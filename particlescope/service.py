from copy import deepcopy
import uuid
import numpy as np
from PIL import Image
from . import validation as v
from .imaging import apply_operation, region_mask, channel_array
from .segmentation import segment, mask_edit
from .measurements import measure_labels, summarize, apply_filters
from .rendering import overlay_image, label_image
from .annotations import recalibrate_annotations


class AnalysisService:
    def __init__(self, store):
        self.store = store
        self._contact_cache = {}

    def require_result(self, sample_id, analysis_id=None):
        result = self.store.result(sample_id, analysis_id)
        if result is None:
            raise v.ValidationError('请先运行图像分析')
        if result.get('stale'):
            raise v.ValidationError('分析对应旧图像，请重新分析')
        return result

    def labels(self, sample_id, analysis_id=None):
        result = self.require_result(sample_id, analysis_id)
        path = self.store.sample_dir(sample_id) / 'analyses' / result['id'] / 'labels.npy'
        if not path.exists():
            raise v.ValidationError('分析掩膜缺失')
        return np.load(path, allow_pickle=False)

    def run(self, sample_id, body, expected_revision=None):
        parameters = v.segmentation(body)
        with self.store.lock:
            meta = self.store.meta(sample_id)
            if expected_revision is not None and meta['revision'] != expected_revision:
                from .storage import ConflictError
                raise ConflictError('样本已修改，请刷新')
            image = self.store.working_image(sample_id)
            labels, gray, selected, details = segment(image, parameters, meta['roi'])
            rows = measure_labels(labels, gray, meta['calibration'], selected)
            return self.save_analysis(
                sample_id,
                labels,
                rows,
                parameters,
                details,
                source='segmentation',
                expected_revision=meta['revision'],
            )

    def save_analysis(self, sample_id, labels, rows, parameters, details, source, expected_revision=None, parent_id=None, measurement_filters=None):
        analysis_id = uuid.uuid4().hex
        with self.store.transaction(sample_id, expected_revision) as (sample_path, meta):
            directory = sample_path / 'analyses' / analysis_id
            directory.mkdir()
            image = self.store.working_image(sample_id)
            selected = region_mask(image.size, meta['roi'])
            result = {
                'id': analysis_id,
                'sample_id': sample_id,
                'created_at': v.now(),
                'image_revision': meta['image_revision'],
                'image_width': image.width,
                'image_height': image.height,
                'parameters': deepcopy(parameters),
                'roi': deepcopy(meta['roi']),
                'calibration': deepcopy(meta['calibration']),
                'details': deepcopy(details),
                'source': source,
                'parent_id': parent_id,
                'measurement_filters': measurement_filters or {},
                'particles': rows,
            }
            result.update(summarize(rows, int(selected.sum()), meta['calibration']))
            np.save(directory / 'labels.npy', labels, allow_pickle=False)
            ids = [r['id'] for r in rows if r.get('included', True)]
            binary = np.isin(labels, ids) & (labels > 0)
            self.store.image_write(directory / 'mask.png', Image.fromarray((binary * 255).astype(np.uint8)))
            self.store.image_write(directory / 'labels.png', label_image(labels, set(ids)))
            self.store.image_write(directory / 'overlay.png', overlay_image(image, labels, result))
            self.store.json_write(directory / 'result.json', result)
            meta['analysis_id'] = analysis_id
        result['sample_revision'] = self.store.meta(sample_id)['revision']
        return result

    def set_roi(self, sample_id, body, expected_revision=None):
        with self.store.transaction(sample_id, expected_revision) as (_, meta):
            meta['roi'] = v.roi(body, meta['width'], meta['height'])
            meta['analysis_id'] = None
        return self.store.meta(sample_id)

    def set_calibration(self, sample_id, body, expected_revision=None):
        calibration = v.calibration(body)
        with self.store.transaction(sample_id, expected_revision) as (path, meta):
            meta['calibration'] = calibration
            meta['analysis_id'] = None
            recalibrate_annotations(self.store, sample_id, calibration)
        return self.store.meta(sample_id)

    def enhance(self, sample_id, body, expected_revision=None):
        parameters = v.enhancement(body)
        with self.store.lock:
            image = self.store.working_image(sample_id)
            result = apply_operation(image, parameters)
            return self.store.commit_image(sample_id, result, parameters['operation'], parameters, expected_revision)

    def edit_mask(self, sample_id, body, expected_revision=None):
        with self.store.lock:
            meta = self.store.meta(sample_id)
            result = self.require_result(sample_id)
            command = v.edit_command(body, meta['width'], meta['height'])
            labels = self.labels(sample_id)
            image = self.store.working_image(sample_id)
            selected = region_mask(image.size, result['roi'])
            changed = mask_edit(labels, command, selected)
            gray = channel_array(image, result['parameters']['channel'])
            rows = measure_labels(changed, gray, meta['calibration'], selected)
            details = dict(result['details'])
            details['manual_command'] = command
            return self.save_analysis(
                sample_id,
                changed,
                rows,
                result['parameters'],
                details,
                'mask_edit',
                expected_revision,
                result['id'],
            )

    def filter_measurements(self, sample_id, body, expected_revision=None):
        with self.store.lock:
            result = self.require_result(sample_id)
            filters = v.filters(body)
            rows = apply_filters(result['particles'], filters)
            labels = self.labels(sample_id)
            return self.save_analysis(
                sample_id,
                labels,
                rows,
                result['parameters'],
                result['details'],
                'measurement_filter',
                expected_revision,
                result['id'],
                filters,
            )

    def contacts(self, sample_id, gap_px):
        with self.store.lock:
            result = self.require_result(sample_id)
            key = (result['id'], round(float(gap_px), 6))
            cached = self._contact_cache.get(key)
            if cached is None:
                from .contacts import contact_network
                cached = contact_network(self.labels(sample_id), result, gap_px)
                if len(self._contact_cache) >= 6:
                    self._contact_cache.pop(next(iter(self._contact_cache)))
                self._contact_cache[key] = cached
            return cached

    def select_analysis(self, sample_id, analysis_id, expected_revision=None):
        with self.store.transaction(sample_id, expected_revision) as (_, meta):
            result = self.store.result(sample_id, analysis_id)
            if result is None:
                raise v.ValidationError('分析记录不存在')
            if result['image_revision'] != meta['image_revision']:
                raise v.ValidationError('不能把旧图像的分析设为当前结果')
            if result['calibration'] != meta['calibration'] or result['roi'] != meta['roi']:
                raise v.ValidationError('分析的标定或区域与当前样本不同，请重新运行')
            meta['analysis_id'] = analysis_id
        return self.store.result(sample_id)

    def render(self, sample_id, style, analysis_id=None):
        result = self.require_result(sample_id, analysis_id)
        labels = self.labels(sample_id, analysis_id)
        image = self.store.working_image(sample_id)
        validated = {}
        for name in ['fill', 'contours', 'boxes', 'ids', 'axes', 'scale_bar', 'show_excluded']:
            validated[name] = v.boolean(style.get(name, name in ['fill', 'contours', 'boxes', 'ids', 'scale_bar']), name)
        validated['opacity'] = v.real(style.get('opacity', 0.25), '覆盖透明度', 0, 1)
        return overlay_image(image, labels, result, validated)

    def import_reference(self, sample_id, stream):
        with self.store.lock:
            meta = self.store.meta(sample_id)
            try:
                with Image.open(stream) as image:
                    if image.size != (meta['width'], meta['height']):
                        raise v.ValidationError('参考掩膜尺寸必须与当前图像相同')
                    binary = np.asarray(image.convert('L')) > 127
            except OSError as exc:
                raise v.ValidationError('参考掩膜不能解码') from exc
            path = self.store.sample_dir(sample_id)
            self.store.image_write(path / 'reference.png', Image.fromarray(binary.astype(np.uint8) * 255))
            self.store.json_write(path / 'reference.json', {
                'image_revision': meta['image_revision'],
                'width': meta['width'],
                'height': meta['height'],
                'created_at': v.now(),
            })
            return {'foreground_pixels': int(binary.sum())}

    def evaluate_reference(self, sample_id):
        from .segmentation import evaluate_reference
        with self.store.lock:
            result = self.require_result(sample_id)
            meta = self.store.meta(sample_id)
            path = self.store.sample_dir(sample_id)
            reference_meta = self.store.json_read(path / 'reference.json')
            if reference_meta is None:
                raise v.ValidationError('请先导入参考掩膜')
            if reference_meta['image_revision'] != meta['image_revision']:
                raise v.ValidationError('参考掩膜对应旧图像，请重新导入')
            reference = np.asarray(Image.open(path / 'reference.png')) > 0
            labels = self.labels(sample_id)
            ids = [r['id'] for r in result['particles'] if r.get('included', True)]
            predicted = np.isin(labels, ids) & (labels > 0)
            valid = region_mask((meta['width'], meta['height']), meta['roi'])
            from .boundaries import reference_boundary_report
            score = evaluate_reference(predicted, reference, valid)
            score.update(reference_boundary_report(predicted, reference, valid, result['scale']))
            return score

    def parameter_sweep(self, sample_id, body):
        from .segmentation import threshold_sweep
        parameters = v.segmentation(body.get('parameters', {}))
        start = v.real(body.get('start', 60), '起始阈值', 0, 255)
        stop = v.real(body.get('stop', 180), '结束阈值', 0, 255)
        steps = v.integer(body.get('steps', 9), '扫描步数', 2, 15)
        if start >= stop:
            raise v.ValidationError('扫描结束阈值必须大于起始阈值')
        with self.store.lock:
            meta = self.store.meta(sample_id)
            image = self.store.working_image(sample_id)
            if image.width * image.height > 4000000:
                raise v.ValidationError('参数扫描限400万像素')
            return threshold_sweep(image, parameters, meta['roi'], np.linspace(start, stop, steps))

    def sample_summary(self, sample_id):
        from .annotations import annotation_summary
        meta = self.store.meta(sample_id)
        result = self.store.result(sample_id)
        annotations = self.store.annotations(sample_id)
        return {
            'meta': meta,
            'analysis': {k: result.get(k) for k in ['id', 'count', 'coverage', 'mean_diameter', 'unit', 'stale']} if result else None,
            'annotations': annotation_summary(annotations),
            'analysis_history_count': len(self.store.list_analyses(sample_id)),
            'preprocessing_history_count': len(meta['history']),
        }
