import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
from flask import Blueprint, request, jsonify, send_file, Response, current_app
from . import validation as v
from .imaging import (
    image_histogram, quality_metrics, intensity_profile,
    region_mask, difference_image, checkerboard, channel_array,
)
from .measurements import included_rows, particle_crop, MEASUREMENT_FIELDS, select_particle
from .statistics import (
    statistics_report, histogram, bootstrap_mean, linear_fit,
    size_classes, particle_size_distribution,
)
from .spatial import (
    spatial_report, cluster_particles, radial_density, pair_correlation, ripley_k,
)
from .texture import texture_report, radial_power_spectrum
from .rendering import (
    png_bytes, heatmap_image, contact_sheet, histogram_chart, draw_annotations,
)
from .exports import (
    particle_csv, annotation_csv, result_json, geojson, binary_tiff,
    label_tiff, analysis_bundle, project_summary, particle_gallery_zip,
)
from .reports import build_report, project_report
from .annotations import add_annotation, delete_annotation, update_annotation, annotation_summary
from .comparison import compare_distributions, compare_analysis_versions, group_summary
from .color_analysis import (
    color_summary, dominant_colors, color_distance, channel_correlations, fluorescence_metrics,
)
from .morphology import (
    porosity_report, granulometry, distance_thickness, minkowski_curve,
    box_counting, skeleton_report,
)


def body():
    result = request.get_json(silent=True)
    if result is None:
        raise v.ValidationError('请提交JSON参数')
    return v.mapping(result)


def expected(data):
    if 'revision' not in data:
        raise v.ValidationError('缺少样本修订号，请刷新页面')
    return v.revision(data)


def binary_response(data, filename, mimetype):
    return send_file(io.BytesIO(data), mimetype=mimetype, as_attachment=True, download_name=filename)


def image_response(image):
    return Response(png_bytes(image), mimetype='image/png', headers={'Cache-Control': 'no-store'})


def create_api(store, service, recipes, jobs, tracking):
    api = Blueprint('api', __name__, url_prefix='/api')

    @api.get('/health')
    def health():
        return jsonify({'status': 'ok', 'application': 'ParticleScope', 'version': '2.0'})

    @api.get('/catalog')
    def catalog():
        from .catalog import tool_catalog
        return jsonify(tool_catalog())

    @api.get('/samples')
    def list_samples():
        return jsonify(store.list_samples(
            project_id=request.args.get('project_id'),
            query=request.args.get('query', ''),
            include_archived=request.args.get('archived') == 'true',
        ))

    @api.post('/samples')
    def import_samples():
        uploads = request.files.getlist('files') or request.files.getlist('file')
        if not uploads:
            raise v.ValidationError('请选择图像')
        if len(uploads) > 20:
            raise v.ValidationError('一次最多导入20个文件')
        project_id = request.form.get('project_id') or None
        imported = []
        failed = []
        for upload in uploads:
            try:
                imported.append(store.import_image(upload.stream, upload.filename, project_id))
            except ValueError as exc:
                failed.append({'name': upload.filename, 'error': str(exc)})
        return jsonify({'imported': imported, 'failed': failed}), 201 if imported else 400

    @api.post('/demo')
    def import_demo():
        from .synthetic import generate_sample
        data = request.get_json(silent=True) or {}
        kind = v.choice(data.get('kind', 'isolated'), '演示类型', ['isolated', 'touching', 'illumination', 'fluorescence', 'fibers', 'porous'])
        image, truth, details = generate_sample(kind)
        stream = io.BytesIO(png_bytes(image))
        meta = store.import_image(stream, details['name'])
        service.import_reference(meta['id'], io.BytesIO(png_bytes(truth)))
        store.json_write(store.sample_dir(meta['id']) / 'demo.json', details)
        return jsonify(meta), 201

    @api.get('/samples/<key>')
    def sample(key):
        with store.lock:
            return jsonify(service.sample_summary(key))

    @api.patch('/samples/<key>')
    def update_sample(key):
        data = body()
        return jsonify(store.update_meta(key, data, expected(data)))

    @api.post('/samples/<key>/archive')
    def archive_sample(key):
        data = body()
        archived = v.boolean(data.get('archived', True), '归档状态')
        return jsonify(store.archive(key, archived, expected(data)))

    @api.post('/samples/<key>/duplicate')
    def duplicate_sample(key):
        return jsonify(store.duplicate(key)), 201

    @api.get('/samples/<key>/image/<kind>')
    def sample_image(key, kind):
        path = store.sample_dir(key)
        if kind in ('original', 'working', 'thumb', 'reference'):
            target = path / (kind + '.png')
        elif kind in ('overlay', 'mask', 'labels'):
            analysis_id = request.args.get('analysis_id')
            result = service.require_result(key, analysis_id)
            target = path / 'analyses' / result['id'] / (kind + '.png')
        else:
            raise v.ValidationError('图像类型不支持')
        if not target.exists():
            raise v.ValidationError('图像尚未生成')
        return send_file(target, max_age=0)

    @api.get('/samples/<key>/history')
    def image_history(key):
        return jsonify(store.meta(key)['history'])

    @api.post('/samples/<key>/restore')
    def restore_image(key):
        data = body()
        version_id = data.get('version_id', 'original')
        return jsonify(store.restore_image(key, version_id, expected(data)))

    @api.post('/samples/<key>/enhance')
    def enhance(key):
        data = body()
        return jsonify(service.enhance(key, data, expected(data)))

    @api.put('/samples/<key>/roi')
    def set_roi(key):
        data = body()
        return jsonify(service.set_roi(key, data.get('roi'), expected(data)))

    @api.put('/samples/<key>/calibration')
    def calibrate(key):
        data = body()
        return jsonify(service.set_calibration(key, data, expected(data)))

    @api.post('/samples/<key>/analyze')
    def analyze(key):
        data = body()
        return jsonify(service.run(key, data, expected(data)))

    @api.get('/samples/<key>/result')
    def result(key):
        return jsonify(store.result(key, request.args.get('analysis_id')))

    @api.get('/samples/<key>/analyses')
    def analyses(key):
        return jsonify(store.list_analyses(key))

    @api.post('/samples/<key>/analyses/select')
    def select_analysis(key):
        data = body()
        return jsonify(service.select_analysis(key, data.get('analysis_id'), expected(data)))

    @api.post('/samples/<key>/mask')
    def edit_mask(key):
        data = body()
        return jsonify(service.edit_mask(key, data, expected(data)))

    @api.post('/samples/<key>/filter')
    def filter_result(key):
        data = body()
        return jsonify(service.filter_measurements(key, data, expected(data)))

    @api.post('/samples/<key>/sweep')
    def sweep(key):
        return jsonify(service.parameter_sweep(key, body()))

    @api.get('/samples/<key>/histogram')
    def intensity_histogram(key):
        with store.lock:
            meta = store.meta(key)
            return jsonify(image_histogram(store.working_image(key), meta['roi']))

    @api.get('/samples/<key>/quality')
    def quality(key):
        with store.lock:
            return jsonify(quality_metrics(store.working_image(key)))

    @api.post('/samples/<key>/profile')
    def profile(key):
        data = body()
        with store.lock:
            meta = store.meta(key)
            start = v.point(data.get('start'), meta['width'], meta['height'], '起点')
            end = v.point(data.get('end'), meta['width'], meta['height'], '终点')
            count = v.integer(data.get('samples', 200), '采样数', 2, 2000)
            width = v.integer(data.get('line_width', 1), '线宽', 1, 51)
            channel = v.choice(data.get('channel', 'gray'), '通道', ['gray', 'red', 'green', 'blue'])
            report = intensity_profile(store.working_image(key), start, end, channel, count, width)
            report['distance'] = [round(d * meta['calibration']['scale'], 6) for d in report['distance_px']]
            report['unit'] = meta['calibration']['unit']
            return jsonify(report)

    @api.get('/samples/<key>/statistics')
    def statistics(key):
        with store.lock:
            report = service.require_result(key)
            bins = v.integer(request.args.get('bins', 15), '分箱数', 3, 100)
            return jsonify(statistics_report(report, bins))

    @api.post('/samples/<key>/distribution')
    def distribution(key):
        data = body()
        weighting = v.choice(data.get('weighting', 'number'), '权重', ['number', 'area', 'volume'])
        bins = v.integer(data.get('bins', 15), '分箱数', 3, 100)
        logarithmic = v.boolean(data.get('logarithmic', False), '对数分箱')
        with store.lock:
            report = service.require_result(key)
            return jsonify(particle_size_distribution(included_rows(report), weighting, bins, logarithmic))

    @api.post('/samples/<key>/bootstrap')
    def bootstrap(key):
        data = body()
        field = v.choice(data.get('field', 'diameter'), '测量指标', MEASUREMENT_FIELDS)
        repeats = v.integer(data.get('repeats', 1000), '重采样次数', 100, 3000)
        confidence = v.real(data.get('confidence', 0.95), '置信水平', 0.5, 0.999)
        with store.lock:
            rows = included_rows(service.require_result(key))
            return jsonify(bootstrap_mean([r[field] for r in rows], repeats, confidence))

    @api.post('/samples/<key>/regression')
    def regression(key):
        data = body()
        x = v.choice(data.get('x', 'diameter'), 'X指标', MEASUREMENT_FIELDS)
        y = v.choice(data.get('y', 'circularity'), 'Y指标', MEASUREMENT_FIELDS)
        with store.lock:
            rows = included_rows(service.require_result(key))
            return jsonify(linear_fit(rows, x, y))

    @api.post('/samples/<key>/size-classes')
    def size_classification(key):
        data = body()
        boundaries = [v.real(a, '分级边界', 0) for a in v.sequence(data.get('boundaries'), '分级边界', 100)]
        with store.lock:
            rows = included_rows(service.require_result(key))
            return jsonify(size_classes(rows, boundaries))

    @api.get('/samples/<key>/spatial')
    def spatial(key):
        with store.lock:
            report = service.require_result(key)
            selected = region_mask((report['image_width'], report['image_height']), report['roi'])
            columns = v.integer(request.args.get('columns', 6), '格网列数', 2, 30)
            rows_count = v.integer(request.args.get('rows', 4), '格网行数', 2, 30)
            return jsonify(spatial_report(report, selected, columns, rows_count))

    @api.post('/samples/<key>/clusters')
    def clusters(key):
        data = body()
        radius = v.real(data.get('radius', 50), '聚类距离', 0.000001)
        min_points = v.integer(data.get('min_points', 3), '最小点数', 2, 1000)
        with store.lock:
            report = service.require_result(key)
            return jsonify(cluster_particles(included_rows(report), report['scale'], radius, min_points))

    @api.post('/samples/<key>/radial-density')
    def radial(key):
        data = body()
        with store.lock:
            report = service.require_result(key)
            center = v.point(data.get('center'), report['image_width'], report['image_height'])
            bins = v.integer(data.get('bins', 12), '环带数', 3, 100)
            selected = region_mask((report['image_width'], report['image_height']), report['roi'])
            return jsonify(radial_density(included_rows(report), center, selected, report['scale'], bins))

    @api.get('/samples/<key>/pair-correlation')
    def pair_distribution(key):
        with store.lock:
            report = service.require_result(key)
            return jsonify(pair_correlation(included_rows(report), report['image_width'], report['image_height'], report['scale']))

    @api.get('/samples/<key>/ripley')
    def ripley(key):
        with store.lock:
            report = service.require_result(key)
            return jsonify(ripley_k(included_rows(report), report['image_width'], report['image_height'], report['scale']))

    @api.get('/samples/<key>/texture')
    def texture(key):
        levels = v.integer(request.args.get('levels', 32), '灰度级数', 4, 64)
        offset = v.integer(request.args.get('distance', 1), '像素偏移', 1, 50)
        with store.lock:
            meta = store.meta(key)
            return jsonify(texture_report(store.working_image(key), meta['roi'], levels, offset))

    @api.get('/samples/<key>/spectrum')
    def spectrum(key):
        with store.lock:
            meta = store.meta(key)
            return jsonify(radial_power_spectrum(store.working_image(key), meta['roi']))

    @api.get('/samples/<key>/color')
    def colors(key):
        with store.lock:
            meta = store.meta(key)
            image = store.working_image(key)
            return jsonify({
                'summary': color_summary(image, meta['roi']),
                'correlation': channel_correlations(image, meta['roi']),
            })

    @api.post('/samples/<key>/dominant-colors')
    def dominant(key):
        data = body()
        count = v.integer(data.get('clusters', 4), '颜色数', 2, 10)
        with store.lock:
            meta = store.meta(key)
            return jsonify(dominant_colors(store.working_image(key), meta['roi'], count))

    @api.post('/samples/<key>/color-distance')
    def color_mask(key):
        data = body()
        rgb = v.sequence(data.get('rgb', [50, 70, 80]), 'RGB', 3)
        if len(rgb) != 3:
            raise v.ValidationError('RGB需要三个通道值')
        rgb = [v.integer(value, '通道值', 0, 255) for value in rgb]
        tolerance = v.real(data.get('tolerance', 20), '色差容限', 0, 200)
        with store.lock:
            meta = store.meta(key)
            mask, info = color_distance(store.working_image(key), meta['roi'], rgb, tolerance)
            store.image_write(store.sample_dir(key) / 'color-mask.png', Image.fromarray(mask.astype(np.uint8) * 255))
            return jsonify(info)

    @api.post('/samples/<key>/fluorescence')
    def fluorescence(key):
        data = body()
        channel = v.choice(data.get('channel', 'green'), '通道', ['gray', 'red', 'green', 'blue'])
        percentile = v.real(data.get('background_percentile', 10), '背景分位数', 0, 50)
        with store.lock:
            meta = store.meta(key)
            return jsonify(fluorescence_metrics(store.working_image(key), meta['roi'], channel, percentile))

    @api.get('/samples/<key>/morphology')
    def morphology(key):
        with store.lock:
            report = service.require_result(key)
            labels = service.labels(key)
            ids = [r['id'] for r in included_rows(report)]
            binary = np.isin(labels, ids) & (labels > 0)
            selected = region_mask((report['image_width'], report['image_height']), report['roi'])
            return jsonify({
                'porosity': porosity_report(binary, selected),
                'thickness': distance_thickness(binary, selected, report['scale']),
                'box_counting': box_counting(binary),
            })

    @api.post('/samples/<key>/granulometry')
    def granulometry_analysis(key):
        data = body()
        radii = [v.integer(a, '结构元素半径', 1, 40) for a in v.sequence(data.get('radii', [1, 2, 4, 8, 12]), '半径序列', 10)]
        if radii != sorted(set(radii)):
            raise v.ValidationError('半径必须严格递增')
        with store.lock:
            report = service.require_result(key)
            labels = service.labels(key)
            ids = [r['id'] for r in included_rows(report)]
            selected = region_mask((report['image_width'], report['image_height']), report['roi'])
            return jsonify(granulometry((labels > 0) & np.isin(labels, ids), selected, radii))

    @api.get('/samples/<key>/skeleton')
    def skeleton(key):
        with store.lock:
            report = service.require_result(key)
            labels = service.labels(key)
            ids = [r['id'] for r in included_rows(report)]
            selected = region_mask((report['image_width'], report['image_height']), report['roi'])
            mask, details = skeleton_report((labels > 0) & np.isin(labels, ids), selected, report['scale'])
            store.image_write(store.sample_dir(key) / 'skeleton.png', Image.fromarray(mask.astype(np.uint8) * 255))
            return jsonify(details)

    @api.get('/samples/<key>/minkowski')
    def minkowski(key):
        with store.lock:
            meta = store.meta(key)
            image = store.working_image(key)
            selected = region_mask(image.size, meta['roi'])
            return jsonify(minkowski_curve(channel_array(image), selected, np.linspace(20, 230, 12)))

    @api.get('/samples/<key>/annotations')
    def annotations(key):
        with store.lock:
            entries = store.annotations(key)
            return jsonify({'annotations': entries, 'summary': annotation_summary(entries)})

    @api.post('/samples/<key>/annotations')
    def create_annotation(key):
        data = body()
        return jsonify(add_annotation(store, key, data, expected(data))), 201

    @api.patch('/samples/<key>/annotations/<annotation_id>')
    def edit_annotation(key, annotation_id):
        data = body()
        v.identifier(annotation_id)
        return jsonify(update_annotation(store, key, annotation_id, data, expected(data)))

    @api.delete('/samples/<key>/annotations/<annotation_id>')
    def remove_annotation(key, annotation_id):
        data = body()
        v.identifier(annotation_id)
        return jsonify(delete_annotation(store, key, annotation_id, expected(data)))

    @api.get('/samples/<key>/annotated')
    def annotated(key):
        with store.lock:
            meta = store.meta(key)
            image = draw_annotations(store.working_image(key), store.annotations(key), meta['calibration'])
            return image_response(image)

    @api.post('/samples/<key>/reference')
    def upload_reference(key):
        file = request.files.get('file')
        if file is None:
            raise v.ValidationError('请选择参考掩膜')
        return jsonify(service.import_reference(key, file.stream))

    @api.get('/samples/<key>/reference-score')
    def reference_score(key):
        return jsonify(service.evaluate_reference(key))

    @api.post('/samples/<key>/render')
    def custom_render(key):
        return image_response(service.render(key, body()))

    @api.post('/samples/<key>/heatmap')
    def heatmap(key):
        data = body()
        field = v.choice(data.get('field', 'diameter'), '热图指标', MEASUREMENT_FIELDS)
        radius = v.real(data.get('radius', 25), '热图核半径', 2, 150)
        with store.lock:
            result = service.require_result(key)
            image, _ = heatmap_image(store.working_image(key), included_rows(result), field, radius)
            return image_response(image)

    @api.get('/samples/<key>/gallery')
    def gallery(key):
        with store.lock:
            result = service.require_result(key)
            image = contact_sheet(store.working_image(key), service.labels(key), included_rows(result))
            return image_response(image)

    @api.get('/samples/<key>/particle/<int:object_id>')
    def particle_image(key, object_id):
        with store.lock:
            image = particle_crop(store.working_image(key), service.labels(key), object_id, 10, request.args.get('isolate') == 'true')
            return image_response(image)

    @api.post('/samples/<key>/pick')
    def pick_particle(key):
        data = body()
        with store.lock:
            meta = store.meta(key)
            position = v.point(data.get('point'), meta['width'], meta['height'])
            labels = service.labels(key)
            result = service.require_result(key)
            x, y = [int(round(a)) for a in position]
            return jsonify(select_particle(labels, result['particles'], x, y))

    @api.get('/samples/<key>/difference')
    def original_difference(key):
        with store.lock:
            image, details = difference_image(store.original_image(key), store.working_image(key))
            return image_response(image)

    @api.get('/samples/<key>/checkerboard')
    def original_checkerboard(key):
        with store.lock:
            return image_response(checkerboard(store.original_image(key), store.working_image(key)))

    @api.get('/samples/<key>/export/<kind>')
    def export(key, kind):
        with store.lock:
            result = service.require_result(key)
            if kind == 'contours-svg':
                from .exports import contour_svg
                return binary_response(contour_svg(service.labels(key), result), 'contours.svg', 'image/svg+xml')
            if kind == 'csv':
                return binary_response(particle_csv(result), 'particles.csv', 'text/csv')
            if kind == 'json':
                return binary_response(result_json(result), 'analysis.json', 'application/json')
            if kind == 'geojson':
                return binary_response(result_json(geojson(result)), 'centroids.geojson', 'application/geo+json')
            if kind == 'mask-tiff':
                return binary_response(binary_tiff(service.labels(key), result), 'mask.tiff', 'image/tiff')
            if kind == 'labels-tiff':
                return binary_response(label_tiff(service.labels(key)), 'labels.tiff', 'image/tiff')
            if kind == 'bundle':
                return binary_response(analysis_bundle(store, service, key), 'analysis.zip', 'application/zip')
            if kind == 'crops':
                return binary_response(particle_gallery_zip(store, service, key), 'particle-crops.zip', 'application/zip')
            if kind == 'annotations':
                return binary_response(annotation_csv(store.annotations(key)), 'annotations.csv', 'text/csv')
            if kind == 'chart':
                chart = histogram_chart(statistics_report(result)['number_distribution'])
                return binary_response(png_bytes(chart), 'distribution.png', 'image/png')
            raise v.ValidationError('不支持的导出格式')

    @api.get('/samples/<key>/report')
    def report(key):
        return Response(build_report(store, service, key), mimetype='text/html')

    @api.post('/compare')
    def compare():
        data = body()
        field = v.choice(data.get('field', 'diameter'), '比较指标', MEASUREMENT_FIELDS)
        with store.lock:
            first = service.require_result(data.get('first'))
            second = service.require_result(data.get('second'))
            return jsonify(compare_distributions(first, second, field))

    @api.post('/compare-versions')
    def compare_versions():
        data = body()
        key = v.identifier(data.get('sample_id'))
        with store.lock:
            first = service.require_result(key, data.get('first'))
            second = service.require_result(key, data.get('second'))
            first_labels = service.labels(key, first['id'])
            second_labels = service.labels(key, second['id'])
            return jsonify(compare_analysis_versions(first, second, first_labels, second_labels))

    @api.post('/group-summary')
    def group():
        data = body()
        keys = v.sequence(data.get('sample_ids'), '样本列表', 100)
        field = v.choice(data.get('field', 'diameter'), '指标', MEASUREMENT_FIELDS)
        with store.lock:
            results = [service.require_result(key) for key in keys]
            return jsonify(group_summary(results, field))

    @api.get('/projects')
    def projects():
        return jsonify(store.list_projects())

    @api.post('/projects')
    def create_project():
        data = body()
        return jsonify(store.create_project(data.get('name'), data.get('description', ''))), 201

    @api.get('/projects/<project_id>/report')
    def project_report_page(project_id):
        store.project(project_id)
        with store.lock:
            keys = [s['id'] for s in store.list_samples(project_id)]
            return Response(project_report(store, service, keys), mimetype='text/html')

    @api.get('/projects/<project_id>/summary.csv')
    def project_csv(project_id):
        store.project(project_id)
        with store.lock:
            keys = [s['id'] for s in store.list_samples(project_id)]
            return binary_response(project_summary(store, keys), 'project-summary.csv', 'text/csv')

    @api.get('/recipes')
    def list_recipes():
        return jsonify(recipes.list())

    @api.post('/recipes')
    def create_recipe():
        return jsonify(recipes.create(body())), 201

    @api.post('/recipes/from-sample/<key>')
    def save_recipe(key):
        data = body()
        return jsonify(recipes.from_analysis(key, data.get('name'), data.get('description', ''))), 201

    @api.put('/recipes/<recipe_id>')
    def update_recipe(recipe_id):
        data = body()
        revision = v.integer(data.get('revision'), '配方修订', 0)
        return jsonify(recipes.update(recipe_id, data, revision))

    @api.get('/recipes/<recipe_id>/export')
    def export_recipe(recipe_id):
        return binary_response(result_json(recipes.export(recipe_id)), 'recipe.json', 'application/json')

    @api.post('/recipes/import')
    def import_recipe():
        return jsonify(recipes.import_recipe(body())), 201

    @api.get('/jobs')
    def list_jobs():
        return jsonify(jobs.list())

    @api.post('/jobs')
    def start_job():
        data = body()
        return jsonify(jobs.start(data.get('sample_ids'), data.get('recipe_id'))), 202

    @api.get('/jobs/<job_id>')
    def job(job_id):
        return jsonify(jobs.get(job_id))

    @api.post('/jobs/<job_id>/cancel')
    def cancel_job(job_id):
        return jsonify(jobs.cancel(job_id))

    @api.post('/jobs/<job_id>/retry')
    def retry_job(job_id):
        return jsonify(jobs.retry(job_id)), 202

    @api.post('/registration')
    def register_images():
        from .registration import registration_report, create_registered_sample
        data = body()
        reference_id = v.identifier(data.get('reference'))
        moving_id = v.identifier(data.get('moving'))
        mode = v.choice(data.get('mode', 'auto'), '配准方式', ['auto', 'manual'])
        dx = v.real(data.get('dx', 0), '平移X', -8000, 8000)
        dy = v.real(data.get('dy', 0), '平移Y', -8000, 8000)
        save = v.boolean(data.get('save', False), '保存新样本')
        with store.lock:
            if save:
                return jsonify(create_registered_sample(store, reference_id, moving_id, mode, dx, dy))
            reference = store.working_image(reference_id)
            moving = store.working_image(moving_id)
            _, _, report = registration_report(reference, moving, mode, dx, dy)
            return jsonify(report)

    @api.post('/mosaic')
    def mosaic():
        from .registration import feather_mosaic
        data = body()
        first_id = v.identifier(data.get('reference'))
        second_id = v.identifier(data.get('moving'))
        dx = v.real(data.get('dx', 0), '拼接偏移X', -8000, 8000)
        dy = v.real(data.get('dy', 0), '拼接偏移Y', -8000, 8000)
        feather = v.integer(data.get('feather', 30), '羽化宽度', 1, 200)
        with store.lock:
            first_meta = store.meta(first_id)
            second_meta = store.meta(second_id)
            if first_meta['calibration'] != second_meta['calibration']:
                raise v.ValidationError('拼接要求两个样本的标定完全一致')
            image, info = feather_mosaic(store.working_image(first_id), store.working_image(second_id), dx, dy, feather)
            meta = store.import_image(io.BytesIO(png_bytes(image)), '拼接视野.png')
            store.json_write(store.sample_dir(meta['id']) / 'provenance.json', {
                'operation': 'mosaic',
                'reference_sample_id': first_id,
                'moving_sample_id': second_id,
                'dx': dx,
                'dy': dy,
                'details': info,
            })
            service.set_calibration(meta['id'], first_meta['calibration'])
            return jsonify({'sample': store.meta(meta['id']), 'details': info})

    @api.post('/affine')
    def affine():
        from .registration import affine_landmarks, apply_affine
        data = body()
        reference_id = v.identifier(data.get('reference'))
        moving_id = v.identifier(data.get('moving'))
        with store.lock:
            reference = store.meta(reference_id)
            moving = store.meta(moving_id)
            source_points = [v.point(p, moving['width'], moving['height']) for p in v.sequence(data.get('source_points'), '移动图控制点', 100)]
            target_points = [v.point(p, reference['width'], reference['height']) for p in v.sequence(data.get('target_points'), '参考图控制点', 100)]
            transform = affine_landmarks(source_points, target_points)
            image = apply_affine(store.working_image(moving_id), transform, (reference['width'], reference['height']))
            meta = store.import_image(io.BytesIO(png_bytes(image)), '仿射配准.png')
            store.json_write(store.sample_dir(meta['id']) / 'provenance.json', {
                'operation': 'affine',
                'reference_sample_id': reference_id,
                'moving_sample_id': moving_id,
                'transform': transform,
            })
            return jsonify({'sample': meta, 'transform': transform})

    @api.post('/samples/<key>/boundary-sensitivity')
    def boundary_error(key):
        from .uncertainty import boundary_sensitivity
        data = body()
        pixels = v.integer(data.get('pixels', 1), '边界变化像素', 1, 5)
        with store.lock:
            report = service.require_result(key)
            return jsonify(boundary_sensitivity(service.labels(key), report, pixels))

    @api.post('/samples/<key>/calibration-uncertainty')
    def calibration_error(key):
        from .uncertainty import calibration_uncertainty
        data = body()
        pixel_error = v.real(data.get('pixel_error', 1), '像素长度标准差', 0, 1000)
        physical_error = v.real(data.get('physical_error', 0), '实际长度标准差', 0, 1000)
        with store.lock:
            report = service.require_result(key)
            calibration = report['calibration']
            return jsonify(calibration_uncertainty(report, calibration['pixel_length'], calibration['physical_length'], pixel_error, physical_error))

    @api.post('/samples/<key>/sampling-precision')
    def sampling(key):
        from .uncertainty import sampling_precision
        data = body()
        error = v.real(data.get('target_relative_error', 0.05), '目标相对误差', 0.001, 0.5)
        with store.lock:
            return jsonify(sampling_precision(service.require_result(key), error))

    @api.post('/samples/<key>/block-bootstrap')
    def block_bootstrap(key):
        from .uncertainty import spatial_block_bootstrap
        data = body()
        columns = v.integer(data.get('columns', 4), '块列数', 2, 20)
        rows = v.integer(data.get('rows', 4), '块行数', 2, 20)
        repeats = v.integer(data.get('repeats', 1000), '重采样次数', 100, 3000)
        with store.lock:
            result = service.require_result(key)
            return jsonify(spatial_block_bootstrap(result, result['image_width'], result['image_height'], columns, rows, repeats))

    @api.post('/samples/<key>/radial-signature')
    def shape_signature(key):
        from .boundaries import radial_signature
        data = body()
        particle_id = v.integer(data.get('particle_id', 1), '颗粒编号', 1, 20000)
        bins = v.integer(data.get('bins', 72), '角度采样数', 12, 360)
        with store.lock:
            return jsonify(radial_signature(service.labels(key), particle_id, bins))

    @api.get('/sequences')
    def sequences():
        return jsonify(tracking.list())

    @api.post('/sequences')
    def create_sequence():
        return jsonify(tracking.create(body())), 201

    @api.get('/sequences/<sid>')
    def sequence(sid):
        return jsonify(tracking.state(tracking.load(sid)))

    @api.patch('/sequences/<sid>')
    def update_sequence(sid):
        return jsonify(tracking.update(sid, body()))

    @api.delete('/sequences/<sid>')
    def delete_sequence(sid):
        return jsonify(tracking.delete(sid, body()))

    @api.post('/sequences/<sid>/relink')
    def relink_sequence(sid):
        data = body()
        return jsonify(tracking.relink(sid, data.get('revision')))

    @api.post('/sequences/<sid>/links')
    def add_sequence_link(sid):
        return jsonify(tracking.add_link(sid, body()))

    @api.delete('/sequences/<sid>/links')
    def remove_sequence_link(sid):
        return jsonify(tracking.remove_link(sid, body()))

    @api.post('/sequences/<sid>/links/confirm')
    def confirm_sequence_link(sid):
        return jsonify(tracking.confirm_link(sid, body()))

    @api.post('/sequences/<sid>/pending/<key>/resolve')
    def resolve_sequence_pending(sid, key):
        return jsonify(tracking.resolve(sid, key, body()))

    @api.post('/sequences/<sid>/rejections/clear')
    def clear_sequence_rejections(sid):
        return jsonify(tracking.clear_rejections(sid, body()))

    @api.post('/sequences/<sid>/frames/<int:index>/rebind')
    def rebind_sequence_frame(sid, index):
        return jsonify(tracking.rebind(sid, index, body()))

    @api.get('/sequences/<sid>/overlay/<int:frame>')
    def sequence_overlay(sid, frame):
        return image_response(tracking.overlay(sid, frame, request.args.get('track')))

    @api.get('/sequences/<sid>/pick')
    def sequence_pick(sid):
        frame = v.integer(request.args.get('frame', 0), '帧序号', 0, 99)
        x = v.real(request.args.get('x', 0), '坐标X', 0, 16000)
        y = v.real(request.args.get('y', 0), '坐标Y', 0, 16000)
        return jsonify(tracking.pick(sid, frame, x, y))

    @api.get('/sequences/<sid>/export/tracks.csv')
    def sequence_tracks_csv(sid):
        return binary_response(tracking.tracks_csv(sid), 'tracks.csv', 'text/csv')

    return api
