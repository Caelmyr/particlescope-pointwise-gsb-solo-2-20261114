import io
import json
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from particlescope import create_app
from particlescope.rendering import png_bytes
from particlescope.synthetic import generate_sample
from particlescope.registration import phase_correlation, translate, affine_landmarks
from particlescope.segmentation import (
    evaluate_reference,
    threshold_sweep,
    clean_binary,
    label_binary,
    filter_labels,
)
from particlescope.imaging import channel_array, region_mask
from particlescope.measurements import crofton_perimeter
from particlescope.morphology import skeletonize


class Workflows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.app = create_app(cls.temp.name)
        cls.app.testing = True
        cls.client = cls.app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls.app.extensions['jobs'].executor.shutdown(wait=True)
        cls.temp.cleanup()

    def demo(self, kind='isolated'):
        response = self.client.post('/api/demo', json={'kind': kind})
        self.assertEqual(response.status_code, 201, response.json)
        return response.json

    def request(self, method, sample, path, data=None):
        response = self.client.open('/api/samples/' + sample['id'] + '/' + path, method=method, json=data)
        self.assertEqual(response.status_code, 200, response.json if response.is_json else response.data[:200])
        value = response.json if response.is_json else response.data
        response.close()
        return value

    def meta(self, sample):
        return self.client.get('/api/samples/' + sample['id']).json['meta']

    def analyze(self, sample, **parameters):
        parameters['revision'] = self.meta(sample)['revision']
        return self.request('POST', sample, 'analyze', parameters)

    def test_isolated_ground_truth(self):
        sample = self.demo()
        result = self.analyze(sample, method='otsu')
        self.assertEqual(result['count'], 40)
        self.assertGreater(result['mean_diameter'], 25)
        score = self.request('GET', sample, 'reference-score')
        self.assertGreater(score['iou'], 0.98)
        self.assertGreater(score['dice'], 0.99)
        self.assertEqual(score['connected_component_matching']['matched_objects'], 40)
        self.assertLess(score['surface_distance']['hausdorff95'], 1.1)

    def test_calibration_scales_lengths_and_areas(self):
        sample = self.demo()
        first = self.analyze(sample)
        self.request('PUT', sample, 'calibration', {
            'revision': self.meta(sample)['revision'], 'pixel_length': 20,
            'physical_length': 10, 'unit': 'μm',
        })
        self.assertIsNone(self.request('GET', sample, 'result'))
        second = self.analyze(sample)
        self.assertAlmostEqual(second['mean_diameter'], first['mean_diameter'] / 2, places=5)
        self.assertAlmostEqual(second['total_area'], first['total_area'] / 4, places=5)
        self.assertEqual(second['unit'], 'μm')

    def test_roi_changes_denominator_and_excludes_boundary(self):
        sample = self.demo()
        self.request('PUT', sample, 'roi', {'revision': 0, 'roi': {'type': 'rectangle', 'x': 0, 'y': 0, 'width': 480, 'height': 640}})
        result = self.analyze(sample)
        self.assertEqual(result['analyzed_area'], 480 * 640)
        self.assertLess(result['count'], 40)
        self.assertGreater(result['count'], 10)
        for particle in result['particles']:
            self.assertFalse(particle['touches_roi'])

    def test_touching_separation(self):
        sample = self.demo('touching')
        first = self.analyze(sample, method='otsu')
        second = self.analyze(sample, method='otsu', split_touching=True, seed_distance=20)
        self.assertEqual(first['count'], 12)
        self.assertEqual(second['count'], 24)
        self.assertAlmostEqual(second['coverage'], first['coverage'], places=3)

    def test_manual_edit_preserves_other_split_instances(self):
        sample = self.demo('touching')
        result = self.analyze(sample, method='otsu', split_touching=True, seed_distance=20)
        self.assertEqual(result['count'], 24)
        first = result['particles'][0]
        mask = self.app.extensions['analysis'].labels(sample['id'])
        y, x = np.nonzero(mask == first['id'])
        changed = self.request('POST', sample, 'mask', {
            'revision': self.meta(sample)['revision'],
            'operation': 'remove_object',
            'point': [int(x[len(x) // 2]), int(y[len(y) // 2])],
        })
        self.assertEqual(changed['count'], 23)

    def test_bright_fluorescence(self):
        sample = self.demo('fluorescence')
        result = self.analyze(sample, method='otsu', polarity='light', channel='green')
        self.assertEqual(result['count'], 40)
        metrics = self.request('POST', sample, 'fluorescence', {'channel': 'green'})
        self.assertGreater(metrics['corrected_integrated_intensity'], 0)
        self.assertGreater(metrics['estimated_snr'], 1)

    def test_filter_preserves_detection_and_history(self):
        sample = self.demo()
        first = self.analyze(sample)
        second = self.request('POST', sample, 'filter', {
            'revision': self.meta(sample)['revision'], 'exclude_ids': [1, 2],
        })
        self.assertEqual(second['count'], 38)
        self.assertEqual(second['total_detected'], 40)
        self.assertLess(second['coverage'], first['coverage'])
        self.assertEqual(second['parent_id'], first['id'])
        old = self.client.get('/api/samples/' + sample['id'] + '/result?analysis_id=' + first['id']).json
        self.assertEqual(old['count'], 40)

    def test_manual_object_removal(self):
        sample = self.demo()
        first = self.analyze(sample)
        p = first['particles'][0]
        changed = self.request('POST', sample, 'mask', {
            'revision': self.meta(sample)['revision'], 'operation': 'remove_object',
            'point': [p['centroid_x'], p['centroid_y']],
        })
        self.assertEqual(changed['count'], 39)
        self.assertEqual(changed['source'], 'mask_edit')

    def test_annotations_recalibrate(self):
        sample = self.demo()
        line = self.client.post('/api/samples/' + sample['id'] + '/annotations', json={
            'revision': 0, 'type': 'line', 'points': [[10, 10], [40, 50]], 'label': '标尺',
        })
        self.assertEqual(line.status_code, 201)
        self.assertEqual(line.json['measurement']['length'], 50)
        angle = self.client.post('/api/samples/' + sample['id'] + '/annotations', json={
            'revision': self.meta(sample)['revision'], 'type': 'angle', 'points': [[10, 20], [10, 10], [20, 10]],
        })
        self.assertEqual(angle.status_code, 201)
        self.assertAlmostEqual(angle.json['measurement']['angle'], 90)
        self.request('PUT', sample, 'calibration', {'revision': self.meta(sample)['revision'], 'pixel_length': 2, 'physical_length': 1, 'unit': 'μm'})
        annotations = self.request('GET', sample, 'annotations')['annotations']
        self.assertEqual(annotations[0]['measurement']['length'], 25)

    def test_preprocessing_invalidates_analysis_and_restore(self):
        sample = self.demo()
        self.analyze(sample)
        original = self.request('GET', sample, 'image/working')
        self.request('POST', sample, 'enhance', {'revision': self.meta(sample)['revision'], 'operation': 'invert'})
        self.assertIsNone(self.request('GET', sample, 'result'))
        self.assertNotEqual(original, self.request('GET', sample, 'image/working'))
        self.request('POST', sample, 'restore', {'revision': self.meta(sample)['revision'], 'version_id': 'original'})
        self.assertEqual(original, self.request('GET', sample, 'image/working'))
        self.assertEqual(len(self.request('GET', sample, 'history')), 2)

    def test_geometry_resets_roi_calibration_annotations(self):
        sample = self.demo()
        self.request('PUT', sample, 'calibration', {'revision': 0, 'unit': 'μm', 'pixel_length': 2, 'physical_length': 1})
        self.request('POST', sample, 'enhance', {'revision': self.meta(sample)['revision'], 'operation': 'resize', 'width': 480, 'height': 320})
        meta = self.meta(sample)
        self.assertEqual((meta['width'], meta['height']), (480, 320))
        self.assertEqual(meta['calibration']['unit'], 'px')
        self.assertEqual(meta['roi'], {'type': 'full'})

    def test_statistics_reconcile_counts(self):
        sample = self.demo()
        result = self.analyze(sample)
        report = self.request('GET', sample, 'statistics')
        self.assertEqual(report['descriptive']['diameter']['n'], result['count'])
        self.assertAlmostEqual(sum(report['number_distribution']['counts']), result['count'])
        self.assertAlmostEqual(sum(report['number_distribution']['percent']), 100, places=4)
        self.assertLessEqual(report['number_distribution']['d10'], report['number_distribution']['d50'])
        self.assertLessEqual(report['number_distribution']['d50'], report['number_distribution']['d90'])
        bootstrap = self.request('POST', sample, 'bootstrap', {'repeats': 200})
        self.assertLess(bootstrap['lower'], result['mean_diameter'])
        self.assertGreater(bootstrap['upper'], result['mean_diameter'])

    def test_spatial_counts_reconcile(self):
        sample = self.demo()
        self.analyze(sample)
        report = self.request('GET', sample, 'spatial')
        self.assertEqual(sum(r['count'] for r in report['quadrat']['cells']), 40)
        self.assertEqual(len(report['nearest_neighbors']['pairs']), 40)
        self.assertGreater(report['nearest_neighbors']['summary']['minimum'], 60)

    def test_color_texture_and_profile(self):
        sample = self.demo()
        colors = self.request('POST', sample, 'dominant-colors', {'clusters': 3})
        self.assertAlmostEqual(sum(c['percent'] for c in colors['colors']), 100, places=4)
        texture = self.request('GET', sample, 'texture')
        self.assertGreaterEqual(texture['glcm']['average']['homogeneity'], 0)
        self.assertLessEqual(texture['glcm']['average']['homogeneity'], 1)
        profile = self.request('POST', sample, 'profile', {'start': [10, 10], 'end': [110, 10], 'samples': 101})
        self.assertEqual(profile['length_px'], 100)
        self.assertEqual(len(profile['intensity']), 101)

    def test_export_contents_and_report(self):
        sample = self.demo()
        self.analyze(sample)
        data = self.request('GET', sample, 'export/bundle')
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            self.assertIn('particles.csv', z.namelist())
            self.assertIn('labels.tiff', z.namelist())
            result = json.loads(z.read('analysis.json'))
            self.assertEqual(result['count'], 40)
            self.assertEqual(len(z.read('particles.csv').decode('utf-8-sig').splitlines()), 41)
        report = self.request('GET', sample, 'report').decode()
        self.assertIn('data:image/png;base64,', report)
        self.assertIn('粒径分布', report)
        labels = self.request('GET', sample, 'export/labels-tiff')
        image = Image.open(io.BytesIO(labels))
        self.assertEqual(np.asarray(image).max(), 40)

    def test_revisions_reject_stale_write(self):
        sample = self.demo()
        self.analyze(sample)
        response = self.client.post('/api/samples/' + sample['id'] + '/analyze', json={'revision': 0})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.meta(sample)['revision'], 1)

    def test_input_validation(self):
        sample = self.demo()
        for payload in [{'threshold': -1}, {'sigma': 'nan'}, {'min_area': True}, {'adaptive_window': 20}, {'polarity': 'bad'}]:
            response = self.client.post('/api/samples/' + sample['id'] + '/analyze', json={'revision': 0, **payload})
            self.assertEqual(response.status_code, 400, payload)
        response = self.client.post('/api/samples', data={'file': (io.BytesIO(b'not an image'), 'bad.png')})
        self.assertEqual(response.status_code, 400)

    def test_same_sample_comparison_and_registration(self):
        first = self.demo()
        second = self.demo()
        self.analyze(first)
        self.analyze(second)
        report = self.client.post('/api/compare', json={'first': first['id'], 'second': second['id']})
        self.assertEqual(report.status_code, 200)
        self.assertEqual(report.json['wasserstein_distance'], 0)
        reg = self.client.post('/api/registration', json={'reference': first['id'], 'moving': second['id']})
        self.assertEqual(reg.status_code, 200)
        self.assertAlmostEqual(reg.json['transform']['dx'], 0, places=3)
        self.assertAlmostEqual(reg.json['after']['ssim_11px_uniform'], 1)

    def test_batch_recipe_and_restart_persistence(self):
        samples = [self.demo(), self.demo('touching')]
        recipe = self.client.post('/api/recipes', json={'name': 'test', 'segmentation': {'method': 'otsu'}}).json
        response = self.client.post('/api/jobs', json={'sample_ids': [s['id'] for s in samples], 'recipe_id': recipe['id']})
        self.assertEqual(response.status_code, 202)
        future = self.app.extensions['jobs'].futures[response.json['id']]
        future.result(timeout=30)
        job = self.client.get('/api/jobs/' + response.json['id']).json
        self.assertEqual(job['status'], 'completed')
        self.assertEqual(job['completed'], 2)
        counts = [self.request('GET', s, 'result')['count'] for s in samples]
        self.assertEqual(counts, [40, 12])
        another = create_app(self.temp.name)
        try:
            c = another.test_client()
            saved = c.get('/api/samples/' + samples[0]['id'] + '/result').json
            self.assertEqual(saved['count'], 40)
            self.assertEqual(c.get('/api/jobs/' + job['id']).json['status'], 'completed')
        finally:
            another.extensions['jobs'].executor.shutdown(wait=True)

    def test_boundary_and_block_uncertainty(self):
        sample = self.demo()
        self.analyze(sample)
        error = self.request('POST', sample, 'boundary-sensitivity', {'pixels': 1})
        for particle in error['particles']:
            self.assertLess(particle['diameter_eroded'], particle['diameter'])
            self.assertGreater(particle['diameter_dilated'], particle['diameter'])
        block = self.request('POST', sample, 'block-bootstrap', {'repeats': 200})
        self.assertGreater(block['upper'], block['lower'])

    def test_affine_landmarks_recover_translation(self):
        transform = affine_landmarks([[0, 0], [10, 0], [0, 10]], [[4, 7], [14, 7], [4, 17]])
        np.testing.assert_allclose(transform['matrix_xy'], np.eye(2), atol=1e-10)
        np.testing.assert_allclose(transform['offset_xy'], [4, 7], atol=1e-10)
        self.assertLess(transform['rmse'], 1e-10)

    def test_phase_correlation_recovers_shift(self):
        image, _, _ = generate_sample('isolated')
        moved, _ = translate(image, 12, -7)
        transform = phase_correlation(image, moved)
        self.assertAlmostEqual(transform['dx'], -12, delta=0.3)
        self.assertAlmostEqual(transform['dy'], 7, delta=0.3)

    def test_mosaic_canvas_and_provenance(self):
        first, second = self.demo(), self.demo()
        response = self.client.post('/api/mosaic', json={'reference': first['id'], 'moving': second['id'], 'dx': 400, 'dy': 0})
        self.assertEqual(response.status_code, 200, response.json)
        self.assertEqual(response.json['sample']['width'], 1360)
        self.assertEqual(response.json['sample']['height'], 640)
        self.assertEqual(response.json['details']['overlap_pixels'], 560 * 640)

    def test_skeleton_preserves_thin_structure(self):
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:40, 23:28] = True
        skeleton, iterations, converged = skeletonize(mask)
        self.assertTrue(converged)
        self.assertGreater(skeleton.sum(), 20)
        self.assertLess(skeleton.sum(), 40)
        self.assertEqual(ndi.label(skeleton, np.ones((3, 3)))[1], 1)

    def test_crofton_circle_approximation(self):
        y, x = np.indices((101, 101))
        circle = (x - 50) ** 2 + (y - 50) ** 2 <= 30 ** 2
        perimeter = crofton_perimeter(circle)
        self.assertAlmostEqual(perimeter, 2 * np.pi * 30, delta=5)

    @staticmethod
    def _sweep_parameters(**overrides):
        params = {
            'method': 'manual', 'threshold': 120, 'channel': 'gray',
            'polarity': 'dark', 'percentile': 25, 'sigma': 1,
            'adaptive_window': 41, 'adaptive_offset': 10,
            'min_area': 40, 'max_area': 16000000, 'opening': 0,
            'closing': 0, 'fill_holes': True, 'exclude_edge': True,
            'split_touching': False, 'seed_distance': 12, 'connectivity': 8,
        }
        params.update(overrides)
        return params

    @staticmethod
    def _reference_sweep(image, parameters, region, thresholds):
        """逐阈值独立分析，作为共享计算扫描的参照实现。"""
        selected = region_mask(image.size, region)
        gray = channel_array(image, parameters['channel'])
        if parameters['sigma']:
            gray = ndi.gaussian_filter(gray, parameters['sigma'])
        rows = []
        for value in thresholds:
            if parameters['polarity'] == 'dark':
                binary = gray <= value
            else:
                binary = gray > value
            binary &= selected
            cleaned = clean_binary(binary, parameters, selected)
            labels, _ = label_binary(cleaned, parameters)
            filtered, details = filter_labels(labels, parameters, selected)
            count = details['accepted_components']
            foreground = int(np.count_nonzero(filtered))
            rows.append({
                'threshold': float(value),
                'count': count,
                'coverage': round(foreground / max(int(selected.sum()), 1) * 100, 4),
                'foreground_pixels': foreground,
                'mean_area_px': round(foreground / count, 4) if count else 0,
            })
        return rows

    def test_sweep_matches_per_threshold_analysis_for_every_cleanup_option(self):
        # 穷尽组合放在小图单元测试上，覆盖全部平滑/开闭/填孔/连通/边缘组合；
        # 与逐阈值独立分析（参照实现）逐字段比对，保证共享计算不改变语义。
        rng = np.random.RandomState(23)
        image = Image.fromarray(rng.randint(0, 256, (96, 128, 3), dtype=np.uint8))
        thresholds = np.linspace(20, 240, 9)
        regions = [
            {'type': 'full'},
            {'type': 'rectangle', 'x': 8, 'y': 6, 'width': 100, 'height': 80},
            {'type': 'polygon', 'points': [[10, 10], [118, 14], [110, 88],
                                            [20, 84], [8, 48]]},
        ]
        cases = 0
        for polarity in ('dark', 'light'):
            for channel in ('gray', 'red', 'blue'):
                for sigma in (0, 1.5):
                    for opening in (0, 2):
                        for closing in (0, 2):
                            for fill_holes in (True, False):
                                for connectivity in (4, 8):
                                    for exclude_edge in (True, False):
                                        params = self._sweep_parameters(
                                            polarity=polarity, channel=channel,
                                            sigma=sigma, opening=opening,
                                            closing=closing,
                                            fill_holes=fill_holes,
                                            connectivity=connectivity,
                                            exclude_edge=exclude_edge)
                                        for region in regions:
                                            got = threshold_sweep(image, params, region, thresholds)
                                            want = self._reference_sweep(image, params, region, thresholds)
                                            self.assertEqual(got, want)
                                            cases += 1
        self.assertGreater(cases, 200)

    def test_sweep_on_demo_image_matches_reference_across_polarities_and_rois(self):
        sample = self.demo()
        image = self.app.extensions['analysis'].store.working_image(sample['id'])
        thresholds = np.linspace(40, 220, 11)
        regions = [
            {'type': 'full'},
            {'type': 'rectangle', 'x': 20, 'y': 20, 'width': 400, 'height': 500},
            {'type': 'polygon', 'points': [[40, 40], [500, 60], [480, 560],
                                            [80, 540], [30, 300]]},
        ]
        for polarity in ('dark', 'light'):
            for opening, closing, fill in ((0, 0, True), (2, 2, True), (1, 0, False)):
                params = self._sweep_parameters(polarity=polarity, opening=opening,
                                                closing=closing, fill_holes=fill)
                for region in regions:
                    got = threshold_sweep(image, params, region, thresholds)
                    want = self._reference_sweep(image, params, region, thresholds)
                    self.assertEqual(got, want)

    def test_sweep_area_filtering_and_empty_threshold_match_reference(self):
        sample = self.demo()
        image = self.app.extensions['analysis'].store.working_image(sample['id'])
        thresholds = np.linspace(0, 255, 13)
        for min_area, max_area in ((1, 30), (300, 100000), (100000, 16000000)):
            params = self._sweep_parameters(polarity='light', min_area=min_area,
                                            max_area=max_area, opening=1)
            got = threshold_sweep(image, params, {'type': 'full'}, thresholds)
            want = self._reference_sweep(image, params, {'type': 'full'}, thresholds)
            self.assertEqual(got, want)
        self.assertTrue(any(row['count'] == 0 and row['foreground_pixels'] == 0
                            and row['mean_area_px'] == 0 for row in got))

    def test_sweep_with_touching_separation_keeps_independent_pipeline(self):
        sample = self.demo('touching')
        image = self.app.extensions['analysis'].store.working_image(sample['id'])
        thresholds = np.linspace(50, 220, 7)
        params = self._sweep_parameters(split_touching=True, seed_distance=20,
                                        opening=1, fill_holes=True)
        got = threshold_sweep(image, params, {'type': 'full'}, thresholds)
        want = self._reference_sweep(image, params, {'type': 'full'}, thresholds)
        self.assertEqual(got, want)

    def test_sweep_does_not_reuse_results_across_channel_or_roi(self):
        sample = self.demo()
        store = self.app.extensions['analysis'].store
        image = store.working_image(sample['id'])
        thresholds = np.linspace(40, 220, 9)
        gray_params = self._sweep_parameters(channel='gray')
        red_params = self._sweep_parameters(channel='red')
        full = {'type': 'full'}
        roi = {'type': 'rectangle', 'x': 30, 'y': 30, 'width': 420, 'height': 520}
        gray_full = threshold_sweep(image, gray_params, full, thresholds)
        red_full = threshold_sweep(image, red_params, full, thresholds)
        gray_roi = threshold_sweep(image, gray_params, roi, thresholds)
        self.assertNotEqual(gray_full, red_full)
        self.assertNotEqual(gray_full, gray_roi)
        # 同输入重复调用仍稳定一致，说明缓存状态没有跨调用泄漏
        self.assertEqual(gray_full, threshold_sweep(image, gray_params, full, thresholds))
        self.assertEqual(red_full, threshold_sweep(image, red_params, full, thresholds))
        self.assertEqual(gray_roi, threshold_sweep(image, gray_params, roi, thresholds))

    def test_sweep_endpoint_does_not_change_current_analysis(self):
        sample = self.demo()
        before = self.analyze(sample, method='otsu')
        meta_before = self.meta(sample)
        response = self.client.post('/api/samples/' + sample['id'] + '/sweep', json={
            'start': 40, 'stop': 220, 'steps': 10,
            'parameters': {'polarity': 'dark', 'opening': 2, 'fill_holes': True},
        })
        self.assertEqual(response.status_code, 200, response.json)
        sweep = response.json
        self.assertEqual(len(sweep), 10)
        self.assertEqual(meta_before['revision'], self.meta(sample)['revision'])
        current = self.request('GET', sample, 'result')
        self.assertEqual(current['id'], before['id'])
        self.assertEqual(current['count'], before['count'])

    def test_sweep_shared_pipeline_is_faster_than_repeated_analysis(self):
        sample = self.demo()
        store = self.app.extensions['analysis'].store
        image = store.working_image(sample['id'])
        region = {'type': 'full'}
        thresholds = np.linspace(40, 220, 15)
        params = self._sweep_parameters(opening=2, fill_holes=True)
        repeated_start = time.perf_counter()
        repeated = self._reference_sweep(image, params, region, thresholds)
        repeated_elapsed = time.perf_counter() - repeated_start
        shared_start = time.perf_counter()
        shared = threshold_sweep(image, params, region, thresholds)
        shared_elapsed = time.perf_counter() - shared_start
        self.assertEqual(shared, repeated)
        self.assertLess(shared_elapsed, repeated_elapsed)


if __name__ == '__main__':
    unittest.main(verbosity=2)
