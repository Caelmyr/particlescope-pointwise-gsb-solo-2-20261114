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
from particlescope.registration import phase_correlation, translate, affine_landmarks, feather_mosaic
from particlescope.mosaic import blend, build_preview, preview_image
from particlescope.segmentation import evaluate_reference
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

    def particle_pair(self, seed_a=11, seed_b=19, dim=1.0, size=(480, 360)):
        from PIL import ImageDraw

        def make(seed, scale):
            rng = np.random.default_rng(seed)
            image = Image.new('RGB', size, (232, 236, 235))
            draw = ImageDraw.Draw(image)
            for _ in range(30):
                x = int(rng.integers(15, size[0] - 15))
                y = int(rng.integers(15, size[1] - 15))
                radius = int(rng.integers(7, 18))
                color = (rng.integers([30, 40, 50], [80, 95, 110]) * scale).astype(int)
                draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                             fill=tuple(int(c) for c in color))
            return image

        return make(seed_a, 1.0), make(seed_b, dim)

    def test_smart_seam_avoids_structure_and_keeps_exclusive_regions(self):
        first, second = self.particle_pair()
        legacy, legacy_info, placed, wa_legacy, wb_legacy, _ = blend(
            first, second, 260, 0, 30, method='feather',
        )
        smart, info, _, wa, wb, detail = blend(first, second, 260, 0, 30, method='smart')
        self.assertEqual(legacy.size, smart.size)
        self.assertEqual(info['seam_orientation'], 'vertical')
        self.assertLess(info['structure_cost_on_seam'], info['structure_cost_overlap_mean'])
        self.assertGreater(info['seam_points'], 0)
        # 两侧独有区域两种方法必须逐像素一致，且权重为纯拷贝
        only_a = placed['valid_a'] & ~placed['valid_b']
        only_b = placed['valid_b'] & ~placed['valid_a']
        np.testing.assert_array_equal(np.asarray(legacy)[only_a], np.asarray(smart)[only_a])
        np.testing.assert_array_equal(np.asarray(legacy)[only_b], np.asarray(smart)[only_b])
        np.testing.assert_array_equal(wa[only_a], 1.0)
        np.testing.assert_array_equal(wb[only_b], 1.0)
        # 重叠区内权重处处为正且互补
        overlap = placed['overlap']
        self.assertTrue(np.all((wa + wb)[overlap] > 0))
        np.testing.assert_allclose((wa + wb)[overlap], 1.0, atol=1e-9)
        # 接缝完全落在重叠框内
        x0, y0, x1, y1 = info['bbox']
        for px, py in info['seam_canvas_points']:
            self.assertTrue(x0 <= px < x1 and y0 <= py < y1)

    def test_smart_mosaic_reduces_ghost_gradient_vs_legacy(self):
        first, second = self.particle_pair()
        legacy, _, placed, wa_l, wb_l, _ = blend(first, second, 260, 0, 30, method='feather')
        smart, _, _, wa_s, wb_s, _ = blend(first, second, 260, 0, 30, method='smart')

        def band_gradient(image, wa, wb):
            band = placed['overlap'] & (wa > 0.02) & (wb > 0.02)
            gray = np.asarray(image, dtype=float).mean(axis=2)
            gradient = np.hypot(ndi.sobel(gray, 0), ndi.sobel(gray, 1))
            return float(gradient[band].mean())

        self.assertLess(band_gradient(smart, wa_s, wb_s), band_gradient(legacy, wa_l, wb_l))

    def test_luminance_gain_harmonizes_dimmed_overlap(self):
        first, second = self.particle_pair()
        # 同内容整体变暗，模拟两视野曝光差异；错位颗粒不影响背景增益估计
        dimmed = Image.fromarray((np.asarray(second).astype(float) * 0.75).clip(0, 255).astype(np.uint8))
        smart, info, _, _, _, _ = blend(first, dimmed, 260, 0, 30, method='smart', harmonize=True)
        self.assertGreater(info['luminance_gain'], 1.1)
        self.assertLess(info['luminance_mae_after_gain'], info['luminance_mae_before'])
        without, raw, _, _, _, _ = blend(first, dimmed, 260, 0, 30, method='smart', harmonize=False)
        self.assertEqual(raw['luminance_gain'], 1.0)

    def test_mosaic_canvas_negative_offset_and_no_overlap(self):
        first, second = self.particle_pair(size=(480, 360))
        negative, neg_info, _, _, _, _ = blend(first, second, -120, -80, 30, method='smart')
        self.assertEqual(negative.size, (600, 440))
        self.assertEqual(neg_info['reference_origin'], [120, 80])
        self.assertEqual(neg_info['overlap_pixels'], 360 * 280)
        gap, gap_info, _, _, _, detail = blend(first, second, 600, 0, 30, method='smart')
        self.assertEqual(gap.size, (1080, 360))
        self.assertEqual(gap_info['overlap_pixels'], 0)
        self.assertIsNone(detail)
        # 无重叠时输出等于两图直接拼接，独有区域保持原色
        gap_array = np.asarray(gap)
        self.assertTrue(np.all(gap_array[:360, :480] == np.asarray(first)))
        self.assertTrue(np.all(gap_array[:360, 600:1080] == np.asarray(second)))
        # 完全包含：画布不大于第一幅图
        small = first.crop((100, 80, 360, 300))
        contained, contained_info, _, _, _, _ = blend(first, small, 60, 40, 20, method='smart')
        self.assertEqual(contained.size, first.size)
        self.assertEqual(contained_info['overlap_pixels'], 260 * 220)
        # 横向重叠给出横向接缝
        _, horizontal, _, _, _, _ = blend(first, second, 0, 200, 30, method='smart')
        self.assertEqual(horizontal['seam_orientation'], 'horizontal')

    def test_legacy_feather_remains_pixel_identical(self):
        first, second = self.particle_pair()
        wrapped, wrapped_info = feather_mosaic(first, second, 260, 0, 30)
        direct, info, _, _, _, _ = blend(first, second, 260, 0, 30, method='feather')
        np.testing.assert_array_equal(np.asarray(wrapped), np.asarray(direct))
        self.assertEqual(wrapped_info['overlap_pixels'], info['overlap_pixels'])
        self.assertEqual(wrapped_info['canvas_width'], info['canvas_width'])

    def test_mosaic_preview_and_smart_sample_api(self):
        first, second = self.demo(), self.demo()
        preview = self.client.post('/api/mosaic/preview', json={
            'reference': first['id'], 'moving': second['id'], 'dx': 400, 'dy': 0, 'feather': 30,
        })
        self.assertEqual(preview.status_code, 200, preview.json)
        token = preview.json['preview_token']
        self.assertEqual(set(preview.json['views']), {'legacy', 'smart', 'seam', 'detail'})
        for view, path in preview.json['views'].items():
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, 'image/png')
            response.close()
        bad = self.client.get(f'/api/mosaic/preview/{token}/unknown')
        self.assertEqual(bad.status_code, 400)
        smart = self.client.post('/api/mosaic', json={
            'reference': first['id'], 'moving': second['id'],
            'dx': 400, 'dy': 0, 'feather': 30, 'method': 'smart',
        })
        self.assertEqual(smart.status_code, 200, smart.json)
        self.assertEqual(smart.json['sample']['width'], 1360)
        self.assertEqual(smart.json['details']['method'], 'smart')
        self.assertIn('luminance_gain', smart.json['details'])
        provenance = self.client.get('/api/samples/' + smart.json['sample']['id']).json['meta']
        self.assertFalse(provenance['archived'])
        legacy = self.client.post('/api/mosaic', json={
            'reference': first['id'], 'moving': second['id'],
            'dx': 400, 'dy': 0, 'feather': 30, 'method': 'feather',
        })
        self.assertEqual(legacy.json['details']['method'], 'feather')
        self.assertEqual(legacy.json['sample']['width'], 1360)

    def test_mosaic_rejects_invalid_method_and_params(self):
        first, second = self.demo(), self.demo()
        response = self.client.post('/api/mosaic', json={
            'reference': first['id'], 'moving': second['id'],
            'dx': 400, 'dy': 0, 'method': 'seam_carve',
        })
        self.assertEqual(response.status_code, 400)
        response = self.client.post('/api/mosaic/preview', json={
            'reference': first['id'], 'moving': second['id'],
            'dx': 400, 'dy': 0, 'band_scale': 99,
        })
        self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main(verbosity=2)
