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
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from particlescope import create_app
from particlescope.rendering import png_bytes
from particlescope.synthetic import generate_sample
from particlescope.registration import (
    phase_correlation, translate, affine_landmarks, assess_registration,
    correlation_candidates, period_vectors, spatial_verification,
    no_data_overlay, overlap_fraction,
)
from particlescope.segmentation import evaluate_reference
from particlescope.measurements import crofton_perimeter
from particlescope.morphology import skeletonize


def periodic_lattice(width=1400, height=1000, spacing=80, jitter=0.5, radius=14, seed=2):
    """生成带轻微位置抖动的周期排列微球大视野，用于模拟两张显微镜照片。"""
    rng = np.random.default_rng(seed)
    image = Image.new('L', (width, height), 235)
    draw = ImageDraw.Draw(image)
    for row in range(height // spacing + 2):
        for column in range(width // spacing + 2):
            x = spacing // 2 + column * spacing + rng.normal(0, jitter)
            y = spacing // 2 + row * spacing + rng.normal(0, jitter)
            r = radius + int(rng.integers(-1, 2))
            draw.ellipse((x - r, y - r, x + r, y + r), fill=60)
    return image.convert('RGB')


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

    def upload_image(self, image, name):
        response = self.client.post(
            '/api/samples',
            data={'file': (io.BytesIO(png_bytes(image)), name)},
            content_type='multipart/form-data',
        )
        self.assertEqual(response.status_code, 201, response.json)
        return response.json['imported'][0]

    def test_periodic_lattice_flags_ambiguous_offsets(self):
        # 两张同视野的周期微球照片错开一个完整晶格周期，存在多个近乎等价的偏移
        fov = 480
        origin_x, origin_y = 200, 150
        field = periodic_lattice(jitter=0.5)
        reference = field.crop((origin_x, origin_y, origin_x + fov, origin_y + fov))
        moving = field.crop((origin_x + 80, origin_y, origin_x + 80 + fov, origin_y + fov))
        result = assess_registration(reference, moving)
        self.assertEqual(result['status'], 'ambiguous')
        self.assertGreaterEqual(len(result.get('alternatives', [])), 1)
        # 最优候选与备选候选都应是强匹配，差距小于歧义阈值
        best = result['selected']
        self.assertGreaterEqual(best['spatial_ncc'], 0.80)
        for alternative in result['alternatives']:
            self.assertGreaterEqual(alternative['geometric_overlap_percent'], 15.0)
            self.assertGreaterEqual(alternative['spatial_ncc'], best['spatial_ncc'] - 0.10)
        # 自相关必须还原出 80px 的晶格周期向量
        vectors = period_vectors(reference)
        self.assertTrue(any(abs(dx) == 80 and dy == 0 for dx, dy, _ in vectors)
                        or any(dx == 0 and abs(dy) == 80 for dx, dy, _ in vectors))

    def test_perfectly_periodic_lattice_is_ambiguous_not_reliable(self):
        # 完全规则的晶格相位相关面会被压平，靠自相关周期向量补候选后仍应判歧义
        fov = 480
        field = periodic_lattice(jitter=0.0)
        reference = field.crop((200, 150, 200 + fov, 150 + fov))
        moving = field.crop((280, 150, 280 + fov, 150 + fov))
        result = assess_registration(reference, moving)
        self.assertEqual(result['status'], 'ambiguous')
        ncc_values = [c['spatial_ncc'] for c in result['candidates'] if c['spatial_ncc'] is not None]
        # 至少两个候选达到强匹配
        self.assertGreaterEqual(sum(v >= 0.80 for v in ncc_values), 2)

    def test_normal_translation_stays_reliable(self):
        # 普通（非周期）图像的平移配准必须继续给出唯一可靠结论
        image, _, _ = generate_sample('isolated')
        moved, _ = translate(image, 12, -7)
        result = assess_registration(image, moved)
        self.assertEqual(result['status'], 'reliable')
        self.assertAlmostEqual(result['selected']['dx'], -12, delta=0.5)
        self.assertAlmostEqual(result['selected']['dy'], 7, delta=0.5)
        self.assertGreaterEqual(result['selected']['spatial_ncc'], 0.90)
        # 同一图像视为唯一匹配
        self.assertEqual(assess_registration(image, image.copy())['status'], 'reliable')

    def test_no_overlap_is_insufficient_and_cannot_be_saved(self):
        image, _, _ = generate_sample('isolated')
        # 平移到几乎完全错开：真实重叠低于 15%
        moved, _ = translate(image, image.width - 40, 0)
        result = assess_registration(image, moved)
        self.assertEqual(result['status'], 'insufficient')
        self.assertLess(overlap_fraction(image.size, image.width - 40, 0) * 100, 15.0)
        # 空间域复核在极小重叠时直接给 None，白色填充不参与评分
        ncc, fraction = spatial_verification(image, moved, image.width - 40, 0)
        self.assertIsNone(ncc)
        # API 层拒绝把无重叠结果另存
        reference_sample = self.upload_image(image, 'no-overlap-a.png')
        moving_sample = self.upload_image(moved, 'no-overlap-b.png')
        response = self.client.post('/api/registration', json={
            'reference': reference_sample['id'], 'moving': moving_sample['id'], 'save': True,
        })
        self.assertEqual(response.status_code, 400)
        # 手动指定无重叠偏移同样被拒绝
        response = self.client.post('/api/registration', json={
            'reference': reference_sample['id'], 'moving': moving_sample['id'],
            'mode': 'manual', 'dx': image.width - 40, 'dy': 0, 'save': True,
        })
        self.assertEqual(response.status_code, 400)

    def test_ambiguous_auto_save_rejected_but_manual_choice_allowed(self):
        fov = 480
        field = periodic_lattice(jitter=0.5)
        reference = field.crop((200, 150, 200 + fov, 150 + fov))
        moving = field.crop((280, 150, 280 + fov, 150 + fov))
        reference_sample = self.upload_image(reference, 'ambig-a.png')
        moving_sample = self.upload_image(moving, 'ambig-b.png')
        # 自动模式 + 另存必须被后端拒绝
        blocked = self.client.post('/api/registration', json={
            'reference': reference_sample['id'], 'moving': moving_sample['id'], 'save': True,
        })
        self.assertEqual(blocked.status_code, 400)
        # 先试算拿到候选，再用手动模式选择最优候选另存
        preview = self.client.post('/api/registration', json={
            'reference': reference_sample['id'], 'moving': moving_sample['id'],
        })
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.json['assessment']['status'], 'ambiguous')
        chosen = preview.json['assessment']['selected']
        saved = self.client.post('/api/registration', json={
            'reference': reference_sample['id'], 'moving': moving_sample['id'],
            'mode': 'manual', 'dx': chosen['dx'], 'dy': chosen['dy'], 'save': True,
        })
        self.assertEqual(saved.status_code, 200, saved.json)
        saved.close()
        # 另存图像的无数据区不再是纯白伪装
        working_response = self.client.get(
            '/api/samples/' + saved.json['sample']['id'] + '/image/working'
        )
        working = working_response.data
        working_response.close()
        with Image.open(io.BytesIO(working)) as decoded:
            pixels = np.asarray(decoded.convert('RGB')).reshape(-1, 3)
        pure_white = int((pixels == 255).all(axis=1).sum())
        self.assertLess(pure_white / len(pixels), 0.02)

    def test_no_data_overlay_only_changes_fill_zone(self):
        image, _, _ = generate_sample('isolated')
        aligned, valid = translate(image, 60, 0, image.size)
        marked = no_data_overlay(aligned, valid)
        original = np.asarray(aligned)
        result = np.asarray(marked)
        # 真实数据像素保持不变
        np.testing.assert_array_equal(result[valid], original[valid])
        # 无数据区不存在纯白像素
        self.assertEqual(int((result[~valid] == 255).all(axis=1).sum()), 0)

    def test_correlation_candidates_returns_multiple_peaks(self):
        fov = 480
        field = periodic_lattice(jitter=1.0)
        reference = field.crop((200, 150, 200 + fov, 150 + fov))
        moving = field.crop((280, 150, 280 + fov, 150 + fov))
        candidates = correlation_candidates(reference, moving)
        self.assertGreaterEqual(len(candidates), 2)
        for candidate in candidates:
            self.assertIn('dx', candidate)
            self.assertIn('dy', candidate)
            self.assertIn('correlation_peak', candidate)

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


if __name__ == '__main__':
    unittest.main(verbosity=2)
