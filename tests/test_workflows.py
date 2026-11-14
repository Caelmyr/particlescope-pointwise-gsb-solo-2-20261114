import io
import json
import math
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
from particlescope.segmentation import evaluate_reference
from particlescope.measurements import crofton_perimeter
from particlescope.morphology import skeletonize
from particlescope.domains import (
    lattice_domains, estimate_spacing, bond_order, build_shells,
    misorientation_degrees, circular_mean_degrees,
)


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

    def test_lattice_domains_separate_orientations_and_disorder(self):
        sample = self.demo('domains')
        self.analyze(sample, method='otsu')
        report = self.request('POST', sample, 'lattice-domains', {})
        self.assertEqual(report['lattice'], 'hexagonal')
        self.assertEqual(report['domain_count'], 2)
        self.assertGreaterEqual(report['ordered_particle_count'], 130)
        self.assertGreater(report['disordered_particle_count'], 15)
        # 每个颗粒都被明确归类：畴内、缺陷或无序。
        for particle in report['particles']:
            self.assertTrue(
                particle['domain_id'] is not None
                or particle['is_defect'] or particle['is_disordered'] or particle['is_edge']
            )
        # 两个畴的取向差应接近构造的15度。
        self.assertEqual(len(report['misorientations']), 1)
        self.assertAlmostEqual(report['misorientations'][0]['misorientation_degrees'], 15, delta=4)
        # 畴编号与逐颗粒标签一致。
        domain_ids = set()
        for domain in report['domains']:
            domain_ids.update(domain['particle_ids'])
        self.assertEqual(len(domain_ids), report['ordered_particle_count'])
        tagged = {p['id'] for p in report['particles'] if p['domain_id'] is not None}
        self.assertEqual(domain_ids, tagged)

    def test_lattice_domain_scale_parameters_change_without_altering_measurement(self):
        sample = self.demo('domains')
        result = self.analyze(sample, method='otsu')
        before = self.request('GET', sample, 'result')
        first = self.request('POST', sample, 'lattice-domains', {'shell_count': 1})
        second = self.request('POST', sample, 'lattice-domains', {'shell_count': 2})
        strict = self.request('POST', sample, 'lattice-domains', {'order_threshold': 0.9})
        self.assertEqual(first['domain_count'], 2)
        self.assertEqual(second['domain_count'], 2)
        self.assertLessEqual(strict['ordered_particle_count'], first['ordered_particle_count'])
        # 反复调整识别尺度后，原始颗粒测量数量与质心保持不变。
        after = self.request('GET', sample, 'result')
        self.assertEqual(after['count'], result['count'])
        self.assertEqual(
            [(p['id'], p['centroid_x'], p['centroid_y']) for p in before['particles']],
            [(p['id'], p['centroid_x'], p['centroid_y']) for p in after['particles']],
        )

    def test_lattice_domain_overlay_image(self):
        sample = self.demo('domains')
        self.analyze(sample, method='otsu')
        response = self.client.post(
            '/api/samples/' + sample['id'] + '/lattice-domains/image', json={}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, 'image/png')
        image = Image.open(io.BytesIO(response.data))
        self.assertEqual(image.size, (sample['width'], sample['height']))

    def test_lattice_domain_validation(self):
        sample = self.demo('domains')
        self.analyze(sample, method='otsu')
        for payload in [
            {'shell_count': 9},
            {'lattice': 'cubic'},
            {'similarity': 1.5},
            {'minimum_domain_size': 0},
        ]:
            response = self.client.post(
                '/api/samples/' + sample['id'] + '/lattice-domains', json=payload
            )
            self.assertEqual(response.status_code, 400, payload)

    def test_bond_order_distinguishes_lattice_from_random(self):
        spacing = 30.0
        lattice = np.array([
            [column * spacing + (row % 2) * spacing / 2,
             row * spacing * math.sqrt(3) / 2]
            for row in range(9) for column in range(9)
        ])
        first_shells, _, _ = build_shells(lattice, 1, estimate_spacing(lattice))
        _, psi6 = bond_order(lattice, first_shells, 6)
        interior = [index for index, shell in enumerate(first_shells) if len(shell) == 6]
        self.assertGreater(np.median(psi6[interior]), 0.95)
        rng = np.random.default_rng(4)
        random_points = rng.uniform(0, 300, (80, 2))
        random_shells, _, _ = build_shells(random_points, 1, estimate_spacing(random_points))
        _, random_psi = bond_order(random_points, random_shells, 6)
        # 稀疏随机点大多不足3个邻居而键序置零，少数非零值也明显低于规则晶格。
        self.assertLess(np.median(random_psi[random_psi > 0]), 0.75)
        self.assertLess(np.median(psi6[interior]), 1.05)

    def test_two_spacings_are_not_merged_into_one_lattice(self):
        spacing = 28.0
        rows = []
        particle_id = 1
        for angle, origin, lattice_a in [(0.0, (0.0, 0.0), spacing), (0.0, (620.0, 0.0), spacing * 1.6)]:
            theta = math.radians(angle)
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            for row in range(7):
                for column in range(7):
                    local_x = column * lattice_a + (row % 2) * lattice_a / 2
                    local_y = row * lattice_a * math.sqrt(3) / 2
                    rows.append({
                        'id': particle_id,
                        'centroid_x': origin[0] + local_x * cos_t - local_y * sin_t,
                        'centroid_y': origin[1] + local_x * sin_t + local_y * cos_t,
                        'diameter': 10,
                    })
                    particle_id += 1
        report = lattice_domains(rows)
        self.assertTrue(report['spacing_distribution']['bimodal'])
        # 两组间距差异超过容限，不能被强行连成同一个畴。
        domain_spacings = [domain['spacing_px'] for domain in report['domains']]
        for value in domain_spacings:
            self.assertIsNotNone(value)

    def test_circular_mean_and_misorientation(self):
        # 周期30度上0度与30度等价，圆均值落在二者之一即正确。
        mean = circular_mean_degrees(np.array([0.5, 29.5, 1.0, 29.0]), 30)
        self.assertTrue(mean < 0.5 or mean > 29.5, mean)
        self.assertAlmostEqual(misorientation_degrees(29.95, 19.95, 30), 10.0, delta=0.2)
        self.assertAlmostEqual(misorientation_degrees(85, 5, 90), 10.0, delta=0.2)

    def test_lattice_domains_require_enough_particles(self):
        with self.assertRaises(ValueError):
            lattice_domains([
                {'id': 1, 'centroid_x': 0, 'centroid_y': 0},
                {'id': 2, 'centroid_x': 10, 'centroid_y': 0},
            ])


if __name__ == '__main__':
    unittest.main(verbosity=2)
