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


class BoundaryCurvature(unittest.TestCase):
    calibration = {'scale': 1.0, 'unit': 'px', 'pixel_length': 1, 'physical_length': 1}

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

    def demo(self):
        response = self.client.post('/api/demo', json={'kind': 'isolated'})
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

    def shape(self, points=None, operation=None):
        from PIL import ImageDraw
        image = Image.new('1', (320, 320), 0)
        draw = ImageDraw.Draw(image)
        if operation:
            draw.polygon(points, fill=1)
            mask = np.asarray(image, dtype=bool)
            if operation == 'notch':
                cleared = Image.fromarray(mask)
                ImageDraw.Draw(cleared).polygon([(120, 82), (200, 82), (160, 130)], fill=0)
                mask = np.asarray(cleared, dtype=bool)
            return mask
        draw.polygon(points, fill=1)
        return np.asarray(image, dtype=bool)

    def analyse(self, mask, sigma=2.0, min_turn=30.0, holes=True):
        from particlescope.curvature import particle_curvature
        return particle_curvature(mask.astype(np.int32), 1, self.calibration, sigma, min_turn, holes)

    def rotated(self, mask, angle, shift=(0, 0)):
        from PIL import Image
        image = Image.fromarray((mask * 255).astype(np.uint8)).rotate(
            angle, resample=Image.BILINEAR, center=(160, 160), translate=shift, fillcolor=0)
        return np.asarray(image) > 127

    def test_circle_has_no_false_sharp_corners_at_any_scale(self):
        y, x = np.indices((320, 320))
        disk = (x - 160) ** 2 + (y - 160) ** 2 <= 60 ** 2
        for sigma in [1.0, 2.0, 4.0, 8.0]:
            report = self.analyse(disk, sigma)
            self.assertEqual(report['outer_contour']['summary']['corner_count'], 0, sigma)
            self.assertEqual(report['outer_contour']['summary']['notch_count'], 0, sigma)
            self.assertEqual(report['corner_count'], 0)

    def test_pixel_staircase_does_not_create_corners_under_rotation(self):
        y, x = np.indices((320, 320))
        disk = (x - 160) ** 2 + (y - 160) ** 2 <= 60 ** 2
        for angle in [0, 15, 45, 31]:
            report = self.analyse(self.rotated(disk, angle), 2.0)
            self.assertEqual(report['outer_contour']['summary']['corner_count'], 0, angle)
            self.assertEqual(report['outer_contour']['summary']['notch_count'], 0, angle)

    def test_translation_keeps_curvature_profile(self):
        y, x = np.indices((320, 320))
        disk = (x - 160) ** 2 + (y - 160) ** 2 <= 60 ** 2
        base = self.analyse(disk, 2.0)['outer_contour']
        moved = self.analyse(self.rotated(disk, 0, (2.5, 1.5)), 2.0)['outer_contour']
        self.assertAlmostEqual(base['perimeter_px'], moved['perimeter_px'], delta=5)
        self.assertEqual(base['summary']['corner_count'], moved['summary']['corner_count'])
        self.assertEqual(base['summary']['notch_count'], moved['summary']['notch_count'])
        first = base['comparable_profile']
        second = moved['comparable_profile']
        self.assertAlmostEqual(first['bin_count'], second['bin_count'], delta=2)
        np.testing.assert_allclose(np.mean(first['normalized_curvature']),
                                   np.mean(second['normalized_curvature']), atol=0.35)

    def test_square_reports_four_corners_and_a_notch_reports_reflex(self):
        square = self.shape([(100, 100), (220, 100), (220, 220), (100, 220)])
        report = self.analyse(square, 2.0)['outer_contour']
        self.assertEqual(report['summary']['corner_count'], 4)
        self.assertEqual(report['summary']['notch_count'], 0)
        for feature in report['features']:
            self.assertAlmostEqual(feature['deflection_deg'], 90, delta=12)
        notched = self.shape([(100, 100), (220, 100), (220, 220), (100, 220)], 'notch')
        report = self.analyse(notched, 2.0)['outer_contour']
        self.assertEqual(report['summary']['notch_count'], 1)
        self.assertGreater(report['summary']['corner_count'], 0)
        reflex = [f for f in report['features'] if f['type'] == 'notch'][0]
        self.assertLess(reflex['deflection_deg'], 0)
        self.assertAlmostEqual(reflex['x_px'], 160, delta=12)
        self.assertAlmostEqual(reflex['y_px'], 130, delta=12)

    def test_ellipse_does_not_report_spurious_corners(self):
        y, x = np.indices((320, 320))
        ellipse = ((x - 160) / 90) ** 2 + ((y - 160) / 35) ** 2 <= 1
        for sigma in [1.0, 2.0, 4.0]:
            report = self.analyse(ellipse, sigma)
            self.assertEqual(report['outer_contour']['summary']['corner_count'], 0, sigma)

    def test_outer_contour_and_hole_are_reported_separately(self):
        y, x = np.indices((320, 320))
        ring = ((x - 160) ** 2 + (y - 160) ** 2 <= 70 ** 2) & ~((x - 160) ** 2 + (y - 160) ** 2 <= 35 ** 2)
        report = self.analyse(ring, 2.0, holes=True)
        self.assertEqual(len(report['holes']), 1)
        self.assertEqual(report['outer_contour']['kind'], 'outer')
        self.assertEqual(report['holes'][0]['kind'], 'hole')
        self.assertLess(report['holes'][0]['area_px'], report['outer_contour']['area_px'])
        excluded = self.analyse(ring, 2.0, holes=False)
        self.assertEqual(excluded['holes'], [])
        self.assertEqual(excluded['skipped_hole_count'], 1)

    def test_scale_smooths_small_angles_and_sharpens_small_ones(self):
        y, x = np.indices((320, 320))
        disk = (x - 160) ** 2 + (y - 160) ** 2 <= 60 ** 2
        notched = Image.fromarray(disk)
        from PIL import ImageDraw
        ImageDraw.Draw(notched).polygon([(120, 82), (200, 82), (160, 130)], fill=0)
        mask = np.asarray(notched, dtype=bool)
        small_scale = self.analyse(mask, 1.0)['outer_contour']
        large_scale = self.analyse(mask, 8.0)['outer_contour']
        self.assertGreater(np.max(np.abs(small_scale['curvature_px'])),
                           np.max(np.abs(large_scale['curvature_px'])))
        self.assertGreaterEqual(small_scale['summary']['corner_count'] + small_scale['summary']['notch_count'], 1)

    def test_material_bulge_into_hole_is_a_corner_and_rim_bite_is_a_notch(self):
        y, x = np.indices((320, 320))
        ring = ((x - 160) ** 2 + (y - 160) ** 2 <= 70 ** 2) & ~((x - 160) ** 2 + (y - 160) ** 2 <= 40 ** 2)
        from PIL import ImageDraw
        protruded = Image.fromarray(ring)
        ImageDraw.Draw(protruded).polygon([(160, 150), (146, 122), (174, 122)], fill=1)
        report = self.analyse(np.asarray(protruded, dtype=bool), 2.0)
        hole = report['holes'][0]
        tip = [f for f in hole['features'] if f['type'] == 'corner']
        self.assertEqual(len(tip), 1)
        self.assertAlmostEqual(tip[0]['x_px'], 160, delta=6)
        self.assertAlmostEqual(tip[0]['y_px'], 150, delta=6)
        self.assertGreater(tip[0]['deflection_deg'], 60)
        bitten = Image.fromarray(ring)
        ImageDraw.Draw(bitten).polygon([(160, 160), (140, 120), (180, 120)], fill=0)
        hole = self.analyse(np.asarray(bitten, dtype=bool), 2.0)['holes'][0]
        notches = [f for f in hole['features'] if f['type'] == 'notch']
        self.assertGreaterEqual(len(notches), 2)
        for feature in notches:
            self.assertLess(feature['deflection_deg'], 0)

    def test_scale_stability_reports_matching_counts_for_a_square(self):
        square = self.shape([(100, 100), (220, 100), (220, 220), (100, 220)])
        report = self.analyse(square, 2.0)
        self.assertTrue(report['scale_stability']['stable'])
        self.assertEqual(len(report['scale_stability']['tiers']), 3)
        for tier in report['scale_stability']['tiers']:
            self.assertEqual(tier['corner_count'], 4)
            self.assertEqual(tier['notch_count'], 0)

    def test_noisy_boundary_is_not_reported_as_a_ring_of_sharp_corners(self):
        from scipy import ndimage as ndi
        y, x = np.indices((400, 400))
        angle = np.arctan2(y - 200, x - 200)
        radius = np.hypot(x - 200, y - 200)
        jitter = ndi.gaussian_filter(np.random.default_rng(11).normal(0, 5.0, (400, 400)), 6)
        rough = radius <= 140 + 1.5 * np.sin(angle * 17) + jitter
        for sigma in [2.0, 4.0]:
            report = self.analyse(rough, sigma)['outer_contour']
            total = report['summary']['corner_count'] + report['summary']['notch_count']
            self.assertLessEqual(total, 1, sigma)
        stable = self.analyse(rough, 4.0)['outer_contour']['summary']
        self.assertEqual(stable['corner_count'], 0)
        self.assertEqual(stable['notch_count'], 0)

    def test_rotation_invariant_descriptor_matches(self):
        shape = self.shape([(100, 90), (220, 90), (220, 180), (170, 180), (170, 240), (100, 240)])
        base = self.analyse(shape, 2.0)['outer_contour']
        turned = self.analyse(self.rotated(shape, 23), 2.0)['outer_contour']
        self.assertEqual(base['summary']['corner_count'], turned['summary']['corner_count'])
        self.assertEqual(base['summary']['notch_count'], turned['summary']['notch_count'])
        first = base['rotation_invariant']['relative_magnitude']
        second = turned['rotation_invariant']['relative_magnitude']
        self.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            self.assertAlmostEqual(a, b, delta=max(0.05, abs(a) * 0.2))

    def test_curvature_endpoint_keeps_labels_and_distinguishes_holes(self):
        y, x = np.indices((320, 320))
        ring = ((x - 160) ** 2 + (y - 160) ** 2 <= 70 ** 2) & ~((x - 160) ** 2 + (y - 160) ** 2 <= 35 ** 2)
        sample = self.demo()
        revision = self.meta(sample)['revision']
        self.request('POST', sample, 'analyze', {'method': 'otsu', 'fill_holes': False, 'revision': revision})
        labels_before = self.app.extensions['analysis'].labels(sample['id'])
        particle_id = int(labels_before.max())
        labels_path = self.app.extensions['store'].sample_dir(sample['id']) / 'analyses'
        target = next(labels_path.iterdir()) / 'labels.npy'
        before = target.read_bytes()
        report = self.request('POST', sample, 'boundary-curvature', {
            'particle_id': particle_id, 'sigma_px': 2.0, 'min_turn_deg': 30, 'include_holes': True,
        })
        self.assertEqual(report['particle_id'], particle_id)
        self.assertEqual(report['outer_contour']['kind'], 'outer')
        self.assertEqual(len(report['outer_contour']['curvature_px']),
                         len(report['outer_contour']['arc_px']))
        self.assertIsNotNone(report['outer_contour']['rotation_invariant']['relative_magnitude'])
        profile = report['outer_contour']['comparable_profile']
        self.assertEqual(len(profile['arc_fraction']), len(profile['normalized_curvature']))
        self.assertGreaterEqual(profile['bin_count'], 8)
        self.assertEqual(before, target.read_bytes())

    def test_circle_report_names_no_sharp_features_on_demo_particles(self):
        sample = self.demo()
        self.analyze(sample, method='otsu')
        for particle_id in [1, 5, 20]:
            for sigma in [1.0, 3.0, 6.0]:
                report = self.request('POST', sample, 'boundary-curvature', {
                    'particle_id': particle_id, 'sigma_px': sigma,
                })
                self.assertEqual(report['corner_count'], 0, (particle_id, sigma))
                self.assertEqual(report['notch_count'], 0, (particle_id, sigma))


if __name__ == '__main__':
    unittest.main(verbosity=2)
