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


class TrackingWorkflows(unittest.TestCase):
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

    def import_frame(self, positions, size=(420, 320), analyze=True):
        image = Image.new('RGB', size, (228, 233, 232))
        draw = ImageDraw.Draw(image)
        for x, y in positions:
            r = 9
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(45, 60, 70))
        stream = io.BytesIO()
        image.save(stream, 'PNG')
        stream.seek(0)
        response = self.client.post('/api/samples', data={'files': (stream, 'frame.png')}, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 201, response.json)
        sample = response.json['imported'][0]
        if analyze:
            revision = self.client.get('/api/samples/' + sample['id']).json['meta']['revision']
            analysis = self.client.post('/api/samples/' + sample['id'] + '/analyze', json={
                'revision': revision, 'method': 'manual', 'threshold': 120,
                'min_area': 20, 'exclude_edge': False,
            })
            self.assertEqual(analysis.status_code, 200, analysis.json)
        return sample

    def make_sequence(self, frames_positions, expected=201, **kwargs):
        samples = [self.import_frame(p) for p in frames_positions]
        payload = {
            'name': '测试时序',
            'sample_ids': [s['id'] for s in samples],
            'interval': 1,
            'interval_unit': 'frame',
        }
        payload.update(kwargs)
        response = self.client.post('/api/sequences', json=payload)
        self.assertEqual(response.status_code, expected, response.json)
        return samples, response.json

    def sequence(self, sid):
        response = self.client.get('/api/sequences/' + sid)
        self.assertEqual(response.status_code, 200, response.json)
        return response.json

    def mutate(self, method, sid, path, data):
        response = self.client.open(f'/api/sequences/{sid}/{path}', method=method, json=data)
        self.assertEqual(response.status_code, 200, response.json)
        return response.json

    def test_tracking_basic_trajectory_and_velocity(self):
        frames = [
            [(60, 60), (200, 100), (350, 250)],
            [(70, 60), (208, 105), (340, 250)],
            [(80, 60), (216, 110), (330, 250)],
            [(90, 60), (224, 115), (320, 250)],
        ]
        samples, sequence = self.make_sequence(frames, interval=2, interval_unit='min')
        self.assertEqual(sequence['summary']['track_count'], 3)
        self.assertEqual(sequence['summary']['complete_tracks'], 3)
        self.assertEqual(sequence['summary']['pending_count'], 0)
        self.assertEqual(sequence['velocity_unit'], 'px/min')
        tracks = {t['id']: t for t in sequence['tracks']}
        self.assertEqual(set(tracks), {'T1', 'T2', 'T3'})
        first = tracks['T1']
        self.assertEqual(first['point_count'], 4)
        self.assertEqual(first['gap_count'], 0)
        self.assertAlmostEqual(first['mean_speed'], 5.0, places=4)
        self.assertAlmostEqual(first['points'][0]['speed'], 5.0, places=4)
        self.assertAlmostEqual(first['net_displacement'], 30.0, places=4)
        second = tracks['T2']
        self.assertAlmostEqual(second['mean_speed'], (8 ** 2 + 5 ** 2) ** 0.5 / 2, places=4)
        self.assertEqual(len(sequence['links']), 9)
        sid = sequence['id']
        picked = self.client.get(f'/api/sequences/{sid}/pick?frame=0&x=60&y=60')
        self.assertEqual(picked.status_code, 200)
        self.assertEqual(picked.json['id'], 1)
        self.assertTrue(picked.json['included'])
        overlay = self.client.get(f'/api/sequences/{sid}/overlay/2')
        self.assertEqual(overlay.status_code, 200)
        self.assertTrue(overlay.data.startswith(b'\x89PNG'))
        focused = self.client.get(f'/api/sequences/{sid}/overlay/2?track=T1')
        self.assertEqual(focused.status_code, 200)
        csv_data = self.client.get(f'/api/sequences/{sid}/export/tracks.csv').data.decode('utf-8-sig')
        lines = csv_data.strip().splitlines()
        self.assertEqual(len(lines), 13)
        self.assertIn('track_id', lines[0])
        duplicate = self.client.post(f'/api/sequences/{sid}/links', json={
            'revision': sequence['revision'], 'from_frame': 0, 'from_id': 1, 'to_frame': 1, 'to_id': 1,
        })
        self.assertEqual(duplicate.status_code, 409)

    def test_tracking_crossing_requires_confirmation(self):
        frames = [
            [(60, 100), (60, 200)],
            [(85, 140), (85, 160)],
            [(110, 170), (110, 130)],
        ]
        _, sequence = self.make_sequence(frames, settings={'max_distance': 60})
        sid = sequence['id']
        self.assertEqual(len([l for l in sequence['links'] if l['from_frame'] == 0]), 2)
        self.assertEqual(len([l for l in sequence['links'] if l['from_frame'] == 1]), 0)
        keys = {p['key'] for p in sequence['pending']}
        self.assertEqual(keys, {'a1:1', 'a1:2'})
        item = next(p for p in sequence['pending'] if p['key'] == 'a1:1')
        self.assertEqual(item['kind'], 'ambiguous')
        self.assertEqual([c['to_id'] for c in item['candidates']], [1, 2])
        state = self.mutate('POST', sid, 'pending/a1:1/resolve', {
            'revision': sequence['revision'], 'action': 'link', 'to_id': 2,
        })
        self.assertIn('a1:2', {p['key'] for p in state['pending']})
        state = self.mutate('POST', sid, 'relink', {'revision': state['revision']})
        self.assertEqual(state['pending'], [])
        pair_links = [l for l in state['links'] if l['from_frame'] == 1]
        self.assertEqual(len(pair_links), 2)
        follow = next(l for l in pair_links if l['from_id'] == 2)
        self.assertEqual((follow['to_frame'], follow['to_id']), (2, 1))
        self.assertEqual(follow['source'], 'auto')
        full = [t for t in state['tracks'] if t['point_count'] == 3]
        self.assertEqual(len(full), 2)
        track_a = next(t for t in full if any(p['frame'] == 2 and p['particle_id'] == 2 for p in t['points']))
        nodes = [[p['frame'], p['particle_id']] for p in track_a['points']]
        self.assertEqual(nodes, [[0, 1], [1, 1], [2, 2]])
        state = self.mutate('POST', sid, 'links/confirm', {
            'revision': state['revision'], 'from_frame': 0, 'from_id': 1,
        })
        state = self.mutate('POST', sid, 'relink', {'revision': state['revision']})
        confirmed = next(l for l in state['links'] if l['from_frame'] == 0 and l['from_id'] == 1)
        self.assertEqual(confirmed['status'], 'confirmed')
        state = self.mutate('DELETE', sid, 'links', {
            'revision': state['revision'], 'from_frame': 0, 'from_id': 1,
        })
        self.assertTrue(state['affected']['changed'])
        state = self.mutate('POST', sid, 'links', {
            'revision': state['revision'], 'from_frame': 0, 'from_id': 1, 'to_frame': 1, 'to_id': 1,
        })
        restored = next(t for t in state['tracks'] if t['id'] == track_a['id'])
        self.assertEqual([[p['frame'], p['particle_id']] for p in restored['points']], [[0, 1], [1, 1], [2, 2]])

    def test_tracking_gap_and_border_events(self):
        frames = [
            [(100, 100), (300, 200), (380, 40), (380, 260)],
            [(110, 100), (300, 200), (405, 40), (405, 260)],
            [(300, 200), (15, 300), (415, 295)],
            [(130, 100), (300, 200), (15, 300)],
        ]
        _, sequence = self.make_sequence(frames, settings={'max_distance': 25, 'border_margin': 20})
        sid = sequence['id']
        keys = {p['key']: p for p in sequence['pending']}
        self.assertEqual(set(keys), {'g1:2', 'b1:4'})
        self.assertEqual(keys['g1:2']['kind'], 'gap')
        self.assertEqual(keys['g1:2']['to_frame'], 3)
        self.assertEqual(keys['b1:4']['kind'], 'border')
        events = {(e['frame'], e['particle_id'], e['kind']) for e in sequence['events']}
        self.assertIn((1, 1, 'left_frame'), events)
        self.assertIn((1, 4, 'left_frame'), events)
        self.assertIn((2, 3, 'entered_frame'), events)
        quiet = next(t for t in sequence['tracks'] if t['point_count'] == 4)
        self.assertEqual(quiet['gap_count'], 0)
        state = self.mutate('POST', sid, 'pending/g1:2/resolve', {
            'revision': sequence['revision'], 'action': 'link', 'to_id': 1,
        })
        gap_track = next(t for t in state['tracks'] if t['gap_count'] == 1)
        self.assertEqual([[p['frame'], p['particle_id']] for p in gap_track['points']], [[0, 2], [1, 2], [3, 1]])
        gap_step = next(s for s in gap_track['steps'] if s['to_frame'] == 3)
        self.assertEqual(gap_step['dt'], 2)
        self.assertAlmostEqual(gap_step['speed'], 10.0, places=4)
        state = self.mutate('POST', sid, 'pending/b1:4/resolve', {
            'revision': state['revision'], 'action': 'reject',
        })
        self.assertEqual(len(state['rejected']), 1)
        state = self.mutate('POST', sid, 'relink', {'revision': state['revision']})
        self.assertNotIn('b1:4', {p['key'] for p in state['pending']})
        state = self.mutate('POST', sid, 'rejections/clear', {'revision': state['revision']})
        self.assertIn('b1:4', {p['key'] for p in state['pending']})

    def test_tracking_preserves_analyses_and_validates(self):
        raw = self.import_frame([(50, 50)], analyze=False)
        frames = [[(80, 80), (250, 200)], [(90, 80), (250, 200)], [(100, 80), (250, 200)]]
        samples, sequence = self.make_sequence(frames)
        sid = sequence['id']
        for payload in [
            {'name': 'x', 'sample_ids': [samples[0]['id']]},
            {'name': 'x', 'sample_ids': [samples[0]['id'], raw['id']]},
            {'name': 'x', 'sample_ids': [samples[0]['id'], samples[1]['id']], 'settings': {'max_distance': -1}},
        ]:
            response = self.client.post('/api/sequences', json=payload)
            self.assertEqual(response.status_code, 400, payload)
        other_size = self.import_frame([(30, 30)], size=(400, 300))
        response = self.client.post('/api/sequences', json={'name': 'x', 'sample_ids': [samples[0]['id'], other_size['id']]})
        self.assertEqual(response.status_code, 400)
        before = []
        for sample in samples:
            meta = self.client.get('/api/samples/' + sample['id']).json['meta']
            result = self.client.get('/api/samples/' + sample['id'] + '/result').json
            before.append((meta['revision'], meta['analysis_id'], result['count']))
        state = self.mutate('DELETE', sid, 'links', {'revision': sequence['revision'], 'from_frame': 0, 'from_id': 1})
        state = self.mutate('POST', sid, 'links', {
            'revision': state['revision'], 'from_frame': 0, 'from_id': 1, 'to_frame': 1, 'to_id': 1,
        })
        state = self.mutate('POST', sid, 'relink', {'revision': state['revision']})
        stale = self.client.post(f'/api/sequences/{sid}/relink', json={'revision': 999})
        self.assertEqual(stale.status_code, 409)
        missing = self.client.post(f'/api/sequences/{sid}/links', json={'from_frame': 0, 'from_id': 1, 'to_id': 1})
        self.assertEqual(missing.status_code, 400)
        for sample, (revision, analysis_id, count) in zip(samples, before):
            meta = self.client.get('/api/samples/' + sample['id']).json['meta']
            result = self.client.get('/api/samples/' + sample['id'] + '/result').json
            self.assertEqual(meta['revision'], revision)
            self.assertEqual(meta['analysis_id'], analysis_id)
            self.assertEqual(result['count'], count)
            self.assertEqual(result['id'], analysis_id)
        listed = self.client.get('/api/sequences').json
        self.assertTrue(any(s['id'] == sid for s in listed))
        deleted = self.client.open(f'/api/sequences/{sid}', method='DELETE', json={'revision': state['revision']})
        self.assertEqual(deleted.status_code, 200)
        gone = self.client.get(f'/api/sequences/{sid}')
        self.assertEqual(gone.status_code, 400)

    def test_tracking_rebind_frame(self):
        frames = [
            [(80, 80), (250, 200)],
            [(90, 80), (250, 200)],
            [(100, 80), (250, 200)],
        ]
        samples, sequence = self.make_sequence(frames)
        sid = sequence['id']
        target = samples[1]
        revision = self.client.get('/api/samples/' + target['id']).json['meta']['revision']
        again = self.client.post('/api/samples/' + target['id'] + '/analyze', json={
            'revision': revision, 'method': 'manual', 'threshold': 120, 'min_area': 20, 'exclude_edge': False,
        })
        self.assertEqual(again.status_code, 200, again.json)
        new_analysis = again.json['id']
        state = self.sequence(sid)
        self.assertFalse(state['frames'][1]['is_current_analysis'])
        state = self.mutate('POST', sid, 'frames/1/rebind', {'revision': state['revision']})
        self.assertTrue(state['frames'][1]['is_current_analysis'])
        self.assertEqual(state['frames'][1]['analysis_id'], new_analysis)
        touching = [l for l in state['links'] if l['from_frame'] == 1 or l['to_frame'] == 1]
        self.assertEqual(touching, [])
        self.assertTrue(all(t['point_count'] == 1 for t in state['tracks']))
        self.assertTrue(state['affected']['changed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
