import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from particlescope import create_app
from particlescope import compensation as comp
from particlescope.rendering import png_bytes
from particlescope.synthetic import SPILLOVER, generate_sample


class SpilloverCompensation(unittest.TestCase):
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

    def demo(self, kind):
        response = self.client.post('/api/demo', json={'kind': kind})
        self.assertEqual(response.status_code, 201, response.json)
        return response.json

    def meta(self, sample):
        return self.client.get('/api/samples/' + sample['id']).json['meta']

    def analyze(self, sample, **parameters):
        parameters['revision'] = self.meta(sample)['revision']
        response = self.client.post('/api/samples/' + sample['id'] + '/analyze', json=parameters)
        self.assertEqual(response.status_code, 200, response.json)
        return response.json

    def calibrate(self, red, green):
        response = self.client.post('/api/compensation/calibrate', json={
            'red_control': red['id'], 'green_control': green['id'],
        })
        self.assertEqual(response.status_code, 201, response.json)
        return response.json

    def test_same_control_is_rejected(self):
        red = self.demo('spillover_red')
        response = self.client.post('/api/compensation/calibrate', json={
            'red_control': red['id'], 'green_control': red['id'],
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn('不同', response.json['error'])

    def test_calibration_recovers_known_spillover(self):
        red = self.demo('spillover_red')
        green = self.demo('spillover_green')
        cal = self.calibrate(red, green)
        matrix = np.asarray(cal['spillover_matrix'])
        self.assertTrue(cal['reliable'])
        self.assertEqual(cal['warnings'], [])
        self.assertAlmostEqual(matrix[1, 0], SPILLOVER['red_to_green'], delta=0.06)
        self.assertAlmostEqual(matrix[0, 1], SPILLOVER['green_to_red'], delta=0.06)
        self.assertLess(cal['condition_number'], 10)
        listing = self.client.get('/api/compensation').json['calibrations']
        self.assertTrue(any(item['id'] == cal['id'] for item in listing))

    def test_apply_classifies_particles_and_preserves_original(self):
        red = self.demo('spillover_red')
        green = self.demo('spillover_green')
        mixed = self.demo('spillover_mixed')
        cal = self.calibrate(red, green)
        result = self.analyze(mixed, method='otsu', polarity='light', channel='value')
        self.assertEqual(result['count'], 40)
        original_bytes = self.client.get('/api/samples/' + mixed['id'] + '/image/original').data
        response = self.client.post(
            '/api/compensation/' + cal['id'] + '/apply', json={'sample_id': mixed['id']}
        )
        self.assertEqual(response.status_code, 201, response.json)
        run = response.json
        summary = run['summary']
        self.assertEqual(summary['particle_count'], 40)
        self.assertEqual(summary['red_only'], 15)
        self.assertEqual(summary['green_only'], 15)
        self.assertEqual(summary['double_positive'], 10)
        self.assertEqual(summary['ambiguous'], 0)
        # 原图与工作图均不得被修改
        self.assertEqual(
            original_bytes,
            self.client.get('/api/samples/' + mixed['id'] + '/image/original').data,
        )
        working = self.client.get('/api/samples/' + mixed['id'] + '/image/working').data
        self.assertEqual(original_bytes, working)
        # 每个颗粒都含校正前后红绿强度
        for row in run['particles']:
            self.assertIn('red_mean', row)
            self.assertIn('green_mean', row)
            self.assertIn('raw_red_mean', row)
            self.assertGreaterEqual(row['red_integrated'], 0)
            self.assertGreaterEqual(row['green_integrated'], 0)
        # 双阳性颗粒同时具备两条通道的真实信号
        for row in run['particles']:
            if row['classification'] == 'double_positive':
                self.assertTrue(row['red_positive'] and row['green_positive'])
        # 资源与 CSV 可访问
        for kind in ['observed', 'separated', 'classification', 'separated-red', 'separated-green', 'run']:
            self.assertEqual(
                self.client.get(f"/api/compensation/{cal['id']}/run/{run['id']}/{kind}").status_code,
                200,
            )
        csv_data = self.client.get(f"/api/compensation/{cal['id']}/run/{run['id']}/csv").data.decode('utf-8-sig')
        self.assertEqual(len(csv_data.splitlines()), 41)
        self.assertIn('classification', csv_data.splitlines()[0])

    def test_size_mismatch_is_rejected_with_clear_message(self):
        red = self.demo('spillover_red')
        green = self.demo('spillover_green')
        cal = self.calibrate(red, green)
        small = Image.new('RGB', (480, 320), (10, 12, 14))
        upload = self.client.post(
            '/api/samples',
            data={'files': (io.BytesIO(png_bytes(small)), 'small.png')},
        )
        small_id = upload.json['imported'][0]['id']
        self.analyze({'id': small_id}, method='otsu', polarity='light')
        response = self.client.post(
            '/api/compensation/' + cal['id'] + '/apply', json={'sample_id': small_id}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn('尺寸', response.json['error'])
        self.assertIn('不一致', response.json['error'])

    def test_control_size_mismatch_and_unknown_asset(self):
        red = self.demo('spillover_red')
        small = Image.new('RGB', (480, 320), (10, 12, 14))
        upload = self.client.post(
            '/api/samples',
            data={'files': (io.BytesIO(png_bytes(small)), 'small.png')},
        )
        small_id = upload.json['imported'][0]['id']
        response = self.client.post('/api/compensation/calibrate', json={
            'red_control': red['id'], 'green_control': small_id,
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn('尺寸不一致', response.json['error'])
        # 未知资源类型
        cal = self.calibrate(red, self.demo('spillover_green'))
        missing = self.client.get(
            f"/api/compensation/{cal['id']}/run/00000000000000000000000000000000/unknown"
        )
        self.assertEqual(missing.status_code, 400)

    def test_degraded_blank_control_is_flagged_unreliable(self):
        green = self.demo('spillover_green')
        blank = Image.new('RGB', (960, 640), (8, 9, 10))
        upload = self.client.post(
            '/api/samples',
            data={'files': (io.BytesIO(png_bytes(blank)), 'blank.png')},
        )
        blank_id = upload.json['imported'][0]['id']
        response = self.client.post('/api/compensation/calibrate', json={
            'red_control': blank_id, 'green_control': green['id'],
        })
        self.assertEqual(response.status_code, 201)
        cal = response.json
        self.assertFalse(cal['reliable'])
        self.assertTrue(any('退化' in w or '过弱' in w for w in cal['warnings']))

    def test_saturated_mixed_sample_emits_warning(self):
        red = self.demo('spillover_red')
        green = self.demo('spillover_green')
        cal = self.calibrate(red, green)
        saturated, _, _ = generate_sample('spillover_mixed')
        array = np.array(saturated, dtype=np.uint8, copy=True)
        # 把红通道大片区域推到饱和，模拟过曝
        array[:, :, 0] = np.where(array[:, :, 0] > 40, 255, array[:, :, 0])
        saturated = Image.fromarray(array)
        upload = self.client.post(
            '/api/samples',
            data={'files': (io.BytesIO(png_bytes(saturated)), 'saturated.png')},
        )
        sample_id = upload.json['imported'][0]['id']
        self.analyze({'id': sample_id}, method='otsu', polarity='light', channel='value')
        response = self.client.post(
            '/api/compensation/' + cal['id'] + '/apply', json={'sample_id': sample_id}
        )
        self.assertEqual(response.status_code, 201, response.json)
        self.assertTrue(any('饱和' in w for w in response.json['warnings']))
        self.assertTrue(any(
            any('饱和' in reason for reason in row['ambiguous_reasons'])
            for row in response.json['particles']
        ))

    def test_fit_helpers_flag_inversion_and_weak_signal(self):
        blank = np.zeros((64, 64, 2), dtype=np.float64) + 5
        fit = comp.fit_single_stain(blank, 'red')
        self.assertTrue(any('退化' in w or '过弱' in w for w in fit['warnings']))
        # 主通道与串色通道负相关时给出告警
        rng = np.random.default_rng(0)
        image = np.zeros((64, 64, 2), dtype=np.float64)
        gradient = np.tile(np.linspace(40, 200, 64), (64, 1))
        image[:, :, 0] = gradient + rng.normal(0, 2, (64, 64))
        image[:, :, 1] = 240 - gradient + rng.normal(0, 2, (64, 64))
        inverse_fit = comp.fit_single_stain(image, 'red')
        self.assertTrue(any('负相关' in w for w in inverse_fit['warnings']))
        self.assertGreaterEqual(inverse_fit['coefficient'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
