import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from particlescope import create_app
from particlescope.core_shell import (
    analyze_core_shell,
    auto_core_mask,
    core_mask_from_polygon,
)
from particlescope.synthetic import generate_sample


def disk_mask(shape, cy, cx, radius):
    y, x = np.ogrid[:shape[0], :shape[1]]
    return (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2


class CoreShellGeometry(unittest.TestCase):
    def test_concentric_shell_thickness_is_interpretable(self):
        shape = (120, 120)
        particle = disk_mask(shape, 60, 60, 50)
        core = disk_mask(shape, 60, 60, 30)
        gray = np.full(shape, 120.0)
        report = analyze_core_shell(gray, particle, core)
        # 同心圆形：所有射线都应有效，壳厚约为 20px
        self.assertEqual(report['invalid_ray_count'], 0)
        self.assertGreater(report['valid_ray_fraction'], 0.99)
        self.assertAlmostEqual(report['thickness_summary']['mean'], 20, delta=1.5)
        # 同心 => 偏心接近 0，壳厚均匀，CV 很小
        self.assertLess(report['eccentricity_ratio'], 0.05)
        self.assertLess(report['thickness_uniformity_cv_percent'], 5)

    def test_eccentric_core_has_thin_and_thick_side(self):
        shape = (120, 120)
        particle = disk_mask(shape, 60, 60, 50)
        core = disk_mask(shape, 70, 70, 25)  # 内核向右下偏移
        gray = np.full(shape, 120.0)
        report = analyze_core_shell(gray, particle, core)
        self.assertEqual(report['invalid_ray_count'], 0)
        self.assertGreater(report['eccentricity_ratio'], 0.15)
        self.assertGreater(
            report['thickness_summary']['maximum'] - report['thickness_summary']['minimum'],
            10,
        )
        # 质心偏移方向与人为偏移一致
        self.assertGreater(report['core_centroid'][0] - report['particle_centroid'][0], 5)
        self.assertGreater(report['core_centroid'][1] - report['particle_centroid'][1], 5)

    def test_ray_through_hole_is_rejected_not_counted(self):
        shape = (140, 140)
        particle = disk_mask(shape, 70, 70, 60)
        core = disk_mask(shape, 70, 70, 25)
        # 在壳层右侧打一个贯穿内核外、外壳内的封闭孔洞
        hole = disk_mask(shape, 70, 112, 8)
        particle_with_hole = particle & ~hole
        self.assertGreater(int(hole.sum()), 0)
        gray = np.full(shape, 120.0)
        report = analyze_core_shell(gray, particle_with_hole, core)
        self.assertEqual(report['hole_pixels'], int(hole.sum()))
        self.assertGreater(report['ray_status_counts']['hole'], 0)
        # 穿孔洞射线必须无效，且不进入壳厚统计
        self.assertEqual(
            report['ray_status_counts']['hole'],
            sum(1 for r in report['rays'] if r['status'] == 'hole'),
        )
        for ray in report['rays']:
            if ray['status'] != 'valid':
                self.assertIsNone(ray['thickness'])
        # 仍然存在大量有效射线
        self.assertGreater(report['valid_ray_fraction'], 0.7)

    def test_exterior_ray_on_dent_is_rejected(self):
        # 颗粒右侧被削平，内核恰好贴到该平边：贴壁方向壳厚为零，
        # 对应射线起点位于颗粒外轮廓上，必须标记 exterior 且不计入壳厚。
        shape = (140, 140)
        particle = disk_mask(shape, 70, 60, 55)
        particle[:, 110:] = False  # 右侧平边 x=109
        core = disk_mask(shape, 70, 88, 21)  # 右缘到达平边
        self.assertEqual(int((core & ~particle).sum()), 0)
        gray = np.full(shape, 120.0)
        report = analyze_core_shell(gray, particle, core)
        self.assertGreater(report['ray_status_counts']['exterior'], 0)
        for ray in report['rays']:
            if ray['status'] == 'exterior':
                self.assertIsNone(ray['thickness'])
        # 开口凹陷（与外部连通）不应被误判为封闭孔洞
        self.assertEqual(report['hole_pixels'], 0)

    def test_open_dent_gives_short_valid_thickness_not_hole(self):
        # 与图像外部连通的凹陷：射线仍跨越外壳外轮廓，得到较短但有效的壳厚，
        # 不能被错误地当成孔洞。
        shape = (140, 140)
        particle = disk_mask(shape, 70, 70, 60)
        core = disk_mask(shape, 70, 70, 25)
        dent = disk_mask(shape, 70, 122, 18)
        dented = particle & ~dent
        gray = np.full(shape, 120.0)
        report = analyze_core_shell(gray, dented, core)
        self.assertEqual(report['ray_status_counts']['hole'], 0)
        self.assertGreater(report['valid_ray_count'], 0)
        self.assertLess(
            report['thickness_summary']['minimum'],
            report['thickness_summary']['maximum'],
        )

    def test_core_outside_particle_is_rejected(self):
        shape = (100, 100)
        particle = disk_mask(shape, 50, 50, 40)
        core = np.zeros(shape, dtype=bool)
        core[5:15, 5:15] = True  # 完全在颗粒外
        gray = np.full(shape, 120.0)
        with self.assertRaises(ValueError):
            analyze_core_shell(gray, particle, core)

    def test_auto_core_mask_selects_dark_region(self):
        shape = (120, 120)
        particle = disk_mask(shape, 60, 60, 50)
        gray = np.full(shape, 200.0)
        core = disk_mask(shape, 60, 60, 25)
        gray[core] = 40.0
        selected, threshold = auto_core_mask(gray, particle, 'dark')
        # 选出的内核应基本覆盖暗区且位于颗粒内部
        self.assertGreater(selected.sum(), core.sum() * 0.9)
        self.assertTrue(np.all(selected <= particle))
        self.assertLessEqual(threshold, 200)

    def test_polygon_rasterization(self):
        points = [[10, 10], [40, 12], [35, 40], [12, 38]]
        mask = core_mask_from_polygon((60, 60), points)
        self.assertTrue(mask[25, 25])
        self.assertFalse(mask[2, 2])


class CoreShellApi(unittest.TestCase):
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

    def _demo_analyzed(self):
        sample = self.client.post('/api/demo', json={'kind': 'core_shell'}).json
        revision = self.client.get('/api/samples/' + sample['id']).json['meta']['revision']
        result = self.client.post(
            '/api/samples/' + sample['id'] + '/analyze',
            json={'revision': revision, 'method': 'otsu', 'fill_holes': False},
        )
        self.assertEqual(result.status_code, 200, result.json)
        return sample, result.json

    def test_full_core_shell_workflow_and_export(self):
        sample, result = self._demo_analyzed()
        self.assertGreaterEqual(result['count'], 8)
        particle_id = result['particles'][0]['id']
        base = '/api/samples/' + sample['id'] + '/core-shell'

        # 手动内核：取该颗粒 bbox 内一个收缩矩形作为多边形
        row = result['particles'][0]
        bx, by = row['bbox_x'], row['bbox_y']
        w, h = row['bbox_width'], row['bbox_height']
        points = [
            [bx + w * 0.35, by + h * 0.35],
            [bx + w * 0.65, by + h * 0.35],
            [bx + w * 0.65, by + h * 0.65],
            [bx + w * 0.35, by + h * 0.65],
        ]
        created = self.client.post(base, json={
            'revision': self.client.get('/api/samples/' + sample['id']).json['meta']['revision'],
            'particle_id': particle_id, 'mode': 'manual', 'points': points,
        })
        self.assertEqual(created.status_code, 201, created.json)
        record = created.json
        self.assertIn('report', record)
        self.assertGreater(record['report']['ray_count'], 0)

        # 列表与详情
        listing = self.client.get(base).json['records']
        self.assertTrue(any(r['id'] == record['id'] for r in listing))
        detail = self.client.get(base + '/' + record['id'])
        self.assertEqual(detail.status_code, 200)

        # 叠加图与内核图可下载
        image = self.client.get(base + '/' + record['id'] + '/image')
        self.assertEqual(image.status_code, 200)
        self.assertEqual(image.mimetype, 'image/png')
        core_image = self.client.get(base + '/' + record['id'] + '/image?suffix=-core')
        self.assertEqual(core_image.status_code, 200)

        # ZIP 导出内容
        export = self.client.get(base + '/' + record['id'] + '/export')
        self.assertEqual(export.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(export.data)) as z:
            names = z.namelist()
            self.assertIn('core-shell.json', names)
            self.assertIn('rays.csv', names)
            self.assertIn('core-shell-overlay.png', names)
            self.assertIn('core-mask.png', names)
            rays_csv = z.read('rays.csv').decode('utf-8-sig')
            self.assertIn('status', rays_csv)

        # 自动内核模式也能跑通
        auto = self.client.post(base, json={
            'revision': self.client.get('/api/samples/' + sample['id']).json['meta']['revision'],
            'particle_id': particle_id, 'mode': 'auto', 'polarity': 'dark', 'channel': 'gray',
        })
        self.assertEqual(auto.status_code, 201, auto.json)
        self.assertIsNotNone(auto.json['definition']['threshold'])

        # 删除
        deleted = self.client.delete(base + '/' + record['id'], json={
            'revision': self.client.get('/api/samples/' + sample['id']).json['meta']['revision'],
        })
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(self.client.get(base + '/' + record['id']).status_code, 400)

    def test_invalid_particle_and_core_validation(self):
        sample, _ = self._demo_analyzed()
        base = '/api/samples/' + sample['id'] + '/core-shell'
        revision = self.client.get('/api/samples/' + sample['id']).json['meta']['revision']
        # 不存在的颗粒
        response = self.client.post(base, json={
            'revision': revision, 'particle_id': 9999, 'mode': 'auto', 'polarity': 'dark',
        })
        self.assertEqual(response.status_code, 400)
        # 手动模式缺少多边形
        response = self.client.post(base, json={
            'revision': revision, 'particle_id': 1, 'mode': 'manual', 'points': [[1, 2]],
        })
        self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main(verbosity=2)
