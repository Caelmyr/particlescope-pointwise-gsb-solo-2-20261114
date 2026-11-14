import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from particlescope import create_app
from particlescope.vectorize import (
    ContourPlan, build_contours, contour_svg_string, preview_report,
    validate_tolerance, _rasterize_loops, _segment_key,
)
from particlescope.validation import ValidationError


def ring_and_touchers():
    """Particle 1 has a hole and touches particle 2; particle 3 is separate."""
    labels = np.zeros((40, 60), dtype=np.int32)
    labels[5:20, 5:20] = 1
    labels[9:16, 9:16] = 0
    labels[5:20, 20:35] = 2
    labels[25:35, 45:56] = 3
    rows = [
        {'id': 1, 'centroid_x': 12, 'centroid_y': 12},
        {'id': 2, 'centroid_x': 27, 'centroid_y': 12},
        {'id': 3, 'centroid_x': 50, 'centroid_y': 30},
    ]
    return labels, rows


class VectorizeUnitTests(unittest.TestCase):
    def test_tolerance_validation(self):
        self.assertEqual(validate_tolerance('1.5'), 1.5)
        for bad in (-0.1, 11, float('inf'), float('nan'), 'abc', None):
            with self.assertRaises(ValidationError):
                validate_tolerance(bad)

    def test_hole_loop_detected_and_never_filled(self):
        labels, rows = ring_and_touchers()
        plan = ContourPlan(labels, rows)
        self.assertEqual(sum(plan.hole_flags[1]), 1)
        self.assertEqual(sum(plan.hole_flags[2]), 0)
        for tolerance in (0.25, 1, 4, 10):
            built = build_contours(labels, rows, tolerance)
            raster = _rasterize_loops(
                built['loops'][1], (slice(5, 20), slice(5, 20)),
            )
            self.assertFalse(raster[7, 7], f'hole filled at tolerance {tolerance}')

    def test_separated_particles_never_intersect(self):
        labels, rows = ring_and_touchers()
        for tolerance in (0.25, 2, 8):
            built = build_contours(labels, rows, tolerance)
            placed_masks = []
            for pid, sy, sx in (
                (2, (5, 20), (20, 35)),
                (3, (25, 35), (45, 56)),
            ):
                raster = _rasterize_loops(
                    built['loops'][pid], (slice(*sy), slice(*sx)),
                )
                placed = np.zeros(labels.shape, bool)
                placed[sy[0]:sy[1], sx[0]:sx[1]] = raster
                placed_masks.append(placed)
            self.assertFalse((placed_masks[0] & placed_masks[1]).any())

    def test_touching_pair_shares_identical_border(self):
        labels, rows = ring_and_touchers()
        for tolerance in (0.25, 1, 4, 8):
            built = build_contours(labels, rows, tolerance)
            edge_owners = {}
            for pid, loops in built['loops'].items():
                for loop in loops:
                    for index in range(len(loop)):
                        key = _segment_key(loop[index], loop[(index + 1) % len(loop)])
                        edge_owners.setdefault(key, set()).add(pid)
            shared = {key: owners for key, owners in edge_owners.items() if len(owners) > 1}
            self.assertTrue(shared, 'shared border must survive simplification')
            for owners in shared.values():
                self.assertEqual(owners, {1, 2})

    def test_node_count_decreases_and_bounded_by_tolerance(self):
        labels, rows = ring_and_touchers()
        previous = None
        for tolerance in (0.25, 0.5, 1, 2, 4, 8):
            built = build_contours(labels, rows, tolerance)
            self.assertLess(built['nodes_simplified'], built['nodes_original'])
            if previous is not None:
                self.assertLessEqual(built['nodes_simplified'], previous + 2)
            previous = built['nodes_simplified']
            for stat in built['stats']:
                if not stat['repaired']:
                    self.assertLessEqual(
                        stat['max_deviation_px'],
                        stat['tolerance_applied_px'] + 1e-6,
                    )

    def test_raw_raster_loops_reconstruct_mask_exactly(self):
        labels, rows = ring_and_touchers()
        plan = ContourPlan(labels, rows)
        raster = _rasterize_loops(
            plan.loops[1], (slice(5, 20), slice(5, 20)),
        )
        np.testing.assert_array_equal(raster, labels[5:20, 5:20] == 1)

    def test_svg_keeps_evenodd_pixel_coords_and_ids(self):
        labels, rows = ring_and_touchers()
        svg = contour_svg_string(labels, rows, 1.5)
        self.assertIn('fill-rule="evenodd"', svg)
        self.assertIn('viewBox="0 0 60 40"', svg)
        self.assertIn('data-particle-id="3"', svg)
        self.assertIn('<text', svg)
        self.assertIn('image pixels', svg)
        raw = contour_svg_string(labels, rows, 0.0)
        self.assertIn('simplified=false', raw)
        self.assertNotIn('<text', raw)
        self.assertGreaterEqual(raw.count(' L '), svg.count(' L '))

    def test_preview_report_contains_two_layers(self):
        labels, rows = ring_and_touchers()
        report = preview_report(labels, rows, 1.0)
        self.assertEqual(len(report['original_paths']), 3)
        self.assertEqual(len(report['simplified_paths']), 3)
        self.assertGreater(report['reduction_percent'], 50)
        self.assertEqual(report['particles_with_holes'], 1)
        self.assertGreater(report['shared_segments'], 0)
        for item in report['particles']:
            self.assertIn('nodes_original', item)
            self.assertIn('nodes_simplified', item)


class ContourApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.app = create_app(cls.temp.name)
        cls.app.testing = True
        cls.client = cls.app.test_client()
        demo = cls.client.post('/api/demo', json={'kind': 'touching'})
        cls.sample = demo.json
        revision = cls.client.get('/api/samples/' + cls.sample['id']).json['meta']['revision']
        cls.client.post('/api/samples/' + cls.sample['id'] + '/analyze', json={
            'revision': revision, 'method': 'otsu', 'split_touching': True, 'seed_distance': 20,
        })

    @classmethod
    def tearDownClass(cls):
        cls.app.extensions['jobs'].executor.shutdown(wait=True)
        cls.temp.cleanup()

    def test_preview_endpoint(self):
        response = self.client.get(
            '/api/samples/' + self.sample['id'] + '/contour-preview?tolerance=1.5'
        )
        self.assertEqual(response.status_code, 200, response.data[:200])
        report = response.json
        self.assertLess(report['nodes_simplified'], report['nodes_original'])
        self.assertGreater(len(report['simplified_paths']), 0)
        self.assertIn('d', report['simplified_paths'][0])

    def test_preview_rejects_bad_tolerance(self):
        response = self.client.get(
            '/api/samples/' + self.sample['id'] + '/contour-preview?tolerance=99'
        )
        self.assertEqual(response.status_code, 400)

    def test_simplified_and_raw_svg_endpoints(self):
        key = self.sample['id']
        simplified = self.client.get(
            '/api/samples/' + key + '/export/contours-svg-simplified?tolerance=2'
        )
        self.assertEqual(simplified.status_code, 200)
        self.assertEqual(simplified.mimetype, 'image/svg+xml')
        body = simplified.data.decode('utf-8')
        self.assertIn('simplified=true', body)
        self.assertIn('fill-rule="evenodd"', body)
        raw = self.client.get('/api/samples/' + key + '/export/contours-svg')
        self.assertEqual(raw.status_code, 200)
        raw_body = raw.data.decode('utf-8')
        self.assertIn('fill-rule="evenodd"', raw_body)
        self.assertNotIn('<metadata', raw_body)
        self.assertGreaterEqual(raw_body.count(' L '), body.count(' L '))

    def test_preview_requires_analysis(self):
        demo = self.client.post('/api/demo', json={'kind': 'isolated'})
        response = self.client.get(
            '/api/samples/' + demo.json['id'] + '/contour-preview?tolerance=1'
        )
        self.assertEqual(response.status_code, 400)


if __name__ == '__main__':
    unittest.main(verbosity=2)
