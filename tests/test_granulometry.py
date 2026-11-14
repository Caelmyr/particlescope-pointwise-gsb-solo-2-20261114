import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from particlescope.imaging import disk
from particlescope.morphology import granulometry


def reference_granulometry(binary, selected, radii):
    """The pre-optimization implementation, kept as the semantic oracle."""
    base = binary & selected
    initial = int(base.sum())
    rows, last = [], initial
    for radius in radii:
        opened = ndi.binary_opening(base, structure=disk(radius)) & selected
        area = int(opened.sum())
        rows.append({
            'radius_px': radius,
            'remaining_pixels': area,
            'removed_from_previous': max(0, last - area),
        })
        last = area
    return initial, rows


def structured_mask(height, width, rng):
    base = rng.random((height, width)) < 0.45
    base = ndi.binary_closing(base, structure=disk(2))
    yy, xx = np.ogrid[:height, :width]
    for _ in range(height * width // 4000 + 2):
        cy, cx = rng.integers(0, height), rng.integers(0, width)
        r = rng.integers(4, 30)
        circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        if rng.random() < 0.5:
            base |= circle
        else:
            base &= ~circle
    base[:, 0] = True
    base[0, :] = True
    return base


class GranulometryEquivalence(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261114)

    def assertMatchesReference(self, base, selected, radii):
        initial, ref_rows = reference_granulometry(base, selected, radii)
        report = granulometry(base, selected, radii)
        self.assertEqual(report['initial_pixels'], initial)
        self.assertEqual(len(report['scales']), len(radii))
        for row, ref, radius in zip(report['scales'], ref_rows, radii):
            self.assertEqual(row['radius_px'], radius)
            self.assertEqual(row['remaining_pixels'], ref['remaining_pixels'])
            self.assertEqual(row['removed_from_previous'], ref['removed_from_previous'])
            if initial:
                self.assertAlmostEqual(
                    row['remaining_percent'],
                    ref['remaining_pixels'] / initial * 100,
                    places=6,
                )
        # Remaining area is monotone non-increasing across increasing radii.
        areas = [row['remaining_pixels'] for row in report['scales']]
        self.assertEqual(areas, sorted(areas, reverse=True))
        # Every area is genuinely computed on the current ROI/foreground.
        self.assertLessEqual(max(areas, default=0), initial)

    def test_random_images_and_rois(self):
        radii = [1, 2, 4, 8, 12, 20, 28, 40]
        for height, width in [(3, 7), (15, 15), (64, 48), (130, 90), (250, 200)]:
            for _ in range(4):
                base = structured_mask(height, width, self.rng)
                selected = np.ones((height, width), dtype=bool)
                selected[:, width // 2:] = False
                selected[0, :] = False
                self.assertMatchesReference(base, selected, radii)

    def test_radius_larger_than_image(self):
        base = np.ones((6, 6), dtype=bool)
        selected = np.ones((6, 6), dtype=bool)
        # A disk of radius 6 cannot fit inside a 6x6 image: every scale empty.
        self.assertMatchesReference(base, selected, [1, 6, 40])

    def test_strict_disk_boundary_survives_correctly(self):
        # An exact disk of radius r must survive opening by r but not r+1.
        size = 81
        yy, xx = np.ogrid[:size, :size]
        center = size // 2
        base = (yy - center) ** 2 + (xx - center) ** 2 <= 12 ** 2
        selected = np.ones((size, size), dtype=bool)
        report = granulometry(base, selected, [12, 13])
        self.assertEqual(report['scales'][0]['remaining_pixels'], int(base.sum()))
        self.assertLess(report['scales'][1]['remaining_pixels'], int(base.sum()))

    def test_result_independent_of_scale_count_and_order(self):
        base = structured_mask(120, 100, self.rng)
        selected = np.ones((120, 100), dtype=bool)
        selected[60:, :] = False
        full = granulometry(base, selected, [1, 4, 12, 40])
        for radii in ([1], [40], [1, 12], [12, 40]):
            sub = granulometry(base, selected, radii)
            wanted = {row['radius_px']: row['remaining_pixels'] for row in full['scales']}
            got = {row['radius_px']: row['remaining_pixels'] for row in sub['scales']}
            self.assertEqual(got, {r: wanted[r] for r in radii})

    def test_empty_foreground_and_empty_radii(self):
        empty = np.zeros((20, 20), dtype=bool)
        selected = np.ones((20, 20), dtype=bool)
        report = granulometry(empty, selected, [1, 2, 4])
        self.assertEqual(report['initial_pixels'], 0)
        self.assertEqual([row['remaining_pixels'] for row in report['scales']], [0, 0, 0])
        self.assertEqual([row['remaining_percent'] for row in report['scales']], [0.0, 0.0, 0.0])
        self.assertEqual(granulometry(empty, selected, [])['scales'], [])

    def test_does_not_mutate_inputs(self):
        base = structured_mask(70, 60, self.rng)
        selected = np.ones((70, 60), dtype=bool)
        selected[:, 40:] = False
        base_copy, selected_copy = base.copy(), selected.copy()
        granulometry(base, selected, [1, 8, 40])
        np.testing.assert_array_equal(base, base_copy)
        np.testing.assert_array_equal(selected, selected_copy)


if __name__ == '__main__':
    unittest.main(verbosity=2)
