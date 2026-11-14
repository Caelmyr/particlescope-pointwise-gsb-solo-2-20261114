"""Topology-preserving vectorization and simplification of label masks.

The raster boundary of every included instance is traced with the same
marching-squares convention as the legacy raw export, but simplification is
planned globally instead of per path:

* every boundary segment is registered once as an undirected pixel-edge key;
* segments shared by two touching instances are detected from the registry
  and simplified only once, so both owners emit the *identical* polyline and
  the common border can neither crack nor overlap;
* the remaining (private) arcs are simplified per particle with
  Ramer-Douglas-Peucker under the user-selected deviation tolerance while the
  shared-arc endpoints stay fixed;
* every simplified polygon is re-validated against the original labeled
  raster: holes may not be filled, originally separated particles may not
  intersect and the per-pixel ownership may not drift beyond the chosen
  tolerance. Particles that cannot satisfy the checks at the requested
  tolerance are repaired at progressively smaller tolerances and, as a last
  resort, keep their unabridged raster contour.

Coordinates stay in image pixels (x right, y down, origin at the upper-left
corner); no scaling or bitmap substitution is performed.
"""

import json
import math
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import find_contours

from .validation import ValidationError


MIN_TOLERANCE = 0.25
MAX_TOLERANCE = 10.0
PREVIEW_NODE_BUDGET = 120000


def validate_tolerance(value):
    try:
        tolerance = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError('轮廓偏差必须为数值') from exc
    if not math.isfinite(tolerance):
        raise ValidationError('轮廓偏差必须为有限数值')
    if tolerance < 0 or tolerance > MAX_TOLERANCE:
        raise ValidationError(f'轮廓偏差需在0到{MAX_TOLERANCE:g}像素之间')
    return tolerance


def _point_key(point):
    """Contour coordinates lie on a 0.5 px grid, so doubled coords are exact."""
    return int(point[0] * 2), int(point[1] * 2)


def _segment_key(a, b):
    ka, kb = _point_key(a), _point_key(b)
    return (ka, kb) if ka <= kb else (kb, ka)


def _loop_segment_keys(points):
    """All undirected edge keys of a closed loop, in traversal order."""
    keys = [None] * len(points)
    previous = _point_key(points[0])
    for index in range(1, len(points)):
        current = _point_key(points[index])
        keys[index - 1] = (previous, current) if previous <= current else (current, previous)
        previous = current
    current = _point_key(points[0])
    keys[-1] = (previous, current) if previous <= current else (current, previous)
    return keys


def trace_instance_loops(labels, object_id, slices):
    """Return the closed contour loops of one instance as (x, y) arrays.

    Cropping to the instance bounding box keeps tracing linear in the total
    foreground instead of repeatedly scanning the whole image. A one-pixel
    pad gives find_contours a background ring identical to the legacy export.
    """
    y0, x0 = slices[0].start, slices[1].start
    crop = labels[slices] == object_id
    padded = np.pad(crop.astype(np.uint8), 1)
    loops = []
    for raw in find_contours(padded, 0.5, fully_connected='high'):
        if len(raw) < 4:
            continue
        # find_contours repeats the first point at the end; drop it.
        if np.allclose(raw[0], raw[-1]):
            raw = raw[:-1]
        if len(raw) < 3:
            continue
        points = np.empty((len(raw), 2), dtype=float)
        points[:, 0] = raw[:, 1] - 1 + x0
        points[:, 1] = raw[:, 0] - 1 + y0
        loops.append(points)
    return loops


def signed_area(points):
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _cross2d(segment, vectors):
    """Scalar z-component of the 2D cross product, array-safe on NumPy 2."""
    return segment[0] * vectors[..., 1] - segment[1] * vectors[..., 0]


def rdp(points, epsilon):
    """Ramer-Douglas-Peucker on an open polyline; endpoints are always kept."""
    if epsilon <= 0 or len(points) < 3:
        return points
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        a, b = points[start], points[end]
        segment = b - a
        norm = float(np.hypot(segment[0], segment[1]))
        middle = points[start + 1:end]
        if norm == 0:
            distances = np.linalg.norm(middle - a, axis=1)
        else:
            distances = np.abs(_cross2d(segment, middle - a)) / norm
        index = int(np.argmax(distances))
        if distances[index] > epsilon:
            pivot = start + 1 + index
            keep[pivot] = True
            stack.append((start, pivot))
            stack.append((pivot, end))
    return points[keep]


def nearest_polyline_deviation(original, simplified, closed=True):
    """Maximum distance from any original vertex to the simplified polyline.

    Closed loops add the wrap-around segment from the last vertex back to the
    first, otherwise vertices near the cyclic seam would be measured against
    the wrong (open) polyline.
    """
    if len(original) == 0 or len(simplified) < 2:
        return 0.0
    segment_count = len(simplified) if closed else len(simplified) - 1
    nearest = np.full(len(original), np.inf)
    for index in range(segment_count):
        a = simplified[index]
        b = simplified[(index + 1) % len(simplified)]
        segment = b - a
        norm = float(np.hypot(segment[0], segment[1]))
        if norm == 0:
            distances = np.linalg.norm(original - a, axis=1)
        else:
            distances = np.abs(_cross2d(segment, original - a)) / norm
        nearest = np.minimum(nearest, distances)
    return float(nearest.max(initial=0.0))


class ContourPlan:
    """Raw loops plus the shared/private arc topology for all instances."""

    def __init__(self, labels, rows):
        self.labels = labels
        self.height, self.width = labels.shape
        self.rows = rows
        self.loops = {}
        self.loop_keys = {}
        self.hole_flags = {}
        self.registry = {}
        self._trace()
        self._classify_loops()
        self.arc_topology = {
            object_id: [
                _split_arcs(points, self.loop_neighbors(object_id, loop_index))
                for loop_index, points in enumerate(loops)
            ]
            for object_id, loops in self.loops.items()
        }

    def _trace(self):
        regions = ndi.find_objects(self.labels)
        for row in self.rows:
            object_id = row['id']
            region = regions[object_id - 1]
            if region is None:
                continue
            loops = trace_instance_loops(self.labels, object_id, region)
            self.loops[object_id] = loops
            keys = [_loop_segment_keys(points) for points in loops]
            self.loop_keys[object_id] = keys
            for loop_index, loop_keys in enumerate(keys):
                for position, key in enumerate(loop_keys):
                    self.registry.setdefault(key, []).append(
                        (object_id, loop_index, position)
                    )

    def _classify_loops(self):
        # Outer loops and hole loops are separated by signed area; the sign
        # convention is fixed per instance by its longest loop.
        for object_id, loops in self.loops.items():
            if not loops:
                self.hole_flags[object_id] = []
                continue
            order = sorted(range(len(loops)), key=lambda i: len(loops[i]), reverse=True)
            dominant_sign = np.sign(signed_area(loops[order[0]])) or 1.0
            self.hole_flags[object_id] = [
                np.sign(signed_area(points)) != dominant_sign
                for points in loops
            ]

    def shared_neighbor(self, key, owner_id):
        for other_id, _, _ in self.registry.get(key, ()):
            if other_id != owner_id:
                return other_id
        return None

    def loop_neighbors(self, object_id, loop_index):
        """Neighbor id (or None) for every segment of one closed loop."""
        result = []
        for key in self.loop_keys[object_id][loop_index]:
            result.append(self.shared_neighbor(key, object_id))
        return result

    def neighbor_ids(self, object_id):
        ids = set()
        for loop_keys in self.loop_keys.get(object_id, ()):
            for key in loop_keys:
                neighbor = self.shared_neighbor(key, object_id)
                if neighbor is not None:
                    ids.add(neighbor)
        return ids

    def original_nodes(self, object_id=None):
        if object_id is not None:
            return sum(len(loop) for loop in self.loops.get(object_id, ()))
        return sum(self.original_nodes(pid) for pid in self.loops)

    def shared_segment_count(self, object_id):
        total = 0
        for loop_keys in self.loop_keys.get(object_id, ()):
            for key in loop_keys:
                if self.shared_neighbor(key, object_id) is not None:
                    total += 1
        return total


def _split_arcs(points, neighbors):
    """Split a loop into maximal shared and private open arcs.

    Each arc is returned as (neighbor_id or None, point sequence). Adjacent
    arcs share their endpoint vertex, which is the fixed anchor that keeps
    touching particles geometrically coupled.
    """
    count = len(points)
    arcs = []
    start = 0
    while start < count:
        neighbor = neighbors[start]
        end = start
        while end + 1 < count and neighbors[end + 1] == neighbor:
            end += 1
        arc_points = np.asarray([
            points[(start + offset) % count]
            for offset in range(end - start + 2)
        ])
        arcs.append((neighbor, arc_points))
        start = end + 1
    # When the cyclic seam splits one class of arc, merge the two ends.
    if len(arcs) > 1 and arcs[0][0] == arcs[-1][0]:
        first_neighbor, first_points = arcs[0]
        _, last_points = arcs[-1]
        merged = np.vstack((last_points[:-1], first_points))
        arcs = arcs[1:-1] + [(first_neighbor, merged)]
    return arcs


def _loop_arcs(plan, object_id, loop_index):
    return plan.arc_topology[object_id][loop_index]


def _shared_arc_id(owner_id, neighbor_id, points):
    # Orientation-independent identity: the undirected set of pixel edges.
    pair = tuple(sorted((owner_id, neighbor_id)))
    keys = tuple(sorted(_segment_key(points[i], points[i + 1])
                        for i in range(len(points) - 1)))
    return pair, keys


def simplify_plan(plan, tolerance, only=None):
    """Build simplified loops for the given instances at one tolerance.

    Arc topology is tolerance-independent, so it is computed once on the
    plan. Shared arcs are cached per (instance pair, edge set); the ``only``
    set restricts work to particles still awaiting repair.
    """
    shared_cache = {}
    simplified = {}
    deviations = {}
    targets = plan.loops if only is None else only
    for object_id in targets:
        loops = plan.loops[object_id]
        new_loops = []
        worst = 0.0
        for loop_index, points in enumerate(loops):
            built = []
            for neighbor, arc_points in plan.arc_topology[object_id][loop_index]:
                if neighbor is None:
                    reduced = rdp(arc_points, tolerance)
                else:
                    arc_id = _shared_arc_id(object_id, neighbor, arc_points)
                    if arc_id not in shared_cache:
                        shared_cache[arc_id] = rdp(arc_points, tolerance)
                    reduced = shared_cache[arc_id]
                    # Reorient the shared polyline to this owner's travel.
                    if not np.allclose(reduced[0], arc_points[0]):
                        reduced = reduced[::-1]
                if built:
                    reduced = reduced[1:]
                built.append(reduced)
            new_loop = np.vstack(built)
            # The closing vertex duplicates the first one; drop it.
            if len(new_loop) > 1 and np.allclose(new_loop[0], new_loop[-1]):
                new_loop = new_loop[:-1]
            worst = max(worst, nearest_polyline_deviation(points, new_loop))
            new_loops.append(new_loop)
        simplified[object_id] = new_loops
        deviations[object_id] = worst
    return simplified, deviations


def _scanline_fill(loops, ys, xs):
    """Even-odd point-in-polygon membership on a pixel grid.

    Marching-squares contours run along the *outer* edge of foreground
    pixels, so membership is tested with a ray offset half a pixel from the
    pixel center toward the upper-left (y-0.5, x-0.5): a boundary pixel then
    falls inside its own contour and outside a hole contour. Loops of every
    orientation interleave via parity, so no winding convention is required.
    """
    ray_y = ys - 0.5
    ray_x = xs - 0.5
    inside = np.zeros((len(ys), len(xs)), dtype=bool)
    for points in loops:
        if len(points) < 3:
            continue
        ax = points[:, 0]
        ay = points[:, 1]
        bx = np.concatenate((ax[1:], ax[:1]))
        by = np.concatenate((ay[1:], ay[:1]))
        minimum_y, maximum_y = float(min(ay.min(), by.min())), float(max(ay.max(), by.max()))
        for row, y in enumerate(ray_y):
            if y < minimum_y or y >= maximum_y:
                continue
            upward = (ay <= y) & (by > y)
            downward = (by <= y) & (ay > y)
            active = upward | downward
            if not active.any():
                continue
            crossings = ax[active] + (y - ay[active]) * (bx[active] - ax[active]) / (by[active] - ay[active])
            crossings.sort()
            for index in range(0, len(crossings) - 1, 2):
                inside[row] ^= (ray_x >= crossings[index]) & (ray_x < crossings[index + 1])
    return inside


def _rasterize_loops(loops, slices):
    # Parity fill handles outer and hole loops directly, so loop orientation
    # flags are not needed here.
    ys = np.arange(slices[0].start, slices[0].stop, dtype=float) + 0.5
    xs = np.arange(slices[1].start, slices[1].stop, dtype=float) + 0.5
    return _scanline_fill(loops, ys, xs)


def validate_particle(plan, object_id, loops, neighbor_ids, hole_pixels,
                      binary, slices, shared_length, tolerance):
    """Check simplified loops against the original labeled raster.

    Hard invariants: no original hole pixel may be filled and no pixel of an
    originally non-neighboring particle may be covered. Soft invariants
    (coverage, IoU, neighbor-border band) trigger automatic repair.
    """
    raster = _rasterize_loops(loops, slices)
    holes_filled = int((raster & hole_pixels).sum())
    labels_crop = plan.labels[slices]
    foreign = (labels_crop != 0) & (labels_crop != object_id)
    if neighbor_ids:
        foreign = foreign & ~np.isin(labels_crop, list(neighbor_ids))
    foreign_overlap = int((raster & foreign).sum())
    if holes_filled or foreign_overlap:
        return False, {'holes_filled': holes_filled, 'foreign_overlap': foreign_overlap}
    covered = int((raster & binary).sum())
    coverage = covered / max(int(binary.sum()), 1)
    symmetric = int((raster ^ binary).sum())
    iou = 1 - symmetric / max(int((raster | binary).sum()), 1)
    neighbor_overlap = int(
        (raster & (labels_crop != object_id) & (labels_crop != 0)).sum()
    )
    # A bulge across a shared border may be at most one tolerance band deep.
    border_budget = max(shared_length * max(1.0, math.ceil(tolerance + 1.0)), 1)
    # Area is allowed to drift in proportion to the chosen deviation. For a
    # convex particle the symmetric-difference rate scales with
    # perimeter * tolerance / area, so small raster objects naturally show a
    # larger relative change at the same pixel tolerance. The hard topology
    # checks above (no holes, no foreign pixels) remain independent of these
    # soft geometric bands and always apply.
    min_coverage = max(0.90, 1.0 - 0.04 * tolerance)
    min_iou = max(0.85, 1.0 - 0.08 * tolerance)
    if coverage < min_coverage or iou < min_iou or neighbor_overlap > border_budget:
        return False, {
            'coverage': round(coverage, 4), 'iou': round(iou, 4),
            'min_coverage': round(min_coverage, 4), 'min_iou': round(min_iou, 4),
            'neighbor_overlap': neighbor_overlap, 'border_budget': int(border_budget),
        }
    return True, {
        'coverage': round(coverage, 4), 'iou': round(iou, 4),
        'neighbor_overlap': neighbor_overlap,
    }


def build_contours(labels, rows, tolerance):
    """Full simplification pipeline with validation and automatic repair.

    Each particle is validated independently at the requested tolerance.
    Particles that fail are retried at halved tolerances; particles that
    already passed keep their result at the larger tolerance. As a last
    resort an unrepairable particle keeps its unabridged raster contour.
    """
    tolerance = validate_tolerance(tolerance)
    plan = ContourPlan(labels, rows)
    if not plan.loops:
        raise ValidationError('当前分析没有可导出的颗粒轮廓')
    regions = ndi.find_objects(plan.labels)
    context = {}
    for pid in plan.loops:
        slices = regions[pid - 1]
        binary = plan.labels[slices] == pid
        hole_pixels = ndi.binary_fill_holes(binary) & ~binary
        context[pid] = {
            'slices': slices,
            'binary': binary,
            'hole_pixels': hole_pixels,
            'neighbors': plan.neighbor_ids(pid),
            'shared_length': plan.shared_segment_count(pid),
        }
    # Halving ladder from the requested tolerance down to the floor. Levels
    # are evaluated lazily: the common case where every particle passes at the
    # requested tolerance costs a single simplification pass.
    requested = max(tolerance, MIN_TOLERANCE)
    levels = []
    level = requested
    while level > MIN_TOLERANCE:
        levels.append(round(level, 6))
        level = max(level / 2, MIN_TOLERANCE)
    levels.append(MIN_TOLERANCE)
    chosen = {}
    applied = {}
    repair_log = []
    pending = set(plan.loops)
    for level in levels:
        if not pending:
            break
        candidate, _ = simplify_plan(plan, level, only=pending)
        still_failing = set()
        for pid in pending:
            loops = candidate[pid]
            info = context[pid]
            ok, detail = validate_particle(
                plan, pid, loops, info['neighbors'], info['hole_pixels'],
                info['binary'], info['slices'], info['shared_length'], level,
            )
            if ok:
                chosen[pid] = loops
                applied[pid] = level
            else:
                still_failing.add(pid)
                repair_log.append({'id': pid, 'tolerance': level, 'reason': detail})
        # A shared border is one geometric polyline: if a particle must drop
        # to a finer level, every neighbor sharing that border drops too, so
        # both owners always emit the identical border (no cracks/overlaps).
        coupled = set()
        for pid in still_failing:
            coupled.update(context[pid]['neighbors'])
        for neighbor in coupled:
            chosen.pop(neighbor, None)
            applied.pop(neighbor, None)
            still_failing.add(neighbor)
        pending = still_failing
    for pid in pending:
        # Exact raster boundary; parity fill reconstructs the original mask.
        chosen[pid] = [loop.copy() for loop in plan.loops[pid]]
        applied[pid] = 0.0
        info = context[pid]
        ok, detail = validate_particle(
            plan, pid, chosen[pid], info['neighbors'], info['hole_pixels'],
            info['binary'], info['slices'], info['shared_length'], 0.0,
        )
        if not ok:
            raise ValidationError(
                f'颗粒{pid}轮廓无法满足拓扑约束，请减小偏差后重试'
            )
    deviations = {}
    for pid, loops in chosen.items():
        # Recompute against the actually chosen loops; a particle repaired at
        # a smaller level must not report the deviation of a rejected level.
        worst = 0.0
        for original, reduced in zip(plan.loops[pid], loops):
            worst = max(worst, nearest_polyline_deviation(original, reduced))
        deviations[pid] = worst
    repaired = set()
    if tolerance > 0:
        for pid, loops in chosen.items():
            originals = plan.loops[pid]
            if len(loops) != len(originals):
                continue
            if all(new_loop.shape == original.shape and np.allclose(new_loop, original)
                   for new_loop, original in zip(loops, originals)):
                repaired.add(pid)
    stats = []
    for row in rows:
        pid = row['id']
        if pid not in chosen:
            continue
        before = plan.original_nodes(pid)
        after = int(sum(len(loop) for loop in chosen[pid]))
        stats.append({
            'id': pid,
            'loops': len(plan.loops[pid]),
            'holes': int(sum(plan.hole_flags[pid])),
            'shared_segments': context[pid]['shared_length'],
            'tolerance_applied_px': applied[pid],
            'nodes_original': before,
            'nodes_simplified': after,
            'max_deviation_px': round(deviations[pid], 3),
            'repaired': pid in repaired,
        })
    return {
        'plan': plan,
        'loops': chosen,
        'hole_flags': plan.hole_flags,
        'stats': stats,
        'tolerance_used': max(applied.values()),
        'tolerance_requested': tolerance,
        'repair_log': repair_log,
        'repaired_ids': sorted(repaired),
        'nodes_original': plan.original_nodes(),
        'nodes_simplified': int(sum(item['nodes_simplified'] for item in stats)),
    }


def format_path_data(loops):
    parts = []
    for points in loops:
        if len(points) < 3:
            continue
        coords = ' L '.join(f'{x:.3f},{y:.3f}' for x, y in points)
        parts.append(f'M {coords} Z')
    return ' '.join(parts)


def particle_color(object_id):
    return '#%06x' % ((object_id * 2654435761) & 0xffffff)


def _metadata_attributes(metadata):
    return ' '.join(
        f'{key}={json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace(chr(34), "&quot;")}'
        for key, value in metadata.items()
    )


def contour_svg_string(labels, rows, tolerance=0.0, labels_on=True):
    """Render the exported SVG document in image pixel coordinates.

    tolerance=0 emits the unabridged raster contour (legacy behavior with
    even-odd hole fills). tolerance>0 emits the topology-validated simplified
    contours plus visible particle numbers when labels_on is set.
    """
    height, width = labels.shape
    if tolerance and tolerance > 0:
        built = build_contours(labels, rows, tolerance)
        loops_by_id = built['loops']
        hole_flags = built['hole_flags']
        metadata = {
            'simplified': True,
            'tolerance_px': built['tolerance_requested'],
            'tolerance_applied_px': built['tolerance_used'],
            'nodes_original': built['nodes_original'],
            'nodes_simplified': built['nodes_simplified'],
            'repaired_particle_ids': built['repaired_ids'],
        }
    else:
        plan = ContourPlan(labels, rows)
        loops_by_id = {pid: [loop.copy() for loop in loops]
                       for pid, loops in plan.loops.items()}
        hole_flags = plan.hole_flags
        metadata = {
            'simplified': False,
            'tolerance_px': 0,
            'nodes_original': plan.original_nodes(),
            'nodes_simplified': plan.original_nodes(),
        }
    body = []
    for row in rows:
        pid = row['id']
        loops = loops_by_id.get(pid)
        if not loops:
            continue
        data = format_path_data(loops)
        color = particle_color(pid)
        body.append(
            f'<path data-particle-id="{pid}" fill="{color}" fill-rule="evenodd" '
            f'data-hole-loops="{int(sum(hole_flags[pid]))}" d="{data}"/>'
        )
    if labels_on and tolerance and tolerance > 0:
        body.append(
            '<g class="particle-labels" fill="#0d1f18" font-family="sans-serif" '
            f'font-size="{max(8, min(width, height) // 90)}" text-anchor="middle">'
        )
        for row in rows:
            pid = row['id']
            if pid not in loops_by_id:
                continue
            body.append(
                f'<text x="{float(row["centroid_x"]):.3f}" '
                f'y="{float(row["centroid_y"]):.3f}" data-particle-id="{pid}">{pid}</text>'
            )
        body.append('</g>')
    metadata.update({
        'coordinate_system': 'image pixels; x right, y down, origin upper-left',
        'width_px': width,
        'height_px': height,
        'particle_count': len(rows),
        'fill_rule': 'evenodd',
    })
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}">'
        f'<metadata>{_metadata_attributes(metadata)}</metadata>'
        f'{"".join(body)}</svg>'
    )


def preview_report(labels, rows, tolerance):
    """Statistics plus bounded two-layer geometry for the page preview."""
    built = build_contours(labels, rows, tolerance)
    plan = built['plan']
    particle_stats = built['stats']
    raw_layers, simple_layers = [], []
    spent = 0
    truncated = False
    for row in rows:
        pid = row['id']
        if pid not in built['loops']:
            continue
        original_nodes = plan.original_nodes(pid)
        if spent + original_nodes > PREVIEW_NODE_BUDGET:
            truncated = True
            break
        spent += original_nodes
        color = particle_color(pid)
        raw_layers.append({'id': pid, 'color': color, 'd': format_path_data(plan.loops[pid])})
        simple_layers.append({'id': pid, 'color': color, 'd': format_path_data(built['loops'][pid])})
    reduction = 0.0
    if built['nodes_original']:
        reduction = round(
            (1 - built['nodes_simplified'] / built['nodes_original']) * 100, 2
        )
    return {
        'tolerance': built['tolerance_requested'],
        'tolerance_applied': built['tolerance_used'],
        'width': plan.width,
        'height': plan.height,
        'nodes_original': built['nodes_original'],
        'nodes_simplified': built['nodes_simplified'],
        'reduction_percent': reduction,
        'shared_segments': sum(item['shared_segments'] for item in particle_stats),
        'particles_with_holes': sum(1 for item in particle_stats if item['holes']),
        'repaired_ids': built['repaired_ids'],
        'truncated_geometry': truncated,
        'preview_particles': len(simple_layers),
        'particles': particle_stats,
        'original_paths': raw_layers,
        'simplified_paths': simple_layers,
    }
