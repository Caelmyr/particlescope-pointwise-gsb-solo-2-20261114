"""Skeleton network analysis.

The thinned foreground is interpreted as a graph of skeleton pixels rather
than a bag of pixels:

* Adjacency is 8-connected, but a diagonal step is ignored when an
  orthogonal two-step bridge already connects the same pixels.  Without that
  rule every diagonal staircase pixel looks like a branch point; with it, a
  25-degree fibre stays a simple path while two lines that merely touch are
  still recognised as a junction.
* Pixels whose graph degree is not two are node pixels.  Adjacent node
  pixels are one node cluster, so a junction blob is a single node with one
  position, and a branch that ends in it is attached exactly once.
* Branch length is measured along the traced spatial path (a one-pixel
  Douglas-Peucker polyline removes raster staircase inflation), so a
  horizontal and an oblique fibre of equal extent report equal lengths.
  A closed loop is one branch measured once around.
* Terminal branches shorter than a chosen spur threshold can be pruned.  The
  raw and pruned networks and their difference are reported together, so the
  effect of the threshold is visible instead of silently changing numbers.

Node pixels and branch pixels are disjoint and together cover the skeleton;
the ``accounting`` block checks that reconciliation.
"""

import math

import numpy as np
from scipy import ndimage as ndi

from .morphology import skeletonize
from .validation import ValidationError

OFFSETS = (
    (-1, -1), (-1, 0), (-1, 1), (0, 1),
    (1, 1), (1, 0), (1, -1), (0, -1),
)
EIGHT = np.ones((3, 3), dtype=int)
MAX_SKELETON_PIXELS = 250000
MAX_BRANCHES = 5000
MAX_PRUNING_ROUNDS = 24
JUNCTION_MERGE_PX = 3.0
SIMPLIFY_TOLERANCE = 1.0
SIMPLIFY_LIMIT = 260


def crossing_numbers(skeleton):
    """Background-to-foreground transitions around each pixel.

    Node detection uses the deduplicated adjacency graph, which is stable on
    staircases.  The crossing number is still reported as a diagnostic: a
    path pixel has two arms, an endpoint one and a branch point three or
    more, so the profile summarises how much of the skeleton is junction
    material.
    """
    height, width = skeleton.shape
    padded = np.pad(skeleton.astype(np.uint8), 1)
    ring = [
        padded[1 + dy:1 + dy + height, 1 + dx:1 + dx + width]
        for dy, dx in OFFSETS
    ]
    counts = np.zeros((height, width), dtype=np.int16)
    for index in range(8):
        counts += (ring[index] == 0) & (ring[(index + 1) % 8] == 1)
    counts[~skeleton] = 0
    return counts


def crossing_profile(skeleton):
    counts = crossing_numbers(skeleton)
    values = counts[skeleton]
    profile = {
        'one_arm_pixels': int(np.count_nonzero(values == 1)),
        'two_arm_pixels': int(np.count_nonzero(values == 2)),
        'three_arm_pixels': int(np.count_nonzero(values == 3)),
        'four_plus_arm_pixels': int(np.count_nonzero(values >= 4)),
    }
    profile['skeleton_pixels'] = int(values.size)
    return profile


def _step(first, second):
    return math.hypot(second[0] - first[0], second[1] - first[1])


def _path_length(path):
    return sum(_step(a, b) for a, b in zip(path, path[1:]))


def _pixel_set(skeleton):
    ys, xs = np.nonzero(skeleton)
    return {(int(y), int(x)) for y, x in zip(ys, xs)}


def _adjacency(pixels):
    """Undirected skeleton graph with redundant diagonal steps removed.

    A diagonal neighbour ``(y+dy, x+dx)`` is redundant when either of the two
    orthogonal pixels between the pair exists: the step then adds no
    connectivity and would otherwise turn every staircase corner into a
    junction.  Genuine crossings still keep their diagonal links because no
    orthogonal bridge exists there.
    """
    graph = {pixel: set() for pixel in pixels}
    for pixel in pixels:
        y, x = pixel
        for dy, dx in OFFSETS:
            neighbour = (y + dy, x + dx)
            if neighbour not in pixels:
                continue
            if dy != 0 and dx != 0:
                if (y + dy, x) in pixels or (y, x + dx) in pixels:
                    continue
            graph[pixel].add(neighbour)
    return {pixel: tuple(sorted(neighbours)) for pixel, neighbours in graph.items()}


def _node_clusters(node_pixels, shape):
    mask = np.zeros(shape, dtype=bool)
    for y, x in node_pixels:
        mask[y, x] = True
    labels, count = ndi.label(mask, EIGHT)
    nodes = []
    node_of = {}
    if not count:
        return nodes, node_of, labels
    flat = labels.ravel()
    order = np.argsort(flat, kind='stable')
    values = flat[order]
    boundaries = np.searchsorted(values, np.arange(1, count + 1), side='left')
    ends = np.searchsorted(values, np.arange(1, count + 1), side='right')
    width = shape[1]
    for index, (start, stop) in enumerate(zip(boundaries.tolist(), ends.tolist()), 1):
        flat_positions = order[start:stop]
        ys = flat_positions // width
        xs = flat_positions % width
        pixels = [(int(y), int(x)) for y, x in zip(ys.tolist(), xs.tolist())]
        nodes.append({
            'pixels': pixels,
            'x': float(xs.mean()),
            'y': float(ys.mean()),
        })
        for pixel in pixels:
            node_of[pixel] = len(nodes) - 1
    return nodes, node_of, labels


def _trace(graph, node_of, order):
    """Trace one branch from a node pixel; returns (path, far_node_index)."""
    visited = set()
    branches = []
    for pixel in order:
        if pixel not in node_of:
            continue
        for neighbour in graph[pixel]:
            if neighbour in node_of or neighbour in visited:
                continue
            path = [pixel, neighbour]
            visited.add(neighbour)
            previous, current = pixel, neighbour
            owner = None
            while True:
                if current in node_of:
                    owner = node_of[current]
                    break
                following = [item for item in graph[current] if item != previous]
                if not following:
                    break
                nxt = following[0]
                path.append(nxt)
                if nxt in node_of:
                    owner = node_of[nxt]
                    break
                visited.add(nxt)
                previous, current = current, nxt
            branches.append({
                'a': node_of[pixel],
                'b': owner,
                'closed': owner is not None and owner == node_of[pixel] and len(path) > 3,
                'path': path,
            })
    return branches, visited


def _loop_branches(graph, visited, node_of, pixels):
    remaining = set(pixels) - visited - set(node_of)
    branches = []
    while remaining:
        seed = min(remaining)
        component = _component(graph, seed, remaining)
        remaining -= component
        if len(component) == 1:
            branches.append({'a': None, 'b': None, 'closed': False, 'path': [seed]})
            continue
        if len(component) < 3:
            path = _walk_path(graph, component)
            branches.append({'a': None, 'b': None, 'closed': False, 'path': path})
            continue
        if all(len([n for n in graph[pixel] if n in component]) == 2 for pixel in component):
            path, closed = _walk_cycle(graph, component)
            if closed:
                branches.append({'a': None, 'b': None, 'closed': True, 'path': path})
                continue
        path = _walk_path(graph, component)
        start = _adjacent_node(graph, path[0], node_of)
        end = _adjacent_node(graph, path[-1], node_of)
        branches.append({
            'a': start,
            'b': end,
            'closed': start is not None and start == end and len(path) > 3,
            'path': path,
        })
    return branches


def _component(graph, seed, pool):
    group = {seed}
    queue = [seed]
    while queue:
        pixel = queue.pop()
        for neighbour in graph[pixel]:
            if neighbour in pool and neighbour not in group:
                group.add(neighbour)
                queue.append(neighbour)
    return group


def _adjacent_node(graph, pixel, node_of):
    for neighbour in graph[pixel]:
        owner = node_of.get(neighbour)
        if owner is not None:
            return owner
    return None


def _walk_path(graph, pixels):
    degree = {pixel: sum(1 for other in graph[pixel] if other in pixels) for pixel in pixels}
    ends = sorted(pixel for pixel in pixels if degree[pixel] <= 1)
    start = ends[0] if ends else min(pixels)
    path = [start]
    used = {start}
    previous, current = None, start
    while True:
        candidates = [n for n in graph[current] if n in pixels and n != previous and n not in used]
        if not candidates:
            break
        candidates.sort()
        following = candidates[0]
        path.append(following)
        used.add(following)
        previous, current = current, following
    return path


def _walk_cycle(graph, pixels):
    start = min(pixels)
    path = [start]
    used = {start}
    previous, current = None, start
    for _ in range(len(pixels) + 2):
        candidates = [
            neighbour for neighbour in graph[current]
            if neighbour in pixels and neighbour != previous
            and (neighbour not in used or neighbour == start)
        ]
        if not candidates:
            return path, False
        if start in candidates:
            return path, True
        following = sorted(candidates)[0]
        path.append(following)
        used.add(following)
        previous, current = current, following
    return path, False


def _degrees(nodes, branches):
    degree = [0] * len(nodes)
    for branch in branches:
        if branch['a'] is not None:
            degree[branch['a']] += 1
        if branch['b'] is not None:
            degree[branch['b']] += 1
    return degree


def _douglas_peucker(points, epsilon):
    """Iterative Douglas-Peucker simplification (avoids recursion limits)."""
    if len(points) <= 2:
        return list(points)
    data = np.asarray(points, dtype=float)
    keep = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    while stack:
        first_index, last_index = stack.pop()
        if last_index <= first_index + 1:
            continue
        first, last = data[first_index], data[last_index]
        segment = last - first
        span = float(np.hypot(*segment))
        window = data[first_index + 1:last_index]
        if span == 0:
            distances = np.hypot(*(window - first).T)
        else:
            distances = np.abs(
                segment[0] * (window[:, 1] - first[1]) - segment[1] * (window[:, 0] - first[0])
            ) / span
        position = int(np.argmax(distances))
        if distances[position] > epsilon:
            split = first_index + 1 + position
            keep.add(split)
            stack.append((first_index, split))
            stack.append((split, last_index))
    return [points[index] for index in sorted(keep)]


def _spatial_length(path):
    """Length along the path with raster staircase corners smoothed out.

    Summing raw pixel steps overstates a 25-degree line by about 24% because
    every staircase corner counts as a diagonal step.  A one-pixel polyline
    tolerance removes that artefact without flattening genuine curvature.
    """
    if len(path) <= 2:
        return _path_length(path)
    return _path_length(_douglas_peucker(path, SIMPLIFY_TOLERANCE))


def _loop_length(path):
    points = list(path)
    if len(points) > 3 and points[0] == points[-1]:
        points = points[:-1]
    if len(points) <= 3:
        return _path_length(list(path))
    start = points[0]
    far = max(
        range(len(points)),
        key=lambda index: (points[index][0] - start[0]) ** 2 + (points[index][1] - start[1]) ** 2,
    )
    first = _douglas_peucker(points[:far + 1], SIMPLIFY_TOLERANCE)
    second = _douglas_peucker(points[far:] + [start], SIMPLIFY_TOLERANCE)
    return _path_length(first[:-1] + second)


def _graph(skeleton):
    pixels = _pixel_set(skeleton)
    graph = _adjacency(pixels)
    node_pixels = {pixel for pixel in pixels if len(graph[pixel]) != 2}
    nodes, node_of, _ = _node_clusters(node_pixels, skeleton.shape)
    order = sorted(node_of)
    branches, visited = _trace(graph, node_of, order)
    for component in _loop_branches(graph, visited, node_of, pixels):
        component['length'] = _loop_length(component['path']) if component['closed'] else _spatial_length(component['path'])
        component['direct_length'] = _path_length(component['path'])
        branches.append(component)
    for branch in branches:
        branch['length'] = _loop_length(branch['path']) if branch['closed'] else _spatial_length(branch['path'])
        branch['direct_length'] = _path_length(branch['path'])
    merged = _merge_close_junctions(nodes, branches)
    for branch in branches:
        branch['length'] = _loop_length(branch['path']) if branch['closed'] else _spatial_length(branch['path'])
        branch['direct_length'] = _path_length(branch['path'])
    return {
        'skeleton': skeleton,
        'pixels': pixels,
        'graph': graph,
        'nodes': nodes,
        'node_of': node_of,
        'branches': branches,
        'merged_junctions': merged,
    }


def _merge_close_junctions(nodes, branches):
    """Fuse junction clusters joined by a very short segment.

    In a thick or shallow crossing the thinning can leave two or three
    junction pixels a few steps apart.  They describe one physical crossing,
    so they are merged into a single positioned node and the tiny connector
    disappears instead of being reported as a branch.
    """
    merged = 0
    changed = True
    while changed:
        changed = False
        degree = _degrees(nodes, branches)
        for index, branch in enumerate(branches):
            a, b = branch['a'], branch['b']
            if a is None or b is None or a == b or branch['closed']:
                continue
            if degree[a] < 3 or degree[b] < 3:
                continue
            if branch['length'] > JUNCTION_MERGE_PX:
                continue
            if len(branch['path']) > 5:
                continue
            keep, drop = min(a, b), max(a, b)
            nodes[keep]['pixels'] = (
                nodes[keep]['pixels'] + branch['path'][1:-1] + nodes[drop]['pixels']
            )
            nodes[keep]['x'] = float(np.mean([p[1] for p in nodes[keep]['pixels']]))
            nodes[keep]['y'] = float(np.mean([p[0] for p in nodes[keep]['pixels']]))
            nodes[drop]['pixels'] = []
            for other in branches:
                if other is branch:
                    continue
                if other['a'] == drop:
                    other['a'] = keep
                if other['b'] == drop:
                    other['b'] = keep
                if other['a'] == keep and other['b'] == keep:
                    other['closed'] = True
                    if other['path'][0] != other['path'][-1]:
                        other['path'] = list(other['path']) + [other['path'][0]]
            branches.pop(index)
            merged += 1
            changed = True
            break
    return merged


def _branch_pixels(branch, nodes):
    owned = set()
    for index in (branch['a'], branch['b']):
        if index is not None:
            owned.update(nodes[index]['pixels'])
    return list(dict.fromkeys(pixel for pixel in branch['path'] if pixel not in owned))


def _kept(nodes):
    return [index for index, node in enumerate(nodes) if node['pixels']]


def _summary(graph_data, scale, unit):
    nodes = graph_data['nodes']
    branches = graph_data['branches']
    degree = _degrees(nodes, branches)
    kept = _kept(nodes)
    lengths = [branch['length'] for branch in branches]
    endpoints = [index for index in kept if degree[index] == 1]
    junctions = [index for index in kept if degree[index] >= 3]
    isolated = [index for index in kept if degree[index] == 0]
    loops = [index for index, branch in enumerate(branches) if branch['closed']]
    return {
        'node_count': len(kept),
        'endpoint_count': len(endpoints),
        'junction_count': len(junctions),
        'isolated_node_count': len(isolated),
        'branch_count': len(branches),
        'loop_count': len(loops),
        'total_length_px': round(sum(lengths), 6),
        'total_length': round(sum(lengths) * scale, 6),
        'mean_branch_length_px': round(sum(lengths) / max(len(lengths), 1), 6),
        'mean_branch_length': round(sum(lengths) / max(len(lengths), 1) * scale, 6),
        'median_branch_length_px': round(float(np.median(lengths)) if lengths else 0.0, 6),
        'longest_branch_px': round(max(lengths) if lengths else 0.0, 6),
        'shortest_branch_px': round(min(lengths) if lengths else 0.0, 6),
        'skeleton_pixels': len(graph_data['pixels']),
        'unit': unit,
        'degree': degree,
        'kept': kept,
    }


def _branch_type(branch, degree, numbering):
    a = numbering.get(branch['a']) if branch['a'] is not None else None
    b = numbering.get(branch['b']) if branch['b'] is not None else None
    if branch['closed']:
        return 'loop'
    if a is None or b is None:
        return 'open'
    if degree[branch['a']] == 1 and degree[branch['b']] == 1:
        return 'isolated'
    if degree[branch['a']] == 1 or degree[branch['b']] == 1:
        return 'terminal'
    return 'connecting'


def _point(pixel):
    return [int(pixel[1]), int(pixel[0])]


def _simplify(points, limit=SIMPLIFY_LIMIT):
    if len(points) <= 2:
        return points
    simplified = _douglas_peucker(points, SIMPLIFY_TOLERANCE)
    epsilon = SIMPLIFY_TOLERANCE
    while len(simplified) > limit and epsilon < 64:
        epsilon *= 2
        simplified = _douglas_peucker(points, epsilon)
    if len(simplified) > limit:
        stride = int(math.ceil(len(simplified) / limit))
        simplified = simplified[::stride] + [simplified[-1]]
    return simplified


def _describe(graph_data, scale, unit):
    nodes = graph_data['nodes']
    branches = graph_data['branches']
    degree = _degrees(nodes, branches)
    summary = _summary(graph_data, scale, unit)
    kept = sorted(
        summary['kept'],
        key=lambda index: (round(nodes[index]['y'], 3), round(nodes[index]['x'], 3), index),
    )
    numbering = {old: number for number, old in enumerate(kept, 1)}
    branch_ids = {index: [] for index in kept}
    ordered = []
    for index, branch in enumerate(branches):
        if branch['a'] is not None and branch['a'] in numbering:
            branch_ids[branch['a']].append(index)
        if branch['b'] is not None and branch['b'] in numbering and branch['b'] != branch['a']:
            branch_ids[branch['b']].append(index)
        anchor = branch['path'][0] if branch['path'] else (0, 0)
        ordered.append(((anchor[0], anchor[1], index), index, branch))
    ordered.sort(key=lambda item: item[0])
    records = []
    for number, (_, index, branch) in enumerate(ordered, 1):
        first = numbering.get(branch['a']) if branch['a'] is not None else None
        second = numbering.get(branch['b']) if branch['b'] is not None else None
        owned = _branch_pixels(branch, nodes)
        points = [_point(pixel) for pixel in branch['path']]
        records.append({
            'id': number,
            'type': _branch_type(branch, degree, numbering),
            'node_a': first,
            'node_b': second,
            'start': points[0],
            'end': points[-1],
            'pixel_count': len(owned),
            'step_count': max(0, len(branch['path']) - 1),
            'direct_length_px': round(branch['direct_length'], 6),
            'length_px': round(branch['length'], 6),
            'length': round(branch['length'] * scale, 6),
            'closed': bool(branch['closed']),
            'points': _simplify(points),
        })
    node_records = []
    endpoints = []
    junctions = []
    for old in kept:
        node = nodes[old]
        value = degree[old]
        kind = 'junction' if value >= 3 else 'endpoint' if value == 1 else 'isolated'
        record = {
            'id': numbering[old],
            'type': kind,
            'x': round(node['x'], 3),
            'y': round(node['y'], 3),
            'pixel_count': len(node['pixels']),
            'degree': value,
            'branch_ids': sorted(branch_ids[old]),
        }
        node_records.append(record)
        if kind == 'junction':
            junctions.append(record)
        elif kind == 'endpoint':
            row = dict(record)
            row['branch_id'] = record['branch_ids'][0] if record['branch_ids'] else None
            endpoints.append(row)
    node_pixels = sum(len(nodes[index]['pixels']) for index in kept)
    branch_pixels = sum(len(_branch_pixels(branch, nodes)) for branch in branches)
    summary.update({
        'nodes': node_records,
        'endpoints': endpoints,
        'junctions': junctions,
        'branches': records,
        'accounting': {
            'node_pixels': node_pixels,
            'branch_pixels': branch_pixels,
            'assigned_pixels': node_pixels + branch_pixels,
            'skeleton_pixels': len(graph_data['pixels']),
            'unassigned_pixels': len(graph_data['pixels']) - node_pixels - branch_pixels,
        },
    })
    return summary


def _prune(skeleton, threshold):
    current = np.asarray(skeleton, dtype=bool).copy()
    removed = []
    if threshold <= 0:
        return current, removed
    for round_number in range(1, MAX_PRUNING_ROUNDS + 1):
        graph_data = _graph(current)
        degree = _degrees(graph_data['nodes'], graph_data['branches'])
        targets = []
        for index, branch in enumerate(graph_data['branches']):
            a, b = branch['a'], branch['b']
            if a is None or b is None or a == b or branch['closed']:
                continue
            if branch['length'] >= threshold:
                continue
            if degree[a] == 1 and degree[b] >= 3:
                targets.append((index, a, b))
            elif degree[b] == 1 and degree[a] >= 3:
                targets.append((index, b, a))
        if not targets:
            break
        changed = 0
        for index, tip, junction in targets:
            branch = graph_data['branches'][index]
            tip_pixels = set(graph_data['nodes'][tip]['pixels'])
            junction_pixels = set(graph_data['nodes'][junction]['pixels'])
            removable = (set(branch['path']) | tip_pixels) - junction_pixels
            tip_pixel = branch['path'][-1] if graph_data['node_of'].get(branch['path'][-1]) == tip else branch['path'][0]
            removed.append({
                'round': round_number,
                'endpoint': _point(tip_pixel),
                'junction': [round(graph_data['nodes'][junction]['x'], 3), round(graph_data['nodes'][junction]['y'], 3)],
                'length_px': round(branch['length'], 6),
                'pixel_count': len(_branch_pixels(branch, graph_data['nodes'])),
                'points': _simplify([_point(pixel) for pixel in branch['path']]),
            })
            for pixel in removable:
                if current[pixel]:
                    current[pixel] = False
                    changed += 1
        if not changed or not np.any(current):
            break
    return current, removed


def skeleton_analysis(binary, selected, scale=1, unit='px', spur_px=0, skeleton=None):
    """Analyse ``binary & selected`` and return positioned nodes and branches."""
    if binary.size > 4000000:
        raise ValidationError('骨架分析限400万像素')
    mask = binary & selected
    if skeleton is None:
        skeleton, iterations, converged = skeletonize(mask)
    else:
        skeleton, iterations, converged = np.asarray(skeleton, dtype=bool) & mask, 0, True
    if int(skeleton.sum()) > MAX_SKELETON_PIXELS:
        raise ValidationError('骨架像素超过250000，请缩小分析区域或减少细碎前景')
    raw_graph = _graph(skeleton.copy())
    raw = _describe(raw_graph, scale, unit)
    pruned, removed = _prune(skeleton, float(spur_px))
    final_graph = _graph(pruned)
    described = _describe(final_graph, scale, unit)
    if described['branch_count'] > MAX_BRANCHES:
        raise ValidationError('骨架支路超过5000条，请提高毛刺阈值或缩小分析区域')
    described.update({
        'iterations': iterations,
        'converged': converged,
        'spur_threshold_px': round(float(spur_px), 6),
        'spur_threshold': round(float(spur_px) * scale, 6),
        'unit': unit,
        'scale': scale,
        'image_width': int(binary.shape[1]),
        'image_height': int(binary.shape[0]),
        'raw_network': {key: value for key, value in raw.items() if key not in (
            'nodes', 'endpoints', 'junctions', 'branches', 'accounting', 'degree', 'kept',
        )},
        'network': {key: described[key] for key in (
            'node_count', 'endpoint_count', 'junction_count', 'isolated_node_count',
            'branch_count', 'loop_count', 'total_length_px', 'total_length',
            'mean_branch_length_px', 'mean_branch_length', 'median_branch_length_px',
            'longest_branch_px', 'shortest_branch_px', 'skeleton_pixels',
        )},
        'removed': {
            'count': len(removed),
            'items': removed,
            'total_length_px': round(sum(item['length_px'] for item in removed), 6),
            'total_length': round(sum(item['length_px'] for item in removed) * scale, 6),
            'pixel_count': int(skeleton.sum()) - int(pruned.sum()),
        },
        'warning': (
            '长度沿骨架空间路径计算，并平滑栅格台阶，水平与斜向纤维可直接比较；'
            '相邻结点像素归为同一结点团，其像素不计入支路长度；闭环只计一次。'
        ),
    })
    described['crossing_profile'] = crossing_profile(pruned)
    described['raw_crossing_profile'] = crossing_profile(skeleton)
    described['impact'] = _impact(raw, described, removed, skeleton, pruned, scale)
    described['raw_geometry'] = skeleton
    described['final_geometry'] = pruned
    described['raw_report'] = raw
    return described


def _impact(raw, described, removed, raw_skeleton, pruned, scale):
    removed_length = sum(item['length_px'] for item in removed)
    return {
        'spur_threshold_px': described['spur_threshold_px'],
        'spur_threshold': described['spur_threshold'],
        'spur_threshold_unit': described['unit'],
        'removed_spur_count': len(removed),
        'removed_spur_length_px': round(removed_length, 6),
        'removed_spur_length': round(removed_length * scale, 6),
        'removed_pixel_count': int(raw_skeleton.sum()) - int(pruned.sum()),
        'removed_length_percent': round(100 * removed_length / max(raw['total_length_px'], 1e-12), 3),
        'node_change': described['node_count'] - raw['node_count'],
        'junction_change': described['junction_count'] - raw['junction_count'],
        'endpoint_change': described['endpoint_count'] - raw['endpoint_count'],
        'branch_change': described['branch_count'] - raw['branch_count'],
        'loop_change': described['loop_count'] - raw['loop_count'],
        'total_length_change_px': round(described['total_length_px'] - raw['total_length_px'], 6),
        'total_length_change': round(
            (described['total_length_px'] - raw['total_length_px']) * scale, 6
        ),
        'skeleton_pixel_change': int(pruned.sum()) - int(raw_skeleton.sum()),
    }
