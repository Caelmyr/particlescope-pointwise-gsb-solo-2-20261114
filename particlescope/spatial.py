import math
import numpy as np
from scipy.spatial import cKDTree, distance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy import ndimage as ndi
from .statistics import describe, scalar
from .measurements import included_rows
from .validation import ValidationError


def coordinates(rows, scale=1):
    if not rows:
        return np.empty((0, 2))
    return np.array([[r['centroid_x'] * scale, r['centroid_y'] * scale] for r in rows])


# 核心点对数超过该阈值时改用分块网格并查，避免一次性物化全部点对导致内存暴涨
_DENSE_PAIR_LIMIT = 20_000_000


def _find_roots(parent, x):
    root = x.copy()
    p = parent[root]
    while (p != root).any():
        parent[root] = parent[p]
        root = parent[root]
        p = parent[root]
    return root


def _hook_edges(parent, first, second):
    while True:
        ra = _find_roots(parent, first)
        rb = _find_roots(parent, second)
        active = ra != rb
        if not active.any():
            return
        lo = np.minimum(ra[active], rb[active])
        hi = np.maximum(ra[active], rb[active])
        np.minimum.at(parent, hi, lo)


def _core_components_blocked(core_points, radius):
    """内存有界的核心点连通分量：按半径大小的网格分块计算候选点对并逐步并查。"""
    n = len(core_points)
    parent = np.arange(n)
    if n < 2:
        return n, np.arange(n)
    cells = np.floor(core_points / radius).astype(np.int64)
    order = np.lexsort((cells[:, 1], cells[:, 0]))
    sorted_cells = cells[order]
    keys, starts = np.unique(sorted_cells, axis=0, return_index=True)
    ends = np.append(starts[1:], n)
    location = {(int(k[0]), int(k[1])): i for i, k in enumerate(keys)}
    r2 = radius * radius
    for ki, key in enumerate(keys):
        # 半邻域偏移，保证每对格子只处理一次
        for dx, dy in ((0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):
            ji = location.get((int(key[0]) + dx, int(key[1]) + dy))
            if ji is None:
                continue
            a_idx = order[starts[ki]:ends[ki]]
            b_idx = order[starts[ji]:ends[ji]]
            pa = core_points[a_idx]
            pb = core_points[b_idx]
            step = max(1, min(len(a_idx), 2_000_000 // max(len(b_idx), 1)))
            for s in range(0, len(a_idx), step):
                block = pa[s:s + step]
                diff_x = np.subtract.outer(block[:, 0], pb[:, 0])
                diff_x *= diff_x
                diff_y = np.subtract.outer(block[:, 1], pb[:, 1])
                diff_y *= diff_y
                diff_x += diff_y
                hit = np.flatnonzero(diff_x <= r2)
                if len(hit):
                    _hook_edges(parent, a_idx[s + hit // len(b_idx)], b_idx[hit % len(b_idx)])
    roots = _find_roots(parent, np.arange(n))
    unique, component = np.unique(roots, return_inverse=True)
    return len(unique), component


def nearest_neighbors(rows, scale=1):
    points = coordinates(rows, scale)
    if len(points) < 2:
        return {'summary': describe([]), 'pairs': [], 'n': len(points)}
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=2)
    pairs = []
    for index, row in enumerate(rows):
        neighbor = int(indices[index, 1])
        pairs.append({
            'id': row['id'],
            'neighbor_id': rows[neighbor]['id'],
            'distance': scalar(distances[index, 1]),
            'surface_gap': scalar(max(0, distances[index, 1] - (row['diameter'] + rows[neighbor]['diameter']) / 2)),
        })
    return {
        'summary': describe(distances[:, 1]),
        'pairs': pairs,
        'n': len(points),
    }


def quadrat_density(rows, selected, scale=1, columns=6, rows_count=4):
    height, width = selected.shape
    x_edges = np.linspace(0, width, columns + 1).astype(int)
    y_edges = np.linspace(0, height, rows_count + 1).astype(int)
    cells = []
    counts = []
    for iy in range(rows_count):
        for ix in range(columns):
            left, right = x_edges[ix:ix + 2]
            top, bottom = y_edges[iy:iy + 2]
            area_px = int(selected[top:bottom, left:right].sum())
            ids = [
                row['id'] for row in rows
                if left <= row['centroid_x'] < right and top <= row['centroid_y'] < bottom
            ]
            area = area_px * scale ** 2
            cells.append({
                'column': ix,
                'row': iy,
                'x': int(left),
                'y': int(top),
                'width': int(right - left),
                'height': int(bottom - top),
                'count': len(ids),
                'area': scalar(area),
                'density': scalar(len(ids) / area) if area else None,
                'ids': ids,
            })
            if area_px:
                counts.append(len(ids))
    mean = float(np.mean(counts)) if counts else 0
    variance = float(np.var(counts, ddof=1)) if len(counts) > 1 else 0
    return {
        'columns': columns,
        'rows': rows_count,
        'cells': cells,
        'count_mean': scalar(mean),
        'count_variance': scalar(variance),
        'dispersion_index': scalar(variance / mean) if mean else None,
        'warning': '不规则ROI各格有效面积不同，离散指数仅适用于近似等面积格网',
    }


def pair_correlation(rows, width, height, scale=1, maximum=None, bins=20):
    points = coordinates(rows, scale)
    if len(points) < 3:
        return {'radii': [], 'g': [], 'pairs': [], 'reason': '至少需要3个颗粒'}
    if len(points) > 5000:
        raise ValidationError('径向分布最多支持5000个颗粒')
    w, h = width * scale, height * scale
    maximum = maximum or min(w, h) / 4
    edges = np.linspace(0, maximum, bins + 1)
    distances = distance.pdist(points)
    counts, _ = np.histogram(distances, bins=edges)
    annulus = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    density = (len(points) - 1) / (w * h)
    expected = len(points) * density * annulus / 2
    g = counts / np.maximum(expected, 1e-12)
    return {
        'radii': [scalar(v) for v in (edges[:-1] + edges[1:]) / 2],
        'g': [scalar(v) for v in g],
        'pairs': counts.tolist(),
        'edges': edges.tolist(),
        'warning': '未进行边缘修正，使用全图矩形面积；不规则ROI不宜据此推断随机性',
    }


def ripley_k(rows, width, height, scale=1, bins=20):
    points = coordinates(rows, scale)
    if len(points) < 3:
        return {'radii': [], 'k': [], 'l_minus_r': [], 'reason': '至少需要3个颗粒'}
    if len(points) > 5000:
        raise ValidationError('K函数最多支持5000个颗粒')
    radii = np.linspace(0, min(width, height) * scale / 4, bins + 1)[1:]
    tree = cKDTree(points)
    pair_counts = tree.count_neighbors(tree, radii) - len(points)
    area = width * height * scale ** 2
    k = area * pair_counts / (len(points) * (len(points) - 1))
    l = np.sqrt(k / math.pi)
    return {
        'radii': [scalar(v) for v in radii],
        'k': [scalar(v) for v in k],
        'l_minus_r': [scalar(v) for v in l - radii],
        'warning': '无边缘修正的探索性K函数，默认观察窗为全图矩形',
    }


def cluster_particles(rows, scale=1, radius=50, min_points=3):
    points = coordinates(rows, scale)
    if not len(points):
        return {'clusters': [], 'noise_ids': [], 'cluster_count': 0}
    tree = cKDTree(points)
    neighborhoods = tree.query_ball_point(points, r=radius, return_length=True)
    core = np.asarray(neighborhoods) >= min_points
    labels = np.full(len(points), -1, dtype=int)
    core_indices = np.flatnonzero(core)
    cluster_id = 0
    if len(core_indices):
        # 核心点之间只要在半径内相连即属同一团簇，用稀疏图连通分量代替逐点BFS，
        # 避免稠密视野下对每个核心点重复查询邻域造成的等待和内存暴涨。
        core_tree = cKDTree(points[core_indices])
        n_core = len(core_indices)
        pair_count = (core_tree.count_neighbors(core_tree, radius) - n_core) // 2
        if pair_count <= _DENSE_PAIR_LIMIT:
            pairs = core_tree.query_pairs(radius, output_type='ndarray')
            if len(pairs):
                # 无向图只存单向边，connected_components(directed=False) 同样按无向处理
                pairs = pairs.astype(np.int32, copy=False)
                graph = csr_matrix(
                    (np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])),
                    shape=(n_core, n_core),
                )
            else:
                graph = csr_matrix((n_core, n_core), dtype=np.int8)
            component_count, component = connected_components(graph, directed=False)
        else:
            component_count, component = _core_components_blocked(points[core_indices], radius)
        # 连通分量按核心点最小原始下标排序编号，与逐点扫描的种子顺序一致，结果可复现。
        first_index = np.full(component_count, len(points), dtype=np.int64)
        np.minimum.at(first_index, component, core_indices)
        remap = np.empty(component_count, dtype=np.int64)
        remap[np.argsort(first_index, kind='stable')] = np.arange(component_count)
        core_labels = remap[component]
        labels[core_indices] = core_labels
        cluster_id = component_count
        border_indices = np.flatnonzero(~core)
        if len(border_indices):
            # 边界点归入半径内最早扩展到的团簇（即相邻核心点中编号最小者）。
            border_tree = cKDTree(points[border_indices])
            cross = border_tree.sparse_distance_matrix(
                core_tree, max_distance=radius, output_type='coo_matrix'
            )
            if cross.nnz:
                best = np.full(len(border_indices), component_count, dtype=np.int64)
                np.minimum.at(best, cross.row, core_labels[cross.col])
                reached = best < component_count
                labels[border_indices[reached]] = best[reached]
    clusters = []
    if cluster_id:
        labeled = np.flatnonzero(labels >= 0)
        ordered = labeled[np.argsort(labels[labeled], kind='stable')]
        sizes = np.bincount(labels[labeled], minlength=cluster_id)
        groups = np.split(ordered, np.cumsum(sizes)[:-1])
        for label, indices in enumerate(groups):
            subset = points[indices]
            clusters.append({
                'id': label + 1,
                'count': len(indices),
                'ids': [rows[i]['id'] for i in indices],
                'centroid_x': scalar(subset[:, 0].mean()),
                'centroid_y': scalar(subset[:, 1].mean()),
                'radius': scalar(np.linalg.norm(subset - subset.mean(axis=0), axis=1).max()),
            })
    return {
        'clusters': clusters,
        'cluster_count': len(clusters),
        'noise_ids': [rows[i]['id'] for i in np.flatnonzero(labels < 0)],
        'radius': radius,
        'min_points': min_points,
    }


def edge_distances(rows, selected, scale=1):
    distance_map = ndi.distance_transform_edt(np.pad(selected, 1))[1:-1, 1:-1]
    values = []
    for row in rows:
        x = min(selected.shape[1] - 1, max(0, int(round(row['centroid_x']))))
        y = min(selected.shape[0] - 1, max(0, int(round(row['centroid_y']))))
        values.append({
            'id': row['id'],
            'distance_to_roi_edge': scalar(distance_map[y, x] * scale),
        })
    return {
        'particles': values,
        'summary': describe([r['distance_to_roi_edge'] for r in values]),
    }


def radial_density(rows, center, selected, scale=1, bins=12):
    height, width = selected.shape
    yy, xx = np.indices((height, width))
    radius_map = np.hypot(xx - center[0], yy - center[1]) * scale
    maximum = float(radius_map[selected].max()) if np.any(selected) else 0
    if maximum == 0:
        raise ValidationError('分析区域为空')
    edges = np.linspace(0, maximum, bins + 1)
    pixel_counts, _ = np.histogram(radius_map[selected], bins=edges)
    radii = [math.hypot(r['centroid_x'] - center[0], r['centroid_y'] - center[1]) * scale for r in rows]
    counts, _ = np.histogram(radii, bins=edges)
    areas = pixel_counts * scale ** 2
    densities = np.divide(counts, areas, out=np.zeros(len(counts), dtype=float), where=areas > 0)
    return {
        'center': center,
        'edges': [scalar(v) for v in edges],
        'counts': counts.tolist(),
        'area': [scalar(v) for v in areas],
        'density': [scalar(v) for v in densities],
    }


def spatial_report(result, selected, columns=6, rows_count=4):
    rows = included_rows(result)
    scale = result['scale']
    nearest = nearest_neighbors(rows, scale)
    area = int(selected.sum()) * scale ** 2
    density = len(rows) / area if area else 0
    expected = 0.5 / math.sqrt(density) if density else None
    observed = nearest['summary']['mean']
    ratio = observed / expected if expected and observed is not None else None
    return {
        'analysis_id': result['id'],
        'unit': result['unit'],
        'nearest_neighbors': nearest,
        'quadrat': quadrat_density(rows, selected, scale, columns, rows_count),
        'edge_distance': edge_distances(rows, selected, scale),
        'random_expected_nn': scalar(expected) if expected else None,
        'clark_evans_ratio': scalar(ratio) if ratio is not None else None,
        'warning': '随机距离参照未做边缘修正，不能单独作为显著性判断',
    }
