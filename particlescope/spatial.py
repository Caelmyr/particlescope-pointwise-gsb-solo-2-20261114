import math
import numpy as np
from scipy.spatial import cKDTree, distance
from scipy import ndimage as ndi
from .statistics import describe, scalar
from .measurements import included_rows
from .validation import ValidationError


def coordinates(rows, scale=1):
    if not rows:
        return np.empty((0, 2))
    return np.array([[r['centroid_x'] * scale, r['centroid_y'] * scale] for r in rows])


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
    core = neighborhoods >= min_points
    labels = np.full(len(points), -1, dtype=int)
    visited = np.zeros(len(points), dtype=bool)
    cluster_id = 0
    for index in range(len(points)):
        if visited[index]:
            continue
        visited[index] = True
        if not core[index]:
            continue
        labels[index] = cluster_id
        queue = [index]
        cursor = 0
        while cursor < len(queue):
            center = queue[cursor]
            cursor += 1
            neighbors = tree.query_ball_point(points[center], r=radius)
            for neighbor in neighbors:
                if labels[neighbor] == -1:
                    labels[neighbor] = cluster_id
                if visited[neighbor]:
                    continue
                visited[neighbor] = True
                if core[neighbor]:
                    queue.append(neighbor)
        cluster_id += 1
    clusters = []
    for label in range(cluster_id):
        indices = np.flatnonzero(labels == label)
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
