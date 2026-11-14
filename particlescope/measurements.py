import math
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, QhullError, distance
from .validation import ValidationError


MEASUREMENT_FIELDS = {
    'area': {'label': '面积', 'power': 2},
    'diameter': {'label': '等效直径', 'power': 1},
    'perimeter': {'label': '周长估计', 'power': 1},
    'major_axis': {'label': '长轴', 'power': 1},
    'minor_axis': {'label': '短轴', 'power': 1},
    'feret_max': {'label': '最大Feret径', 'power': 1},
    'feret_min': {'label': '最小Feret径', 'power': 1},
    'circularity': {'label': '圆度', 'power': 0},
    'aspect_ratio': {'label': '长宽比', 'power': 0},
    'solidity': {'label': '实心度', 'power': 0},
    'extent': {'label': '矩形填充率', 'power': 0},
    'mean_intensity': {'label': '平均灰度', 'power': 0},
    'std_intensity': {'label': '灰度标准差', 'power': 0},
    'orientation': {'label': '方向角', 'power': 0},
    'eccentricity': {'label': '偏心率', 'power': 0},
    'roughness': {'label': '边界粗糙度', 'power': 0},
}


def round_value(value, digits=6):
    value = float(value)
    if not math.isfinite(value):
        return None
    return round(value, digits)


def crofton_perimeter(mask):
    padded = np.pad(mask.astype(np.int8), 1)
    horizontal = np.count_nonzero(np.diff(padded, axis=1))
    vertical = np.count_nonzero(np.diff(padded, axis=0))
    diagonal_a = np.count_nonzero(padded[1:, 1:] != padded[:-1, :-1])
    diagonal_b = np.count_nonzero(padded[1:, :-1] != padded[:-1, 1:])
    perimeter = math.pi / 8 * (horizontal + vertical + (diagonal_a + diagonal_b) / math.sqrt(2))
    return max(float(perimeter), 1)


def covariance_axes(x, y):
    if len(x) < 2:
        return {'major': 1, 'minor': 1, 'orientation': 0, 'eccentricity': 0}
    coordinates = np.column_stack((x, y)).astype(float)
    covariance = np.cov(coordinates, rowvar=False, bias=True)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0)
    vectors = vectors[:, order]
    major = 4 * math.sqrt(values[0])
    minor = 4 * math.sqrt(values[1])
    angle = math.degrees(math.atan2(vectors[1, 0], vectors[0, 0])) % 180
    eccentricity = math.sqrt(max(0, 1 - values[1] / values[0])) if values[0] else 0
    return {
        'major': max(major, 1),
        'minor': max(minor, 1),
        'orientation': angle,
        'eccentricity': eccentricity,
    }


def hull_metrics(mask, x, y):
    boundary = mask & ~ndi.binary_erosion(mask)
    by, bx = np.nonzero(boundary)
    if len(bx) > 10000:
        step = math.ceil(len(bx) / 10000)
        bx = bx[::step]
        by = by[::step]
    points = np.column_stack((bx, by)).astype(float)
    if len(points) < 3:
        return {'area': float(len(x)), 'perimeter': crofton_perimeter(mask), 'max': 1, 'min': 1}
    try:
        hull = ConvexHull(points)
    except QhullError:
        span = float(np.ptp(x)) + 1
        return {'area': float(len(x)), 'perimeter': 2 * span + 2, 'max': span, 'min': 1}
    vertices = points[hull.vertices]
    if len(vertices) > 2000:
        vertices = vertices[::math.ceil(len(vertices) / 2000)]
    maximum = float(distance.pdist(vertices).max()) + 1
    widths = []
    for index, a in enumerate(vertices):
        b = vertices[(index + 1) % len(vertices)]
        edge = b - a
        norm = np.linalg.norm(edge)
        if norm == 0:
            continue
        normal = np.array([-edge[1], edge[0]]) / norm
        projection = vertices @ normal
        widths.append(float(projection.max() - projection.min()) + 1)
    area = max(float(hull.volume), float(len(x)))
    return {
        'area': area,
        'perimeter': float(hull.area),
        'max': maximum,
        'min': min(widths) if widths else 1,
    }


def intensity_metrics(values):
    if not len(values):
        return {
            'mean_intensity': 0,
            'std_intensity': 0,
            'min_intensity': 0,
            'max_intensity': 0,
            'median_intensity': 0,
            'integrated_intensity': 0,
        }
    return {
        'mean_intensity': round_value(values.mean()),
        'std_intensity': round_value(values.std()),
        'min_intensity': round_value(values.min()),
        'max_intensity': round_value(values.max()),
        'median_intensity': round_value(np.median(values)),
        'integrated_intensity': round_value(values.sum()),
    }


def measure_object(object_id, region, mask, gray, scale, image_shape, roi_boundary):
    slices_y, slices_x = region
    y, x = np.nonzero(mask)
    area_px = int(len(x))
    center_x = float(x.mean() + slices_x.start)
    center_y = float(y.mean() + slices_y.start)
    perimeter_px = crofton_perimeter(mask)
    axes = covariance_axes(x, y)
    hull = hull_metrics(mask, x, y)
    border = (
        slices_x.start == 0 or slices_y.start == 0
        or slices_x.stop == image_shape[1]
        or slices_y.stop == image_shape[0]
    )
    touches_roi = bool(np.any(roi_boundary[region] & mask))
    width = slices_x.stop - slices_x.start
    height = slices_y.stop - slices_y.start
    filled = ndi.binary_fill_holes(mask)
    holes, hole_count = ndi.label(filled & ~mask)
    circularity = min(1, 4 * math.pi * area_px / perimeter_px ** 2)
    row = {
        'id': int(object_id),
        'area_px': area_px,
        'area': round_value(area_px * scale ** 2),
        'diameter': round_value(2 * math.sqrt(area_px / math.pi) * scale),
        'perimeter': round_value(perimeter_px * scale),
        'perimeter_px': round_value(perimeter_px),
        'centroid_x': round_value(center_x),
        'centroid_y': round_value(center_y),
        'centroid_x_scaled': round_value(center_x * scale),
        'centroid_y_scaled': round_value(center_y * scale),
        'bbox_x': slices_x.start,
        'bbox_y': slices_y.start,
        'bbox_width': width,
        'bbox_height': height,
        'major_axis': round_value(axes['major'] * scale),
        'minor_axis': round_value(axes['minor'] * scale),
        'orientation': round_value(axes['orientation']),
        'eccentricity': round_value(axes['eccentricity']),
        'aspect_ratio': round_value(axes['major'] / axes['minor']),
        'circularity': round_value(circularity),
        'solidity': round_value(min(1, area_px / max(hull['area'], 1))),
        'extent': round_value(area_px / (width * height)),
        'feret_max': round_value(hull['max'] * scale),
        'feret_min': round_value(hull['min'] * scale),
        'convex_area': round_value(hull['area'] * scale ** 2),
        'roughness': round_value(perimeter_px / max(hull['perimeter'], 1)),
        'hole_count': int(hole_count),
        'hole_area_px': int(np.count_nonzero(filled & ~mask)),
        'touches_edge': bool(border),
        'touches_roi': touches_roi,
        'included': True,
    }
    row.update(intensity_metrics(gray[region][mask]))
    return row


def measure_labels(labels, gray, calibration, selected):
    if labels.shape != gray.shape:
        raise ValidationError('掩膜尺寸与图像不一致')
    scale = calibration['scale']
    roi_boundary = selected & ~ndi.binary_erosion(selected, border_value=0)
    rows = []
    regions = ndi.find_objects(labels)
    for label, region in enumerate(regions, 1):
        if region is None:
            continue
        mask = labels[region] == label
        if not np.any(mask):
            continue
        rows.append(measure_object(label, region, mask, gray, scale, labels.shape, roi_boundary))
    return rows


def apply_filters(rows, filters):
    result = []
    excluded = set(filters.get('exclude_ids', []))
    for source in rows:
        row = dict(source)
        reasons = []
        if row['id'] in excluded:
            reasons.append('人工排除')
        for name in ['area', 'diameter', 'circularity', 'aspect_ratio', 'solidity', 'mean_intensity']:
            lower = filters.get(name + '_min')
            upper = filters.get(name + '_max')
            value = row.get(name)
            if value is None:
                reasons.append(name + '缺失')
                continue
            if lower is not None and value < lower:
                reasons.append(name + '低于下限')
            if upper is not None and value > upper:
                reasons.append(name + '高于上限')
        row['included'] = not reasons
        row['exclusion_reasons'] = reasons
        result.append(row)
    return result


def included_rows(result):
    return [row for row in result['particles'] if row.get('included', True)]


def summarize(rows, selected_pixels, calibration):
    included = [row for row in rows if row.get('included', True)]
    total_area_px = sum(row['area_px'] for row in included)
    diameters = [row['diameter'] for row in included]
    areas = [row['area'] for row in included]
    return {
        'count': len(included),
        'total_detected': len(rows),
        'excluded_count': len(rows) - len(included),
        'coverage': round_value(total_area_px / max(selected_pixels, 1) * 100),
        'mean_diameter': round_value(np.mean(diameters)) if diameters else 0,
        'median_diameter': round_value(np.median(diameters)) if diameters else 0,
        'total_area': round_value(sum(areas)),
        'analyzed_area': round_value(selected_pixels * calibration['scale'] ** 2),
        'number_density': round_value(len(included) / max(selected_pixels * calibration['scale'] ** 2, 1e-12)),
        'unit': calibration['unit'],
        'scale': calibration['scale'],
    }


def select_particle(labels, rows, x, y):
    object_id = int(labels[y, x])
    if object_id == 0:
        return None
    return next((r for r in rows if r['id'] == object_id), None)


def particle_crop(image, labels, object_id, margin=10, isolate=False):
    mask = labels == object_id
    if not np.any(mask):
        raise ValidationError('颗粒编号不存在')
    y, x = np.nonzero(mask)
    left = max(0, int(x.min()) - margin)
    right = min(image.width, int(x.max()) + margin + 1)
    top = max(0, int(y.min()) - margin)
    bottom = min(image.height, int(y.max()) + margin + 1)
    cropped = image.crop((left, top, right, bottom)).convert('RGBA')
    if isolate:
        from PIL import Image
        alpha = (mask[top:bottom, left:right] * 255).astype(np.uint8)
        cropped.putalpha(Image.fromarray(alpha))
    return cropped


def measurement_units(calibration):
    unit = calibration['unit']
    result = {}
    for key, definition in MEASUREMENT_FIELDS.items():
        power = definition['power']
        if power == 0:
            result[key] = ''
        elif power == 1:
            result[key] = unit
        else:
            result[key] = unit + '²'
    result['orientation'] = '°'
    result['mean_intensity'] = '0–255'
    return result
