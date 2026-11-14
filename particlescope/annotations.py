import math
import re
import uuid
from copy import deepcopy
import numpy as np
from .validation import ValidationError, choice, sequence, point, polygon, text, now


def line_measurement(points, scale):
    first, second = points
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    length = math.hypot(dx, dy)
    if length < 0.01:
        raise ValidationError('测量线长度过小')
    return {
        'length_px': round(length, 6),
        'length': round(length * scale, 6),
        'angle': round(math.degrees(math.atan2(dy, dx)) % 180, 6),
        'delta_x': round(dx * scale, 6),
        'delta_y': round(dy * scale, 6),
    }


def angle_measurement(points):
    a, center, b = np.asarray(points, dtype=float)
    first = a - center
    second = b - center
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator < 1e-9:
        raise ValidationError('角度的两条边不能为零')
    cosine = float(np.clip(np.dot(first, second) / denominator, -1, 1))
    angle = math.degrees(math.acos(cosine))
    return {'angle': round(angle, 6)}


def polygon_measurement(points, scale):
    signed = 0
    perimeter = 0
    weighted_x = 0
    weighted_y = 0
    for index, a in enumerate(points):
        b = points[(index + 1) % len(points)]
        cross = a[0] * b[1] - b[0] * a[1]
        signed += cross
        weighted_x += (a[0] + b[0]) * cross
        weighted_y += (a[1] + b[1]) * cross
        perimeter += math.dist(a, b)
    area = abs(signed) / 2
    cx = weighted_x / (3 * signed) if signed else 0
    cy = weighted_y / (3 * signed) if signed else 0
    return {
        'area_px': round(area, 6),
        'area': round(area * scale ** 2, 6),
        'perimeter': round(perimeter * scale, 6),
        'centroid_x': round(cx, 6),
        'centroid_y': round(cy, 6),
    }


def validate_annotation(body, meta):
    kind = choice(body.get('type'), '标注类型', ['line', 'angle', 'polygon', 'point'])
    raw = sequence(body.get('points'), '测量坐标', 200)
    if kind == 'line' and len(raw) != 2:
        raise ValidationError('线段需要两个点')
    if kind == 'angle' and len(raw) != 3:
        raise ValidationError('角度需要三个点，中间点为顶点')
    if kind == 'point' and len(raw) != 1:
        raise ValidationError('点标记需要一个点')
    if kind == 'polygon':
        points = polygon(raw, meta['width'], meta['height'])
    else:
        points = [point(p, meta['width'], meta['height']) for p in raw]
    color = body.get('color', '#ffcf66')
    if not isinstance(color, str) or not re.fullmatch('#[a-fA-F0-9]{6}', color):
        raise ValidationError('颜色必须为十六进制RGB')
    return {
        'type': kind,
        'points': points,
        'label': text(body.get('label', ''), '标注说明', 200),
        'color': color,
    }


def measure_annotation(annotation, calibration):
    result = deepcopy(annotation)
    kind = result['type']
    points = result['points']
    scale = calibration['scale']
    if kind == 'line':
        measured = line_measurement(points, scale)
    elif kind == 'angle':
        measured = angle_measurement(points)
    elif kind == 'polygon':
        measured = polygon_measurement(points, scale)
    else:
        measured = {'x': points[0][0], 'y': points[0][1]}
    result['measurement'] = measured
    result['unit'] = calibration['unit']
    return result


def add_annotation(store, sample_id, body, expected_revision=None):
    with store.transaction(sample_id, expected_revision) as (path, meta):
        annotations = store.json_read(path / 'annotations.json', [])
        if len(annotations) >= 1000:
            raise ValidationError('每张图像最多1000个标注')
        annotation = validate_annotation(body, meta)
        annotation['id'] = uuid.uuid4().hex
        annotation['created_at'] = now()
        annotation['image_revision'] = meta['image_revision']
        annotation = measure_annotation(annotation, meta['calibration'])
        annotations.append(annotation)
        store.json_write(path / 'annotations.json', annotations)
    return annotation


def delete_annotation(store, sample_id, annotation_id, expected_revision=None):
    with store.transaction(sample_id, expected_revision) as (path, meta):
        annotations = store.json_read(path / 'annotations.json', [])
        selected = [a for a in annotations if a['id'] != annotation_id]
        if len(selected) == len(annotations):
            raise ValidationError('标注不存在')
        store.json_write(path / 'annotations.json', selected)
    return {'deleted': annotation_id}


def update_annotation(store, sample_id, annotation_id, body, expected_revision=None):
    with store.transaction(sample_id, expected_revision) as (path, meta):
        annotations = store.json_read(path / 'annotations.json', [])
        selected = next((a for a in annotations if a['id'] == annotation_id), None)
        if selected is None:
            raise ValidationError('标注不存在')
        merged = dict(selected)
        merged.update(body)
        validated = validate_annotation(merged, meta)
        validated['id'] = annotation_id
        validated['created_at'] = selected['created_at']
        validated['updated_at'] = now()
        validated['image_revision'] = meta['image_revision']
        validated = measure_annotation(validated, meta['calibration'])
        index = annotations.index(selected)
        annotations[index] = validated
        store.json_write(path / 'annotations.json', annotations)
    return validated


def recalibrate_annotations(store, sample_id, calibration):
    path = store.sample_dir(sample_id)
    annotations = store.json_read(path / 'annotations.json', [])
    result = [measure_annotation(a, calibration) for a in annotations]
    store.json_write(path / 'annotations.json', result)
    return result


def annotation_summary(annotations):
    groups = {'line': [], 'angle': [], 'polygon': [], 'point': []}
    for annotation in annotations:
        groups[annotation['type']].append(annotation)
    lengths = [a['measurement']['length'] for a in groups['line']]
    areas = [a['measurement']['area'] for a in groups['polygon']]
    angles = [a['measurement']['angle'] for a in groups['angle']]
    return {
        'count': len(annotations),
        'line_count': len(lengths),
        'polygon_count': len(areas),
        'angle_count': len(angles),
        'point_count': len(groups['point']),
        'total_length': round(sum(lengths), 6),
        'total_polygon_area': round(sum(areas), 6),
        'mean_line_length': round(float(np.mean(lengths)), 6) if lengths else None,
        'mean_angle': round(float(np.mean(angles)), 6) if angles else None,
    }
