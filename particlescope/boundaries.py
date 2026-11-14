import math
import numpy as np
from scipy import ndimage as ndi
from .statistics import scalar, describe
from .validation import ValidationError


def boundary_pixels(mask):
    return mask & ~ndi.binary_erosion(mask, border_value=0)


def surface_distances(predicted, reference, valid=None, scale=1):
    if predicted.shape != reference.shape:
        raise ValidationError('两个掩膜尺寸必须一致')
    if valid is None:
        valid = np.ones(predicted.shape, dtype=bool)
    first = boundary_pixels(predicted & valid)
    second = boundary_pixels(reference & valid)
    if not np.any(first) or not np.any(second):
        return {
            'hausdorff': None,
            'hausdorff95': None,
            'average_symmetric_surface_distance': None,
            'reason': '至少一个掩膜没有边界',
        }
    to_second = ndi.distance_transform_edt(~second)[first] * scale
    to_first = ndi.distance_transform_edt(~first)[second] * scale
    combined = np.concatenate([to_second, to_first])
    return {
        'hausdorff': scalar(combined.max()),
        'hausdorff95': scalar(np.percentile(combined, 95)),
        'average_symmetric_surface_distance': scalar(combined.mean()),
        'predicted_to_reference': describe(to_second),
        'reference_to_predicted': describe(to_first),
    }


def tolerance_scores(predicted, reference, tolerances=None):
    tolerances = tolerances or [1, 2, 3, 5]
    first = boundary_pixels(predicted)
    second = boundary_pixels(reference)
    if not np.any(first) or not np.any(second):
        return []
    distance_to_first = ndi.distance_transform_edt(~first)
    distance_to_second = ndi.distance_transform_edt(~second)
    results = []
    for tolerance in tolerances:
        precision = float(np.mean(distance_to_second[first] <= tolerance))
        recall = float(np.mean(distance_to_first[second] <= tolerance))
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        results.append({
            'tolerance_px': tolerance,
            'boundary_precision': scalar(precision),
            'boundary_recall': scalar(recall),
            'boundary_f1': scalar(f1),
        })
    return results


def radial_signature(labels, particle_id, bins=72):
    mask = labels == particle_id
    if not np.any(mask):
        raise ValidationError('颗粒不存在')
    y, x = np.nonzero(mask)
    center_x, center_y = float(x.mean()), float(y.mean())
    by, bx = np.nonzero(boundary_pixels(mask))
    angles = np.arctan2(by - center_y, bx - center_x) % (2 * math.pi)
    radii = np.hypot(bx - center_x, by - center_y)
    indices = np.minimum((angles / (2 * math.pi) * bins).astype(int), bins - 1)
    maximum = np.zeros(bins)
    counts = np.bincount(indices, minlength=bins)
    np.maximum.at(maximum, indices, radii)
    known = np.flatnonzero(counts)
    if len(known) < 2:
        raise ValidationError('边界点不足')
    interpolated = np.interp(np.arange(bins), known, maximum[known], period=bins)
    average = float(interpolated.mean())
    normalized = interpolated / average if average else interpolated
    fourier = np.abs(np.fft.rfft(normalized)) / bins
    return {
        'particle_id': particle_id,
        'centroid': [scalar(center_x), scalar(center_y)],
        'angles_degrees': [scalar(v) for v in np.linspace(0, 360, bins, endpoint=False)],
        'radius_px': [scalar(v) for v in interpolated],
        'normalized_radius': [scalar(v) for v in normalized],
        'fourier_magnitudes': [scalar(v) for v in fourier[:16]],
        'radial_cv': scalar(interpolated.std() / average) if average else None,
        'warning': '每个角度取最外侧边界；对凹形和带孔对象会丢失内部细节',
    }


def object_overlap(first, second, minimum_iou=0.5):
    if first.shape != second.shape:
        raise ValidationError('实例掩膜尺寸必须一致')
    first_count = int(first.max())
    second_count = int(second.max())
    first_sizes = np.bincount(first.ravel(), minlength=first_count + 1)
    second_sizes = np.bincount(second.ravel(), minlength=second_count + 1)
    valid = (first > 0) & (second > 0)
    packed = first[valid].astype(np.int64) * (second_count + 1) + second[valid]
    pair_values, intersections = np.unique(packed, return_counts=True)
    candidates = []
    for pair, intersection in zip(pair_values, intersections):
        first_id = int(pair // (second_count + 1))
        second_id = int(pair % (second_count + 1))
        union = int(first_sizes[first_id] + second_sizes[second_id] - intersection)
        iou = float(intersection / union)
        if iou >= minimum_iou:
            candidates.append((iou, first_id, second_id))
    candidates.sort(reverse=True)
    matches = []
    used_first = set()
    used_second = set()
    for iou, first_id, second_id in candidates:
        if first_id in used_first or second_id in used_second:
            continue
        used_first.add(first_id)
        used_second.add(second_id)
        matches.append({'first_id': first_id, 'second_id': second_id, 'iou': scalar(iou)})
    n_first = int(np.count_nonzero(first_sizes[1:]))
    n_second = int(np.count_nonzero(second_sizes[1:]))
    precision = len(matches) / n_first if n_first else None
    recall = len(matches) / n_second if n_second else None
    return {
        'matches': matches,
        'first_objects': n_first,
        'second_objects': n_second,
        'matched_objects': len(matches),
        'precision': scalar(precision) if precision is not None else None,
        'recall': scalar(recall) if recall is not None else None,
        'iou_threshold': minimum_iou,
        'method': '按IoU降序贪心一对一匹配',
    }


def reference_boundary_report(predicted, reference, selected, scale=1):
    p = predicted & selected
    r = reference & selected
    first_labels, _ = ndi.label(p)
    second_labels, _ = ndi.label(r)
    return {
        'surface_distance': surface_distances(p, r, selected, scale),
        'boundary_tolerance': tolerance_scores(p, r),
        'connected_component_matching': object_overlap(first_labels, second_labels),
        'warning': '参考为二值连通对象，粘连实例无法作为独立实例评分',
    }
