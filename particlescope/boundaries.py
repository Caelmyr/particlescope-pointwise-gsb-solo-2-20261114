import math
import numpy as np
from scipy import ndimage as ndi
from .statistics import scalar, describe
from .validation import ValidationError


TRUNCATION_NOTE = '分析区域截断产生的人造边缘不参与边界评分，仅比较真实对象轮廓'
TRUNCATION_REASON = '有效区域内没有可比较的真实轮廓（边界均来自分析区域或图像边缘的裁切），无法评价'


def boundary_pixels(mask):
    return mask & ~ndi.binary_erosion(mask, border_value=0)


def split_boundary(mask, valid):
    """把有效区域内掩膜的边界拆成真实对象轮廓与观测截断边缘。

    掩膜被分析区域裁切后，贴着区域边界的边缘是观测截断的产物而非对象
    真实轮廓；两侧掩膜在同一区域裁切下这些人造边缘必然重合，参与评分
    会掩盖真实轮廓的缺失。分析区域始终受图像范围约束，图像边缘处的
    掩膜边界同样属于截断。这里按是否与无效区域或图像外相邻把二者分开。
    """
    boundary = boundary_pixels(mask & valid)
    if not np.any(boundary):
        return boundary, boundary
    near_invalid = ndi.binary_dilation(
        ~valid, structure=ndi.generate_binary_structure(2, 1), border_value=1)
    truncated = boundary & near_invalid
    return boundary & ~truncated, truncated


def boundary_scope(first, first_truncated, second, second_truncated):
    return {
        'predicted_real_boundary_pixels': int(np.count_nonzero(first)),
        'reference_real_boundary_pixels': int(np.count_nonzero(second)),
        'predicted_truncated_pixels': int(np.count_nonzero(first_truncated)),
        'reference_truncated_pixels': int(np.count_nonzero(second_truncated)),
    }


def surface_distances(predicted, reference, valid=None, scale=1):
    if predicted.shape != reference.shape:
        raise ValidationError('两个掩膜尺寸必须一致')
    if valid is None:
        valid = np.ones(predicted.shape, dtype=bool)
    first, first_truncated = split_boundary(predicted, valid)
    second, second_truncated = split_boundary(reference, valid)
    scope = boundary_scope(first, first_truncated, second, second_truncated)
    if not np.any(first) or not np.any(second):
        if scope['predicted_truncated_pixels'] or scope['reference_truncated_pixels']:
            reason = TRUNCATION_REASON
        else:
            reason = '至少一个掩膜在有效区域内没有边界'
        return {
            'hausdorff': None,
            'hausdorff95': None,
            'average_symmetric_surface_distance': None,
            'predicted_to_reference': None,
            'reference_to_predicted': None,
            'evaluable': False,
            'reason': reason,
            **scope,
        }
    to_second = ndi.distance_transform_edt(~second)[first] * scale
    to_first = ndi.distance_transform_edt(~first)[second] * scale
    combined = np.concatenate([to_second, to_first])
    result = {
        'hausdorff': scalar(combined.max()),
        'hausdorff95': scalar(np.percentile(combined, 95)),
        'average_symmetric_surface_distance': scalar(combined.mean()),
        'predicted_to_reference': describe(to_second),
        'reference_to_predicted': describe(to_first),
        'evaluable': True,
        **scope,
    }
    if scope['predicted_truncated_pixels'] or scope['reference_truncated_pixels']:
        result['note'] = TRUNCATION_NOTE
    return result


def tolerance_scores(predicted, reference, valid=None, tolerances=None):
    tolerances = tolerances or [1, 2, 3, 5]
    if valid is None:
        valid = np.ones(predicted.shape, dtype=bool)
    first, _ = split_boundary(predicted, valid)
    second, _ = split_boundary(reference, valid)
    if not np.any(first) or not np.any(second):
        return [{
            'tolerance_px': tolerance,
            'boundary_precision': None,
            'boundary_recall': None,
            'boundary_f1': None,
            'evaluable': False,
        } for tolerance in tolerances]
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
            'evaluable': True,
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
    surface = surface_distances(p, r, selected, scale)
    report = {
        'surface_distance': surface,
        'boundary_tolerance': tolerance_scores(p, r, selected),
        'connected_component_matching': object_overlap(first_labels, second_labels),
        'warning': '参考为二值连通对象，粘连实例无法作为独立实例评分',
    }
    if surface['predicted_truncated_pixels'] or surface['reference_truncated_pixels']:
        report['note'] = TRUNCATION_NOTE + '；像素重叠指标（IoU、Dice等）仍按有效区域内全部像素计算'
    return report
