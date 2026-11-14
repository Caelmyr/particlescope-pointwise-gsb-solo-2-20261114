import math
import numpy as np
from scipy import ndimage as ndi
from .statistics import scalar, describe
from .measurements import included_rows
from .validation import ValidationError


def calibration_uncertainty(result, pixel_length, physical_length, pixel_error, physical_error):
    if pixel_length <= 0 or physical_length <= 0:
        raise ValidationError('标定长度必须大于0')
    relative = math.sqrt((pixel_error / pixel_length) ** 2 + (physical_error / physical_length) ** 2)
    rows = []
    for particle in included_rows(result):
        length_error = particle['diameter'] * relative
        area_error = particle['area'] * 2 * relative
        rows.append({
            'id': particle['id'],
            'diameter': particle['diameter'],
            'diameter_calibration_std': scalar(length_error),
            'area': particle['area'],
            'area_calibration_std': scalar(area_error),
        })
    return {
        'relative_length_std': scalar(relative),
        'relative_area_std': scalar(2 * relative),
        'particles': rows,
        'assumption': '独立小误差的一阶传播，仅计标定误差，不含边界分割偏差',
    }


def boundary_sensitivity(labels, result, pixels=1):
    rows = []
    for particle in included_rows(result):
        object_id = particle['id']
        left = max(0, particle['bbox_x'] - pixels - 1)
        top = max(0, particle['bbox_y'] - pixels - 1)
        right = min(labels.shape[1], particle['bbox_x'] + particle['bbox_width'] + pixels + 1)
        bottom = min(labels.shape[0], particle['bbox_y'] + particle['bbox_height'] + pixels + 1)
        mask = labels[top:bottom, left:right] == object_id
        smaller = ndi.binary_erosion(mask, iterations=pixels)
        larger = ndi.binary_dilation(mask, iterations=pixels)
        area_min = int(smaller.sum()) * result['scale'] ** 2
        area_max = int(larger.sum()) * result['scale'] ** 2
        diameter_min = 2 * math.sqrt(area_min / math.pi)
        diameter_max = 2 * math.sqrt(area_max / math.pi)
        rows.append({
            'id': object_id,
            'area': particle['area'],
            'area_eroded': scalar(area_min),
            'area_dilated': scalar(area_max),
            'diameter': particle['diameter'],
            'diameter_eroded': scalar(diameter_min),
            'diameter_dilated': scalar(diameter_max),
            'diameter_span_percent': scalar((diameter_max - diameter_min) / max(particle['diameter'], 1e-12) * 100),
        })
    return {
        'boundary_shift_px': pixels,
        'particles': rows,
        'span_summary': describe([r['diameter_span_percent'] for r in rows]),
        'warning': '独立腐蚀和膨胀估计边界敏感性，不是统计置信区间；碰到图边可能截断',
    }


def stability_plateaus(sweep, relative_count_tolerance=0.05, absolute_coverage_tolerance=1):
    if not sweep:
        return {'plateaus': [], 'recommended_threshold': None}
    plateaus = []
    start = 0
    for index in range(1, len(sweep) + 1):
        if index < len(sweep):
            first = sweep[index - 1]
            second = sweep[index]
            relative = abs(second['count'] - first['count']) / max(first['count'], 1)
            coverage = abs(second['coverage'] - first['coverage'])
            stable = relative <= relative_count_tolerance and coverage <= absolute_coverage_tolerance
        else:
            stable = False
        if stable:
            continue
        segment = sweep[start:index]
        if len(segment) >= 2 and any(row['count'] for row in segment):
            counts = [row['count'] for row in segment]
            plateaus.append({
                'threshold_start': segment[0]['threshold'],
                'threshold_end': segment[-1]['threshold'],
                'samples': len(segment),
                'count_min': min(counts),
                'count_max': max(counts),
                'midpoint': (segment[0]['threshold'] + segment[-1]['threshold']) / 2,
            })
        start = index
    plateaus.sort(key=lambda row: row['threshold_end'] - row['threshold_start'], reverse=True)
    return {
        'plateaus': plateaus,
        'recommended_threshold': plateaus[0]['midpoint'] if plateaus else None,
        'warning': '稳定平台不保证分割准确，必须结合原图、掩膜或参考图检查',
    }


def sampling_precision(result, target_relative_error=0.05):
    values = np.asarray([r['diameter'] for r in included_rows(result)], dtype=float)
    if len(values) < 2 or values.mean() == 0:
        raise ValidationError('至少需要两个有效颗粒估计抽样精度')
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    relative_se = std / math.sqrt(len(values)) / abs(mean)
    approximate_required = math.ceil((1.96 * std / (target_relative_error * abs(mean))) ** 2)
    return {
        'observed_count': len(values),
        'mean_diameter': scalar(mean),
        'sample_std': scalar(std),
        'relative_standard_error': scalar(relative_se),
        'target_relative_half_width': target_relative_error,
        'approximate_required_count': max(2, approximate_required),
        'additional_count': max(0, approximate_required - len(values)),
        'assumption': '正态近似95%均值区间与独立同分布颗粒；不是实验设计保证',
    }


def spatial_block_bootstrap(result, width, height, blocks_x=4, blocks_y=4, repeats=1000, seed=42):
    rows = included_rows(result)
    groups = [[] for _ in range(blocks_x * blocks_y)]
    for row in rows:
        column = min(blocks_x - 1, int(row['centroid_x'] / width * blocks_x))
        line = min(blocks_y - 1, int(row['centroid_y'] / height * blocks_y))
        groups[line * blocks_x + column].append(row['diameter'])
    occupied = sum(bool(group) for group in groups)
    if occupied < 2:
        raise ValidationError('至少需要两个非空空间块')
    rng = np.random.default_rng(seed)
    sums = np.asarray([sum(group) for group in groups])
    counts = np.asarray([len(group) for group in groups])
    means = []
    for _ in range(repeats):
        indices = rng.integers(0, len(groups), len(groups))
        total_count = int(counts[indices].sum())
        if total_count:
            means.append(float(sums[indices].sum() / total_count))
    lower, upper = np.percentile(means, [2.5, 97.5])
    return {
        'lower': scalar(lower),
        'upper': scalar(upper),
        'block_count': len(groups),
        'occupied_blocks': occupied,
        'blocks_x': blocks_x,
        'blocks_y': blocks_y,
        'valid_repeats': len(means),
        'seed': seed,
        'warning': '整块重采样保留块内关联，区间对块大小敏感，不替代独立视野重复',
    }
