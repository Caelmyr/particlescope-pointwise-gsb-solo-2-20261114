import math
import numpy as np
from scipy import ndimage as ndi
from .imaging import disk
from .statistics import scalar, describe
from .validation import ValidationError


def porosity_report(binary, selected):
    foreground = binary & selected
    void = ~binary & selected
    labels, count = ndi.label(void)
    boundary = selected & ~ndi.binary_erosion(selected, border_value=0)
    exterior_ids = set(np.unique(labels[boundary]).tolist())
    sizes = np.bincount(labels.ravel())
    closed = []
    for object_id in range(1, count + 1):
        if object_id in exterior_ids:
            continue
        closed.append(int(sizes[object_id]))
    area = int(selected.sum())
    return {
        'roi_pixels': area,
        'foreground_pixels': int(foreground.sum()),
        'void_pixels': int(void.sum()),
        'void_fraction': scalar(void.sum() / max(area, 1)),
        'closed_void_count': len(closed),
        'closed_void_area_px': sum(closed),
        'closed_void_summary': describe(closed),
        'warning': '孔隙按分割背景定义；图像阈值改变会影响孔隙率',
    }


def granulometry(binary, selected, radii):
    base = binary & selected
    initial = int(base.sum())
    rows = []
    last = initial
    for radius in radii:
        opened = ndi.binary_opening(base, structure=disk(radius)) & selected
        area = int(opened.sum())
        rows.append({
            'radius_px': radius,
            'remaining_pixels': area,
            'remaining_percent': scalar(area / max(initial, 1) * 100),
            'removed_from_previous': max(0, last - area),
        })
        last = area
    return {
        'initial_pixels': initial,
        'scales': rows,
        'method': '圆盘结构元素开运算颗粒谱，不等同于逐颗粒粒径分布',
    }


def distance_thickness(binary, selected, scale=1):
    foreground = binary & selected
    distance = ndi.distance_transform_edt(np.pad(foreground, 1))[1:-1, 1:-1]
    maxima = (distance == ndi.maximum_filter(distance, size=3)) & foreground
    radii = distance[maxima] * scale
    return {
        'inscribed_diameter_summary': describe(radii * 2),
        'maximum_inscribed_diameter': scalar(distance.max() * scale * 2),
        'mean_foreground_distance': scalar(distance[foreground].mean() * scale) if np.any(foreground) else 0,
        'local_peak_count': int(maxima.sum()),
        'warning': '局部内切直径峰值可能多个属于同一个对象，不是对象数量',
    }


def minkowski_curve(gray, selected, thresholds):
    rows = []
    from .measurements import crofton_perimeter
    for threshold in thresholds:
        binary = (gray <= threshold) & selected
        components, component_count = ndi.label(binary)
        filled = ndi.binary_fill_holes(binary)
        holes, hole_count = ndi.label(filled & ~binary)
        rows.append({
            'threshold': float(threshold),
            'area_pixels': int(binary.sum()),
            'perimeter_px': scalar(crofton_perimeter(binary)) if np.any(binary) else 0,
            'components': int(component_count),
            'holes': int(hole_count),
            'euler_number': int(component_count - hole_count),
        })
    return rows


def box_counting(binary, sizes=None):
    binary = np.asarray(binary, dtype=bool)
    height, width = binary.shape
    sizes = sizes or [2, 4, 8, 16, 32, 64]
    scales = []
    for size in sizes:
        if size > min(height, width):
            continue
        pad_h = (-height) % size
        pad_w = (-width) % size
        padded = np.pad(binary, ((0, pad_h), (0, pad_w)))
        blocks = padded.reshape(padded.shape[0] // size, size, padded.shape[1] // size, size)
        occupied = blocks.any(axis=(1, 3))
        count = int(occupied.sum())
        scales.append({'box_size': size, 'occupied_boxes': count})
    positive = [r for r in scales if r['occupied_boxes'] > 0]
    if len(positive) >= 3:
        x = np.log([1 / r['box_size'] for r in positive])
        y = np.log([r['occupied_boxes'] for r in positive])
        slope, intercept = np.polyfit(x, y, 1)
        fitted = slope * x + intercept
        denominator = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - np.sum((y - fitted) ** 2) / denominator if denominator else None
    else:
        slope = r_squared = None
    return {
        'scales': scales,
        'estimated_dimension': scalar(slope) if slope is not None else None,
        'r_squared': scalar(r_squared) if r_squared is not None else None,
        'warning': '有限尺度盒计数估计，对分辨率、阈值和盒网格起点敏感',
    }


def skeletonize(binary, max_iterations=500):
    current = np.pad(binary.astype(np.uint8), 1)
    iterations = 0
    for iteration in range(max_iterations):
        removed = 0
        for phase in (0, 1):
            center = current[1:-1, 1:-1]
            neighbors = [
                current[:-2, 1:-1],
                current[:-2, 2:],
                current[1:-1, 2:],
                current[2:, 2:],
                current[2:, 1:-1],
                current[2:, :-2],
                current[1:-1, :-2],
                current[:-2, :-2],
            ]
            total = sum(neighbors)
            transitions = sum((neighbors[i] == 0) & (neighbors[(i + 1) % 8] == 1) for i in range(8))
            condition = (center == 1) & (total >= 2) & (total <= 6) & (transitions == 1)
            if phase == 0:
                condition &= neighbors[0] * neighbors[2] * neighbors[4] == 0
                condition &= neighbors[2] * neighbors[4] * neighbors[6] == 0
            else:
                condition &= neighbors[0] * neighbors[2] * neighbors[6] == 0
                condition &= neighbors[0] * neighbors[4] * neighbors[6] == 0
            removed += int(condition.sum())
            center[condition] = 0
        iterations = iteration + 1
        if removed == 0:
            break
    return current[1:-1, 1:-1].astype(bool), iterations, removed == 0


def skeleton_report(binary, selected, scale=1):
    if binary.size > 4000000:
        raise ValidationError('骨架分析限400万像素')
    skeleton, iterations, converged = skeletonize(binary & selected)
    neighbors = ndi.convolve(skeleton.astype(int), np.ones((3, 3), dtype=int)) - skeleton.astype(int)
    endpoints = skeleton & (neighbors == 1)
    branch = skeleton & (neighbors >= 3)
    branch_labels, branch_count = ndi.label(branch)
    components, count = ndi.label(skeleton, np.ones((3, 3)))
    y, x = np.nonzero(endpoints)
    return skeleton, {
        'skeleton_pixels': int(skeleton.sum()),
        'approximate_length': scalar(skeleton.sum() * scale),
        'endpoint_count': int(endpoints.sum()),
        'branch_region_count': int(branch_count),
        'component_count': int(count),
        'endpoints': [[int(a), int(b)] for a, b in zip(x, y)],
        'iterations': iterations,
        'converged': converged,
        'warning': '长度为骨架像素数近似，未对斜边和分叉重叠修正',
    }
