import math
import numpy as np
from scipy import ndimage as ndi
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


def _squared_distances_into(edt_float64, target_int32):
    """Square a float64 EDT in place and store its exact integer result.

    EDT values are square roots of integer squared grid distances, so their
    square rounds to that integer. The float64 array is squared in place, then
    rounded directly into the caller's int32 map without a second full-size
    float buffer.
    """
    edt_float64 *= edt_float64
    np.rint(edt_float64, out=edt_float64)
    np.copyto(target_int32, edt_float64, casting='unsafe')


def granulometry(binary, selected, radii):
    """Disk-opening granulometry with per-scale, exact area results.

    Each reported area is the pixel count of the image opened by the closed
    Euclidean disk of that radius and intersected with the current analysis
    region ``selected`` — the same definition as
    ``ndi.binary_opening(base, disk(r)) & selected``. No image is downscaled
    and no surrogate size estimator is substituted.

    Speed and memory come from expressing the two morphological primitives as
    exact Euclidean distance transforms (EDT), which scipy evaluates in linear
    time with a small constant instead of its generic O(radius^2) neighborhood
    filters:

    * disk erosion keeps a foreground pixel iff its distance to the nearest
      background is strictly greater than the radius (the disk includes its
      boundary, so a background point at distance == radius still excludes it);
    * disk dilation keeps a pixel iff its distance to an eroded pixel is at
      most the radius.

    The foreground EDT is evaluated once on a zero-padded buffer whose margin
    is the largest requested radius; every scale's erosion is the same fixed
    image window thresholded at that scale, so adding a scale costs only one
    extra dilation EDT plus an O(N) bool copy. Each scale's erosion is embedded
    in a max-margin buffer with zeros spanning its own radius-wide border,
    because the dilation's outside-image zero border must be exactly one disk
    wide (erosion can legitimately reach the image edge).

    scipy only emits float64 EDTs. The one-time foreground map is converted to
    its exact integer squared distance (grid distances are integral) and held
    in int32, since it is reused for every scale; the per-scale dilation map is
    consumed immediately and thresholded directly as float64 squared distance,
    which is exact at every supported radius (no achievable grid squared
    distance can round across an integer threshold). A single float64 EDT
    scratch map, one padded bool erosion buffer, its preallocated complement
    and one image-sized bool buffer (erosion, then the opened mask) are reused
    across all scales, so the retained working set does not grow with the
    number of radii. Zero padding reproduces scipy's ``border_value=0``
    border handling.
    """
    base = binary & selected
    initial = int(base.sum())
    rows = []
    last = initial
    if initial == 0 or not radii:
        for radius in radii:
            rows.append({
                'radius_px': radius,
                'remaining_pixels': 0,
                'remaining_percent': 0.0,
                'removed_from_previous': 0,
            })
        return {
            'initial_pixels': initial,
            'scales': rows,
            'method': '圆盘结构元素开运算颗粒谱，不等同于逐颗粒粒径分布',
        }

    height, width = base.shape
    # The shared foreground distance map is padded by the largest requested
    # radius. Image pixel (y, x) always lives at padded index (y + margin,
    # x + margin); thresholding that fixed window reproduces exactly the disk
    # erosion obtained from an image padded by any smaller radius. Capping the
    # margin at the image extent keeps the footprint bounded when a radius
    # exceeds the image, in which case every scale's erosion is already empty.
    margin = min(max(radii), max(height, width))
    padded_shape = (height + 2 * margin, width + 2 * margin)
    # Reusable float64 EDT output (scipy requires float64) and the retained
    # integer foreground squared-distance map used by every scale's erosion.
    edt_scratch = np.empty(padded_shape, dtype=np.float64)
    foreground_squared = np.empty(padded_shape, dtype=np.int32)
    padded_erosion = np.zeros(padded_shape, dtype=bool)
    complement = np.empty(padded_shape, dtype=bool)
    padded_base = np.pad(base, margin, mode='constant', constant_values=False)
    ndi.distance_transform_edt(padded_base, distances=edt_scratch)
    del padded_base
    _squared_distances_into(edt_scratch, foreground_squared)
    inner = (slice(margin, margin + height), slice(margin, margin + width))
    # Holds one scale's erosion first, then is overwritten with its opened mask.
    mask = np.empty(base.shape, dtype=bool)

    for radius in radii:
        radius_squared = radius * radius
        np.greater(foreground_squared[inner], radius_squared, out=mask)
        if not mask.any():
            area = 0
        else:
            # Place the erosion at a radius-wide zero border inside the shared
            # padded buffer; the unused outer band stays zero, which is exactly
            # the radius-padded input the disk dilation EDT requires.
            padded_erosion.fill(False)
            shift = margin - min(radius, margin)
            target = (
                slice(shift, shift + height),
                slice(shift, shift + width),
            )
            padded_erosion[target] = mask
            # Reuse the complement buffer instead of allocating ~padded_erosion.
            np.logical_not(padded_erosion, out=complement)
            ndi.distance_transform_edt(complement, distances=edt_scratch)
            edt_scratch *= edt_scratch
            np.less_equal(edt_scratch[target], radius_squared, out=mask)
            area = int(np.count_nonzero(mask & selected))
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
