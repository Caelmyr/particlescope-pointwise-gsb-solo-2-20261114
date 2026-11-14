import math
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from skimage.measure import find_contours
from .statistics import scalar
from .validation import ValidationError


DEFAULT_SIGMA_PX = 2.0
DEFAULT_MIN_TURN_DEG = 30.0
SAMPLE_STEP_PX = 0.5
MIN_SAMPLES = 64
MAX_SAMPLES = 4000
TURN_WIDENING_LIMIT = 1.35
PROMINENCE_MAD = 4.0


def contour_loops(mask):
    padded = np.pad(np.asarray(mask, dtype=np.uint8), 1, mode='constant')
    loops = []
    for contour in find_contours(padded, 0.5):
        points = np.column_stack((contour[:, 1] - 1, contour[:, 0] - 1)).astype(float)
        if len(points) > 1 and np.allclose(points[0], points[-1]):
            points = points[:-1]
        if len(points) >= 12:
            loops.append(points)
    return loops


def signed_area(points):
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def resample(points):
    closed = np.vstack((points, points[0]))
    segment = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment)))
    length = float(cumulative[-1])
    if length < 12:
        raise ValidationError('颗粒边界过短，无法估计曲率')
    count = int(np.clip(round(length / SAMPLE_STEP_PX), MIN_SAMPLES, MAX_SAMPLES))
    target = np.arange(count) * length / count
    q = np.column_stack((np.interp(target, cumulative, closed[:, 0]),
                         np.interp(target, cumulative, closed[:, 1])))
    return q, length, count, bool(length / SAMPLE_STEP_PX > MAX_SAMPLES)


def left_foreground_fraction(q, mask, offsets=(1.0, 2.0, 3.0)):
    step = np.roll(q, -1, axis=0) - q
    norm = np.maximum(np.linalg.norm(step, axis=1, keepdims=True), 1e-9)
    tangent = step / norm
    left = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    middle = (q + np.roll(q, -1, axis=0)) / 2
    scores = []
    for offset in offsets:
        probe = middle + left * offset
        x = np.clip(np.rint(probe[:, 0]).astype(int), 0, mask.shape[1] - 1)
        y = np.clip(np.rint(probe[:, 1]).astype(int), 0, mask.shape[0] - 1)
        scores.append(float(mask[y, x].mean()))
    return float(np.mean(scores))


def orient(q, mask):
    forward = left_foreground_fraction(q, mask)
    backward = left_foreground_fraction(q[::-1], mask)
    if forward >= backward:
        return q, forward
    return q[::-1], backward


def curvature(q, sigma_px, length):
    n = len(q)
    ds = length / n
    step = np.roll(q, -1, axis=0) - np.roll(q, 1, axis=0)
    angle = np.arctan2(step[:, 1], step[:, 0])
    if sigma_px > 0:
        width = sigma_px / ds
        cosine = ndi.gaussian_filter1d(np.cos(angle), width, mode='wrap')
        sine = ndi.gaussian_filter1d(np.sin(angle), width, mode='wrap')
        angle = np.angle(cosine + 1j * sine)
    return np.gradient(np.unwrap(angle), ds)


def turn_series(q, half):
    n = len(q)
    index = np.arange(n)
    a = q[(index - half) % n]
    b = q[index]
    c = q[(index + half) % n]
    first = b - a
    second = c - b
    cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    dot = first[:, 0] * second[:, 0] + first[:, 1] * second[:, 1]
    return np.degrees(np.arctan2(cross, dot))


def peak_indices(signal, prominence, distance):
    n = len(signal)
    found, _ = find_peaks(np.concatenate((signal, signal, signal)), prominence=prominence, distance=distance)
    keep = (found >= n) & (found < 2 * n)
    return sorted({int(i) for i in (found[keep] - n)})


def median_absolute_deviation(values):
    return 1.4826 * float(np.median(np.abs(values - np.median(values))))


def detect_features(q, sigma_px, min_turn_deg):
    n = len(q)
    length = _arc_length(q)
    ds = length / n
    detection_scale = max(sigma_px, 2.0)
    half_near = max(int(round(4.0 * detection_scale / ds)), 3)
    if 3 * half_near >= n // 2:
        half_near = max(3, n // 8)
    half_far = min(3 * half_near, max(n // 3, half_near + 1))
    near = turn_series(q, half_near)
    far = turn_series(q, half_far)
    strength = near - far / 3.0
    kappa = curvature(q, detection_scale, length)
    prominence = max(PROMINENCE_MAD * median_absolute_deviation(strength), min_turn_deg * 0.5)
    candidates = sorted(set(peak_indices(strength, prominence, max(3, half_near))) |
                        set(peak_indices(-strength, prominence, max(3, half_near))))
    window = max(int(round(2.5 * detection_scale / ds)), 2)
    rows = []
    for index in candidates:
        value = float(strength[index])
        if abs(value) < min_turn_deg:
            continue
        widening = abs(float(far[index])) / max(abs(float(near[index])), 1e-6)
        if widening > TURN_WIDENING_LIMIT:
            continue
        samples = (index + np.arange(-window, window + 1)) % n
        deflection = math.degrees(float(np.sum(kappa[samples])) * ds)
        rows.append({
            'sample_index': int(index),
            'arc_px': scalar(index * ds),
            'arc_fraction': scalar(index / n),
            'x_px': float(q[index, 0]),
            'y_px': float(q[index, 1]),
            'type': 'corner' if value > 0 else 'notch',
            'deflection_deg': scalar(deflection),
            'strength_deg': scalar(value),
            'widening_ratio': scalar(widening),
        })
    rows.sort(key=lambda row: row['arc_px'])
    corners = notches = 0
    for row in rows:
        if row['type'] == 'corner':
            corners += 1
            row['id'] = 'C%d' % corners
        else:
            notches += 1
            row['id'] = 'N%d' % notches
    return rows, kappa


def _arc_length(q):
    closed = np.vstack((q, q[0]))
    return float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1)))


def loop_report(q, kind, hole_index, calibration, sigma_px, min_turn_deg):
    length = _arc_length(q)
    n = len(q)
    ds = length / n
    scale = float(calibration['scale'])
    kappa_px = curvature(q, sigma_px, length)
    equivalent_radius_px = length / (2 * math.pi)
    normalized = kappa_px * equivalent_radius_px
    area_px = abs(signed_area(q))
    features, _ = detect_features(q, sigma_px, min_turn_deg)
    centroid_x = float(q[:, 0].mean())
    centroid_y = float(q[:, 1].mean())
    for row in features:
        value = kappa_px[row['sample_index']]
        row['x'] = scalar(row['x_px'] * scale)
        row['y'] = scalar(row['y_px'] * scale)
        row['position_angle_deg'] = scalar(
            math.degrees(math.atan2(row['y_px'] - centroid_y, row['x_px'] - centroid_x)) % 360)
        row['normalized_curvature'] = scalar(normalized[row['sample_index']])
        row['radius_of_curvature'] = scalar(scale / abs(value)) if value else None
        row['curvature'] = scalar(value / scale) if scale else None
    invariant = _invariant_signature(q, scale)
    profile = _resampled_profile(q, kappa_px, normalized, length, sigma_px)
    return {
        'kind': kind,
        'hole_index': hole_index,
        'perimeter_px': scalar(length),
        'perimeter': scalar(length * scale),
        'area_px': scalar(area_px),
        'area': scalar(area_px * scale ** 2),
        'equivalent_radius_px': scalar(equivalent_radius_px),
        'equivalent_radius': scalar(equivalent_radius_px * scale),
        'point_count': n,
        'sample_step_px': scalar(ds),
        'arc_px': [scalar(value) for value in np.arange(n) * ds],
        'arc_fraction': [scalar(value) for value in np.arange(n) / n],
        'x_px': [scalar(value) for value in q[:, 0]],
        'y_px': [scalar(value) for value in q[:, 1]],
        'curvature_px': [scalar(value) for value in kappa_px],
        'curvature': [scalar(value / scale) if scale else None for value in kappa_px],
        'normalized_curvature': [scalar(value) for value in normalized],
        'features': features,
        'rotation_invariant': invariant,
        'comparable_profile': profile,
        'summary': {
            'corner_count': sum(1 for row in features if row['type'] == 'corner'),
            'notch_count': sum(1 for row in features if row['type'] == 'notch'),
            'mean_normalized_curvature': scalar(float(normalized.mean())),
            'normalized_curvature_std': scalar(float(normalized.std())),
            'maximum_normalized_curvature': scalar(float(normalized.max())),
            'minimum_normalized_curvature': scalar(float(normalized.min())),
        },
    }


def _invariant_signature(q, scale):
    descriptors = fourier_descriptors(q)
    magnitudes = descriptors[1:9]
    magnitudes = magnitudes / magnitudes[0] if magnitudes[0] else magnitudes
    return {
        'descriptor_count': len(magnitudes),
        'relative_magnitude': [scalar(value) for value in magnitudes],
        'scale': scalar(float(scale)),
        'note': '按弧长均匀重采样并归一化的傅里叶描述子相对幅度；平移和旋转不改变幅度，可比较不同位姿的同一形状',
    }


def _resampled_profile(q, kappa_px, normalized, length, sigma_px, bins=180):
    closed = np.vstack((q, q[0]))
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))))
    width = max(4.0 * sigma_px, 8.0)
    bins = int(np.clip(round(length / width), 24, 240))
    fraction = np.arange(bins) / bins
    target = fraction * length
    index = np.clip(np.searchsorted(cumulative, target, side='left'), 1, len(cumulative) - 1)
    values_px = []
    values_norm = []
    for arc, position in zip(target, index):
        start = cumulative[position - 1]
        span = cumulative[position] - start
        weight = 0.0 if span <= 0 else (arc - start) / span
        first = (position - 1) % len(q)
        second = position % len(q)
        values_px.append(kappa_px[first] + (kappa_px[second] - kappa_px[first]) * weight)
        values_norm.append(normalized[first] + (normalized[second] - normalized[first]) * weight)
    values_px = np.asarray(values_px, dtype=float)
    values_norm = np.asarray(values_norm, dtype=float)
    step = max(1, int(round(width / (length / len(q)))))
    if step > 1 and bins >= 8:
        kernel = np.ones(step) / step
        values_px = np.convolve(np.r_[values_px[-step:], values_px, values_px[:step]], kernel, mode='same')[step:-step]
        values_norm = np.convolve(np.r_[values_norm[-step:], values_norm, values_norm[:step]], kernel, mode='same')[step:-step]
    return {
        'bin_count': bins,
        'bin_width_px': scalar(length / bins),
        'arc_fraction': [scalar(value) for value in fraction],
        'curvature_px': [scalar(value) for value in values_px],
        'normalized_curvature': [scalar(value) for value in values_norm],
        'note': '按周长等比例分箱并按观察尺度疏化的曲率，便于比较同一形状在不同平移/旋转/缩放后的轮廓描述',
    }


def fourier_descriptors(q, count=64):
    length = _arc_length(q)
    closed = np.vstack((q, q[0]))
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))))
    target = np.arange(count) * length / count
    points = np.column_stack((np.interp(target, cumulative, closed[:, 0]),
                              np.interp(target, cumulative, closed[:, 1])))
    complex_points = (points[:, 0] - points[:, 0].mean()) + 1j * (points[:, 1] - points[:, 1].mean())
    spectrum = np.fft.fft(complex_points) / len(complex_points)
    return np.abs(spectrum[:count // 2 + 1])


def scale_stability(q, sigma_px, min_turn_deg):
    tiers = sorted({round(max(0.5, sigma_px * factor), 1) for factor in (0.5, 1.0, 2.0)})
    rows = []
    for value in tiers:
        features, _ = detect_features(q, value, min_turn_deg)
        rows.append({
            'sigma_px': scalar(value),
            'corner_count': sum(1 for row in features if row['type'] == 'corner'),
            'notch_count': sum(1 for row in features if row['type'] == 'notch'),
        })
    stable = len({(row['corner_count'], row['notch_count']) for row in rows}) == 1
    return {
        'tiers': rows,
        'stable': bool(stable),
        'note': '对外轮廓在观察尺度的0.5倍、1倍和2倍下重复计数；计数一致表示该轮廓的尖角/凹口对尺度不敏感',
    }


def particle_curvature(labels, particle_id, calibration, sigma_px=DEFAULT_SIGMA_PX,
                       min_turn_deg=DEFAULT_MIN_TURN_DEG, include_holes=True):
    mask = labels == particle_id
    if not np.any(mask):
        raise ValidationError('颗粒不存在')
    loops = contour_loops(mask)
    if not loops:
        raise ValidationError('颗粒没有可用边界')
    scored = []
    sampled = False
    for points in loops:
        q, _, _, limited = resample(points)
        sampled = sampled or limited
        q, _ = orient(q, mask)
        scored.append({'q': q, 'area': signed_area(q)})
    outer = [item for item in scored if item['area'] > 0]
    holes = [item for item in scored if item['area'] <= 0]
    if not outer:
        raise ValidationError('未能确定颗粒外轮廓')
    outer.sort(key=lambda item: abs(item['area']), reverse=True)
    holes.sort(key=lambda item: abs(item['area']), reverse=True)
    outer_report = loop_report(outer[0]['q'], 'outer', None, calibration, sigma_px, min_turn_deg)
    hole_reports = [loop_report(item['q'], 'hole', index, calibration, sigma_px, min_turn_deg)
                    for index, item in enumerate(holes)] if include_holes else []
    holes_skipped = 0 if include_holes else len(holes)
    stability = scale_stability(outer[0]['q'], sigma_px, min_turn_deg)
    return {
        'particle_id': int(particle_id),
        'sigma_px': scalar(float(sigma_px)),
        'sigma': scalar(float(sigma_px) * float(calibration['scale'])),
        'min_turn_deg': scalar(float(min_turn_deg)),
        'unit': calibration['unit'],
        'scale': scalar(float(calibration['scale'])),
        'include_holes': bool(include_holes),
        'outer_contour': outer_report,
        'holes': hole_reports,
        'skipped_hole_count': holes_skipped,
        'scale_stability': stability,
        'sampling_limited': sampled,
        'corner_count': outer_report['summary']['corner_count'] + sum(item['summary']['corner_count'] for item in hole_reports),
        'notch_count': outer_report['summary']['notch_count'] + sum(item['summary']['notch_count'] for item in hole_reports),
        'method': '按弧长重采样后平滑切向角得到曲率；角点与凹口由多尺度转角差、显著度以及转角随尺度是否继续放大共同判定',
        'convention': '沿轮廓行进时材料在左侧，外轮廓与孔洞都按此定向。deflection_deg 为正表示材料向外凸出的尖角，为负表示材料内凹的缺口；孔洞轮廓上恒为负的归一化曲率表示材料绕孔内凹，并不代表缺口',
        'warning': '曲率估计受观察尺度影响；尺度大于特征尺寸时细小尖角会被平滑掉，过小尺度仍可能放大掩膜像素台阶。圆滑曲线的渐进尖端（如细长椭圆端部）不会被当作尖角',
    }
