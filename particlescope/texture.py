import math
import numpy as np
from scipy import ndimage as ndi
from .imaging import channel_array, region_mask
from .statistics import scalar
from .validation import ValidationError


def glcm_matrix(gray, selected, levels, dx, dy, symmetric=True):
    quantized = np.minimum((gray.astype(float) * levels / 256).astype(int), levels - 1)
    height, width = gray.shape
    x0 = max(0, -dx)
    x1 = min(width, width - dx)
    y0 = max(0, -dy)
    y1 = min(height, height - dy)
    if x1 <= x0 or y1 <= y0:
        raise ValidationError('纹理偏移超过图像尺寸')
    first = quantized[y0:y1, x0:x1]
    second = quantized[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    valid = selected[y0:y1, x0:x1] & selected[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    packed = first[valid] * levels + second[valid]
    matrix = np.bincount(packed, minlength=levels * levels).reshape(levels, levels).astype(float)
    if symmetric:
        matrix += matrix.T
    total = matrix.sum()
    return matrix / total if total else matrix, int(valid.sum())


def glcm_features(matrix):
    size = matrix.shape[0]
    i, j = np.indices(matrix.shape)
    contrast = np.sum(matrix * (i - j) ** 2)
    dissimilarity = np.sum(matrix * np.abs(i - j))
    homogeneity = np.sum(matrix / (1 + (i - j) ** 2))
    asm = np.sum(matrix ** 2)
    row = matrix.sum(axis=1)
    column = matrix.sum(axis=0)
    mean_i = np.sum(np.arange(size) * row)
    mean_j = np.sum(np.arange(size) * column)
    variance_i = np.sum((np.arange(size) - mean_i) ** 2 * row)
    variance_j = np.sum((np.arange(size) - mean_j) ** 2 * column)
    denominator = math.sqrt(variance_i * variance_j)
    correlation = np.sum(matrix * (i - mean_i) * (j - mean_j)) / denominator if denominator else None
    positive = matrix[matrix > 0]
    entropy = -np.sum(positive * np.log2(positive))
    return {
        'contrast': scalar(contrast),
        'dissimilarity': scalar(dissimilarity),
        'homogeneity': scalar(homogeneity),
        'asm': scalar(asm),
        'energy': scalar(math.sqrt(asm)),
        'entropy': scalar(entropy),
        'correlation': scalar(correlation) if correlation is not None else None,
    }


def glcm_report(image, region, levels=32, distance=1):
    gray = channel_array(image)
    selected = region_mask(image.size, region)
    offsets = [(distance, 0), (distance, distance), (0, distance), (-distance, distance)]
    angles = [0, 45, 90, 135]
    directions = []
    for angle, (dx, dy) in zip(angles, offsets):
        matrix, pairs = glcm_matrix(gray, selected, levels, dx, dy)
        features = glcm_features(matrix)
        directions.append({'angle': angle, 'pairs': pairs, 'features': features})
    average = {}
    for name in directions[0]['features']:
        values = [d['features'][name] for d in directions if d['features'][name] is not None]
        average[name] = scalar(np.mean(values)) if values else None
    return {
        'levels': levels,
        'distance': distance,
        'directions': directions,
        'average': average,
        'method': '对称归一化灰度共生矩阵',
    }


def local_binary_pattern(image, region):
    gray = channel_array(image)
    selected = region_mask(image.size, region)
    center = gray[1:-1, 1:-1]
    valid = selected[1:-1, 1:-1].copy()
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    codes = np.zeros(center.shape, dtype=np.uint8)
    height, width = gray.shape
    for bit, (dy, dx) in enumerate(offsets):
        neighbor = gray[1 + dy:height - 1 + dy, 1 + dx:width - 1 + dx]
        codes |= ((neighbor >= center).astype(np.uint8) << bit)
        valid &= selected[1 + dy:height - 1 + dy, 1 + dx:width - 1 + dx]
    counts = np.bincount(codes[valid], minlength=256)
    probabilities = counts / max(int(counts.sum()), 1)
    uniform = np.zeros(10)
    for code, probability in enumerate(probabilities):
        bits = [(code >> i) & 1 for i in range(8)]
        transitions = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
        index = sum(bits) if transitions <= 2 else 9
        uniform[index] += probability
    return {
        'histogram': counts.tolist(),
        'probability': [scalar(v) for v in probabilities],
        'rotation_invariant_uniform': [scalar(v) for v in uniform],
        'pixel_count': int(counts.sum()),
        'method': '8邻域、半径1的局部二值模式',
    }


def gradient_report(image, region, bins=18):
    gray = channel_array(image)
    selected = region_mask(image.size, region)
    gx = ndi.sobel(gray, axis=1) / 8
    gy = ndi.sobel(gray, axis=0) / 8
    magnitude = np.hypot(gx, gy)
    angle = np.degrees(np.arctan2(gy, gx)) % 180
    values = magnitude[selected]
    weights, edges = np.histogram(angle[selected], bins=np.linspace(0, 180, bins + 1), weights=values)
    total = weights.sum()
    if total:
        weights /= total
    return {
        'orientation_edges': edges.tolist(),
        'orientation_weight': [scalar(v) for v in weights],
        'mean_gradient': scalar(values.mean()) if len(values) else 0,
        'std_gradient': scalar(values.std()) if len(values) else 0,
        'edge_pixel_fraction': scalar(np.mean(values > 20)) if len(values) else 0,
    }


# 峰判定阈值：显著性（峰功率/背景中位数）与能量集中度（峰环占非直流能量比例）
STABLE_PROMINENCE = 8.0
STABLE_CONCENTRATION = 0.05
TENTATIVE_PROMINENCE = 3.0
TENTATIVE_CONCENTRATION = 0.02
# 可信周期至少覆盖的条纹周期数；周期短于该像素数视为采样伪影而非材料结构
MIN_CYCLES = 3.0
MIN_RESOLVABLE_PERIOD_PX = 3.0


def radial_power_spectrum(image, region, bins=32):
    gray = channel_array(image).astype(float)
    selected = region_mask(image.size, region)
    if image.width * image.height > 4000000:
        raise ValidationError('频谱分析限400万像素，请先缩放或裁切')
    if not int(selected.sum()):
        raise ValidationError('分析区域为空')

    # 仅在观察窗（区域外接矩形）内做加窗 FFT，避免区域外像素和矩形边界制造伪周期
    rows, cols = np.nonzero(selected)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    mask = selected[r0:r1, c0:c1].astype(float)
    field = gray[r0:r1, c0:c1]
    win_h, win_w = mask.shape
    window = np.outer(np.hanning(win_h), np.hanning(win_w)) * mask
    window_energy = float(np.sum(window ** 2))
    region_mean = float(np.sum(field * mask) / mask.sum())
    region_values = field[mask > 0]
    contrast = float(region_values.std())

    spectrum = np.abs(np.fft.fftshift(np.fft.fft2((field - region_mean) * window))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(win_h))
    fx = np.fft.fftshift(np.fft.fftfreq(win_w))
    radii = np.hypot(fy[:, None], fx[None, :])
    edges = np.linspace(0, 0.5, bins + 1)
    energy, _ = np.histogram(radii, bins=edges, weights=spectrum)
    counts, _ = np.histogram(radii, bins=edges)
    # 归一化到加窗能量，使不同大小、形状观察窗的功率曲线仍可横向比较
    mean_power = np.divide(
        energy, counts, out=np.zeros(bins), where=counts > 0
    ) / max(window_energy, 1e-12)
    centers = (edges[:-1] + edges[1:]) / 2
    total_ac_energy = float(energy[1:].sum())

    analysis = _classify_spectral_peak(
        spectrum, radii, edges, centers, counts, mean_power, energy,
        total_ac_energy, window_energy, contrast, win_w, win_h,
    )

    result = {
        'frequency_cycles_per_pixel': [scalar(v) for v in centers],
        'mean_power': [scalar(v) for v in mean_power],
        'method': '观察窗内均值移除、Hann窗、径向功率谱与亚分箱频率细化',
        'window': {
            'width': int(win_w),
            'height': int(win_h),
            'pixel_count': int(mask.sum()),
            'contrast_std': scalar(contrast),
        },
        'resolution': {
            'min_resolvable_frequency': scalar(MIN_CYCLES / min(win_w, win_h)),
            'max_resolvable_frequency': scalar(1.0 / MIN_RESOLVABLE_PERIOD_PX),
            'min_observed_cycles': MIN_CYCLES,
            'note': '峰对应周期需在观察窗短边内至少覆盖3个周期，且周期不短于3像素',
        },
    }
    result.update(analysis)
    return result


def _classify_spectral_peak(spectrum, radii, edges, centers, counts, mean_power,
                            energy, total_ac_energy, window_energy, contrast,
                            win_w, win_h):
    span = min(win_w, win_h)
    f_min = MIN_CYCLES / span
    f_max = 1.0 / MIN_RESOLVABLE_PERIOD_PX

    def unconfirmed(reason, note):
        return {
            'peak_status': 'unconfirmed',
            'peak_label': '无法确认',
            'peak_frequency': None,
            'peak_period_px': None,
            'peak_prominence': None,
            'peak_energy_concentration': None,
            'observed_cycles': None,
            'unconfirmed_reason': reason,
            'interpretation': note,
        }

    # 纯色/零能量：直流以外没有任何结构信息
    if contrast < 1e-6 or total_ac_energy <= 1e-12 * window_energy:
        return unconfirmed(
            'uniform_region',
            '区域灰度近似恒定，不存在可测量的纹理周期；功率曲线仅反映数值噪声。',
        )

    valid_bins = [
        i for i in range(1, len(centers))
        if counts[i] > 0 and f_min <= centers[i] <= f_max
    ]
    if not valid_bins:
        return unconfirmed(
            'no_resolvable_frequency',
            '观察窗过短，所有非直流频率都低于可分辨极限，无法在该区域内确认周期。',
        )

    strongest = mean_power[valid_bins].max()
    strong_bins = [i for i in valid_bins if mean_power[i] >= 1e-4 * strongest]
    candidates = [
        i for i in strong_bins
        if mean_power[i] == mean_power[max(0, i - 1):i + 2].max()
    ]
    if not candidates:
        return unconfirmed(
            'no_resolved_peak',
            '可分辨频段内没有局部极大峰，功率分布平坦，缺少周期性证据。',
        )

    peak_bin = max(candidates, key=lambda i: mean_power[i])
    guarded = np.array([
        i for i in valid_bins if all(abs(i - c) > 1 for c in candidates)
    ])
    if guarded.size:
        floor = float(np.median(mean_power[guarded]))
    else:
        floor = float(np.median(mean_power[valid_bins]))
    prominence = float(mean_power[peak_bin] / max(floor, 1e-30))
    concentration = float(energy[peak_bin] / max(total_ac_energy, 1e-30))

    # 亚分箱频率：在峰环（±1 径向分箱）内按二维功率加权求半径，
    # 消除32分箱量化把16px周期读成14.2px之类的系统性偏差
    lo, hi = edges[max(0, peak_bin - 1)], edges[min(len(centers), peak_bin + 2)]
    ring = (radii >= lo) & (radii < hi) & (radii > 0)
    ring_power = spectrum[ring]
    ring_total = float(ring_power.sum())
    if ring_total <= 0:
        return unconfirmed('low_frequency_leakage', '峰环能量异常，无法细化频率。')
    peak_frequency = float(np.sum(ring_power * radii[ring]) / ring_total)
    if peak_frequency <= 0:
        return unconfirmed(
            'low_frequency_leakage',
            '能量集中在直流附近，属于平缓渐变或边界泄漏而非周期纹理。',
        )
    observed_cycles = float(span * peak_frequency)
    peak_period = 1.0 / peak_frequency

    if observed_cycles < MIN_CYCLES:
        return {
            'peak_status': 'unconfirmed',
            'peak_label': '无法确认',
            'peak_frequency': scalar(peak_frequency),
            'peak_period_px': scalar(peak_period),
            'peak_prominence': scalar(prominence),
            'peak_energy_concentration': scalar(concentration),
            'observed_cycles': scalar(observed_cycles),
            'unconfirmed_reason': 'insufficient_cycles',
            'interpretation': (
                '观察窗短边内仅覆盖约{:.1f}个周期（需至少{:.0f}个），'
                '该峰可能由窗口截断造成，已给出候选频率但不作为材料周期。'
            ).format(observed_cycles, MIN_CYCLES),
        }

    if prominence >= STABLE_PROMINENCE and concentration >= STABLE_CONCENTRATION:
        status, label = 'stable', '稳定峰'
        interpretation = (
            '峰显著高于背景且集中了足够的纹理能量，在观察窗内覆盖约{:.1f}个周期，'
            '可作为材料的主周期；更换大小或形状的有效区域时该频率应保持一致。'
        ).format(observed_cycles)
    elif prominence >= TENTATIVE_PROMINENCE and concentration >= TENTATIVE_CONCENTRATION:
        status, label = 'tentative', '弱候选峰'
        interpretation = (
            '存在可重复的频率抬升但峰功率或能量集中度有限，周期解释偏弱，'
            '建议扩大观察窗或与其他区域结果交叉确认。'
        )
    else:
        status, label = 'unconfirmed', '无法确认'
        unconfirmed_reason = 'weak_peak'
        interpretation = (
            '最强局部峰与背景差异不显著（纯噪声或边界泄漏也会产生此类峰），'
            '没有足够纹理证据支持材料周期。'
        )

    return {
        'peak_status': status,
        'peak_label': label,
        'peak_frequency': scalar(peak_frequency),
        'peak_period_px': scalar(peak_period),
        'peak_prominence': scalar(prominence),
        'peak_energy_concentration': scalar(concentration),
        'observed_cycles': scalar(observed_cycles),
        'unconfirmed_reason': unconfirmed_reason if status == 'unconfirmed' else None,
        'interpretation': interpretation,
    }


def multiscale_contrast(image, region, sigmas=None):
    sigmas = sigmas or [1, 2, 4, 8, 16]
    gray = channel_array(image)
    selected = region_mask(image.size, region)
    result = []
    for sigma in sigmas:
        smooth = ndi.gaussian_filter(gray, sigma)
        residual = gray - smooth
        values = residual[selected]
        result.append({
            'sigma': sigma,
            'residual_std': scalar(values.std()) if len(values) else 0,
            'residual_abs_mean': scalar(np.abs(values).mean()) if len(values) else 0,
            'laplacian_energy': scalar(np.mean(ndi.gaussian_laplace(gray, sigma)[selected] ** 2)),
        })
    return result


def texture_report(image, region, levels=32, distance=1):
    return {
        'glcm': glcm_report(image, region, levels, distance),
        'lbp': local_binary_pattern(image, region),
        'gradient': gradient_report(image, region),
        'multiscale': multiscale_contrast(image, region),
    }
