import math
import numpy as np
from scipy import ndimage as ndi
from .imaging import channel_array, region_mask
from .statistics import scalar
from .validation import ValidationError


# GLCM directions use image axes: x to the right, y downwards. The angle is
# measured from +x and taken modulo 180 degrees.
GLCM_DIRECTIONS = (
    (1, 0, 0),
    (1, 1, 45),
    (0, 1, 90),
    (-1, 1, 135),
)

GLCM_FEATURES = ('contrast', 'dissimilarity', 'homogeneity', 'asm', 'energy', 'entropy', 'correlation')

GAUSSIAN_TRUNCATE = 3.0


def _selected_region(image, region):
    selected = region_mask(image.size, region)
    pixels = int(selected.sum())
    if not pixels:
        raise ValidationError('分析区域为空')
    return selected, pixels


def _distance_to_outside(selected):
    """Chebyshev distance from each region pixel to the nearest non-region pixel.

    The image border is treated as outside the observed region, so stencils
    that would extend past the image are rejected for the same reason as
    stencils that extend past an ROI boundary.
    """
    padded = np.pad(selected, 1, mode='constant', constant_values=False)
    distance = ndi.distance_transform_cdt(padded, metric='chessboard')
    return distance[1:-1, 1:-1]


def _gaussian_radius(sigma):
    return int(GAUSSIAN_TRUNCATE * sigma + 0.5)


def _normalised_gaussian(values, selected, sigma):
    """Gaussian weighted mean of observed region pixels only.

    The filter of the region mask is the fraction of kernel mass that landed
    inside the region. At pixels where it is 1 the result equals ordinary
    Gaussian filtering of the region; elsewhere the estimate rests on fewer
    observations and is never used for reported statistics.
    """
    weight = ndi.gaussian_filter(
        selected.astype(float), sigma, mode='constant', truncate=GAUSSIAN_TRUNCATE
    )
    smooth = ndi.gaussian_filter(
        values, sigma, mode='constant', truncate=GAUSSIAN_TRUNCATE
    )
    return np.divide(smooth, weight, out=np.zeros_like(smooth), where=weight > 1e-12)


def _laplacian_value(values, smooth, sigma):
    """Laplacian-of-Gaussian of the region-only image.

    The function is only reported at pixels whose full kernel lies inside the
    region, so no pixel outside the region contributes. A truncated discrete
    LoG kernel does not sum to exactly zero; ``defect`` measures that residual
    and is removed so that a constant region yields a true zero response
    instead of a small truncation offset.
    """
    laplacian = ndi.gaussian_laplace(values, sigma, mode='constant', truncate=GAUSSIAN_TRUNCATE)
    defect = ndi.gaussian_laplace(
        np.ones_like(values), sigma, mode='constant', truncate=GAUSSIAN_TRUNCATE
    )
    return laplacian - smooth * defect


def glcm_matrix(gray, selected, levels, dx, dy, symmetric=True):
    quantized = np.minimum((gray.astype(float) * levels / 256).astype(int), levels - 1)
    height, width = gray.shape
    x0 = max(0, -dx)
    x1 = min(width, width - dx)
    y0 = max(0, -dy)
    y1 = min(height, height - dy)
    if x1 <= x0 or y1 <= y0:
        raise ValidationError('纹理偏移超过图像尺寸')
    first = selected[y0:y1, x0:x1]
    second = selected[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    # Both ends of a pair must be observed region pixels. A pair touching the
    # background outside the ROI is never counted.
    valid = first & second
    pairs = int(valid.sum())
    packed = (
        quantized[y0:y1, x0:x1][valid] * levels
        + quantized[y0 + dy:y1 + dy, x0 + dx:x1 + dx][valid]
    )
    matrix = np.bincount(packed, minlength=levels * levels).reshape(levels, levels).astype(float)
    if symmetric:
        matrix += matrix.T
    total = matrix.sum()
    if total:
        matrix /= total
    return matrix, pairs


def glcm_features(matrix):
    """Features of a normalised GLCM.

    An all-zero matrix means no valid pair was observed, which is not a real
    zero-contrast response. The caller must check ``pairs`` and treat this
    function's output as undefined in that case.
    """
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
    selected, area = _selected_region(image, region)
    directions = []
    for step_x, step_y, angle in GLCM_DIRECTIONS:
        dx, dy = step_x * distance, step_y * distance
        matrix, pairs = glcm_matrix(gray, selected, levels, dx, dy)
        if pairs:
            features = glcm_features(matrix)
            defined = True
        else:
            features = {name: None for name in GLCM_FEATURES}
            defined = False
        row = {
            'angle': angle,
            'dx': dx,
            'dy': dy,
            'pairs': pairs,
            'pair_coverage': scalar(pairs / area),
            'defined': defined,
        }
        row.update(features)
        directions.append(row)
    average = {}
    for name in GLCM_FEATURES:
        values = [row[name] for row in directions if row['defined'] and row[name] is not None]
        average[name] = scalar(np.mean(values)) if values else None
    defined_directions = sum(1 for row in directions if row['defined'])
    total_pairs = sum(row['pairs'] for row in directions)
    pair_coverage = total_pairs / (area * len(GLCM_DIRECTIONS))

    warnings = []
    if defined_directions == 0:
        warnings.append(
            '四个方向的偏移对都没有完整落在分析区域内；共生矩阵未定义，'
            '各项特征为 null，而不是真实的零对比度。'
        )
    else:
        missing = [f"{row['angle']}°" for row in directions if not row['defined']]
        if missing:
            warnings.append(
                '以下方向没有任何同时落在区域内的像素对，其结果为 null：'
                + '、'.join(missing) + '。窄条或凹形区域常出现这种情况，不要把它读成平滑纹理。'
            )
        if pair_coverage < 0.25:
            warnings.append(
                f'四个方向的像素对合计仅覆盖区域内像素的 {pair_coverage * 100:.1f}%；'
                '结果只由这些可观测像素对支撑。'
            )
    return {
        'levels': levels,
        'distance': distance,
        'directions': directions,
        'average': average,
        'defined': defined_directions > 0,
        'defined_directions': defined_directions,
        'region_pixels': area,
        'total_pairs': total_pairs,
        'pair_coverage': scalar(pair_coverage),
        'warnings': warnings,
        'method': (
            '对称归一化灰度共生矩阵；灰度按 floor(g*levels/256) 量化；'
            '仅统计偏移两端都落在分析区域内的像素对；角度按图像坐标自X轴量起、Y向下为正，0°与180°等价；'
            'average 是**可用方向**的算术平均，跳过无有效像素对的方向；'
            'correlation 在任一边际方差为0时为 null。'
        ),
    }


def local_binary_pattern(image, region):
    gray = channel_array(image)
    selected, area = _selected_region(image, region)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    distance = _distance_to_outside(selected)
    # A centre is only evaluated when its complete 8-neighbour ring is inside
    # the region. Comparing against background outside the ROI would invent
    # binary transitions at the region border.
    valid = selected & (distance > 1)
    centres = int(valid.sum())
    height, width = gray.shape
    if centres:
        padded = np.pad(gray, 1, mode='edge')
        center = padded[1:1 + height, 1:1 + width]
        codes = np.zeros((height, width), dtype=np.uint8)
        for bit, (dy, dx) in enumerate(offsets):
            neighbor = padded[1 + dy:1 + dy + height, 1 + dx:1 + dx + width]
            codes |= ((neighbor >= center).astype(np.uint8) << bit)
        counts = np.bincount(codes[valid], minlength=256).astype(np.int64)
        probabilities = counts / centres
        uniform = np.zeros(10)
        for code, probability in enumerate(probabilities):
            bits = [(code >> i) & 1 for i in range(8)]
            transitions = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
            index = sum(bits) if transitions <= 2 else 9
            uniform[index] += probability
    else:
        counts = None
        probabilities = None
        uniform = None
    coverage = centres / area
    warnings = []
    if centres == 0:
        warnings.append(
            '区域内没有任何像素拥有完整的8邻域；LBP 未定义，直方图为空、概率为 null，'
            '而不是全零响应。'
        )
    elif coverage < 0.5:
        warnings.append(
            f'仅 {coverage * 100:.1f}% 的区域内像素拥有完整8邻域；'
            '直方图只描述这些内部像素，不代表整块区域。'
        )
    return {
        'histogram': counts.tolist() if counts is not None else [],
        'probability': [scalar(v) for v in probabilities] if probabilities is not None else None,
        'rotation_invariant_uniform': [scalar(v) for v in uniform] if uniform is not None else None,
        'pixel_count': centres,
        'region_pixels': area,
        'valid_fraction': scalar(coverage),
        'defined': centres > 0,
        'warnings': warnings,
        'method': (
            '8邻域、半径1的局部二值模式；仅统计8个邻域全部落在分析区域内且不越出图像的像素。'
            '旋转不变均匀模式共10档：0–8对应转移数≤2且1的个数为该值，第9档为非均匀模式。'
        ),
    }


def gradient_report(image, region, bins=18):
    gray = channel_array(image)
    selected, area = _selected_region(image, region)
    gx = ndi.sobel(gray, axis=1, mode='constant') / 8
    gy = ndi.sobel(gray, axis=0, mode='constant') / 8
    magnitude = np.hypot(gx, gy)
    angle = np.degrees(np.arctan2(gy, gx)) % 180
    distance = _distance_to_outside(selected)
    # The 3x3 Sobel stencil includes diagonals, so all eight neighbours must
    # be observed. Pixels next to the ROI boundary or image border are not
    # evaluated; the background can therefore never become a gradient.
    evaluated = selected & (distance > 1)
    count = int(evaluated.sum())
    values = magnitude[evaluated]
    edges = np.linspace(0, 180, bins + 1)
    coverage = count / area
    weights = None
    if count:
        weights, _ = np.histogram(angle[evaluated], bins=edges, weights=values)
        total = weights.sum()
        if total:
            weights = weights / total
    warnings = []
    if count == 0:
        warnings.append(
            '区域内没有完整3×3梯度模板的像素；梯度未定义，均值、标准差和方向权重均为 null，'
            '而不是零响应。'
        )
    elif coverage < 0.5:
        warnings.append(
            f'仅 {coverage * 100:.1f}% 的区域内像素拥有完整3×3模板；'
            '其余像素靠近区域边界，其结果会受区域外背景影响，因此已排除。'
        )
    return {
        'orientation_edges': edges.tolist(),
        'orientation_weight': [scalar(v) for v in weights] if weights is not None else None,
        'mean_gradient': scalar(values.mean()) if count else None,
        'std_gradient': scalar(values.std()) if count else None,
        'edge_pixel_fraction': scalar(np.mean(values > 20)) if count else None,
        'gradient_pixels': count,
        'region_pixels': area,
        'valid_fraction': scalar(coverage),
        'defined': count > 0,
        'warnings': warnings,
        'method': (
            'Sobel 3×3 梯度（除以8），方向为 atan2(gy,gx) 取模180°，按幅值加权统计；'
            '仅统计完整模板落在分析区域内的像素；'
            'edge_pixel_fraction 为有效像素中梯度幅值大于20灰度级/像素的比例；'
            '方向权重在总幅值为0时保持全零（此时为真实零响应）。'
        ),
    }


def radial_power_spectrum(image, region, bins=32):
    gray = channel_array(image).astype(float)
    selected, area = _selected_region(image, region)
    if image.width * image.height > 4000000:
        raise ValidationError('频谱分析限400万像素，请先缩放或裁切')
    values = gray[selected]
    centered = np.where(selected, gray - values.mean(), 0)
    window = np.outer(np.hanning(image.height), np.hanning(image.width))
    # The window is multiplied by the region mask: no background pixel can
    # contribute, and the mean is also the region mean.
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centered * window * selected))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(image.height))
    fx = np.fft.fftshift(np.fft.fftfreq(image.width))
    radii = np.hypot(fy[:, None], fx[None, :])
    edges = np.linspace(0, 0.5, bins + 1)
    energy, _ = np.histogram(radii, bins=edges, weights=spectrum)
    counts, _ = np.histogram(radii, bins=edges)
    mean_power = np.divide(energy, counts, out=np.zeros(bins), where=counts > 0)
    populated = mean_power[1:]
    peak = None
    if len(populated) and populated.max() > 0:
        peak = int(np.argmax(populated) + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    return {
        'frequency_cycles_per_pixel': [scalar(v) for v in centers],
        'mean_power': [scalar(v) for v in mean_power],
        'peak_frequency': scalar(centers[peak]) if peak is not None else None,
        'peak_period_px': scalar(1 / centers[peak]) if peak is not None and centers[peak] else None,
        'region_pixels': area,
        'defined': True,
        'warnings': [
            '频谱在整幅图像尺寸上做零填充，区域形状会在低频产生泄漏；功率未按区域面积归一，'
            '区域大小不同的结果不能直接比较幅值。peak_frequency 为 null 表示区域内没有非零频率功率。'
        ],
        'method': (
            '先减去分析区域内部均值，再乘以限制在区域内的Hann窗和区域掩膜，'
            'Fourier功率按径向频率箱取均值；区域外像素恒为0，不参与计算。'
        ),
    }


def multiscale_contrast(image, region, sigmas=None):
    sigmas = sigmas or [1, 2, 4, 8, 16]
    gray = channel_array(image)
    selected, area = _selected_region(image, region)
    distance = _distance_to_outside(selected)
    values = np.where(selected, gray, 0)
    result = []
    for sigma in sigmas:
        radius = _gaussian_radius(sigma)
        # Only pixels whose entire 3σ Gaussian stencil lies inside the ROI are
        # reported. Elsewhere the local estimate would rest on too few
        # observations (or, with naive filtering, on background outside the
        # ROI), which can look like a smooth texture that was never measured.
        valid = selected & (distance > radius)
        count = int(valid.sum())
        entry = {
            'sigma': sigma,
            'kernel_radius_px': radius,
            'region_pixels': area,
            'evaluated_pixels': count,
            'valid_fraction': scalar(count / area),
            'residual_std': None,
            'residual_abs_mean': None,
            'laplacian_energy': None,
            'defined': count > 0,
        }
        if count:
            smooth = _normalised_gaussian(values, selected, sigma)
            residual = (gray - smooth)[valid]
            laplacian = _laplacian_value(values, smooth, sigma)
            entry['residual_std'] = scalar(residual.std())
            entry['residual_abs_mean'] = scalar(np.abs(residual).mean())
            entry['laplacian_energy'] = scalar(np.mean(laplacian[valid] ** 2))
        if count == 0:
            entry['note'] = '区域内没有完整3σ邻域的像素，本尺度未定义（null），不代表平滑纹理'
        elif entry['valid_fraction'] < 0.5:
            entry['note'] = f'仅{entry["valid_fraction"] * 100:.1f}%的区域内像素有完整3σ邻域，结果仅基于该子集'
        result.append(entry)
    return result


def texture_report(image, region, levels=32, distance=1):
    _, pixels = _selected_region(image, region)
    parts = {
        'glcm': glcm_report(image, region, levels, distance),
        'lbp': local_binary_pattern(image, region),
        'gradient': gradient_report(image, region),
        'multiscale': multiscale_contrast(image, region),
    }
    warnings = []
    for name, part in parts.items():
        if isinstance(part, list):
            continue
        for message in part.get('warnings', []):
            warnings.append(f'{name}: {message}')
    for entry in parts['multiscale']:
        if entry.get('note'):
            warnings.append(f'multiscale σ={entry["sigma"]}: {entry["note"]}')
    undefined = [
        name for name, part in parts.items()
        if not isinstance(part, list) and not part.get('defined', True)
    ]
    for entry in parts['multiscale']:
        if not entry.get('defined', True):
            undefined.append(f'multiscale σ={entry["sigma"]}')
    return {
        'region_pixels': pixels,
        **parts,
        'defined': not undefined,
        'undefined_parts': undefined,
        'warnings': warnings,
        'multiscale_method': (
            '各σ尺度用只含区域内像素的归一化高斯平滑计算残差与高斯拉普拉斯能量；'
            '仅统计核半径(3σ, 截断3.0)完全落在区域内的像素；'
            'Laplacian 核做了离散截断缺陷校正，使区域内常值得到真实0；'
            'evaluated_pixels 与 valid_fraction 给出该尺度的实际支撑，'
            'note 说明支撑不足时的含义，defined=false 表示该尺度未定义。'
        ),
        'isolation': (
            '所有邻域描述只在分析区域内部计算：区域外像素不参与配对、卷积或梯度模板；'
            '邻域不完整的像素被排除，其数量见各项的 evaluated_pixels / gradient_pixels / '
            'pixel_count 与 valid_fraction。无有效邻域时数值为 null 且 defined=false，'
            '与区域内真实存在的零响应（有有效像素、数值为0）严格区分。'
            'region_pixels 为分析区域内像素总数，作为各有效比例的共同分母。'
        ),
    }
