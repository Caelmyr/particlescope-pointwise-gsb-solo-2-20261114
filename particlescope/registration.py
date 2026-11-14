import math
import io
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
from .imaging import channel_array
from .statistics import scalar
from .validation import ValidationError
from .rendering import png_bytes


# 决策阈值：空间域归一化互相关低于该值视为没有可靠匹配证据
STRONG_NCC = 0.80
# 次优候选与最优候选的 NCC 差距不超过该值，且自身足够强 -> 位移存在歧义
AMBIGUITY_GAP = 0.10
# 真实几何重叠低于该比例时，无论相关多高都不能接受配准
MIN_OVERLAP_FRACTION = 0.15
# 相关性峰值非极大抑制的最小间距（像素）
PEAK_MIN_DISTANCE = 8
# 参与空间域复核的候选数量上限
MAX_CANDIDATES = 8
# 空间域评分时从每个填充边界向内收缩的保护带（像素），避免边缘结构与填充参与评分
EDGE_GUARD = 10


def compatible_arrays(reference, moving, maximum_pixels=4000000):
    if reference.size != moving.size:
        raise ValidationError('平移配准要求两个样本尺寸一致')
    if reference.width * reference.height > maximum_pixels:
        raise ValidationError('配准限400万像素，请先缩放')
    first = channel_array(reference).astype(float)
    second = channel_array(moving).astype(float)
    if np.std(first) < 1e-6 or np.std(second) < 1e-6:
        raise ValidationError('恒定图像没有可供配准的结构')
    return first, second


def parabolic_offset(left, center, right):
    denominator = left - 2 * center + right
    if abs(denominator) < 1e-12:
        return 0
    return float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))


def _wrapped_correlation(first, second):
    """加 Hann 窗的归一化相位相关面（循环位移假设，仅用于产生候选）。"""
    window = np.outer(np.hanning(first.shape[0]), np.hanning(first.shape[1]))
    fft_first = np.fft.fft2((first - first.mean()) * window)
    fft_second = np.fft.fft2((second - second.mean()) * window)
    cross = fft_first * np.conj(fft_second)
    normalized = cross / np.maximum(np.abs(cross), 1e-12)
    return np.fft.ifft2(normalized).real


def _signed_index(index, length):
    return index if index <= length // 2 else index - length


def correlation_candidates(reference, moving, max_candidates=MAX_CANDIDATES):
    """从相位相关面提取多个局部峰，而不是只取单一最大值。

    周期排列的微球会在相关面上产生一组重复峰；只取 argmax 会把“错开整排”
    的歧义位移当作唯一答案，因此这里保留全部显著候选供空间域复核。
    """
    first, second = compatible_arrays(reference, moving)
    correlation = _wrapped_correlation(first, second)
    height, width = correlation.shape
    maxed = ndi.maximum_filter(correlation, size=2 * PEAK_MIN_DISTANCE + 1)
    peak_mask = correlation == maxed
    values = correlation[peak_mask]
    positions = np.argwhere(peak_mask)
    order = np.argsort(values)[::-1][:max_candidates]
    candidates = []
    for item in order:
        peak_y, peak_x = positions[item]
        dy = float(_signed_index(peak_y, height))
        dx = float(_signed_index(peak_x, width))
        candidates.append({
            'dx': scalar(dx),
            'dy': scalar(dy),
            'correlation_peak': scalar(float(correlation[peak_y, peak_x])),
        })
    return candidates


# 自相关峰强度达到该比例才认定为晶格周期向量（普通随机纹理低于此值）
PERIOD_STRENGTH = 0.50


def period_vectors(reference, max_vectors=4):
    """从参考图自相关中提取晶格周期向量。

    完全周期化的图像在相位相关面上所有周期峰可能同高甚至被窗函数压平，
    导致 FFT 候选漏掉真实位移；自相关能稳定还原“整排错开”的周期向量，
    用它们补充候选假设，再统一交给空间域复核裁决。
    """
    first, _ = compatible_arrays(reference, reference)
    height, width = first.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    centered = (first - first.mean()) * window
    autocorr = np.fft.ifft2(np.abs(np.fft.fft2(centered)) ** 2).real
    origin = float(autocorr[0, 0])
    autocorr[0, 0] = 0
    min_period = max(12, int(round(min(height, width) * 0.03)))
    yy, xx = np.indices(autocorr.shape)
    distance2 = np.minimum(yy, height - yy) ** 2 + np.minimum(xx, width - xx) ** 2
    autocorr[distance2 < min_period ** 2] = 0
    maxed = ndi.maximum_filter(autocorr, size=2 * min_period + 1)
    mask = (autocorr == maxed) & (autocorr > 0)
    values = autocorr[mask]
    if not values.size:
        return []
    positions = np.argwhere(mask)
    order = np.argsort(values)[::-1][:max_vectors]
    vectors = []
    for item in order:
        peak_y, peak_x = positions[item]
        strength = float(autocorr[peak_y, peak_x] / origin)
        if strength < PERIOD_STRENGTH:
            continue
        dy = float(_signed_index(peak_y, height))
        dx = float(_signed_index(peak_x, width))
        vectors.append((dx, dy, strength))
    return vectors


def phase_correlation(reference, moving, subpixel=True):
    """保留原接口：返回最强峰的亚像素平移估计与峰旁瓣比。"""
    first, second = compatible_arrays(reference, moving)
    correlation = _wrapped_correlation(first, second)
    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    height, width = correlation.shape
    dy = float(_signed_index(peak_y, height))
    dx = float(_signed_index(peak_x, width))
    if subpixel:
        dy += parabolic_offset(
            correlation[(peak_y - 1) % height, peak_x],
            correlation[peak_y, peak_x],
            correlation[(peak_y + 1) % height, peak_x],
        )
        dx += parabolic_offset(
            correlation[peak_y, (peak_x - 1) % width],
            correlation[peak_y, peak_x],
            correlation[peak_y, (peak_x + 1) % width],
        )
    sidelobe = correlation.copy()
    for offset_y in range(-5, 6):
        for offset_x in range(-5, 6):
            sidelobe[(peak_y + offset_y) % height, (peak_x + offset_x) % width] = 0
    peak = float(correlation[peak_y, peak_x])
    noise = float(sidelobe.std())
    psr = (peak - float(sidelobe.mean())) / noise if noise else None
    return {
        'dx': scalar(dx),
        'dy': scalar(dy),
        'peak': scalar(peak),
        'peak_sidelobe_ratio': scalar(psr) if psr is not None else None,
        'subpixel': subpixel,
        'convention': '把移动样本向右dx、向下dy移动以对齐参考样本',
        'warning': '仅估计平移，不适用于旋转、尺度变化或重叠过小的视野',
    }


def overlap_fraction(size, dx, dy):
    """移动样本按 (dx, dy) 平移后与参考视野的真实几何重叠比例。"""
    width, height = size
    overlap_x = max(0, width - abs(int(round(dx))))
    overlap_y = max(0, height - abs(int(round(dy))))
    return overlap_x * overlap_y / float(width * height)


def _shift_with_mask(array, dx, dy):
    """整数平移灰度图，越界区域为 NaN，绝不使用白色等填充值参与评分。"""
    height, width = array.shape
    shifted = np.full(array.shape, np.nan, dtype=float)
    dst_y0, dst_y1 = max(0, dy), min(height, height + dy)
    dst_x0, dst_x1 = max(0, dx), min(width, width + dx)
    src_y0, src_x0 = max(0, -dy), max(0, -dx)
    shifted[dst_y0:dst_y1, dst_x0:dst_x1] = array[
        src_y0:src_y0 + (dst_y1 - dst_y0),
        src_x0:src_x0 + (dst_x1 - dst_x0),
    ]
    valid = ~np.isnan(shifted)
    return shifted, valid


def spatial_verification(reference, moving, dx, dy, guard=EDGE_GUARD):
    """在真实重叠区域内计算归一化互相关，白色填充区域一律排除。

    与相位相关的循环卷积不同，这里按真实（非循环）几何关系取交集，并从
    填充边界向内收缩保护带，使边缘结构和任何填充像素都无法抬高分数。
    """
    first = channel_array(reference).astype(float)
    second = channel_array(moving).astype(float)
    integer_dx, integer_dy = int(round(dx)), int(round(dy))
    shifted, valid = _shift_with_mask(second, integer_dx, integer_dy)
    if guard > 0:
        valid = ndi.binary_erosion(valid, iterations=guard)
        frame = np.zeros_like(valid)
        frame[guard:-guard, guard:-guard] = True
        valid &= frame
    fraction = float(valid.mean())
    if fraction < 0.05:
        return None, fraction
    a, b = first[valid], shifted[valid]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return None, fraction
    return float(np.corrcoef(a, b)[0, 1]), fraction


def assess_registration(reference, moving):
    """多候选位移评估，返回可靠性结论与全部候选证据。

    结论分三类：
    - reliable：单一强匹配，可直接采用；
    - ambiguous：多个位移的空间域证据接近（典型为周期排列微球错开整排），
      页面必须列出候选让用户人工核对；
    - insufficient：没有满足最小重叠与最低相关的候选（无重叠/无关图像），
      不允许另存为配准结果。
    """
    fft_candidates = correlation_candidates(reference, moving)
    fft_best = fft_candidates[0] if fft_candidates else {'dx': 0, 'dy': 0}
    # 假设集合：零位移、FFT 显著峰，以及“主峰/零位移 ± 晶格周期向量”。
    # 后者覆盖相位相关面被周期结构压平时漏掉的整排错开位移。
    hypotheses = [{'dx': 0, 'dy': 0, 'correlation_peak': None}]
    for candidate in fft_candidates:
        hypotheses.append({'dx': candidate['dx'], 'dy': candidate['dy'],
                           'correlation_peak': candidate['correlation_peak']})
    for base_dx, base_dy in ((0.0, 0.0), (fft_best['dx'], fft_best['dy'])):
        for period_dx, period_dy, _strength in period_vectors(reference):
            hypotheses.append({'dx': base_dx + period_dx, 'dy': base_dy + period_dy,
                               'correlation_peak': None})
    candidates = []
    seen = set()
    for hypothesis in hypotheses:
        dx, dy = hypothesis['dx'], hypothesis['dy']
        # 合并空间上几乎重合的候选
        key = (int(round(dx / 5)), int(round(dy / 5)))
        if key in seen:
            continue
        seen.add(key)
        ncc, fraction = spatial_verification(reference, moving, dx, dy)
        geometric = overlap_fraction(reference.size, dx, dy)
        candidates.append({
            'dx': scalar(dx),
            'dy': scalar(dy),
            'correlation_peak': hypothesis['correlation_peak'],
            'spatial_ncc': scalar(ncc) if ncc is not None else None,
            'scored_overlap_percent': scalar(fraction * 100),
            'geometric_overlap_percent': scalar(geometric * 100),
        })
    viable = [c for c in candidates if c['spatial_ncc'] is not None]
    viable.sort(key=lambda c: c['spatial_ncc'], reverse=True)
    if not viable:
        return {
            'status': 'insufficient',
            'candidates': candidates,
            'selected': None,
            'message': '所有候选位移都没有足够的真实重叠，无法配准；请勿依据白色填充区域判断匹配。',
        }
    best = viable[0]
    if best['geometric_overlap_percent'] < MIN_OVERLAP_FRACTION * 100:
        return {
            'status': 'insufficient',
            'candidates': candidates,
            'selected': best,
            'message': '最佳候选的真实重叠仅 %.1f%%，低于 %.0f%% 的最低要求，证据不足。'
                       % (best['geometric_overlap_percent'], MIN_OVERLAP_FRACTION * 100),
        }
    if best['spatial_ncc'] < STRONG_NCC:
        return {
            'status': 'insufficient',
            'candidates': candidates,
            'selected': best,
            'message': '最佳候选在真实重叠区内的相关系数仅 %.3f，低于 %.2f，两个视野可能并不同源。'
                       % (best['spatial_ncc'], STRONG_NCC),
        }
    runners = [
        c for c in viable[1:]
        if c['geometric_overlap_percent'] >= MIN_OVERLAP_FRACTION * 100
        and c['spatial_ncc'] >= STRONG_NCC
        and best['spatial_ncc'] - c['spatial_ncc'] <= AMBIGUITY_GAP
    ]
    if runners:
        return {
            'status': 'ambiguous',
            'candidates': candidates,
            'selected': best,
            'alternatives': runners,
            'message': '检测到 %d 个空间域证据接近的平移（常见于周期排列微球错开整排），'
                       '自动结果不唯一，请核对重叠后人工选择。' % (1 + len(runners)),
        }
    return {
        'status': 'reliable',
        'candidates': candidates,
        'selected': best,
        'message': '单一强匹配，最佳位移在真实重叠区内相关系数 %.3f。' % best['spatial_ncc'],
    }


def translate(image, dx, dy, output_size=None):
    output_size = output_size or image.size
    array = np.asarray(image.convert('RGB'), dtype=float)
    channels = []
    for channel in range(3):
        shifted = ndi.affine_transform(
            array[:, :, channel],
            matrix=np.eye(2),
            offset=[-dy, -dx],
            output_shape=(output_size[1], output_size[0]),
            order=1,
            mode='constant',
            cval=255,
        )
        channels.append(shifted)
    rgb = np.stack(channels, axis=2)
    valid = ndi.affine_transform(
        np.ones(array.shape[:2], dtype=np.uint8),
        matrix=np.eye(2),
        offset=[-dy, -dx],
        output_shape=(output_size[1], output_size[0]),
        order=0,
        mode='constant',
        cval=0,
    ).astype(bool)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)), valid


def no_data_overlay(image, valid):
    """把无数据（白色填充）区域渲染为醒目的斜纹警示，避免被误认为真实匹配。"""
    base = np.asarray(image.convert('RGB'), dtype=np.uint8).copy()
    missing = ~valid
    if not np.any(missing):
        return Image.fromarray(base)
    yy, xx = np.indices(valid.shape)
    hatch = ((xx + yy) // 12) % 2 == 0
    warning = np.array([240, 170, 60], dtype=np.uint8)
    dark = np.array([58, 42, 20], dtype=np.uint8)
    pattern = np.where(hatch[:, :, None], warning[None, None, :], dark[None, None, :])
    base[missing] = (0.45 * base[missing] + 0.55 * pattern[missing]).astype(np.uint8)
    return Image.fromarray(base)


def similarity_metrics(reference, moving, valid=None):
    if reference.size != moving.size:
        raise ValidationError('相似度比较要求尺寸一致')
    first = channel_array(reference).astype(float)
    second = channel_array(moving).astype(float)
    if valid is None:
        valid = np.ones(first.shape, dtype=bool)
    if not np.any(valid):
        raise ValidationError('两个样本没有有效重叠')
    a, b = first[valid], second[valid]
    difference = a - b
    mse = float(np.mean(difference ** 2))
    correlation = float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else None
    mean_a = ndi.uniform_filter(first, size=11)
    mean_b = ndi.uniform_filter(second, size=11)
    variance_a = np.maximum(ndi.uniform_filter(first ** 2, size=11) - mean_a ** 2, 0)
    variance_b = np.maximum(ndi.uniform_filter(second ** 2, size=11) - mean_b ** 2, 0)
    covariance = ndi.uniform_filter(first * second, size=11) - mean_a * mean_b
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    ssim = ((2 * mean_a * mean_b + c1) * (2 * covariance + c2)) / ((mean_a ** 2 + mean_b ** 2 + c1) * (variance_a + variance_b + c2))
    safe_valid = ndi.binary_erosion(valid, iterations=5)
    if not np.any(safe_valid):
        safe_valid = valid
    histogram, _, _ = np.histogram2d(a, b, bins=32, range=[[0, 256], [0, 256]])
    joint = histogram / histogram.sum()
    product = joint.sum(axis=1)[:, None] * joint.sum(axis=0)[None, :]
    nonzero = joint > 0
    mutual_information = np.sum(joint[nonzero] * np.log2(joint[nonzero] / product[nonzero]))
    return {
        'mae': scalar(np.abs(difference).mean()),
        'rmse': scalar(math.sqrt(mse)),
        'psnr_db': scalar(10 * math.log10(255 ** 2 / mse)) if mse else None,
        'identical': mse == 0,
        'pearson': scalar(correlation) if correlation is not None else None,
        'ssim_11px_uniform': scalar(ssim[safe_valid].mean()),
        'mutual_information_bits': scalar(mutual_information),
        'valid_pixels': int(valid.sum()),
        'valid_percent': scalar(valid.mean() * 100),
        'no_data_percent': scalar((~valid).mean() * 100),
    }


def affine_landmarks(source_points, target_points):
    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValidationError('两组配准点必须一一对应')
    if len(source) < 3:
        raise ValidationError('仿射配准至少需要三对非共线点')
    design = np.column_stack([source, np.ones(len(source))])
    coefficients, residuals, rank, singular = np.linalg.lstsq(design, target, rcond=None)
    if rank < 3:
        raise ValidationError('控制点共线，无法估计仿射变换')
    matrix = coefficients[:2].T
    offset = coefficients[2]
    if abs(np.linalg.det(matrix)) < 1e-8:
        raise ValidationError('仿射变换退化')
    predicted = design @ coefficients
    error = np.linalg.norm(predicted - target, axis=1)
    return {
        'matrix_xy': matrix.tolist(),
        'offset_xy': offset.tolist(),
        'point_errors': [scalar(v) for v in error],
        'rmse': scalar(np.sqrt(np.mean(error ** 2))),
        'determinant': scalar(np.linalg.det(matrix)),
        'point_count': len(source),
    }


def apply_affine(image, transform, output_size):
    forward = np.asarray(transform['matrix_xy'])
    translation = np.asarray(transform['offset_xy'])
    inverse = np.linalg.inv(forward)
    offset = -inverse @ translation
    swap = np.array([[0, 1], [1, 0]])
    matrix_yx = swap @ inverse @ swap
    offset_yx = swap @ offset
    array = np.asarray(image.convert('RGB'), dtype=float)
    transformed = []
    for channel in range(3):
        transformed.append(ndi.affine_transform(
            array[:, :, channel],
            matrix_yx,
            offset=offset_yx,
            output_shape=(output_size[1], output_size[0]),
            order=1,
            mode='constant',
            cval=255,
        ))
    return Image.fromarray(np.clip(np.stack(transformed, axis=2), 0, 255).astype(np.uint8))


def feather_mosaic(first, second, dx, dy, feather=30):
    left = math.floor(min(0, dx))
    top = math.floor(min(0, dy))
    right = math.ceil(max(first.width, dx + second.width))
    bottom = math.ceil(max(first.height, dy + second.height))
    size = (right - left, bottom - top)
    if size[0] * size[1] > 16000000:
        raise ValidationError('拼接画布超过1600万像素')
    a, valid_a = translate(first, -left, -top, size)
    b, valid_b = translate(second, dx - left, dy - top, size)
    weight_a = np.minimum(ndi.distance_transform_edt(np.pad(valid_a, 1))[1:-1, 1:-1], feather) * valid_a
    weight_b = np.minimum(ndi.distance_transform_edt(np.pad(valid_b, 1))[1:-1, 1:-1], feather) * valid_b
    total = weight_a + weight_b
    output = np.full((size[1], size[0], 3), 255, dtype=float)
    active = total > 0
    numerator = np.asarray(a, dtype=float) * weight_a[:, :, None] + np.asarray(b, dtype=float) * weight_b[:, :, None]
    output[active] = numerator[active] / total[active, None]
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8)), {
        'canvas_width': size[0],
        'canvas_height': size[1],
        'reference_origin': [-left, -top],
        'overlap_pixels': int(np.count_nonzero(valid_a & valid_b)),
        'feather_px': feather,
    }


def registration_report(reference, moving, mode='auto', dx=0, dy=0):
    if mode == 'auto':
        assessment = assess_registration(reference, moving)
        selected = assessment['selected']
        if selected is None:
            # 没有任何可用候选：保留最强 FFT 峰信息，但明确标记为不可用
            transform = phase_correlation(reference, moving)
            dx, dy = transform['dx'], transform['dy']
        else:
            # 可靠性裁决基于整数候选的空间域复核；实际平移仍采用亚像素精化
            transform = phase_correlation(reference, moving)
            if abs(transform['dx'] - selected['dx']) <= 1.5 and abs(transform['dy'] - selected['dy']) <= 1.5:
                dx, dy = transform['dx'], transform['dy']
            else:
                dx, dy = selected['dx'], selected['dy']
                transform['dx'], transform['dy'] = dx, dy
    else:
        fraction = overlap_fraction(reference.size, dx, dy)
        ncc, scored_fraction = spatial_verification(reference, moving, dx, dy)
        if fraction < MIN_OVERLAP_FRACTION:
            raise ValidationError(
                '手动偏移的真实重叠仅 %.1f%%，低于 %.0f%%，不能另存为配准结果；白色填充不会被当作匹配。'
                % (fraction * 100, MIN_OVERLAP_FRACTION * 100)
            )
        weak = ncc is None or ncc < STRONG_NCC
        assessment = {
            'status': 'manual',
            'candidates': [{
                'dx': scalar(dx), 'dy': scalar(dy), 'correlation_peak': None,
                'spatial_ncc': scalar(ncc) if ncc is not None else None,
                'scored_overlap_percent': scalar(scored_fraction * 100),
                'geometric_overlap_percent': scalar(fraction * 100),
            }],
            'selected': {
                'dx': scalar(dx), 'dy': scalar(dy),
                'spatial_ncc': scalar(ncc) if ncc is not None else None,
                'geometric_overlap_percent': scalar(fraction * 100),
            },
            'message': (
                '手动偏移在真实重叠区内相关系数仅 %s，匹配证据偏弱，建议核对后再另存。'
                % ('不可计算' if ncc is None else ('%.3f' % ncc))
            ) if weak else '手动指定偏移，真实重叠区相关系数 %.3f，配准质量良好。' % ncc,
            'weak_match': weak,
        }
        transform = {'dx': scalar(dx), 'dy': scalar(dy), 'method': 'manual'}
    aligned, valid = translate(moving, dx, dy, reference.size)
    before = similarity_metrics(reference, moving) if reference.size == moving.size else None
    after = similarity_metrics(reference, aligned, valid)
    report = {
        'transform': transform,
        'before': before,
        'after': after,
        'assessment': assessment,
        'overlap': {
            'geometric_overlap_percent': after['valid_percent'],
            'no_data_fill_percent': after['no_data_percent'],
            'minimum_required_percent': scalar(MIN_OVERLAP_FRACTION * 100),
        },
    }
    return aligned, valid, report


def create_registered_sample(store, reference_id, moving_id, mode='auto', dx=0, dy=0):
    with store.lock:
        reference = store.working_image(reference_id)
        moving = store.working_image(moving_id)
        aligned, valid, report = registration_report(reference, moving, mode, dx, dy)
        status = report['assessment']['status']
        if status == 'insufficient':
            raise ValidationError(
                '配准证据不足，已拒绝另存：' + report['assessment']['message']
                + ' 请调整视野或改用控制点仿射配准。'
            )
        if status == 'ambiguous':
            raise ValidationError(
                '存在多个合理平移、自动配准不唯一，已拒绝直接另存。请在页面选择候选偏移并核对真实重叠后，'
                '改用“手动偏移”模式另存。'
            )
        # 另存的样本在无数据区叠加警示斜纹，避免白色填充被误读为配准成功
        marked = no_data_overlay(aligned, valid)
        meta = store.import_image(io.BytesIO(png_bytes(marked)), '配准 · ' + store.meta(moving_id)['name'][:100])
        provenance = {
            'operation': 'registration',
            'reference_sample_id': reference_id,
            'moving_sample_id': moving_id,
            'report': report,
            'assessment_status': status,
            'note': '新样本未继承标定，请按参考视野重新标定；橙黑斜纹区域为无数据填充，不代表真实图像内容',
        }
        store.json_write(store.sample_dir(meta['id']) / 'provenance.json', provenance)
        return {'sample': meta, 'report': report}
