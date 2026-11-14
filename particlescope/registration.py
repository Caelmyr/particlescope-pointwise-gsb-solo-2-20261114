import math
import io
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
from .imaging import channel_array
from .statistics import scalar
from .validation import ValidationError
from .rendering import png_bytes


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


def phase_correlation(reference, moving, subpixel=True):
    first, second = compatible_arrays(reference, moving)
    window = np.outer(np.hanning(first.shape[0]), np.hanning(first.shape[1]))
    first = (first - first.mean()) * window
    second = (second - second.mean()) * window
    fft_first = np.fft.fft2(first)
    fft_second = np.fft.fft2(second)
    cross = fft_first * np.conj(fft_second)
    normalized = cross / np.maximum(np.abs(cross), 1e-12)
    correlation = np.fft.ifft2(normalized).real
    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    height, width = correlation.shape
    dy = float(peak_y if peak_y <= height // 2 else peak_y - height)
    dx = float(peak_x if peak_x <= width // 2 else peak_x - width)
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
    from .mosaic import blend
    image, info, _, _, _, _ = blend(first, second, dx, dy, feather, method='feather')
    return image, {
        'canvas_width': info['canvas_width'],
        'canvas_height': info['canvas_height'],
        'reference_origin': info['reference_origin'],
        'overlap_pixels': info['overlap_pixels'],
        'feather_px': info['feather_px'],
    }


def registration_report(reference, moving, mode='auto', dx=0, dy=0):
    if mode == 'auto':
        transform = phase_correlation(reference, moving)
        dx, dy = transform['dx'], transform['dy']
    else:
        transform = {'dx': dx, 'dy': dy, 'method': 'manual'}
    aligned, valid = translate(moving, dx, dy, reference.size)
    before = similarity_metrics(reference, moving) if reference.size == moving.size else None
    after = similarity_metrics(reference, aligned, valid)
    return aligned, valid, {
        'transform': transform,
        'before': before,
        'after': after,
    }


def create_registered_sample(store, reference_id, moving_id, mode='auto', dx=0, dy=0):
    with store.lock:
        reference = store.working_image(reference_id)
        moving = store.working_image(moving_id)
        aligned, valid, report = registration_report(reference, moving, mode, dx, dy)
        meta = store.import_image(io.BytesIO(png_bytes(aligned)), '配准 · ' + store.meta(moving_id)['name'][:100])
        provenance = {
            'operation': 'registration',
            'reference_sample_id': reference_id,
            'moving_sample_id': moving_id,
            'report': report,
            'note': '新样本未继承标定，请按参考视野重新标定；白色区域为无数据填充',
        }
        store.json_write(store.sample_dir(meta['id']) / 'provenance.json', provenance)
        return {'sample': meta, 'report': report}
