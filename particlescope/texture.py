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


def radial_power_spectrum(image, region, bins=32):
    gray = channel_array(image).astype(float)
    selected = region_mask(image.size, region)
    if image.width * image.height > 4000000:
        raise ValidationError('频谱分析限400万像素，请先缩放或裁切')
    values = gray[selected]
    if not len(values):
        raise ValidationError('分析区域为空')
    centered = np.where(selected, gray - values.mean(), 0)
    window = np.outer(np.hanning(image.height), np.hanning(image.width))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centered * window))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(image.height))
    fx = np.fft.fftshift(np.fft.fftfreq(image.width))
    radii = np.hypot(fy[:, None], fx[None, :])
    edges = np.linspace(0, 0.5, bins + 1)
    energy, _ = np.histogram(radii, bins=edges, weights=spectrum)
    counts, _ = np.histogram(radii, bins=edges)
    mean_power = np.divide(energy, counts, out=np.zeros(bins), where=counts > 0)
    peak = int(np.argmax(mean_power[1:]) + 1) if bins > 1 else 0
    centers = (edges[:-1] + edges[1:]) / 2
    return {
        'frequency_cycles_per_pixel': [scalar(v) for v in centers],
        'mean_power': [scalar(v) for v in mean_power],
        'peak_frequency': scalar(centers[peak]),
        'peak_period_px': scalar(1 / centers[peak]) if centers[peak] else None,
        'method': '均值移除、Hann窗、径向功率谱',
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
