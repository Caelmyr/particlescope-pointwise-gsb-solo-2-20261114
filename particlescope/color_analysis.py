import numpy as np
from scipy import ndimage as ndi
from .imaging import region_mask, channel_array
from .statistics import scalar, describe
from .validation import ValidationError


def rgb_to_lab(rgb):
    array = np.asarray(rgb, dtype=float) / 255
    linear = np.where(array <= 0.04045, array / 12.92, ((array + 0.055) / 1.055) ** 2.4)
    transform = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = linear @ transform.T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    delta = 6 / 29
    corrected = np.where(xyz > delta ** 3, np.cbrt(xyz), xyz / (3 * delta ** 2) + 4 / 29)
    l = 116 * corrected[..., 1] - 16
    a = 500 * (corrected[..., 0] - corrected[..., 1])
    b = 200 * (corrected[..., 1] - corrected[..., 2])
    return np.stack([l, a, b], axis=-1)


def color_summary(image, region):
    selected = region_mask(image.size, region)
    rgb = np.asarray(image.convert('RGB'))
    values = rgb[selected]
    if not len(values):
        raise ValidationError('分析区域为空')
    if len(values) > 500000:
        sample = values[::int(np.ceil(len(values) / 500000))]
    else:
        sample = values
    lab = rgb_to_lab(sample)
    mean_rgb = np.mean(values, axis=0)
    mean_lab = np.mean(lab, axis=0)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    return {
        'mean_rgb': [scalar(v) for v in mean_rgb],
        'median_rgb': [scalar(v) for v in np.median(values, axis=0)],
        'std_rgb': [scalar(v) for v in np.std(values, axis=0)],
        'mean_lab_d65': [scalar(v) for v in mean_lab],
        'chroma': describe(chroma),
        'pixel_count': len(values),
        'lab_sample_count': len(sample),
        'hex_mean': '#%02x%02x%02x' % tuple(np.rint(mean_rgb).astype(int)),
    }


def dominant_colors(image, region, clusters=4, maximum_samples=50000, seed=42):
    selected = region_mask(image.size, region)
    rgb = np.asarray(image.convert('RGB'))
    values = rgb[selected]
    if len(values) < clusters:
        raise ValidationError('区域像素不足以进行颜色聚类')
    rng = np.random.default_rng(seed)
    if len(values) > maximum_samples:
        values = values[rng.choice(len(values), maximum_samples, replace=False)]
    lab = rgb_to_lab(values)
    centers = [lab[int(rng.integers(len(lab)))]]
    for _ in range(1, clusters):
        distances = np.min(np.sum((lab[:, None, :] - np.asarray(centers)[None, :, :]) ** 2, axis=2), axis=1)
        total = float(distances.sum())
        if total <= 1e-12:
            centers.append(lab[int(rng.integers(len(lab)))])
        else:
            centers.append(lab[int(rng.choice(len(lab), p=distances / total))])
    centers = np.asarray(centers)
    labels = np.zeros(len(lab), dtype=int)
    iterations = 0
    for iteration in range(50):
        distances = np.sum((lab[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        updated_labels = np.argmin(distances, axis=1)
        updated_centers = centers.copy()
        for cluster in range(clusters):
            subset = lab[updated_labels == cluster]
            if len(subset):
                updated_centers[cluster] = subset.mean(axis=0)
        iterations = iteration + 1
        movement = float(np.max(np.linalg.norm(updated_centers - centers, axis=1)))
        centers = updated_centers
        labels = updated_labels
        if movement < 0.01:
            break
    colors = []
    for cluster in range(clusters):
        indices = labels == cluster
        count = int(indices.sum())
        if not count:
            continue
        mean_rgb = values[indices].mean(axis=0)
        colors.append({
            'rgb': [int(round(v)) for v in mean_rgb],
            'hex': '#%02x%02x%02x' % tuple(np.rint(mean_rgb).astype(int)),
            'lab': [scalar(v) for v in centers[cluster]],
            'percent': scalar(count / len(values) * 100),
            'sample_count': count,
        })
    return {
        'colors': sorted(colors, key=lambda c: c['percent'], reverse=True),
        'sample_count': len(values),
        'iterations': iterations,
        'seed': seed,
        'method': 'Lab D65空间K-means，随机种子固定，结果为采样估计',
    }


def color_distance(image, region, target_rgb, tolerance=20):
    selected = region_mask(image.size, region)
    rgb = np.asarray(image.convert('RGB'))
    lab = rgb_to_lab(rgb)
    target_lab = rgb_to_lab(np.asarray(target_rgb))
    delta = np.linalg.norm(lab - target_lab, axis=2)
    accepted = (delta <= tolerance) & selected
    labels, count = ndi.label(accepted)
    sizes = np.bincount(labels.ravel())[1:]
    return accepted, {
        'target_rgb': target_rgb,
        'target_lab': [scalar(v) for v in target_lab],
        'delta_e76_tolerance': tolerance,
        'matched_pixels': int(accepted.sum()),
        'matched_percent': scalar(accepted.sum() / max(int(selected.sum()), 1) * 100),
        'connected_regions': int(count),
        'largest_region_px': int(sizes.max()) if len(sizes) else 0,
        'delta_e_summary': describe(delta[selected]),
    }


def channel_correlations(image, region):
    selected = region_mask(image.size, region)
    rgb = np.asarray(image.convert('RGB'), dtype=float)
    values = rgb[selected]
    matrix = []
    for a in range(3):
        line = []
        for b in range(3):
            if np.std(values[:, a]) == 0 or np.std(values[:, b]) == 0:
                line.append(None)
            else:
                line.append(scalar(np.corrcoef(values[:, a], values[:, b])[0, 1]))
        matrix.append(line)
    return {'channels': ['red', 'green', 'blue'], 'pearson': matrix, 'pixel_count': len(values)}


def fluorescence_metrics(image, region, channel='green', background_percentile=10):
    selected = region_mask(image.size, region)
    values = channel_array(image, channel)[selected]
    if not len(values):
        raise ValidationError('分析区域为空')
    background = float(np.percentile(values, background_percentile))
    corrected = np.maximum(values - background, 0)
    noise = values[values <= np.percentile(values, 25)]
    noise_std = float(noise.std()) if len(noise) else 0
    return {
        'channel': channel,
        'background_percentile': background_percentile,
        'background_level': scalar(background),
        'raw_integrated_intensity': scalar(values.sum()),
        'corrected_integrated_intensity': scalar(corrected.sum()),
        'corrected_mean': scalar(corrected.mean()),
        'background_noise_std': scalar(noise_std),
        'estimated_snr': scalar(corrected.mean() / noise_std) if noise_std else None,
        'saturated_percent': scalar(np.mean(values >= 254) * 100),
        'warning': '背景为区域内低分位数估计；并非真实仪器噪声标定',
    }
