"""荧光双通道串色（spillover）校正。

用两张单染对照估计线性混合矩阵，再对混染样本做矩阵解混。
所有结果写入独立的 compensation 目录，不修改任何原始图像或工作图像。
"""
import math
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from .imaging import region_mask
from .statistics import scalar
from .validation import ValidationError


SATURATION_LEVEL = 250
SATURATION_WARN_PERCENT = 0.5
WEAK_SIGNAL_LEVEL = 25.0
WEAK_FOREGROUND_PERCENT = 0.5
MIN_R_SQUARED = 0.8
MAX_CONDITION_NUMBER = 15.0
NEGATIVE_FRACTION_WARN = 0.02
CHANNEL_NAMES = ('red', 'green')
CHANNEL_INDEX = {'red': 0, 'green': 1}


def two_channel_array(image):
    rgb = np.asarray(image.convert('RGB'), dtype=np.float64)
    return np.stack([rgb[:, :, 0], rgb[:, :, 1]], axis=2)


def background_levels(array, percentile=10.0):
    return np.array([
        float(np.percentile(array[:, :, 0], percentile)),
        float(np.percentile(array[:, :, 1], percentile)),
    ])


def saturated_fraction(channel):
    return float(np.mean(channel >= SATURATION_LEVEL) * 100.0)


def fit_single_stain(array, primary, background_percentile=10.0):
    """估计单染对照的主通道信号与向另一通道的溢出系数。

    返回溢出系数（主通道单位强度在另一通道产生的强度）、线性拟合优度和诊断信息。
    """
    primary_idx = CHANNEL_INDEX[primary]
    spill_idx = 1 - primary_idx
    primary_raw = array[:, :, primary_idx]
    spill_raw = array[:, :, spill_idx]
    backgrounds = background_levels(array, background_percentile)
    primary_bg = backgrounds[primary_idx]
    spill_bg = backgrounds[spill_idx]
    primary_net = np.maximum(primary_raw - primary_bg, 0.0)
    spill_net = np.maximum(spill_raw - spill_bg, 0.0)

    active = primary_net > max(3.0, 0.02 * float(primary_net.max() or 1.0))
    active_pixels = int(active.sum())
    total_pixels = int(primary_net.size)
    primary_max = float(primary_net.max())
    foreground_ratio = active_pixels / max(total_pixels, 1) * 100.0

    coefficient = 0.0
    r_squared = None
    inversion = False
    if active_pixels >= 20 and primary_max >= WEAK_SIGNAL_LEVEL:
        x = primary_net[active]
        y = spill_net[active]
        denominator = float(np.dot(x, x))
        if denominator > 1e-9:
            coefficient = float(np.dot(x, y) / denominator)
            predicted = coefficient * x
            residual = y - predicted
            ss_res = float(np.dot(residual, residual))
            ss_tot = float(np.dot(y - y.mean(), y - y.mean()))
            r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-9 else 1.0
        inversion = bool(np.corrcoef(x, y)[0, 1] < 0) if active_pixels >= 2 else False
        coefficient = max(0.0, min(coefficient, 0.95))

    warnings = []
    if primary_max < WEAK_SIGNAL_LEVEL or foreground_ratio < WEAK_FOREGROUND_PERCENT:
        warnings.append(
            f'{primary_name(primary)}单染信号过弱（主通道净信号峰值 {primary_max:.1f}，'
            f'有效像素 {foreground_ratio:.2f}%），对照可能退化，串色系数不可靠'
        )
    if r_squared is not None and r_squared < MIN_R_SQUARED:
        warnings.append(
            f'{primary_name(primary)}单染的线性拟合优度偏低（R²={r_squared:.3f}），'
            '通道间关系偏离线性，校正关系不可靠'
        )
    if inversion:
        warnings.append(
            f'{primary_name(primary)}单染的主通道与串色通道呈负相关，疑似选错对照或图像异常'
        )
    primary_saturated = saturated_fraction(primary_raw)
    spill_saturated = saturated_fraction(spill_raw)
    if primary_saturated > SATURATION_WARN_PERCENT:
        warnings.append(
            f'{primary_name(primary)}单染主通道 {primary_saturated:.2f}% 像素达到饱和，'
            '高强度区域的串色系数被截断，请勿用于定量'
        )

    return {
        'primary': primary,
        'coefficient': scalar(coefficient),
        'r_squared': scalar(r_squared) if r_squared is not None else None,
        'background': [scalar(backgrounds[0]), scalar(backgrounds[1])],
        'primary_peak_net': scalar(primary_max),
        'active_pixel_percent': scalar(foreground_ratio),
        'saturated_percent': {
            'red': scalar(saturated_fraction(array[:, :, 0])),
            'green': scalar(saturated_fraction(array[:, :, 1])),
        },
        'warnings': warnings,
    }


def primary_name(primary):
    return '红色染料' if primary == 'red' else '绿色染料'


def build_matrix(red_fit, green_fit):
    # observed = M @ true，行是观测通道（红、绿），列是真实染料（红、绿）
    matrix = np.array([
        [1.0, green_fit['coefficient']],
        [red_fit['coefficient'], 1.0],
    ], dtype=np.float64)
    determinant = float(np.linalg.det(matrix))
    try:
        condition = float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        condition = math.inf
    return matrix, determinant, condition


def calibrate(red_control, green_control, background_percentile=10.0):
    if red_control.size != green_control.size:
        raise ValidationError(
            f'两张单染对照尺寸不一致：红色对照 {red_control.size[0]}×{red_control.size[1]}，'
            f'绿色对照 {green_control.size[0]}×{green_control.size[1]}，无法逐像素建立校正关系'
        )
    red_array = two_channel_array(red_control)
    green_array = two_channel_array(green_control)
    red_fit = fit_single_stain(red_array, 'red', background_percentile)
    green_fit = fit_single_stain(green_array, 'green', background_percentile)
    matrix, determinant, condition = build_matrix(red_fit, green_fit)
    warnings = list(red_fit['warnings']) + list(green_fit['warnings'])
    reliable = not warnings and math.isfinite(condition) and condition <= MAX_CONDITION_NUMBER
    if math.isfinite(condition) and condition > MAX_CONDITION_NUMBER:
        warnings.append(
            f'混合矩阵条件数 {condition:.2f} 过大（阈值 {MAX_CONDITION_NUMBER:g}），'
            '两种染料光谱高度重叠，解混会放大噪声，结果无法可靠区分'
        )
    if not math.isfinite(condition) or abs(determinant) < 1e-6:
        warnings.append('混合矩阵接近奇异，无法稳定求逆，双阳性判定不可靠')
        reliable = False
    inverse = np.linalg.pinv(matrix) if abs(determinant) < 1e-12 else np.linalg.inv(matrix)
    return {
        'red_control': red_fit,
        'green_control': green_fit,
        'spillover_matrix': np.round(matrix, 6).tolist(),
        'inverse_matrix': np.round(inverse, 6).tolist(),
        'determinant': scalar(determinant),
        'condition_number': scalar(condition) if math.isfinite(condition) else None,
        'reliable': bool(reliable),
        'warnings': warnings,
        'size': {'width': red_control.size[0], 'height': red_control.size[1]},
    }


def unmix(mixed_image, calibration, region=None, negative_floor=True):
    size = (mixed_image.width, mixed_image.height)
    if tuple(calibration['size'].values()) != size:
        raise ValidationError(
            f"混染样本尺寸 {size[0]}×{size[1]} 与单染对照 "
            f"{calibration['size']['width']}×{calibration['size']['height']} 不一致，"
            '请使用同一视野、同一分辨率的图像'
        )
    array = two_channel_array(mixed_image)
    selected = region_mask(size, region or {'type': 'full'})
    red_bg, green_bg = _mixed_backgrounds(calibration, array, selected)
    net = np.stack([
        np.maximum(array[:, :, 0] - red_bg, 0.0),
        np.maximum(array[:, :, 1] - green_bg, 0.0),
    ], axis=2)
    inverse = np.asarray(calibration['inverse_matrix'], dtype=np.float64)
    corrected = net @ inverse.T
    # 背景像素受噪声影响普遍出现微小负值，属正常现象；
    # 用背景噪声尺度区分“噪声级波动”和“对照失配导致的显著负信号”
    foreground = selected & (np.max(net, axis=2) > WEAK_SIGNAL_LEVEL)
    background_pixels = selected & ~foreground
    noise = np.array([
        float(corrected[:, :, channel][background_pixels].std()) if background_pixels.any() else 0.0
        for channel in range(2)
    ])
    significant_negative = corrected < -3.0 * np.maximum(noise, 1e-9)[None, None, :]
    if foreground.any():
        negative_fraction = float(np.mean(np.any(significant_negative, axis=2)[foreground]))
    else:
        negative_fraction = 0.0
    clipped = np.maximum(corrected, 0.0) if negative_floor else corrected
    saturated = (array[:, :, 0] >= SATURATION_LEVEL) | (array[:, :, 1] >= SATURATION_LEVEL)
    saturated_fraction_roi = float(np.mean(saturated & selected) * 100.0)
    warnings = list(calibration.get('warnings', []))
    if saturated_fraction_roi > SATURATION_WARN_PERCENT:
        warnings.append(
            f'混染样本分析区域内 {saturated_fraction_roi:.2f}% 像素饱和，'
            '饱和像素的解混信号只能截断，无法恢复真实强度'
        )
    if negative_fraction > NEGATIVE_FRACTION_WARN:
        warnings.append(
            f'{negative_fraction * 100:.2f}% 前景像素解混后出现负信号（已置零），'
            '说明对照关系与样本不匹配或噪声偏大，这些区域无法可靠区分'
        )
    return {
        'observed': array,
        'net': net,
        'corrected': clipped,
        'raw_corrected': corrected,
        'selected': selected,
        'saturated': saturated,
        'backgrounds': [scalar(red_bg), scalar(green_bg)],
        'negative_fraction': scalar(negative_fraction),
        'saturated_percent': scalar(saturated_fraction_roi),
        'warnings': warnings,
    }


def _mixed_backgrounds(calibration, array, selected):
    # 用低分位背景与单染对照背景的较小值，避免把弱信号当背景减掉
    red_fit_bg = calibration['red_control']['background'][0]
    green_fit_bg = calibration['green_control']['background'][1]
    red_bg = min(float(np.percentile(array[:, :, 0][selected], 10)), red_fit_bg)
    green_bg = min(float(np.percentile(array[:, :, 1][selected], 10)), green_fit_bg)
    return red_bg, green_bg


def particle_signal_rows(labels, unmix_result, rows, positive_z=3.0, calibration_reliable=True):
    """按已有颗粒实例统计校正前后的红、绿信号强度与判定。"""
    corrected = unmix_result['corrected']
    observed = unmix_result['observed']
    selected = unmix_result['selected']
    saturated = unmix_result['saturated']
    # 用 ROI 内背景像素估计各真实通道的噪声尺度
    foreground = labels > 0
    background_pixels = selected & ~foreground
    noise = []
    for channel in range(2):
        values = corrected[:, :, channel][background_pixels]
        noise.append(float(values.std()) if values.size else 0.0)
    included_by_id = {row['id']: row.get('included', True) for row in rows}
    result_rows = []
    regions = ndi.find_objects(labels)
    for object_id, region_slice in enumerate(regions, 1):
        if region_slice is None:
            continue
        mask = labels[region_slice] == object_id
        if not np.any(mask):
            continue
        entry = {'id': int(object_id), 'included': included_by_id.get(object_id, True)}
        ambiguous = False
        reasons = []
        if not calibration_reliable:
            ambiguous = True
            reasons.append('校正关系未通过可靠性检查，分类结果不可直接采信')
        for channel, name in enumerate(CHANNEL_NAMES):
            obs_values = observed[:, :, channel][region_slice][mask]
            corrected_values = corrected[:, :, channel][region_slice][mask]
            entry[f'raw_{name}_mean'] = scalar(obs_values.mean())
            entry[f'raw_{name}_integrated'] = scalar(obs_values.sum())
            entry[f'{name}_mean'] = scalar(corrected_values.mean())
            entry[f'{name}_integrated'] = scalar(corrected_values.sum())
            entry[f'{name}_peak'] = scalar(corrected_values.max())
            threshold = max(WEAK_SIGNAL_LEVEL, positive_z * max(noise[channel], 1e-9))
            entry[f'{name}_positive'] = bool(corrected_values.mean() >= threshold)
            entry[f'{name}_threshold'] = scalar(threshold)
        particle_saturated = bool(np.any(saturated[region_slice][mask]))
        if particle_saturated:
            ambiguous = True
            reasons.append('颗粒内含饱和像素，强度被截断')
        red_positive = entry['red_positive']
        green_positive = entry['green_positive']
        if red_positive and green_positive:
            # 双阳性需两通道都明显高于噪声，否则标记为不可靠
            if min(entry['red_mean'], entry['green_mean']) < WEAK_SIGNAL_LEVEL:
                ambiguous = True
                reasons.append('双阳性中较弱通道接近检测限，无法确认共定位')
            entry['classification'] = 'double_positive'
        elif red_positive:
            entry['classification'] = 'red_only'
        elif green_positive:
            entry['classification'] = 'green_only'
        else:
            entry['classification'] = 'negative'
            ambiguous = True
            reasons.append('两通道校正后均低于阳性阈值')
        entry['ambiguous'] = ambiguous
        entry['ambiguous_reasons'] = reasons
        result_rows.append(entry)
    return result_rows, [scalar(n) for n in noise]


def summarize_rows(particle_rows):
    total = len(particle_rows)
    def count(label):
        return sum(1 for row in particle_rows if row['classification'] == label)
    ambiguous = sum(1 for row in particle_rows if row['ambiguous'])
    return {
        'particle_count': total,
        'red_only': count('red_only'),
        'green_only': count('green_only'),
        'double_positive': count('double_positive'),
        'negative': count('negative'),
        'ambiguous': ambiguous,
        'reliable_count': total - ambiguous,
    }


def separated_signal_image(unmix_result):
    """红、绿真实信号分通道渲染（黑底）。"""
    corrected = unmix_result['corrected']
    height, width = corrected.shape[:2]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    red = _scale_channel(corrected[:, :, 0])
    green = _scale_channel(corrected[:, :, 1])
    canvas[:, :, 0] = red
    canvas[:, :, 1] = green
    return Image.fromarray(canvas)


def classification_image(mixed_image, labels, particle_rows, unmix_result):
    """在混染图上叠加颗粒分类颜色与不可靠标记。"""
    base = np.asarray(mixed_image.convert('RGB'), dtype=np.float64) * 0.45
    overlay = np.zeros_like(base)
    class_color = {
        'red_only': (235, 92, 92),
        'green_only': (96, 220, 120),
        'double_positive': (230, 200, 90),
        'negative': (130, 140, 150),
    }
    row_by_id = {row['id']: row for row in particle_rows}
    for object_id in np.unique(labels):
        if object_id == 0:
            continue
        row = row_by_id.get(int(object_id))
        if row is None:
            continue
        color = class_color.get(row['classification'], (150, 150, 150))
        if row['ambiguous']:
            color = (170, 120, 230)
        overlay[labels == object_id] = color
    mask = labels > 0
    base[mask] = base[mask] * 0.3 + overlay[mask] * 0.7
    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for row in particle_rows:
        object_id = row['id']
        region = np.nonzero(labels == object_id)
        if not region[0].size:
            continue
        cy, cx = float(region[0].mean()), float(region[1].mean())
        if row['ambiguous']:
            radius = 7
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                         outline=(235, 160, 255), width=2)
            draw.line((cx - radius, cy, cx + radius, cy), fill=(235, 160, 255), width=2)
    return image


def _scale_channel(channel):
    maximum = float(np.percentile(channel, 99.5))
    if maximum <= 1e-9:
        return np.zeros(channel.shape, dtype=np.uint8)
    return np.clip(channel / maximum * 255.0, 0, 255).astype(np.uint8)


PARTICLE_CSV_FIELDS = [
    'id', 'included', 'classification', 'ambiguous', 'ambiguous_reasons',
    'raw_red_mean', 'raw_red_integrated', 'raw_green_mean', 'raw_green_integrated',
    'red_mean', 'red_integrated', 'red_peak', 'red_positive', 'red_threshold',
    'green_mean', 'green_integrated', 'green_peak', 'green_positive', 'green_threshold',
]


def particle_csv(particle_rows):
    import csv
    import io
    import json
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=PARTICLE_CSV_FIELDS, extrasaction='ignore')
    writer.writeheader()
    for row in particle_rows:
        safe = {}
        for key in PARTICLE_CSV_FIELDS:
            value = row.get(key)
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            safe[key] = value
        writer.writerow(safe)
    return ('﻿' + stream.getvalue()).encode('utf8')
