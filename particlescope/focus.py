import io
import colorsys
import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
from . import validation as v
from .imaging import channel_array
from .rendering import png_bytes, font
from .statistics import scalar

MAX_PLANES = 24
MAX_PLANE_PIXELS = 4000000
# 焦面清晰度差异的对数扩散阈值：小于SPREAD_UNIFORM视为各焦面一致（等权平均），
# 大于SPREAD_DECISIVE视为存在真实景深差异（按清晰度权重融合），之间线性过渡。
SPREAD_UNIFORM = 1.0
SPREAD_DECISIVE = 2.0


def focus_parameters(data):
    ids = v.sequence(data.get('sample_ids'), '焦面样本列表', MAX_PLANES)
    if len(ids) < 2:
        raise v.ValidationError('焦面融合至少需要两个样本')
    ids = [v.identifier(item, '焦面样本编号') for item in ids]
    if len(set(ids)) != len(ids):
        raise v.ValidationError('焦面样本列表中存在重复样本')
    return {
        'sample_ids': ids,
        'smooth_sigma': v.real(data.get('smooth_sigma', 2), '清晰度评价平滑σ', 0.5, 10),
        'selectivity': v.real(data.get('selectivity', 8), '焦面选择锐度', 1, 32),
        'blend_sigma': v.real(data.get('blend_sigma', 1.5), '权重过渡平滑σ', 0, 10),
    }


def focus_measure(gray, smooth_sigma):
    laplace = ndi.laplace(gray, mode='reflect')
    energy = ndi.gaussian_filter(laplace * laplace, smooth_sigma, mode='reflect')
    noise = 1.4826 * float(np.median(np.abs(laplace)))
    floor = max(2.5 * noise * noise, 1e-6)
    return energy.astype(np.float32), floor


def fuse_stack(loader, count, size, smooth_sigma=2.0, selectivity=8.0, blend_sigma=1.5):
    width, height = size
    measures = np.empty((count, height, width), dtype=np.float32)
    floors = []
    mean_measures = []
    for index in range(count):
        measure, floor = focus_measure(channel_array(loader(index)), smooth_sigma)
        measures[index] = measure
        floors.append(floor)
        mean_measures.append(float(measure.mean()))
    noise_floor = max(floors)
    # 等权程度u：焦面间对数扩散小（结构一致）时等权平均；全部焦面低于噪声基准时也等权。
    # 但噪声基准本身可能被纹理污染（视野几乎无平坦区时MAD反映纹理而非噪声）：
    # 若过半像素低于基准且这些像素的焦面扩散大，说明基准不可信，改用纯相对判据。
    flat_absolute = measures.max(axis=0) < noise_floor
    logs = np.log(np.maximum(measures, 1e-12))
    spread = logs.max(axis=0) - logs.min(axis=0)
    uniform = np.clip((SPREAD_DECISIVE - spread) / (SPREAD_DECISIVE - SPREAD_UNIFORM), 0, 1)
    if flat_absolute.mean() < 0.5 or float(np.median(spread[flat_absolute])) < SPREAD_DECISIVE:
        uniform = np.maximum(uniform, flat_absolute.astype(np.float32))
    # 清晰度权重：对数域内按selectivity幂次竞争，数值稳定
    power_floor = max(0.02 * float(np.median(measures)), 1e-6)
    np.maximum(measures, power_floor, out=measures)
    np.log(measures, out=measures)
    measures -= measures.max(axis=0)
    measures *= selectivity
    np.exp(measures, out=measures)
    measures /= measures.sum(axis=0)
    measures *= (1 - uniform)
    measures += uniform / count
    total = measures.sum(axis=0)

    def smoothed(index):
        weight = measures[index] / total
        if blend_sigma > 0:
            weight = ndi.gaussian_filter(weight, blend_sigma, mode='reflect')
        return weight

    norm = np.zeros((height, width), dtype=np.float32)
    for index in range(count):
        norm += smoothed(index)
    norm = np.maximum(norm, 1e-12)
    fused = np.zeros((height, width, 3), dtype=np.float64)
    best_weight = np.zeros((height, width), dtype=np.float32)
    best_index = np.zeros((height, width), dtype=np.uint8)
    contributions = []
    for index in range(count):
        weight = smoothed(index) / norm
        rgb = np.asarray(loader(index).convert('RGB'), dtype=np.float32)
        fused += weight[:, :, None] * rgb
        better = weight > best_weight
        best_weight[better] = weight[better]
        best_index[better] = index
        contributions.append(float(weight.sum()) / weight.size * 100)
    confidence = np.clip((best_weight - 1.0 / count) / (1 - 1.0 / count), 0, 1)
    stats = {
        'floor': noise_floor,
        'mean_measures': mean_measures,
        'contributions': contributions,
        'flat_percent': float((uniform > 0.99).mean() * 100),
        'decisive_percent': float((confidence > 0.5).mean() * 100),
    }
    return fused, best_index, confidence.astype(np.float32), stats


def plane_colors(count):
    colors = []
    for index in range(count):
        red, green, blue = colorsys.hsv_to_rgb(index / count, 0.75, 1.0)
        colors.append((int(red * 255), int(green * 255), int(blue * 255)))
    return colors


def depth_map_image(indices, confidence, contributions):
    count = len(contributions)
    colors = plane_colors(count)
    height, width = indices.shape
    lookup = np.asarray(colors, dtype=np.float32)
    brightness = 0.45 + 0.55 * confidence
    mapped = lookup[indices] * brightness[:, :, None]
    base = Image.fromarray(np.clip(mapped, 0, 255).astype(np.uint8))
    legend_width = 150
    row = max(18, min(30, (height - 20) // count))
    canvas = Image.new('RGB', (width + legend_width, max(height, 26 + count * row)), (23, 29, 35))
    canvas.paste(base, (0, 0))
    draw = ImageDraw.Draw(canvas)
    text_font = font(13)
    for index, color in enumerate(colors):
        top = 8 + index * row
        chip = min(16, row - 4)
        draw.rectangle((width + 8, top, width + 8 + chip, top + chip), fill=color)
        draw.text((width + 30, top - 2), f'{index + 1} · {contributions[index]:.0f}%', fill=(225, 232, 238), font=text_font)
    return canvas


def create_focus_sample(store, service, params):
    with store.lock:
        ids = params['sample_ids']
        metas = [store.meta(sample_id) for sample_id in ids]
        sizes = {}
        for meta in metas:
            sizes.setdefault((meta['width'], meta['height']), []).append(meta['name'])
        if len(sizes) > 1:
            detail = '；'.join(f"{'、'.join(names)} 为 {size[0]}×{size[1]}" for size, names in sizes.items())
            raise v.ValidationError('焦面组内样本尺寸不一致，拒绝融合：' + detail)
        width, height = next(iter(sizes))
        if width * height > MAX_PLANE_PIXELS:
            raise v.ValidationError('焦面融合单张限400万像素，请先缩放')
        count = len(ids)
        fused, indices, confidence, stats = fuse_stack(
            lambda index: store.working_image(ids[index]),
            count,
            (width, height),
            params['smooth_sigma'],
            params['selectivity'],
            params['blend_sigma'],
        )
        image = Image.fromarray(np.clip(fused, 0, 255).astype(np.uint8))
        meta = store.import_image(io.BytesIO(png_bytes(image)), f"焦面融合 · {count}层 · {metas[0]['name'][:80]}")
        path = store.sample_dir(meta['id'])
        contributions = stats['contributions']
        store.image_write(path / 'focus-depth.png', depth_map_image(indices, confidence, contributions))
        np.save(path / 'focus-depth.npy', indices, allow_pickle=False)
        calibrations = [item['calibration'] for item in metas]
        inherited = all(item == calibrations[0] for item in calibrations)
        if inherited:
            service.set_calibration(meta['id'], calibrations[0])
        report = {
            'plane_count': count,
            'width': width,
            'height': height,
            'planes': [
                {
                    'index': index + 1,
                    'sample_id': ids[index],
                    'name': metas[index]['name'],
                    'contribution_percent': scalar(contributions[index]),
                    'mean_focus_measure': scalar(stats['mean_measures'][index]),
                }
                for index in range(count)
            ],
            'flat_averaged_percent': scalar(stats['flat_percent']),
            'decisive_percent': scalar(stats['decisive_percent']),
            'noise_floor': scalar(stats['floor']),
            'parameters': {
                'smooth_sigma': params['smooth_sigma'],
                'selectivity': params['selectivity'],
                'blend_sigma': params['blend_sigma'],
            },
            'calibration_inherited': inherited,
            'note': '各焦面清晰度一致或低于噪声基准的区域按等权平均，不凭空产生纹理；权重经空间平滑以减少景深交界光晕',
            'warning': '焦面融合是二维清晰度合成，不代表真实三维形貌；对融合图测量前请核对标定',
        }
        store.json_write(path / 'provenance.json', {
            'operation': 'focus_stack',
            'source_sample_ids': ids,
            'report': report,
        })
        return {'sample': store.meta(meta['id']), 'report': report}
