"""双视野拼接：原距离羽化与结构感知智能接缝。

两种方法共用同一套画布几何，完整保留负偏移、无重叠和两侧独有区域：
- feather：与历史版本完全一致的距离变换线性羽化；
- smart：在重叠区内搜索绕开明显结构的最小代价接缝，按接缝两侧平滑
  过渡，并先以稳定区域估计亮度增益，减少细小颗粒的双边缘与冲淡。
"""
import math
import time
import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
from .registration import translate
from .rendering import font
from .validation import ValidationError


MAX_CANVAS_PIXELS = 16000000
SEAM_GRID_LIMIT = 1000000
PREVIEW_CACHE_LIMIT = 12
_PREVIEW_CACHE = {}


def canvas_geometry(first, second, dx, dy):
    left = math.floor(min(0, dx))
    top = math.floor(min(0, dy))
    right = math.ceil(max(first.width, dx + second.width))
    bottom = math.ceil(max(first.height, dy + second.height))
    size = (right - left, bottom - top)
    if size[0] * size[1] > MAX_CANVAS_PIXELS:
        raise ValidationError('拼接画布超过1600万像素')
    return {
        'size': size,
        'left': left,
        'top': top,
        'origin': (-left, -top),
    }


def place_images(first, second, dx, dy):
    geometry = canvas_geometry(first, second, dx, dy)
    size = geometry['size']
    origin_x, origin_y = geometry['origin']
    image_a, valid_a = translate(first, origin_x, origin_y, size)
    image_b, valid_b = translate(second, dx + origin_x, dy + origin_y, size)
    overlap = valid_a & valid_b
    return {
        'geometry': geometry,
        'image_a': image_a,
        'image_b': image_b,
        'valid_a': valid_a,
        'valid_b': valid_b,
        'overlap': overlap,
    }


def _edge_field(gray, sigma):
    blurred = ndi.gaussian_filter(gray.astype(np.float32), sigma)
    grad_x = ndi.sobel(blurred, axis=1)
    grad_y = ndi.sobel(blurred, axis=0)
    magnitude = np.hypot(grad_x, grad_y)
    del grad_x, grad_y, blurred
    denominator = max(float(np.percentile(magnitude, 99)), 1e-6)
    return (magnitude / denominator).astype(np.float32)


def structure_cost(gray_a, gray_b, overlap, strength=1.0):
    """重叠区接缝代价：两侧结构越强、灰度越不一致，代价越高。"""
    edge_a = _edge_field(gray_a, 1.2)
    edge_b = _edge_field(gray_b, 1.2)
    structure = np.maximum(edge_a, edge_b)
    del edge_a, edge_b
    disagreement = ndi.gaussian_filter(
        (np.abs(gray_a - gray_b) / 255.0).astype(np.float32), 2.0,
    )
    cost = ((1.0 + strength * structure) * (1.0 + 2.0 * disagreement)).astype(np.float32)
    del structure
    cost[~overlap] = 0.0
    scale = float(np.percentile(cost[overlap], 95)) if overlap.any() else 1.0
    cost = cost / max(scale, 1e-6)
    cost[~overlap] = 0.0
    return cost.astype(np.float32), disagreement


def _optimal_seam_grid(cost, vertical):
    """带连续性约束的最小代价路径（Viterbi 动态规划）。

    vertical=True 时输出纵向接缝：每个网格列 c 对应一行 r(c)，
    相邻列的行号变化不超过 2；horizontal 时行列互换。
    """
    if vertical:
        grid = cost
    else:
        grid = cost.T
    rows, cols = grid.shape
    previous = np.empty((cols - 1, rows), dtype=np.int32)
    accumulated = grid[:, 0].astype(np.float64).copy()
    for col in range(1, cols):
        shifted = []
        for delta in (-2, -1, 0, 1, 2):
            shifted.append(np.roll(accumulated, delta))
            if delta > 0:
                shifted[-1][:delta] = np.inf
            elif delta < 0:
                shifted[-1][delta:] = np.inf
        options = np.stack(shifted, axis=0)
        best = np.argmin(options, axis=0)
        previous[col - 1] = best - 2
        accumulated = grid[:, col] + np.min(options, axis=0)
    seam_rows = [int(np.argmin(accumulated))]
    for col in range(cols - 1, 0, -1):
        current_row = seam_rows[-1]
        seam_rows.append(int(np.clip(current_row + previous[col - 1, current_row], 0, rows - 1)))
    seam_rows.reverse()
    path = [(row, col) for col, row in enumerate(seam_rows)]
    if not vertical:
        path = [(col, row) for row, col in path]
    return path


def _downscaled_cost(cost, overlap):
    cells = int(overlap.sum())
    factor = math.sqrt(SEAM_GRID_LIMIT / cells)
    if factor >= 1.0:
        return cost, 1.0, 1.0
    bounded = np.where(overlap, cost, 0.0)
    image = Image.fromarray(np.clip(bounded * 255.0, 0, 255).astype(np.uint8))
    grid_w = max(2, int(round(image.width * factor)))
    grid_h = max(2, int(round(image.height * factor)))
    resized = np.asarray(image.resize((grid_w, grid_h), Image.BILINEAR), dtype=np.float64) / 255.0
    return resized, image.width / grid_w, image.height / grid_h


def find_seam(cost, overlap, bbox):
    x0, y0, x1, y1 = bbox
    full_h, full_w = y1 - y0, x1 - x0
    region = cost[y0:y1, x0:x1]
    region_overlap = overlap[y0:y1, x0:x1]
    grid_cost, scale_x, scale_y = _downscaled_cost(region, region_overlap)
    vertical = full_w <= full_h
    grid_path = _optimal_seam_grid(grid_cost, vertical)
    grid_rows = np.array([p[0] for p in grid_path], dtype=np.float64)
    grid_cols = np.array([p[1] for p in grid_path], dtype=np.float64)

    if vertical:
        # 纵向接缝：对每个完整列插值出接缝行，长度等于区域宽度
        full_cols = np.arange(full_w, dtype=np.float64)
        seam_rows = np.interp(
            full_cols,
            grid_cols * scale_x,
            grid_rows * scale_y,
        )
        seam_rows = np.clip(np.round(seam_rows), 0, full_h - 1).astype(int)
        if full_w >= 5:
            seam_rows = ndi.median_filter(seam_rows, size=5, mode='nearest')
            seam_rows = np.round(ndi.convolve1d(
                seam_rows, np.ones(5) / 5.0, mode='nearest',
            )).astype(int)
            seam_rows = np.clip(seam_rows, 0, full_h - 1)
        points = [(int(c), int(r)) for c, r in enumerate(seam_rows)]
    else:
        # 横向接缝：对每个完整行插值出接缝列，长度等于区域高度
        full_rows = np.arange(full_h, dtype=np.float64)
        seam_cols = np.interp(
            full_rows,
            grid_rows * scale_y,
            grid_cols * scale_x,
        )
        seam_cols = np.clip(np.round(seam_cols), 0, full_w - 1).astype(int)
        if full_h >= 5:
            seam_cols = ndi.median_filter(seam_cols, size=5, mode='nearest')
            seam_cols = np.round(ndi.convolve1d(
                seam_cols, np.ones(5) / 5.0, mode='nearest',
            )).astype(int)
            seam_cols = np.clip(seam_cols, 0, full_w - 1)
        points = [(int(c), int(r)) for r, c in enumerate(seam_cols)]
    return points, vertical


def smart_weights(placed, feather, structure_strength=1.0, band_scale=1.0):
    valid_a = placed['valid_a']
    valid_b = placed['valid_b']
    overlap = placed['overlap']
    image_a = np.asarray(placed['image_a'], dtype=np.float32)
    image_b = np.asarray(placed['image_b'], dtype=np.float32)
    height, width = overlap.shape
    wa = np.zeros((height, width), dtype=np.float32)
    wb = np.zeros_like(wa)
    wa[valid_a & ~overlap] = 1.0
    wb[valid_b & ~overlap] = 1.0
    metrics = {
        'overlap_pixels': int(overlap.sum()),
        'has_overlap': bool(overlap.any()),
        'seam_points': 0,
        'seam_length_px': 0,
        'flatness': 0.0,
        'band_width_px': float(feather),
        'structure_cost_on_seam': None,
        'structure_cost_overlap_mean': None,
    }
    if not overlap.any():
        return wa, wb, metrics, None
    gray_a = image_a.mean(axis=2).astype(np.float32)
    gray_b = image_b.mean(axis=2).astype(np.float32)
    cost, disagreement = structure_cost(gray_a, gray_b, overlap, structure_strength)
    rows = np.where(overlap.any(axis=1))[0]
    cols = np.where(overlap.any(axis=0))[0]
    bbox = (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
    points, vertical = find_seam(cost, overlap, bbox)
    x0, y0, x1, y1 = bbox
    region_h, region_w = y1 - y0, x1 - x0
    line = np.zeros((region_h, region_w), dtype=bool)
    for index, (px, py) in enumerate(points):
        line[np.clip(py, 0, region_h - 1), np.clip(px, 0, region_w - 1)] = True
        if index:
            prev_x, prev_y = points[index - 1]
            steps = max(abs(px - prev_x), abs(py - prev_y), 1)
            for step in range(1, steps):
                inter_x = int(round(prev_x + (px - prev_x) * step / steps))
                inter_y = int(round(prev_y + (py - prev_y) * step / steps))
                line[np.clip(inter_y, 0, region_h - 1), np.clip(inter_x, 0, region_w - 1)] = True
    distance_region = ndi.distance_transform_edt(~line)
    path_costs = np.array([cost[y0 + py, x0 + px] for px, py in points], dtype=np.float64)
    overlap_mean = float(cost[overlap].mean())
    flatness = float(np.clip(1.0 - path_costs.mean() / max(overlap_mean, 1e-6), 0.0, 1.0))
    band_width = float(np.clip(
        feather * band_scale * (0.7 + 1.6 * (1.0 - flatness)),
        feather * 0.5,
        feather * 4.0,
    ))
    if vertical:
        # points 逐列给出该列的接缝行；用接缝的平均列位置判断 A 位于左侧还是右侧
        seam_row_per_col = np.array([p[1] for p in points])
        seam_col_mean = float(np.mean([p[0] for p in points]))
        a_on_left = seam_col_mean < region_w / 2.0
        row_index = np.arange(region_h)[:, None]
        seam_row = seam_row_per_col[None, :]
        side_a = (row_index <= seam_row) if a_on_left else (row_index >= seam_row)
        a_side = 'left' if a_on_left else 'right'
    else:
        # points 逐行给出该行的接缝列；用接缝的平均行位置判断 A 位于顶部还是底部
        seam_col_per_row = np.array([p[0] for p in points])
        seam_row_mean = float(np.mean([p[1] for p in points]))
        a_on_top = seam_row_mean < region_h / 2.0
        column_index = np.arange(region_w)[None, :]
        seam_column = seam_col_per_row[:, None]
        side_a = (column_index <= seam_column) if a_on_top else (column_index >= seam_column)
        a_side = 'top' if a_on_top else 'bottom'
    transition = np.clip(distance_region / max(band_width, 1.0), 0.0, 1.0)
    smooth = transition * transition * (3.0 - 2.0 * transition)
    region_wa = np.where(side_a, 1.0 - 0.5 * smooth, 0.5 * smooth)
    wa[y0:y1, x0:x1] = region_wa
    wb[y0:y1, x0:x1] = 1.0 - region_wa
    metrics.update({
        'seam_points': len(points),
        'seam_length_px': int(line.sum()),
        'flatness': flatness,
        'band_width_px': band_width,
        'structure_cost_on_seam': float(path_costs.mean()),
        'structure_cost_overlap_mean': overlap_mean,
        'a_side': a_side,
        'seam_orientation': 'vertical' if vertical else 'horizontal',
        'bbox': [x0, y0, x1, y1],
        'seam_canvas_points': [[x0 + px, y0 + py] for px, py in points],
    })
    return wa, wb, metrics, {
        'cost': cost,
        'disagreement': disagreement,
        'distance_region': distance_region,
        'bbox': bbox,
        'points': points,
        'vertical': vertical,
        'band_width': band_width,
        'side_a': side_a,
    }


def legacy_weights(placed, feather):
    valid_a = placed['valid_a']
    valid_b = placed['valid_b']
    distance_a = ndi.distance_transform_edt(np.pad(valid_a, 1))[1:-1, 1:-1]
    distance_b = ndi.distance_transform_edt(np.pad(valid_b, 1))[1:-1, 1:-1]
    wa = np.minimum(distance_a, feather) * valid_a
    wb = np.minimum(distance_b, feather) * valid_b
    return wa, wb


def luminance_gain(image_a, image_b, overlap, seam_detail=None):
    if not overlap.any():
        return 1.0, 0
    gray_a = image_a.mean(axis=2)
    gray_b = image_b.mean(axis=2)
    low_structure = (
        (_edge_field(gray_a, 1.5) < 0.12)
        & (_edge_field(gray_b, 1.5) < 0.12)
    )
    stable = overlap & low_structure
    if seam_detail is not None:
        stable &= seam_detail['disagreement'] <= float(
            np.percentile(seam_detail['disagreement'][overlap], 60)
        )
    if stable.sum() < 100:
        stable = overlap & low_structure
    if stable.sum() < 100:
        stable = overlap
    gains = []
    for channel in range(3):
        values_a = image_a[:, :, channel][stable]
        values_b = image_b[:, :, channel][stable]
        ratio = float(values_a.mean() / max(values_b.mean(), 1e-6))
        if math.isfinite(ratio) and 0.2 < ratio < 5.0:
            gains.append(ratio)
    if not gains:
        return 1.0, int(stable.sum())
    gain = float(np.clip(np.median(gains), 0.5, 2.0))
    return gain, int(stable.sum())


def _compose(image_a, image_b, wa, wb, gain):
    corrected_b = np.clip(image_b * gain, 0, 255)
    total = wa + wb
    output = np.full(image_a.shape, 255.0, dtype=np.float32)
    active = total > 0
    numerator = image_a * wa[:, :, None] + corrected_b * wb[:, :, None]
    output[active] = (numerator[active] / total[active, None]).astype(np.float32)
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8)), corrected_b


def _quality_metrics(image_a, image_b, corrected_b, output, overlap, wa, wb):
    if not overlap.any():
        return {
            'luminance_mae_before': None,
            'luminance_mae_after_gain': None,
            'seam_band_gradient': None,
        }
    gray_a = image_a.mean(axis=2)
    gray_b = image_b.mean(axis=2)
    gray_corrected = corrected_b.mean(axis=2)
    transition = overlap & (wa > 0.02) & (wb > 0.02)
    result_gray = np.asarray(output, dtype=np.float32).mean(axis=2)
    gradient = np.hypot(ndi.sobel(result_gray, axis=0), ndi.sobel(result_gray, axis=1))
    return {
        'luminance_mae_before': float(np.abs(gray_a - gray_b)[overlap].mean()),
        'luminance_mae_after_gain': float(np.abs(gray_a - gray_corrected)[overlap].mean()),
        'seam_band_gradient': float(gradient[transition].mean()) if transition.any() else 0.0,
        'transition_pixels': int(transition.sum()),
    }


def blend(first, second, dx, dy, feather=30, method='smart',
          structure_strength=1.0, band_scale=1.0, harmonize=True):
    placed = place_images(first, second, dx, dy)
    image_a = np.asarray(placed['image_a'], dtype=np.float32)
    image_b = np.asarray(placed['image_b'], dtype=np.float32)
    overlap = placed['overlap']
    if method == 'feather':
        wa, wb = legacy_weights(placed, feather)
        seam_metrics = {
            'overlap_pixels': int(overlap.sum()),
            'has_overlap': bool(overlap.any()),
            'method': 'feather',
        }
        gain = 1.0
        corrected_b = image_b
        output_image, _ = _compose(image_a, image_b, wa, wb, gain)
        quality = _quality_metrics(image_a, image_b, corrected_b, output_image, overlap, wa, wb)
    elif method == 'smart':
        wa, wb, seam_metrics, detail = smart_weights(placed, feather, structure_strength, band_scale)
        if harmonize:
            gain, stable_pixels = luminance_gain(image_a, image_b, overlap, detail)
        else:
            gain, stable_pixels = 1.0, int(overlap.sum())
        output_image, corrected_b = _compose(image_a, image_b, wa, wb, gain)
        seam_metrics['method'] = 'smart'
        seam_metrics['luminance_gain'] = gain
        seam_metrics['gain_stable_pixels'] = stable_pixels
        quality = _quality_metrics(image_a, image_b, corrected_b, output_image, overlap, wa, wb)
    else:
        raise ValidationError('拼接方法应为 feather 或 smart')
    geometry = placed['geometry']
    info = {
        'method': method,
        'canvas_width': geometry['size'][0],
        'canvas_height': geometry['size'][1],
        'reference_origin': list(geometry['origin']),
        'feather_px': feather,
        'structure_strength': structure_strength,
        'band_scale': band_scale,
        'harmonize': harmonize,
        **seam_metrics,
        **quality,
    }
    return output_image, info, placed, wa, wb, detail if method == 'smart' else None


def _seam_map(placed, detail, feather):
    geometry = placed['geometry']
    width, height = geometry['size']
    canvas = Image.new('RGBA', (width, height), (245, 248, 247, 255))
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    valid_a = placed['valid_a']
    valid_b = placed['valid_b']
    draw.bitmap((0, 0), Image.fromarray((valid_a & ~placed['overlap']).astype(np.uint8) * 255), fill=(70, 140, 210, 60))
    draw.bitmap((0, 0), Image.fromarray((valid_b & ~placed['overlap']).astype(np.uint8) * 255), fill=(210, 150, 60, 60))
    if detail is None:
        canvas.alpha_composite(overlay)
        return canvas.convert('RGB')
    x0, y0, x1, y1 = detail['bbox']
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(40, 90, 160, 255), width=2)
    points = [(x0 + px, y0 + py) for px, py in detail['points']]
    band = detail['distance_region'] <= detail['band_width']
    band_full = np.zeros((height, width), dtype=np.uint8)
    band_full[y0:y1, x0:x1] = band.astype(np.uint8) * 255
    draw.bitmap((0, 0), Image.fromarray(band_full), fill=(150, 90, 210, 85))
    if detail['vertical']:
        legacy_x = (x0 + x1) // 2
        draw.line((legacy_x, y0, legacy_x, y1 - 1), fill=(255, 255, 255, 255), width=4)
        draw.line((legacy_x, y0, legacy_x, y1 - 1), fill=(235, 160, 20, 255), width=2)
    else:
        legacy_y = (y0 + y1) // 2
        draw.line((x0, legacy_y, x1 - 1, legacy_y), fill=(255, 255, 255, 255), width=4)
        draw.line((x0, legacy_y, x1 - 1, legacy_y), fill=(235, 160, 20, 255), width=2)
    draw.line(points, fill=(255, 255, 255, 255), width=5)
    draw.line(points, fill=(20, 30, 45, 255), width=3)
    draw.line(points, fill=(214, 45, 130, 255), width=2)
    canvas.alpha_composite(overlay)
    labeled = ImageDraw.Draw(canvas)
    text_font = font(max(12, min(18, width // 70)))
    legend = (
        'blue: A only   orange: B only   purple: transition band   '
        'orange line: legacy feather midline   pink line: structure-aware seam'
    )
    labeled.rectangle((0, 0, width, 26), fill=(255, 255, 255, 225))
    labeled.text((8, 6), legend, font=text_font, fill=(30, 45, 58, 255))
    return canvas.convert('RGB')


def _detail_crop(legacy_image, smart_image, detail, feather):
    if detail is None:
        tile = Image.new('RGB', (760, 200), (245, 248, 247))
        draw = ImageDraw.Draw(tile)
        draw.text((24, 80), 'No overlap: each exclusive region is copied directly, no seam is produced',
                  font=font(16), fill=(60, 75, 85))
        return tile
    x0, y0, x1, y1 = detail['bbox']
    points_x = [p[0] for p in detail['points']]
    points_y = [p[1] for p in detail['points']]
    center_x = x0 + int(np.median(points_x))
    center_y = y0 + int(np.median(points_y))
    window = int(np.clip(feather * 6, 200, min(legacy_image.width, legacy_image.height)))
    half_w = min(window // 2, center_x, legacy_image.width - 1 - center_x)
    half_h = min(window // 2, center_y, legacy_image.height - 1 - center_y)
    box = (center_x - half_w, center_y - half_h, center_x + half_w + 1, center_y + half_h + 1)
    if box[2] - box[0] < 24 or box[3] - box[1] < 24:
        box = (x0, y0, x1, y1)
    left = legacy_image.crop(box)
    right = smart_image.crop(box)
    divider = 10
    title_h = 30
    tile = Image.new('RGB', (left.width * 2 + divider, left.height + title_h), (245, 248, 247))
    tile.paste(left, (0, title_h))
    tile.paste(right, (left.width + divider, title_h))
    draw = ImageDraw.Draw(tile)
    title_font = font(14)
    draw.rectangle((0, 0, left.width, title_h), fill=(35, 52, 64))
    draw.rectangle((left.width + divider, 0, tile.width, title_h), fill=(96, 48, 84))
    left_title = 'Legacy feather'
    right_title = 'Smart seam'
    draw.text((10, 7), left_title, font=title_font, fill=(240, 245, 248))
    draw.text((left.width + divider + 10, 7), right_title, font=title_font, fill=(240, 245, 248))
    draw.line((left.width + divider // 2, title_h, left.width + divider // 2, tile.height), fill=(214, 58, 142), width=divider)
    return tile


def build_preview(first, second, dx, dy, feather=30, structure_strength=1.0, band_scale=1.0):
    legacy_image, legacy_info, placed, _, _, _ = blend(
        first, second, dx, dy, feather, method='feather',
    )
    smart_image, smart_info, _, _, _, detail = blend(
        first, second, dx, dy, feather, method='smart',
        structure_strength=structure_strength, band_scale=band_scale,
    )
    seam_map = _seam_map(placed, detail, feather)
    detail_crop = _detail_crop(legacy_image, smart_image, detail, feather)
    token = f'{int(time.time() * 1000):x}{np.random.randint(0, 1 << 24):06x}'
    entry = {
        'created': time.time(),
        'images': {
            'legacy': legacy_image,
            'smart': smart_image,
            'seam': seam_map,
            'detail': detail_crop,
        },
        'info': {'feather': legacy_info, 'smart': smart_info},
    }
    _PREVIEW_CACHE[token] = entry
    while len(_PREVIEW_CACHE) > PREVIEW_CACHE_LIMIT:
        oldest = min(_PREVIEW_CACHE, key=lambda key: _PREVIEW_CACHE[key]['created'])
        _PREVIEW_CACHE.pop(oldest)
    return token, entry['info']


def preview_image(token, view):
    entry = _PREVIEW_CACHE.get(token)
    if entry is None or view not in entry['images']:
        raise ValidationError('预览已过期或视图不存在，请重新生成对照预览')
    return entry['images'][view]
