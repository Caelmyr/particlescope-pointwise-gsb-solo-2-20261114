import io
import math
import os
import colorsys
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy import fft as sfft
from scipy import ndimage as ndi
from .validation import ValidationError
from .measurements import included_rows


HEATMAP_TILE_PIXELS = 4_000_000
HEATMAP_BAND_PIXELS = 500_000
FFT_WORKERS = max(1, min(4, os.cpu_count() or 1))


PALETTE = [
    (73, 220, 185),
    (248, 180, 75),
    (94, 168, 244),
    (230, 111, 151),
    (178, 142, 236),
    (237, 214, 103),
    (81, 202, 219),
    (244, 139, 91),
]


def font(size=14):
    candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        'C:/Windows/Fonts/arial.ttf',
        '/System/Library/Fonts/Helvetica.ttc',
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def label_color(object_id):
    hue = (object_id * 0.61803398875) % 1
    rgb = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return tuple(int(v * 255) for v in rgb)


def label_image(labels, included=None):
    maximum = int(labels.max())
    lookup = np.zeros((maximum + 1, 3), dtype=np.uint8)
    for index in range(1, maximum + 1):
        if included is None or index in included:
            lookup[index] = label_color(index)
    return Image.fromarray(lookup[labels])


def draw_scale_bar(draw, image_size, calibration, length=None):
    scale = calibration['scale']
    target_pixels = image_size[0] / 5
    if length is None:
        raw = target_pixels * scale
        exponent = math.floor(math.log10(max(raw, 1e-12)))
        magnitude = 10 ** exponent
        normalized = raw / magnitude
        nice = 1 if normalized < 2 else 2 if normalized < 5 else 5
        length = nice * magnitude
    pixels = length / scale
    if pixels > image_size[0] * 0.8:
        raise ValidationError('比例尺长度超过图像宽度的80%')
    margin = max(12, image_size[0] // 50)
    x = margin
    y = image_size[1] - margin
    text_label = f'{length:g} {calibration["unit"]}'
    text_font = font(max(12, image_size[0] // 65))
    bounds = draw.textbbox((0, 0), text_label, font=text_font)
    text_height = bounds[3] - bounds[1]
    draw.rectangle((x - 6, y - text_height - 23, x + max(pixels, bounds[2]) + 8, y + 7), fill=(10, 20, 28, 210))
    draw.line((x, y, x + pixels, y), fill='white', width=max(3, image_size[0] // 250))
    draw.text((x, y - text_height - 17), text_label, fill='white', font=text_font)


def overlay_image(image, labels, result, style=None):
    style = style or {}
    included = {r['id'] for r in included_rows(result)}
    opacity = style.get('opacity', 0.25)
    base = np.asarray(image.convert('RGB'), dtype=float)
    colors = np.asarray(label_image(labels, included), dtype=float)
    selection = np.isin(labels, list(included)) & (labels > 0)
    if style.get('fill', True):
        base[selection] = base[selection] * (1 - opacity) + colors[selection] * opacity
    output = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert('RGBA')
    draw = ImageDraw.Draw(output)
    text_font = font(max(11, image.width // 100))
    if style.get('contours', True):
        boundary = selection & ~ndi.binary_erosion(selection)
        array = np.asarray(output).copy()
        array[boundary] = (65, 245, 198, 255)
        output = Image.fromarray(array)
        draw = ImageDraw.Draw(output)
    for row in result['particles']:
        if not row.get('included', True) and not style.get('show_excluded', False):
            continue
        color = label_color(row['id']) if row.get('included', True) else (130, 130, 130)
        x, y = row['bbox_x'], row['bbox_y']
        if style.get('boxes', True):
            draw.rectangle((x, y, x + row['bbox_width'] - 1, y + row['bbox_height'] - 1), outline=color, width=2)
        if style.get('ids', True):
            label = str(row['id'])
            position = (row['centroid_x'] + 3, row['centroid_y'] + 3)
            draw.text(position, label, font=text_font, fill='white', stroke_width=1, stroke_fill='black')
        if style.get('axes', False):
            angle = math.radians(row['orientation'])
            half = row['major_axis'] / result['scale'] / 2
            cx, cy = row['centroid_x'], row['centroid_y']
            dx, dy = half * math.cos(angle), half * math.sin(angle)
            draw.line((cx - dx, cy - dy, cx + dx, cy + dy), fill=(255, 211, 84), width=2)
    if style.get('scale_bar', True):
        draw_scale_bar(draw, image.size, result['calibration'])
    return output.convert('RGB')


def gaussian_1d(sigma):
    radius = int(4.0 * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1, dtype=float)
    weights = np.exp(-0.5 / sigma ** 2 * offsets ** 2)
    return weights / weights.sum(), radius


def reflected_indices(start, stop, length):
    index = np.mod(np.arange(start, stop), 2 * length)
    return np.where(index < length, index, 2 * length - 1 - index)


def folded_matrix(length, kernel):
    matrix = np.zeros((length, length), dtype=float)
    origin = np.arange(length)[:, None]
    offsets = np.arange(len(kernel)) - len(kernel) // 2
    source = np.mod(origin - offsets[None, :], 2 * length)
    source = np.where(source < length, source, 2 * length - 1 - source)
    np.add.at(matrix, (np.broadcast_to(origin, source.shape).ravel(), source.ravel()),
              np.broadcast_to(kernel, source.shape).ravel())
    return matrix


def prefers_matrix(length, radius):
    return length * length <= (length + 2 * radius) * math.log2(length + 2 * radius + 2)


def axis_convolve(block, axis, kernel, radius, matrix, spectra):
    if matrix is not None:
        if axis == 0:
            return np.tensordot(matrix, block, axes=(1, 0))
        return np.tensordot(block, matrix.T, axes=(1, 0))
    length = block.shape[axis]
    spectrum = spectra.get(length)
    if spectrum is None:
        padded = np.zeros(length, dtype=float)
        padded[:2 * radius + 1] = kernel
        spectrum = sfft.rfft(np.roll(padded, -radius), workers=FFT_WORKERS)
        spectra[length] = spectrum
    moved = np.moveaxis(block, axis, -1)
    filtered = sfft.irfft(sfft.rfft(moved, workers=FFT_WORKERS) * spectrum, n=length, workers=FFT_WORKERS)
    return np.moveaxis(filtered, -1, axis)


def tile_extent(height, width, pad_y, pad_x):
    budget = max(1, HEATMAP_TILE_PIXELS)
    side = max(1, int(math.sqrt(budget)))
    minimum = 32
    tile_y = height if pad_y == 0 else min(height, max(1, side - 2 * pad_y))
    tile_x = width if pad_x == 0 else min(width, max(1, side - 2 * pad_x))
    if pad_x and tile_x < min(width, minimum):
        tile_x = width
    if pad_y and tile_y < min(height, minimum):
        tile_y = height
    if pad_x and tile_x < width:
        tile_x = min(tile_x, max(1, budget // (tile_y + 2 * pad_y) - 2 * pad_x))
    if pad_y and tile_y < height:
        tile_y = min(tile_y, max(1, budget // (tile_x + 2 * pad_x) - 2 * pad_y))
    return max(1, tile_y), max(1, tile_x)


def blockwise_gaussian(py, px, fields, height, width, sigma):
    kernel, radius = gaussian_1d(sigma)
    matrix_y = folded_matrix(height, kernel) if prefers_matrix(height, radius) else None
    matrix_x = folded_matrix(width, kernel) if prefers_matrix(width, radius) else None
    pad_y = 0 if matrix_y is not None else radius
    pad_x = 0 if matrix_x is not None else radius
    tile_y, tile_x = tile_extent(height, width, pad_y, pad_x)
    outputs = [np.zeros((height, width), dtype=float) for _ in fields]
    spectra = {}
    for y0 in range(0, height, tile_y):
        y1 = min(height, y0 + tile_y)
        for x0 in range(0, width, tile_x):
            x1 = min(width, x0 + tile_x)
            sy0, sy1 = (max(0, y0 - pad_y), min(height, y1 + pad_y)) if pad_y else (0, height)
            sx0, sx1 = (max(0, x0 - pad_x), min(width, x1 + pad_x)) if pad_x else (0, width)
            inside = (py >= sy0) & (py < sy1) & (px >= sx0) & (px < sx1)
            local_y = py[inside] - sy0
            local_x = px[inside] - sx0
            if pad_y:
                rows = reflected_indices(y0 - radius, y1 + radius, height) - sy0
            else:
                rows = np.arange(sy1 - sy0)
            if pad_x:
                cols = reflected_indices(x0 - radius, x1 + radius, width) - sx0
            else:
                cols = np.arange(sx1 - sx0)
            for output, values in zip(outputs, fields):
                source = np.zeros((sy1 - sy0, sx1 - sx0), dtype=float)
                if np.any(inside):
                    np.add.at(source, (local_y, local_x), values[inside])
                extended = source[np.ix_(rows, cols)]
                extended = axis_convolve(extended, 0, kernel, radius, matrix_y, spectra)
                extended = axis_convolve(extended, 1, kernel, radius, matrix_x, spectra)
                top = radius if pad_y else y0
                left = radius if pad_x else x0
                output[y0:y1, x0:x1] = extended[top:top + (y1 - y0), left:left + (x1 - x0)]
    return outputs


def heatmap_image(image, rows, field='diameter', radius=20):
    height, width = image.height, image.width
    if not rows:
        return image.convert('RGB'), {'minimum': 0.0, 'maximum': 0.0, 'field': field}
    count = len(rows)
    px = np.empty(count, dtype=np.intp)
    py = np.empty(count, dtype=np.intp)
    values = np.empty(count, dtype=float)
    for index, row in enumerate(rows):
        x = int(round(row['centroid_x']))
        y = int(round(row['centroid_y']))
        px[index] = max(0, min(width - 1, x))
        py[index] = max(0, min(height - 1, y))
        values[index] = row[field]
    smooth_values, smooth_weights = blockwise_gaussian(
        py, px, [values, np.ones(count)], height, width, radius)
    np.divide(smooth_values, smooth_weights, out=smooth_values, where=smooth_weights > 1e-8)
    smooth_values[smooth_weights <= 1e-8] = 0
    active = smooth_weights > max(float(smooth_weights.max()) * 0.01, 1e-8)
    del smooth_weights
    normalized = smooth_values
    if np.any(active):
        lo, hi = np.percentile(normalized[active], [5, 95])
        normalized -= lo
        normalized /= max(hi - lo, 1e-9)
        np.clip(normalized, 0, 1, out=normalized)
    else:
        lo = hi = 0
    output = np.empty((height, width, 3), dtype=np.uint8)
    band_rows = max(1, HEATMAP_BAND_PIXELS // max(1, width))
    for y0 in range(0, height, band_rows):
        y1 = min(height, y0 + band_rows)
        band = normalized[y0:y1]
        tint = np.stack([
            band * 255,
            200 * (1 - np.abs(band * 2 - 1)),
            (1 - band) * 255,
        ], axis=2)
        strip = np.asarray(image.crop((0, y0, width, y1)).convert('RGB'), dtype=float)
        mask = active[y0:y1]
        strip[mask] = strip[mask] * 0.35 + tint[mask] * 0.65
        output[y0:y1] = strip.astype(np.uint8)
    return Image.fromarray(output), {'minimum': float(lo), 'maximum': float(hi), 'field': field}


def contact_sheet(image, labels, rows, columns=6, cell_size=130):
    from .measurements import particle_crop
    if len(rows) > 600:
        raise ValidationError('颗粒图册最多600个颗粒，请先筛选')
    count = max(len(rows), 1)
    row_count = math.ceil(count / columns)
    sheet = Image.new('RGB', (columns * cell_size, row_count * (cell_size + 28)), '#18232c')
    draw = ImageDraw.Draw(sheet)
    text_font = font(12)
    for index, row in enumerate(rows):
        crop = particle_crop(image, labels, row['id'], margin=5, isolate=False).convert('RGB')
        crop.thumbnail((cell_size - 10, cell_size - 10))
        x = index % columns * cell_size + (cell_size - crop.width) // 2
        y = index // columns * (cell_size + 28) + (cell_size - crop.height) // 2
        sheet.paste(crop, (x, y))
        draw.text((index % columns * cell_size + 7, index // columns * (cell_size + 28) + cell_size), f'#{row["id"]}  D={row["diameter"]:.2f}', font=text_font, fill='white')
    return sheet


def histogram_chart(distribution, title='Particle size distribution', size=(1000, 600)):
    image = Image.new('RGB', size, '#15212b')
    draw = ImageDraw.Draw(image)
    heading = font(22)
    small = font(13)
    draw.text((55, 25), title, font=heading, fill='white')
    left, top, right, bottom = 65, 90, size[0] - 45, size[1] - 75
    draw.line((left, top, left, bottom, right, bottom), fill='#8ba5b7', width=2)
    values = distribution.get('percent', [])
    if not values:
        draw.text((left + 30, top + 40), 'No particles', font=heading, fill='#8ba5b7')
        return image
    peak = max(max(values), 1)
    span = (right - left) / len(values)
    for i, value in enumerate(values):
        x = left + i * span
        y = bottom - value / peak * (bottom - top - 15)
        if value > 0:
            draw.rectangle((x + 2, min(y, bottom - 1), x + span - 2, bottom - 1), fill='#55d8b1')
        if len(values) <= 25:
            draw.text((x + 3, bottom + 12), f'{distribution["edges"][i]:.1f}', font=small, fill='#9db1bf')
    for i in range(6):
        y = bottom - (bottom - top - 15) * i / 5
        value = peak * i / 5
        draw.text((8, y - 7), f'{value:.1f}%', font=small, fill='#9db1bf')
        draw.line((left, y, right, y), fill='#2c3a45', width=1)
    return image


def draw_annotations(image, annotations, calibration):
    output = image.convert('RGBA')
    draw = ImageDraw.Draw(output)
    text_font = font(max(12, image.width // 85))
    for item in annotations:
        points = [tuple(p) for p in item['points']]
        color = item.get('color', '#ffcf66')
        kind = item['type']
        if kind == 'line':
            draw.line(points, fill=color, width=3)
            a, b = points
            length = math.dist(a, b) * calibration['scale']
            label = f'{length:.3g} {calibration["unit"]}'
        elif kind == 'angle':
            draw.line(points, fill=color, width=3)
            label = f'{item["measurement"]["angle"]:.2f} deg'
        elif kind == 'polygon':
            draw.polygon(points, outline=color, width=3)
            label = f'{item["measurement"]["area"]:.3g} {calibration["unit"]}2'
        else:
            x, y = points[0]
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
            label = item['label']
        if item.get('label') and kind != 'point':
            label = item['label'] + ' ' + label
        draw.text((points[0][0] + 6, points[0][1] + 6), label, font=text_font, fill=color, stroke_width=1, stroke_fill='black')
    return output.convert('RGB')


def png_bytes(image):
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return stream.getvalue()
