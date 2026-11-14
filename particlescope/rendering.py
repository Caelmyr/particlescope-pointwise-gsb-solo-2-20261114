import io
import math
import colorsys
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy import ndimage as ndi
from .validation import ValidationError
from .measurements import included_rows


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


def heatmap_image(image, rows, field='diameter', radius=20):
    values = np.zeros((image.height, image.width), dtype=float)
    weight = np.zeros_like(values)
    for row in rows:
        x = int(round(row['centroid_x']))
        y = int(round(row['centroid_y']))
        x = max(0, min(image.width - 1, x))
        y = max(0, min(image.height - 1, y))
        values[y, x] += row[field]
        weight[y, x] += 1
    smooth_values = ndi.gaussian_filter(values, radius)
    smooth_weights = ndi.gaussian_filter(weight, radius)
    average = np.divide(smooth_values, smooth_weights, out=np.zeros_like(values), where=smooth_weights > 1e-8)
    active = smooth_weights > max(float(smooth_weights.max()) * 0.01, 1e-8)
    if np.any(active):
        lo, hi = np.percentile(average[active], [5, 95])
        normalized = np.clip((average - lo) / max(hi - lo, 1e-9), 0, 1)
    else:
        normalized = average
        lo = hi = 0
    rgb = np.stack([
        normalized * 255,
        200 * (1 - np.abs(normalized * 2 - 1)),
        (1 - normalized) * 255,
    ], axis=2)
    base = np.asarray(image.convert('RGB'), dtype=float)
    base[active] = base[active] * 0.35 + rgb[active] * 0.65
    return Image.fromarray(base.astype(np.uint8)), {'minimum': float(lo), 'maximum': float(hi), 'field': field}


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


def skeleton_overlay(image, report, highlight_branch=None, highlight_node=None):
    """Draw the skeleton network with numbered nodes and branches."""
    output = image.convert('RGBA')
    draw = ImageDraw.Draw(output)
    scale_factor = max(image.width, image.height, 1)
    line_width = max(2, scale_factor // 320)
    node_radius = max(3, scale_factor // 180)
    text_font = font(max(10, scale_factor // 70))
    text_size = getattr(text_font, 'size', max(10, scale_factor // 70))
    branch_colors = [
        (86, 232, 196), (247, 189, 98), (116, 176, 250), (238, 132, 170),
        (188, 156, 244), (238, 220, 124), (105, 214, 232), (243, 156, 112),
    ]
    for branch in report.get('branches', []):
        points = [tuple(point) for point in branch.get('points') or []]
        if len(points) < 2:
            continue
        chosen = branch['id'] == highlight_branch
        color = (255, 236, 120) if chosen else branch_colors[branch['id'] % len(branch_colors)]
        draw.line(points, fill=color + (235,), width=line_width + (2 if chosen else 0), joint='curve')
        middle = points[len(points) // 2]
        draw.text(
            (middle[0] + 3, middle[1] - text_size - 2),
            f'B{branch["id"]}', font=text_font, fill=color, stroke_width=1, stroke_fill='black',
        )
    for node in report.get('junctions', []):
        chosen = node['id'] == highlight_node
        radius = node_radius + (3 if chosen else 0)
        x, y = node['x'], node['y']
        draw.rectangle(
            (x - radius, y - radius, x + radius, y + radius),
            outline=(255, 244, 170) if chosen else (255, 170, 64), width=max(2, line_width),
        )
        draw.text(
            (x + radius + 2, y - radius - text_size),
            f'J{node["id"]}', font=text_font, fill=(255, 208, 120), stroke_width=1, stroke_fill='black',
        )
    for node in report.get('endpoints', []):
        chosen = node['id'] == highlight_node
        radius = node_radius + (3 if chosen else 0)
        x, y = node['x'], node['y']
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(255, 214, 92) if chosen else (255, 120, 120), outline='black',
        )
        draw.text(
            (x + radius + 2, y - radius - text_size),
            f'E{node["id"]}', font=text_font, fill=(255, 158, 158), stroke_width=1, stroke_fill='black',
        )
    return output.convert('RGB')


def png_bytes(image):
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return stream.getvalue()
