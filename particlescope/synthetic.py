import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from .validation import ValidationError


NAMES = {
    'isolated': '离散微球 · 合成演示',
    'touching': '粘连微球 · 合成演示',
    'illumination': '不均匀照明 · 合成演示',
    'fluorescence': '荧光颗粒 · 合成演示',
    'fibers': '取向纤维 · 合成演示',
    'porous': '多孔材料 · 合成演示',
}


def blank(size=(960, 640), background=(231, 235, 234)):
    return Image.new('RGB', size, background), Image.new('L', size, 0)


def isolated(rng):
    image, mask = blank()
    draw = ImageDraw.Draw(image)
    truth = ImageDraw.Draw(mask)
    objects = []
    for row in range(5):
        for column in range(8):
            x = 60 + column * 118 + int(rng.integers(-12, 13))
            y = 60 + row * 126 + int(rng.integers(-12, 13))
            radius = int(rng.integers(13, 35))
            box = (x - radius, y - radius, x + radius, y + radius)
            color = tuple(int(c) for c in rng.integers([28, 42, 51], [75, 89, 103]))
            draw.ellipse(box, fill=color)
            truth.ellipse(box, fill=255)
            highlight = (x - radius * 0.6, y - radius * 0.6, x + radius * 0.1, y + radius * 0.1)
            draw.ellipse(highlight, fill=(99, 113, 121))
            objects.append({'x': x, 'y': y, 'radius': radius})
    return image, mask, {'expected_objects': 40, 'objects': objects}


def touching(rng):
    image, mask = blank()
    draw = ImageDraw.Draw(image)
    truth = ImageDraw.Draw(mask)
    centers = []
    for row in range(3):
        for column in range(4):
            center_x = 105 + column * 235
            center_y = 105 + row * 210
            radius = int(rng.integers(30, 43))
            for shift in [-radius * 0.72, radius * 0.72]:
                x = int(center_x + shift)
                y = center_y + int(rng.integers(-7, 8))
                box = (x - radius, y - radius, x + radius, y + radius)
                draw.ellipse(box, fill=(48, 66, 77))
                truth.ellipse(box, fill=255)
                centers.append([x, y])
    return image, mask, {
        'expected_physical_disks': 24,
        'expected_connected_components': 12,
        'centers': centers,
        'note': '真实掩膜表示前景并集；实例分离数量需人工核对',
    }


def illumination_shading(shape):
    y, x = np.indices(shape)
    shading = 0.45 + 0.55 * x / shape[1]
    shading *= 0.85 + 0.15 * np.cos(y / shape[0] * math.pi)
    return shading


CHANNEL_RESPONSE = np.array([1.0, 0.86, 0.7])
DARK_OFFSET = np.array([9.0, 8.0, 11.0])


def illumination(rng):
    image, mask, details = isolated(rng)
    signal = np.asarray(image, dtype=float)
    shading = illumination_shading(signal.shape[:2])
    rgb = signal * shading[:, :, None] * CHANNEL_RESPONSE[None, None, :] + DARK_OFFSET[None, None, :]
    rgb += rng.normal(0, 2.0, signal.shape[:2])[:, :, None]
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)), mask, details


def flat_dark_references(rng, shape):
    """Same-size flat and dark fields with channel response and sensor defects."""
    shading = illumination_shading(shape)
    noise = rng.normal(0, 1.2, shape)
    flat = 236 * shading[:, :, None] * CHANNEL_RESPONSE[None, None, :] + DARK_OFFSET[None, None, :]
    flat += noise[:, :, None]
    flat[24, 30] = 255
    flat[shape[0] // 3:shape[0] // 3 + 14, shape[1] - 40:shape[1] - 26] = 255
    dark = np.full(shape + (3,), 10.0, dtype=float) + rng.normal(0, 0.6, shape + (3,))
    dark[40, 60] = [255, 255, 255]
    near_dead = np.zeros(shape, dtype=bool)
    near_dead[shape[0] - 90:shape[0] - 78, shape[1] - 120:shape[1] - 100] = True
    flat[near_dead] = dark[near_dead] + 0.6
    return (
        Image.fromarray(np.clip(flat, 0, 255).astype(np.uint8)),
        Image.fromarray(np.clip(dark, 0, 255).astype(np.uint8)),
    )


def generate_flatfield_references(seed=19):
    """Illumination demo plus same-size flat/dark references for the UI workflow."""
    rng = np.random.default_rng(seed)
    image, mask, details = illumination(rng)
    flat, dark = flat_dark_references(rng, (image.height, image.width))
    details.update({
        'name': '光照不均 · 平场参考演示',
        'kind': 'flatfield',
        'seed': seed,
        'synthetic': True,
        'note': '同尺寸明场与暗场参考含通道响应差异、坏点、饱和和响应过低区域',
    })
    return (
        image, mask, details,
        flat, dict(details, name='明场参考 · 合成演示', kind='flatfield-reference', reference='flat'),
        dark, dict(details, name='暗场参考 · 合成演示', kind='flatfield-reference', reference='dark'),
    )


def fluorescence(rng):
    image, mask = blank(background=(6, 9, 15))
    draw = ImageDraw.Draw(image)
    truth = ImageDraw.Draw(mask)
    for row in range(5):
        for column in range(8):
            x = 60 + column * 118 + int(rng.integers(-15, 16))
            y = 60 + row * 126 + int(rng.integers(-15, 16))
            radius = int(rng.integers(8, 24))
            box = (x - radius, y - radius, x + radius, y + radius)
            green = int(rng.integers(110, 250))
            draw.ellipse(box, fill=(20, green, 54))
            truth.ellipse(box, fill=255)
    image = image.filter(ImageFilter.GaussianBlur(1.8))
    rgb = np.asarray(image, dtype=float)
    rgb += rng.normal(0, 2, rgb.shape)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)), mask, {'expected_objects': 40}


def fibers(rng):
    image, mask = blank()
    draw = ImageDraw.Draw(image)
    truth = ImageDraw.Draw(mask)
    angles = []
    for row in range(5):
        for column in range(6):
            cx = 80 + column * 155
            cy = 60 + row * 126
            angle = float(rng.normal(25, 12))
            radians = math.radians(angle)
            length = int(rng.integers(55, 92))
            width = int(rng.integers(7, 16))
            dx = math.cos(radians) * length / 2
            dy = math.sin(radians) * length / 2
            points = [(cx - dx, cy - dy), (cx + dx, cy + dy)]
            draw.line(points, fill=(52, 71, 82), width=width)
            truth.line(points, fill=255, width=width)
            angles.append(angle)
    return image, mask, {'expected_objects': 30, 'angles': angles}


def porous(rng):
    image, mask = blank(background=(45, 60, 69))
    mask.paste(255, (0, 0, image.width, image.height))
    draw = ImageDraw.Draw(image)
    truth = ImageDraw.Draw(mask)
    for row in range(5):
        for column in range(8):
            x = 60 + column * 118 + int(rng.integers(-12, 13))
            y = 60 + row * 126 + int(rng.integers(-12, 13))
            radius_x = int(rng.integers(14, 35))
            radius_y = int(rng.integers(12, 32))
            box = (x - radius_x, y - radius_y, x + radius_x, y + radius_y)
            draw.ellipse(box, fill=(222, 231, 229))
            truth.ellipse(box, fill=0)
    return image, mask, {'expected_closed_pores': 40, 'note': '暗色为基体，分析孔洞请使用亮颗粒极性'}


def generate_sample(kind, seed=19):
    rng = np.random.default_rng(seed)
    generators = {
        'isolated': isolated,
        'touching': touching,
        'illumination': illumination,
        'fluorescence': fluorescence,
        'fibers': fibers,
        'porous': porous,
    }
    if kind not in generators:
        raise ValidationError('演示类型不存在')
    image, mask, details = generators[kind](rng)
    details['name'] = NAMES[kind]
    details['kind'] = kind
    details['seed'] = seed
    details['synthetic'] = True
    return image, mask, details
