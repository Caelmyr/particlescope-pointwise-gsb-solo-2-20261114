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
    'spillover_red': '红染料单染对照 · 串色演示',
    'spillover_green': '绿染料单染对照 · 串色演示',
    'spillover_mixed': '红绿混染样本 · 串色演示',
}

# 演示用真实串扰系数：绿染料向红通道溢出、红染料向绿通道溢出
SPILLOVER = {'red_to_green': 0.28, 'green_to_red': 0.22}


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


def illumination(rng):
    image, mask, details = isolated(rng)
    rgb = np.asarray(image, dtype=float)
    y, x = np.indices(rgb.shape[:2])
    shading = 0.45 + 0.55 * x / rgb.shape[1]
    shading *= 0.85 + 0.15 * np.cos(y / rgb.shape[0] * math.pi)
    rgb *= shading[:, :, None]
    noise = rng.normal(0, 2.5, rgb.shape[:2])
    rgb += noise[:, :, None]
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)), mask, details


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


def _fluorescence_truth(rng, mixed=False, single='red'):
    """生成红绿两通道的真实荧光强度图、并集掩膜和颗粒真值。"""
    width, height = 960, 640
    red_true = np.zeros((height, width), dtype=np.float64)
    green_true = np.zeros((height, width), dtype=np.float64)
    mask = Image.new('L', (width, height), 0)
    truth = ImageDraw.Draw(mask)
    ground_truth = []
    yy, xx = np.ogrid[:height, :width]
    for row in range(5):
        for column in range(8):
            x = 60 + column * 118 + int(rng.integers(-15, 16))
            y = 60 + row * 126 + int(rng.integers(-15, 16))
            radius = int(rng.integers(9, 22))
            disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
            if mixed:
                kind = ('red', 'green', 'double')[column % 3]
            else:
                kind = single
            red_level = float(rng.integers(120, 190))
            green_level = float(rng.integers(120, 190))
            if kind in ('red', 'double'):
                red_true[disk] = red_level
            if kind in ('green', 'double'):
                green_true[disk] = green_level
            truth.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
            ground_truth.append({'x': x, 'y': y, 'radius': radius, 'kind': kind})
    return red_true, green_true, mask, ground_truth


def _compose_spillover(rng, red_true, green_true):
    """按线性串扰混合并加入暗背景、噪声与轻微失焦。"""
    red_obs = red_true + SPILLOVER['green_to_red'] * green_true
    green_obs = green_true + SPILLOVER['red_to_green'] * red_true
    red_obs += 8.0
    green_obs += 10.0
    rgb = np.stack([red_obs, green_obs, np.zeros_like(red_true)], axis=2)
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.4))
    array = np.asarray(image, dtype=np.float64)
    array += rng.normal(0, 2.2, array.shape)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def spillover_red(rng):
    red_true, green_true, mask, ground_truth = _fluorescence_truth(rng, mixed=False)
    image = _compose_spillover(rng, red_true, green_true)
    return image, mask, {
        'expected_objects': len(ground_truth),
        'control': 'red',
        'true_spillover': SPILLOVER,
        'ground_truth': ground_truth,
        'note': '红染料单染对照：红通道为真实信号，绿通道仅含串色与背景',
    }


def spillover_green(rng):
    red_true, green_true, mask, ground_truth = _fluorescence_truth(rng, mixed=False, single='green')
    image = _compose_spillover(rng, red_true, green_true)
    return image, mask, {
        'expected_objects': len(ground_truth),
        'control': 'green',
        'true_spillover': SPILLOVER,
        'ground_truth': ground_truth,
        'note': '绿染料单染对照：绿通道为真实信号，红通道仅含串色与背景',
    }


def spillover_mixed(rng):
    red_true, green_true, mask, ground_truth = _fluorescence_truth(rng, mixed=True)
    image = _compose_spillover(rng, red_true, green_true)
    counts = {kind: sum(1 for item in ground_truth if item['kind'] == kind) for kind in ('red', 'green', 'double')}
    return image, mask, {
        'expected_objects': len(ground_truth),
        'control': 'mixed',
        'true_spillover': SPILLOVER,
        'ground_truth': ground_truth,
        'expected_classification': counts,
        'note': '混染样本：含红单阳、绿单阳与双阳性颗粒，直接比较通道亮度会误判',
    }


def generate_sample(kind, seed=19):
    rng = np.random.default_rng(seed)
    generators = {
        'isolated': isolated,
        'touching': touching,
        'illumination': illumination,
        'fluorescence': fluorescence,
        'fibers': fibers,
        'porous': porous,
        'spillover_red': spillover_red,
        'spillover_green': spillover_green,
        'spillover_mixed': spillover_mixed,
    }
    if kind not in generators:
        raise ValidationError('演示类型不存在')
    image, mask, details = generators[kind](rng)
    details['name'] = NAMES[kind]
    details['kind'] = kind
    details['seed'] = seed
    details['synthetic'] = True
    return image, mask, details
