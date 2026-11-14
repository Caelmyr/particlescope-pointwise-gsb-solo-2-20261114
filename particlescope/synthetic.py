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
    'core_shell': '核壳颗粒 · 合成演示',
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


# 核壳演示的四种典型构型：均匀同心、偏心、外壳凹陷、壳层带孔
CORE_SHELL_KINDS = ['uniform', 'eccentric', 'dent', 'porous_shell']


def core_shell(rng):
    image, mask = blank(background=(236, 238, 236))
    draw = ImageDraw.Draw(image)
    truth = ImageDraw.Draw(mask)
    shell_color = (214, 142, 70)
    core_color = (58, 96, 150)
    pore_color = (236, 238, 236)
    objects = []
    columns = 4
    for index in range(8):
        row = index // columns
        column = index % columns
        cx = 130 + column * 240 + int(rng.integers(-8, 9))
        cy = 165 + row * 320 + int(rng.integers(-8, 9))
        outer = int(rng.integers(48, 58))
        kind = CORE_SHELL_KINDS[index % len(CORE_SHELL_KINDS)]
        core_radius = int(outer * rng.uniform(0.42, 0.55))
        offset_x = offset_y = 0
        if kind == 'eccentric':
            offset_x = int(outer * 0.28)
            offset_y = int(outer * 0.16)
        # 外壳（整体颗粒）
        outer_box = (cx - outer, cy - outer, cx + outer, cy + outer)
        draw.ellipse(outer_box, fill=shell_color)
        truth.ellipse(outer_box, fill=255)
        if kind == 'dent':
            # 在外壳右侧咬出一个凹陷（与背景同色），使该方向壳厚射线穿出外部
            dent_r = int(outer * 0.42)
            dx = cx + outer - int(dent_r * 0.35)
            draw.ellipse((dx - dent_r, cy - dent_r, dx + dent_r, cy + dent_r), fill=pore_color)
            truth.ellipse((dx - dent_r, cy - dent_r, dx + dent_r, cy + dent_r), fill=0)
        if kind == 'porous_shell':
            # 在壳层内打两个封闭小孔（背景色孔洞，仍被外壳实体包围）
            for _ in range(2):
                angle = float(rng.uniform(0, 2 * math.pi))
                ring = (core_radius + outer) / 2
                px = cx + int(ring * math.cos(angle))
                py = cy + int(ring * math.sin(angle))
                pr = int(outer * 0.12)
                draw.ellipse((px - pr, py - pr, px + pr, py + pr), fill=pore_color)
                truth.ellipse((px - pr, py - pr, px + pr, py + pr), fill=0)
        # 内核
        kx = cx + offset_x
        ky = cy + offset_y
        draw.ellipse(
            (kx - core_radius, ky - core_radius, kx + core_radius, ky + core_radius),
            fill=core_color,
        )
        objects.append({
            'x': cx, 'y': cy, 'outer_radius': outer,
            'core_radius': core_radius,
            'core_offset_x': offset_x, 'core_offset_y': offset_y,
            'kind': kind,
        })
    return image, mask, {
        'expected_objects': 8,
        'objects': objects,
        'shell_color': list(shell_color),
        'core_color': list(core_color),
        'note': '亮橙色为外壳、深蓝色为内核；包含同心、偏心、凹陷和带孔壳层四种构型',
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
        'core_shell': core_shell,
    }
    if kind not in generators:
        raise ValidationError('演示类型不存在')
    image, mask, details = generators[kind](rng)
    details['name'] = NAMES[kind]
    details['kind'] = kind
    details['seed'] = seed
    details['synthetic'] = True
    return image, mask, details
