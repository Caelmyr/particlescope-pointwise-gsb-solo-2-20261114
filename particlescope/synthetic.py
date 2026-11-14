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
    'domains': '晶格畴与无序区 · 合成演示',
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


def lattice_domains(rng):
    """两个不同取向的六方晶格畴，中间夹一条无序颗粒带和点缺陷。"""
    image, mask = blank(size=(980, 620))
    draw = ImageDraw.Draw(image)
    truth = ImageDraw.Draw(mask)
    nn_distance = 36
    # 标准三角（六方）晶格：同行相邻距离a，行间距a*sqrt(3)/2，奇数行
    # 水平错开a/2，任意第一壳层邻居的距离都严格等于nn_distance。
    column_pitch = nn_distance
    row_pitch = nn_distance * math.sqrt(3) / 2
    row_offset = nn_distance / 2
    radius = 12
    domain_specs = [
        {'origin': (48, 55), 'rows': 9, 'cols': 9, 'angle': 5.0, 'label': 1},
        {'origin': (600, 55), 'rows': 9, 'cols': 8, 'angle': 20.0, 'label': 2},
    ]
    domain_members = {1: [], 2: []}
    for spec in domain_specs:
        theta = math.radians(spec['angle'])
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        for row in range(spec['rows']):
            for column in range(spec['cols']):
                local_x = column * column_pitch + (row % 2) * row_offset
                local_y = row * row_pitch
                x = spec['origin'][0] + local_x * cos_t - local_y * sin_t + float(rng.normal(0, 1.0))
                y = spec['origin'][1] + local_x * sin_t + local_y * cos_t + float(rng.normal(0, 1.0))
                # 记录旋转前的行列号，保证空位删除与绘制使用同一坐标。
                domain_members[spec['label']].append((x, y, row, column))
    # 从每个畴中移除两个颗粒模拟点缺陷（空位）。
    vacancies = set()
    for label in (1, 2):
        choices = rng.choice(len(domain_members[label]), size=2, replace=False)
        vacancies.update((label, int(index)) for index in choices)
    particle_id = 0
    centers = []
    for label, members in domain_members.items():
        for index, (x, y, _, _) in enumerate(members):
            if (label, index) in vacancies:
                continue
            particle_id += 1
            jitter_x, jitter_y = float(rng.normal(0, 0.8)), float(rng.normal(0, 0.8))
            box = (x - radius + jitter_x, y - radius + jitter_y, x + radius + jitter_x, y + radius + jitter_y)
            draw.ellipse(box, fill=(46, 62, 74))
            truth.ellipse(box, fill=255)
            centers.append([x + jitter_x, y + jitter_y])
    # 中间无序带：随机位置、不形成晶格。
    disordered = []
    while len(disordered) < 26:
        x = float(rng.uniform(385, 560))
        y = float(rng.uniform(60, 520))
        if all(math.hypot(x - p[0], y - p[1]) > radius * 1.7 for p in disordered):
            disordered.append((x, y))
    for x, y in disordered:
        particle_id += 1
        box = (x - radius, y - radius, x + radius, y + radius)
        draw.ellipse(box, fill=(46, 62, 74))
        truth.ellipse(box, fill=255)
        centers.append([x, y])
    return image, mask, {
        'expected_ordered_domains': 2,
        'expected_domain_labels': [1, 2],
        'domain_angles_degrees': [5.0, 20.0],
        'vacancy_count': len(vacancies),
        'disordered_count': len(disordered),
        'lattice_spacing_px': nn_distance,
        'centers': centers,
        'note': '两个六方晶格畴取向不同，中间为无序带，空位用于检验缺陷不被强行并入晶格',
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
        'domains': lattice_domains,
    }
    if kind not in generators:
        raise ValidationError('演示类型不存在')
    image, mask, details = generators[kind](rng)
    details['name'] = NAMES[kind]
    details['kind'] = kind
    details['seed'] = seed
    details['synthetic'] = True
    return image, mask, details
