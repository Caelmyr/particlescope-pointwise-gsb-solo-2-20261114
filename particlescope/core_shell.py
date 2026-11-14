"""核壳颗粒（core-shell）单颗粒内核与外壳分析。

整体面积只能反映颗粒大小，无法判断壳层是否均匀。本模块对**单个已分割颗粒**
指定一个内核区域，沿内核边界的外法线发射射线，测量每条射线从内核边界到外壳
边界的可解释距离，从而得到局部壳厚、壳厚均匀性和内核偏心程度。

可解释性约定（关键）：
- 射线必须从内核边界出发、全程落在该颗粒内部，并终止于该颗粒的外壳边界；
- 射线一旦在到达外壳边界前进入颗粒外部（图像背景）或落入颗粒内部孔洞
  （被前景包围、与图像外部连通性不同的背景），该射线标记为 invalid，
    其距离不计入任何壳厚统计，避免把穿过孔洞或颗粒外部的距离当成壳厚；
- 外壳凹陷、内核偏心、局部接触会导致部分射线失效，结果同时给出有效/无效
    射线计数与失效原因，供使用者判断哪些厚度可以解释。
"""

import math
import uuid
from copy import deepcopy

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from . import validation as v
from .imaging import channel_array
from .statistics import describe, scalar


# 单颗粒计算规模上限，防止在超大颗粒上发射过多射线导致请求过慢
MAX_PARTICLE_PIXELS = 4_000_000
# 单颗粒最多保留的逐射线记录条数，完整射线仍参与统计，只是不全量回传
MAX_RETURNED_RAYS = 2000


def boundary_pixels(mask):
    """区域中至少有一个四邻域为背景的像素。"""
    return mask & ~ndi.binary_erosion(mask, border_value=0)


def _raster_polygon(width, height, points):
    image = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(image)
    draw.polygon([(float(x), float(y)) for x, y in points], fill=1)
    return np.asarray(image, dtype=bool)


def core_mask_from_polygon(shape, polygon_points):
    height, width = shape
    return _raster_polygon(width, height, polygon_points)


def auto_core_mask(gray, particle_mask, polarity, threshold=None):
    """在颗粒内部按灰度阈值提取内核，取与阈值结果最大的连通分量。

    polarity='dark' 表示内核比壳层暗；'light' 表示内核更亮。
    threshold 为 None 时在颗粒内部用 Otsu 思路自动取类间方差最大的阈值。
    """
    values = gray[particle_mask].astype(np.float64)
    if values.size < 4:
        raise v.ValidationError('颗粒过小，无法自动识别内核')
    if threshold is None:
        threshold = _otsu_threshold(values)
    if polarity == 'dark':
        candidate = particle_mask & (gray <= threshold)
    else:
        candidate = particle_mask & (gray >= threshold)
    if not np.any(candidate):
        raise v.ValidationError('当前阈值在颗粒内没有选出内核像素，请改用手动绘制内核')
    labeled, count = ndi.label(candidate)
    if count <= 1:
        return candidate, float(threshold)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    biggest = int(np.argmax(sizes))
    return labeled == biggest, float(threshold)


def _otsu_threshold(values):
    hist, edges = np.histogram(values, bins=256, range=(0, 255))
    total = hist.sum()
    if total == 0:
        raise v.ValidationError('颗粒内没有可用灰度')
    centers = (edges[:-1] + edges[1:]) / 2
    weight_total = total
    sum_total = float((centers * hist).sum())
    weight_bg = 0.0
    sum_bg = 0.0
    best_variance = -1.0
    best_threshold = float(centers[len(centers) // 2])
    for index, count in enumerate(hist):
        weight_bg += count
        if weight_bg == 0:
            continue
        weight_fg = weight_total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += centers[index] * count
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = float(centers[index])
    return best_threshold


def _outward_normals(core_mask):
    """内核边界像素的外法线方向（单位向量，x 向右、y 向下）。

    用边界像素相对内核质心的径向方向为主、Sobel 梯度为辅：
    圆形内核可得到稳定径向法线；非凸内核上若径向指向实体内则改用梯度。
    """
    structure = core_mask.astype(np.float32)
    grad_x = ndi.sobel(structure, axis=1)
    grad_y = ndi.sobel(structure, axis=0)
    cy, cx = ndi.center_of_mass(core_mask)
    by, bx = np.nonzero(boundary_pixels(core_mask))
    # 法线统一按 (y 分量, x 分量) 存储，与数组索引 (row=y, col=x) 对齐
    # 径向方向由质心指向边界，恒为外法线
    radial = np.column_stack((by - cy, bx - cx)).astype(np.float32)
    radial_norm = np.hypot(radial[:, 0], radial[:, 1])
    radial_norm[radial_norm == 0] = 1.0
    radial = radial / radial_norm[:, None]
    # 填充掩膜的 Sobel 梯度由高处(内)指向低处(外)，取负号才是外法线
    sobel = np.column_stack((-grad_y[by, bx], -grad_x[by, bx]))
    sobel_norm = np.hypot(sobel[:, 0], sobel[:, 1])
    valid = sobel_norm > 1e-9
    sobel[valid] /= sobel_norm[valid, None]
    # 两种外法线方向一致（点积为正）时采用平滑的径向方向，否则用梯度兜底
    dot = radial[:, 0] * sobel[:, 0] + radial[:, 1] * sobel[:, 1]
    use_radial = ~valid | (dot > 0.2)
    normals = np.where(use_radial[:, None], radial, sobel)
    return by, bx, normals


def background_components(particle_mask):
    """把颗粒外的背景按连通性分成“外部背景”和“封闭孔洞”。

    返回 (背景标签矩阵, 外部背景标签集合)。与图像四周边界连通的背景分量为
    外部背景；其余被颗粒完全包围的背景分量为封闭孔洞。
    """
    background = ~particle_mask
    labels, _ = ndi.label(background)
    border_labels = set()
    for edge in (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]):
        border_labels.update(int(value) for value in np.unique(edge) if value != 0)
    return labels, border_labels


def _trace_ray(start_y, start_x, dir_y, dir_x, step, particle_mask,
               bg_labels, exterior_labels, max_steps):
    """沿单位外法线方向步进，返回 (状态, 行进像素长度)。

    可解释性规则（与用户约定一致）：
      valid       起点位于颗粒实体，射线全程在实体内行进，最终跨越颗粒外轮廓
                  进入外部背景，这段距离就是该方向真实壳厚；
      hole        射线在遇到外轮廓前进入某个被颗粒包围的封闭背景分量（孔洞），
                  该距离穿过孔洞，不能当作壳厚；
      exterior    射线走出图像边界且未遇到可界定外壳，或法线第一步即落在背景上
                  （内核贴壁/局部接触方向没有可测量壳层）；
      unreachable 达到步数上限仍未离开实体（数值保护，按无效处理）。
    """
    origin_y = float(start_y) + 0.5
    origin_x = float(start_x) + 0.5
    y, x = origin_y, origin_x
    step_y, step_x = dir_y * step, dir_x * step
    for _ in range(max_steps):
        y += step_y
        x += step_x
        iy = int(math.floor(y))
        ix = int(math.floor(x))
        traveled = math.hypot(x - origin_x, y - origin_y)
        if not (0 <= iy < particle_mask.shape[0] and 0 <= ix < particle_mask.shape[1]):
            return 'exterior', traveled - step
        if particle_mask[iy, ix]:
            continue
        label = int(bg_labels[iy, ix])
        if label in exterior_labels:
            # 从实体跨入外部背景即跨越外壳外轮廓，记为有效壳厚
            return 'valid', traveled - step / 2
        # 进入的是被颗粒包围的封闭背景分量（孔洞）
        return 'hole', traveled
    return 'unreachable', max_steps * step


def analyze_core_shell(gray, particle_mask, core_mask, scale=1.0, ray_step=1.0):
    """对单个颗粒执行核壳分析。

    gray          与掩膜同尺寸的灰度数组（float）
    particle_mask 该颗粒的实体掩膜（True 为颗粒，含壳与核）
    core_mask     内核掩膜（必须完全位于颗粒实体内部）
    scale         每像素物理长度
    ray_step      射线步进长度（像素），越小越精细、越慢
    """
    if particle_mask.shape != core_mask.shape or gray.shape != particle_mask.shape:
        raise v.ValidationError('内核掩膜、颗粒掩膜与图像尺寸必须一致')
    particle_pixels = int(particle_mask.sum())
    if particle_pixels == 0:
        raise v.ValidationError('颗粒掩膜为空')
    if particle_pixels > MAX_PARTICLE_PIXELS:
        raise v.ValidationError(f'单颗粒超过{MAX_PARTICLE_PIXELS}像素，请先裁切图像再分析')
    if int(core_mask.sum()) == 0:
        raise v.ValidationError('内核区域为空，请重新指定内核')
    outside_core = core_mask & ~particle_mask
    if np.any(outside_core):
        raise v.ValidationError(
            f'内核有 {int(outside_core.sum())} 个像素落在颗粒外部，请把内核绘制在该颗粒内部'
        )

    # 背景按连通性区分：与图像边界连通的是外部背景，被颗粒包围的是封闭孔洞
    bg_labels, exterior_labels = background_components(particle_mask)
    hole_mask = ~particle_mask & ~np.isin(bg_labels, list(exterior_labels))
    hole_pixels = int(hole_mask.sum())

    # 实体壳层 = 颗粒实体减去内核（孔洞本来就不是实体）
    shell_mask = particle_mask & ~core_mask
    shell_pixels = int(shell_mask.sum())

    # 内核相对颗粒的偏心：质心偏移，以及各自等效半径
    py, px = ndi.center_of_mass(particle_mask)
    cy, cx = ndi.center_of_mass(core_mask)
    offset_px = math.hypot(cx - px, cy - py)
    particle_equiv_radius = math.sqrt(particle_pixels / math.pi)
    core_equiv_radius = math.sqrt(int(core_mask.sum()) / math.pi)

    by, bx, normals = _outward_normals(core_mask)
    max_steps = int(math.ceil(max(particle_mask.shape) / max(ray_step, 0.1))) + 2
    particle_boundary = boundary_pixels(particle_mask)
    rays = []
    status_counter = {'valid': 0, 'hole': 0, 'exterior': 0, 'unreachable': 0}
    valid_thickness_px = []
    for index in range(len(by)):
        dir_y, dir_x = normals[index]
        sy, sx = int(by[index]), int(bx[index])
        # 内核边界与颗粒外轮廓重合：该方向壳厚为零（贴壁/局部接触），不可解释为壳厚
        if particle_boundary[sy, sx]:
            status, length_px = 'exterior', 0.0
        else:
            status, length_px = _trace_ray(
                float(sy), float(sx),
                dir_y, dir_x, ray_step,
                particle_mask, bg_labels, exterior_labels, max_steps,
            )
        status_counter[status] = status_counter.get(status, 0) + 1
        thickness = float(length_px) * scale if status == 'valid' else None
        if status == 'valid':
            valid_thickness_px.append(float(length_px))
        rays.append({
            'index': index,
            'start': [round(float(sx), 3), round(float(sy), 3)],
            'direction': [round(float(dir_x), 5), round(float(dir_y), 5)],
            'angle_degrees': round(math.degrees(math.atan2(dir_y, dir_x)) % 360, 3),
            'status': status,
            'thickness_px': round(float(length_px), 3) if status == 'valid' else None,
            'thickness': round(thickness, 6) if thickness is not None else None,
        })

    total_rays = len(rays)
    valid_rays = status_counter['valid']
    invalid_rays = total_rays - valid_rays
    valid_fraction = valid_rays / total_rays if total_rays else 0.0
    thickness_values = np.asarray(valid_thickness_px, dtype=float) * scale if valid_thickness_px else np.asarray([])
    thickness_summary = describe(thickness_values) if valid_thickness_px else describe([])

    # 壳厚均匀性：有效厚度的变异系数；越低越均匀
    uniformity_cv = thickness_summary.get('cv_percent')
    # 偏心程度：质心偏移相对颗粒等效半径归一化
    eccentricity_ratio = scalar(offset_px / particle_equiv_radius) if particle_equiv_radius else None

    warnings = []
    if valid_rays == 0:
        warnings.append('没有任何射线获得有效壳厚：内核可能贴到外壳、颗粒多孔或轮廓严重凹陷，壳厚不可解释')
    elif valid_fraction < 0.6:
        warnings.append(
            f'仅 {valid_fraction:.0%} 的射线有效，大量方向被孔洞、凹陷或接触打断，'
            '统计壳厚只代表可解释方向，不能代表整圈壳层'
        )
    if status_counter['hole']:
        warnings.append(
            f"{status_counter['hole']} 条射线穿过内部孔洞，已剔除其距离，不计入壳厚"
        )
    if status_counter['exterior']:
        warnings.append(
            f"{status_counter['exterior']} 条射线起点已贴在颗粒外轮廓上（该方向壳厚为零，"
            "常见于内核贴壁或局部接触），没有可测量壳层，已剔除"
        )
    if eccentricity_ratio is not None and eccentricity_ratio > 0.25:
        warnings.append('内核质心相对颗粒质心明显偏移，存在偏心，最薄与最厚壳厚方向应结合角度图查看')
    if shell_pixels == 0:
        warnings.append('内核已覆盖整个颗粒实体，不存在可测量的壳层')

    # 角度分箱（0-360°），用于前端极向壳厚曲线；无效角度给 null
    bins = 36
    angle_bins = [None] * bins
    angle_counts = [0] * bins
    for ray in rays:
        bucket = min(bins - 1, int(ray['angle_degrees'] / 360 * bins))
        angle_counts[bucket] += 1
        if ray['status'] == 'valid':
            value = ray['thickness']
            angle_bins[bucket] = value if angle_bins[bucket] is None else angle_bins[bucket] + value
    angle_centers = [round((i + 0.5) * 360 / bins, 2) for i in range(bins)]
    polar_thickness = [
        round(angle_bins[i] / angle_counts[i], 6) if angle_bins[i] is not None else None
        for i in range(bins)
    ]

    returned_rays = rays
    if len(rays) > MAX_RETURNED_RAYS:
        pick = np.linspace(0, len(rays) - 1, MAX_RETURNED_RAYS).astype(int)
        returned_rays = [rays[i] for i in pick]

    return {
        'particle_pixels': particle_pixels,
        'core_pixels': int(core_mask.sum()),
        'shell_pixels': shell_pixels,
        'hole_pixels': hole_pixels,
        'core_area_fraction': scalar(int(core_mask.sum()) / particle_pixels),
        'shell_area_fraction': scalar(shell_pixels / particle_pixels),
        'particle_centroid': [round(scalar(px), 4), round(scalar(py), 4)],
        'core_centroid': [round(scalar(cx), 4), round(scalar(cy), 4)],
        'centroid_offset_px': round(scalar(offset_px), 4),
        'centroid_offset': round(scalar(offset_px * scale), 6),
        'particle_equivalent_radius_px': round(scalar(particle_equiv_radius), 4),
        'core_equivalent_radius_px': round(scalar(core_equiv_radius), 4),
        'eccentricity_ratio': eccentricity_ratio,
        'thickness_summary': thickness_summary,
        'thickness_uniformity_cv_percent': uniformity_cv,
        'ray_count': total_rays,
        'valid_ray_count': valid_rays,
        'invalid_ray_count': invalid_rays,
        'valid_ray_fraction': round(scalar(valid_fraction), 4),
        'ray_status_counts': status_counter,
        'polar_angles_degrees': angle_centers,
        'polar_thickness': polar_thickness,
        'rays': returned_rays,
        'rays_truncated': len(rays) > MAX_RETURNED_RAYS,
        'warnings': warnings,
        'interpretation': (
            '壳厚仅统计全程位于颗粒实体、终止于外壳边界的射线；'
            '穿过孔洞或穿出颗粒外部的射线标记为无效并剔除。'
        ),
    }


def render_core_shell(image, particle_mask, core_mask, report):
    """生成核壳分析叠加图：内核、壳层、有效/无效射线与最薄最厚方向。"""
    base = np.asarray(image.convert('RGB'), dtype=np.float32).copy()
    # 内核半透明蓝，壳层半透明橙
    overlay = base.copy()
    overlay[particle_mask & ~core_mask] = (239, 169, 92)
    overlay[core_mask] = (94, 168, 244)
    blended = base * 0.55 + overlay * 0.45
    output = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)).convert('RGBA')
    draw = ImageDraw.Draw(output)

    valid_color = (81, 226, 186, 255)
    hole_color = (255, 120, 120, 255)
    exterior_color = (200, 200, 200, 255)
    status_color = {
        'valid': valid_color,
        'hole': hole_color,
        'exterior': exterior_color,
        'unreachable': exterior_color,
    }
    # 射线过多时抽样绘制，保持图面可读
    rays = report['rays']
    for ray in rays:
        sx, sy = ray['start']
        dx, dy = ray['direction']
        length = ray['thickness_px'] if ray['status'] == 'valid' else 4
        if length is None:
            length = 4
        end = (sx + dx * length, sy + dy * length)
        draw.line((sx, sy, end[0], end[1]), fill=status_color[ray['status']], width=1)

    # 内核边界与颗粒外边界
    core_edge = boundary_pixels(core_mask)
    array = np.asarray(output).copy()
    array[core_edge] = (40, 110, 230, 255)
    particle_edge = boundary_pixels(particle_mask)
    array[particle_edge] = (255, 255, 255, 255)
    output = Image.fromarray(array)
    draw = ImageDraw.Draw(output)

    # 质心标记：颗粒（绿）与内核（蓝），连线表示偏心方向
    px, py = report['particle_centroid']
    cx, cy = report['core_centroid']
    r = max(3, image.width // 250)
    draw.line((px, py, cx, cy), fill=(255, 230, 90, 255), width=2)
    draw.ellipse((px - r, py - r, px + r, py + r), outline=(120, 255, 200, 255), width=2)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(150, 200, 255, 255), width=2)

    # 最薄/最厚有效射线
    valid = [ray for ray in rays if ray['status'] == 'valid' and ray['thickness_px'] is not None]
    if valid:
        thinnest = min(valid, key=lambda r: r['thickness_px'])
        thickest = max(valid, key=lambda r: r['thickness_px'])
        for ray, color, name in [
            (thinnest, (255, 90, 90, 255), 'min'),
            (thickest, (120, 255, 120, 255), 'max'),
        ]:
            sx, sy = ray['start']
            dx, dy = ray['direction']
            length = ray['thickness_px']
            draw.line((sx, sy, sx + dx * length, sy + dy * length), fill=color, width=3)
    return output.convert('RGB')


def validate_core_definition(body, width, height):
    """解析前端提交的内核定义。"""
    data = v.mapping(body, '内核定义')
    mode = v.choice(data.get('mode', 'manual'), '内核指定方式', ['manual', 'auto'])
    definition = {'mode': mode}
    if mode == 'manual':
        points = v.polygon(data.get('points'), width, height)
        definition['points'] = points
    else:
        definition['polarity'] = v.choice(
            data.get('polarity', 'dark'), '内核极性', ['dark', 'light']
        )
        threshold = data.get('threshold')
        definition['threshold'] = v.real(threshold, '灰度阈值', 0, 255) if threshold is not None else None
        definition['channel'] = v.choice(
            data.get('channel', 'gray'), '分析通道',
            ['gray', 'red', 'green', 'blue', 'hue', 'saturation', 'value'],
        )
    definition['ray_step'] = v.real(data.get('ray_step', 1.0), '射线步进像素', 0.25, 4.0)
    return definition


def build_core_shell_record(store, service, sample_id, particle_id, definition, expected_revision=None):
    """取当前分析中的指定颗粒，执行核壳分析并持久化。"""
    from .storage import ConflictError

    with store.lock:
        meta = store.meta(sample_id)
        if expected_revision is not None and meta['revision'] != expected_revision:
            raise ConflictError('样本已被其他操作修改，请刷新后重试')
        result = service.require_result(sample_id)
        labels = service.labels(sample_id)
        particle_mask = labels == particle_id
        if not np.any(particle_mask):
            raise v.ValidationError('颗粒编号不存在或已被修订，请重新选择颗粒')
        image = store.working_image(sample_id)
        scale = meta['calibration']['scale']

        if definition['mode'] == 'manual':
            core_mask = core_mask_from_polygon(particle_mask.shape, definition['points'])
            used_threshold = None
            channel = 'gray'
            polarity = None
        else:
            channel = definition['channel']
            gray_full = channel_array(image, channel)
            gray = gray_full
            core_mask, used_threshold = auto_core_mask(
                gray, particle_mask, definition['polarity'], definition['threshold']
            )
            polarity = definition['polarity']
        if definition['mode'] == 'manual':
            gray = channel_array(image, 'gray')

        report = analyze_core_shell(gray, particle_mask, core_mask, scale, definition['ray_step'])
        overlay = render_core_shell(image, particle_mask, core_mask, report)

        record_id = uuid.uuid4().hex
        directory = store.sample_dir(sample_id) / 'core_shell'
        directory.mkdir(exist_ok=True)
        record = {
            'id': record_id,
            'sample_id': sample_id,
            'analysis_id': result['id'],
            'image_revision': meta['image_revision'],
            'particle_id': int(particle_id),
            'created_at': v.now(),
            'definition': {
                'mode': definition['mode'],
                'ray_step': definition['ray_step'],
                'channel': channel,
                'polarity': polarity,
                'threshold': used_threshold,
                'points': definition.get('points'),
            },
            'calibration': deepcopy(meta['calibration']),
            'report': report,
        }
        store.json_write(directory / f'{record_id}.json', record)
        store.image_write(directory / f'{record_id}.png', overlay)
        # 同时保存内核掩膜，便于下载与复核
        store.image_write(
            directory / f'{record_id}-core.png',
            Image.fromarray((core_mask * 255).astype(np.uint8)),
        )
        return record


def list_records(store, sample_id):
    directory = store.sample_dir(sample_id) / 'core_shell'
    if not directory.exists():
        return []
    records = []
    for path in directory.glob('*.json'):
        data = store.json_read(path)
        if not data:
            continue
        meta = store.meta(sample_id)
        stale = data.get('image_revision') != meta['image_revision']
        records.append({
            'id': data['id'],
            'particle_id': data['particle_id'],
            'created_at': data['created_at'],
            'mode': data['definition']['mode'],
            'stale': stale,
            'valid_ray_fraction': data['report']['valid_ray_fraction'],
            'mean_thickness': data['report']['thickness_summary']['mean'],
            'eccentricity_ratio': data['report']['eccentricity_ratio'],
            'uniformity_cv_percent': data['report']['thickness_uniformity_cv_percent'],
        })
    return sorted(records, key=lambda r: r['created_at'], reverse=True)


def get_record(store, sample_id, record_id):
    v.identifier(record_id, '核壳分析编号')
    path = store.sample_dir(sample_id) / 'core_shell' / f'{record_id}.json'
    record = store.json_read(path)
    if record is None:
        raise v.ValidationError('核壳分析记录不存在')
    meta = store.meta(sample_id)
    record['stale'] = record.get('image_revision') != meta['image_revision']
    return record


def delete_record(store, sample_id, record_id, expected_revision=None):
    v.identifier(record_id, '核壳分析编号')
    with store.transaction(sample_id, expected_revision):
        directory = store.sample_dir(sample_id) / 'core_shell'
        json_path = directory / f'{record_id}.json'
        if not json_path.exists():
            raise v.ValidationError('核壳分析记录不存在')
        json_path.unlink()
        for suffix in ['.png', '-core.png']:
            target = directory / f'{record_id}{suffix}'
            target.unlink(missing_ok=True)
    return {'deleted': record_id}
