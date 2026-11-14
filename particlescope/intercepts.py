import math
from collections import Counter
import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
from .statistics import describe, histogram, scalar
from .validation import ValidationError

MAX_LINES = 4000
MAX_SAMPLES = 8000000
PHASE_OUTSIDE = 0
PHASE_PARTICLE = 1
PHASE_VOID = 2
PHASE_HOLE = 3
PHASE_EXCLUDED = 4
PHASE_NAMES = {
    PHASE_PARTICLE: 'particle',
    PHASE_VOID: 'void',
    PHASE_HOLE: 'hole',
    PHASE_EXCLUDED: 'excluded',
}


def phase_map(labels, selected, included_ids, include_holes=True):
    labels = np.asarray(labels)
    selected = np.asarray(selected, dtype=bool)
    phases = np.full(labels.shape, PHASE_VOID, dtype=np.uint8)
    keep = np.isin(labels, [int(v) for v in included_ids]) & (labels > 0)
    if include_holes:
        void_labels, count = ndi.label(labels == 0)
        if count:
            boundary = selected & ~ndi.binary_erosion(selected, border_value=0)
            exterior = set(int(v) for v in np.unique(void_labels[boundary]).tolist() if v)
            exterior.update(int(v) for v in np.unique(void_labels[~selected]).tolist() if v)
            structure = np.ones((3, 3), dtype=bool)
            void_region = void_labels > 0
            touching_any = np.unique(void_labels[ndi.binary_dilation(labels > 0, structure) & void_region])
            touching_included = np.unique(void_labels[ndi.binary_dilation(keep, structure) & void_region])
            enclosed = [int(v) for v in np.unique(void_labels[void_region]).tolist() if v not in exterior]
            excluded_enclosed = set(int(v) for v in touching_any.tolist() if v) - set(
                int(v) for v in touching_included.tolist() if v
            )
            hole_ids = [v for v in enclosed if v not in excluded_enclosed]
            excluded_ids = [v for v in enclosed if v in excluded_enclosed]
            if hole_ids:
                phases[np.isin(void_labels, hole_ids)] = PHASE_HOLE
            if excluded_ids:
                phases[np.isin(void_labels, excluded_ids)] = PHASE_EXCLUDED
    phases[labels > 0] = PHASE_EXCLUDED
    phases[keep] = PHASE_PARTICLE
    phases[~selected] = PHASE_VOID
    return phases


def line_geometry(shape, selected, angle, spacing_px, offset_fraction):
    height, width = shape
    ys, xs = np.nonzero(selected)
    if not len(xs):
        raise ValidationError('分析区域为空，无法布置测线')
    left, right = float(xs.min()) - 1, float(xs.max()) + 2
    top, bottom = float(ys.min()) - 1, float(ys.max()) + 2
    center_x, center_y = width / 2.0, height / 2.0
    radians = math.radians(angle)
    dx, dy = math.cos(radians), math.sin(radians)
    nx, ny = -dy, dx
    projections = [
        (x - center_x) * nx + (y - center_y) * ny
        for x, y in [(left, top), (right, top), (left, bottom), (right, bottom)]
    ]
    u_min, u_max = min(projections), max(projections)
    offset = (offset_fraction % 1.0) * spacing_px
    count = int(math.floor((u_max - u_min - offset) / spacing_px)) + 1
    positions = u_min + offset + np.arange(max(count, 1)) * spacing_px
    positions = positions[positions <= u_max + 1e-9]
    if not len(positions):
        positions = np.array([u_min + offset])
    if len(positions) > MAX_LINES:
        raise ValidationError(f'测线超过{MAX_LINES}条，请增大间距')
    t_corners = [
        (x - center_x) * dx + (y - center_y) * dy
        for x, y in [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]
    ]
    return {
        'center': (center_x, center_y),
        'direction': (dx, dy),
        'normal': (nx, ny),
        'positions': positions,
        't_min': min(t_corners),
        't_max': max(t_corners),
    }


def line_point(geometry, position, location):
    center_x, center_y = geometry['center']
    dx, dy = geometry['direction']
    nx, ny = geometry['normal']
    return (
        center_x + nx * position + dx * location,
        center_y + ny * position + dy * location,
    )


def sample_point(phases, labels, selected, geometry, position, location):
    x, y = line_point(geometry, position, location)
    height, width = phases.shape
    xi, yi = int(round(x)), int(round(y))
    if xi < 0 or yi < 0 or xi >= width or yi >= height:
        return PHASE_OUTSIDE, 0, 'image'
    if not selected[yi, xi]:
        return PHASE_OUTSIDE, 0, 'roi'
    return int(phases[yi, xi]), int(labels[yi, xi]), 'inside'


def _sample_run(phases, labels, selected, geometry, position, step):
    center_x, center_y = geometry['center']
    dx, dy = geometry['direction']
    nx, ny = geometry['normal']
    base_x, base_y = center_x + nx * position, center_y + ny * position
    length = geometry['t_max'] - geometry['t_min']
    count = max(2, int(math.ceil(length / step)) + 1)
    locations = np.linspace(geometry['t_min'], geometry['t_max'], count)
    xs = np.rint(base_x + dx * locations).astype(np.int64)
    ys = np.rint(base_y + dy * locations).astype(np.int64)
    height, width = phases.shape
    inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    values = np.zeros(len(locations), dtype=np.uint8)
    label_values = np.zeros(len(locations), dtype=np.int64)
    valid = np.flatnonzero(inside)
    values[valid] = phases[ys[valid], xs[valid]]
    label_values[valid] = labels[ys[valid], xs[valid]]
    selected_values = np.zeros(len(locations), dtype=bool)
    selected_values[valid] = selected[ys[valid], xs[valid]]
    values[~selected_values] = PHASE_OUTSIDE
    keys = np.where(values == PHASE_PARTICLE, label_values, -values.astype(np.int64))
    keys[values == PHASE_OUTSIDE] = 0
    return locations, values, keys


def _refine(phases, labels, selected, geometry, position, left, right):
    left_code = sample_point(phases, labels, selected, geometry, position, left)[0]
    left_label = sample_point(phases, labels, selected, geometry, position, left)[1]
    left_key = left_label if left_code == PHASE_PARTICLE else -left_code
    for _ in range(20):
        middle = (left + right) / 2
        code, label_id, _ = sample_point(phases, labels, selected, geometry, position, middle)
        key = label_id if code == PHASE_PARTICLE else -code
        if key == left_key:
            left = middle
        else:
            right = middle
    return (left + right) / 2


def _neighbour(phases, labels, selected, geometry, position, location):
    code, _, kind = sample_point(phases, labels, selected, geometry, position, location)
    if code == PHASE_OUTSIDE:
        return kind
    return PHASE_NAMES.get(code, 'void')


def run_intercepts(phases, labels, selected, geometry, position, step, line_index):
    locations, values, keys = _sample_run(phases, labels, selected, geometry, position, step)
    changes = np.flatnonzero(keys[1:] != keys[:-1]) + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(keys)]])
    rows = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        code = int(values[start])
        if code == PHASE_OUTSIDE:
            continue
        if start == 0:
            start_t = float(locations[0])
            start_boundary = 'image'
            start_neighbour = 'image'
        else:
            start_t = _refine(
                phases, labels, selected, geometry, position,
                float(locations[start - 1]), float(locations[start]),
            )
            previous = int(values[start - 1])
            if previous == PHASE_OUTSIDE:
                start_boundary = _neighbour(
                    phases, labels, selected, geometry, position,
                    float(locations[start - 1]),
                )
                start_neighbour = start_boundary
            else:
                start_boundary = 'transition'
                start_neighbour = PHASE_NAMES.get(previous, 'void')
        if end == len(keys):
            end_t = float(locations[-1])
            end_boundary = 'image'
            end_neighbour = 'image'
        else:
            end_t = _refine(
                phases, labels, selected, geometry, position,
                float(locations[end - 1]), float(locations[end]),
            )
            following = int(values[end])
            if following == PHASE_OUTSIDE:
                end_boundary = _neighbour(
                    phases, labels, selected, geometry, position,
                    float(locations[end]),
                )
                end_neighbour = end_boundary
            else:
                end_boundary = 'transition'
                end_neighbour = PHASE_NAMES.get(following, 'void')
        length_px = end_t - start_t
        if length_px <= 1e-6:
            continue
        middle = (start_t + end_t) / 2
        _, label_id, _ = sample_point(phases, labels, selected, geometry, position, middle)
        start_point = line_point(geometry, position, start_t)
        end_point = line_point(geometry, position, end_t)
        middle_point = line_point(geometry, position, middle)
        rows.append({
            'line': line_index,
            'phase': PHASE_NAMES[code],
            'label_id': int(label_id) if code == PHASE_PARTICLE else None,
            'start': [round(start_point[0], 3), round(start_point[1], 3)],
            'end': [round(end_point[0], 3), round(end_point[1], 3)],
            'midpoint': [round(middle_point[0], 3), round(middle_point[1], 3)],
            'length_px': length_px,
            'complete': start_boundary == 'transition' and end_boundary == 'transition',
            'start_boundary': start_boundary,
            'end_boundary': end_boundary,
            'start_neighbour': start_neighbour,
            'end_neighbour': end_neighbour,
        })
    in_roi = int(np.count_nonzero(values != PHASE_OUTSIDE))
    return rows, in_roi


def _length_summary(values):
    if not values:
        return {
            'count': 0, 'mean_length': None, 'median_length': None,
            'minimum': None, 'maximum': None, 'std': None, 'total_length': 0,
        }
    description = describe(values)
    return {
        'count': len(values),
        'mean_length': description['mean'],
        'median_length': description['median'],
        'minimum': description['minimum'],
        'maximum': description['maximum'],
        'std': description['std'],
        'total_length': scalar(sum(values)),
    }


def _distribution(values, bins):
    if not values:
        return {
            'edges': [], 'counts': [], 'percent': [], 'cumulative_percent': [],
            'n': 0, 'mean': None, 'median': None,
        }
    result = histogram(values, bins=bins)
    result['n'] = len(values)
    result['mean'] = scalar(np.mean(values))
    result['median'] = scalar(np.median(values))
    return result


def intercept_analysis(labels, selected, included_ids, calibration, parameters, bins=20):
    labels = np.asarray(labels)
    selected = np.asarray(selected, dtype=bool)
    if labels.shape != selected.shape:
        raise ValidationError('分析区域与掩膜尺寸不一致')
    scale = calibration['scale']
    unit = calibration['unit']
    spacing_px = parameters.get('spacing_px')
    physical_spacing = parameters.get('spacing')
    if spacing_px is None:
        if physical_spacing is None:
            raise ValidationError('请填写测线间距')
        spacing_px = physical_spacing / scale
    spacing_px = float(spacing_px)
    if not math.isfinite(spacing_px) or spacing_px < 0.5:
        raise ValidationError('测线间距不能小于0.5像素')
    geometry = line_geometry(labels.shape, selected, parameters['angle'], spacing_px, parameters['offset'])
    step = parameters.get('sample_step', 0.5)
    estimate = len(geometry['positions']) * (geometry['t_max'] - geometry['t_min']) / step
    if estimate > MAX_SAMPLES:
        raise ValidationError('测线采样量过大，请增大采样步长或测线间距')
    phases = phase_map(labels, selected, included_ids, parameters.get('include_holes', True))
    intercepts = []
    per_line = []
    for line_index, position in enumerate(geometry['positions'].tolist()):
        rows, in_roi = run_intercepts(phases, labels, selected, geometry, position, step, line_index)
        if not rows:
            continue
        intercepts.extend(rows)
        per_line.append({
            'line': line_index,
            'position': round(position, 3),
            'in_roi_samples': in_roi,
            'intercepts': len(rows),
            'particle_intercepts': sum(1 for r in rows if r['phase'] == 'particle'),
            'complete_particle_intercepts': sum(
                1 for r in rows if r['phase'] == 'particle' and r['complete']
            ),
            'hole_intercepts': sum(1 for r in rows if r['phase'] == 'hole'),
            'excluded_intercepts': sum(1 for r in rows if r['phase'] == 'excluded'),
            'particle_length': round(sum(r['length_px'] for r in rows if r['phase'] == 'particle'), 3),
            'void_length': round(sum(r['length_px'] for r in rows if r['phase'] == 'void'), 3),
        })
    intercepts.sort(key=lambda r: (r['line'], r['start'][0] + r['start'][1]))
    for index, row in enumerate(intercepts, 1):
        row['id'] = index
        row['length'] = round(row['length_px'] * scale, 6)
        row['length_px'] = round(row['length_px'], 3)
    particles = [r for r in intercepts if r['phase'] == 'particle']
    complete_particles = [r for r in particles if r['complete']]
    truncated_particles = [r for r in particles if not r['complete']]
    holes = [r for r in intercepts if r['phase'] == 'hole']
    void = [r for r in intercepts if r['phase'] == 'void']
    excluded = [r for r in intercepts if r['phase'] == 'excluded']
    truncation = Counter()
    neighbour_pairs = Counter()
    for row in particles:
        if row['complete']:
            truncation['complete'] += 1
            neighbour_pairs[f"{row['start_neighbour']}|{row['end_neighbour']}"] += 1
            continue
        truncation['truncated'] += 1
        boundaries = {row['start_boundary'], row['end_boundary']} - {'transition'}
        truncation['both_ends' if len(boundaries) == 2 else 'one_end'] += 1
        for boundary in boundaries:
            truncation['image_edge' if boundary == 'image' else 'roi_edge'] += 1
    summary = {
        'particle': dict(
            _length_summary([r['length'] for r in complete_particles]),
            intercepts=len(particles),
        ),
        'particle_truncated': dict(
            _length_summary([r['length'] for r in truncated_particles]),
            intercepts=len(truncated_particles),
        ),
        'void': dict(_length_summary([r['length'] for r in void if r['complete']]), intercepts=len(void)),
        'hole': dict(_length_summary([r['length'] for r in holes if r['complete']]), intercepts=len(holes)),
        'excluded': dict(
            _length_summary([r['length'] for r in excluded if r['complete']]),
            intercepts=len(excluded),
        ),
    }
    return {
        'angle_degrees': parameters['angle'],
        'spacing': round(spacing_px * scale, 6) if physical_spacing is not None else None,
        'spacing_px': round(spacing_px, 6),
        'unit': unit,
        'scale': scale,
        'offset_fraction': parameters['offset'],
        'sample_step_px': step,
        'include_holes': bool(parameters.get('include_holes', True)),
        'geometry': {
            'center': [round(geometry['center'][0], 3), round(geometry['center'][1], 3)],
            'direction': [round(geometry['direction'][0], 6), round(geometry['direction'][1], 6)],
            'normal': [round(geometry['normal'][0], 6), round(geometry['normal'][1], 6)],
            't_min': round(geometry['t_min'], 3),
            't_max': round(geometry['t_max'], 3),
        },
        'line_count': len(per_line),
        'line_positions': len(geometry['positions']),
        'lines_intersecting_roi': len(per_line),
        'intercept_total': len(intercepts),
        'summary': summary,
        'truncation': {
            'complete': truncation.get('complete', 0),
            'truncated': truncation.get('truncated', 0),
            'one_end': truncation.get('one_end', 0),
            'both_ends': truncation.get('both_ends', 0),
            'image_edge': truncation.get('image_edge', 0),
            'roi_edge': truncation.get('roi_edge', 0),
        },
        'length_distribution': _distribution([r['length'] for r in complete_particles], bins),
        'hole_length_distribution': _distribution([r['length'] for r in holes if r['complete']], bins),
        'complete_neighbour_pairs': [
            {'start_phase': key.split('|')[0], 'end_phase': key.split('|')[1], 'count': count}
            for key, count in sorted(neighbour_pairs.items(), key=lambda item: -item[1])
        ],
        'per_line': per_line,
        'intercepts': intercepts,
        'definitions': {
            'particle': '纳入统计的颗粒实例内的截段',
            'void': '分析区域内的基体/背景截段',
            'hole': '不与分析区域或图像边界连通的背景连通域，即封闭孔洞',
            'excluded': '位于测量筛选排除颗粒内（含其内部孔洞）的截段，不进入长度分布',
            'complete': '两端均为相变；端点落在分析区域或图像边界均判为不完整截段。相邻颗粒直接接触时按实例编号切开，不合并为一段',
        },
        'warning': '测线仅读取当前分割结果，不修改掩膜、分析参数或已有分析记录；长度分布只统计两端完整的颗粒截段，图像边界与ROI边界截断分开计数。',
    }


def intercept_csv(result):
    rows = result.get('intercepts', [])
    header = [
        'id', 'line', 'phase', 'category', 'label_id',
        'start_x', 'start_y', 'end_x', 'end_y',
        'length_px', 'length', 'complete', 'start_boundary', 'end_boundary',
    ]
    lines = [','.join(header)]
    for row in rows:
        category = row['phase']
        if row['phase'] == 'particle':
            category = 'complete' if row['complete'] else 'truncated'
        values = [
            row['id'], row['line'], row['phase'], category,
            '' if row['label_id'] is None else row['label_id'],
            row['start'][0], row['start'][1], row['end'][0], row['end'][1],
            row['length_px'], row['length'], 'yes' if row['complete'] else 'no',
            row['start_boundary'], row['end_boundary'],
        ]
        lines.append(','.join(str(v) for v in values))
    return ('﻿' + '\n'.join(lines) + '\n').encode('utf8')


def intercept_image(image, result, font_factory=None):
    from .rendering import font as default_font

    text_font = font_factory or default_font
    output = image.convert('RGB').copy()
    draw = ImageDraw.Draw(output, 'RGBA')
    width, height = output.size
    line_width = max(1, round(width / 900))
    marker = max(2, round(width / 500))
    geometry = result['geometry']
    shape = {
        'center': (geometry['center'][0], geometry['center'][1]),
        'direction': (geometry['direction'][0], geometry['direction'][1]),
        'normal': (geometry['normal'][0], geometry['normal'][1]),
    }
    for entry in result.get('per_line', []):
        start = line_point(shape, entry['position'], geometry['t_min'])
        end = line_point(shape, entry['position'], geometry['t_max'])
        draw.line([start, end], fill=(255, 255, 255, 45), width=line_width)
    colors = {
        'complete': (86, 232, 190, 240),
        'truncated': (248, 176, 72, 240),
        'hole': (238, 111, 191, 240),
        'excluded': (168, 178, 188, 220),
        'void': (120, 170, 210, 120),
    }
    for row in result.get('intercepts', []):
        if row['phase'] == 'void':
            continue
        if row['phase'] == 'particle':
            key = 'complete' if row['complete'] else 'truncated'
        else:
            key = row['phase']
        color = colors.get(key, colors['excluded'])
        draw.line([tuple(row['start']), tuple(row['end'])], fill=color, width=line_width)
        for point in (row['start'], row['end']):
            x, y = point
            draw.ellipse((x - marker, y - marker, x + marker, y + marker), fill=color)
    summary = result['summary']
    entries = [
        ('Complete particle', summary['particle']['count'], colors['complete']),
        ('Truncated particle', summary['particle_truncated']['count'], colors['truncated']),
        ('Hole', summary['hole']['count'], colors['hole']),
        ('Excluded particle', summary['excluded']['count'], colors['excluded']),
    ]
    size = max(12, round(width / 70))
    text = text_font(size)
    x = max(10, round(width / 90))
    y = max(10, round(height / 60))
    line_height = size + 4
    draw.rectangle(
        (x - 8, y - 7, x + max(215, round(width / 4)), y + line_height * (len(entries) + 1) + 4),
        fill=(14, 24, 32, 200),
    )
    draw.text(
        (x, y), f"{result['angle_degrees']:g} deg, {result['line_count']} lines",
        font=text, fill=(240, 248, 250, 255),
    )
    for index, (name, value, color) in enumerate(entries, 1):
        draw.text((x, y + index * line_height), f'{name}: {value}', font=text, fill=color[:3] + (255,))
    return output
