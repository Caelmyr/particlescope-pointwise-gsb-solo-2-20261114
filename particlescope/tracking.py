import math
import uuid
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from . import validation as v
from .storage import ConflictError
from .measurements import round_value
from .rendering import font, label_color


INTERVAL_UNITS = ['frame', 's', 'min', 'h', 'd']
TIME_LABELS = {'frame': '帧', 's': 's', 'min': 'min', 'h': 'h', 'd': 'd'}
MAX_FRAMES = 100
MAX_SEQUENCES = 100
DEFAULT_SETTINGS = {
    'max_distance': 40.0,
    'margin_ratio': 1.35,
    'margin_abs': 3.0,
    'gap_enabled': True,
    'gap_factor': 2.0,
    'border_margin': 20.0,
}
TRACK_CSV_FIELDS = [
    'track_id', 'frame', 'time', 'particle_id', 'x_px', 'y_px', 'x', 'y',
    'step_vx', 'step_vy', 'step_speed', 'space_unit', 'time_unit',
]


def settings(body):
    body = v.mapping(body, '跟踪参数')
    return {
        'max_distance': v.real(body.get('max_distance', DEFAULT_SETTINGS['max_distance']), '帧间最大位移', 0.1, 5000),
        'margin_ratio': v.real(body.get('margin_ratio', DEFAULT_SETTINGS['margin_ratio']), '歧义判定比例', 1, 3),
        'margin_abs': v.real(body.get('margin_abs', DEFAULT_SETTINGS['margin_abs']), '歧义判定余量', 0, 100),
        'gap_enabled': v.boolean(body.get('gap_enabled', DEFAULT_SETTINGS['gap_enabled']), '允许跨帧候选'),
        'gap_factor': v.real(body.get('gap_factor', DEFAULT_SETTINGS['gap_factor']), '跨帧距离倍数', 1, 4),
        'border_margin': v.real(body.get('border_margin', DEFAULT_SETTINGS['border_margin']), '边缘判定宽度', 0, 500),
    }


def positions(rows):
    ids = sorted(rows)
    if not ids:
        return ids, np.zeros((0, 2))
    points = np.array([[rows[i]['centroid_x'], rows[i]['centroid_y']] for i in ids], dtype=float)
    return ids, points


def pair_candidates(source, target, max_distance):
    if not source or not target:
        return []
    source_ids, source_points = positions(source)
    target_ids, target_points = positions(target)
    tree = cKDTree(target_points)
    pairs = []
    for index, point in enumerate(source_points):
        for hit in tree.query_ball_point(point, max_distance):
            distance = float(np.hypot(*(point - target_points[hit])))
            pairs.append((round(distance, 6), source_ids[index], target_ids[hit]))
    pairs.sort()
    return pairs


def auto_links_for_pair(source, target, rejected, tracking_settings):
    pairs = [
        (d, a, b) for d, a, b in pair_candidates(source, target, tracking_settings['max_distance'])
        if (a, b) not in rejected
    ]
    by_source = {}
    by_target = {}
    for distance, a, b in pairs:
        by_source.setdefault(a, []).append((distance, b))
        by_target.setdefault(b, []).append((distance, a))
    matched = {}
    used_targets = set()
    for distance, a, b in pairs:
        if a in matched or b in used_targets:
            continue
        matched[a] = (b, distance)
        used_targets.add(b)
    links = []
    for a, (b, distance) in matched.items():
        limit = distance * tracking_settings['margin_ratio'] + tracking_settings['margin_abs']
        rivals = [d for d, other in by_source[a] if other != b]
        competitors = [d for d, other in by_target[b] if other != a]
        if (rivals and rivals[0] <= limit) or (competitors and competitors[0] <= limit):
            continue
        links.append((a, b))
    return links


def near_border(row, meta, margin):
    if margin <= 0:
        return False
    x, y = row['centroid_x'], row['centroid_y']
    return x <= margin or y <= margin or x >= meta['width'] - 1 - margin or y >= meta['height'] - 1 - margin


def dashed_line(draw, start, end, fill, width=2, dash=8, gap=6):
    x0, y0 = start
    x1, y1 = end
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1:
        return
    dx, dy = (x1 - x0) / length, (y1 - y0) / length
    position = 0.0
    while position < length:
        stop = min(position + dash, length)
        draw.line((x0 + dx * position, y0 + dy * position, x0 + dx * stop, y0 + dy * stop), fill=fill, width=width)
        position = stop + gap


def dashed_circle(draw, center, radius, fill, width=2, segments=28):
    points = []
    for k in range(segments + 1):
        angle = 2 * math.pi * k / segments
        points.append((center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)))
    for k in range(0, segments, 2):
        draw.line((points[k][0], points[k][1], points[k + 1][0], points[k + 1][1]), fill=fill, width=width)


class TrackingService:
    def __init__(self, store):
        self.store = store

    def path(self, sequence_id):
        v.identifier(sequence_id, '时序编号')
        return self.store.sequences / (sequence_id + '.json')

    def load(self, sequence_id):
        sequence = self.store.json_read(self.path(sequence_id))
        if sequence is None:
            raise v.ValidationError('时序不存在')
        return sequence

    def save(self, sequence):
        sequence['updated_at'] = v.now()
        self.store.json_write(self.path(sequence['id']), sequence)

    def check_revision(self, sequence, revision):
        if revision is None:
            raise v.ValidationError('缺少时序修订号，请刷新页面')
        if v.integer(revision, '修订号', 0) != sequence['revision']:
            raise ConflictError('时序已被其他操作修改，请刷新后重试')

    def bump(self, sequence):
        sequence['revision'] += 1

    def list(self):
        with self.store.lock:
            result = []
            for path in self.store.sequences.glob('*.json'):
                sequence = self.store.json_read(path)
                if sequence is None:
                    continue
                result.append({
                    'id': sequence['id'],
                    'name': sequence['name'],
                    'frame_count': len(sequence['frames']),
                    'link_count': len(sequence['links']),
                    'track_count': len(sequence.get('tracks', [])),
                    'interval': sequence['interval'],
                    'interval_unit': sequence['interval_unit'],
                    'created_at': sequence['created_at'],
                    'updated_at': sequence['updated_at'],
                })
            return sorted(result, key=lambda s: s['updated_at'], reverse=True)

    def frame_data(self, sequence):
        frames = []
        for frame in sequence['frames']:
            meta = self.store.meta(frame['sample_id'])
            result = self.store.result(frame['sample_id'], frame['analysis_id'])
            particles = {row['id']: row for row in result['particles'] if row.get('included', True)}
            frames.append({'meta': meta, 'result': result, 'particles': particles})
        return frames

    def calibration(self, frames):
        first = frames[0]['result']['calibration']
        uniform = all(f['result']['calibration'] == first for f in frames)
        return first, uniform

    def create(self, body):
        body = v.mapping(body, '时序参数')
        name = v.text(body.get('name', ''), '时序名称', 180, False)
        sample_ids = [v.identifier(s, '样本编号') for s in v.sequence(body.get('sample_ids'), '样本列表', MAX_FRAMES)]
        if len(sample_ids) < 2:
            raise v.ValidationError('时序至少需要两个样本')
        interval = v.real(body.get('interval', 1), '拍摄间隔', 0.000001, 1000000000)
        interval_unit = v.choice(body.get('interval_unit', 'min'), '时间单位', INTERVAL_UNITS)
        tracking_settings = settings(body.get('settings', {}))
        with self.store.lock:
            if len(self.list()) >= MAX_SEQUENCES:
                raise v.ValidationError(f'时序数量最多{MAX_SEQUENCES}个')
            frames = []
            size = None
            for index, sample_id in enumerate(sample_ids):
                meta = self.store.meta(sample_id)
                if not meta.get('analysis_id'):
                    raise v.ValidationError(f'样本「{meta["name"]}」还没有分析结果，请先运行图像分析')
                result = self.store.result(sample_id)
                if result.get('stale'):
                    raise v.ValidationError(f'样本「{meta["name"]}」的分析对应旧图像，请重新分析后再加入时序')
                if size is None:
                    size = (meta['width'], meta['height'])
                elif (meta['width'], meta['height']) != size:
                    raise v.ValidationError('所有帧的图像尺寸必须一致（同一视野）')
                frames.append({
                    'sample_id': sample_id,
                    'analysis_id': meta['analysis_id'],
                    'name': meta['name'],
                    'time': round(index * interval, 9),
                })
            sequence = {
                'id': uuid.uuid4().hex,
                'name': name,
                'created_at': v.now(),
                'updated_at': v.now(),
                'revision': 0,
                'interval': interval,
                'interval_unit': interval_unit,
                'frames': frames,
                'settings': tracking_settings,
                'links': [],
                'rejected': [],
                'tracks': [],
                'next_track': 1,
            }
            data = self.frame_data(sequence)
            for index in range(len(data) - 1):
                for a, b in auto_links_for_pair(data[index]['particles'], data[index + 1]['particles'], set(), tracking_settings):
                    sequence['links'].append({
                        'from_frame': index, 'from_id': a,
                        'to_frame': index + 1, 'to_id': b,
                        'source': 'auto', 'status': 'auto',
                    })
            self.recompute(sequence, data)
            self.save(sequence)
            return self.state(sequence, data)

    def update(self, sequence_id, body):
        body = v.mapping(body, '时序设置')
        with self.store.lock:
            sequence = self.load(sequence_id)
            self.check_revision(sequence, body.get('revision'))
            if 'name' in body:
                sequence['name'] = v.text(body['name'], '时序名称', 180, False)
            interval_changed = False
            if 'interval' in body:
                sequence['interval'] = v.real(body['interval'], '拍摄间隔', 0.000001, 1000000000)
                interval_changed = True
            if 'interval_unit' in body:
                sequence['interval_unit'] = v.choice(body['interval_unit'], '时间单位', INTERVAL_UNITS)
            if interval_changed:
                for index, frame in enumerate(sequence['frames']):
                    frame['time'] = round(index * sequence['interval'], 9)
            settings_changed = 'settings' in body
            if settings_changed:
                sequence['settings'] = settings(body['settings'])
            data = self.frame_data(sequence)
            if settings_changed:
                self.relink_auto(sequence, data)
            affected, retired = self.recompute(sequence, data)
            self.bump(sequence)
            self.save(sequence)
            return self.state(sequence, data, {'changed': affected, 'retired': retired})

    def delete(self, sequence_id, body):
        body = v.mapping(body or {}, '时序设置')
        with self.store.lock:
            sequence = self.load(sequence_id)
            self.check_revision(sequence, body.get('revision'))
            self.path(sequence_id).unlink()
            return {'deleted': sequence_id}

    def relink_auto(self, sequence, data):
        protected = [l for l in sequence['links'] if l['source'] == 'manual' or l['status'] == 'confirmed']
        links = list(protected)
        for index in range(len(data) - 1):
            outgoing = {l['from_id'] for l in links if l['from_frame'] == index}
            incoming = {l['to_id'] for l in links if l['to_frame'] == index + 1}
            source = {pid: row for pid, row in data[index]['particles'].items() if pid not in outgoing}
            target = {pid: row for pid, row in data[index + 1]['particles'].items() if pid not in incoming}
            rejected = {
                (r['from_id'], r['to_id']) for r in sequence['rejected']
                if r['from_frame'] == index and r['to_frame'] == index + 1
            }
            for a, b in auto_links_for_pair(source, target, rejected, sequence['settings']):
                links.append({
                    'from_frame': index, 'from_id': a,
                    'to_frame': index + 1, 'to_id': b,
                    'source': 'auto', 'status': 'auto',
                })
        sequence['links'] = links

    def relink(self, sequence_id, revision):
        with self.store.lock:
            sequence = self.load(sequence_id)
            self.check_revision(sequence, revision)
            data = self.frame_data(sequence)
            self.relink_auto(sequence, data)
            affected, retired = self.recompute(sequence, data)
            self.bump(sequence)
            self.save(sequence)
            return self.state(sequence, data, {'changed': affected, 'retired': retired})

    def find_link(self, sequence, from_frame, from_id):
        return next(
            (l for l in sequence['links'] if l['from_frame'] == from_frame and l['from_id'] == from_id),
            None,
        )

    def add_link(self, sequence_id, body):
        body = v.mapping(body, '手动关联')
        with self.store.lock:
            sequence = self.load(sequence_id)
            self.check_revision(sequence, body.get('revision'))
            count = len(sequence['frames'])
            from_frame = v.integer(body.get('from_frame'), '起始帧', 0, count - 2)
            to_frame = v.integer(body.get('to_frame', from_frame + 1), '目标帧', from_frame + 1, count - 1)
            if to_frame - from_frame > 2:
                raise v.ValidationError('只能关联相邻帧或隔一帧的颗粒')
            from_id = v.integer(body.get('from_id'), '起始颗粒编号', 1)
            to_id = v.integer(body.get('to_id'), '目标颗粒编号', 1)
            data = self.frame_data(sequence)
            if from_id not in data[from_frame]['particles']:
                raise v.ValidationError('起始颗粒不存在或在该帧分析中已被排除')
            if to_id not in data[to_frame]['particles']:
                raise v.ValidationError('目标颗粒不存在或在该帧分析中已被排除')
            if self.find_link(sequence, from_frame, from_id):
                raise ConflictError('该颗粒已有出向关联，请先删除原关联')
            if any(l['to_frame'] == to_frame and l['to_id'] == to_id for l in sequence['links']):
                raise ConflictError('目标颗粒已有入向关联，请先删除原关联')
            sequence['rejected'] = [
                r for r in sequence['rejected']
                if not (r['from_frame'] == from_frame and r['from_id'] == from_id
                        and r['to_frame'] == to_frame and r['to_id'] == to_id)
            ]
            sequence['links'].append({
                'from_frame': from_frame, 'from_id': from_id,
                'to_frame': to_frame, 'to_id': to_id,
                'source': 'manual', 'status': 'confirmed',
            })
            affected, retired = self.recompute(sequence, data)
            self.bump(sequence)
            self.save(sequence)
            return self.state(sequence, data, {'changed': affected, 'retired': retired})

    def remove_link(self, sequence_id, body):
        body = v.mapping(body, '删除关联')
        with self.store.lock:
            sequence = self.load(sequence_id)
            self.check_revision(sequence, body.get('revision'))
            from_frame = v.integer(body.get('from_frame'), '起始帧', 0, len(sequence['frames']) - 1)
            from_id = v.integer(body.get('from_id'), '起始颗粒编号', 1)
            link = self.find_link(sequence, from_frame, from_id)
            if link is None:
                raise v.ValidationError('该颗粒没有出向关联')
            sequence['links'].remove(link)
            data = self.frame_data(sequence)
            affected, retired = self.recompute(sequence, data)
            self.bump(sequence)
            self.save(sequence)
            return self.state(sequence, data, {'changed': affected, 'retired': retired})

    def confirm_link(self, sequence_id, body):
        body = v.mapping(body, '确认关联')
        with self.store.lock:
            sequence = self.load(sequence_id)
            self.check_revision(sequence, body.get('revision'))
            from_frame = v.integer(body.get('from_frame'), '起始帧', 0, len(sequence['frames']) - 1)
            from_id = v.integer(body.get('from_id'), '起始颗粒编号', 1)
            link = self.find_link(sequence, from_frame, from_id)
            if link is None:
                raise v.ValidationError('该颗粒没有出向关联')
            link['status'] = 'confirmed'
            data = self.frame_data(sequence)
            self.bump(sequence)
            self.save(sequence)
            return self.state(sequence, data)

    def resolve(self, sequence_id, key, body):
        body = v.mapping(body, '待定处理')
        with self.store.lock:
            sequence = self.load(sequence_id)
            self.check_revision(sequence, body.get('revision'))
            data = self.frame_data(sequence)
            pending, _ = self.pending(sequence, data)
            item = next((p for p in pending if p['key'] == key), None)
            if item is None:
                raise v.ValidationError('该待定对应已处理或跟踪参数已变化，请刷新')
            action = v.choice(body.get('action'), '处理方式', ['link', 'reject'])
            if action == 'link':
                to_id = v.integer(body.get('to_id'), '目标颗粒编号', 1)
                if to_id not in [c['to_id'] for c in item['candidates']]:
                    raise v.ValidationError('目标不在候选列表中；如需关联其他颗粒请使用手动关联')
                sequence['links'].append({
                    'from_frame': item['frame'], 'from_id': item['from_id'],
                    'to_frame': item['to_frame'], 'to_id': to_id,
                    'source': 'manual', 'status': 'confirmed',
                })
            else:
                for candidate in item['candidates']:
                    pair = {
                        'from_frame': item['frame'], 'from_id': item['from_id'],
                        'to_frame': item['to_frame'], 'to_id': candidate['to_id'],
                    }
                    if pair not in sequence['rejected']:
                        sequence['rejected'].append(pair)
            affected, retired = self.recompute(sequence, data)
            self.bump(sequence)
            self.save(sequence)
            return self.state(sequence, data, {'changed': affected, 'retired': retired})

    def clear_rejections(self, sequence_id, body):
        body = v.mapping(body or {}, '清除否决')
        with self.store.lock:
            sequence = self.load(sequence_id)
            self.check_revision(sequence, body.get('revision'))
            sequence['rejected'] = []
            data = self.frame_data(sequence)
            self.bump(sequence)
            self.save(sequence)
            return self.state(sequence, data)

    def rebind(self, sequence_id, index, body):
        body = v.mapping(body or {}, '重新绑定')
        with self.store.lock:
            sequence = self.load(sequence_id)
            self.check_revision(sequence, body.get('revision'))
            if not 0 <= index < len(sequence['frames']):
                raise v.ValidationError('帧序号超出范围')
            frame = sequence['frames'][index]
            meta = self.store.meta(frame['sample_id'])
            if not meta.get('analysis_id'):
                raise v.ValidationError('样本当前没有分析结果，请先运行图像分析')
            result = self.store.result(frame['sample_id'])
            if result.get('stale'):
                raise v.ValidationError('样本当前分析对应旧图像，请重新分析')
            others = [f for i, f in enumerate(sequence['frames']) if i != index]
            if others:
                reference = self.store.meta(others[0]['sample_id'])
                if (meta['width'], meta['height']) != (reference['width'], reference['height']):
                    raise v.ValidationError('重新绑定后图像尺寸与其他帧不一致')
            frame['analysis_id'] = meta['analysis_id']
            frame['name'] = meta['name']
            sequence['links'] = [
                l for l in sequence['links']
                if l['from_frame'] != index and l['to_frame'] != index
            ]
            sequence['rejected'] = [
                r for r in sequence['rejected']
                if r['from_frame'] != index and r['to_frame'] != index
            ]
            data = self.frame_data(sequence)
            affected, retired = self.recompute(sequence, data)
            self.bump(sequence)
            self.save(sequence)
            return self.state(sequence, data, {'changed': affected, 'retired': retired})

    def pending(self, sequence, data):
        tracking_settings = sequence['settings']
        links = sequence['links']
        rejected = sequence['rejected']
        count = len(data)
        items = []
        candidate_targets = {i: set() for i in range(count)}
        unlinked = {i: [] for i in range(count)}
        for index in range(count - 1):
            outgoing = {l['from_id'] for l in links if l['from_frame'] == index}
            next_incoming = {l['to_id'] for l in links if l['to_frame'] == index + 1}
            free_next = {pid: row for pid, row in data[index + 1]['particles'].items() if pid not in next_incoming}
            scoped = {
                (r['from_id'], r['to_id']) for r in rejected
                if r['from_frame'] == index and r['to_frame'] == index + 1
            }
            gap_scoped = {
                (r['from_id'], r['to_id']) for r in rejected
                if r['from_frame'] == index and r['to_frame'] == index + 2
            }
            for pid, row in data[index]['particles'].items():
                if pid in outgoing:
                    continue
                candidates = [
                    (d, b) for d, a, b in pair_candidates({pid: row}, free_next, tracking_settings['max_distance'])
                    if (pid, b) not in scoped
                ]
                if candidates:
                    candidate_targets[index + 1].update(b for _, b in candidates)
                    items.append(self.pending_item(
                        'ambiguous', index, index + 1, pid, row, candidates,
                        data[index + 1]['particles'], '相邻帧存在多个相近候选，未自动连线',
                    ))
                    continue
                if tracking_settings['gap_enabled'] and index + 2 < count:
                    gap_incoming = {l['to_id'] for l in links if l['to_frame'] == index + 2}
                    free_gap = {p: r for p, r in data[index + 2]['particles'].items() if p not in gap_incoming}
                    gap_candidates = [
                        (d, b) for d, a, b in pair_candidates(
                            {pid: row}, free_gap,
                            tracking_settings['max_distance'] * tracking_settings['gap_factor'],
                        )
                        if (pid, b) not in gap_scoped
                    ]
                    if gap_candidates:
                        candidate_targets[index + 2].update(b for _, b in gap_candidates)
                        items.append(self.pending_item(
                            'gap', index, index + 2, pid, row, gap_candidates,
                            data[index + 2]['particles'], '下一帧没有候选，疑似暂时漏检，未自动跨帧连线',
                        ))
                        continue
                unlinked[index].append(pid)
        events = []
        for index in range(count - 1):
            for pid in unlinked[index]:
                row = data[index]['particles'][pid]
                kind = 'left_frame' if near_border(row, data[index]['meta'], tracking_settings['border_margin']) else 'ended'
                events.append({'frame': index, 'particle_id': pid, 'kind': kind})
        for index in range(1, count):
            incoming = {l['to_id'] for l in links if l['to_frame'] == index}
            for pid, row in data[index]['particles'].items():
                if pid in incoming or pid in candidate_targets[index]:
                    continue
                kind = 'entered_frame' if near_border(row, data[index]['meta'], tracking_settings['border_margin']) else 'started'
                events.append({'frame': index, 'particle_id': pid, 'kind': kind})
        if tracking_settings['border_margin'] > 0:
            gate = tracking_settings['max_distance'] * tracking_settings['gap_factor']
            for index in range(count - 1):
                leavers = [e['particle_id'] for e in events if e['frame'] == index and e['kind'] == 'left_frame']
                entrants = [e['particle_id'] for e in events if e['frame'] == index + 1 and e['kind'] == 'entered_frame']
                if not leavers or not entrants:
                    continue
                scoped = {
                    (r['from_id'], r['to_id']) for r in rejected
                    if r['from_frame'] == index and r['to_frame'] == index + 1
                }
                for pid in leavers:
                    row = data[index]['particles'][pid]
                    candidates = []
                    for bid in entrants:
                        target = data[index + 1]['particles'][bid]
                        if (pid, bid) in scoped:
                            continue
                        distance = math.hypot(
                            row['centroid_x'] - target['centroid_x'],
                            row['centroid_y'] - target['centroid_y'],
                        )
                        if distance <= gate:
                            candidates.append((round(distance, 6), bid))
                    candidates.sort()
                    if candidates:
                        items.append(self.pending_item(
                            'border', index, index + 1, pid, row, candidates,
                            data[index + 1]['particles'], '颗粒在视野边缘消失且附近有新颗粒出现，未自动连线',
                        ))
        items.sort(key=lambda p: (p['frame'], p['from_id']))
        events.sort(key=lambda e: (e['frame'], e['particle_id']))
        return items, events

    def pending_item(self, kind, frame, to_frame, pid, row, candidates, target_rows, reason):
        prefix = {'ambiguous': 'a', 'gap': 'g', 'border': 'b'}[kind]
        return {
            'key': f'{prefix}{frame}:{pid}',
            'kind': kind,
            'frame': frame,
            'to_frame': to_frame,
            'from_id': pid,
            'position': [row['centroid_x'], row['centroid_y']],
            'candidates': [
                {
                    'to_id': b,
                    'distance': d,
                    'position': [target_rows[b]['centroid_x'], target_rows[b]['centroid_y']],
                }
                for d, b in candidates[:5]
            ],
            'reason': reason,
        }

    def recompute(self, sequence, data):
        count = len(data)
        outgoing = {}
        incoming = {}
        for link in sequence['links']:
            outgoing[(link['from_frame'], link['from_id'])] = (link['to_frame'], link['to_id'])
            incoming[(link['to_frame'], link['to_id'])] = (link['from_frame'], link['from_id'])
        chains = []
        visited = set()
        for index in range(count):
            for pid in data[index]['particles']:
                node = (index, pid)
                if node in visited:
                    continue
                start = node
                while start in incoming:
                    start = incoming[start]
                chain = []
                current = start
                while True:
                    chain.append(current)
                    visited.add(current)
                    if current not in outgoing:
                        break
                    current = outgoing[current]
                chains.append(chain)
        chains.sort(key=lambda c: c[0])
        old = {t['id']: {tuple(n) for n in t['nodes']} for t in sequence.get('tracks', [])}
        overlaps = []
        for ci, chain in enumerate(chains):
            nodes = set(chain)
            for tid, old_nodes in old.items():
                shared = len(nodes & old_nodes)
                if shared:
                    overlaps.append((-shared, ci, tid))
        overlaps.sort()
        assigned = {}
        taken = set()
        for _, ci, tid in overlaps:
            if ci in assigned or tid in taken:
                continue
            assigned[ci] = tid
            taken.add(tid)
        next_number = sequence.get('next_track', 1)
        tracks = []
        affected = []
        for ci, chain in enumerate(chains):
            tid = assigned.get(ci)
            if tid is None:
                tid = f'T{next_number}'
                next_number += 1
                affected.append(tid)
            elif old[tid] != set(chain):
                affected.append(tid)
            tracks.append({'id': tid, 'nodes': [[f, p] for f, p in chain]})
        retired = sorted(tid for tid in old if tid not in taken)
        sequence['tracks'] = tracks
        sequence['next_track'] = next_number
        return sorted(affected), retired

    def track_payload(self, track, data, times, scale):
        points = []
        for frame_index, pid in track['nodes']:
            row = data[frame_index]['particles'].get(pid)
            if row is None:
                continue
            points.append({
                'frame': frame_index,
                'particle_id': pid,
                'x': row['centroid_x'],
                'y': row['centroid_y'],
                'time': times[frame_index],
            })
        steps = []
        for k in range(len(points) - 1):
            a, b = points[k], points[k + 1]
            dt = b['time'] - a['time']
            dx = (b['x'] - a['x']) * scale
            dy = (b['y'] - a['y']) * scale
            distance = math.hypot(dx, dy)
            steps.append({
                'from_frame': a['frame'],
                'to_frame': b['frame'],
                'dt': round_value(dt),
                'distance': round_value(distance),
                'vx': round_value(dx / dt) if dt > 0 else None,
                'vy': round_value(dy / dt) if dt > 0 else None,
                'speed': round_value(distance / dt) if dt > 0 else None,
            })
        for k, point in enumerate(points):
            point['speed'] = steps[k]['speed'] if k < len(steps) else None
        path = sum(s['distance'] for s in steps)
        duration = points[-1]['time'] - points[0]['time'] if len(points) > 1 else 0
        net = 0.0
        if len(points) > 1:
            net = math.hypot(
                (points[-1]['x'] - points[0]['x']) * scale,
                (points[-1]['y'] - points[0]['y']) * scale,
            )
        speeds = [s['speed'] for s in steps if s['speed'] is not None]
        return {
            'id': track['id'],
            'start_frame': points[0]['frame'] if points else None,
            'end_frame': points[-1]['frame'] if points else None,
            'point_count': len(points),
            'gap_count': sum(1 for s in steps if s['to_frame'] - s['from_frame'] > 1),
            'duration': round_value(duration),
            'path_length': round_value(path),
            'net_displacement': round_value(net),
            'mean_speed': round_value(path / duration) if duration > 0 else None,
            'max_speed': round_value(max(speeds)) if speeds else None,
            'mean_vx': round_value(sum(s['vx'] for s in steps) / len(steps)) if steps else None,
            'mean_vy': round_value(sum(s['vy'] for s in steps) / len(steps)) if steps else None,
            'points': points,
            'steps': steps,
        }

    def state(self, sequence, data=None, affected=None):
        data = data or self.frame_data(sequence)
        calibration, uniform = self.calibration(data)
        scale = calibration['scale'] if uniform else 1.0
        space_unit = calibration['unit'] if uniform else 'px'
        time_unit = TIME_LABELS[sequence['interval_unit']]
        times = [f['time'] for f in sequence['frames']]
        pending, events = self.pending(sequence, data)
        tracks = [self.track_payload(t, data, times, scale) for t in sequence.get('tracks', [])]
        all_speeds = [s['speed'] for t in tracks for s in t['steps'] if s['speed'] is not None]
        frames = []
        for index, frame in enumerate(sequence['frames']):
            meta = data[index]['meta']
            result = data[index]['result']
            frames.append({
                'index': index,
                'sample_id': frame['sample_id'],
                'analysis_id': frame['analysis_id'],
                'name': meta['name'],
                'time': frame['time'],
                'particle_count': len(data[index]['particles']),
                'image_mismatch': meta['image_revision'] != result['image_revision'],
                'is_current_analysis': meta['analysis_id'] == frame['analysis_id'],
            })
        particle_total = sum(len(data[i]['particles']) for i in range(len(data)))
        linked_particles = sum(t['point_count'] for t in tracks if t['point_count'] > 1)
        state = {
            'id': sequence['id'],
            'name': sequence['name'],
            'revision': sequence['revision'],
            'created_at': sequence['created_at'],
            'updated_at': sequence['updated_at'],
            'interval': sequence['interval'],
            'interval_unit': sequence['interval_unit'],
            'settings': sequence['settings'],
            'frames': frames,
            'links': sequence['links'],
            'rejected': sequence['rejected'],
            'pending': pending,
            'events': events,
            'tracks': tracks,
            'space_unit': space_unit,
            'time_unit': time_unit,
            'velocity_unit': f'{space_unit}/{time_unit}',
            'uniform_calibration': uniform,
            'summary': {
                'frame_count': len(frames),
                'particle_total': particle_total,
                'track_count': len(tracks),
                'singleton_tracks': sum(1 for t in tracks if t['point_count'] == 1),
                'complete_tracks': sum(1 for t in tracks if t['start_frame'] == 0 and t['end_frame'] == len(frames) - 1),
                'linked_particles': linked_particles,
                'pending_count': len(pending),
                'mean_track_length': round_value(np.mean([t['point_count'] for t in tracks])) if tracks else 0,
                'mean_speed': round_value(np.mean(all_speeds)) if all_speeds else None,
                'max_speed': round_value(max(all_speeds)) if all_speeds else None,
                'calibration_warning': None if uniform else '各帧标定不一致，位移与速度以像素为单位',
            },
        }
        if affected is not None:
            state['affected'] = affected
        return state

    def pick(self, sequence_id, frame_index, x, y):
        sequence = self.load(sequence_id)
        if not 0 <= frame_index < len(sequence['frames']):
            raise v.ValidationError('帧序号超出范围')
        frame = sequence['frames'][frame_index]
        result = self.store.result(frame['sample_id'], frame['analysis_id'])
        path = self.store.sample_dir(frame['sample_id']) / 'analyses' / result['id'] / 'labels.npy'
        if not path.exists():
            raise v.ValidationError('该帧分析掩膜缺失')
        labels = np.load(path, allow_pickle=False)
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < labels.shape[1] and 0 <= yi < labels.shape[0]):
            raise v.ValidationError('坐标超出图像范围')
        pid = int(labels[yi, xi])
        if pid == 0:
            return None
        return next((r for r in result['particles'] if r['id'] == pid), None)

    def overlay(self, sequence_id, frame_index, track_filter=None):
        sequence = self.load(sequence_id)
        if not 0 <= frame_index < len(sequence['frames']):
            raise v.ValidationError('帧序号超出范围')
        data = self.frame_data(sequence)
        frame = sequence['frames'][frame_index]
        image = self.store.working_image(frame['sample_id']).convert('RGBA')
        draw = ImageDraw.Draw(image, 'RGBA')
        calibration, uniform = self.calibration(data)
        scale = calibration['scale'] if uniform else 1.0
        times = [f['time'] for f in sequence['frames']]
        tracks = [self.track_payload(t, data, times, scale) for t in sequence.get('tracks', [])]
        text_font = font(max(11, image.width // 90))
        for track in tracks:
            number = int(track['id'][1:])
            color = label_color(number)
            emphasized = track_filter is None or track['id'] == track_filter
            alpha = 210 if emphasized else 45
            width = 3 if emphasized else 1
            pts = [(p['x'], p['y']) for p in track['points']]
            if track_filter and not emphasized:
                pts = []
            if len(pts) > 1:
                draw.line(pts, fill=color + (alpha,), width=width)
            for point in track['points']:
                x, y = point['x'], point['y']
                if point['frame'] == frame_index and emphasized:
                    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color + (230,), outline=(255, 255, 255, 255))
                    draw.text((x + 8, y - 8), track['id'], font=text_font, fill=(255, 255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0, 255))
                elif not track_filter or emphasized:
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color + (alpha,))
        pending, _ = self.pending(sequence, data)
        amber = (255, 196, 66, 255)
        for item in pending:
            if item['frame'] != frame_index:
                continue
            x, y = item['position']
            dashed_circle(draw, (x, y), 14, amber, width=3)
            draw.text((x + 16, y + 6), '?', font=text_font, fill=amber, stroke_width=1, stroke_fill=(0, 0, 0, 255))
            for candidate in item['candidates']:
                cx, cy = candidate['position']
                dashed_line(draw, (x, y), (cx, cy), (255, 196, 66, 170), width=2)
                dashed_circle(draw, (cx, cy), 9, (255, 196, 66, 220), width=2)
                draw.text((cx + 10, cy + 4), f"#{candidate['to_id']}", font=text_font, fill=amber, stroke_width=1, stroke_fill=(0, 0, 0, 255))
        return image.convert('RGB')

    def tracks_csv(self, sequence_id):
        from .exports import csv_bytes
        sequence = self.load(sequence_id)
        state = self.state(sequence)
        scale = None
        if state['uniform_calibration']:
            data = self.frame_data(sequence)
            scale = data[0]['result']['calibration']['scale']
        rows = []
        for track in state['tracks']:
            for k, point in enumerate(track['points']):
                step = track['steps'][k] if k < len(track['steps']) else None
                factor = scale if scale is not None else 1.0
                rows.append({
                    'track_id': track['id'],
                    'frame': point['frame'],
                    'time': point['time'],
                    'particle_id': point['particle_id'],
                    'x_px': point['x'],
                    'y_px': point['y'],
                    'x': round_value(point['x'] * factor),
                    'y': round_value(point['y'] * factor),
                    'step_vx': step['vx'] if step else None,
                    'step_vy': step['vy'] if step else None,
                    'step_speed': step['speed'] if step else None,
                    'space_unit': state['space_unit'],
                    'time_unit': state['time_unit'],
                })
        return csv_bytes(rows, TRACK_CSV_FIELDS)
