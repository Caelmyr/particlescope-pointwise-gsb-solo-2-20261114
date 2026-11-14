import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from .measurements import included_rows, round_value
from .validation import ValidationError

MAX_CANDIDATE_PAIRS = 200000
TREE_CACHE_LIMIT = 512


def boundary_mask_of(work):
    cross = ndi.generate_binary_structure(2, 1)
    low = ndi.minimum_filter(work, footprint=cross, mode='constant', cval=0)
    high = ndi.maximum_filter(work, footprint=cross, mode='constant', cval=0)
    return (work > 0) & ((low != work) | (high != work))


def particle_boundaries(work, boundary, ids):
    by, bx = np.nonzero(boundary)
    bl = work[by, bx]
    order = np.argsort(bl, kind='stable')
    bl, by, bx = bl[order], by[order], bx[order]
    unique, starts = np.unique(bl, return_index=True)
    wanted = set(ids)
    coords = {}
    for index, pid in enumerate(unique):
        pid = int(pid)
        if pid not in wanted:
            continue
        end = starts[index + 1] if index + 1 < len(starts) else len(bl)
        coords[pid] = np.column_stack((by[starts[index]:end], bx[starts[index]:end])).astype(np.float64)
    return coords


def candidate_pairs(rows, reach):
    boxes = []
    for row in rows:
        x0 = row['bbox_x'] - reach
        y0 = row['bbox_y'] - reach
        x1 = row['bbox_x'] + row['bbox_width'] + reach
        y1 = row['bbox_y'] + row['bbox_height'] + reach
        boxes.append((x0, y0, x1, y1, row['id']))
    boxes.sort()
    active = []
    pairs = set()
    for x0, y0, x1, y1, pid in boxes:
        active = [b for b in active if b[2] >= x0]
        for other in active:
            if other[1] < y1 and other[3] > y0:
                pairs.add((min(pid, other[4]), max(pid, other[4])))
        active.append((x0, y0, x1, y1, pid))
    return sorted(pairs)


def pair_contact_zones(points_a, points_b, threshold, tree_b):
    distances, indices = tree_b.query(points_a, k=1, distance_upper_bound=threshold + 1e-9)
    close = np.isfinite(distances)
    if not np.any(close):
        return None
    pts = points_a[close]
    dists = distances[close]
    matched = points_b[indices[close]]
    y0, x0 = np.floor(pts.min(axis=0)).astype(int) - 1
    y1, x1 = np.ceil(pts.max(axis=0)).astype(int) + 2
    mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    iy = (pts[:, 0] - y0).astype(int)
    ix = (pts[:, 1] - x0).astype(int)
    mask[iy, ix] = True
    zones, count = ndi.label(mask, structure=np.ones((3, 3)))
    membership = zones[iy, ix]
    result = []
    for zone in range(1, count + 1):
        members = membership == zone
        zone_pts = pts[members]
        zone_dists = dists[members]
        zone_matched = matched[members]
        nearest = int(np.argmin(zone_dists))
        pa = zone_pts[nearest]
        pb = zone_matched[nearest]
        midpoints = (zone_pts + zone_matched) / 2
        result.append({
            'gap': max(0.0, float(zone_dists[nearest]) - 1.0),
            'x': float(midpoints[:, 1].mean()),
            'y': float(midpoints[:, 0].mean()),
            'point_a': (float(pa[1]), float(pa[0])),
            'point_b': (float(pb[1]), float(pb[0])),
            'pixels': int(len(zone_pts)),
        })
    return result


def connected_groups(ids, edges):
    parent = {pid: pid for pid in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups = {}
    for pid in ids:
        groups.setdefault(find(pid), []).append(pid)
    clusters = [sorted(members) for members in groups.values() if len(members) > 1]
    clusters.sort(key=lambda members: (-len(members), members[0]))
    return clusters


def contact_network(labels, result, gap_px):
    gap_px = float(gap_px)
    rows = included_rows(result)
    ids = [row['id'] for row in rows]
    scale = result['scale']
    excluded_count = len(result['particles']) - len(ids)
    report = {
        'analysis_id': result['id'],
        'gap_px': round_value(gap_px),
        'unit': result['unit'],
        'scale': scale,
        'included_count': len(ids),
        'excluded_count': excluded_count,
        'pair_count': 0,
        'contact_count': 0,
        'multi_contact_pairs': 0,
        'cluster_count': 0,
        'largest_cluster_size': 0,
        'isolated_count': len(ids),
        'isolated_ids': sorted(ids),
        'contacts': [],
        'pairs': [],
        'adjacency': {},
        'clusters': [],
        'method': '表面间隙=两颗粒边界最近距离−1px：直接相邻为0，间隔一个背景像素为1；间隙≤允许值即判定接触',
        'warning': '仅纳入统计的颗粒参与组网，被筛除颗粒不会形成或传递接触；对角相邻的间隙约为0.41px',
    }
    if len(ids) < 2:
        return report
    keep = np.zeros(int(labels.max()) + 1, dtype=bool)
    keep[ids] = True
    work = labels * keep[labels]
    boundary = boundary_mask_of(work)
    coords = particle_boundaries(work, boundary, ids)
    reach = gap_px + 1.0
    candidates = candidate_pairs(rows, reach)
    if len(candidates) > MAX_CANDIDATE_PAIRS:
        raise ValidationError('候选颗粒对超过200000，请减小允许间隙')
    trees = {}

    def tree_of(pid):
        if pid not in trees:
            if len(trees) >= TREE_CACHE_LIMIT:
                trees.clear()
            trees[pid] = cKDTree(coords[pid])
        return trees[pid]

    adjacency = {pid: set() for pid in ids}
    pair_zones = {}
    contact_entries = []
    for a, b in candidates:
        if a not in coords or b not in coords:
            continue
        zones = pair_contact_zones(coords[a], coords[b], reach, tree_of(b))
        if not zones:
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)
        pair_zones[(a, b)] = []
        for zone in zones:
            pair_zones[(a, b)].append(len(contact_entries))
            contact_entries.append({'a': a, 'b': b, **zone})
    edges = sorted(pair_zones)
    clusters = connected_groups(ids, edges)
    cluster_of = {}
    cluster_entries = []
    for index, members in enumerate(clusters, 1):
        member_set = set(members)
        pair_count = sum(1 for a, _ in edges if a in member_set)
        contact_count = sum(len(pair_zones[edge]) for edge in edges if edge[0] in member_set)
        cluster_entries.append({
            'cluster_id': index,
            'size': len(members),
            'particle_ids': members,
            'pair_count': pair_count,
            'contact_count': contact_count,
        })
        for pid in members:
            cluster_of[pid] = index
    contacts_out = []
    for number, entry in enumerate(contact_entries, 1):
        contacts_out.append({
            'contact_id': number,
            'particle_a': entry['a'],
            'particle_b': entry['b'],
            'gap_px': round_value(entry['gap']),
            'gap_scaled': round_value(entry['gap'] * scale),
            'x': round_value(entry['x']),
            'y': round_value(entry['y']),
            'boundary_pixels': entry['pixels'],
            'point_a': [round_value(entry['point_a'][0]), round_value(entry['point_a'][1])],
            'point_b': [round_value(entry['point_b'][0]), round_value(entry['point_b'][1])],
            'cluster_id': cluster_of.get(entry['a']),
        })
    pairs_out = []
    for edge in edges:
        zone_indices = pair_zones[edge]
        pairs_out.append({
            'particle_a': edge[0],
            'particle_b': edge[1],
            'contact_count': len(zone_indices),
            'min_gap_px': min(contacts_out[i]['gap_px'] for i in zone_indices),
            'contact_ids': [i + 1 for i in zone_indices],
            'cluster_id': cluster_of.get(edge[0]),
        })
    isolated = sorted(pid for pid in ids if not adjacency[pid])
    report.update({
        'pair_count': len(pairs_out),
        'contact_count': len(contacts_out),
        'multi_contact_pairs': sum(1 for p in pairs_out if p['contact_count'] > 1),
        'cluster_count': len(cluster_entries),
        'largest_cluster_size': max((c['size'] for c in cluster_entries), default=0),
        'isolated_count': len(isolated),
        'isolated_ids': isolated,
        'contacts': contacts_out,
        'pairs': pairs_out,
        'adjacency': {str(pid): sorted(adjacency[pid]) for pid in ids},
        'clusters': cluster_entries,
    })
    return report


def contact_csv(report):
    from .exports import csv_bytes
    fields = [
        'contact_id', 'particle_a', 'particle_b', 'gap_px', 'gap_physical',
        'unit', 'x_px', 'y_px', 'boundary_pixels', 'cluster_id',
    ]
    rows = []
    for contact in report['contacts']:
        rows.append({
            'contact_id': contact['contact_id'],
            'particle_a': contact['particle_a'],
            'particle_b': contact['particle_b'],
            'gap_px': contact['gap_px'],
            'gap_physical': contact['gap_scaled'],
            'unit': report['unit'],
            'x_px': contact['x'],
            'y_px': contact['y'],
            'boundary_pixels': contact['boundary_pixels'],
            'cluster_id': contact['cluster_id'],
        })
    return csv_bytes(rows, fields)


def contact_network_image(image, labels, result, report):
    from PIL import Image, ImageDraw
    from .rendering import font, label_color, draw_scale_bar
    width, height = image.size
    base = np.asarray(image.convert('RGB'), dtype=float) * 0.45
    lookup = np.zeros((int(labels.max()) + 1, 3), dtype=float)
    cluster_of = {}
    for cluster in report['clusters']:
        for pid in cluster['particle_ids']:
            cluster_of[pid] = cluster['cluster_id']
    for row in result['particles']:
        pid = row['id']
        if not row.get('included', True):
            lookup[pid] = (120, 120, 120)
        elif pid in cluster_of:
            lookup[pid] = label_color(cluster_of[pid])
        else:
            lookup[pid] = (96, 128, 140)
    foreground = labels > 0
    base[foreground] = base[foreground] * 0.45 + lookup[labels][foreground] * 0.55
    output = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(output)
    contacts = report['contacts']
    connector = max(2, width // 500)
    for contact in contacts:
        draw.line([tuple(contact['point_a']), tuple(contact['point_b'])], fill=(255, 138, 61), width=connector)
    marker = max(3, width // 300)
    small = font(max(10, width // 120))
    for contact in contacts:
        x, y = contact['x'], contact['y']
        draw.ellipse((x - marker, y - marker, x + marker, y + marker), fill=(230, 60, 48), outline=(255, 255, 255))
        if len(contacts) <= 400:
            draw.text((x + marker + 2, y - marker - 2), str(contact['contact_id']), font=small, fill=(255, 220, 170), stroke_width=1, stroke_fill=(0, 0, 0))
    text_font = font(max(11, width // 100))
    for row in result['particles']:
        color = (255, 255, 255) if row.get('included', True) else (150, 150, 150)
        draw.text((row['centroid_x'] + 3, row['centroid_y'] + 3), str(row['id']), font=text_font, fill=color, stroke_width=1, stroke_fill=(0, 0, 0))
    legend_font = font(max(11, width // 90))
    step = max(15, width // 90 + 5)
    for index, line in enumerate([
        'cluster colors = connected clusters',
        'dim teal = isolated, gray = excluded',
        'red dots = contact points (contact id)',
    ]):
        draw.text((10, 8 + index * step), line, font=legend_font, fill=(235, 240, 244), stroke_width=1, stroke_fill=(0, 0, 0))
    draw_scale_bar(draw, (width, height), result['calibration'])
    return output
