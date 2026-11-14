"""晶格畴识别。

从颗粒质心出发，用二维复键序psi_l（六方取l=6、四方取l=4）估计每个颗粒的
局部晶格取向，再按取向一致性、最近邻壳层和局部间距把颗粒连接成畴。边缘缺
邻居、点缺陷与无序区不会被强行并入完整晶格。本模块只读取既有测量结果，不
修改颗粒测量。
"""
import math
import numpy as np
from scipy.spatial import cKDTree
from .measurements import included_rows
from .statistics import scalar
from .validation import ValidationError


HEX_SYMMETRY = 6
SQUARE_SYMMETRY = 4


def point_array(rows, scale=1):
    if not rows:
        return np.empty((0, 2), dtype=float), []
    points = np.array(
        [[r['centroid_x'] * scale, r['centroid_y'] * scale] for r in rows],
        dtype=float,
    )
    ids = [int(r['id']) for r in rows]
    return points, ids


def estimate_spacing(points):
    """估计最小规则晶格的最近邻间距。

    规则晶格的最近邻距离在晶格常数附近形成显著尖峰；无序区的随机距离分
    散偏小，多种间距共存时则出现多个峰。先用直方图定位计数最集中的距
    离，再在其±25%窗口内自举取中位数精炼，避免被无序区或大晶格带偏。
    """
    if len(points) < 2:
        return None
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)
    nearest = distances[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
    if not len(nearest):
        return None
    if float(nearest.max() - nearest.min()) <= 1e-9:
        return float(nearest[0])
    bins = min(36, max(12, len(nearest) // 5))
    histogram, edges = np.histogram(nearest, bins=bins)
    peak = int(np.argmax(histogram))
    spacing = float(nearest[
        (nearest >= edges[peak]) & (nearest < edges[peak + 1])
    ].mean())
    for _ in range(20):
        window = nearest[(nearest >= spacing * 0.75) & (nearest <= spacing * 1.25)]
        if len(window) < 3:
            break
        updated = float(np.median(window))
        if abs(updated - spacing) <= 1e-6 * max(updated, 1.0):
            spacing = updated
            break
        spacing = updated
    return spacing


def build_shells(points, shell_count, spacing):
    """划分每个颗粒的第一壳层与扩展邻居。

    第一壳层由距离间隙客观划定（次近邻/最近邻之比在规则晶格中至少为
    sqrt(2)≈1.41），只依赖各点自身的距离序列；扩展壳层用于畴连接时跨越
    更宽的识别尺度。键序始终只由第一壳层计算，避免次近邻键角旋转稀释。
    """
    tree = cKDTree(points)
    candidate_radius = spacing * (shell_count + 0.9)
    gathered = tree.query_ball_tree(tree, candidate_radius)
    fallback_k = min(13, len(points))
    fallback_indices = tree.query(points, k=fallback_k)[1]
    first_shells = []
    extended = []
    local_spacing = np.full(len(points), np.nan)
    for index, members in enumerate(gathered):
        members = [m for m in members if m != index]
        if not members:
            row = np.atleast_1d(fallback_indices[index])
            members = [int(m) for m in row if m != index]
        if not members:
            first_shells.append(np.empty(0, dtype=int))
            extended.append(np.empty(0, dtype=int))
            continue
        order = sorted(members, key=lambda m: float(np.linalg.norm(points[m] - points[index])))
        distances = np.array([float(np.linalg.norm(points[m] - points[index])) for m in order])
        # 晶格键必须落在全局间距的0.75~1.18倍窗口内：无序区的异常近邻
        # （远小于间距）和次近邻（至少1.41倍间距）都被排除。若窗口内
        # 没有任何邻居，退而保留最近邻并将该颗粒留给边缘/无序逻辑处理。
        lattice_bonds = np.flatnonzero(
            (distances >= spacing * 0.75) & (distances <= spacing * 1.18)
        )
        if len(lattice_bonds):
            first_end = int(lattice_bonds[-1]) + 1
            bond_distances = distances[lattice_bonds]
            local_spacing[index] = np.median(bond_distances)
        else:
            first_end = 1
            local_spacing[index] = distances[0]
        first = np.array(order[:first_end], dtype=int)
        first_shells.append(first)
        if shell_count <= 1:
            extended.append(first)
            continue
        outer = [
            member for rank, member in enumerate(order)
            if rank >= first_end
            and distances[rank] <= local_spacing[index] * (shell_count + 0.25)
        ]
        extended.append(np.array(order[:first_end] + outer, dtype=int))
    return first_shells, extended, local_spacing


def bond_order(points, first_shells, symmetry):
    """计算二维复键序psi_l的模长与归一化相位向量。

    psi_l(i)是第一壳层键角exp(i*l*theta)的平均。理想六方晶格的psi6、
    四方晶格的psi4模长为1；随机点云的模长期望约1/sqrt(配位数)。少于3
    个键无法定义局部晶格环境，模长置零以避免单键造成的伪完美序。
    """
    raw = np.zeros(len(points), dtype=complex)
    for index, chosen in enumerate(first_shells):
        if len(chosen) < 3:
            continue
        delta = points[chosen] - points[index]
        phi = np.arctan2(delta[:, 1], delta[:, 0])
        raw[index] = np.exp(1j * symmetry * phi).mean()
    magnitude = np.abs(raw)
    normalized = np.divide(raw, magnitude, out=np.zeros_like(raw), where=magnitude > 0)
    vectors = np.stack([normalized.real, normalized.imag], axis=1)
    return vectors, magnitude


def phase_similarity(vectors, a, b):
    """两个归一化键序向量的点积，等于相位差余弦。"""
    return float(vectors[a, 0] * vectors[b, 0] + vectors[a, 1] * vectors[b, 1])


def orientation_angle(vectors, symmetry):
    """从键序相位反演局部主取向，角度在0到180/symmetry度之间。"""
    angles = np.full(len(vectors), np.nan)
    for index in range(len(vectors)):
        if vectors[index, 0] == 0 and vectors[index, 1] == 0:
            continue
        phase = math.atan2(vectors[index, 1], vectors[index, 0])
        angles[index] = math.degrees(phase / symmetry) % (180.0 / symmetry)
    return angles


def edge_flags(points, first_shells, expected_coordination):
    """只把点云物理外缘且缺邻居的颗粒标记为边缘。

    畴内部因缺陷而缺邻居的颗粒不属于图像边缘，仍可参与畴生长；这里用
    点云凸包附近（半径90分位以外）且配位数不足两个以上来判定。
    """
    edge = np.zeros(len(points), dtype=bool)
    if len(points) < 4:
        return edge
    center = points.mean(axis=0)
    radii = np.linalg.norm(points - center, axis=1)
    outer = np.quantile(radii, 0.90)
    for index in range(len(points)):
        if radii[index] >= outer and len(first_shells[index]) <= expected_coordination - 2:
            edge[index] = True
    return edge


def assign_domains(vectors, order_magnitude, neighbors, local_spacing, edge,
                   order_threshold, similarity, spacing_tolerance, minimum_size):
    """按取向与间距一致性在邻居图上生长畴，边缘颗粒不参与搭桥。"""
    count = len(vectors)
    labels = np.full(count, -1, dtype=int)
    valid_spacing = local_spacing[np.isfinite(local_spacing)]
    # 用25~75分位区间的中位数代表规则晶格间距，避免无序区异常间距拉偏。
    if len(valid_spacing):
        low, high = np.quantile(valid_spacing, [0.25, 0.75])
        core = valid_spacing[(valid_spacing >= low) & (valid_spacing <= high)]
        spacing_median = float(np.median(core if len(core) else valid_spacing))
    else:
        spacing_median = 1.0
    if spacing_median <= 0:
        spacing_median = 1.0

    def compatible(a, b):
        if order_magnitude[a] < order_threshold or order_magnitude[b] < order_threshold:
            return False
        if phase_similarity(vectors, a, b) < similarity:
            return False
        if math.isfinite(local_spacing[a]) and math.isfinite(local_spacing[b]):
            pair_spacing = 0.5 * (local_spacing[a] + local_spacing[b])
            if abs(pair_spacing - spacing_median) > spacing_tolerance * spacing_median:
                return False
        return True

    domain_id = 0
    for seed in range(count):
        if labels[seed] != -1 or edge[seed] or order_magnitude[seed] < order_threshold:
            continue
        labels[seed] = domain_id
        queue = [seed]
        cursor = 0
        while cursor < len(queue):
            center = queue[cursor]
            cursor += 1
            for neighbor in neighbors[center]:
                if labels[neighbor] != -1 or edge[neighbor]:
                    continue
                if compatible(center, neighbor):
                    labels[neighbor] = domain_id
                    queue.append(neighbor)
        domain_id += 1

    # 边缘颗粒只允许以多数票并入已经成形的相邻畴，不能连接两个畴。
    for index in np.flatnonzero(edge):
        if order_magnitude[index] < order_threshold:
            continue
        votes = {}
        for neighbor in neighbors[index]:
            if labels[neighbor] < 0:
                continue
            if compatible(index, neighbor):
                votes[labels[neighbor]] = votes.get(labels[neighbor], 0) + 1
        if votes:
            labels[index] = max(votes, key=votes.get)

    if minimum_size > 1 and np.any(labels >= 0):
        sizes = np.bincount(labels[labels >= 0])
        for index in range(count):
            if labels[index] >= 0 and sizes[labels[index]] < minimum_size:
                labels[index] = -1
    return labels


def boundary_flags(labels, neighbors):
    """畴内颗粒若与其它畴或无序颗粒相邻，则位于畴界。"""
    boundary = np.zeros(len(labels), dtype=bool)
    for index, label in enumerate(labels):
        if label < 0:
            continue
        for neighbor in neighbors[index]:
            if labels[neighbor] != label:
                boundary[index] = True
                break
    return boundary


def spacing_report(local_spacing, scale):
    """检查局部间距是否出现两群不能强行连成同一晶格的数值。"""
    values = np.sort(local_spacing[np.isfinite(local_spacing)])
    report = {
        'median_px': scalar(float(np.median(values)) / scale) if len(values) else None,
        'q1_px': scalar(float(np.quantile(values, 0.25)) / scale) if len(values) else None,
        'q3_px': scalar(float(np.quantile(values, 0.75)) / scale) if len(values) else None,
        'bimodal': False,
        'groups': [],
    }
    if len(values) < 8:
        return report
    # 规则晶格的局部间距集中在少数离散值附近，按相对间隙聚类，再判断
    # 是否形成两个内部紧凑、均值比超过1.3的间距群。
    groups = [[values[0]]]
    for value in values[1:]:
        if value - groups[-1][-1] > 0.18 * groups[-1][-1]:
            groups.append([])
        groups[-1].append(value)
    if len(groups) < 2:
        return report
    sizes = [len(group) for group in groups]
    first, second = sorted(range(len(groups)), key=lambda index: sizes[index], reverse=True)[:2]
    left = np.array(groups[min(first, second)])
    right = np.array(groups[max(first, second)])
    if (
        len(left) >= 4 and len(right) >= 4
        and right.mean() > 1.3 * left.mean()
    ):
        report['bimodal'] = True
        report['groups'] = [
            scalar(float(left.mean()) / scale),
            scalar(float(right.mean()) / scale),
        ]
    return report


def circular_mean_degrees(angles, period):
    """周期角度的圆均值，避免在周期边界附近算术平均失真。"""
    if not len(angles):
        return None
    scaled = np.deg2rad(np.asarray(angles) * 360.0 / period)
    phase = math.atan2(np.sin(scaled).mean(), np.cos(scaled).mean())
    return math.degrees(phase) * period / 360.0 % period


def misorientation_degrees(angle_a, angle_b, period):
    """两个周期取向之间的最小夹角。

    角度被限制在0到period之间，等价取向会落在周期两端（如六方晶格的0°
    与30°），因此同时比较直接差和跨周期差，取较小者。
    """
    difference = abs(angle_a - angle_b)
    return min(difference, period - difference)


def domain_misorientations(domains, symmetry):
    """计算所有畴对之间的最小取向差，供判断畴界两侧的晶格偏转。"""
    period = 180.0 / symmetry
    result = []
    for index, first in enumerate(domains):
        for second in domains[index + 1:]:
            if first['orientation_degrees'] is None or second['orientation_degrees'] is None:
                continue
            result.append({
                'domain_a': first['domain_id'],
                'domain_b': second['domain_id'],
                'misorientation_degrees': scalar(
                    misorientation_degrees(first['orientation_degrees'], second['orientation_degrees'], period)
                ),
            })
    return result


def build_summary(labels, order_magnitude, angles, local_spacing, points, ids, scale, symmetry):
    """汇总各畴并把标签重新编号为从1开始的连续序号。"""
    period = 180.0 / symmetry
    domains = []
    for label in [value for value in np.unique(labels) if value >= 0]:
        members = np.flatnonzero(labels == label)
        member_angles = angles[members]
        member_angles = member_angles[np.isfinite(member_angles)]
        member_spacing = local_spacing[members]
        member_spacing = member_spacing[np.isfinite(member_spacing)]
        center = points[members].mean(axis=0)
        domains.append({
            'count': int(len(members)),
            'particle_ids': [ids[i] for i in members],
            'orientation_degrees': scalar(circular_mean_degrees(member_angles, period)) if len(member_angles) else None,
            'orientation_spread_degrees': scalar(float(member_angles.std(ddof=1))) if len(member_angles) > 1 else 0.0,
            'mean_order': scalar(float(order_magnitude[members].mean())),
            'spacing_px': scalar(float(member_spacing.mean()) / scale) if len(member_spacing) else None,
            'centroid_x': scalar(center[0] / scale),
            'centroid_y': scalar(center[1] / scale),
        })
    domains.sort(key=lambda item: item['count'], reverse=True)
    id_map = {}
    for rank, domain in enumerate(domains, 1):
        domain['domain_id'] = rank
        for particle_id in domain['particle_ids']:
            id_map[particle_id] = rank
    return domains, id_map


def lattice_domains(rows, scale=1, shell_count=1, order_threshold=None,
                    similarity=0.85, spacing_tolerance=0.25, minimum_size=4,
                    spacing_override=None, lattice=None):
    """完整的晶格畴识别流程，返回可序列化报告。"""
    points, ids = point_array(rows, scale)
    if len(points) < 4:
        raise ValidationError('晶格畴识别至少需要4个颗粒')
    if len(points) > 20000:
        raise ValidationError('晶格畴识别最多支持20000个颗粒')
    shell_count = max(1, min(4, int(shell_count)))
    minimum_size = max(1, min(1000, int(minimum_size)))

    spacing = float(spacing_override) * scale if spacing_override else estimate_spacing(points)
    if not spacing or not math.isfinite(spacing) or spacing <= 0:
        raise ValidationError('无法估计晶格间距，请显式指定间距')
    first_shells, neighbors, local_spacing = build_shells(points, shell_count, spacing)

    hex_vectors, hex_magnitude = bond_order(points, first_shells, HEX_SYMMETRY)
    square_vectors, square_magnitude = bond_order(points, first_shells, SQUARE_SYMMETRY)
    if lattice == 'square' or (lattice != 'hexagonal' and np.nanmedian(square_magnitude) > np.nanmedian(hex_magnitude)):
        symmetry, lattice_name = SQUARE_SYMMETRY, 'square'
        vectors, order_magnitude = square_vectors, square_magnitude
    else:
        symmetry, lattice_name = HEX_SYMMETRY, 'hexagonal'
        vectors, order_magnitude = hex_vectors, hex_magnitude

    if order_threshold is None:
        order_threshold = 0.55 if symmetry == HEX_SYMMETRY else 0.5
    order_threshold = max(0.05, min(0.98, float(order_threshold)))
    similarity = max(0.5, min(0.999, float(similarity)))
    spacing_tolerance = max(0.05, min(1.0, float(spacing_tolerance)))

    angles = orientation_angle(vectors, symmetry)
    edge = edge_flags(points, first_shells, symmetry)
    raw_labels = assign_domains(
        vectors, order_magnitude, neighbors, local_spacing, edge,
        order_threshold, similarity, spacing_tolerance, minimum_size,
    )
    boundary = boundary_flags(raw_labels, neighbors)
    coordination = np.array([len(chosen) for chosen in first_shells])
    defect = (raw_labels < 0) & (~edge) & (order_magnitude >= order_threshold)
    disordered = (raw_labels < 0) & (~defect)

    domains, id_map = build_summary(raw_labels, order_magnitude, angles, local_spacing, points, ids, scale, symmetry)
    particles = []
    for index in range(len(rows)):
        particles.append({
            'id': ids[index],
            'domain_id': id_map.get(ids[index]),
            'order': scalar(order_magnitude[index]),
            'orientation_degrees': scalar(angles[index]) if math.isfinite(angles[index]) else None,
            'coordination': int(coordination[index]),
            'local_spacing_px': scalar(local_spacing[index] / scale) if math.isfinite(local_spacing[index]) else None,
            'is_edge': bool(edge[index]),
            'is_boundary': bool(boundary[index]),
            'is_defect': bool(defect[index]),
            'is_disordered': bool(disordered[index]),
        })

    return {
        'lattice': lattice_name,
        'symmetry': symmetry,
        'orientation_period_degrees': scalar(180.0 / symmetry),
        'misorientations': domain_misorientations(domains, symmetry),
        'particle_count': len(rows),
        'distance_unit': 'pixel',
        'spacing_px': scalar(spacing / scale),
        'spacing_distribution': spacing_report(local_spacing, scale),
        'shell_count': shell_count,
        'order_threshold': scalar(order_threshold),
        'similarity_threshold': scalar(similarity),
        'spacing_tolerance': scalar(spacing_tolerance),
        'minimum_domain_size': minimum_size,
        'domain_count': len(domains),
        'ordered_particle_count': int(sum(domain['count'] for domain in domains)),
        'disordered_particle_count': int(np.sum(disordered)),
        'defect_particle_count': int(np.sum(defect)),
        'edge_particle_count': int(np.sum(edge)),
        'boundary_particle_count': int(np.sum(boundary)),
        'mean_order': scalar(float(np.nanmean(order_magnitude))),
        'domains': domains,
        'particles': particles,
        'note': '畴识别只读取当前颗粒质心与测量结果，不修改原始分析；边缘缺邻居、点缺陷和无序区不参与完整晶格连接。',
    }


def domain_report(result, parameters):
    rows = included_rows(result)
    return lattice_domains(
        rows,
        scale=result['scale'],
        shell_count=parameters.get('shell_count', 1),
        order_threshold=parameters.get('order_threshold'),
        similarity=parameters.get('similarity', 0.85),
        spacing_tolerance=parameters.get('spacing_tolerance', 0.25),
        minimum_size=parameters.get('minimum_domain_size', 4),
        spacing_override=parameters.get('spacing'),
        lattice=parameters.get('lattice'),
    )
