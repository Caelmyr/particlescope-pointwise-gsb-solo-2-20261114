import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
from .validation import ValidationError
from .measurements import included_rows
from .statistics import describe, scalar, cumulative_curve, histogram
from .segmentation import evaluate_reference


UNIT_FACTORS = {'nm': 0.001, 'μm': 1, 'mm': 1000}


def normalized_rows(result, target_unit):
    source_unit = result['unit']
    if source_unit == target_unit:
        factor = 1
    elif source_unit == 'px' or target_unit == 'px':
        raise ValidationError('不能直接比较像素样本和已物理标定的样本')
    else:
        factor = UNIT_FACTORS[source_unit] / UNIT_FACTORS[target_unit]
    result_rows = []
    lengths = ['diameter', 'perimeter', 'major_axis', 'minor_axis', 'feret_max', 'feret_min']
    for original in included_rows(result):
        row = dict(original)
        for field in lengths:
            row[field] *= factor
        row['area'] *= factor ** 2
        row['convex_area'] *= factor ** 2
        result_rows.append(row)
    return result_rows


def compare_distributions(first_result, second_result, field='diameter', bins=15):
    unit = first_result['unit']
    first_rows = normalized_rows(first_result, unit)
    second_rows = normalized_rows(second_result, unit)
    first = np.asarray([r[field] for r in first_rows], dtype=float)
    second = np.asarray([r[field] for r in second_rows], dtype=float)
    if not len(first) or not len(second):
        raise ValidationError('两个样本都需要至少一个纳入统计的颗粒')
    ks = stats.ks_2samp(first, second)
    if len(first) > 1 and len(second) > 1 and (np.std(first) or np.std(second)):
        welch = stats.ttest_ind(first, second, equal_var=False)
        t_stat = scalar(welch.statistic)
        t_p = scalar(welch.pvalue)
    else:
        t_stat = t_p = None
    if len(first) > 1 and len(second) > 1:
        pooled = ((len(first) - 1) * first.var(ddof=1) + (len(second) - 1) * second.var(ddof=1)) / (len(first) + len(second) - 2)
        effect = (first.mean() - second.mean()) / np.sqrt(pooled) if pooled else None
    else:
        effect = None
    combined = np.concatenate([first, second])
    edges = np.histogram_bin_edges(combined, bins=bins)
    first_hist, _ = np.histogram(first, bins=edges)
    second_hist, _ = np.histogram(second, bins=edges)
    return {
        'field': field,
        'unit': unit,
        'first': describe(first),
        'second': describe(second),
        'mean_difference': scalar(first.mean() - second.mean()),
        'mean_ratio': scalar(first.mean() / second.mean()) if second.mean() else None,
        'wasserstein_distance': scalar(stats.wasserstein_distance(first, second)),
        'ks_statistic': scalar(ks.statistic),
        'ks_p_value': scalar(ks.pvalue),
        'welch_t_statistic': t_stat,
        'welch_p_value': t_p,
        'cohen_d': scalar(effect) if effect is not None else None,
        'edges': [scalar(v) for v in edges],
        'first_percent': [scalar(v) for v in first_hist / len(first) * 100],
        'second_percent': [scalar(v) for v in second_hist / len(second) * 100],
        'first_cdf': cumulative_curve(first),
        'second_cdf': cumulative_curve(second),
        'warning': '按颗粒独立观测计算，不能替代独立实验重复；显著性仅作探索',
    }


def match_particles(first_rows, second_rows, maximum_distance=20):
    if not first_rows or not second_rows:
        return {
            'matches': [],
            'unmatched_first': [r['id'] for r in first_rows],
            'unmatched_second': [r['id'] for r in second_rows],
        }
    points = np.asarray([[r['centroid_x'], r['centroid_y']] for r in second_rows])
    tree = cKDTree(points)
    candidates = []
    for index, row in enumerate(first_rows):
        point = np.asarray([row['centroid_x'], row['centroid_y']])
        for neighbor in tree.query_ball_point(point, maximum_distance):
            distance = float(np.linalg.norm(points[neighbor] - point))
            candidates.append((distance, index, neighbor))
    candidates.sort()
    used_first = set()
    used_second = set()
    matches = []
    for distance, first_index, second_index in candidates:
        if first_index in used_first or second_index in used_second:
            continue
        used_first.add(first_index)
        used_second.add(second_index)
        first = first_rows[first_index]
        second = second_rows[second_index]
        matches.append({
            'first_id': first['id'],
            'second_id': second['id'],
            'distance_px': scalar(distance),
            'area_change_percent': scalar((second['area_px'] - first['area_px']) / first['area_px'] * 100),
            'diameter_change': scalar(second['diameter'] - first['diameter']),
        })
    return {
        'matches': matches,
        'unmatched_first': [r['id'] for i, r in enumerate(first_rows) if i not in used_first],
        'unmatched_second': [r['id'] for i, r in enumerate(second_rows) if i not in used_second],
        'method': '按质心距离排序的贪心一对一匹配，适用于同一视野的分析版本',
    }


def lineage_matching(first, second, ancestor_lookup=None):
    """Match particles across versions using recorded mask-edit lineage.

    Handles direct parent/child mask edits and chains where measurement filter
    versions sit between the two compared analyses. ``ancestor_lookup`` maps an
    analysis id to its stored result so the parent chain can be walked.
    Returns None when the analyses are not connected by local mask revisions,
    so callers can use centroid matching instead.
    """
    lineage = second.get('lineage')
    if not lineage:
        return None
    if ancestor_lookup is None:
        return None
    # Walk parent links until reaching the first analysis, crossing
    # filter-only versions (which have parent_id but no lineage of their own).
    current = second
    hops = 0
    while current is not None and current.get('id') != first['id']:
        parent_id = (current.get('lineage') or {}).get('parent_analysis_id') or current.get('parent_id')
        if not parent_id or hops > 100:
            return None
        current = ancestor_lookup(parent_id)
        if current is None:
            return None
        hops += 1
    if current.get('id') != first['id']:
        return None
    first_rows = {row['id']: row for row in first['particles']}
    second_rows = {row['id']: row for row in second['particles']}
    matches, split, merged, created, modified = [], [], [], [], []
    for new_id, entry in lineage.get('chain', {}).items():
        new_id = int(new_id)
        parents = entry['parent_ids']
        relations = entry['relations']
        current_row = second_rows.get(new_id)
        if current_row is None:
            continue
        if not parents:
            created.append(new_id)
        elif len(parents) > 1:
            merged.append({'new_id': new_id, 'old_ids': parents})
        else:
            old_id = parents[0]
            previous = first_rows.get(old_id)
            if previous is None:
                continue
            if 'split' in relations:
                split.append({'old_id': old_id, 'new_id': new_id})
            elif 'modified' in relations:
                modified.append(new_id)
            matches.append({
                'first_id': old_id,
                'second_id': new_id,
                'relation': relations[0],
                'area_change_percent': scalar((current_row['area_px'] - previous['area_px']) / previous['area_px'] * 100),
                'diameter_change': scalar(current_row['diameter'] - previous['diameter']),
            })
    removed = list(lineage.get('removed_ids', []))
    unchanged = [
        entry['new_ids'][0]
        for entry in lineage.get('relations', [])
        if entry['relation'] == 'unchanged'
    ]
    return {
        'method': '按掩膜修订血缘匹配（未涉及对象编号保持不变）',
        'lineage_hops': hops,
        'matches': matches,
        'unmatched_first': sorted(removed),
        'unmatched_second': sorted(created),
        'unchanged_ids': sorted(unchanged),
        'modified_ids': sorted(modified),
        'split': split,
        'merged': merged,
        'created_ids': sorted(created),
        'removed_ids': sorted(removed),
    }


def compare_analysis_versions(first, second, first_labels, second_labels, maximum_distance=20, ancestor_lookup=None):
    if first['image_revision'] != second['image_revision']:
        raise ValidationError('分析版本来自不同图像版本，不能直接像素比较')
    if first_labels.shape != second_labels.shape:
        raise ValidationError('分析掩膜尺寸不一致')
    first_ids = [r['id'] for r in included_rows(first)]
    second_ids = [r['id'] for r in included_rows(second)]
    first_mask = np.isin(first_labels, first_ids) & (first_labels > 0)
    second_mask = np.isin(second_labels, second_ids) & (second_labels > 0)
    lineage_report = lineage_matching(first, second, ancestor_lookup)
    matching = lineage_report or match_particles(included_rows(first), included_rows(second), maximum_distance)
    return {
        'first_id': first['id'],
        'second_id': second['id'],
        'count_change': second['count'] - first['count'],
        'coverage_change': scalar(second['coverage'] - first['coverage']),
        'pixel_agreement': evaluate_reference(first_mask, second_mask),
        'matching': matching,
    }


def group_summary(results, field='diameter'):
    if not results:
        raise ValidationError('没有分析结果')
    unit = results[0]['unit']
    groups = []
    sample_means = []
    pooled_values = []
    for result in results:
        rows = normalized_rows(result, unit)
        values = [r[field] for r in rows]
        summary = describe(values)
        groups.append({
            'sample_id': result['sample_id'],
            'analysis_id': result['id'],
            'description': summary,
        })
        if values:
            sample_means.append(summary['mean'])
            pooled_values.extend(values)
    return {
        'field': field,
        'unit': unit,
        'groups': groups,
        'pooled_particle_summary': describe(pooled_values),
        'sample_mean_summary': describe(sample_means),
        'warning': '合并颗粒统计按颗粒数加权；样本均值统计按样本等权',
    }
