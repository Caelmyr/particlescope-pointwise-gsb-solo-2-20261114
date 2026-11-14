import math
import numpy as np
from scipy import stats
from .measurements import MEASUREMENT_FIELDS, included_rows
from .validation import ValidationError


def clean_values(values):
    result = np.asarray(values, dtype=float)
    return result[np.isfinite(result)]


def scalar(value, digits=6):
    number = float(value)
    return round(number, digits) if math.isfinite(number) else None


def describe(values):
    array = clean_values(values)
    size = len(array)
    if size == 0:
        return {
            'n': 0,
            'mean': None,
            'std': None,
            'median': None,
            'minimum': None,
            'maximum': None,
            'q1': None,
            'q3': None,
            'iqr': None,
            'cv_percent': None,
            'sem': None,
            'ci95_low': None,
            'ci95_high': None,
            'skewness': None,
            'kurtosis': None,
        }
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if size > 1 else 0
    q1, q3 = np.percentile(array, [25, 75])
    sem = std / math.sqrt(size)
    critical = float(stats.t.ppf(0.975, size - 1)) if size > 1 else 0
    nonconstant = bool(array.max() > array.min())
    return {
        'n': size,
        'mean': scalar(mean),
        'std': scalar(std),
        'median': scalar(np.median(array)),
        'minimum': scalar(array.min()),
        'maximum': scalar(array.max()),
        'q1': scalar(q1),
        'q3': scalar(q3),
        'iqr': scalar(q3 - q1),
        'cv_percent': scalar(std / abs(mean) * 100) if mean else None,
        'sem': scalar(sem),
        'ci95_low': scalar(mean - critical * sem),
        'ci95_high': scalar(mean + critical * sem),
        'skewness': scalar(stats.skew(array, bias=False)) if size >= 3 and nonconstant else None,
        'kurtosis': scalar(stats.kurtosis(array, bias=False)) if size >= 4 and nonconstant else None,
    }


def histogram(values, bins=15, logarithmic=False, weights=None):
    array = clean_values(values)
    if not len(array):
        return {'edges': [], 'counts': [], 'percent': [], 'cumulative_percent': []}
    if logarithmic:
        if np.any(array <= 0):
            raise ValidationError('对数分箱要求全部测量值大于0')
        minimum, maximum = float(array.min()), float(array.max())
        if minimum == maximum:
            minimum *= 0.9
            maximum *= 1.1
        edges = np.geomspace(minimum, maximum, bins + 1)
    else:
        edges = np.histogram_bin_edges(array, bins=bins)
    counts, edges = np.histogram(array, bins=edges, weights=weights)
    total = float(counts.sum())
    percent = counts.astype(float) / total * 100 if total else np.zeros(len(counts))
    return {
        'edges': [scalar(v) for v in edges],
        'counts': [scalar(v) for v in counts],
        'percent': [scalar(v) for v in percent],
        'cumulative_percent': [scalar(v) for v in np.cumsum(percent)],
        'logarithmic': logarithmic,
    }


def weighted_quantile(values, quantiles, weights):
    array = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if not len(array):
        return [None for _ in quantiles]
    order = np.argsort(array)
    array = array[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    total = cumulative[-1]
    if total <= 0:
        return [None for _ in quantiles]
    positions = (cumulative - weights / 2) / total
    return [scalar(np.interp(q, positions, array)) for q in quantiles]


def particle_size_distribution(rows, weighting='number', bins=15, logarithmic=False):
    diameters = [row['diameter'] for row in rows]
    if weighting == 'area':
        weights = [row['area'] for row in rows]
    elif weighting == 'volume':
        weights = [math.pi / 6 * row['diameter'] ** 3 for row in rows]
    else:
        weights = [1 for _ in rows]
    d10, d50, d90 = weighted_quantile(diameters, [0.1, 0.5, 0.9], weights)
    distribution = histogram(diameters, bins, logarithmic, weights)
    distribution.update({
        'weighting': weighting,
        'd10': d10,
        'd50': d50,
        'd90': d90,
        'span': scalar((d90 - d10) / d50) if d50 else None,
        'assumption': '体积权重根据等效球体估算' if weighting == 'volume' else '',
    })
    return distribution


def outliers(rows, field, multiplier=1.5):
    values = clean_values([row[field] for row in rows])
    if not len(values):
        return {'ids': [], 'lower': None, 'upper': None}
    q1, q3 = np.percentile(values, [25, 75])
    spread = q3 - q1
    lower = q1 - multiplier * spread
    upper = q3 + multiplier * spread
    ids = [row['id'] for row in rows if row[field] < lower or row[field] > upper]
    return {'ids': ids, 'lower': scalar(lower), 'upper': scalar(upper), 'method': 'Tukey IQR'}


def correlation(rows, fields):
    if len(rows) < 3:
        return {'fields': fields, 'matrix': [], 'n': len(rows)}
    columns = np.array([[row[field] for row in rows] for field in fields], dtype=float)
    matrix = []
    for first in columns:
        line = []
        for second in columns:
            if np.std(first) == 0 or np.std(second) == 0:
                line.append(None)
            else:
                line.append(scalar(np.corrcoef(first, second)[0, 1]))
        matrix.append(line)
    return {'fields': fields, 'matrix': matrix, 'n': len(rows)}


def linear_fit(rows, x_field, y_field):
    if len(rows) < 3:
        raise ValidationError('回归至少需要三个颗粒')
    x = np.asarray([row[x_field] for row in rows], dtype=float)
    y = np.asarray([row[y_field] for row in rows], dtype=float)
    if np.ptp(x) == 0:
        raise ValidationError('自变量没有变化，无法回归')
    fit = stats.linregress(x, y)
    predicted = fit.intercept + fit.slope * x
    residual = y - predicted
    return {
        'x_field': x_field,
        'y_field': y_field,
        'slope': scalar(fit.slope),
        'intercept': scalar(fit.intercept),
        'r_squared': scalar(fit.rvalue ** 2),
        'p_value': scalar(fit.pvalue),
        'slope_stderr': scalar(fit.stderr),
        'residual_std': scalar(residual.std(ddof=2)),
        'n': len(rows),
        'points': [{'id': r['id'], 'x': r[x_field], 'y': r[y_field]} for r in rows],
    }


def orientation_distribution(rows, bins=18):
    if not rows:
        return {'counts': [], 'edges': [], 'preferred_angle': None, 'alignment': None}
    angles = np.asarray([r['orientation'] for r in rows])
    radians = np.deg2rad(angles * 2)
    cosine = float(np.cos(radians).mean())
    sine = float(np.sin(radians).mean())
    preferred = math.degrees(math.atan2(sine, cosine)) / 2 % 180
    alignment = math.hypot(cosine, sine)
    counts, edges = np.histogram(angles, bins=np.linspace(0, 180, bins + 1))
    return {
        'counts': counts.tolist(),
        'edges': edges.tolist(),
        'preferred_angle': scalar(preferred),
        'alignment': scalar(alignment),
        'interpretation': '0为各向均匀，1为方向一致；接近圆形的颗粒方向不稳定',
    }


def bootstrap_mean(values, repeats=1000, confidence=0.95, seed=42):
    array = clean_values(values)
    if not len(array):
        raise ValidationError('没有可用于重采样的数据')
    if len(array) > 20000:
        raise ValidationError('重采样最多支持20000个颗粒')
    rng = np.random.default_rng(seed)
    means = np.empty(repeats)
    for start in range(0, repeats, 50):
        count = min(50, repeats - start)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        means[start:start + count] = np.mean(array[indices], axis=1)
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(means, [alpha, 1 - alpha])
    return {
        'mean': scalar(array.mean()),
        'confidence': confidence,
        'lower': scalar(lower),
        'upper': scalar(upper),
        'repeats': repeats,
        'seed': seed,
        'n': len(array),
    }


def normality(values):
    array = clean_values(values)
    if len(array) < 8 or np.ptp(array) == 0:
        return {'statistic': None, 'p_value': None, 'reason': '至少需要8个非恒定测量值'}
    test = stats.normaltest(array)
    return {
        'statistic': scalar(test.statistic),
        'p_value': scalar(test.pvalue),
        'method': 'D’Agostino–Pearson',
        'warning': '颗粒可能存在空间相关性；结果仅作探索',
    }


def statistics_report(result, bins=15):
    rows = included_rows(result)
    description = {}
    for field in MEASUREMENT_FIELDS:
        description[field] = describe([row[field] for row in rows])
    return {
        'analysis_id': result['id'],
        'count': len(rows),
        'unit': result['unit'],
        'descriptive': description,
        'number_distribution': particle_size_distribution(rows, 'number', bins),
        'area_distribution': particle_size_distribution(rows, 'area', bins),
        'volume_distribution': particle_size_distribution(rows, 'volume', bins),
        'orientation': orientation_distribution(rows),
        'outliers': outliers(rows, 'diameter'),
        'correlation': correlation(rows, ['diameter', 'circularity', 'aspect_ratio', 'solidity', 'mean_intensity']),
        'normality': normality([row['diameter'] for row in rows]),
    }


def cumulative_curve(values, weighting=None):
    array = clean_values(values)
    if not len(array):
        return {'x': [], 'cumulative_percent': []}
    order = np.argsort(array)
    if weighting is None:
        weights = np.ones(len(array))
    else:
        weights = np.asarray(weighting, dtype=float)
    weights = weights[order]
    cumulative = np.cumsum(weights) / weights.sum() * 100
    return {
        'x': [scalar(v) for v in array[order]],
        'cumulative_percent': [scalar(v) for v in cumulative],
    }


def size_classes(rows, boundaries):
    if len(boundaries) < 2:
        raise ValidationError('至少需要两个分级边界')
    edges = np.asarray(boundaries, dtype=float)
    if np.any(np.diff(edges) <= 0):
        raise ValidationError('分级边界必须严格递增')
    classes = []
    total_area = sum(r['area'] for r in rows)
    for index in range(len(edges) - 1):
        lower, upper = edges[index:index + 2]
        is_last = index == len(edges) - 2
        subset = [r for r in rows if lower <= r['diameter'] and (r['diameter'] <= upper if is_last else r['diameter'] < upper)]
        area = sum(r['area'] for r in subset)
        classes.append({
            'lower': float(lower),
            'upper': float(upper),
            'includes_upper': is_last,
            'count': len(subset),
            'number_percent': scalar(len(subset) / max(len(rows), 1) * 100),
            'area_percent': scalar(area / total_area * 100) if total_area else 0,
            'ids': [r['id'] for r in subset],
        })
    covered = sum(item['count'] for item in classes)
    return {'classes': classes, 'outside_count': len(rows) - covered}
