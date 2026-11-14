import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
from .imaging import channel_array, disk, region_mask
from .validation import ValidationError


def otsu_threshold(values):
    hist = np.bincount(np.clip(values, 0, 255).astype(np.uint8).ravel(), minlength=256).astype(float)
    total = hist.sum()
    if total == 0:
        raise ValidationError('分析区域为空')
    probability = hist / total
    weight = np.cumsum(probability)
    moment = np.cumsum(probability * np.arange(256))
    overall = moment[-1]
    denominator = weight * (1 - weight)
    between = np.zeros(256)
    valid = denominator > 1e-12
    between[valid] = (overall * weight[valid] - moment[valid]) ** 2 / denominator[valid]
    return float(np.argmax(between))


def threshold_image(image, parameters, selected):
    gray = channel_array(image, parameters['channel'])
    sigma = parameters['sigma']
    if sigma:
        gray = ndi.gaussian_filter(gray, sigma=sigma, mode='reflect')
    method = parameters['method']
    values = gray[selected]
    if not len(values):
        raise ValidationError('分析区域没有像素')
    threshold = parameters['threshold']
    threshold_map = None
    if method == 'otsu':
        threshold = otsu_threshold(values)
    elif method == 'percentile':
        threshold = float(np.percentile(values, parameters['percentile']))
    elif method == 'adaptive':
        local = ndi.uniform_filter(gray, size=parameters['adaptive_window'], mode='reflect')
        if parameters['polarity'] == 'dark':
            threshold_map = local - parameters['adaptive_offset']
        else:
            threshold_map = local + parameters['adaptive_offset']
    limit = threshold_map if threshold_map is not None else threshold
    if parameters['polarity'] == 'dark':
        binary = gray <= limit
    else:
        binary = gray > limit
    binary &= selected
    info = {
        'method': method,
        'effective_threshold': round(float(np.mean(limit)), 4),
        'channel': parameters['channel'],
        'roi_pixel_count': int(selected.sum()),
        'foreground_before_cleanup': int(binary.sum()),
    }
    return binary, gray, info


def clean_binary(binary, parameters, selected):
    cleaned = binary.copy()
    if parameters['opening']:
        cleaned = ndi.binary_opening(cleaned, structure=disk(parameters['opening']))
    if parameters['closing']:
        cleaned = ndi.binary_closing(cleaned, structure=disk(parameters['closing']))
    if parameters['fill_holes']:
        cleaned = ndi.binary_fill_holes(cleaned)
    cleaned &= selected
    return cleaned


def split_regions(binary, seed_distance):
    distance = ndi.distance_transform_edt(binary)
    if not np.any(distance):
        return np.zeros(binary.shape, dtype=np.int32), {'seed_count': 0, 'split': True}
    smooth = ndi.gaussian_filter(distance, 1)
    maximum = ndi.maximum_filter(smooth, size=seed_distance * 2 + 1, mode='constant')
    maxima = (smooth >= maximum - 1e-7) & binary & (distance >= 1)
    maxima_labels, n = ndi.label(maxima)
    seeds = np.zeros(binary.shape, dtype=np.int32)
    seed_index = 0
    for region in ndi.find_objects(maxima_labels):
        if region is None:
            continue
        area = maxima_labels[region] > 0
        scores = np.where(area, distance[region], -1)
        local_y, local_x = np.unravel_index(np.argmax(scores), scores.shape)
        seed_index += 1
        seeds[region[0].start + local_y, region[1].start + local_x] = seed_index
    components, component_count = ndi.label(binary)
    for label, region in enumerate(ndi.find_objects(components), 1):
        if region is None:
            continue
        component = components[region] == label
        if np.any(seeds[region][component]):
            continue
        scores = np.where(component, distance[region], -1)
        y, x = np.unravel_index(np.argmax(scores), scores.shape)
        seed_index += 1
        seeds[region[0].start + y, region[1].start + x] = seed_index
    if seed_index > 20000:
        raise ValidationError('分离种子超过20000，请增大种子间距或最小面积')
    from skimage.segmentation import watershed
    flooded = watershed(
        -distance,
        markers=seeds,
        connectivity=np.ones((3, 3)),
        mask=binary,
        watershed_line=False,
    )
    return flooded.astype(np.int32), {
        'seed_count': seed_index,
        'components_before_split': int(component_count),
        'split': True,
    }


def label_binary(binary, parameters):
    if parameters['split_touching']:
        return split_regions(binary, parameters['seed_distance'])
    structure = ndi.generate_binary_structure(2, 2 if parameters['connectivity'] == 8 else 1)
    labels, count = ndi.label(binary, structure)
    return labels.astype(np.int32), {'split': False, 'component_count': int(count)}


def filter_labels(labels, parameters, selected):
    counts = np.bincount(labels.ravel())
    keep = np.ones(len(counts), dtype=bool)
    keep[0] = False
    keep &= counts >= parameters['min_area']
    keep &= counts <= parameters['max_area']
    edge_ids = set()
    if parameters['exclude_edge']:
        boundary = selected & ~ndi.binary_erosion(selected, border_value=0)
        edge_ids.update(np.unique(labels[boundary]).tolist())
        keep[list(edge_ids)] = False
    ids = np.flatnonzero(keep)
    if len(ids) > 20000:
        raise ValidationError('颗粒超过20000，请提高最小面积或缩小分析区域')
    mapping = np.zeros(len(counts), dtype=np.int32)
    mapping[ids] = np.arange(1, len(ids) + 1)
    result = mapping[labels]
    return result, {
        'accepted_components': int(len(ids)),
        'rejected_by_area': int(np.count_nonzero((counts[1:] < parameters['min_area']) | (counts[1:] > parameters['max_area']))),
        'edge_components': int(len(edge_ids - {0})),
    }


def segment(image, parameters, region):
    selected = region_mask(image.size, region)
    binary, gray, info = threshold_image(image, parameters, selected)
    cleaned = clean_binary(binary, parameters, selected)
    labels, split_info = label_binary(cleaned, parameters)
    labels, filtering_info = filter_labels(labels, parameters, selected)
    info.update(split_info)
    info.update(filtering_info)
    info['foreground_after_cleanup'] = int(np.count_nonzero(labels))
    return labels, gray, selected, info


def mask_edit(labels, command, selected):
    edited = labels.copy()
    operation = command['operation']
    paint = Image.new('1', (labels.shape[1], labels.shape[0]), 0)
    drawing = ImageDraw.Draw(paint)
    if operation in ('add', 'erase'):
        drawing.polygon([tuple(p) for p in command['points']], fill=1)
    elif operation == 'merge':
        drawing.line([tuple(p) for p in command['points']], fill=1, width=2 * command['radius'] + 1)
    painted = np.asarray(paint, dtype=bool) & selected
    if operation == 'remove_object':
        x, y = [int(round(v)) for v in command['point']]
        target = int(edited[y, x])
        if target == 0:
            raise ValidationError('选择的位置没有颗粒')
        edited[edited == target] = 0
    elif operation == 'erase':
        edited[painted] = 0
    else:
        affected = np.unique(edited[painted])
        affected = affected[affected > 0]
        target = int(affected.min()) if len(affected) else int(edited.max()) + 1
        if len(affected):
            edited[np.isin(edited, affected)] = target
        edited[painted] = target
    edited[~selected] = 0
    output = np.zeros(labels.shape, dtype=np.int32)
    next_id = 1
    for object_id, region in enumerate(ndi.find_objects(edited), 1):
        if region is None:
            continue
        local, count = ndi.label(edited[region] == object_id, np.ones((3, 3)))
        for component in range(1, count + 1):
            output[region][local == component] = next_id
            next_id += 1
            if next_id > 20001:
                raise ValidationError('修改后颗粒数超过限制')
    return output


def threshold_sweep(image, parameters, region, thresholds):
    selected = region_mask(image.size, region)
    gray = channel_array(image, parameters['channel'])
    if parameters['sigma']:
        gray = ndi.gaussian_filter(gray, parameters['sigma'])
    result = []
    for value in thresholds:
        if parameters['polarity'] == 'dark':
            binary = gray <= value
        else:
            binary = gray > value
        binary &= selected
        cleaned = clean_binary(binary, parameters, selected)
        labels, _ = label_binary(cleaned, parameters)
        filtered, details = filter_labels(labels, parameters, selected)
        count = details['accepted_components']
        foreground = int(np.count_nonzero(filtered))
        result.append({
            'threshold': float(value),
            'count': count,
            'coverage': round(foreground / max(int(selected.sum()), 1) * 100, 4),
            'foreground_pixels': foreground,
            'mean_area_px': round(foreground / count, 4) if count else 0,
        })
    return result


def evaluate_reference(predicted, reference, valid=None):
    predicted = np.asarray(predicted, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    if predicted.shape != reference.shape:
        raise ValidationError('参考掩膜必须与当前图像尺寸一致')
    if valid is None:
        valid = np.ones(predicted.shape, dtype=bool)
    p = predicted[valid]
    r = reference[valid]
    tp = int(np.count_nonzero(p & r))
    fp = int(np.count_nonzero(p & ~r))
    fn = int(np.count_nonzero(~p & r))
    tn = int(np.count_nonzero(~p & ~r))
    union = tp + fp + fn
    denominator = 2 * tp + fp + fn
    return {
        'true_positive': tp,
        'false_positive': fp,
        'false_negative': fn,
        'true_negative': tn,
        'iou': round(tp / union, 6) if union else 1,
        'dice': round(2 * tp / denominator, 6) if denominator else 1,
        'precision': round(tp / (tp + fp), 6) if tp + fp else None,
        'recall': round(tp / (tp + fn), 6) if tp + fn else None,
        'pixel_accuracy': round((tp + tn) / max(len(p), 1), 6),
    }
