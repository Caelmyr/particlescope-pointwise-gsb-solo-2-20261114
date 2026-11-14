import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
from .imaging import channel_array, disk, region_mask
from .validation import ValidationError


# 与 ndi.binary_fill_holes 默认结构元素一致的 4-连通十字
_FILL_STRUCTURE = ndi.generate_binary_structure(2, 1)


def _sweep_level_array(gray, selected, parameters):
    """把与阈值无关的开/闭运算和填孔合并为一次灰度形态学级联。

    对每个阈值 v，返回的 level 满足：
        亮极性 clean_binary(gray > v) == (level > v) & selected
        暗极性 clean_binary(gray <= v) == (level <= v) & selected
    灰度形态学是二值形态学对所有阈值集的提升，
    灰度孔洞重建与二值填孔逐层对应，因此结果与逐阈值调用完全一致。
    ROI 外的哨兵值和边界常量保证 ROI 边界外不会被当作孔洞填充。
    """
    from skimage.morphology import reconstruction
    light = parameters['polarity'] == 'light'
    infinite = np.float32(np.inf)
    sentinel = -infinite if light else infinite
    level = np.where(selected, gray, sentinel)
    if parameters['opening']:
        structure = disk(parameters['opening'])
        if light:
            level = ndi.grey_erosion(level, footprint=structure,
                                     mode='constant', cval=-infinite)
            level = ndi.grey_dilation(level, footprint=structure,
                                      mode='constant', cval=-infinite)
        else:
            level = ndi.grey_dilation(level, footprint=structure,
                                      mode='constant', cval=infinite)
            level = ndi.grey_erosion(level, footprint=structure,
                                     mode='constant', cval=infinite)
    if parameters['closing']:
        structure = disk(parameters['closing'])
        if light:
            level = ndi.grey_dilation(level, footprint=structure,
                                      mode='constant', cval=-infinite)
            level = ndi.grey_erosion(level, footprint=structure,
                                     mode='constant', cval=-infinite)
        else:
            level = ndi.grey_erosion(level, footprint=structure,
                                     mode='constant', cval=infinite)
            level = ndi.grey_dilation(level, footprint=structure,
                                      mode='constant', cval=infinite)
    if parameters['fill_holes']:
        if light:
            marker = np.full_like(level, infinite)
            marker[0, :] = level[0, :]
            marker[-1, :] = level[-1, :]
            marker[:, 0] = level[:, 0]
            marker[:, -1] = level[:, -1]
            level = reconstruction(marker, level, method='erosion',
                                   footprint=_FILL_STRUCTURE)
        else:
            marker = np.full_like(level, -infinite)
            marker[0, :] = level[0, :]
            marker[-1, :] = level[-1, :]
            marker[:, 0] = level[:, 0]
            marker[:, -1] = level[:, -1]
            level = reconstruction(marker, level, method='dilation',
                                   footprint=_FILL_STRUCTURE)
    return level


def _sweep_filter_summary(labels, parameters, boundary):
    """只统计扫描需要的颗粒数和前景像素，不做全图重编号。

    接受/拒绝判定与 filter_labels 相同：面积区间一致，
    边缘集合来自同一张分析区域边界，结果计数和像素数完全一致。
    """
    counts = np.bincount(labels.ravel())
    keep = (counts >= parameters['min_area']) & (counts <= parameters['max_area'])
    keep[0] = False
    if parameters['exclude_edge']:
        edge_ids = np.unique(labels[boundary])
        keep[edge_ids[edge_ids > 0]] = False
    accepted = np.flatnonzero(keep)
    return int(len(accepted)), int(counts[accepted].sum())


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
    dark = parameters['polarity'] == 'dark'
    roi_pixels = max(int(selected.sum()), 1)

    def summarize(value, labels, details):
        count = details['accepted_components']
        foreground = int(np.count_nonzero(labels))
        return {
            'threshold': float(value),
            'count': count,
            'coverage': round(foreground / roi_pixels * 100, 4),
            'foreground_pixels': foreground,
            'mean_area_px': round(foreground / count, 4) if count else 0,
        }

    result = []
    if parameters['split_touching']:
        # 分水岭依赖每个阈值下的距离场和种子，无法跨阈值共享，保持逐阈值完整流程
        for value in thresholds:
            binary = (gray <= value) if dark else (gray > value)
            binary &= selected
            cleaned = clean_binary(binary, parameters, selected)
            labels, _ = label_binary(cleaned, parameters)
            filtered, details = filter_labels(labels, parameters, selected)
            result.append(summarize(value, filtered, details))
        return result

    # 不分离粘连：平滑只做一次；开/闭/填孔合并为一次灰度形态学级联；
    # 区域边界只计算一次；每个阈值只做二值化、连通标记和计数筛选。
    level = _sweep_level_array(gray, selected, parameters)
    structure = ndi.generate_binary_structure(
        2, 2 if parameters['connectivity'] == 8 else 1)
    boundary = selected & ~ndi.binary_erosion(selected, border_value=0)
    for value in thresholds:
        binary = ((level <= value) if dark else (level > value)) & selected
        labels, _ = ndi.label(binary, structure)
        labels = labels.astype(np.int32)
        count, foreground = _sweep_filter_summary(labels, parameters, boundary)
        result.append({
            'threshold': float(value),
            'count': count,
            'coverage': round(foreground / roi_pixels * 100, 4),
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
