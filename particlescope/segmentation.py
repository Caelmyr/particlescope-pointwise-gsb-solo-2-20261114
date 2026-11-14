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


LINEAGE_RELATIONS = ('unchanged', 'modified', 'split', 'merged', 'created', 'removed')


def _paint_command(command, shape, selected):
    paint = Image.new('1', (shape[1], shape[0]), 0)
    drawing = ImageDraw.Draw(paint)
    operation = command['operation']
    if operation in ('add', 'erase'):
        drawing.polygon([tuple(p) for p in command['points']], fill=1)
    elif operation == 'merge':
        drawing.line([tuple(p) for p in command['points']], fill=1, width=2 * command['radius'] + 1)
    return np.asarray(paint, dtype=bool) & selected


def mask_edit(labels, command, selected, high_water=None):
    """Apply a local mask revision while preserving untouched object identities.

    Objects whose pixels are never touched keep their exact label id. Only
    objects intersecting the edited region are relabelled; fresh ids are
    allocated above every id ever used so an id is never silently reused for a
    different object. The returned lineage maps every old/new id explicitly,
    including split and merge relationships.

    ``high_water`` is the largest id ever allocated in previous revisions
    (including ids since deleted). It defaults to the current maximum and is
    returned in the lineage so callers can persist it.
    """
    edited = labels.copy()
    operation = command['operation']
    previous_max = int(labels.max())
    high_water = int(high_water) if high_water is not None else previous_max
    next_id = high_water + 1
    lineage = {
        'operation': operation,
        'command': dict(command),
        'parent_max_id': previous_max,
        'high_water': high_water,
        'relations': [],
        'old_ids': sorted(int(value) for value in np.unique(labels) if value > 0),
        'new_ids': [],
    }

    def record(relation, old_ids=None, new_ids=None):
        entry = {'relation': relation}
        if old_ids is not None:
            entry['old_ids'] = sorted(int(value) for value in old_ids)
        if new_ids is not None:
            entry['new_ids'] = sorted(int(value) for value in new_ids)
        lineage['relations'].append(entry)

    if operation == 'remove_object':
        x, y = [int(round(v)) for v in command['point']]
        target = int(edited[y, x])
        if target == 0:
            raise ValidationError('选择的位置没有颗粒')
        edited[edited == target] = 0
        record('removed', old_ids=[target], new_ids=[])
    else:
        painted = _paint_command(command, labels.shape, selected)
        # Painted pixels are already clipped to the ROI, so only objects with
        # actual pixels inside the editable region count as affected.
        affected = [int(value) for value in np.unique(labels[painted]) if value > 0]
        if not np.any(painted):
            # Nothing lies inside the editable region: return labels unchanged
            # rather than erasing pixels outside the ROI.
            for old_id in lineage['old_ids']:
                record('unchanged', old_ids=[old_id], new_ids=[old_id])
            lineage['new_ids'] = list(lineage['old_ids'])
            lineage['counts'] = {relation: sum(1 for entry in lineage['relations'] if entry['relation'] == relation) for relation in LINEAGE_RELATIONS}
            return labels.copy(), lineage
        if operation == 'merge' and not affected:
            raise ValidationError('连接区域没有接触任何颗粒')
        if operation == 'erase':
            edited[painted] = 0
        else:
            if affected:
                target = int(min(affected))
                edited[np.isin(edited, affected)] = target
                edited[painted] = target
            # When nothing is touched, painted pixels stay background (id 0)
            # and are picked up as new components by the relabelling step.

        # Build the relabelling region. Erase needs the post-erasure mask
        # (the gap must split components); add/merge need the original labels
        # so the parent ids stay traceable after the merge rewrite.
        affected_set = set(affected)
        if operation == 'erase':
            affected_mask = np.isin(edited, list(affected_set)) if affected_set else np.zeros(labels.shape, dtype=bool)
        else:
            affected_mask = np.isin(labels, list(affected_set)) if affected_set else np.zeros(labels.shape, dtype=bool)
            if operation == 'add':
                affected_mask |= painted & (labels == 0)
        components, component_count = ndi.label(affected_mask, np.ones((3, 3)))
        old_by_component = {
            int(index): sorted(int(value) for value in np.unique(labels[components == index]) if value > 0)
            for index in range(1, component_count + 1)
        }
        # Only components that still have foreground after the edit survive.
        # A pure addition starts as background in edited, so painted pixels
        # count as surviving foreground for add.
        def component_survives(index):
            region = components == index
            if operation == 'add':
                return np.any(region & (edited > 0)) or np.any(region & painted)
            return np.any(edited[region] > 0)

        surviving = [index for index in range(1, component_count + 1) if component_survives(index)]
        new_ids = list(range(next_id, next_id + len(surviving)))
        if next_id + len(surviving) - 1 > 20000:
            raise ValidationError('修改后颗粒数超过限制')
        # Rebuild from scratch: untouched objects keep their original id,
        # affected surviving components receive freshly allocated ids.
        output = np.zeros(labels.shape, dtype=np.int32)
        untouched = ~np.isin(labels, list(affected_set)) if affected_set else np.ones(labels.shape, dtype=bool)
        output[untouched] = labels[untouched]
        for component_index, new_id in zip(surviving, new_ids):
            output[components == component_index] = new_id
        edited = output
        next_id += len(surviving)

        if operation == 'erase':
            surviving_old_in_region = {
                value for ids in old_by_component.values() for value in ids
            }
            for old_id in affected:
                if old_id not in surviving_old_in_region:
                    record('removed', old_ids=[old_id], new_ids=[])
                    continue
                children = sorted(
                    new_id
                    for component_index, new_id in zip(surviving, new_ids)
                    if old_id in old_by_component[component_index]
                )
                if len(children) == 1 and old_by_component[surviving[new_ids.index(children[0])]] == [old_id]:
                    record('modified', old_ids=[old_id], new_ids=children)
                else:
                    record('split', old_ids=[old_id], new_ids=children)
        elif operation == 'add':
            for component_index, new_id in zip(surviving, new_ids):
                parents = old_by_component[component_index]
                if not parents:
                    record('created', old_ids=[], new_ids=[new_id])
                elif len(parents) == 1:
                    old_id = parents[0]
                    siblings = [
                        candidate
                        for index, candidate in zip(surviving, new_ids)
                        if old_id in old_by_component[index]
                    ]
                    relation = 'split' if len(siblings) > 1 else 'modified'
                    record(relation, old_ids=[old_id], new_ids=[new_id])
                else:
                    record('merged', old_ids=parents, new_ids=[new_id])
        else:  # merge: every touched object is meant to become one object
            surviving_parents = sorted({
                value for ids in old_by_component.values() for value in ids
            })
            for new_id in new_ids:
                record('merged', old_ids=surviving_parents or affected, new_ids=[new_id])

    edited[~selected] = 0
    lineage['new_ids'] = sorted(int(value) for value in np.unique(edited) if value > 0)
    lineage['high_water'] = max(high_water, int(edited.max()))
    handled_old = {
        old_id
        for entry in lineage['relations']
        for old_id in entry.get('old_ids', [])
    }
    for old_id in lineage['old_ids']:
        if old_id not in handled_old:
            record('unchanged', old_ids=[old_id], new_ids=[old_id])
    lineage['relations'].sort(key=lambda entry: (entry['relation'], entry.get('old_ids', [0])[0] if entry.get('old_ids') else 0))
    lineage['counts'] = {relation: sum(1 for entry in lineage['relations'] if entry['relation'] == relation) for relation in LINEAGE_RELATIONS}
    return edited.astype(np.int32), lineage


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
