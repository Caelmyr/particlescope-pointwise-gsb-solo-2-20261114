import csv
import io
import json
import zipfile
from pathlib import PurePosixPath
import numpy as np
from PIL import Image
from .validation import ValidationError
from .measurements import included_rows, measurement_units
from .rendering import png_bytes, contact_sheet, histogram_chart
from .statistics import statistics_report


CSV_FIELDS = [
    'id', 'included', 'area_px', 'area', 'diameter', 'perimeter',
    'centroid_x', 'centroid_y', 'major_axis', 'minor_axis',
    'feret_max', 'feret_min', 'orientation', 'eccentricity',
    'circularity', 'aspect_ratio', 'solidity', 'extent', 'roughness',
    'mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity',
    'median_intensity', 'integrated_intensity', 'bbox_x', 'bbox_y',
    'bbox_width', 'bbox_height', 'hole_count', 'hole_area_px',
    'touches_edge', 'touches_roi',
]


def csv_bytes(rows, fields, header=True):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    if header:
        writer.writeheader()
    for row in rows:
        safe = {}
        for key in fields:
            value = row.get(key)
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            if isinstance(value, str) and value.startswith(('=', '+', '-', '@', '\t', '\r')):
                value = "'" + value
            safe[key] = value
        writer.writerow(safe)
    return ('\ufeff' + stream.getvalue()).encode('utf8')


def particle_csv(result, include_excluded=True):
    rows = result['particles'] if include_excluded else included_rows(result)
    return csv_bytes(rows, CSV_FIELDS)


def annotation_csv(annotations):
    rows = []
    for item in annotations:
        row = {
            'id': item['id'],
            'type': item['type'],
            'label': item['label'],
            'unit': item['unit'],
            'points': item['points'],
        }
        row.update(item['measurement'])
        rows.append(row)
    fields = ['id', 'type', 'label', 'unit', 'points', 'length', 'length_px', 'angle', 'area', 'area_px', 'perimeter', 'centroid_x', 'centroid_y']
    return csv_bytes(rows, fields)


def summary_csv(results):
    fields = [
        'sample_id', 'sample_name', 'id', 'count', 'excluded_count',
        'coverage', 'mean_diameter', 'median_diameter', 'total_area',
        'analyzed_area', 'number_density', 'unit', 'scale', 'created_at',
    ]
    return csv_bytes(results, fields)


def result_json(result):
    return json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False).encode('utf8')


def geojson(result):
    features = []
    for row in result['particles']:
        properties = {key: value for key, value in row.items() if not key.startswith('centroid')}
        properties['unit'] = result['unit']
        features.append({
            'type': 'Feature',
            'id': row['id'],
            'geometry': {'type': 'Point', 'coordinates': [row['centroid_x'], row['centroid_y']]},
            'properties': properties,
        })
    return {
        'type': 'FeatureCollection',
        'name': 'Particle centroids in image pixel coordinates',
        'coordinate_system': 'image pixels: x right, y down, origin upper-left; not geographic',
        'features': features,
    }


def binary_tiff(labels, result):
    ids = [row['id'] for row in included_rows(result)]
    mask = np.isin(labels, ids) & (labels > 0)
    image = Image.fromarray((mask * 255).astype(np.uint8))
    stream = io.BytesIO()
    image.save(stream, format='TIFF', compression='tiff_lzw')
    return stream.getvalue()


def label_tiff(labels):
    image = Image.fromarray(labels.astype(np.int32), mode='I')
    stream = io.BytesIO()
    image.save(stream, format='TIFF', compression='tiff_deflate')
    return stream.getvalue()


def contour_svg(labels, result):
    from skimage.measure import find_contours
    # Each object's unabridged raster contour is retained. This deliberately
    # favors an auditable vector representation over path simplification.
    paths = []
    for row in included_rows(result):
        binary = np.pad((labels == row['id']).astype(np.uint8), 1)
        loops = find_contours(binary, .5, fully_connected='high')
        parts = []
        for loop in loops:
            if len(loop) < 3:
                continue
            points = [(float(x - 1), float(y - 1)) for y, x in loop]
            parts.append('M ' + ' L '.join(f'{x:.3f},{y:.3f}' for x, y in points) + ' Z')
        color = '#%06x' % ((row['id'] * 2654435761) & 0xffffff)
        paths.append(f'<path data-particle-id="{row["id"]}" fill="{color}" fill-rule="evenodd" d="{" ".join(parts)}"/>')
    width, height = labels.shape[1], labels.shape[0]
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-0.5 -0.5 {width} {height}" '
            f'width="{width}" height="{height}">' + ''.join(paths) + '</svg>').encode('utf-8')


def analysis_bundle(store, service, sample_id):
    with store.lock:
        meta = store.meta(sample_id)
        result = service.require_result(sample_id)
        path = store.sample_dir(sample_id) / 'analyses' / result['id']
        labels = service.labels(sample_id)
        report = statistics_report(result)
        annotations = store.annotations(sample_id)
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('sample.json', result_json(meta))
            archive.writestr('analysis.json', result_json(result))
            archive.writestr('statistics.json', result_json(report))
            archive.writestr('measurement-units.json', result_json(measurement_units(result['calibration'])))
            archive.writestr('particles.csv', particle_csv(result))
            archive.writestr('annotations.csv', annotation_csv(annotations))
            archive.writestr('centroids.geojson', result_json(geojson(result)))
            archive.writestr('labels.tiff', label_tiff(labels))
            archive.writestr('distribution.png', png_bytes(histogram_chart(report['number_distribution'])))
            for name in ['mask.png', 'overlay.png', 'labels.png']:
                archive.write(path / name, name)
            archive.write(store.sample_dir(sample_id) / 'working.png', 'working.png')
            unreliable = store.sample_dir(sample_id) / 'unreliable.png'
            if unreliable.exists():
                archive.write(unreliable, 'unreliable-regions.png')
            archive.writestr('README.txt', 'ParticleScope analysis export. Coordinates are image pixels.\nPhysical lengths use calibration in sample.json.\nVolume weighting assumes equivalent spheres.\n')
        return stream.getvalue()


def project_summary(store, sample_ids):
    results = []
    for sample_id in sample_ids:
        meta = store.meta(sample_id)
        result = store.result(sample_id)
        if result is None or result.get('stale'):
            continue
        result['sample_name'] = meta['name']
        results.append(result)
    return summary_csv(results)


def particle_gallery_zip(store, service, sample_id, isolate=True):
    from .measurements import particle_crop
    with store.lock:
        result = service.require_result(sample_id)
        rows = included_rows(result)
        if len(rows) > 600:
            raise ValidationError('颗粒裁片导出最多600个，请先筛选')
        labels = service.labels(sample_id)
        image = store.working_image(sample_id)
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
            for row in rows:
                crop = particle_crop(image, labels, row['id'], 5, isolate)
                archive.writestr(f'particle-{row["id"]:05d}.png', png_bytes(crop))
            archive.writestr('particles.csv', particle_csv(result, False))
        return stream.getvalue()


def validate_archive_paths(names):
    seen = set()
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
            raise ValidationError('归档包含无效路径')
        normalized = str(path)
        if normalized in seen:
            raise ValidationError('归档包含重复路径')
        seen.add(normalized)
    return seen
