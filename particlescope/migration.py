import re
from .validation import now


def migrate_first_version(store, logger):
    marker_path = store.root / 'legacy-imports.json'
    with store.lock:
        marker = store.json_read(marker_path, {'version': 1, 'imports': {}, 'errors': {}})
        candidates = [
            path for path in store.root.iterdir()
            if path.is_dir() and re.fullmatch('[a-f0-9]{32}', path.name)
        ]
        for path in candidates:
            if path.name in marker['imports']:
                continue
            if not (path / 'meta.json').exists() or not (path / 'original.png').exists():
                continue
            try:
                original = store.json_read(path / 'meta.json')
                sample = store.import_image(path / 'original.png', original.get('name', '早期样本'))
                provenance = {
                    'operation': 'legacy_import',
                    'legacy_id': path.name,
                    'source_directory': path.name,
                    'imported_at': now(),
                    'note': '保留早期目录全部数据；新测量引擎需要重新分析',
                }
                store.json_write(store.sample_dir(sample['id']) / 'provenance.json', provenance)
                marker['imports'][path.name] = sample['id']
                marker['errors'].pop(path.name, None)
                store.json_write(marker_path, marker)
            except (OSError, ValueError) as error:
                marker['errors'][path.name] = str(error)
                store.json_write(marker_path, marker)
                logger.warning('早期样本导入失败 %s: %s', path.name, error)
        return marker
