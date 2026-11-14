from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
target = root / 'particlescope-source.zip'
excluded = {'.venv', '__pycache__', 'data', '.git', '.pytest_cache'}
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if not path.is_file() or set(relative.parts).intersection(excluded):
            continue
        if path == target or path.suffix in ('.pyc', '.log', '.zip'):
            continue
        archive.write(path, 'solo_4/' + relative.as_posix())
print(target)
