import hashlib
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if sys.version_info < (3, 12):
    raise SystemExit('请安装Python 3.12或更高版本；依赖按Python 3.12环境锁定。')
requirements = root / 'requirements.txt'
fingerprint = hashlib.sha256(requirements.read_bytes()).hexdigest()
marker = Path(sys.prefix) / '.particlescope-requirements'
if not marker.exists() or marker.read_text() != fingerprint:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(requirements)], check=True)
    marker.write_text(fingerprint)
