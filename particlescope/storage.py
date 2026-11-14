"""Archival and persistence helpers for ParticleScope."""

from pathlib import Path
from contextlib import contextmanager
from threading import RLock
from copy import deepcopy
import hashlib
import json
import os
import shutil
import uuid
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from .validation import ValidationError, identifier, text, tags, now


class ConflictError(ValueError):
    pass


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.samples = self.root / 'samples'
        self.projects = self.root / 'projects'
        self.recipes = self.root / 'recipes'
        self.jobs = self.root / 'jobs'
        self.lock = RLock()
        for path in [self.root, self.samples, self.projects, self.recipes, self.jobs]:
            path.mkdir(parents=True, exist_ok=True)

    def json_read(self, path, default=None):
        if not path.exists():
            return deepcopy(default)
        try:
            return json.loads(path.read_text(encoding='utf8'))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValidationError('数据文件无法读取：' + path.name) from exc

    def json_write(self, path, value):
        temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            with temporary.open('w', encoding='utf8') as stream:
                json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def image_write(self, path, image):
        temporary = path.with_name(path.stem + '.' + uuid.uuid4().hex + '.tmp')
        try:
            image.save(temporary, format='PNG')
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def sample_dir(self, sample_id):
        identifier(sample_id)
        path = self.samples / sample_id
        if not (path / 'meta.json').exists():
            raise ValidationError('样本不存在')
        return path

    def meta(self, sample_id):
        path = self.sample_dir(sample_id)
        return self.json_read(path / 'meta.json')

    def list_samples(self, project_id=None, query='', include_archived=False):
        result = []
        query = query.casefold().strip()
        with self.lock:
            for path in self.samples.glob('*/meta.json'):
                meta = self.json_read(path)
                if meta.get('archived') and not include_archived:
                    continue
                if project_id and meta.get('project_id') != project_id:
                    continue
                searchable = ' '.join([meta['name'], meta.get('notes', ''), *meta.get('tags', [])])
                if query and query not in searchable.casefold():
                    continue
                result.append(meta)
        return sorted(result, key=lambda m: m.get('updated_at', ''), reverse=True)

    def import_image(self, stream, filename, project_id=None, unreliable_stream=None, provenance=None, calibration=None):
        if project_id:
            self.project(project_id)
        try:
            with Image.open(stream) as decoded:
                if decoded.width * decoded.height > 16000000:
                    raise ValidationError('图像不超过1600万像素')
                if min(decoded.size) < 16:
                    raise ValidationError('图像最小尺寸为16×16')
                original_format = decoded.format
                original_mode = decoded.mode
                source = ImageOps.exif_transpose(decoded)
                rgba = source.convert('RGBA')
                background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
                background.alpha_composite(rgba)
                image = background.convert('RGB')
                image.load()
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise ValidationError('无法解码图像，支持PNG、JPEG、TIFF、BMP、WebP') from exc
        unreliable = None
        if unreliable_stream is not None:
            try:
                with Image.open(unreliable_stream) as decoded:
                    if decoded.size != image.size:
                        raise ValidationError('不可靠区域掩膜必须与新样本同尺寸')
                    unreliable = decoded.convert('L').copy()
            except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
                raise ValidationError('不可靠区域掩膜不能解码') from exc
        with self.lock:
            sample_id = uuid.uuid4().hex
            path = self.samples / sample_id
            path.mkdir()
            (path / 'history').mkdir()
            (path / 'analyses').mkdir()
            (path / 'annotations').mkdir()
            self.image_write(path / 'original.png', image)
            self.image_write(path / 'working.png', image)
            thumbnail = image.copy()
            thumbnail.thumbnail((320, 220))
            self.image_write(path / 'thumb.png', thumbnail)
            timestamp = now()
            meta = {
                'id': sample_id,
                'name': text(Path(filename).name or '未命名图像', '文件名', 180, False),
                'width': image.width,
                'height': image.height,
                'original_width': image.width,
                'original_height': image.height,
                'source_format': original_format,
                'source_mode': original_mode,
                'pixel_sha256': hashlib.sha256(image.tobytes()).hexdigest(),
                'project_id': project_id,
                'tags': [],
                'notes': '',
                'created_at': timestamp,
                'updated_at': timestamp,
                'revision': 0,
                'image_revision': 0,
                'analysis_id': None,
                'roi': {'type': 'full'},
                'calibration': {'unit': 'px', 'scale': 1, 'pixel_length': 1, 'physical_length': 1},
                'archived': False,
                'history': [],
                'unreliable_mask': unreliable is not None,
            }
            if unreliable is not None:
                self.image_write(path / 'unreliable.png', unreliable)
            if calibration:
                meta['calibration'] = deepcopy(calibration)
            if provenance:
                self.json_write(path / 'provenance.json', provenance)
            self.json_write(path / 'meta.json', meta)
            self.json_write(path / 'annotations.json', [])
            return meta

    @contextmanager
    def transaction(self, sample_id, expected_revision=None):
        with self.lock:
            path = self.sample_dir(sample_id)
            meta = self.json_read(path / 'meta.json')
            if expected_revision is not None and meta['revision'] != expected_revision:
                raise ConflictError('样本已被其他操作修改，请刷新后重试')
            working = deepcopy(meta)
            yield path, working
            working['revision'] = meta['revision'] + 1
            working['updated_at'] = now()
            self.json_write(path / 'meta.json', working)

    def update_meta(self, sample_id, body, expected_revision=None):
        with self.transaction(sample_id, expected_revision) as (_, meta):
            if 'name' in body:
                meta['name'] = text(body['name'], '样本名称', 180, False)
            if 'notes' in body:
                meta['notes'] = text(body['notes'], '备注', 10000)
            if 'tags' in body:
                meta['tags'] = tags(body['tags'])
            if 'project_id' in body:
                project_id = body['project_id']
                if project_id:
                    self.project(project_id)
                meta['project_id'] = project_id
        return self.meta(sample_id)

    def archive(self, sample_id, value, expected_revision=None):
        with self.transaction(sample_id, expected_revision) as (_, meta):
            meta['archived'] = value
        return self.meta(sample_id)

    def working_image(self, sample_id):
        path = self.sample_dir(sample_id)
        with Image.open(path / 'working.png') as image:
            return image.convert('RGB')

    def reliable_mask(self, sample_id):
        """Boolean mask of pixels the analysis may use; None means all pixels."""
        path = self.sample_dir(sample_id) / 'unreliable.png'
        if not path.exists():
            return None
        with Image.open(path) as image:
            return np.asarray(image.convert('L')) <= 127

    def original_image(self, sample_id):
        path = self.sample_dir(sample_id)
        with Image.open(path / 'original.png') as image:
            return image.convert('RGB')

    def commit_image(self, sample_id, image, operation, parameters, expected_revision=None):
        with self.transaction(sample_id, expected_revision) as (path, meta):
            version = uuid.uuid4().hex
            previous = f'history/{version}.png'
            shutil.copyfile(path / 'working.png', path / previous)
            meta['history'].append({
                'id': version,
                'operation': operation,
                'parameters': parameters,
                'path': previous,
                'width': meta['width'],
                'height': meta['height'],
                'roi': deepcopy(meta['roi']),
                'calibration': deepcopy(meta['calibration']),
                'time': now(),
            })
            self.image_write(path / 'working.png', image)
            thumbnail = image.copy()
            thumbnail.thumbnail((320, 220))
            self.image_write(path / 'thumb.png', thumbnail)
            geometry_changed = image.size != (meta['width'], meta['height'])
            meta['width'], meta['height'] = image.size
            if geometry_changed or operation in ('rotate', 'flip', 'crop', 'resize', 'restore'):
                meta['roi'] = {'type': 'full'}
                meta['calibration'] = {'unit': 'px', 'scale': 1, 'pixel_length': 1, 'physical_length': 1}
                self.json_write(path / 'annotations.json', [])
                (path / 'unreliable.png').unlink(missing_ok=True)
                meta['unreliable_mask'] = False
            meta['image_revision'] += 1
            meta['analysis_id'] = None
        return self.meta(sample_id)

    def restore_image(self, sample_id, version_id, expected_revision=None):
        meta = self.meta(sample_id)
        if version_id == 'original':
            image = self.original_image(sample_id)
        else:
            identifier(version_id, '图像版本')
            entry = next((h for h in meta['history'] if h['id'] == version_id), None)
            if entry is None:
                raise ValidationError('图像版本不存在')
            with Image.open(self.sample_dir(sample_id) / entry['path']) as source:
                image = source.convert('RGB')
        return self.commit_image(sample_id, image, 'restore', {'version_id': version_id}, expected_revision)

    def duplicate(self, sample_id):
        with self.lock:
            meta = self.meta(sample_id)
            path = self.sample_dir(sample_id)
            new_id = uuid.uuid4().hex
            target = self.samples / new_id
            shutil.copytree(path, target)
            meta['id'] = new_id
            meta['name'] = meta['name'][:170] + ' · 副本'
            meta['created_at'] = now()
            meta['updated_at'] = now()
            meta['revision'] = 0
            meta['archived'] = False
            self.json_write(target / 'meta.json', meta)
            return meta

    def project(self, project_id):
        identifier(project_id, '项目编号')
        path = self.projects / (project_id + '.json')
        project = self.json_read(path)
        if project is None:
            raise ValidationError('项目不存在')
        return project

    def create_project(self, name, description=''):
        with self.lock:
            project = {
                'id': uuid.uuid4().hex,
                'name': text(name, '项目名称', 100, False),
                'description': text(description, '项目说明', 2000),
                'created_at': now(),
            }
            self.json_write(self.projects / (project['id'] + '.json'), project)
            return project

    def list_projects(self):
        with self.lock:
            result = [self.json_read(p) for p in self.projects.glob('*.json')]
            for project in result:
                project['sample_count'] = len(self.list_samples(project['id']))
            return sorted(result, key=lambda p: p['created_at'], reverse=True)

    def result(self, sample_id, analysis_id=None):
        with self.lock:
            meta = self.meta(sample_id)
            analysis_id = analysis_id or meta.get('analysis_id')
            if not analysis_id:
                return None
            identifier(analysis_id, '分析编号')
            path = self.sample_dir(sample_id) / 'analyses' / analysis_id / 'result.json'
            result = self.json_read(path)
            if result is None:
                raise ValidationError('分析记录不存在')
            result['stale'] = result['image_revision'] != meta['image_revision']
            return result

    def list_analyses(self, sample_id):
        with self.lock:
            directory = self.sample_dir(sample_id) / 'analyses'
            result = []
            for path in directory.glob('*/result.json'):
                data = self.json_read(path)
                result.append({k: data.get(k) for k in [
                    'id', 'created_at', 'count', 'coverage', 'mean_diameter',
                    'image_revision', 'parameters', 'source',
                ]})
            return sorted(result, key=lambda r: r['created_at'], reverse=True)

    def annotations(self, sample_id):
        return self.json_read(self.sample_dir(sample_id) / 'annotations.json', [])
