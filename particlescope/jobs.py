import uuid
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from . import validation as v


class JobService:
    def __init__(self, store, analysis, recipes):
        self.store = store
        self.analysis = analysis
        self.recipes = recipes
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='particlescope-batch')
        self.cancellations = {}
        self.futures = {}
        self.recover_interrupted()

    def recover_interrupted(self):
        with self.store.lock:
            for path in self.store.jobs.glob('*.json'):
                job = self.store.json_read(path)
                if job['status'] in ('queued', 'running', 'cancelling'):
                    job['status'] = 'interrupted'
                    job['finished_at'] = v.now()
                    for item in job['items']:
                        if item['status'] in ('queued', 'running'):
                            item['status'] = 'interrupted'
                            item['error'] = '服务重启中断；已完成样本保留'
                    self.store.json_write(path, job)

    def save(self, job):
        job['updated_at'] = v.now()
        self.store.json_write(self.store.jobs / (job['id'] + '.json'), job)

    def get(self, job_id):
        v.identifier(job_id, '任务编号')
        with self.store.lock:
            job = self.store.json_read(self.store.jobs / (job_id + '.json'))
        if job is None:
            raise v.ValidationError('任务不存在')
        job['completed'] = sum(item['status'] == 'completed' for item in job['items'])
        job['failed'] = sum(item['status'] == 'failed' for item in job['items'])
        job['progress_percent'] = round(sum(item['status'] not in ('queued', 'running') for item in job['items']) / max(len(job['items']), 1) * 100, 2)
        return job

    def list(self):
        with self.store.lock:
            ids = [p.stem for p in self.store.jobs.glob('*.json')]
        jobs = [self.get(key) for key in ids]
        return sorted(jobs, key=lambda j: j['created_at'], reverse=True)

    def start(self, sample_ids, recipe_id):
        ids = v.sequence(sample_ids, '样本编号', 100)
        if not ids:
            raise v.ValidationError('至少选择一个样本')
        ids = list(dict.fromkeys(v.identifier(key) for key in ids))
        recipe = self.recipes.get(recipe_id)
        with self.store.lock:
            active = [j for j in self.list() if j['status'] in ('queued', 'running', 'cancelling')]
            if len(active) >= 5:
                raise v.ValidationError('最多排队5个批处理任务')
            snapshots = []
            for key in ids:
                meta = self.store.meta(key)
                snapshots.append({
                    'sample_id': key,
                    'name': meta['name'],
                    'expected_revision': meta['revision'],
                    'status': 'queued',
                    'analysis_id': None,
                    'error': None,
                })
            job = {
                'id': uuid.uuid4().hex,
                'created_at': v.now(),
                'updated_at': v.now(),
                'status': 'queued',
                'recipe': deepcopy(recipe),
                'items': snapshots,
                'finished_at': None,
            }
            self.save(job)
            self.cancellations[job['id']] = Event()
            self.futures[job['id']] = self.executor.submit(self.run, job['id'])
        return self.get(job['id'])

    def run(self, job_id):
        cancellation = self.cancellations[job_id]
        with self.store.lock:
            job = self.get(job_id)
            job['status'] = 'running'
            self.save(job)
        for index in range(len(job['items'])):
            with self.store.lock:
                job = self.get(job_id)
                if cancellation.is_set():
                    for remaining in job['items'][index:]:
                        remaining['status'] = 'cancelled'
                    job['status'] = 'cancelled'
                    job['finished_at'] = v.now()
                    self.save(job)
                    return
                item = job['items'][index]
                item['status'] = 'running'
                self.save(job)
            try:
                with self.store.lock:
                    current = self.store.meta(item['sample_id'])
                    if current['revision'] != item['expected_revision']:
                        raise v.ValidationError('排队后样本已修改，跳过以避免覆盖')
                    for step in job['recipe']['steps']:
                        self.analysis.enhance(item['sample_id'], step, current['revision'])
                        current = self.store.meta(item['sample_id'])
                    result = self.analysis.run(item['sample_id'], job['recipe']['segmentation'], current['revision'])
                    filters = job['recipe'].get('filters', {})
                    if any(value not in (None, []) for value in filters.values()):
                        current = self.store.meta(item['sample_id'])
                        result = self.analysis.filter_measurements(item['sample_id'], filters, current['revision'])
                    item['analysis_id'] = result['id']
                    item['count'] = result['count']
                    item['status'] = 'completed'
            except Exception as exc:
                item['status'] = 'failed'
                item['error'] = str(exc)
            with self.store.lock:
                latest = self.get(job_id)
                latest['items'][index] = item
                self.save(latest)
        with self.store.lock:
            job = self.get(job_id)
            failures = any(i['status'] == 'failed' for i in job['items'])
            job['status'] = 'completed_with_errors' if failures else 'completed'
            job['finished_at'] = v.now()
            self.save(job)

    def cancel(self, job_id):
        with self.store.lock:
            job = self.get(job_id)
            if job['status'] not in ('running', 'queued', 'cancelling'):
                raise v.ValidationError('任务已结束')
            event = self.cancellations.get(job_id)
            if event:
                event.set()
            job['status'] = 'cancelling'
            self.save(job)
        return self.get(job_id)

    def retry(self, job_id):
        job = self.get(job_id)
        if job['status'] in ('running', 'queued', 'cancelling'):
            raise v.ValidationError('运行中的任务不能重试')
        ids = [item['sample_id'] for item in job['items'] if item['status'] in ('failed', 'interrupted', 'cancelled')]
        if not ids:
            raise v.ValidationError('没有可重试的样本')
        recipe = self.recipes.create(job['recipe'])
        return self.start(ids, recipe['id'])
