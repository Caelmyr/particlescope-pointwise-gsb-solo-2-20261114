import uuid
from copy import deepcopy
from . import validation as v
from .storage import ConflictError


class RecipeService:
    def __init__(self, store):
        self.store = store

    def validate(self, body):
        name = v.text(body.get('name'), '配方名称', 120, False)
        steps = v.sequence(body.get('steps', []), '预处理步骤', 12)
        validated_steps = []
        for step in steps:
            parameters = v.enhancement(step)
            if parameters['operation'] in ('resize', 'crop', 'rotate', 'flip'):
                raise v.ValidationError('批量配方不包含几何操作，避免静默改变标定与区域')
            validated_steps.append(parameters)
        return {
            'name': name,
            'description': v.text(body.get('description', ''), '配方说明', 2000),
            'steps': validated_steps,
            'segmentation': v.segmentation(body.get('segmentation', {})),
            'filters': v.filters(body.get('filters', {})),
        }

    def create(self, body):
        recipe = self.validate(body)
        recipe['id'] = uuid.uuid4().hex
        recipe['created_at'] = v.now()
        recipe['updated_at'] = recipe['created_at']
        recipe['revision'] = 0
        with self.store.lock:
            self.store.json_write(self.store.recipes / (recipe['id'] + '.json'), recipe)
        return recipe

    def get(self, recipe_id):
        v.identifier(recipe_id, '配方编号')
        with self.store.lock:
            recipe = self.store.json_read(self.store.recipes / (recipe_id + '.json'))
        if recipe is None:
            raise v.ValidationError('配方不存在')
        return recipe

    def list(self):
        with self.store.lock:
            recipes = [self.store.json_read(p) for p in self.store.recipes.glob('*.json')]
        return sorted(recipes, key=lambda r: r['updated_at'], reverse=True)

    def update(self, recipe_id, body, revision):
        validated = self.validate(body)
        with self.store.lock:
            recipe = self.get(recipe_id)
            if recipe['revision'] != revision:
                raise ConflictError('配方已被修改，请刷新')
            recipe.update(validated)
            recipe['revision'] += 1
            recipe['updated_at'] = v.now()
            self.store.json_write(self.store.recipes / (recipe_id + '.json'), recipe)
        return recipe

    def from_analysis(self, sample_id, name, description=''):
        result = self.store.result(sample_id)
        if result is None:
            raise v.ValidationError('请先分析样本')
        return self.create({
            'name': name,
            'description': description,
            'steps': [],
            'segmentation': result['parameters'],
            'filters': result.get('measurement_filters', {}),
        })

    def export(self, recipe_id):
        recipe = deepcopy(self.get(recipe_id))
        return {
            'format': 'particlescope-recipe',
            'version': 1,
            'recipe': {key: recipe[key] for key in ['name', 'description', 'steps', 'segmentation', 'filters']},
        }

    def import_recipe(self, body):
        v.mapping(body)
        if body.get('format') != 'particlescope-recipe' or body.get('version') != 1:
            raise v.ValidationError('不支持的配方文件')
        return self.create(v.mapping(body.get('recipe'), '配方'))


BUILTIN_RECIPES = [
    {
        'name': '暗色离散微球',
        'description': '均匀亮背景，Otsu全局分割，不拆分粘连。',
        'steps': [],
        'segmentation': {'method': 'otsu', 'polarity': 'dark', 'min_area': 40},
    },
    {
        'name': '暗色粘连颗粒',
        'description': '距离变换种子分水岭，需根据颗粒尺寸调整种子间距。',
        'steps': [],
        'segmentation': {'method': 'otsu', 'polarity': 'dark', 'min_area': 40, 'split_touching': True, 'seed_distance': 16},
    },
    {
        'name': '非均匀背景',
        'description': '局部阈值适合有缓慢亮度变化的暗颗粒图。',
        'steps': [],
        'segmentation': {'method': 'adaptive', 'polarity': 'dark', 'min_area': 40, 'adaptive_window': 61, 'adaptive_offset': 12},
    },
    {
        'name': '亮色荧光颗粒',
        'description': '分析绿色通道亮区域；按图像实际信号调整阈值。',
        'steps': [],
        'segmentation': {'method': 'otsu', 'channel': 'green', 'polarity': 'light', 'min_area': 20},
    },
]


def ensure_builtins(service):
    marker = service.store.root / 'builtin-recipes.json'
    with service.store.lock:
        if marker.exists():
            return
        ids = []
        for body in BUILTIN_RECIPES:
            ids.append(service.create(body)['id'])
        service.store.json_write(marker, {'ids': ids, 'version': 1})
