import math
import re
from datetime import datetime, timezone


class ValidationError(ValueError):
    pass


def mapping(value, name='参数'):
    if not isinstance(value, dict):
        raise ValidationError(f'{name}必须是对象')
    return value


def sequence(value, name='列表', maximum=10000):
    if not isinstance(value, list):
        raise ValidationError(f'{name}必须是列表')
    if len(value) > maximum:
        raise ValidationError(f'{name}最多包含{maximum}项')
    return value


def real(value, name, minimum=None, maximum=None):
    if isinstance(value, bool):
        raise ValidationError(f'{name}必须为数值')
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        raise ValidationError(f'{name}必须为数值')
    if not math.isfinite(number):
        raise ValidationError(f'{name}必须是有限数值')
    if minimum is not None and number < minimum:
        raise ValidationError(f'{name}不能小于{minimum}')
    if maximum is not None and number > maximum:
        raise ValidationError(f'{name}不能大于{maximum}')
    return number


def integer(value, name, minimum=None, maximum=None):
    number = real(value, name, minimum, maximum)
    if number != int(number):
        raise ValidationError(f'{name}必须是整数')
    return int(number)


def boolean(value, name):
    if not isinstance(value, bool):
        raise ValidationError(f'{name}必须是布尔值')
    return value


def text(value, name, maximum=500, allow_empty=True):
    if not isinstance(value, str):
        raise ValidationError(f'{name}必须是文本')
    value = value.strip()
    if not allow_empty and not value:
        raise ValidationError(f'{name}不能为空')
    if len(value) > maximum:
        raise ValidationError(f'{name}不能超过{maximum}字符')
    if '\x00' in value:
        raise ValidationError(f'{name}包含无效字符')
    return value


def choice(value, name, options):
    if value not in options:
        raise ValidationError(f'{name}应为：' + '、'.join(map(str, options)))
    return value


def identifier(value, name='编号'):
    if not isinstance(value, str) or not re.fullmatch('[a-f0-9]{32}', value):
        raise ValidationError(f'{name}无效')
    return value


def point(value, width, height, name='坐标'):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValidationError(f'{name}必须包含x和y')
    return [
        real(value[0], name + 'X', 0, width - 1),
        real(value[1], name + 'Y', 0, height - 1),
    ]


def polygon(value, width, height):
    points = sequence(value, '多边形', 200)
    if len(points) < 3:
        raise ValidationError('多边形至少需要三个点')
    vertices = [point(p, width, height) for p in points]
    area = 0
    for i, a in enumerate(vertices):
        b = vertices[(i + 1) % len(vertices)]
        area += a[0] * b[1] - b[0] * a[1]
    if abs(area) < 2:
        raise ValidationError('多边形面积过小')
    return vertices


def rectangle(value, width, height):
    body = mapping(value, '矩形')
    x = integer(body.get('x', 0), '起点X', 0, width - 1)
    y = integer(body.get('y', 0), '起点Y', 0, height - 1)
    w = integer(body.get('width', width - x), '宽度', 1, width - x)
    h = integer(body.get('height', height - y), '高度', 1, height - y)
    return {'x': x, 'y': y, 'width': w, 'height': h}


def now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def calibration(body):
    body = mapping(body, '标定')
    unit = choice(body.get('unit', 'px'), '单位', ['px', 'μm', 'mm', 'nm'])
    pixel_length = real(body.get('pixel_length', 1), '像素长度', 0.000001, 1000000)
    physical_length = real(body.get('physical_length', 1), '实际长度', 0.000001, 1000000)
    if unit == 'px':
        physical_length = pixel_length
    return {
        'unit': unit,
        'pixel_length': pixel_length,
        'physical_length': physical_length,
        'scale': physical_length / pixel_length,
    }


def roi(body, width, height):
    if body is None:
        return {'type': 'full'}
    body = mapping(body, '分析区域')
    kind = choice(body.get('type', 'full'), '区域类型', ['full', 'rectangle', 'polygon', 'ellipse'])
    result = {'type': kind}
    if kind in ('rectangle', 'ellipse'):
        result.update(rectangle(body, width, height))
    if kind == 'polygon':
        result['points'] = polygon(body.get('points'), width, height)
    return result


def segmentation(body):
    body = mapping(body, '分割参数')
    result = {
        'method': choice(body.get('method', 'manual'), '阈值方法', ['manual', 'otsu', 'adaptive', 'percentile']),
        'channel': choice(body.get('channel', 'gray'), '颜色通道', ['gray', 'red', 'green', 'blue', 'hue', 'saturation', 'value']),
        'polarity': choice(body.get('polarity', 'dark'), '极性', ['dark', 'light']),
        'threshold': real(body.get('threshold', 120), '灰度阈值', 0, 255),
        'percentile': real(body.get('percentile', 25), '分位数', 0, 100),
        'sigma': real(body.get('sigma', 1), '平滑强度', 0, 8),
        'adaptive_window': integer(body.get('adaptive_window', 41), '局部窗口', 3, 301),
        'adaptive_offset': real(body.get('adaptive_offset', 10), '局部偏移', -100, 100),
        'min_area': integer(body.get('min_area', 40), '最小面积', 1, 16000000),
        'max_area': integer(body.get('max_area', 16000000), '最大面积', 1, 16000000),
        'opening': integer(body.get('opening', 0), '开运算半径', 0, 20),
        'closing': integer(body.get('closing', 0), '闭运算半径', 0, 20),
        'fill_holes': boolean(body.get('fill_holes', True), '填充孔洞'),
        'exclude_edge': boolean(body.get('exclude_edge', True), '排除边缘'),
        'split_touching': boolean(body.get('split_touching', False), '粘连分离'),
        'seed_distance': integer(body.get('seed_distance', 12), '种子间距', 2, 100),
        'connectivity': integer(body.get('connectivity', 8), '连通性', 4, 8),
    }
    if result['connectivity'] not in (4, 8):
        raise ValidationError('连通性必须为4或8')
    if result['adaptive_window'] % 2 == 0:
        raise ValidationError('局部窗口必须是奇数')
    if result['max_area'] < result['min_area']:
        raise ValidationError('最大面积不能小于最小面积')
    return result


def filters(body):
    body = mapping(body, '测量筛选')
    result = {}
    for name in ['area', 'diameter', 'circularity', 'aspect_ratio', 'solidity', 'mean_intensity']:
        lower = body.get(name + '_min')
        upper = body.get(name + '_max')
        if lower is not None:
            lower = real(lower, name + '下限', 0)
        if upper is not None:
            upper = real(upper, name + '上限', 0)
        if lower is not None and upper is not None and lower > upper:
            raise ValidationError(name + '范围倒置')
        result[name + '_min'] = lower
        result[name + '_max'] = upper
    result['exclude_ids'] = [
        integer(v, '排除颗粒编号', 1)
        for v in sequence(body.get('exclude_ids', []), '排除编号', 20000)
    ]
    return result


def enhancement(body):
    body = mapping(body, '预处理')
    return {
        'operation': choice(body.get('operation'), '图像操作', [
            'gaussian', 'median', 'bilateral', 'normalize', 'gamma',
            'background', 'unsharp', 'invert', 'channel', 'contrast',
            'equalize', 'clahe', 'rotate', 'flip', 'crop', 'resize',
        ]),
        'amount': real(body.get('amount', 1), '强度', 0.01, 100),
        'radius': integer(body.get('radius', 3), '半径', 1, 100),
        'channel': choice(body.get('channel', 'gray'), '通道', ['gray', 'red', 'green', 'blue', 'hue', 'saturation', 'value']),
        'angle': real(body.get('angle', 90), '角度', -360, 360),
        'direction': choice(body.get('direction', 'horizontal'), '翻转方向', ['horizontal', 'vertical']),
        'width': integer(body.get('width', 1024), '目标宽度', 16, 8000),
        'height': integer(body.get('height', 768), '目标高度', 16, 8000),
        'crop': body.get('crop'),
    }


def revision(body):
    return integer(body.get('revision'), '修订号', 0)


def flatfield(body):
    body = mapping(body, '平场校正参数')
    return {
        'saturation_level': real(body.get('saturation_level', 250), '饱和阈值', 1, 255),
        'min_response': real(body.get('min_response', 1), '最小参考响应', 0, 255),
        'min_response_fraction': real(body.get('min_response_fraction', 0.05), '响应比例下限', 0, 1),
        'max_gain': real(body.get('max_gain', 20), '最大增益', 1, 1000),
        'bad_pixel_sigma': real(body.get('bad_pixel_sigma', 6), '坏点判定倍数', 1, 50),
        'margin': integer(body.get('margin', 1), '不可靠区域外扩', 0, 20),
    }


def flatfield_pair(body):
    body = mapping(body, '平场校正')
    return {
        'flat_id': identifier(body.get('flat_id'), '明场参考编号'),
        'dark_id': identifier(body.get('dark_id'), '暗场参考编号'),
        'parameters': flatfield(body.get('parameters', {})),
    }


def flatfield_correction(body):
    value = flatfield_pair(body)
    value.update({
        'sample_id': identifier(body.get('sample_id'), '样本编号'),
        'save': boolean(body.get('save', False), '另存新样本'),
        'name': text(body.get('name', ''), '新样本名称', 180),
    })
    return value


def tags(value):
    items = sequence(value, '标签', 20)
    result = []
    for item in items:
        name = text(item, '标签', 40, False)
        if name not in result:
            result.append(name)
    return result


def edit_command(body, width, height):
    body = mapping(body, '掩膜修改')
    operation = choice(body.get('operation'), '修订方式', ['add', 'erase', 'remove_object', 'merge'])
    result = {'operation': operation}
    if operation in ('add', 'erase'):
        result['points'] = polygon(body.get('points'), width, height)
    elif operation == 'remove_object':
        result['point'] = point(body.get('point'), width, height)
    else:
        result['points'] = [point(p, width, height) for p in sequence(body.get('points'), '连接点', 100)]
        if len(result['points']) < 2:
            raise ValidationError('至少需要两个连接点')
        result['radius'] = integer(body.get('radius', 2), '连接笔刷', 1, 30)
    return result
