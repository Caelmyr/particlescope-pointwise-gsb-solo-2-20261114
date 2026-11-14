from copy import deepcopy


def number(name, label, default, minimum=None, maximum=None, step='any'):
    return {
        'name': name,
        'label': label,
        'type': 'number',
        'default': default,
        'min': minimum,
        'max': maximum,
        'step': step,
    }


def select(name, label, default, options):
    return {
        'name': name,
        'label': label,
        'type': 'select',
        'default': default,
        'options': [{'value': value, 'label': title} for value, title in options],
    }


def checkbox(name, label, default=False):
    return {'name': name, 'label': label, 'type': 'checkbox', 'default': default}


def coordinates(name, label, default):
    return {'name': name, 'label': label, 'type': 'array', 'default': default}


def tool(key, label, group, path, method='GET', fields=None, note='', output='json'):
    return {
        'key': key,
        'label': label,
        'group': group,
        'path': path,
        'method': method,
        'fields': fields or [],
        'note': note,
        'output': output,
    }


MEASUREMENTS = [
    ('diameter', '等效直径'),
    ('area', '面积'),
    ('circularity', '圆度'),
    ('aspect_ratio', '长宽比'),
    ('solidity', '实心度'),
    ('mean_intensity', '平均灰度'),
    ('orientation', '方向角'),
    ('feret_max', '最大Feret径'),
    ('major_axis', '长轴'),
    ('minor_axis', '短轴'),
]
CHANNELS = [('gray', '灰度'), ('red', '红'), ('green', '绿'), ('blue', '蓝')]


def tool_catalog():
    tools = []
    tools.append(tool(
        'quality', '图像质量诊断', '图像', 'quality',
        note='显示清晰度、曝光截断、照明变化和对比度指标。诊断是启发式提示。',
    ))
    tools.append(tool(
        'histogram', '颜色与灰度直方图', '图像', 'histogram',
        note='统计当前ROI中的像素，不依赖颗粒分析。',
    ))
    tools.append(tool(
        'profile', '灰度线剖面', '图像', 'profile', 'POST', [
            coordinates('start', '起点 [x,y]', [20, 100]),
            coordinates('end', '终点 [x,y]', [800, 100]),
            number('samples', '采样点数', 200, 2, 2000, 1),
            number('line_width', '平均线宽 px', 1, 1, 51, 1),
            select('channel', '分析通道', 'gray', CHANNELS),
        ], '也可在画布选择“剖面”工具点击两个端点。',
    ))
    tools.append(tool(
        'statistics', '完整测量统计', '统计', 'statistics',
        note='均值、四分位、置信区间、相关矩阵和异常值均以纳入的颗粒计算。',
    ))
    tools.append(tool(
        'distribution', '加权粒径分布', '统计', 'distribution', 'POST', [
            select('weighting', '分布权重', 'number', [('number', '数量'), ('area', '面积'), ('volume', '等效球体积')]),
            number('bins', '分箱数量', 15, 3, 100, 1),
            checkbox('logarithmic', '对数分箱'),
        ], '体积权重假设等效球，不代表真实三维粒径。',
    ))
    tools.append(tool(
        'bootstrap', '均值重采样区间', '统计', 'bootstrap', 'POST', [
            select('field', '测量指标', 'diameter', MEASUREMENTS),
            number('repeats', '重采样次数', 1000, 100, 3000, 1),
            number('confidence', '置信水平', 0.95, 0.5, 0.999),
        ], '固定随机种子，可复现；不替代独立实验重复。',
    ))
    tools.append(tool(
        'regression', '指标散点与回归', '统计', 'regression', 'POST', [
            select('x', 'X指标', 'diameter', MEASUREMENTS),
            select('y', 'Y指标', 'circularity', MEASUREMENTS),
        ], '自变量须有变化，至少三个颗粒。',
    ))
    tools.append(tool(
        'size-classes', '自定义粒径分级', '统计', 'size-classes', 'POST', [
            coordinates('boundaries', '分级边界（当前长度单位）', [0, 20, 40, 60, 80, 120]),
        ], '区间左闭右开，最后一区间右端点也计入。范围外另行统计。',
    ))
    tools.append(tool(
        'spatial', '最近邻与密度格网', '空间', 'spatial',
        note='质心间距、格网密度和ROI边缘距离。随机距离参照未做边缘修正。',
    ))
    tools.append(tool(
        'clusters', '颗粒空间聚类', '空间', 'clusters', 'POST', [
            number('radius', '邻域半径（当前长度单位）', 100, 0.000001),
            number('min_points', '最小邻域点数', 3, 2, 1000, 1),
        ], '基于密度的聚类，邻域点数包括自身。',
    ))
    tools.append(tool(
        'radial-density', '中心环带密度', '空间', 'radial-density', 'POST', [
            coordinates('center', '中心 [x,y]', [480, 320]),
            number('bins', '环带数量', 12, 3, 100, 1),
        ], '各环带按ROI内实际有效像素面积归一化。',
    ))
    tools.append(tool(
        'pair-correlation', '径向对分布', '空间', 'pair-correlation',
        note='最多5000个颗粒，无边缘修正，仅适合全图矩形观察窗的探索。',
    ))
    tools.append(tool(
        'ripley', 'Ripley K函数', '空间', 'ripley',
        note='最多5000个颗粒，无边缘修正，L(r)-r用于探索聚集尺度。',
    ))
    tools.append(tool(
        'texture', '共生矩阵与局部纹理', '纹理', 'texture',
        note='四方向GLCM、局部二值模式、梯度方向和多尺度细节。',
    ))
    tools.append(tool(
        'spectrum', '径向频谱', '纹理', 'spectrum',
        note='最大400万像素，移除均值并加Hann窗，输出频率及对应周期。',
    ))
    tools.append(tool(
        'color', '颜色空间统计', '颜色', 'color',
        note='RGB、Lab D65、色度及通道相关性。Lab最多采样50万像素。',
    ))
    tools.append(tool(
        'dominant-colors', '主色聚类', '颜色', 'dominant-colors', 'POST', [
            number('clusters', '主色数量', 4, 2, 10, 1),
        ], '最多采样5万像素，在Lab空间聚类。',
    ))
    tools.append(tool(
        'color-distance', '目标色差匹配', '颜色', 'color-distance', 'POST', [
            coordinates('rgb', '目标RGB [r,g,b]', [50, 70, 80]),
            number('tolerance', 'ΔE76容限', 20, 0, 200),
        ], '输出目标颜色覆盖率、连通区域和色差统计。',
    ))
    tools.append(tool(
        'fluorescence', '荧光通道定量', '颜色', 'fluorescence', 'POST', [
            select('channel', '荧光通道', 'green', CHANNELS),
            number('background_percentile', '背景估计分位数', 10, 0, 50),
        ], '根据区域低分位估计背景，输出扣背景强度和饱和比例。',
    ))
    tools.append(tool(
        'morphology', '孔隙与厚度分析', '形态', 'morphology',
        note='前景、孔隙率、内切直径和多尺度盒计数；依赖当前分割结果。',
    ))
    tools.append(tool(
        'granulometry', '形态学颗粒谱', '形态', 'granulometry', 'POST', [
            coordinates('radii', '递增圆盘半径 px', [1, 2, 4, 8, 12]),
        ], '按不同结构元素开运算，不等同于逐颗粒粒径。',
    ))
    tools.append(tool(
        'skeleton', '细化骨架分析', '形态', 'skeleton',
        note='最多400万像素，输出骨架端点、分支和长度近似。',
    ))
    tools.append(tool(
        'minkowski', '阈值拓扑曲线', '形态', 'minkowski',
        note='扫描12个灰度阈值，观察面积、边界长度和欧拉数的变化。',
    ))
    tools.append(tool(
        'sweep', '分割阈值扫描', '分割', 'sweep', 'POST', [
            number('start', '起始阈值', 60, 0, 255),
            number('stop', '结束阈值', 180, 0, 255),
            number('steps', '扫描步数', 9, 2, 15, 1),
        ], '沿用当前页面分割设置，不修改当前分析结果；最多400万像素。',
    ))
    tools.append(tool(
        'reference-score', '参考掩膜评估', '分割', 'reference-score',
        note='先上传同尺寸二值参考图；演示样本自带参考掩膜。显示IoU、Dice、精确率与召回率。',
    ))
    tools.append(tool(
        'gallery', '颗粒裁片图册', '导出', 'gallery',
        note='按纳入颗粒生成编号图册，最多600个。', output='image',
    ))
    tools.append(tool(
        'heatmap', '测量值空间热图', '空间', 'heatmap', 'POST', [
            select('field', '热图指标', 'diameter', MEASUREMENTS),
            number('radius', '平滑半径 px', 25, 2, 150),
        ], '质心测量值按高斯权重局部平均，仅覆盖有支持的区域。', output='image',
    ))
    tools.append(tool(
        'difference', '原图差异热图', '图像', 'difference',
        note='当前图像与原图必须同尺寸，反映预处理带来的像素差异。', output='image',
    ))
    tools.append(tool(
        'checkerboard', '原图棋盘对照', '图像', 'checkerboard',
        note='与原图交错显示，要求图像尺寸一致。', output='image',
    ))
    tools.append(tool(
        'annotated', '人工测量叠加', '图像', 'annotated',
        note='按当前标定渲染线段、角度、多边形和点标记。', output='image',
    ))
    tools.append(tool(
        'boundary-sensitivity', '分割边界敏感性', '精度', 'boundary-sensitivity', 'POST', [
            number('pixels', '边界变化像素', 1, 1, 5, 1),
        ], '逐颗粒腐蚀与膨胀，比较直径和面积变化；不是置信区间。',
    ))
    tools.append(tool(
        'calibration-uncertainty', '标定误差传播', '精度', 'calibration-uncertainty', 'POST', [
            number('pixel_error', '标尺像素长度标准差', 1, 0, 1000),
            number('physical_error', '实际长度标准差', 0, 0, 1000),
        ], '一阶独立误差传播，仅估计标定影响。',
    ))
    tools.append(tool(
        'sampling-precision', '颗粒抽样精度估计', '精度', 'sampling-precision', 'POST', [
            number('target_relative_error', '目标相对半宽', 0.05, 0.001, 0.5),
        ], '独立同分布和正态近似，仅作参考。',
    ))
    tools.append(tool(
        'block-bootstrap', '空间块重采样', '精度', 'block-bootstrap', 'POST', [
            number('columns', '块列数', 4, 2, 20, 1),
            number('rows', '块行数', 4, 2, 20, 1),
            number('repeats', '重采样次数', 1000, 100, 3000, 1),
        ], '整块重采样保留局部空间相关性。',
    ))
    tools.append(tool(
        'radial-signature', '颗粒径向形状描述', '形态', 'radial-signature', 'POST', [
            number('particle_id', '颗粒编号', 1, 1, 20000, 1),
            number('bins', '角度采样数', 72, 12, 360, 1),
        ], '相对质心的径向边界与傅里叶幅度，不适合表达凹形内部结构。',
    ))
    tools.append(tool(
        'boundary-curvature', '颗粒边界曲率与尖角', '形态', 'boundary-curvature', 'POST', [
            number('particle_id', '颗粒编号', 1, 1, 20000, 1),
            number('sigma_px', '观察尺度 σ px', 2.0, 0.5, 40, 0.5),
            number('min_turn_deg', '最小转角 °', 30, 5, 170, 1),
            checkbox('include_holes', '包含孔洞轮廓', True),
        ], '沿外轮廓和孔洞轮廓的曲率序列，标记凸角与凹口；尺度可调，像素台阶不应当作尖角。',
    ))
    return {
        'tools': tools,
        'measurements': [{'key': key, 'label': label} for key, label in MEASUREMENTS],
        'preprocessing': [
            {'key': 'gaussian', 'label': '高斯平滑'},
            {'key': 'median', 'label': '中值去噪'},
            {'key': 'bilateral', 'label': '保边双边滤波'},
            {'key': 'normalize', 'label': '分位拉伸'},
            {'key': 'gamma', 'label': 'Gamma校正'},
            {'key': 'background', 'label': '缓变背景校正'},
            {'key': 'unsharp', 'label': '反锐化增强'},
            {'key': 'invert', 'label': '颜色反相'},
            {'key': 'channel', 'label': '通道提取'},
            {'key': 'contrast', 'label': '对比度增强'},
            {'key': 'equalize', 'label': '直方图均衡'},
            {'key': 'clahe', 'label': '局部对比增强'},
            {'key': 'rotate', 'label': '旋转'},
            {'key': 'flip', 'label': '翻转'},
            {'key': 'crop', 'label': '裁切'},
            {'key': 'resize', 'label': '缩放'},
        ],
        'limits': {
            'upload_mb': 96,
            'image_megapixels': 16,
            'max_particles': 20000,
            'max_batch_samples': 100,
        },
    }
