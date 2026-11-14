import base64
import html
import io
from .measurements import included_rows, MEASUREMENT_FIELDS
from .statistics import statistics_report
from .rendering import histogram_chart, png_bytes
from .annotations import annotation_summary


def escape(value):
    return html.escape(str(value), quote=True)


def number(value, digits=4):
    if value is None:
        return '—'
    if isinstance(value, (int, float)):
        return f'{value:.{digits}g}'
    return escape(value)


def image_uri(data):
    return 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')


def html_table(headers, rows):
    head = ''.join('<th>' + escape(h) + '</th>' for h in headers)
    body = []
    for row in rows:
        cells = ''.join('<td>' + escape(cell) + '</td>' for cell in row)
        body.append('<tr>' + cells + '</tr>')
    return '<table><thead><tr>' + head + '</tr></thead><tbody>' + ''.join(body) + '</tbody></table>'


def metadata_section(meta, result):
    pairs = [
        ('样本', meta['name']),
        ('图像尺寸', f'{meta["width"]} × {meta["height"]} px'),
        ('分析时间', result['created_at']),
        ('分析编号', result['id']),
        ('图像版本', result['image_revision']),
        ('标定', f'1 px = {result["scale"]:g} {result["unit"]}'),
        ('标签', ' / '.join(meta.get('tags', []))),
        ('样本说明', meta.get('notes', '')),
    ]
    return html_table(['项目', '内容'], pairs)


def measurement_section(result, statistics):
    rows = []
    for name, definition in MEASUREMENT_FIELDS.items():
        description = statistics['descriptive'][name]
        power = definition['power']
        unit = result['unit'] + ('²' if power == 2 else '') if power else ''
        rows.append([
            definition['label'],
            unit,
            number(description['mean']),
            number(description['std']),
            number(description['median']),
            number(description['minimum']),
            number(description['maximum']),
        ])
    return html_table(['指标', '单位', '均值', '标准差', '中位数', '最小值', '最大值'], rows)


def parameter_section(result):
    labels = {
        'method': '阈值方法',
        'channel': '分析通道',
        'polarity': '颗粒极性',
        'threshold': '手动阈值',
        'sigma': '平滑σ',
        'min_area': '最小面积px²',
        'max_area': '最大面积px²',
        'opening': '开运算半径',
        'closing': '闭运算半径',
        'fill_holes': '填充孔洞',
        'exclude_edge': '排除边缘',
        'split_touching': '粘连分离',
        'seed_distance': '种子间距',
    }
    pairs = [(labels.get(k, k), v) for k, v in result['parameters'].items()]
    pairs.append(('实际阈值', result['details'].get('effective_threshold')))
    pairs.append(('分析区域', str(result['roi'])))
    pairs.append(('结果来源', result['source']))
    return html_table(['参数', '取值'], pairs)


def psd_section(statistics):
    rows = []
    for key, label in [('number_distribution', '数量加权'), ('area_distribution', '面积加权'), ('volume_distribution', '等效球体积加权')]:
        distribution = statistics[key]
        rows.append([
            label,
            number(distribution['d10']),
            number(distribution['d50']),
            number(distribution['d90']),
            number(distribution['span']),
        ])
    return html_table(['权重', 'D10', 'D50', 'D90', 'Span'], rows)


def annotation_section(annotations):
    rows = []
    for item in annotations:
        measured = ', '.join(f'{key}={number(value)}' for key, value in item['measurement'].items())
        rows.append([item['type'], item['label'], measured, item['unit']])
    if not rows:
        return '<p>暂无人工测量标注。</p>'
    return html_table(['类型', '标签', '测量', '单位'], rows)


def build_report(store, service, sample_id):
    with store.lock:
        meta = store.meta(sample_id)
        result = service.require_result(sample_id)
        statistics = statistics_report(result)
        annotations = store.annotations(sample_id)
        overlay = store.sample_dir(sample_id) / 'analyses' / result['id'] / 'overlay.png'
        overlay_uri = image_uri(overlay.read_bytes())
        chart_uri = image_uri(png_bytes(histogram_chart(statistics['number_distribution'])))
        cards = []
        for label, value in [
            ('纳入颗粒', result['count']),
            ('排除颗粒', result['excluded_count']),
            ('覆盖率', number(result['coverage']) + '%'),
            ('平均直径', number(result['mean_diameter']) + ' ' + result['unit']),
        ]:
            cards.append('<div class="card"><small>' + escape(label) + '</small><strong>' + escape(value) + '</strong></div>')
        sections = [
            '<section><h2>样本与标定</h2>' + metadata_section(meta, result) + '</section>',
            '<div class="cards">' + ''.join(cards) + '</div>',
            '<section><h2>检测叠加</h2><img src="' + overlay_uri + '"></section>',
            '<section><h2>粒径分布</h2><img src="' + chart_uri + '">' + psd_section(statistics) + '</section>',
            '<section><h2>形态与灰度统计</h2>' + measurement_section(result, statistics) + '</section>',
            '<section><h2>分析参数</h2>' + parameter_section(result) + '</section>',
            '<section><h2>人工测量</h2>' + annotation_section(annotations) + '</section>',
        ]
        notices = [
            '所有坐标以图像左上角为原点，X向右、Y向下。',
            '二维图像的等效圆直径不等于三维真实粒径；体积权重采用等效球假设。',
            '周长为Crofton近似；凸包与Feret径按像素所占单位方格的实际平面区域计算。',
            '自动分离依赖距离峰值，须人工检查掩膜，不能保证每个相接对象准确分离。',
            '统计以纳入颗粒为单位，不替代独立实验重复。',
        ]
        notice_html = '<ul>' + ''.join('<li>' + escape(item) + '</li>' for item in notices) + '</ul>'
        title = escape(meta['name'])
        style = 'body{font:14px sans-serif;max-width:1100px;margin:30px auto;color:#243442;padding:20px}h1{color:#167b66}h2{margin-top:32px}table{border-collapse:collapse;width:100%;font-size:12px}th,td{border:1px solid #ccd5db;padding:8px;text-align:left;overflow-wrap:anywhere}th{background:#eff5f3}img{max-width:100%;max-height:650px;object-fit:contain}.cards{display:flex;gap:18px;margin-top:24px}.card{background:#edf6f3;padding:20px;flex:1}.card strong{display:block;font-size:24px;margin-top:10px}small,footer{color:#5e7280}.print{float:right;padding:10px}@media print{.print{display:none}section{break-inside:avoid}body{margin:0}table{font-size:10px}}'
        report = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>粒境报告 · ' + title + '</title><style>' + style + '</style><body>'
        report += '<button class="print" onclick="window.print()">打印 / 另存PDF</button><h1>粒境 · 颗粒图像分析报告</h1><p>' + title + '</p>'
        report += ''.join(sections)
        report += '<section><h2>测量说明</h2>' + notice_html + '</section><footer>ParticleScope · 本地生成 · 分析编号 ' + escape(result['id']) + '</footer></body></html>'
        return report


def project_report(store, service, sample_ids):
    entries = []
    unit_groups = {}
    for key in sample_ids:
        meta = store.meta(key)
        result = store.result(key)
        if result is None or result.get('stale'):
            entries.append([meta['name'], '未分析或结果过期', '—', '—', '—'])
            continue
        entries.append([
            meta['name'],
            str(result['count']),
            number(result['mean_diameter']) + ' ' + result['unit'],
            number(result['coverage']) + '%',
            result['created_at'],
        ])
        unit_groups.setdefault(result['unit'], []).append(result)
    summaries = []
    for unit, results in unit_groups.items():
        count = sum(r['count'] for r in results)
        weighted = sum(r['mean_diameter'] * r['count'] for r in results) / count if count else 0
        summaries.append([unit, len(results), count, number(weighted)])
    body = '<h1>粒境 · 多样本汇总报告</h1>'
    body += html_table(['样本', '数量', '平均直径', '覆盖率', '分析时间'], entries)
    body += '<h2>按相同单位汇总</h2>'
    body += html_table(['单位', '已分析样本', '总颗粒', '颗粒加权平均直径'], summaries)
    body += '<p>不同单位分别汇总；像素样本的放大倍率可能不同，需确认后再比较。</p>'
    return '<!doctype html><meta charset="utf-8"><title>多样本汇总</title><style>body{font:14px sans-serif;padding:30px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #ccc}</style>' + body
