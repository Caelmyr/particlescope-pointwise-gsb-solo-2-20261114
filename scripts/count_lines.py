import ast
import hashlib
import io
import json
from pathlib import Path
import tokenize

ROOT = Path(__file__).resolve().parents[1]


def measure(path):
    raw = path.read_bytes()
    source = raw.decode('utf-8-sig')
    lines = source.splitlines()
    excluded = set()
    for node in ast.walk(ast.parse(source, filename=str(path))):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            excluded.update(range(node.lineno, node.end_lineno + 1))
    ignored = {
        tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
        tokenize.DEDENT, tokenize.ENDMARKER, tokenize.ENCODING, tokenize.STRING,
    }
    for name in ['FSTRING_START', 'FSTRING_MIDDLE', 'FSTRING_END']:
        if hasattr(tokenize, name):
            ignored.add(getattr(tokenize, name))
    meaningful = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in ignored or not token.string.strip():
            continue
        for number in range(token.start[0], token.end[0] + 1):
            if number not in excluded and lines[number - 1].strip():
                meaningful.add(number)
    return {
        'file': path.relative_to(ROOT).as_posix(),
        'effective_lines': len(meaningful),
        'sha256': hashlib.sha256(raw).hexdigest(),
    }


def main():
    paths = [ROOT / 'app.py', *sorted((ROOT / 'particlescope').glob('*.py'))]
    files = [measure(path) for path in paths]
    total = sum(item['effective_lines'] for item in files)
    report = {
        'method': '只统计app.py和particlescope/*.py自有后端。tokenize排除空行、注释、缩进、纯字符串内容；AST排除独立字符串表达式及文档字符串。前端、测试、脚本、依赖、虚拟环境不计。与根目录统计后端有效行数.py采用同一measure算法。',
        'files': files,
        'total': total,
        'passes_5000': total > 5000,
    }
    directory = ROOT / 'docs'
    directory.mkdir(exist_ok=True)
    (directory / 'line-count.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    text = '\n'.join(f'{item["file"]}: {item["effective_lines"]}' for item in files)
    text += f'\n有效后端合计：{total}\n超过5000：{total > 5000}\n'
    (directory / 'line-count.txt').write_text(text, encoding='utf8')
    print(text)


if __name__ == '__main__':
    main()
