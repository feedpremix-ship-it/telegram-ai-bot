"""Bounded math AST interpreter: no eval, exec, imports, attributes or Python code."""
import ast
import io
import math
import os
import multiprocessing
import operator

FUNCS = {n: getattr(math, n) for n in ('sin','cos','tan','sqrt','log','log10','exp','fabs')}
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow}

def expression(source):
    if not isinstance(source, str) or len(source) > 160:
        raise ValueError('Формула слишком длинная.')
    tree = ast.parse(source.replace('^', '**'), mode='eval')
    if len(list(ast.walk(tree))) > 60:
        raise ValueError('Формула слишком сложная.')
    def calc(node, x):
        if isinstance(node, ast.Expression):
            value = calc(node.body, x)
        elif isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = float(node.value)
        elif isinstance(node, ast.Name) and node.id in ('x','pi','e'):
            value = {'x': x, 'pi': math.pi, 'e': math.e}[node.id]
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = calc(node.operand, x) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in OPS:
            a, b = calc(node.left, x), calc(node.right, x)
            if isinstance(node.op, ast.Pow) and abs(b) > 12:
                raise ValueError('Степень должна быть от -12 до 12.')
            value = OPS[type(node.op)](a, b)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCS and len(node.args) == 1 and not node.keywords:
            value = FUNCS[node.func.id](calc(node.args[0], x))
        else:
            raise ValueError('Разрешены только x, числа, pi, e, арифметика и стандартные функции.')
        if not isinstance(value, (float, int)) or not math.isfinite(value) or abs(value) > 1e12:
            raise ValueError('Значение вне безопасного диапазона.')
        return value
    # Validate all syntax even when the first sample has a domain error.
    allowed = (ast.Expression, ast.Constant, ast.Name, ast.Load, ast.UnaryOp, ast.UAdd, ast.USub,
               ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Call)
    if any(not isinstance(n, allowed) for n in ast.walk(tree)):
        raise ValueError('Недопустимая формула.')
    return lambda x: calc(tree, x)

def render(spec):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    title = str(spec.get('title', 'График'))[:100]
    fig = Figure(figsize=(7, 4), dpi=110)
    ax = fig.subplots()
    kind = spec.get('kind', 'function')
    if kind == 'function':
        f = expression(spec.get('expression', ''))
        lo, hi = float(spec.get('min', -10)), float(spec.get('max', 10))
        if not math.isfinite(lo+hi) or not -10000 <= lo < hi <= 10000:
            raise ValueError('Диапазон: -10000 ≤ min < max ≤ 10000.')
        xs = [lo + (hi-lo)*i/399 for i in range(400)]
        ys = []
        for x in xs:
            try:
                ys.append(f(x))
            except (ValueError, OverflowError, ZeroDivisionError):
                ys.append(float('nan'))
        if not any(math.isfinite(y) for y in ys):
            raise ValueError('Нет допустимых значений функции.')
        ax.plot(xs, ys)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=.25)
    elif kind in ('bar', 'pie'):
        labels, values = spec.get('labels'), spec.get('values')
        if not isinstance(labels, list) or not isinstance(values, list) or not 1 <= len(values) <= 20 or len(labels) != len(values):
            raise ValueError('Нужно 1–20 подписей и столько же значений.')
        values = [float(x) for x in values]
        if not all(math.isfinite(x) and abs(x) <= 1e12 for x in values):
            raise ValueError('Некорректные значения.')
        labels = [str(x)[:30] for x in labels]
        if kind == 'pie':
            if min(values) < 0 or sum(values) <= 0:
                raise ValueError('Для долей нужны неотрицательные значения с положительной суммой.')
            ax.pie(values, labels=labels, autopct='%1.1f%%')
        else:
            ax.bar(labels, values)
            ax.tick_params(axis='x', labelrotation=30)
    else:
        raise ValueError('Поддерживаются function, bar, pie.')
    ax.set_title(title, parse_math=False)
    fig.tight_layout()
    data = io.BytesIO()
    fig.savefig(data, format='png')
    return data.getvalue()

def _child(spec, connection):
    try:
        if os.name == "posix":
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (8, 8))
        connection.send((True, render(spec)))
    except Exception:
        connection.send((False, 'Не удалось построить график. Проверьте формулу и диапазон.'))
    finally:
        connection.close()

def isolated_render(spec):
    ctx = multiprocessing.get_context('spawn')
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_child, args=(spec, child), daemon=True)
    process.start()
    child.close()
    try:
        if not parent.poll(12):
            raise ValueError('Построение графика превысило время ожидания.')
        try:
            ok, value = parent.recv()
        except EOFError:
            raise ValueError('Процесс построения графика остановлен.') from None
        if not ok:
            raise ValueError(value)
        return value
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
        parent.close()
