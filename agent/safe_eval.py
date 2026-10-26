"""Small AST evaluator for model-generated pandas expressions.

This is intentionally not a general Python sandbox. Only expressions rooted at
the provided ``df`` object and a fixed set of DataFrame/Series methods are
accepted. Attribute names starting with ``_`` are rejected, so dunder access
(``__class__``, ``__globals__``) never reaches the evaluator.
"""

from __future__ import annotations

import ast
from typing import Any

ALLOWED_METHODS = frozenset({
    "head", "tail", "nlargest", "nsmallest", "sort_values", "sort_index",
    "groupby", "agg", "sum", "mean", "median", "min", "max", "count",
    "value_counts", "reset_index", "set_index", "rename", "drop", "dropna",
    "fillna", "astype", "query", "loc", "iloc", "nunique", "unique",
    "to_datetime", "corr", "corrwith", "cov",
})

_ALLOWED_NODES = (
    ast.Expression, ast.Module, ast.Expr, ast.Assign, ast.Name, ast.Constant, ast.Subscript,
    ast.Attribute, ast.Call, ast.keyword, ast.List, ast.Tuple, ast.Dict,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.And, ast.Or, ast.Not, ast.UnaryOp, ast.USub, ast.UAdd, ast.Load,
    ast.Store, ast.Slice, ast.Index if hasattr(ast, "Index") else ast.Slice,
)


class SafeEvalError(ValueError):
    pass


def _reject(message: str) -> None:
    raise SafeEvalError(message)


def _node_ok(node: ast.AST, root_name: str) -> None:
    if not isinstance(node, _ALLOWED_NODES):
        _reject(f"node tidak diizinkan: {type(node).__name__}")
    if isinstance(node, ast.Name):
        if node.id not in {root_name, "result"}:
            raise NameError(node.id)
    if isinstance(node, ast.Attribute):
        if node.attr.startswith("_"):
            _reject(f"atribut tidak diizinkan: {node.attr}")
        if node.attr not in ALLOWED_METHODS and node.attr not in {"loc", "iloc"}:
            _reject(f"method/atribut tidak diizinkan: {node.attr}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Attribute):
            _reject("panggilan harus berupa method DataFrame/Series")
        if any(isinstance(arg, ast.Lambda) for arg in node.args):
            _reject("lambda tidak diizinkan")
    for child in ast.iter_child_nodes(node):
        _node_ok(child, root_name)


def evaluate(expression: str, df: Any, *, root_name: str = "df") -> Any:
    expression = str(expression or "").strip()
    if not expression:
        _reject("query kosong")
    import re as _re

    is_assignment = bool(_re.search(r"(?m)^\s*result\s*=", expression))
    try:
        tree = ast.parse(expression, mode="exec" if is_assignment else "eval")
    except SyntaxError as exc:
        raise SafeEvalError(f"syntax tidak valid: {exc}") from exc
    body = tree.body[-1] if isinstance(tree, ast.Module) else tree
    if isinstance(body, ast.Assign):
        if len(body.targets) != 1 or not isinstance(body.targets[0], ast.Name) or body.targets[0].id != "result":
            _reject("hanya assignment ke 'result' yang diizinkan")
        value_node = body.value
    elif isinstance(tree, ast.Expression):
        value_node = tree.body
    else:
        _reject("format query tidak didukung")

    for node in ast.walk(tree):
        _node_ok(node, root_name)
    return _eval(value_node, {root_name: df})


def _eval(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise NameError(node.id)
        return env[node.id]
    if isinstance(node, ast.Subscript):
        base = _eval(node.value, env)
        key = _eval(node.slice, env) if not isinstance(node.slice, ast.Slice) else slice(
            _eval(node.slice.lower, env) if node.slice.lower else None,
            _eval(node.slice.upper, env) if node.slice.upper else None,
            _eval(node.slice.step, env) if node.slice.step else None,
        )
        return base[key]
    if isinstance(node, ast.Attribute):
        base = _eval(node.value, env)
        if node.attr.startswith("_"):
            raise SafeEvalError("atribut privat tidak diizinkan")
        return getattr(base, node.attr)
    if isinstance(node, ast.Call):
        func = _eval(node.func, env)
        args = [_eval(arg, env) for arg in node.args]
        kwargs = {kw.arg: _eval(kw.value, env) for kw in node.keywords if kw.arg}
        return func(*args, **kwargs)
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(item, env) for item in node.elts] if isinstance(node, ast.List) else tuple(_eval(item, env) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {_eval(k, env): _eval(v, env) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.BinOp):
        left, right = _eval(node.left, env), _eval(node.right, env)
        return {ast.Add: lambda: left + right, ast.Sub: lambda: left - right, ast.Mult: lambda: left * right, ast.Div: lambda: left / right, ast.Mod: lambda: left % right, ast.Pow: lambda: left ** right}[type(node.op)]()
    if isinstance(node, ast.Compare):
        left = _eval(node.left, env)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval(comparator, env)
            if not {ast.Eq: lambda: left == right, ast.NotEq: lambda: left != right, ast.Lt: lambda: left < right, ast.LtE: lambda: left <= right, ast.Gt: lambda: left > right, ast.GtE: lambda: left >= right}[type(op)]():
                return False
            left = right
        return True
    if isinstance(node, ast.BoolOp):
        values = [_eval(value, env) for value in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.UnaryOp):
        value = _eval(node.operand, env)
        return not value if isinstance(node.op, ast.Not) else (-value if isinstance(node.op, ast.USub) else +value)
    if isinstance(node, ast.Slice):
        return slice(_eval(node.lower, env) if node.lower else None, _eval(node.upper, env) if node.upper else None, _eval(node.step, env) if node.step else None)
    raise SafeEvalError(f"node tidak didukung: {type(node).__name__}")
