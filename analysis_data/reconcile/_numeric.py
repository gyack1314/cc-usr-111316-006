"""精确数值工具。

资料把有理量写成字符串（如 ``"1/10000000000"``），缺测写成 JSON 的
``null``。本模块负责把它们翻译成 :class:`~fractions.Fraction`，并提供
矩阵/向量的小工具。``None``（缺测）在任何地方都不等于 0，由调用方显式
决定删行而不是补零。
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Iterable, Sequence

__all__ = [
    "parse_number",
    "parse_optional",
    "frac_matrix",
    "frac_vector",
    "zeros",
    "identity",
    "matmul",
    "transpose",
    "dot",
    "vec_add",
    "vec_scale",
    "to_float",
]


def parse_number(x: Any, *, what: str = "数值") -> Fraction:
    """把 JSON 标量解析成 :class:`Fraction`。

    接受字符串（含 ``"p/q"``、十进制、科学计数法）与 int/float；
    显式拒绝 bool 与 None，避免 ``True`` 被当成 1、缺测被当成 0。
    """
    if isinstance(x, bool) or x is None:
        raise ValueError(f"{what}不能是布尔或null（缺测必须在结构层显式删除）")
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, float):
        return Fraction(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            raise ValueError(f"{what}是空字符串")
        try:
            return Fraction(s)
        except (ZeroDivisionError, ValueError, OverflowError) as exc:
            raise ValueError(f"{what}不是合法有理数: {x!r}") from exc
    raise ValueError(f"{what}类型非法: {type(x).__name__}")


def parse_optional(x: Any, *, what: str = "数值") -> Fraction | None:
    """``null`` 透传成 ``None``；其余按 :func:`parse_number` 解析。"""
    if x is None:
        return None
    return parse_number(x, what=what)


def zeros(n: int, m: int | None = None) -> list[list[Fraction]]:
    if m is None:
        return [Fraction(0) for _ in range(n)]
    return [[Fraction(0) for _ in range(m)] for _ in range(n)]


def identity(n: int) -> list[list[Fraction]]:
    a = zeros(n, n)
    for i in range(n):
        a[i][i] = Fraction(1)
    return a


def frac_vector(rows: Iterable[Any], *, what: str) -> list[Fraction]:
    return [parse_number(x, what=what) for x in rows]


def frac_matrix(rows: Sequence[Sequence[Any]], *, what: str) -> list[list[Fraction]]:
    if not rows:
        return []
    width = len(rows[0])
    out: list[list[Fraction]] = []
    for i, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(f"{what}第{i}行宽度不一致")
        out.append([parse_number(x, what=f"{what}[{i}]") for x in row])
    return out


def transpose(a: list[list[Fraction]]) -> list[list[Fraction]]:
    if not a:
        return []
    return [[a[i][j] for i in range(len(a))] for j in range(len(a[0]))]


def matmul(a: list[list[Fraction]], b: list[list[Fraction]]) -> list[list[Fraction]]:
    if not a or not b:
        return []
    k = len(a[0])
    if len(b) != k:
        raise ValueError("矩阵乘法维度不匹配")
    bt = transpose(b)
    return [[dot(row, col) for col in bt] for row in a]


def dot(u: Sequence[Fraction], v: Sequence[Fraction]) -> Fraction:
    if len(u) != len(v):
        raise ValueError("向量内积长度不一致")
    return sum((x * y for x, y in zip(u, v)), Fraction(0))


def vec_add(u: Sequence[Fraction], v: Sequence[Fraction]) -> list[Fraction]:
    return [x + y for x, y in zip(u, v)]


def vec_scale(s: Fraction, v: Sequence[Fraction]) -> list[Fraction]:
    return [s * x for x in v]


def to_float(v: Any) -> Any:
    """递归地把 Fraction 转成 float，用于人读报告与浮点交叉核对。"""
    if isinstance(v, Fraction):
        return float(v)
    if isinstance(v, dict):
        return {k: to_float(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [to_float(x) for x in v]
    return v
