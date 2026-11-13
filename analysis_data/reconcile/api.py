"""可调用分析接口。

典型用法::

    from analysis_data.io import load
    from analysis_data.reconcile import reconcile_file, reconcile_dict
    from analysis_data.reconcile.api import reconcile_problem

    result = reconcile_file("data/normal.json")
    result.to_dict()           # 精确分数字符串的完整结果包
    result.to_dict(exact=False)  # 人读浮点版

入口分三层，便于在任意一层接入：

* :func:`reconcile_file`：读冻结 JSON、校验身份、一次跑完整分析；
* :func:`reconcile_dict`：内存中的资料字典（测试/合成场景）；
* :func:`reconcile_problem`：已经物化的 :class:`~materialize.Problem`。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import load
from .engine import ReconciliationResult, reconcile
from .materialize import Problem, materialize

__all__ = [
    "reconcile_file",
    "reconcile_dict",
    "reconcile_problem",
    "materialize_dict",
]


def reconcile_file(path: str | Path) -> ReconciliationResult:
    v, sha = load(str(path))
    problem = materialize(v, manifest_sha256=sha)
    return reconcile(problem)


def reconcile_dict(v: dict[str, Any]) -> ReconciliationResult:
    return reconcile(materialize(v))


def materialize_dict(v: dict[str, Any]) -> Problem:
    return materialize(v)


def reconcile_problem(problem: Problem) -> ReconciliationResult:
    return reconcile(problem)
