"""把冻结资料物化成可求解的显式问题。

登记口径（与 ``docs/data-contract.md`` 一致）：

* ``bottom_ids`` 唯一确定底层向量 ``b`` 的坐标顺序；
* 每个节点显式登记它在底层上的系数，允许重复/线性依赖的汇总行
  （重复行不在这里删除——它与完全相关的误差共同构成 ``W`` 的零方向，
  是必须保留的信息，而不是冗余垃圾）；
* ``predictions`` 中的 ``null`` 表示该节点没有预测：整行从观测向量和
  协方差中显式删除，**绝不补 0**；
* 下界一般非负；只有在 ``net_transfer`` 名单里明确标成净转出的底层项
  才允许负下界；
* 协方差必须对称且半正定，明显负特征值直接判非法，不容忍。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from . import _linalg
from ._numeric import frac_matrix, frac_vector, parse_number, parse_optional, zeros

__all__ = ["Problem", "materialize", "MaterialError"]


class MaterialError(ValueError):
    """资料不满足分析契约时抛出。"""


@dataclass
class Query:
    id: str
    c: list[Fraction]


@dataclass
class Problem:
    bottom_ids: list[str]
    # 全部已登记节点（含缺测），顺序与资料一致
    all_node_ids: list[str]
    observed_node_ids: list[str]
    missing_node_ids: list[str]
    # 全量登记映射（all nodes × m），用于逐行回代核对
    s_full: list[list[Fraction]]
    # 观测子问题：S、y、W 只含有预测的节点
    s: list[list[Fraction]]
    y: list[Fraction]
    w: list[list[Fraction]]
    lower: list[Fraction]
    upper: list[Fraction | None]  # None = +inf
    b0: list[Fraction]
    rho: Fraction
    rank_tolerance: Fraction
    queries: list[Query]
    net_transfer: frozenset[str]
    node_registry: dict[str, dict[str, Fraction]]
    ldl: dict = field(repr=False, default=None)
    manifest_sha256: str | None = None

    @property
    def m(self) -> int:
        return len(self.bottom_ids)

    @property
    def n_obs(self) -> int:
        return len(self.observed_node_ids)


def _coeff_map(node: dict, bottom_index: dict[str, int], m: int) -> list[Fraction]:
    row = zeros(m)
    coeffs = node.get("coefficients", {})
    if not isinstance(coeffs, dict) or not coeffs:
        raise MaterialError(f"节点{node.get('id')!r}缺少显式系数登记")
    for key, val in coeffs.items():
        if key not in bottom_index:
            raise MaterialError(f"节点{node.get('id')!r}引用了未登记底层{key!r}")
        row[bottom_index[key]] = parse_number(val, what=f"节点{node.get('id')!r}系数")
    return row


def materialize(v: dict, *, manifest_sha256: str | None = None) -> Problem:
    if not isinstance(v, dict) or v.get("kind") != "reconciliation":
        raise MaterialError("kind 必须是 reconciliation")

    bottom_ids = v.get("bottom_ids")
    if not isinstance(bottom_ids, list) or not bottom_ids:
        raise MaterialError("bottom_ids 必须是非空列表")
    if len(set(bottom_ids)) != len(bottom_ids):
        raise MaterialError("底层身份不唯一")
    m = len(bottom_ids)
    bottom_index = {bid: i for i, bid in enumerate(bottom_ids)}

    nodes = v.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise MaterialError("nodes 必须是非空列表")
    all_ids: list[str] = []
    registry: dict[str, dict[str, Fraction]] = {}
    s_full: list[list[Fraction]] = []
    for node in nodes:
        nid = node.get("id")
        if not isinstance(nid, str) or not nid or nid in registry:
            raise MaterialError("节点身份缺失或重复")
        all_ids.append(nid)
        row = _coeff_map(node, bottom_index, m)
        registry[nid] = {bid: row[i] for i, bid in enumerate(bottom_ids) if row[i] != 0}
        s_full.append(row)

    predictions = v.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != len(nodes):
        raise MaterialError("predictions 长度必须与 nodes 一致；缺测写 null，不要删位")

    # 显式区分“有预测”和“缺测”，缺测节点整行删除而非补零
    observed_idx: list[int] = []
    missing_ids: list[str] = []
    y: list[Fraction] = []
    for i, raw in enumerate(predictions):
        val = parse_optional(raw, what=f"预测{nodes[i].get('id')!r}")
        if val is None:
            missing_ids.append(all_ids[i])
        else:
            observed_idx.append(i)
            y.append(val)
    if not observed_idx:
        raise MaterialError("至少要有一个有预测的节点")
    observed_ids = [all_ids[i] for i in observed_idx]
    s = [s_full[i][:] for i in observed_idx]

    cov_raw = v.get("covariance")
    if not isinstance(cov_raw, list) or len(cov_raw) != len(nodes) or any(
        not isinstance(r, list) or len(r) != len(nodes) for r in cov_raw
    ):
        raise MaterialError("协方差必须是与 nodes 等阶的方阵（缺测节点的行列随后删除）")
    w_full = frac_matrix(cov_raw, what="协方差")
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if w_full[i][j] != w_full[j][i]:
                raise MaterialError("协方差不对称")
    # 删除缺测节点对应的行列——null 不参与代数
    w = [[w_full[i][j] for j in observed_idx] for i in observed_idx]

    tol = parse_number(v.get("rank_tolerance", "0"), what="rank_tolerance")
    if tol < 0:
        raise MaterialError("rank_tolerance 不能为负")
    ldl = _linalg.ldl_psd(w, tol)
    if ldl["negative"]:
        raise MaterialError(
            f"协方差存在明显负主元 {ldl['negative']}，不是半正定矩阵；"
            "不允许通过加岭项把它修成正定"
        )

    lower_raw = v.get("lower_bounds")
    upper_raw = v.get("upper_bounds")
    if not isinstance(lower_raw, list) or not isinstance(upper_raw, list) or len(
        lower_raw
    ) != m or len(upper_raw) != m:
        raise MaterialError("lower_bounds/upper_bounds 必须按底层逐项给出")
    net_transfer = frozenset(v.get("net_transfer", ()))
    extra = net_transfer - set(bottom_ids)
    if extra:
        raise MaterialError(f"net_transfer 含未登记底层: {sorted(extra)}")
    lower: list[Fraction] = []
    upper: list[Fraction | None] = []
    for i, bid in enumerate(bottom_ids):
        lo = parse_optional(lower_raw[i], what=f"下界{bid!r}")
        hi = parse_optional(upper_raw[i], what=f"上界{bid!r}")
        if lo is None:
            raise MaterialError(f"底层{bid!r}下界不能为 null（资金需求至少给 -inf 显式登记）")
        if lo < 0 and bid not in net_transfer:
            raise MaterialError(
                f"底层{bid!r}下界为负但未在 net_transfer 中明确标成净转出"
            )
        if hi is not None and hi < lo:
            raise MaterialError(f"底层{bid!r}上界小于下界")
        lower.append(lo)
        upper.append(hi)

    b0 = frac_vector(v.get("reference", []), what="reference")
    if len(b0) != m:
        raise MaterialError("reference(b0) 必须按底层逐项给出")
    rho = parse_number(v.get("rho", "0"), what="rho")
    if rho < 0:
        raise MaterialError("rho 是加权残差平方上限，不能为负")

    queries: list[Query] = []
    seen_q: set[str] = set()
    for q in v.get("queries", []):
        qid = q.get("id")
        if not isinstance(qid, str) or not qid or qid in seen_q:
            raise MaterialError("查询 id 缺失或重复")
        seen_q.add(qid)
        c = _coeff_map(q, bottom_index, m)
        queries.append(Query(qid, c))

    return Problem(
        bottom_ids=bottom_ids,
        all_node_ids=all_ids,
        observed_node_ids=observed_ids,
        missing_node_ids=missing_ids,
        s_full=s_full,
        s=s,
        y=y,
        w=w,
        lower=lower,
        upper=upper,
        b0=b0,
        rho=rho,
        rank_tolerance=tol,
        queries=queries,
        net_transfer=net_transfer,
        node_registry=registry,
        ldl=ldl,
        manifest_sha256=manifest_sha256,
    )
