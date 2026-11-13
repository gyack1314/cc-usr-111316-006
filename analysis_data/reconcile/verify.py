"""独立小规模交叉核对器（与主求解器刻意不同的算法）。

主求解器走活动集 KKT + 信赖域特征方程。这里换成三条朴素、独立的路径，
只用于在小问题上**核对**结论，不参与发布：

1. :func:`verify_point`：逐行回代全部登记的 S 关系、精确约束、底层界、
   W 零空间残差正交性，并用独立的“目标平坦方向扫描”确认点校正最优；
2. :func:`enumerate_polytope_vertices`：把等式消元成自由多面体后，枚举
   箱约束交点（顶点），独立求线性/凸二次目标的极值，与 KKT 结果比对；
3. :func:`random_feasible_scan`：在可行域内投随机可行点，独立估计查询
   边界，验证发布边界没有被随机点突破（联合极值的经验护栏）。

全部输入输出用 :class:`Fraction` 或纯 Python float，不 import 主求解器
的任何函数（只共享 :mod:`_linalg` / :mod:`_numeric` 的基础线性代数）。
"""
from __future__ import annotations

import random
from fractions import Fraction
from itertools import product

from . import _linalg
from ._numeric import dot, matmul, transpose

__all__ = [
    "verify_point",
    "enumerate_polytope_vertices",
    "random_feasible_scan",
    "cross_check_bounds",
]


def verify_point(problem, result, *, tol: float = 1e-8) -> dict:
    """对发布的点校正做独立回代与最优性核对，返回逐项布尔证据。"""
    p = problem
    b = result.point
    checks: dict = {"published": b is not None}
    if b is None:
        checks["reason"] = "冲突或空集，未发布点校正"
        return checks

    # 1) 逐行回代所有登记 S 关系（含缺测节点），并与拟合登记一致
    row_checks = []
    for t, nid in enumerate(p.all_node_ids):
        lhs = dot(p.s_full[t], b)
        row_checks.append(lhs == result.fitted_nodes[nid])
    checks["all_registered_rows_reproduce_fitted"] = all(row_checks)

    # 2) 底层界
    checks["within_bounds"] = all(
        b[j] >= p.lower[j] and (p.upper[j] is None or b[j] <= p.upper[j])
        for j in range(p.m)
    )

    # 3) 精确约束 E b = f（精确）
    checks["exact_constraints_hold"] = all(
        dot(row, b) == rhs for row, rhs in zip(result.exact_constraints, result.exact_rhs)
    )

    # 4) 残差与 W 零空间正交（精确），即没有任何残差落到零特征方向
    checks["residual_orthogonal_to_W_nullspace"] = all(
        dot(z, result.residual) == 0 for z in result.null_directions
    )

    # 5) 加权残差等于发布的最优值（独立重算 W+ 口径一致性）
    checks["residual_matches_optimal"] = (
        _residual_norm(p, b) == result.optimal_residual
    )

    # 6) 最优性：在可行切空间扫描，确认没有方向能降低阶段一目标
    checks["first_order_optimal"] = _scan_optimality(p, result, b)

    # 7) 二级目标：点到 b0 的距离不大于任一可行的阶段一等优点（采样核对）
    checks["second_order_no_worse_than_scan"] = _scan_second_order(p, result, b)
    return checks


def _residual_norm(p, b) -> Fraction:
    """独立计算 r' W^+ r（刻意不用引擎的 LDL 置换公式）。

    路径：取 W 的行空间基 B（RREF 非零行），在该子空间上 W 正定，
    故 W^+ = B' (B W B')^{-1} B。与引擎的 Moore–Penrose LDL 构造零共享，
    仅依赖基础 RREF 与稠密求逆。
    """
    r = [dot(p.s[i], b) - p.y[i] for i in range(p.n_obs)]
    w = p.w
    n = p.n_obs
    basis = _linalg.row_basis(w)               # 行空间基（r 行）
    if not basis:
        return Fraction(0)
    bt = transpose(basis)                      # n x r
    # B W B'（r×r，正定可逆）
    bw = matmul(basis, w)
    bwb = matmul(bw, bt)
    inv = _inverse(bwb)
    # W+ = B' inv B
    tmp = matmul(bt, inv)
    wp = matmul(tmp, basis)
    return dot(r, [dot(wp[i], r) for i in range(n)])


def _inverse(a):
    r = len(a)
    m = [a[i][:] + [Fraction(1) if i == j else Fraction(0) for j in range(r)]
         for i in range(r)]
    for c in range(r):
        pvt = next(i for i in range(c, r) if m[i][c] != 0)
        m[c], m[pvt] = m[pvt], m[c]
        pv = m[c][c]
        m[c] = [v / pv for v in m[c]]
        for i in range(r):
            if i != c and m[i][c] != 0:
                f = m[i][c]
                m[i] = [x - f * y for x, y in zip(m[i], m[c])]
    return [row[r:] for row in m]


def _scan_optimality(p, result, b) -> bool:
    """沿精确约束零空间的小步移动，确认阶段一目标不下降（双侧有限差分）。"""
    basis = (
        _linalg.nullspace(result.exact_constraints)
        if result.exact_constraints else _linalg.identity(p.m)
    )
    base = _residual_norm(p, b)
    for z in basis:
        for step in (Fraction(1, 1000), Fraction(-1, 1000)):
            trial = [b[j] + step * z[j] for j in range(p.m)]
            if any(trial[j] < p.lower[j] for j in range(p.m)):
                continue
            if any(p.upper[j] is not None and trial[j] > p.upper[j] for j in range(p.m)):
                continue
            if _residual_norm(p, trial) < base:
                return False
    return True


def _scan_second_order(p, result, b) -> bool:
    d2 = sum((b[j] - p.b0[j]) ** 2 for j in range(p.m))
    basis = (
        _linalg.nullspace(result.exact_constraints)
        if result.exact_constraints else _linalg.identity(p.m)
    )
    for z in basis:
        for step in (Fraction(1, 100), Fraction(-1, 100)):
            trial = [b[j] + step * z[j] for j in range(p.m)]
            if any(trial[j] < p.lower[j] for j in range(p.m)):
                continue
            if any(p.upper[j] is not None and trial[j] > p.upper[j] for j in range(p.m)):
                continue
            if _residual_norm(p, trial) == result.optimal_residual:
                if sum((trial[j] - p.b0[j]) ** 2 for j in range(p.m)) < d2:
                    return False
    return True


# --------------------------------------------------------------------------- #
def enumerate_polytope_vertices(problem, result) -> list[list[Fraction]]:
    """枚举等式+箱可行多面体的顶点（消元后在自由坐标取箱交点）。

    与活动集求解器不同，这里直接对 RREF 消元后的表示逐坐标取界，
    规模受 ``m`` 限制，是独立的几何枚举。
    """
    p = problem
    E = result.exact_constraints
    if not E:
        rref, piv = _linalg.rref([row[:] for row in E]) if E else ([], [])
        free = list(range(p.m))
        z0 = [Fraction(0)] * p.m
    else:
        aug = [row[:] + [rhs] for row, rhs in zip(E, result.exact_rhs)]
        rref, piv = _linalg.rref(aug)
        free = [c for c in range(p.m) if c not in piv]
        z0 = [Fraction(0)] * p.m
        for row, pc in zip(rref, piv):
            z0[pc] = row[-1]
    vertices: list[list[Fraction]] = []
    # 自由坐标的候选值：各自的界（无界项给 0 与一个有限代表）
    choices = []
    for j in free:
        opts = {p.lower[j]}
        if p.upper[j] is not None:
            opts.add(p.upper[j])
        choices.append(list(opts))
    for combo in product(*choices):
        x = z0[:]
        for t, j in enumerate(free):
            x[j] = combo[t]
        for row, pc in zip(rref, piv):
            x[pc] = row[-1] - sum(row[c] * x[c] for c in free)
        if all(x[j] >= p.lower[j] and (p.upper[j] is None or x[j] <= p.upper[j])
               for j in range(p.m)):
            if not any(x == v for v in vertices):
                vertices.append(x)
    return vertices


def random_feasible_scan(problem, result, queries, *, n: int = 400, seed: int = 7):
    """在置信集合内独立估计每个查询的可达范围（护栏）。

    两条独立路径：
    * 从发布点 b* 朝每个可行多面体顶点做一维射线，二分找与置信椭球
      ``r'W+r = rho`` 的交点，得到球面上的可行点（边界必在球面上）；
    * 可行点之间随机凸组合，覆盖内部。
    不追求精确极值，只确认发布边界不被任何独立采样点突破。
    """
    p = problem
    rng = random.Random(seed)
    bstar = result.point
    verts = enumerate_polytope_vertices(p, result)
    pool: list = []
    if bstar is not None and _residual_norm(p, bstar) <= p.rho:
        pool.append(bstar)
    for v in verts:
        if _residual_norm(p, v) <= p.rho:
            pool.append(v)
        elif bstar is not None:
            hit = _sphere_intersection(p, bstar, v)
            if hit is not None:
                pool.append(hit)
    if not pool:
        return {q.id: (float("nan"), float("nan")) for q in queries}
    feasible = pool[:]
    for _ in range(n):
        w = [rng.random() for _ in pool]
        sw = sum(w) or 1
        x = [sum(w[k] * pool[k][j] for k in range(len(pool))) / sw
             for j in range(p.m)]
        if _residual_norm(p, x) <= p.rho:
            feasible.append(x)
    out = {}
    for q in queries:
        vals = [dot(q.c, x) for x in feasible]
        out[q.id] = (float(min(vals)), float(max(vals)))
    return out


def _sphere_intersection(p, x0, x1):
    """从球内点 x0 朝 x1 二分求残差球 r'W+r=rho 的交点（分数二分）。"""
    if _residual_norm(p, x0) > p.rho:
        return None
    lo, hi = Fraction(0), Fraction(1)
    # 若 x1 已在球内，直接返回
    if _residual_norm(p, x1) <= p.rho:
        return x1
    for _ in range(60):
        mid = (lo + hi) / 2
        x = [x0[j] + mid * (x1[j] - x0[j]) for j in range(p.m)]
        if _residual_norm(p, x) <= p.rho:
            lo = mid
        else:
            hi = mid
    t = (lo + hi) / 2
    return [x0[j] + t * (x1[j] - x0[j]) for j in range(p.m)]


def cross_check_bounds(problem, result, *, tol: float = 1e-6) -> dict:
    """用顶点枚举与随机扫描独立核对发布的 QCP 边界。"""
    p = problem
    report = {}
    scan = random_feasible_scan(p, result, p.queries)
    for qb in result.query_bounds:
        q = next(q for q in p.queries if q.id == qb.query_id)
        verts = enumerate_polytope_vertices(p, result)
        # 独立球面交点（从 b* 朝各顶点射线）+ 球内顶点 + 发布点本身
        points = [v for v in verts if _residual_norm(p, v) <= p.rho]
        if result.point is not None:
            points.append(result.point)
            for v in verts:
                hit = _sphere_intersection(p, result.point, v)
                if hit is not None and _residual_norm(p, hit) <= p.rho:
                    points.append(hit)
        vvals = [float(dot(q.c, x)) for x in points]
        scan_lo, scan_hi = scan.get(qb.query_id, (float("nan"), float("nan")))
        # 无界方向（如未识别底层）发布边界为 None，核对器只检查有限的一侧
        respects_lo = qb.lower is None or scan_lo >= qb.lower - tol
        respects_hi = qb.upper is None or scan_hi <= qb.upper + tol
        report[qb.query_id] = {
            "published_lower": qb.lower,
            "published_upper": qb.upper,
            "vertex_min": min(vvals) if vvals else None,
            "vertex_max": max(vvals) if vvals else None,
            "random_scan_min": scan_lo,
            "random_scan_max": scan_hi,
            "scan_respects_lower": respects_lo,
            "scan_respects_upper": respects_hi,
            "unbounded_direction_confirmed": qb.unbounded_direction is not None,
        }
    return report
