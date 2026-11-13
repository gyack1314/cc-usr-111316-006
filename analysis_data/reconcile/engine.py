"""联合一致校正的分析引擎。

口径（见 ``docs/data-contract.md``）：

1. ``W = Q diag(lambda) Q'``：正特征方向给加权费用，零特征方向
   ``z'r = 0`` 是**精确约束**。本实现不用任何岭项，零方向由精确
   LDL/零空间算得，矛盾时只输出冲突证据、不发布校正值。
2. 词典序两阶段：先在精确约束与底层界上最小化加权残差
   ``(Sb-y)' W^+ (Sb-y)``；仅当最优底层解不唯一时，再在“最优面”
   （精确约束外加残差固定为最优值的线性化等式）上最小化
   ``||b-b0||^2``。两个目标绝不加权混合。
3. 置信集合 ``{b : 精确约束, 界, 加权残差 <= rho}``：对每个查询
   ``c'b`` 分别求最小/最大，并保留取得边界的完整底层向量；
   ``rho`` 小于最优残差时集合为空，明确报告，不取绝对值、不放宽。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from . import _linalg
from ._numeric import dot, matmul, to_float, transpose, zeros
from ._qcp_float import QCPResult, qcp_solve_float
from ._solver import QPResult, linear_feasibility, qp_solve
from .materialize import Problem

__all__ = ["ReconciliationResult", "reconcile", "QueryBound", "ConflictReport"]


@dataclass
class QueryBound:
    query_id: str
    lower: float | None
    upper: float | None
    lower_vector: list[float] | None
    upper_vector: list[float] | None
    lower_residual: float | None
    upper_residual: float | None
    # 浮点 QCP 的精度证据（点校正本身仍是精确分数）
    lower_numeric: dict | None = None
    upper_numeric: dict | None = None
    unbounded_direction: list[Fraction] | None = None


@dataclass
class ConflictReport:
    kind: str
    detail: dict
    message: str


@dataclass
class ReconciliationResult:
    problem: Problem
    status: str                         # "published" | "conflict" | "empty_set"
    # W 结构诊断
    w_rank: int
    w_nullity: int
    eigenvalues: list[float]
    eigenvectors: list[list[float]]
    rank_tolerance: Fraction
    dropped_pivots: list[Fraction]
    exact_constraints: list[list[Fraction]]   # E b = f 的 E（原坐标）
    exact_rhs: list[Fraction]
    null_directions: list[list[Fraction]]     # W 零空间（残差坐标）
    # 点校正
    point: list[Fraction] | None = None
    stage1_objective: Fraction | None = None
    stage1_optima_count: int | None = None
    stage2_used: bool | None = None
    stage2_objective: Fraction | None = None
    kkt_stage1: dict | None = None
    kkt_stage2: dict | None = None
    residual: list[Fraction] | None = None
    fitted_nodes: dict[str, Fraction] | None = None
    # 置信集合
    rho: Fraction | None = None
    optimal_residual: Fraction | None = None
    set_empty: bool | None = None
    query_bounds: list[QueryBound] = field(default_factory=list)
    # 未识别方向（可行域内的自由度）
    unidentified_directions: list[list[Fraction]] = field(default_factory=list)
    # 冲突证据
    conflict: ConflictReport | None = None
    # 回代核对
    backsub: dict | None = None

    # ---- 序列化 ----
    def to_dict(self, *, exact: bool = True) -> dict:
        def f(x):
            if x is None:
                return None
            return str(x) if exact else float(x)

        p = self.problem
        out: dict = {
            "status": self.status,
            "manifest_sha256": p.manifest_sha256,
            "bottom_ids": p.bottom_ids,
            "observed_nodes": p.observed_node_ids,
            "missing_nodes": p.missing_node_ids,
            "net_transfer": sorted(p.net_transfer),
            "covariance": {
                "rank": self.w_rank,
                "nullity": self.w_nullity,
                "rank_tolerance": str(self.rank_tolerance),
                "dropped_pivots": [str(x) for x in self.dropped_pivots],
                "eigenvalues_float": self.eigenvalues,
                "null_directions_residual": [
                    [str(x) for x in z] for z in self.null_directions
                ],
            },
            "exact_constraints": {
                "E": [[str(x) for x in row] for row in self.exact_constraints],
                "f": [str(x) for x in self.exact_rhs],
            },
            "unidentified_directions": [
                [str(x) for x in z] for z in self.unidentified_directions
            ],
        }
        if self.point is not None:
            out["point_correction"] = {
                "b": [f(x) for x in self.point],
                "residual": [f(x) for x in self.residual],
                "stage1_weighted_residual": f(self.stage1_objective),
                "stage1_optima_count": self.stage1_optima_count,
                "stage2_used": self.stage2_used,
                "stage2_distance_squared": f(self.stage2_objective),
                "fitted_nodes": {k: f(v) for k, v in self.fitted_nodes.items()},
                "kkt": _kkt_to_dict(self.kkt_stage1, f),
                "kkt_stage2": _kkt_to_dict(self.kkt_stage2, f),
            }
        if self.conflict is not None:
            out["conflict"] = {
                "kind": self.conflict.kind,
                "message": self.conflict.message,
                "detail": to_float(self.conflict.detail) if not exact else
                          _frac_str(self.conflict.detail),
            }
        out["confidence_set"] = {
            "rho": f(self.rho),
            "optimal_residual": f(self.optimal_residual),
            "empty": self.set_empty,
            "bounds": [
                {
                    "query_id": qb.query_id,
                    "lower": qb.lower,
                    "upper": qb.upper,
                    "lower_vector": qb.lower_vector,
                    "upper_vector": qb.upper_vector,
                    "lower_residual": qb.lower_residual,
                    "upper_residual": qb.upper_residual,
                    "lower_numeric_evidence": qb.lower_numeric,
                    "upper_numeric_evidence": qb.upper_numeric,
                    "unbounded_direction": (
                        [f(x) for x in qb.unbounded_direction]
                        if qb.unbounded_direction else None
                    ),
                }
                for qb in self.query_bounds
            ],
        }
        if self.backsub is not None:
            out["backsubstitution_checks"] = self.backsub
        return out


def _frac_str(obj):
    if isinstance(obj, Fraction):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _frac_str(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_frac_str(v) for v in obj]
    return obj


def _kkt_to_dict(kkt: dict | None, f) -> dict | None:
    if kkt is None:
        return None
    return {
        "active_set": list(kkt.get("active", ())),
        "equality_multipliers": [f(x) for x in kkt.get("equality_multipliers", [])],
        "lower_multipliers": [f(x) for x in kkt.get("lower_multipliers", [])],
        "upper_multipliers": [f(x) for x in kkt.get("upper_multipliers", [])],
        "quadratic_slack": f(kkt.get("quad_slack")),
        "candidates_checked": kkt.get("candidates_checked"),
    }


# --------------------------------------------------------------------------- #
def _weighted_forms(problem: Problem):
    """构建 W^+、精确约束 E b = f 与阶段一 H,g。"""
    p = problem
    ldl = p.ldl
    d = ldl["d"]
    r = ldl["rank"]
    n = p.n_obs
    # Moore–Penrose 逆，用行空间基稳健构造（天然免疫 LDL 行列置换）：
    # 取 W 的行空间基 B（RREF 非零行，r 行），W 在该子空间正定，故
    # W^+ = B' (B W B')^{-1} B。零空间方向由 ldl 另行精确登记为约束。
    basis = _linalg.row_basis(p.w)
    if not basis:
        w_pinv = zeros(n, n)
    else:
        b_t = transpose(basis)                    # n×r
        bwb = matmul(matmul(basis, p.w), b_t)     # r×r 正定
        w_pinv = matmul(matmul(b_t, _invert_dense(bwb)), basis)

    # 零空间基（残差坐标），转成底层精确约束：z'(Sb-y)=0，即 (z'S)b = z'y。
    # 注意 z'S 可能是零行（重复汇总行），此时约束退化为 0 = z'y：
    # 绝不能丢弃，它正是“两个完全相关预测值不一致”的冲突证据来源。
    null_dirs = ldl["null_basis"]
    E = [
        [sum(z[i] * p.s[i][j] for i in range(n)) for j in range(p.m)]
        for z in null_dirs
    ]
    f = [dot(z, p.y) for z in null_dirs]

    # 阶段一目标 1/2 b'H b + g'b；其最优值 + const 即加权残差 r'W+r
    St = transpose(p.s)
    H = matmul(matmul(St, w_pinv), p.s)
    g = [(-1) * sum(St[j][i] * sum(w_pinv[i][k] * p.y[k] for k in range(n))
                    for i in range(n)) for j in range(p.m)]
    const = sum(p.y[i] * sum(w_pinv[i][k] * p.y[k] for k in range(n))
                for i in range(n)) / 2
    return w_pinv, null_dirs, E, f, H, g, const


def _invert_dense(a: list[list[Fraction]]) -> list[list[Fraction]]:
    """精确求可逆稠密矩阵的逆（Gauss–Jordan 增广单位阵）。"""
    r = len(a)
    m = [a[i][:] + [Fraction(1) if i == j else Fraction(0) for j in range(r)]
         for i in range(r)]
    for c in range(r):
        pvt = next((i for i in range(c, r) if m[i][c] != 0), None)
        if pvt is None:  # pragma: no cover - 调用方保证可逆
            raise ValueError("矩阵奇异，无法求逆")
        m[c], m[pvt] = m[pvt], m[c]
        pv = m[c][c]
        m[c] = [v / pv for v in m[c]]
        for i in range(r):
            if i != c and m[i][c] != 0:
                f = m[i][c]
                m[i] = [x - f * y for x, y in zip(m[i], m[c])]
    return [row[r:] for row in m]


def reconcile(problem: Problem) -> ReconciliationResult:
    p = problem
    w_pinv, null_dirs, E, f, H, g, const = _weighted_forms(p)
    eig, qvec = _linalg.jacobi_eigen(p.w)
    rank = p.ldl["rank"]

    result = ReconciliationResult(
        problem=p,
        status="published",
        w_rank=rank,
        w_nullity=p.n_obs - rank,
        eigenvalues=eig,
        eigenvectors=qvec,
        rank_tolerance=p.rank_tolerance,
        dropped_pivots=p.ldl["dropped"],
        exact_constraints=E,
        exact_rhs=f,
        null_directions=null_dirs,
        rho=p.rho,
    )

    # ---- 阶段 0：精确约束与底层界的可行性（冲突则不发布校正值）----
    witness = linear_feasibility(E, f, p.lower, p.upper)
    if witness is not None:
        result.status = "conflict"
        result.conflict = ConflictReport(
            kind="exact_constraint_vs_bounds",
            detail=witness,
            message=(
                "W 零空间要求的精确合计与登记的底层上下界冲突；"
                "不允许加岭或放宽，故不发布校正值。"
            ),
        )
        result.backsub = _backsub_only(p, result)
        return result

    # ---- 阶段 1：最小化加权残差 ----
    s1: QPResult = qp_solve(H, g, E, f, p.lower, p.upper)
    if s1.status != "optimal":
        result.status = "conflict"
        result.conflict = ConflictReport(
            kind=f"stage1_{s1.status}",
            detail={"solver": s1.status},
            message="阶段一加权最小化未得到最优解。",
        )
        return result

    b1 = s1.x
    r1 = [sum(p.s[i][j] * b1[j] for j in range(p.m)) - p.y[i] for i in range(p.n_obs)]
    obj1 = _weighted_sq(r1, w_pinv)
    result.stage1_objective = obj1
    result.stage1_optima_count = len(s1.all_optima)
    result.kkt_stage1 = {**s1.kkt, "candidates_checked": s1.candidates_checked}

    # ---- 阶段 2：仅当阶段一存在多个最优底层解时才启用 ----
    # 凸 QP 的最优面是仿射集。令 grad = H b1 + g：对任意可行方向 d
    # （E d = 0 且保持箱内），一阶最优性给出 grad'd = 0，反之在凸二次下
    # grad'(b-b1)=0 与目标值不变等价。因此最优面＝精确约束 ∩ {grad'b = grad'b1}。
    # 若阶段一解唯一（枚举只找到一个 KKT 最优点，且零空间内无目标平坦方向），
    # 就没有第二阶段可言，直接发布 b1，绝不能在整个空间上另做最近点投影。
    unique_stage1 = len(s1.all_optima) == 1 and _is_unique_quadratic_optimum(
        H, E, f, b1, p
    )
    if unique_stage1:
        result.point = b1
        result.stage2_used = False
        result.stage2_objective = sum((b1[i] - p.b0[i]) ** 2 for i in range(p.m))
    else:
        grad = [sum(H[i][j] * b1[j] for j in range(p.m)) + g[i] for i in range(p.m)]
        face_rhs = dot(grad, b1)
        if any(v != 0 for v in grad):
            E2 = E + [grad]
            f2 = f + [face_rhs]
        else:
            E2, f2 = E, f
        H2 = [[Fraction(2) if i == j else Fraction(0) for j in range(p.m)]
              for i in range(p.m)]
        g2 = [Fraction(-2) * p.b0[i] for i in range(p.m)]
        s2: QPResult = qp_solve(H2, g2, E2, f2, p.lower, p.upper)
        if s2.status != "optimal":
            # 阶段二退化（不应发生：阶段一点本身可行）；保守发布阶段一点并标注
            result.point = b1
            result.stage2_used = False
        else:
            result.point = s2.x
            result.stage2_used = any(s2.x[i] != b1[i] for i in range(p.m))
            result.stage2_objective = sum(
                (s2.x[i] - p.b0[i]) ** 2 for i in range(p.m)
            )
            result.kkt_stage2 = {**s2.kkt, "candidates_checked": s2.candidates_checked}

    b = result.point
    result.residual = [
        sum(p.s[i][j] * b[j] for j in range(p.m)) - p.y[i]
        for i in range(p.n_obs)
    ]
    # 全部登记节点（含缺测）的拟合值，逐行回代用
    result.fitted_nodes = {
        nid: dot(p.s_full[t], b) for t, nid in enumerate(p.all_node_ids)
    }
    result.optimal_residual = _weighted_sq(result.residual, w_pinv)

    # ---- 未识别方向：精确约束零空间与可行箱内的自由度 ----
    result.unidentified_directions = _unidentified(E, p)

    # ---- 置信集合边界 ----
    result.set_empty = p.rho + 0 < result.optimal_residual
    if result.set_empty:
        result.status = "empty_set"
    else:
        for q in p.queries:
            result.query_bounds.append(
                _query_bound(p, q, E, f, H, g, w_pinv, b, result.optimal_residual)
            )

    result.backsub = _backsub_only(p, result)
    return result


def _is_unique_quadratic_optimum(H, E, f, b1, p: Problem) -> bool:
    """阶段一最优是否唯一：可行切空间里不存在 H 的零曲率方向。

    精确约束把可行集写成仿射参数形式 ``b = b_p + N t``（RREF 消元，
    N 的列是自由坐标方向，主元变量随 t 线性变化）。目标在自由坐标上的
    约化 Hessian 是 ``N'H N``：它在某个自由方向退化（零曲率）才说明
    阶段一存在连续多个最优解。直接在原始坐标判 ``z'Hz`` 会漏掉主元
    变量跟随变化带来的抵消，因此这里必须先参数化。
    """
    m = p.m
    if E:
        # 先剥掉 0=0 平凡行（W 零方向落在 S 零空间时产生），它们不约束任何变量
        nz = [(row, rhs) for row, rhs in zip(E, f) if any(v != 0 for v in row)]
        if nz:
            rref, piv = _linalg.rref(
                [row[:] + [rhs] for row, rhs in nz]
            )
        else:
            rref, piv = [], []
        free_cols = [c for c in range(m) if c not in piv]
        # b = bp + N t：bp 取自由变量为 0 的特解，N 列为自由坐标单位扰动
        bp = [Fraction(0)] * m
        for row, pc in zip(rref, piv):
            bp[pc] = row[-1]
        n_basis = []
        for fc in free_cols:
            col = [Fraction(0)] * m
            col[fc] = Fraction(1)
            for row, pc in zip(rref, piv):
                col[pc] = -row[fc]
            n_basis.append(col)
    else:
        bp = [Fraction(0)] * m
        n_basis = _linalg.identity(m)

    k = len(n_basis)
    # 约化 Hessian Hr = N' H N；半正定，出现零特征即非唯一
    for a in range(k):
        # Hr 在第 a 个对角元及与已判定方向的二次型
        za = n_basis[a]
        ha = [sum(H[i][j] * za[j] for j in range(m)) for i in range(m)]
        if all(sum(ha[i] * n_basis[b][i] for i in range(m)) == 0 for b in range(k)):
            # 该自由方向与 H 完全正交：零曲率，检查界是否钉死
            z = za
            can_plus = any(
                (z[j] > 0 and (p.upper[j] is None or b1[j] < p.upper[j]))
                or (z[j] < 0 and b1[j] > p.lower[j])
                for j in range(m)
            )
            can_minus = any(
                (z[j] < 0 and (p.upper[j] is None or b1[j] < p.upper[j]))
                or (z[j] > 0 and b1[j] > p.lower[j])
                for j in range(m)
            )
            if can_plus or can_minus:
                return False
    return True


def _weighted_sq(r: list[Fraction], w_pinv) -> Fraction:
    return sum(r[i] * sum(w_pinv[i][k] * r[k] for k in range(len(r)))
               for i in range(len(r)))


def _unidentified(E: list[list[Fraction]], p: Problem) -> list[list[Fraction]]:
    """可行域切空间：E 的零空间中、在非活动界处仍可移动的方向。"""
    if not E:
        basis = _linalg.identity(p.m)
    else:
        basis = _linalg.nullspace(E)
    # 去掉被有限双侧箱完全钉死的坐标方向（保留业务可解释的自由度）
    out = []
    for z in basis:
        blocked = False
        for j, v in enumerate(z):
            if v == 0:
                continue
            if p.upper[j] is not None and p.upper[j] == p.lower[j]:
                blocked = True
        if not blocked:
            out.append(z)
    return out


def _query_bound(
    p: Problem,
    query,
    E, f, H, g, w_pinv,
    b_star: list[Fraction],
    opt_residual: Fraction,
) -> QueryBound:
    m = p.m
    c = query.c
    # 二次约束：加权残差 1/2 b'H b + g'b + const <= rho
    kappa = p.rho - _const_term(p, w_pinv)
    # 求解器最小化 sign*c'x：sign=+1 给 c'x 的最小值（下界），
    # sign=-1 给 -c'x 的最小值即 c'x 的最大值（上界）。
    lo_res = qcp_solve_float(+1, c, E, f, p.lower, p.upper, H, g, kappa)
    hi_res = qcp_solve_float(-1, c, E, f, p.lower, p.upper, H, g, kappa)

    def unpack(qr: QCPResult):
        if qr.status != "optimal" or qr.x is None:
            return None, None, None, None
        xf = qr.x
        # 回代核验：在精确分数约束上评估浮点解的违反量与残差
        r_fl = [sum(float(p.s[i][j]) * xf[j] for j in range(m)) - float(p.y[i])
                for i in range(p.n_obs)]
        residual = _weighted_sq_float(r_fl, w_pinv)
        eq_viol = max(
            (abs(sum(float(E[e][j]) * xf[j] for j in range(m)) - float(f[e]))
             for e in range(len(E))),
            default=0.0,
        )
        bound_viol = 0.0
        for j in range(m):
            bound_viol = max(bound_viol, float(p.lower[j]) - xf[j])
            if p.upper[j] is not None:
                bound_viol = max(bound_viol, xf[j] - float(p.upper[j]))
        numeric = {
            "lambda": qr.lam,
            "quadratic_slack": qr.quad_slack,
            "active_set": list(qr.active),
            "equality_violation_float": eq_viol,
            "bound_violation_float": max(0.0, bound_viol),
            "candidates_checked": qr.candidates_checked,
            "arithmetic": "float64 (QCP 根一般为代数无理数；点校正仍为精确分数)",
        }
        return xf, sum(float(c[j]) * xf[j] for j in range(m)), residual, numeric

    lv, lb, lres, lnum = unpack(lo_res)
    hv, hb, hres, hnum = unpack(hi_res)
    # 无界方向直接取 QCP 求解器识别到的零曲率射线，转成精确分数登记
    unb = None
    ray = None
    if lo_res.status == "unbounded" and lo_res.unbounded_ray is not None:
        ray = lo_res.unbounded_ray
    elif hi_res.status == "unbounded" and hi_res.unbounded_ray is not None:
        ray = hi_res.unbounded_ray
    if ray is not None:
        unb = [Fraction(x).limit_denominator(10 ** 12) for x in ray]
    return QueryBound(
        query_id=query.id,
        lower=lb,
        upper=hb,
        lower_vector=lv,
        upper_vector=hv,
        lower_residual=lres,
        upper_residual=hres,
        lower_numeric=lnum,
        upper_numeric=hnum,
        unbounded_direction=unb,
    )


def _weighted_sq_float(r: list[float], w_pinv) -> float:
    n = len(r)
    wp = [[float(w_pinv[i][k]) for k in range(n)] for i in range(n)]
    return sum(r[i] * sum(wp[i][k] * r[k] for k in range(n)) for i in range(n))


def _const_term(p: Problem, w_pinv) -> Fraction:
    n = p.n_obs
    return sum(p.y[i] * sum(w_pinv[i][k] * p.y[k] for k in range(n))
               for i in range(n)) / 2


def _backsub_only(p: Problem, result: ReconciliationResult) -> dict:
    """逐行回代所有登记的 S 关系并核对结构性质。"""
    checks: dict = {"rows": [], "all_exact_constraints_hold": None}
    b = result.point
    exact_ok: list[bool] = []
    for t, nid in enumerate(p.all_node_ids):
        row = p.s_full[t]
        lhs = dot(row, b) if b is not None else None
        checks["rows"].append({
            "node": nid,
            "linear_form": {p.bottom_ids[j]: str(row[j]) for j in range(p.m) if row[j]},
            "fitted": None if lhs is None else str(lhs),
        })
    if b is not None:
        for i, (erow, rhs) in enumerate(zip(result.exact_constraints, result.exact_rhs)):
            exact_ok.append(dot(erow, b) == rhs)
        checks["all_exact_constraints_hold"] = all(exact_ok) if exact_ok else True
        checks["within_bounds"] = all(
            b[j] >= p.lower[j] and (p.upper[j] is None or b[j] <= p.upper[j])
            for j in range(p.m)
        )
        checks["nullspace_residual_zero"] = all(
            dot(z, result.residual) == 0 for z in result.null_directions
        )
    return checks
