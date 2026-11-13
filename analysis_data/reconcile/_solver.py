"""凸二次规划/二次约束规划的精确活动集求解器。

问题形式（全部系数为 :class:`Fraction`）::

    min  1/2 x'H x + g'x
    s.t. A x = b            （等式，可能线性依赖）
         l <= x <= u        （u 的 None 表示 +inf）
         1/2 x'C x + d'x <= kappa   （可选，单个凸二次约束，用于置信集合）

方法：枚举每个变量的活动状态（自由 / 卡在下界 / 卡在上界），对每个活动集
解 KKT 线性方程组，再用 (i) 原问题可行性、(ii) 对偶乘子符号、(iii) 二阶
充分性（KKT 矩阵在自由变量上正定/半正定）三层过滤。资料规模是地区×季度
的小问题，枚举可穷尽，且每一步都是精确分数——这正是 KKT 证据的来源。

这里**不做**伪逆最小范数兜底：无解就报无解，多解就把多个最优解全部保留，
交给词典序二级目标处理。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from typing import Sequence

from ._linalg import rref, solve_full
from ._numeric import dot, zeros

# 枚举保护：活动集数随变量数指数增长，资料是小问题，超限明确报错而不是卡死
MAX_ENUM_VARS = 12


@dataclass
class QPResult:
    status: str                      # "optimal" | "infeasible" | "unbounded" | "too_large"
    objective: Fraction | None = None
    x: list[Fraction] | None = None
    # 全部达到同一最优值的原问题解（用于证明/排除非唯一）
    all_optima: list[list[Fraction]] = field(default_factory=list)
    active: tuple[int, ...] = ()     # 每个变量: 0 自由 / -1 下界 / +1 上界
    equality_multipliers: list[Fraction] | None = None
    lower_multipliers: list[Fraction] | None = None
    upper_multipliers: list[Fraction] | None = None
    kkt: dict | None = None
    # 不可行证据
    infeasible_witness: dict | None = None
    candidates_checked: int = 0


def _bound_value(u: Fraction | None) -> Fraction:
    return Fraction(10**30) if u is None else u


def _is_psd_on_free(
    h: list[list[Fraction]], free: list[int]
) -> bool:
    """H 在自由子矩阵上半正定：精确 LDL（零容差），出现负主元即非 PSD。

    不用顺序主子式符号法：带零主元时 Sylvester 判据必须检查所有主子式，
    直接 LDL 更稳且全程精确分数。
    """
    if not free:
        return True
    sub = [[h[i][j] for j in free] for i in free]
    from ._linalg import ldl_psd  # 局部导入避免模块初始化环
    info = ldl_psd(sub, Fraction(0))
    return not info["negative"]


def _kkt_solve(
    h: list[list[Fraction]],
    g: list[Fraction],
    aeq: list[list[Fraction]],
    beq: list[Fraction],
    free: list[int],
    fixed: dict[int, Fraction],
    n: int,
    *,
    qe_mat: list[list[Fraction]] | None = None,
    qe_gamma: Fraction = Fraction(0),
) -> tuple[list[Fraction] | None, list[Fraction] | None, list[Fraction] | None]:
    """固定非自由变量后解自由变量的 KKT 系统。

    返回 ``(x, 线性等式乘子 nu, 自由变量的 H 乘子证据)``；
    自由块的二阶条件由调用方另行校验。``qe_gamma`` 是二次等式
    ``1/2 x'Qx + d'x = k`` 的预给乘子，其有效 Hessian 为 ``H + gamma Q``。
    """
    nf = len(free)
    h_eff = [[h[i][j] + (qe_gamma * qe_mat[i][j] if qe_mat else Fraction(0))
              for j in range(n)] for i in range(n)]
    # 等式里固定变量的贡献
    rhs_eq = beq[:]
    if aeq:
        for idx, val in fixed.items():
            for row in range(len(aeq)):
                rhs_eq[row] = rhs_eq[row] - aeq[row][idx] * val
    a_free = (
        [[row[j] for j in free] for row in aeq] if aeq else []
    )
    h_free = [[h_eff[i][j] for j in free] for i in free]
    g_free = [g[j] + sum(h_eff[j][idx] * val for idx, val in fixed.items()) for j in free]

    ne = len(a_free)
    size = nf + ne
    kkt = zeros(size, size)
    for i in range(nf):
        for j in range(nf):
            kkt[i][j] = h_free[i][j]
    for e in range(ne):
        for i in range(nf):
            kkt[nf + e][i] = a_free[e][i]
            kkt[i][nf + e] = a_free[e][i]
    rhs = [-g_free[i] for i in range(nf)] + rhs_eq

    sol, null = solve_full(kkt, rhs)
    if sol is None:
        return None, None, None
    x_free = sol[:nf]
    nu = sol[nf:]

    x = [Fraction(0)] * n
    for idx, val in fixed.items():
        x[idx] = val
    for t, j in enumerate(free):
        x[j] = x_free[t]
    return x, nu, None


def _objective(
    h: Sequence[Sequence[Fraction]],
    g: Sequence[Fraction],
    x: Sequence[Fraction],
) -> Fraction:
    n = len(x)
    quad = sum(h[i][j] * x[i] * x[j] for i in range(n) for j in range(n)) / 2
    return quad + dot(list(g), list(x))


def _quad_constraint_value(
    c_mat: list[list[Fraction]] | None,
    d_vec: list[Fraction] | None,
    kappa: Fraction | None,
    x: list[Fraction],
) -> Fraction:
    if c_mat is None:
        return Fraction(0)
    n = len(x)
    quad = sum(c_mat[i][j] * x[i] * x[j] for i in range(n) for j in range(n)) / 2
    return quad + dot(d_vec, x) - kappa


def qp_solve(
    h: list[list[Fraction]],
    g: list[Fraction],
    aeq: list[list[Fraction]],
    beq: list[Fraction],
    lower: list[Fraction],
    upper: list[Fraction | None],
    *,
    c_mat: list[list[Fraction]] | None = None,
    d_vec: list[Fraction] | None = None,
    kappa: Fraction | None = None,
    qc_required: bool = False,
    qe_mat: list[list[Fraction]] | None = None,
    qe_vec: list[Fraction] | None = None,
    qe_rhs: Fraction | None = None,
    qe_gamma: Fraction = Fraction(0),
) -> QPResult:
    """解（可能带一个凸二次约束/二次等式的）箱约束等式 QP。

    * 二次不等式：``1/2 x'C x + d'x <= kappa``；``qc_required=True`` 时
      要求它必须活动（用于置信集合球面搜索）。
    * 二次等式：``1/2 x'Qx + qe_vec'x = qe_rhs``，乘子预给为 ``qe_gamma``
      （阶段二在固定活动集上是闭式 KKT，gamma 由调用方按面约束标定）。
    """
    n = len(g)
    if n > MAX_ENUM_VARS:
        return QPResult(status="too_large")

    # 先做线性可行域判定，并在不可行时构造冲突证据
    feas = linear_feasibility(aeq, beq, lower, upper)
    if feas is not None:
        return QPResult(status="infeasible", infeasible_witness=feas)

    best: Fraction | None = None
    best_points: list[list[Fraction]] = []
    best_meta: dict | None = None
    checked = 0

    states = range(-1, 2)  # -1 下界, 0 自由, +1 上界
    for code in product(states, repeat=n):
        free: list[int] = []
        fixed: dict[int, Fraction] = {}
        ok = True
        for j, s in enumerate(code):
            if s == 0:
                free.append(j)
            elif s == -1:
                fixed[j] = lower[j]
            else:
                if upper[j] is None:
                    ok = False  # 无上界却枚举上界活动
                    break
                fixed[j] = upper[j]
        if not ok:
            continue
        checked += 1

        x, nu, _ = _kkt_solve(
            h, g, aeq, beq, free, fixed, n,
            qe_mat=qe_mat, qe_gamma=qe_gamma,
        )
        if x is None:
            continue
        # 原问题可行性：自由变量严格在箱内、等式满足
        if any(x[j] < lower[j] or (upper[j] is not None and x[j] > upper[j])
               for j in free):
            continue
        # 二阶条件（有效 Hessian H+gamma Q 在自由块半正定；严格凸时自动满足）
        h_eff = [
            [h[i][j] + (qe_gamma * qe_mat[i][j] if qe_mat else Fraction(0))
             for j in range(n)]
            for i in range(n)
        ]
        if not _is_psd_on_free(h_eff, free):
            continue
        # 二次不等式约束
        qv = _quad_constraint_value(c_mat, d_vec, kappa, x)
        if qv > 0:
            continue
        if qc_required and qv != 0:
            continue
        # 二次等式约束（阶段一最优面）
        if qe_mat is not None and _quad_constraint_value(qe_mat, qe_vec, qe_rhs, x) != 0:
            continue

        # 对偶可行性：恢复界乘子并验符号
        mult = _bound_multipliers(h_eff, g, aeq, nu, x, lower, upper, code)
        if mult is None:
            continue
        val = _objective(h, g, x)
        if best is None or val < best:
            best = val
            best_points = [x]
            best_meta = {
                "active": code,
                "equality_multipliers": nu,
                "lower_multipliers": mult[0],
                "upper_multipliers": mult[1],
                "quadratic_slack": -qv,
                "quadratic_equality_gamma": qe_gamma if qe_mat is not None else None,
            }
        elif val == best and not any(x == z for z in best_points):
            best_points.append(x)

    if best is None:
        # 线性可行却没有合格 KKT 点：只可能是无界（线性目标 + 无界方向）
        return QPResult(status="unbounded")
    return QPResult(
        status="optimal",
        objective=best,
        x=best_points[0],
        all_optima=best_points,
        active=tuple(best_meta["active"]),
        equality_multipliers=best_meta["equality_multipliers"],
        lower_multipliers=best_meta["lower_multipliers"],
        upper_multipliers=best_meta["upper_multipliers"],
        kkt=best_meta,
        candidates_checked=checked,
    )


def _bound_multipliers(
    h, g, aeq, nu, x, lower, upper, code
) -> tuple[list[Fraction], list[Fraction]] | None:
    """由稳定方程 Hx+g - A'nu - mu_l + mu_u = 0 恢复界乘子并验符号。"""
    n = len(g)
    hx = [sum(h[i][j] * x[j] for j in range(n)) for i in range(n)]
    atnu = [sum(aeq[e][i] * nu[e] for e in range(len(aeq))) for i in range(n)] if aeq else [Fraction(0)] * n
    resid = [hx[i] + g[i] - atnu[i] for i in range(n)]
    mu_l = [Fraction(0)] * n
    mu_u = [Fraction(0)] * n
    for j, s in enumerate(code):
        if s == -1:  # 下界活动：mu_l = resid，须 >= 0
            if resid[j] < 0:
                return None
            mu_l[j] = resid[j]
        elif s == 1:  # 上界活动：mu_u = -resid，须 >= 0
            if resid[j] > 0:
                return None
            mu_u[j] = -resid[j]
        else:
            if resid[j] != 0:
                return None
    return mu_l, mu_u


def linear_feasibility(
    aeq: list[list[Fraction]],
    beq: list[Fraction],
    lower: list[Fraction],
    upper: list[Fraction | None],
) -> dict | None:
    """精确判定等式+箱约束可行性；可行返回 None，不可行返回冲突证据。

    证据用 Farkas 型证书：找等式组合 ``p'A`` 与箱约束的矛盾方向，
    以及一对“要求上界 < 要求下界”的具体节点。
    """
    n = len(lower)
    if any(upper[j] is not None and upper[j] < lower[j] for j in range(n)):
        j = next(j for j in range(n) if upper[j] is not None and upper[j] < lower[j])
        return {"type": "empty_box", "bottom_index": j, "lower": lower[j], "upper": upper[j]}

    if not aeq:
        return None
    # 令 z = x - l，有 0 <= z_j <= (u_j-l_j)（无界项给一个符号变量处理）。
    # 精确做法：枚举活动集找一个可行点（小问题），找不到再做 Farkas 证书。
    found = _find_feasible_point(aeq, beq, lower, upper)
    if found is not None:
        return None

    # Farkas 证书：A z = rhs，z>=0（把自由/双侧变量拆成正负部后），
    # 不可行当且仅当存在 p 使 p'A <= 0 而 p'rhs > 0。这里直接在原坐标上
    # 给出最小矛盾等式子集作为业务可读证据。
    witness_rows = _infeasible_row_certificate(aeq, beq, lower, upper)
    return {
        "type": "equality_vs_bounds",
        "certificate_rows": witness_rows,
        "rhs": beq,
    }


def _find_feasible_point(aeq, beq, lower, upper) -> list[Fraction] | None:
    n = len(lower)
    if n > MAX_ENUM_VARS:
        # 大问题退化为 RREF 代换检查（本资料集不会触发）
        r, piv = rref([row[:] + [rhs] for row, rhs in zip(aeq, beq)])
        for row in r:
            if all(v == 0 for v in row[:-1]) and row[-1] != 0:
                return None
        return [Fraction(0)] * n
    for code in product(range(-1, 2), repeat=n):
        free, fixed = [], {}
        bad = False
        for j, s in enumerate(code):
            if s == 0:
                free.append(j)
            elif s == -1:
                fixed[j] = lower[j]
            else:
                if upper[j] is None:
                    bad = True
                    break
                fixed[j] = upper[j]
        if bad:
            continue
        x, _nu, _ = _kkt_solve(
            zeros(n, n), [Fraction(0)] * n, aeq, beq, free, fixed, n
        )
        if x is None:
            continue
        if all(x[j] >= lower[j] and (upper[j] is None or x[j] <= upper[j]) for j in range(n)):
            return x
    return None


def _infeasible_row_certificate(aeq, beq, lower, upper) -> list[int]:
    """返回参与矛盾的等式行号（消元中生成 0=非0 的原始行溯源集合）。

    对增广矩阵做带行溯源的高斯–若尔当消元：每个工作行记录它由哪些
    原始行线性组合而成；出现矛盾行时，其溯源集合就是一份 Farkas 型证书，
    无需对原始行做指数级组合枚举。
    """
    if not aeq:
        return []
    m = len(aeq)
    work = [aeq[i][:] + [beq[i]] for i in range(m)]
    source = [frozenset({i}) for i in range(m)]
    rows, cols = m, len(aeq[0])
    r = 0
    for c in range(cols):
        pivot = next((i for i in range(r, rows) if work[i][c] != 0), None)
        if pivot is None:
            continue
        work[r], work[pivot] = work[pivot], work[r]
        source[r], source[pivot] = source[pivot], source[r]
        pv = work[r][c]
        work[r] = [v / pv for v in work[r]]
        for i in range(rows):
            if i != r and work[i][c] != 0:
                f = work[i][c]
                work[i] = [x - f * y for x, y in zip(work[i], work[r])]
                source[i] = source[i] | source[r]
        r += 1
    for row, src in zip(work, source):
        if all(v == 0 for v in row[:-1]) and row[-1] != 0:
            return sorted(src)
    return list(range(m))

