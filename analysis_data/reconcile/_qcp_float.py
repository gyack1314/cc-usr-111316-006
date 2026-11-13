"""置信集合边界的浮点 QCP 求解器（与精确 QP 路径刻意分离）。

为什么这里用浮点
==================
点校正、可行性、零空间与秩判定全部走 :mod:`_solver` 的精确有理数路径；
但置信集合边界是凸 QCQP，其 KKT 乘子 lambda 在一般有理输入下是
*代数无理数*，不存在有限分数表示。这里诚实地用双精度浮点解特征方程，
并在 :mod:`engine` 中把返回向量回代到精确等式/界/二次约束，报告真实
残差精度，而不是把浮点结果伪装成精确分数。

方法
====
枚举箱活动集（下界/上界/自由），固定变量代入后在自由子空间解信赖域型
凸 QCQP：用 LDL 对角化得到特征方程 phi(lambda)=r，浮点二分求 lambda>=0，
恢复 t 后做 KKT 相容性与界乘子符号核验。凸性保证合格 KKT 点即全局最优。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Sequence

from ._numeric import dot
from ._solver import MAX_ENUM_VARS, linear_feasibility

FEAS_TOL = 1e-10
ROOT_TOL = 1e-13
UNBOUNDED = object()  # 活动集内检测到无界射线的哨兵


@dataclass
class QCPResult:
    status: str
    objective: float | None = None
    x: list[float] | None = None
    active: tuple[int, ...] = ()
    lam: float | None = None
    quad_slack: float | None = None
    candidates_checked: int = 0
    infeasible_witness: dict | None = None
    unbounded_ray: list[float] | None = None


def _fmat(a) -> list[list[float]]:
    return [[float(v) for v in row] for row in a]


def _fvec(v) -> list[float]:
    return [float(x) for x in v]


def _ldl(a: list[list[float]]):
    """对称半正定矩阵的 LDL（对角选主元），返回 L、正对角 d、秩。"""
    n = len(a)
    work = [row[:] for row in a]
    l = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    d: list[float] = []
    for k in range(n):
        j = max(range(k, n), key=lambda i: abs(work[i][i]))
        if abs(work[j][j]) <= 1e-14:
            break
        if j != k:
            work[k], work[j] = work[j], work[k]
            for i in range(n):
                work[i][k], work[i][j] = work[i][j], work[i][k]
            for i in range(k):
                l[i][k], l[i][j] = l[i][j], l[i][k]
        pivot = work[k][k]
        if pivot < -1e-12:
            raise ValueError("二次约束矩阵非半正定")
        d.append(pivot)
        for i in range(k + 1, n):
            l[i][k] = work[i][k] / pivot
        for i in range(k + 1, n):
            for h in range(k + 1, n):
                work[i][h] -= l[i][k] * pivot * l[h][k]
    return l, d, len(d)


def _invert_lower(L: list[list[float]]) -> list[list[float]]:
    n = len(L)
    inv = [[0.0] * n for _ in range(n)]
    for k in range(n):
        col = [0.0] * n
        for i in range(n):
            rhs = 1.0 if i == k else 0.0
            col[i] = rhs - sum(L[i][t] * col[t] for t in range(i))
        for i in range(n):
            inv[i][k] = col[i]
    return inv


def _nullspace_float(a: list[list[float]], tol: float = 1e-12) -> list[list[float]]:
    """浮点零空间（RREF 风格消元，返回行向量基）。"""
    if not a:
        return []
    m, n = len(a), len(a[0])
    work = [row[:] for row in a]
    pivots: list[int] = []
    r = 0
    for c in range(n):
        p = next((i for i in range(r, m) if abs(work[i][c]) > tol), None)
        if p is None:
            continue
        work[r], work[p] = work[p], work[r]
        pv = work[r][c]
        work[r] = [v / pv for v in work[r]]
        for i in range(m):
            if i != r and abs(work[i][c]) > tol:
                f = work[i][c]
                work[i] = [x - f * y for x, y in zip(work[i], work[r])]
        pivots.append(c)
        r += 1
    free = [c for c in range(n) if c not in pivots]
    basis = []
    for fc in free:
        z = [0.0] * n
        z[fc] = 1.0
        for row, pc in zip(work[:r], pivots):
            z[pc] = -row[fc]
        basis.append(z)
    return basis


def _solve(a: list[list[float]], b: list[float]):
    """高斯消元解（可能矩形的）线性方程组；无解返回 None，欠定返回特解。"""
    if not a:
        return [0.0] * (len(b) and 0)
    rows, n = len(a), len(a[0])
    m = [a[i][:] + [b[i]] for i in range(rows)]
    piv_cols = []
    r = 0
    for c in range(n):
        p = next((i for i in range(r, rows) if abs(m[i][c]) > 1e-14), None)
        if p is None:
            continue
        m[r], m[p] = m[p], m[r]
        pv = m[r][c]
        m[r] = [v / pv for v in m[r]]
        for i in range(rows):
            if i != r and abs(m[i][c]) > 1e-14:
                f = m[i][c]
                m[i] = [x - f * y for x, y in zip(m[i], m[r])]
        piv_cols.append(c)
        r += 1
    for i in range(r, rows):
        if abs(m[i][n]) > 1e-10 and all(abs(m[i][c]) <= 1e-14 for c in range(n)):
            return None
    x = [0.0] * n
    for row, c in zip(m[:r], piv_cols):
        x[c] = row[n]
    return x


def qcp_solve_float(
    sign: int,
    c,
    aeq,
    beq,
    lower,
    upper,
    c_mat,
    d_vec,
    kappa,
) -> QCPResult:
    """浮点解 ``extremize c'x``（sign=+1 最大 / -1 最小）于凸 QCQP 可行域。"""
    n = len(c)
    if n > MAX_ENUM_VARS:
        return QCPResult(status="too_large")
    # 可行性仍用精确路径给证书（入参为 Fraction）
    witness = linear_feasibility(aeq, beq, lower, upper)
    if witness is not None:
        return QCPResult(status="infeasible", infeasible_witness=witness)

    C = _fmat(c_mat)
    d = _fvec(d_vec)
    g = [sign * float(x) for x in c]
    A = _fmat(aeq) if aeq else []
    b = _fvec(beq)
    lo = _fvec(lower)
    hi = [float("inf") if u is None else float(u) for u in upper]
    kappa_f = float(kappa)

    best: float | None = None
    best_x: list[float] | None = None
    best_meta: dict | None = {}
    best_ray: list[float] | None = None
    checked = 0

    unbounded = False
    for code in product(range(-1, 2), repeat=n):
        free, fixed = [], {}
        ok = True
        for j, s in enumerate(code):
            if s == 0:
                free.append(j)
            elif s == -1:
                fixed[j] = lo[j]
            else:
                if upper[j] is None:
                    ok = False
                    break
                fixed[j] = hi[j]
        if not ok:
            continue
        checked += 1
        for item in _active_set_stationary(
            code, free, fixed, n, g, A, b, C, d, kappa_f, lo, hi
        ):
            if item[0] is UNBOUNDED:
                unbounded = True
                if best_ray is None:
                    best_ray = item[1]
                continue
            x, lam, slack, _ray = item
            if not _box_ok(x, free, lo, hi):
                continue
            val = sum(g[i] * x[i] for i in range(n))
            if best is None or val < best - 1e-12:
                best, best_x = val, x
                best_meta = {"active": code, "lambda": lam, "quadratic_slack": slack}
            elif abs(val - best) <= 1e-12 and best_x is not None:
                pass

    if unbounded:
        # 存在沿目标下降方向的无界射线：该极值严格无界，
        # 任何有限活动集点都不能覆盖它。
        return QCPResult(
            status="unbounded", candidates_checked=checked, unbounded_ray=best_ray
        )
    if best is None:
        return QCPResult(status="unbounded", candidates_checked=checked)
    return QCPResult(
        status="optimal",
        objective=best,
        x=best_x,
        active=tuple(best_meta["active"]),
        lam=best_meta["lambda"],
        quad_slack=best_meta["quadratic_slack"],
        candidates_checked=checked,
    )


def _box_ok(x, free, lo, hi):
    return all(x[j] >= lo[j] - FEAS_TOL and x[j] <= hi[j] + FEAS_TOL for j in free)


def _active_set_stationary(code, free, fixed, n, g, A, b, C, d, kappa, lo, hi):
    nf = len(free)
    rhs = b[:]
    for idx, val in fixed.items():
        for row in range(len(A)):
            rhs[row] -= A[row][idx] * val
    Af = [[A[e][j] for j in free] for e in range(len(A))] if A else []
    Cr = [[C[i][j] for j in free] for i in free]
    d_hat = [d[j] + sum(C[j][idx] * val for idx, val in fixed.items()) for j in free]
    q0 = (
        sum(C[i][j] * fixed.get(i, 0.0) * fixed.get(j, 0.0)
            for i in range(n) for j in range(n)) / 2
        + sum(d[idx] * val for idx, val in fixed.items())
    )
    g_hat = [g[j] for j in free]
    radius = kappa - q0

    if Af:
        z0 = _solve([row[:] for row in Af], rhs[:])
        if z0 is None:
            return
        null = _nullspace_float(Af)
    else:
        z0, null = [0.0] * nf, _identity_rows(nf)

    def assemble(t):
        z = z0[:]
        for tt, vec in enumerate(null):
            for i in range(nf):
                z[i] += vec[i] * t[tt]
        x = [0.0] * n
        for idx, val in fixed.items():
            x[idx] = val
        for tt, j in enumerate(free):
            x[j] = z[tt]
        return x, z

    if not null:
        # 自由变量被等式钉死：只校验球面
        _, z = assemble([])
        qv = _quad(Cr, d_hat, z)
        if abs(qv - radius) <= FEAS_TOL:
            yield assemble([])[0], 0.0, radius - qv, None
        return

    # 降维到零空间坐标 t：z = z0 + N t
    N = [[null[tt][i] for tt in range(len(null))] for i in range(nf)]  # nf x k
    k = len(null)
    Nt = [[N[i][t] for i in range(nf)] for t in range(k)]
    Ck = _gram(Nt, Cr, N)
    cz0 = [sum(Cr[i][j] * z0[j] for j in range(nf)) for i in range(nf)]
    dk = [sum(Nt[t][i] * (cz0[i] + d_hat[i]) for i in range(nf)) for t in range(k)]
    gk = [sum(Nt[t][i] * g_hat[i] for i in range(nf)) for t in range(k)]
    r_const = _quad(Cr, d_hat, z0)
    rk = radius - r_const

    # 零曲率方向检测：Ck 的零空间方向 v 满足 v'Ck v=0，二次约束不限制它。
    # 若线性目标沿 v 有非零分量，且该方向在底层箱内可无限延伸，则问题无界。
    for v in _nullspace_float(Ck):
        gproj = sum(gk[t] * v[t] for t in range(k))
        if abs(gproj) <= 1e-12:
            continue
        # v（零空间坐标）映回完整底层坐标：raw = N v，固定变量分量保持 0
        raw = [0.0] * n
        for t in range(k):
            for tt, j in enumerate(free):
                raw[j] += N[tt][t] * v[t]
        sign = 1.0 if gproj < 0 else -1.0  # 最小化目标，取下降方向
        if _ray_unbounded(raw, sign, free, fixed, lo, hi):
            full_ray = [sign * x for x in raw]
            yield UNBOUNDED, full_ray
            return  # 该活动集已确认无界，无需再枚举球面驻点

    # 情形 A：lambda=0，约束松弛，线性驻点要求 gk=0（自由方向无梯度）
    if all(abs(v) <= 1e-10 for v in gk):
        t = [0.0] * k
        x, z = assemble(t)
        slack = radius - _quad(Cr, d_hat, z)
        if slack > FEAS_TOL:
            yield x, 0.0, slack, None

    # 情形 B：lambda>0，球面活动，特征方程求根
    for lam, t in _trust_roots(Ck, dk, gk, rk):
        x, z = assemble(t)
        qv = _quad(Cr, d_hat, z)
        if abs(qv - radius) <= FEAS_TOL:
            yield x, lam, 0.0, None


def _ray_unbounded(raw, sign, free, fixed, lo, hi) -> bool:
    """方向 sign*raw 是否能在箱约束内无限延伸（固定变量保持不变）。"""
    for j in free:
        d = sign * raw[j]
        if abs(d) <= 1e-15:
            continue
        if d > 0 and not (hi[j] == float("inf")):
            return False
        if d < 0 and not (lo[j] == float("-inf")):
            return False
    return True


def _quad(C, d, z):
    n = len(z)
    return sum(C[i][j] * z[i] * z[j] for i in range(n) for j in range(n)) / 2 + sum(
        d[i] * z[i] for i in range(n)
    )


def _identity_rows(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _gram(a, m, b):
    tmp = [[sum(a[i][t] * m[t][j] for t in range(len(m))) for j in range(len(m))]
           for i in range(len(a))]
    return [[sum(tmp[i][t] * b[t][j] for t in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(tmp))]


def _trust_roots(Ck, dk, gk, rk):
    """中心化信赖域求 (lambda, t)。

    约束 ``1/2 t'Ck t + dk't = rk``。先解 ``Ck t0 = -dk`` 取椭球中心
    （Ck 在其值域上正定；零曲率坐标单独处理），令 w = t - t0，则约束
    化为标准信赖域 ``1/2 w'Ck w = R``，其中
    ``R = rk - (1/2 t0'Ck t0 + dk't0)``。目标线性项变为
    ``gk'(w+t0)``，驻点方程 ``lambda Ck w = -gk``（常数项不影响 argmin）。
    """
    k = len(Ck)
    L, diag, r = _ldl(Ck)
    Linv = _invert_lower(L)
    # 椭球中心 t0：解 Ck t0 = -dk（只在值域坐标上有定义）
    t0 = _solve([row[:] for row in Ck], [-x for x in dk])
    if t0 is None:
        return
    # 中心化半径
    center_val = (
        sum(Ck[i][j] * t0[i] * t0[j] for i in range(k) for j in range(k)) / 2
        + sum(dk[i] * t0[i] for i in range(k))
    )
    radius = rk - center_val
    if radius < -FEAS_TOL:
        return  # 该活动集下椭球内部为空（被界削掉的情形由别的活动集覆盖）

    # 转到 LDL 坐标 u = L' w，Ck = L D L'，则 1/2 u' D u = R
    Lti = [[Linv[j][i] for j in range(k)] for i in range(k)]  # L^{-T}
    gu = [sum(Lti[i][j] * gk[j] for j in range(k)) for i in range(k)]

    # 零曲率坐标（i>=r）：驻点要求 gu_i=0，否则无内部驻点
    for i in range(r, k):
        if abs(gu[i]) > 1e-9:
            return

    def w_of(lam):
        u = [0.0] * k
        for i in range(r):
            denom = lam * diag[i]
            if abs(denom) < 1e-300:
                return None
            u[i] = -gu[i] / denom
        # w = L^{-T} u
        w = [sum(Lti[i][j] * u[j] for j in range(k)) for i in range(k)]
        return w

    def psi(lam):
        w = w_of(lam)
        if w is None:
            return None
        return sum(Ck[i][j] * w[i] * w[j] for i in range(k) for j in range(k)) / 2

    if radius <= FEAS_TOL:
        # 半径为 0：唯一可行点就是中心
        yield 0.0, t0
        return

    # psi(lambda) 在 lambda>0 上从 +inf（若有 gu!=0）单调降到 0
    lo_l = 1e-12
    psi_lo = psi(lo_l)
    if psi_lo is None:
        return
    if psi_lo <= radius:
        # lambda 极小就已在球内：无约束线性驻点在球内（情形 A 处理），
        # 这里仍给出 lambda->0 边界
        w = w_of(lo_l)
        if w is not None:
            yield lo_l, [t0[i] + w[i] for i in range(k)]
        return
    hi_l = 1.0
    for _ in range(120):
        if psi(hi_l) is not None and psi(hi_l) <= radius:
            break
        hi_l *= 2.0
    else:
        return
    for _ in range(90):
        mid = (lo_l + hi_l) / 2
        pm = psi(mid)
        if pm is None:
            break
        if abs(pm - radius) <= ROOT_TOL:
            lo_l = hi_l = mid
            break
        if pm > radius:
            lo_l = mid
        else:
            hi_l = mid
    lam = (lo_l + hi_l) / 2
    w = w_of(lam)
    if w is not None:
        yield lam, [t0[i] + w[i] for i in range(k)]
