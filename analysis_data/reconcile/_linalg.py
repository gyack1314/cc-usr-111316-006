"""零依赖线性代数（精确有理数为主，浮点特征诊断为辅）。

设计原则
========
* 结构判定（秩、零空间、线性依赖、KKT 求解）全部用 :class:`Fraction`
  精确完成，不依赖浮点阈值，因此奇异 ``W`` 的零空间方向是精确约束，
  不会被一个小岭项掩盖。
* 唯一的浮点例程是 :func:`jacobi_eigen`，仅用于把 ``W = Q diag(lambda) Q'``
  的特征值/特征向量写进结果包做诊断；半正定性与秩的判定仍以精确 LDL 为准。
"""
from __future__ import annotations

import math
from fractions import Fraction

from ._numeric import identity, transpose, zeros

__all__ = [
    "rref",
    "rank",
    "nullspace",
    "row_basis",
    "ldl_psd",
    "jacobi_eigen",
    "solve_full",
    "projection_matrix",
]


# --------------------------------------------------------------------------- #
# 精确高斯消元
# --------------------------------------------------------------------------- #
def rref(a: list[list[Fraction]]) -> tuple[list[list[Fraction]], list[int]]:
    """行最简形（Gauss–Jordan），返回 (R, 主元列)。"""
    m = [row[:] for row in a]
    if not m:
        return [], []
    rows, cols = len(m), len(m[0])
    pivots: list[int] = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        pivot = next((i for i in range(r, rows) if m[i][c] != 0), None)
        if pivot is None:
            continue
        m[r], m[pivot] = m[pivot], m[r]
        pv = m[r][c]
        m[r] = [x / pv for x in m[r]]
        for i in range(rows):
            if i != r and m[i][c] != 0:
                f = m[i][c]
                m[i] = [x - f * y for x, y in zip(m[i], m[r])]
        pivots.append(c)
        r += 1
    return m, pivots


def rank(a: list[list[Fraction]]) -> int:
    return len(rref(a)[1])


def nullspace(a: list[list[Fraction]]) -> list[list[Fraction]]:
    """返回右零空间 ``{z : a z = 0}`` 的一组基（列向量列表）。"""
    if not a:
        n = 0
    else:
        n = len(a[0])
    r_mat, pivots = rref(a)
    free = [c for c in range(n) if c not in pivots]
    basis: list[list[Fraction]] = []
    for f in free:
        z = [Fraction(0) for _ in range(n)]
        z[f] = Fraction(1)
        for row, pc in zip(r_mat, pivots):
            z[pc] = -row[f]
        basis.append(z)
    return basis


def row_basis(a: list[list[Fraction]]) -> list[list[Fraction]]:
    """返回行空间的一组基（非零 RREF 行）。"""
    r_mat, pivots = rref(a)
    return [r_mat[i] for i in range(len(pivots))]


# --------------------------------------------------------------------------- #
# 半正定 LDL（带秩容差，精确分数）
# --------------------------------------------------------------------------- #
def ldl_psd(
    w: list[list[Fraction]], tol: Fraction
) -> dict:
    """对称矩阵的秩揭示 LDL 分解（对称选主元）。

    返回::

        {
          "rank": int,
          "pivot": [行列置换顺序],
          "ld": 下三角 L（含单位对角）,
          "d":    对角元列表（精确值）,
          "positive": 被保留的正主元（精确值）,
          "dropped":  被容差剔除的小主元（精确值）,
          "negative": 明显负主元（若非空则 W 非法）,
          "null_basis": W 零空间基（按原坐标，列向量列表）,
          "tolerance": tol,
        }

    ``tol`` 按主元的相对尺度使用：主元 ``d`` 满足
    ``|d| <= tol * max_j |W_jj|`` 视为数值零。明显为负（超过容差）则报错信号。
    """
    n = len(w)
    scale = max((abs(w[i][i]) for i in range(n)), default=Fraction(0))
    cutoff = tol * scale if scale > 0 else tol
    perm = list(range(n))
    work = [row[:] for row in w]
    l = identity(n)
    d: list[Fraction] = []
    positive: list[Fraction] = []
    dropped: list[Fraction] = []
    negative: list[Fraction] = []
    null_rows: list[list[Fraction]] = []

    for k in range(n):
        # 在剩余子矩阵里选绝对值最大的对角主元
        j = max(range(k, n), key=lambda i: abs(work[i][i]))
        if abs(work[j][j]) <= cutoff:
            # 剩余对角全在容差内；检查非对角是否也真正归零
            if any(work[i][h] != 0 for i in range(k, n) for h in range(k, n)):
                # 容差内非零但对角为零：对称不定的信号，按负主元处理
                negative.append(work[j][j])
            break
        if work[j][j] < 0:
            negative.append(work[j][j])
            break
        if j != k:
            work[k], work[j] = work[j], work[k]
            for i in range(n):
                work[i][k], work[i][j] = work[i][j], work[i][k]
            perm[k], perm[j] = perm[j], perm[k]
            for i in range(k):
                l[i][k], l[i][j] = l[i][j], l[i][k]
        pivot = work[k][k]
        d.append(pivot)
        positive.append(pivot)
        for i in range(k + 1, n):
            l[i][k] = work[i][k] / pivot
        for i in range(k + 1, n):
            for h in range(k + 1, n):
                work[i][h] -= l[i][k] * pivot * l[h][k]
            work[i][k] = Fraction(0)
        for i in range(k + 1, n):
            work[k][i] = Fraction(0)

    r = len(d)
    # 数值零主元（容差内但非严格 0）单独登记，保留精确值
    for i in range(r, n):
        v = work[i][i]
        if v != 0:
            dropped.append(v)

    # 精确零空间（严格代数），按原始坐标返回
    null_basis = nullspace(w)

    return {
        "rank": r,
        "pivot": perm,
        "ld": l,
        "d": d,
        "positive": positive,
        "dropped": dropped,
        "negative": negative,
        "null_basis": null_basis,
        "tolerance": tol,
        "cutoff": cutoff,
        "scale": scale,
    }


# --------------------------------------------------------------------------- #
# 浮点 Jacobi 特征分解（仅诊断用）
# --------------------------------------------------------------------------- #
def jacobi_eigen(
    w: list[list[Fraction]], sweeps: int = 100, tol: float = 1e-14
) -> tuple[list[float], list[list[float]]]:
    """对称矩阵的循环 Jacobi 特征分解，返回 (特征值, 各行是特征向量)。

    只用标准库 :mod:`math`，规模小（预测节点数），足够给出
    ``W = Q diag(lambda) Q'`` 的诊断量；业务判定不依赖这里的浮点结果。
    """
    n = len(w)
    a = [[float(w[i][j]) for j in range(n)] for i in range(n)]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(sweeps):
        off = math.sqrt(
            sum(2.0 * a[i][j] ** 2 for i in range(n) for j in range(i + 1, n))
        )
        if off <= tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) <= tol:
                    continue
                tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = (1.0 if tau >= 0 else -1.0) / (
                    abs(tau) + math.sqrt(1.0 + tau * tau)
                )
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = t * c
                # 旋转 A
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    eig = [a[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: eig[i], reverse=True)
    eig = [eig[i] for i in order]
    q = [[v[row][i] for i in order] for row in range(n)]
    return eig, q


# --------------------------------------------------------------------------- #
# 线性方程组与投影
# --------------------------------------------------------------------------- #
def solve_full(
    a: list[list[Fraction]], b: list[Fraction]
) -> tuple[list[Fraction] | None, list[list[Fraction]]]:
    """解 ``a x = b``。

    返回 ``(特解或 None（无解）, 齐次零空间基)``。零空间非空表示解不唯一，
    调用方必须自行施加二级准则，不能把任意最小范数解当成业务解。
    """
    if not a:
        return [Fraction(0) for _ in b], []
    n = len(a[0])
    aug = [row[:] + [rhs] for row, rhs in zip(a, b)]
    r_mat, pivots = rref(aug)
    # 矛盾行：系数全 0 而右端非 0
    for row in r_mat:
        if all(x == 0 for x in row[:-1]) and row[-1] != 0:
            return None, []
    x = [Fraction(0) for _ in range(n)]
    for row, pc in zip(r_mat, pivots):
        x[pc] = row[-1]
    return x, nullspace([row[:-1] for row in r_mat])


def projection_matrix(basis: list[list[Fraction]], n: int) -> list[list[Fraction]]:
    """把列基 ``basis`` 张成子空间的正交投影矩阵 P = B (B'B)^-1 B'。"""
    if not basis:
        return zeros(n, n)
    b_mat = transpose(basis)  # n x k
    bt = transpose(b_mat)
    gram = [[sum(bt[i][t] * b_mat[t][j] for t in range(n)) for j in range(len(basis))]
            for i in range(len(basis))]
    rhs = identity(len(basis))
    inv_cols: list[list[Fraction]] = []
    for j in range(len(basis)):
        sol, null = solve_full(gram, [rhs[i][j] for i in range(len(basis))])
        if sol is None:  # pragma: no cover - 基线性无关时不会发生
            raise ValueError("投影基线性相关")
        inv_cols.append(sol)
    gram_inv = transpose(inv_cols)
    tmp = matmul(b_mat, gram_inv)
    return matmul(tmp, bt)
