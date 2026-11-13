"""联合一致校正的性质测试。

覆盖分析口径要求的关键不变量：

* 点校正满足全部精确约束、底层界、W 零空间正交；
* 删除完全重复的汇总预测及其同一随机误差不改变结论；
* 随机打乱节点顺序只改变输出顺序、不改变数值结论；
* 逐行回代所有登记 S 关系；
* 独立小规模解法（不同算法）核对目标值与置信集合边界；
* rho 小于最优残差时置信集合为空，不放宽；
* 精确约束与界冲突时只输出冲突证据、不发布校正值；
* 负的净转出、缺季度（null 不补零）、奇异 W、二级平局。
"""
from __future__ import annotations

import copy
import unittest
from fractions import Fraction

from analysis_data.io import load
from analysis_data.reconcile import _linalg, verify
from analysis_data.reconcile._numeric import dot
from analysis_data.reconcile.api import materialize_dict, reconcile_dict, reconcile_file
from analysis_data.reconcile.engine import _weighted_forms


def _load(path):
    v, _ = load(path)
    return v


class PointCorrectionTests(unittest.TestCase):
    def test_normal_point_satisfies_all_contracts(self):
        r = reconcile_file("data/normal.json")
        p = r.problem
        self.assertEqual(r.status, "published")
        # 精确分数，不是浮点
        self.assertTrue(all(isinstance(x, Fraction) for x in r.point))
        # 界
        self.assertTrue(all(r.point[j] >= p.lower[j] for j in range(p.m)))
        # 残差与 W 零空间正交（满秩 W 时零空间为空，自然成立）
        self.assertEqual(r.w_nullity, 0)
        # 逐行回代
        checks = verify.verify_point(p, r)
        self.assertTrue(all(v is True for v in checks.values()))

    def test_net_transfer_negative_and_missing_quarter(self):
        r = reconcile_file("data/net-transfer-missing-quarter.json")
        p = r.problem
        self.assertEqual(r.status, "published")
        # 两个缺季度被显式删除而非补零
        self.assertEqual(set(p.missing_node_ids), {"a:q2", "b:year"})
        self.assertEqual(len(p.observed_node_ids), p.n_obs)
        # 净转出项允许为负且确实取到负值
        idx = p.bottom_ids.index("x:net-transfer")
        self.assertIn("x:net-transfer", p.net_transfer)
        self.assertLess(r.point[idx], 0)
        # 非净转出项保持非负
        for j, bid in enumerate(p.bottom_ids):
            if bid not in p.net_transfer:
                self.assertGreaterEqual(r.point[j], 0)
        checks = verify.verify_point(p, r)
        self.assertTrue(all(v is True for v in checks.values()))

    def test_singular_conflict_blocks_publication(self):
        r = reconcile_file("data/inconsistent-exact-total.json")
        self.assertEqual(r.status, "conflict")
        self.assertIsNone(r.point)
        self.assertIsNotNone(r.conflict)
        # 冲突证据：完全相关的两个预测 26 vs 27 给出 0=1
        self.assertEqual(r.exact_rhs, [Fraction(1)])
        self.assertTrue(all(x == 0 for row in r.exact_constraints for x in row))
        # 冲突时不发布置信边界
        self.assertEqual(r.query_bounds, [])


class DuplicateInvarianceTests(unittest.TestCase):
    def test_removing_duplicate_total_preserves_conclusion(self):
        base = reconcile_file("data/normal.json")
        dup = reconcile_file("data/duplicate-correlated-total.json")
        # 重复案例 W 奇异，多出一个零方向
        self.assertEqual(dup.w_nullity, 1)
        self.assertEqual(base.w_nullity, 0)
        # 但精确约束是平凡的 0=0，点校正与残差完全一致
        self.assertEqual(dup.point, base.point)
        self.assertEqual(dup.optimal_residual, base.optimal_residual)
        # 置信边界数值一致
        self.assertEqual(len(dup.query_bounds), len(base.query_bounds))
        for a, b in zip(dup.query_bounds, base.query_bounds):
            self.assertAlmostEqual(a.lower, b.lower, places=9)
            self.assertAlmostEqual(a.upper, b.upper, places=9)

    def test_manual_duplicate_removal_matches(self):
        v = _load("data/duplicate-correlated-total.json")
        # 手工删除复制节点及其预测、协方差行列
        keep = [i for i, n in enumerate(v["nodes"]) if n["id"] != "all:year-copy"]
        v2 = copy.deepcopy(v)
        v2["nodes"] = [v["nodes"][i] for i in keep]
        v2["predictions"] = [v["predictions"][i] for i in keep]
        v2["covariance"] = [
            [v["covariance"][i][j] for j in keep] for i in keep
        ]
        r_dup = reconcile_dict(v)
        r_removed = reconcile_dict(v2)
        self.assertEqual(r_dup.point, r_removed.point)
        self.assertEqual(r_dup.optimal_residual, r_removed.optimal_residual)


class PermutationInvarianceTests(unittest.TestCase):
    def test_node_order_shuffle_only_changes_order(self):
        v = _load("data/normal.json")
        n = len(v["nodes"])
        perm = [3, 0, 4, 1, 2]  # 固定随机种子的乱序
        self.assertEqual(sorted(perm), list(range(n)))
        vv = copy.deepcopy(v)
        vv["nodes"] = [v["nodes"][i] for i in perm]
        vv["predictions"] = [v["predictions"][i] for i in perm]
        vv["covariance"] = [
            [v["covariance"][i][j] for j in perm] for i in perm
        ]
        r1 = reconcile_dict(v)
        r2 = reconcile_dict(vv)
        # 底层身份顺序未变，点校正完全相同
        self.assertEqual(r1.point, r2.point)
        self.assertEqual(r1.optimal_residual, r2.optimal_residual)
        # 观测节点集合一致（仅顺序不同）
        self.assertEqual(
            set(r1.problem.observed_node_ids),
            set(r2.problem.observed_node_ids),
        )
        for a, b in zip(r1.query_bounds, r2.query_bounds):
            self.assertAlmostEqual(a.lower, b.lower, places=9)
            self.assertAlmostEqual(a.upper, b.upper, places=9)


class EmptySetTests(unittest.TestCase):
    def test_rho_below_optimal_residual_is_empty(self):
        r = reconcile_file("data/normal.json")
        self.assertTrue(r.optimal_residual > 0)
        v = _load("data/normal.json")
        v["rho"] = "0"  # 小于正的最优残差
        r0 = reconcile_dict(v)
        self.assertTrue(r0.set_empty)
        self.assertEqual(r0.status, "empty_set")
        # 点校正仍发布（它不依赖 rho），但边界为空
        self.assertIsNotNone(r0.point)
        self.assertEqual(r0.query_bounds, [])

    def test_rho_equal_optimal_is_nonempty(self):
        v = _load("data/normal.json")
        r = reconcile_dict(v)
        v["rho"] = str(r.optimal_residual)
        r2 = reconcile_dict(v)
        self.assertFalse(r2.set_empty)
        self.assertTrue(r2.query_bounds)


class TieBreakTests(unittest.TestCase):
    def test_second_stage_pulls_unobserved_bottom_to_reference(self):
        r = reconcile_file("data/singular-tie-break.json")
        p = r.problem
        # a:q2 无任何观测，是零费用自由方向
        self.assertIn("a:q2", p.missing_node_ids)
        self.assertTrue(r.stage2_used)
        # 二级目标把它放到冻结参考 b0（距离平方为 0）
        self.assertEqual(r.point, p.b0)
        self.assertEqual(r.stage2_objective, 0)
        # 未识别方向被登记
        self.assertTrue(r.unidentified_directions)

    def test_pseudoinverse_satisfies_penrose(self):
        for path in (
            "data/normal.json",
            "data/duplicate-correlated-total.json",
            "data/net-transfer-missing-quarter.json",
        ):
            v = _load(path)
            p = materialize_dict(v)
            wp, *_ = _weighted_forms(p)
            from analysis_data.reconcile._numeric import matmul
            self.assertEqual(matmul(matmul(p.w, wp), p.w), p.w, path)
            self.assertEqual(matmul(matmul(wp, p.w), wp), wp, path)


class IndependentCrossCheckTests(unittest.TestCase):
    def test_independent_solver_confirms_bounds(self):
        for path in (
            "data/normal.json",
            "data/duplicate-correlated-total.json",
            "data/singular-tie-break.json",
        ):
            with self.subTest(file=path):
                v, _ = load(path)
                p = materialize_dict(v)
                r = reconcile_dict(v)
                report = verify.cross_check_bounds(p, r)
                for qid, d in report.items():
                    self.assertTrue(d["scan_respects_lower"], (path, qid))
                    self.assertTrue(d["scan_respects_upper"], (path, qid))

    def test_backsubstitution_every_registered_row(self):
        r = reconcile_file("data/net-transfer-missing-quarter.json")
        bs = r.backsub
        self.assertTrue(bs["all_exact_constraints_hold"])
        self.assertTrue(bs["within_bounds"])
        self.assertTrue(bs["nullspace_residual_zero"])
        # 每个登记节点（含缺测）都有回代拟合值
        self.assertEqual(len(bs["rows"]), len(r.problem.all_node_ids))


class CovarianceDiagnosticsTests(unittest.TestCase):
    def test_rank_eigenvalues_and_tolerance_reported(self):
        r = reconcile_file("data/duplicate-correlated-total.json")
        self.assertEqual(r.w_rank, r.problem.n_obs - r.w_nullity)
        self.assertEqual(len(r.eigenvalues), r.problem.n_obs)
        # 半正定：浮点特征值不含显著负值
        self.assertTrue(all(e > -1e-9 for e in r.eigenvalues))
        self.assertEqual(r.rank_tolerance, Fraction(1, 10 ** 10))
        # 零方向精确登记
        self.assertEqual(len(r.null_directions), r.w_nullity)

    def test_non_psd_rejected(self):
        v = _load("data/normal.json")
        v["covariance"][0][1] = "5"
        v["covariance"][1][0] = "5"  # 制造不定矩阵
        with self.assertRaises(ValueError):
            reconcile_dict(v)


if __name__ == "__main__":
    unittest.main()
