# 汇总映射及误差集合

nodes顺序对应预测y、W行列；节点可以是任意已登记底层线性和，不要求一棵树。有缺预测时显式从观测向量和W删对应行列，不能用null参与代数。bottom_ids确定b顺序，reference为二级目标b0。残差r=S*b-y。

对W=Q diag(lambda) Q'，正特征方向费用为r分量平方/lambda，零空间分量必须0；数值阈值须输出实际秩与残差，明显负特征值非法。主目标精确最小时才应用二级最近reference，不能用一个微小权重混合两个目标。

rho为联合加权残差平方上限而非半径；对每query分别优化。重复预测例子对复制的年度行使用同一误差完全相关，W因此奇异；两个不同预测值会违反精确关系。初始读取只检查维度及节点名，不解QP或置信集合。

## 联合一致校正口径（analysis_data.reconcile）

`reconcile_file`/`reconcile_dict`/`reconcile_problem` 在上述资料上做联合一致校正，结果包见 `ReconciliationResult.to_dict()`。

- **两层算术**。点校正、可行性、零空间、秩、KKT 全部以 `Fraction` 精确执行；只有置信集合 QCP 的 KKT 乘子 lambda 在一般有理输入下是代数无理数，用 float64 解特征方程，并把等式/界违反量与残差回代到精确约束写进 `*_numeric_evidence`，不把浮点伪装成精确分数。
- **奇异 W 不加岭**。LDL（容差 `rank_tolerance`，相对对角尺度）给秩与零空间；零方向 z 转成底层精确约束 `(z'S)b = z'y`。`z'S` 为零行时退化为 `0 = z'y`，是重复预测不一致的冲突证据，绝不丢弃。Moore–Penrose 逆由行空间基 `W^+ = B'(BWB')^{-1}B` 构造，满足 Penrose 条件。
- **词典序两阶段**。先在精确约束与底层界上最小化 `r'W^+r`；仅当约化 Hessian（RREF 参数化到 `ker E` 后）出现零曲率可行方向、即阶段一多解时，才在最优面上最小化到 `reference` 的欧氏距离。两目标不加权混合，不用伪逆最小范数解冒充业务解。
- **底层界与净转出**。默认下界非负；只有 `net_transfer` 名单内的底层允许负下界（资金净转出）。精确约束与界冲突时 `status="conflict"`，给 Farkas 型行溯源证书，不发布校正值。
- **置信集合**。`{b : E b=f, 界, r'W^+r <= rho}`，对每个 query 的 `c'b` 分别求 min/max，并保留取到边界的完整底层向量；单项边界不可相加当联合极值。`rho < 最优残差` 时 `status="empty_set"`，不取绝对值、不放宽。零费用且无界的查询方向返回 `unbounded_direction` 而非有限边界。
- **缺测**。预测为 null 的节点整行整列从 S、y、W 删除，不参与代数；其底层自由度进入 `unidentified_directions`。
- **独立核对**（`reconcile.verify`）。用与主求解器不同的算法（RREF 顶点枚举、球面射线二分、随机凸组合扫描、行空间基 W⁺）回代所有登记 S 关系并核对目标值与边界，测试套件据此交叉验证。

