"""联合一致校正（reconciliation）分析包。

公开接口见 :mod:`analysis_data.reconcile.api`。本包只依赖 Python 标准库，
全部代数以 :class:`fractions.Fraction` 精确执行，避免浮点把奇异误差结构
（零特征方向＝精确约束）悄悄掩盖掉。
"""
