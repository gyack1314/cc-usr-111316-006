# 气候资金预测的跨地域跨期一致校正

本项目保存气候行动研究的合成资料与领域口径。`analysis_data.io`读取JSON及核验原始文件身份，`analysis_data.domain`核对基本形状，`analysis_data.cli`展示数量。详细量纲、时间、契约和统计口径见[资料说明](docs/data-contract.md)。地区、家庭、合同、价格及排放数据均为合成，不代表真实政策评价或投资建议。

Python 3.11以上，无第三方依赖。项目根目录可运行：

```sh
python -m unittest discover -s tests -v
python -m compileall -q analysis_data tests
python -m analysis_data.cli inspect data/normal.json
```

`inspect`只展示资料，不做拟合、求解或分析发布。`invalid-config.json`包含基本配置错误，其余清单在身份上完整，可能包含需领域分析判断的不可识别、无界、冲突或不适用情况。有理数以字符串保留，null不等于0，随机种子属于分析设计。

## 联合一致校正

`analysis_data.reconcile` 在冻结资料上做跨地区跨期的联合一致校正，纯标准库、零第三方依赖。点校正与可行性用有理数精确求解（奇异 W 的零空间方向是精确约束，不加岭），置信集合 QCP 的无理 KKT 乘子用 float64 并附精确回代证据。

```sh
python -m analysis_data.cli reconcile data/normal.json            # 精确分数结果包
python -m analysis_data.cli reconcile data/normal.json --float    # 人读浮点版
```

可编程接口：

```python
from analysis_data.reconcile.api import reconcile_file
from analysis_data.reconcile import verify

result = reconcile_file("data/normal.json")
result.to_dict()                          # 完整结果包（点校正、KKT、置信边界、回代核对）
problem = result.problem
verify.verify_point(problem, result)      # 独立算法逐行回代与最优性核对
verify.cross_check_bounds(problem, result)  # 顶点枚举+随机扫描核对置信边界
```

资料场景：`normal` 为基线；`duplicate-correlated-total` 含完全相关的重复年度行（W 奇异，删除重复不改变结论）；`inconsistent-exact-total` 的重复预测值矛盾，输出冲突证据而不发布；`net-transfer-missing-quarter` 展示负的净转出与 null 缺季度；`singular-tie-break` 展示未识别零费用方向上的二级最近参考平局。口径细节见[资料说明](docs/data-contract.md)。

分析测试：

```sh
python -m unittest discover -s tests -t .
```


Dockerfile固定Linux镜像；`docker build -t climate-analysis .`构建，`docker run --rm climate-analysis`显示摘要，`docker run --rm climate-analysis python -m unittest discover -s tests -v`运行已有读取检查。资料均可本地读取，运行不需要线上账户。
