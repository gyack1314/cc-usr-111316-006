"""查看合成分析资料，或运行联合一致校正并输出结果包。

子命令：

* ``inspect``：只展示资料身份与基本形状（不求解，保持原有行为）；
* ``reconcile``：运行完整联合一致校正，输出精确分数结果包
  （``--float`` 切换为人读浮点版）。
"""
import argparse
import json
from pathlib import Path

from .io import load
from .domain import summarize
from .reconcile.api import reconcile_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["inspect", "reconcile"])
    p.add_argument("file")
    p.add_argument("--float", action="store_true", help="结果包改用浮点（默认精确分数字符串）")
    a = p.parse_args()
    if a.command == "inspect":
        v, h = load(a.file)
        print(json.dumps(
            {"manifest_sha256": h, "summary": summarize(v, Path(a.file).parent)},
            ensure_ascii=False, indent=2,
        ))
    else:
        result = reconcile_file(a.file)
        print(json.dumps(result.to_dict(exact=not a.float), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
