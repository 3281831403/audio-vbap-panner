# VBAP 矢量基幅度声像修复

## 题面

换 VBAP 声像后，上半球的声音跑到脚下、扫过扇区边界还会爆音：`speaker_vectors` 把仰角符号写反，上半球声像落到下半球；`find_triangle` 只取点积最大的三个扬声器而不是真正包含方向的凸包三角面，扇区边界漏判；`inverse_matrix` 的伴随矩阵转置顺序写错，行列式虽然还对但增益串位；`gains_for` 把负增益钳到 0 后没有重新归一，总能量小于 1；`gains_for` 用 L2 归一后没有再做能量守恒检查，方向还原出现偏差；`pan` 在扇区边界不做判定，两个三角形同时命中时随机取一个；`energy` 用增益和而不是平方和，非负增益下几乎永远不等于 1；`transition_weights` 用线性角插值，边界处权重和不等于 1；`pan_batch` 复用三角形缓存，第二个方向沿用第一个方向的解；`find_triangle` 不先归一化方向，非单位方向会选错扇区。难点是几何：三角包含判定、逆矩阵转置、钳负后归一，三处错一个声像就崩。vbap/panner.py 是核心，tests/test_panner.py 15个红。要修的是：上述十处几何、归一化与边界缺陷，直到 15 个测试全部通过。

## 运行方式

在本目录执行：

```bash
python -m pytest tests -q
```

初始状态必须恰好是 `15 failed`，且没有 collection error、import error 或 syntax error。修好 8 个桩函数后，15 个测试应全部通过。测试只依赖 Python 标准库 `math` 和 `pytest`，离线运行，不使用随机数。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 文件与约束

```text
audio-vbap-panner/
├── README.md
├── conftest.py
├── .gitignore
├── vbap/
│   ├── __init__.py
│   └── panner.py
└── tests/
    └── test_panner.py
```

- `vbap/panner.py` 中的 8 个公开函数目前全部 `raise NotImplementedError`，docstring 是接口契约。
- 角度单位为度；坐标约定为 `+x` 右、`+y` 前、`+z` 上；仰角为正表示上半球。
- 增益范围为 `0..1`，容差为 `1e-9`（方向还原夹角容差为 `1e-6`）。
- `tests/test_panner.py` 不可修改；不得通过改测试、跳过测试或放宽断言来得到绿色。
- 不得引入第三方依赖，不得联网，不得使用随机数。
