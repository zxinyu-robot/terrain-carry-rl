# 迁移学习

`load_partial_checkpoint` 支持普通 state dict，以及包含 `model` 或 `state_dict` 的检查点。
它会报告：

- `loaded`：形状完全一致并加载的参数；
- `missing`：目标模型存在但检查点缺失的参数；
- `unexpected`：仅检查点中存在的参数；
- `shape_mismatch`：新旧形状；
- `partially_loaded`：按每个维度的相交切片复制过的参数。

相交复制适用于追加观测、扩展动作头等结构演进；新增区域保留目标模型初始化值。它不是
语义对齐的替代品：改变观测顺序时必须先显式重排权重。

`BackboneFreezeSchedule` 可先冻结已迁移 backbone，仅训练任务头，然后在指定 epoch
统一解冻。优化器应在冻结前包含全部参数，以便解冻后继续更新。
