# 策略蒸馏

教师在仿真中读取完整特权状态，学生读取固定长度部署观测历史。蒸馏目标由动作 MSE 和
特征 MSE 组成；可选 uncertainty weighting 为两项学习 log variance，自动调整相对尺度。
教师在 `DistillationTrainer` 中固定为 eval 且不计算梯度。

建议真实流程：

1. 在仿真任务中训练并冻结 privileged teacher；
2. 用同一 rollout 同步记录 teacher 状态与部署观测历史；
3. 先离线蒸馏，再按需做 student-in-the-loop 在线修正；
4. 按地形、载荷和扰动分层评估，避免只看总体 MSE。

`SyntheticDistillationDataset` 仅为 CI 与 smoke test 构造随机张量，不能代表机器人数据、
策略性能或 Sim2Real 证据。
