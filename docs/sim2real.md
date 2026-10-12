# Sim2Real 工程准备

当前组件覆盖三类可复现扰动：

- `DomainRandomizer`：质量、质心、摩擦、motor strength 与观测噪声幅度；
- `ObservationNoise`、`ObservationActionLatency`：传感噪声及独立观测/动作延迟；
- `ActuatorLag`：一阶执行器响应和强度偏差。

所有随机入口均接受 seed。生产适配器还应记录每个 episode 的随机化样本，以支持失败复现。
范围需要依据系统辨识结果设置，而不是直接把默认值用于真机。

进入硬件前仍需完成：仿真资产与接触参数校准、控制频率和时间戳验证、关节/力矩/功率安全
限制、急停与通信超时、shadow mode、逐级载荷测试以及硬件回归指标。本仓库未执行 Isaac
Lab 仿真或真机试验，也不声称已经完成 Sim2Real。
