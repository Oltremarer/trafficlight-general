# E-CoLight：Jinan1 五种子实验

专用入口：`python -m cityflow_tsc.ecolight_experiment train|evaluate|compare`。运行参数和文件结构见 [CoLight 说明](colight.md)。

作者固定提交 `d5d4180f34edb843e1d1b462d5846c75d6d4533a` 的 `run_efficient_colight.py` 使用 `cur_phase + traffic_movement_pressure_queue_efficient + adjacency_matrix`，奖励为 `−0.25 × queue_length`。它与 CoLight 共用 `models/colight_agent.py` 图注意力网络和训练配置。

本次输入为8维当前相位编码与12维 efficient queue pressure。每条进口车道的压力等于本车道排队数减去目标出口道路所有车道排队数的均值；Jinan每条出口道路三车道，故减去出口排队总数/3。不是车辆总数压力，也不是只减实际 laneLink 对应的一条出口车道。奖励仍为进口排队总数的−0.25倍，与压力输入分开计算。

共享MLP `[32,32]`、5个注意力头、每头32维、含自身最多5个邻居、四相位输出；Adam lr=.001、epsilon=1e−7；γ=.8；TD奖励/20；整网回放12000、每轮最多抽3000；batch20，最多100拟合epochs，验证集30%，patience10。每个种子100 rounds×3600秒，每轮训练后冻结评估，末十轮均值再做跨种子统计。

保留本项目 CoLight 的独立 PyTorch 实现边界：实际回合边界next state、逐路口独立探索、项目随机数；作者使用全网一次探索硬币。控制30秒含切换黄灯5秒、全红0秒，与论文文字30/3/2秒有差异。换道开启；shadow只在真实车辆完成量记账中处理，不改变输入、奖励或原生ATT。

首先对照 [LLMLight 表2](https://arxiv.org/html/2312.16044v5#S4.T2) Efficient-CoLight 行，Jinan1参考ATT **277.11秒**。表中还列FuzzyLight无噪声表2与Traffic-R1主表2/附录表5，明确预算、时序与训练域差异。不得把尚在训练的成绩当作最终结果。
