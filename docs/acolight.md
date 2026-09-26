# A-CoLight：Jinan1 五种子实验

入口：`python -m cityflow_tsc.acolight_experiment train|evaluate|compare`。训练配置和统计方法沿用 [E-CoLight](ecolight.md)。

已对照作者固定提交 `d5d4180f34edb843e1d1b462d5846c75d6d4533a` 的 `run_advanced_colight.py`、`models/colight_agent.py`、`utils/config.py` 和 `utils/cityflow_env.py`，本地缓存的 Git blob hash 与该提交 tree 一致。

输入按顺序拼接当前相位8维、efficient queue pressure12维、停止线前167米运行车辆数12维，共32维。压力输入仍使用整条车道排队数；新增车辆特征满足 distance >= lane_length - 167 且 speed > 0.1。换道shadow ID按作者代码映射到原车ID读取位置/速度，各车道列表中的出现仍计数，不擅自去重。

网络为32维输入、共享MLP[32,32]、5头图注意力（每头32维）、四相位Q值；邻居含自身最多5个。奖励为进口排队总数乘−0.25。Adam lr=.001、epsilon=1e−7；γ=.8；TD奖励/20；回放12000、每轮最多3000样本、batch20；最多100拟合epochs、validation30%、patience10。五种子各100 rounds×3600秒，末十轮评估均值再算跨种子均值与样本标准差。

控制每30秒决策，切换时包含5秒黄灯和25秒新绿灯，全红0秒。与LLMLight论文文字30/3/2秒有差异；保留逐路口探索、实际回合边界next state和独立PyTorch随机数等移植边界，不宣称历史结果完全一致复现。laneChange=true；LLMTSCS口径ATT与引擎ATT分别报告。

首先对照LLMLight表2 Advanced-CoLight，Jinan1参考274.67秒；另列FuzzyLight无噪声表2、Traffic-R1表2零样本/附录表5目标训练，注明预算、时序和训练域差异。参考值来自本地已保存论文表格，不是本项目实测。
