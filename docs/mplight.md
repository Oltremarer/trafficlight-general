# MPLight：Jinan1 五种子实验

使用 `python -m cityflow_tsc.mplight_experiment train|evaluate|compare`；输入、输出与种子等命令参数见 [CoLight 运行说明](colight.md)。这是 MPLight 的专用配置和 FRAP 网络，环境、指标及固定 TD target 的拟合循环复用已有代码。

参考源为 LLMTSCS 提交 `d5d4180f34edb843e1d1b462d5846c75d6d4533a`：`run_mplight.py`、`models/mplight_agent.py`、`models/network_agent.py`、`utils/updater.py`、`utils/config.py`。缓存源码的 Git blob SHA1 已与该提交的 tree.json 核对一致。源入口虽用 `EfficientMPLight` 作内部 dispatch 名，实际 `MODEL=MPLight`，状态是车辆数压力，并不是 efficient queue pressure。

| 项目 | 本次配置 |
|---|---|
| 场景和预算 | Jinan1；种子 0–4；各 100 rounds × 3600 秒；每轮训练后冻结评估 |
| 输入/奖励 | 当前相位8维＋进口/出口车辆数压力12维；奖励为 −0.25 × 进口与出口排队总数差的绝对值，区间 pre-state 均值 |
| FRAP | 压力和相位分别嵌入4维；车道嵌入16维；相位需求求和；有序相位对竞争；中间卷积宽度20；四相位 Q 输出 |
| 初始化 | Linear 对应 Dense/1×1 Conv：GlorotUniform、bias=0；Embedding：Uniform[-0.05,0.05] |
| 优化 | Adam lr=0.001，epsilon=1e−8，γ=0.8，TD奖励/20；batch=20 |
| 拟合 | 固定全 Q-vector targets、MSE；最多100 epochs；验证末30%，不 shuffle，patience10，不恢复最优epoch |
| 探索/目标网 | 每次全网一次探索硬币，ε=max(0.8×0.95^round,0.2)；评估ε=0；目标网滞后5轮 |
| 控制 | 30秒决策包含切换黄灯5秒，全红0秒；步长1秒；换道开启 |
| 汇总 | 各种子末10轮评估均值，再计算5个种子的均值和样本标准差 |

## 回放的两级抽样

每路口保留最后12000条时间记录，第一次抽取最多3000个时间索引，所有路口共享这些索引。按路口顺序拼接后，再保留尾部12000条单路口样本，第二次抽取最多3000条用于拟合。实现用整网回放保存逐路口历史，但实际 minibatch 是单路口样本。

这保留了源代码的一个偏置：当每个路口已抽满3000条时，Jinan的12个路口只有顺序最后4个路口的样本进入第二次抽样。没有悄悄改成均衡抽样。本项目按路口编号自然排序，对应作者的 x 外循环、y 内循环；实际顺序写入 protocol.json。通用 FRAP 类中的单相位回退分支在本次四相位输出中不生效。

这仍是独立 PyTorch 实现，使用项目随机数及真实回合边界 next state，不复刻作者最后一条样本的3599秒 workaround，也不声称与其历史运行逐位一致。持续 Adam 状态与作者跨轮保存、恢复优化器的做法一致。

## 论文对照

首先对照 [LLMLight 表2](https://arxiv.org/html/2312.16044v5#S4.T2) 的 MPLight 行：Jinan1 ATT **307.82秒**。自动表还列 FuzzyLight 无噪声表2、Traffic-R1 主表2/附录表5和 Astra 表1的 MPLight 行，注明各自预算、迁移与时序差异。

作者代码30秒决策含5秒黄灯，与论文文字30/3/2秒仍有差异。两种 ATT、完成车辆记账及跨论文比较边界与 CoLight 相同。当前 MPLight 的交通成绩须等真实训练完成后才能给出。
