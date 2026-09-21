# 最小证伪实验与 5090 执行规格

## 1. 要回答的唯一主问题

在完全相同的 CityFlow 状态、MaxPressure 提案、候选集合和搜索预算下，显式加入相位—movement 动作语义、邻居联合动作和多步目标，能否提高候选动作的真实排序；经过校准的保守门控能否把这种排序改善转化为 Jinan、Hangzhou 独立需求上的控制收益，同时限制最差 flow 风险？

本轮不把“训练 loss 更低”或“单个 flow 略好”视为成功。实验按 E0–E4 顺序执行；前一阶段达到继续条件后才进入下一阶段，但这些是研究死亡条件，不是需要用户逐次确认的 gate。

## 2. 固定数据协议

### 2.1 独立性的统计单位

- 统计单位是一个独立 traffic-demand flow 文件，不是训练 seed 或评估 seed。
- Jinan 与 Hangzhou 分别建立 `train/calibration/test` 三个互斥集合；按 flow 文件内容 SHA256 去重，同一内容不得跨集合。
- 最低数据量为每城市 6 个训练 flow、2 个校准 flow、4 个最终测试 flow。若现有仓库达不到这个数量，先补充或生成具有不同时间需求曲线、方向比例和峰值强度的 demand；在满足前，任何闭环结果只能标记为诊断结果。
- 数据划分在训练前冻结为 `data_manifest.json`。它记录城市、roadnet、flow 绝对源路径、内容 SHA256、时长、车辆数、生成来源和 split；最终测试 flow 在模型和阈值冻结前不得用于调参。
- 每个配置使用 5 个模型初始化 seed。统计时先在同一个 flow 内对 seed 求均值，再以 flow 为单位做配对推断，禁止把 seed 当作额外独立样本。

### 2.2 轨迹采集

训练数据由 Random、FixedTime、MaxPressure 和 SharedDQN 的混合行为策略产生，并在 manifest 中记录每条 episode 的来源策略。Random 提供动作覆盖，另外三类保持现实状态分布。各模型变体使用完全相同的训练轨迹和 split。

每个 transition 至少保留：

- `observation_t`、`joint_action_t`、`reward_t`、`observation_t+1`、`done/truncated`；
- `phase_movement_mask`、movement/road 有向连接、邻居索引和有效 action mask；
- city、roadnet、flow、episode、step、行为策略、采集 seed；
- 原始量纲的 queue、vehicle count、pressure、travel-time proxy，以及归一化统计量的训练集来源。

窗口样本长度至少为 `H+L=13`：候选控制 horizon `H=3`，之后用 MaxPressure 延续 `L=10` 步构造固定的终端目标。跨 episode 的窗口不得拼接。

## 3. 固定候选与 CityFlow 反事实真值

### 3.1 候选集合

每个决策状态先产生 MaxPressure 联合动作，然后用当前确定性局部变异器产生总计 64 条、长度为 3 的联合动作序列。第 0 条始终是重复滚动的 MaxPressure 提案。候选生成器版本、顺序和候选张量 SHA256 必须记录；所有模型、消融和 baseline 复用同一个候选张量，不得因为方法不同增加搜索预算。

### 3.2 可重放的反事实标签

对每个选定诊断状态，在独立 CityFlow engine 中从 episode 起点重放相同动作历史，确认重放状态摘要与原轨迹一致后，再分别执行 64 个候选。真实候选回报定义为：

`G_CF = sum(t=0..H-1) gamma^t r_t + gamma^H * sum(j=0..L-1) gamma^j r_(H+j)^MP`

其中 `gamma=0.99`，前 3 步执行候选序列，后 10 步由 MaxPressure 接管。训练 reward 的符号和聚合方式沿用当前环境契约。由该回报产生 CityFlow 真值排序、最优候选、相对 MaxPressure 的真实 advantage 和所选动作 regret。

CityFlow 若不能直接 snapshot/restore，不允许用近似状态手工重建代替。先执行确定性重放可行性检查：相同 roadnet、flow、seed 和动作历史必须在分支点得到一致的 phase、lane vehicle count、queue、累计车辆统计及哈希。若不一致，E1 停止，改用从起点同步推进的平行 engine；仍不一致则先修复模拟器确定性契约，不生成排序结论。

为控制成本，E1 每城市先在校准集均匀抽取不少于 200 个状态，覆盖低/中/高拥堵和 episode 前/中/后段；最终 E3 在所有测试 flow 上每隔固定决策间隔采样，间隔在看测试结果前冻结。

## 4. 模型变体与接口草案

本研究阶段不修改现有公共 API。实现阶段在现有 `ObservationBatch`、trajectory、checkpoint 和 planner 周围做版本化扩展，并保留旧 checkpoint 的只读加载路径。

### 4.1 动力学消融

| 编号 | 变体 | 唯一新增因素 |
|---|---|---|
| M0 | 当前 `GraphWorldModel` | 单步状态/奖励，普通 action embedding |
| M0-cap | 容量匹配对照 | 增大隐藏层，使参数量与 M2 相差不超过 5%，但不加入交通语义 |
| M1 | Movement-gated | 用所选相位对应的 `phase_movement_mask` 门控各 movement 的服务/状态更新 |
| M2 | Action-semantic graph | M1 加上沿有向 road/movement edge 传播的邻居所选动作消息 |
| M3 | Multi-step | M2 加 1–3 步 free-running 状态和奖励联合损失，时间权重固定并报告 |
| M4 | Terminal-value | M3 加 `V_MP(s_H)` 或等价的 MaxPressure continuation value head |
| M5 | Ensemble | 5 个 bootstrap M4；每个成员使用不同 episode bootstrap 和初始化 seed |

M1–M2 的接口至少需要 `phase_movement_mask [N,A,M]`、`selected_actions [B,N]`、有向 edge/index/mask 和 movement 方向映射。M3 的 batch 增加连续窗口 `[B,H+1,N,M,F]` 及 `[B,H,N]` 动作。M4 checkpoint 明确记录 value-target 协议。M5 的 planner 接收逐成员预测，而不是只接收已平均的一个标量。

### 4.2 规划与门控消融

所有规划器在相同的 64×3 候选张量上比较：

- P0：按预测均值直接 argmax；
- P1：`LCB = mean(advantage) - beta * calibrated_uncertainty`；
- P2：P1 加校准集确定的最小优势阈值 `tau`；
- P3：P2 加显式 fallback；不通过支持度、置信度或优势阈值时执行 MaxPressure。

`beta`、`tau`、支持度阈值只可在 calibration split 上确定。阈值选择目标为：在校准集约束“被接受候选的真实负 advantage 比例”和最坏 flow regret 后，最大化覆盖率；不得在测试集按最终旅行时间调阈值。

## 5. E0–E4 执行顺序与死亡条件

### E0：数据与重放契约

- 产出 flow-disjoint manifest、轨迹 schema 检查、归一化统计来源和重放一致性报告。
- 继续条件：无 split 哈希重叠；窗口不跨 episode；至少 99.9% 重放状态逐字段一致，任何不一致均有可解释的数值容差。
- 死亡条件：无法得到可靠分支真值。此时不得训练候选排序方法。

### E1：当前模型诊断

对 M0 计算 1–5 步 teacher-forced 与 free-running 状态/奖励误差、64 候选的 Spearman、Kendall、top-1 accuracy、top-k recall、所选动作 regret，以及“预测比 MaxPressure 好但真实更差”的比例。

- 继续条件：确认 M0 的失败主要出现在可测的 horizon、movement、拥堵或候选支持区间。
- 若 M0 排序已很高但控制仍差，优先检查 reward、相位时序、候选执行与评估契约，不进入大模型改造。

### E2：动作语义与多步模型

按 M0、M0-cap、M1、M2、M3、M4 顺序训练；训练预算、数据、early-stopping 规则和候选完全匹配。

- 主继续条件：M2 相对 M0-cap 在 Jinan、Hangzhou 校准 flow 上都提高 Spearman/Kendall，并降低真实 regret；M3/M4 至少不破坏该改善。
- G01 死亡条件：M1/M2 在任一城市都无排序改善，或改善被 M0-cap 复现。回退到真正改善的最小模块；若无模块改善，停止动作语义主路线。
- G02 死亡条件：M3/M4 只降低状态 MSE，不提高排序或 regret。回退为直接 H-step 回报/排序模型 I02，不再宣称多步生成模型必要。

### E3：不确定性与保守门控

训练 M5，在 calibration split 检验 ensemble disagreement、最近训练样本距离/密度和真实 absolute return error、真实 regret 的关系。报告可靠性曲线、分箱 ECE、Spearman、错误候选识别 AUROC/AUPRC 和 coverage-risk 曲线。

- 继续条件：至少一种未看测试集的分数能随误差或 regret 单调上升，并在两城市都优于随机识别。
- G03 死亡条件：uncertainty/support 与真实误差不相关，或 fallback 在固定覆盖率下不能改善最差 flow。此时 ensemble 不能被称为安全机制；回退为 I03 的纯支持度门控或只保留 MaxPressure。

### E4：最终闭环配对评估

冻结模型、候选生成器、`beta/tau`、checkpoint 和代码 commit 后，在 Jinan、Hangzhou test flows 上运行 FixedTime、MaxPressure、SharedDQN、M0+P0、M4+P0 和 M5+P3。正式论文再加入维护良好的 pressure-RL、图协调 RL 和直接 model-based TSC baseline，见 `baseline_boundary_matrix.csv`。

每个 flow/seed/控制器使用相同模拟输入并记录配对 run ID。报告平均旅行时间、平均 queue、平均 waiting time、throughput、切相次数、模型接受率、fallback 原因、相对 MaxPressure 的逐步 regret 和 wall-clock 决策延迟。

## 6. 统计与成功标准

- 对每个城市分别报告逐 flow 配对差值和均值；另给城市等权的 pooled 结果。
- 用 flow 为 block 做 10,000 次 paired bootstrap；seed 只在 block 内聚合。报告百分比改善的 95% CI、原始量纲差值和全部 flow 散点。
- 主成功标准：Jinan 和 Hangzhou 各自的 held-out 平均旅行时间均比 MaxPressure 至少改善 2%，且每城市配对 bootstrap 95% CI 均不跨 0。
- 保护指标：平均 queue、waiting time 不得反向恶化超过 1%；throughput 不得下降超过 0.5%；P3 在任一最差 test flow 的旅行时间退化不得超过 1%。同时报告 CI，不能只看点估计。
- 机制归因：M2 相对 M0-cap、M3 相对 M2、P3 相对 P0 的排序/regret 或风险改善必须在固定搜索预算下成立。候选排序改善若仅来自更多候选或更长 horizon，判为失败。
- 计算可行性：在线每次决策耗时应低于环境决策间隔的 20%；超出时报告吞吐瓶颈并用固定候选批处理优化，不能减少 baseline 的搜索预算来制造优势。

## 7. Baseline 的公平实现边界

最小性能链必须包含 FixedTime、MaxPressure、SharedDQN 和当前 World Model。正式投稿实验还需要：

- pressure-based 强 RL：PressLight 或维护良好的兼容实现；
- 图协调 RL：CoLight/UniLight 类方法，用于排除“只是加入图消息”的解释；
- 直接 model-based TSC：ModelLight/PLight/PRLight/ADAC 中至少一个协议可对齐的方法；
- 跨城市主张出现时才加入 X-Light/CrossLight，并采用其真正的跨场景协议。

所有 baseline 共用 roadnet、flow split、决策间隔、黄灯/全红、最小绿灯、reward 定义和指标计算。无许可证仓库只允许依据论文独立实现机制，不复制代码。若某方法无法在同一离散多路口动作契约下复现，应标为“不具可比实现”，不能用未经对齐的论文数字代替本地实验。

## 8. 5090 运行与产物契约

每次启动前先确认 `/mnt/pan` 存在且可写，并先报告绝对目录：

`/mnt/pan/world-model/runs/<UTC时间>_<source-sha>_<stage>/`

目录至少包含：

- `manifest/`：代码 SHA、dirty diff 摘要、依赖、GPU、CityFlow 版本、flow split 和命令；
- `data/`：本轮生成的轨迹、反事实标签和哈希索引；
- `checkpoints/`：模型、优化器、归一化统计、模型 schema 版本；
- `logs/`：stdout/stderr、逐 epoch 指标、资源和异常；
- `evaluation/`：逐 step、逐 episode、逐 flow 原始结果；
- `reports/`：误差曲线、排序/校准、bootstrap 和最终汇总。

若 `/mnt/pan` 不可用或不可写，停止，不回退到系统盘。launch 状态、训练诊断和最终指标分开记录；只有所有预注册 test flows 完成且原始文件可追溯时，才生成最终结果表。

## 9. 最终决策规则

- I01 进入完整论文实验：E2 排序证据、E3 校准证据和 E4 两城市性能门槛全部通过。
- 收缩为 I02：动作语义有效，但递归多步/terminal value 无排序收益。
- 收缩为 I03：主模型能排序，但 ensemble 不可校准；只保留经验证的数据支持度门控。
- 延后 I04：只有 I01 在两城市成立，且增加预训练不会改变目标城市数据量、搜索预算或测试协议时才研究跨城市预训练。
- 全部失败：保留 MaxPressure 和诊断工具，公开负结果边界，不用更大模型或更多候选掩盖核心失败。
