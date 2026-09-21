# 交通信号 World Model 调研综合结论

## 结论先说

当前最值得做的主路线，不是把 DreamerV3 整体搬进 CityFlow，也不是先做跨城市大预训练，而是先验证一个更小、能被证伪的系统：**以 MaxPressure 为提案和安全锚点的动作语义图 World Model**。

它只在同一组 MaxPressure 邻域候选中做三件事：

1. 用 `phase_movement_mask` 明确告诉模型“这个相位到底放行哪些 movement”；
2. 让上游路口选择的动作通过有向道路/车流连接影响下游预测；
3. 用多步模型和 ensemble 给每个候选计算保守优势，优势不够大或不确定性过高时直接执行 MaxPressure。

工作标签是 **MaxPressure-Guarded Action-Semantic Graph World Model**。这个名称只是为了讨论方便，不是已经成立的创新点。ModelLight 已经用相位向量预测交通转移，UniLight 已经预测流量对邻居的影响，MaCAR 已经建模同步动作后的交通，PETS/MOPO/ADAC 已经分别研究 ensemble、不确定性惩罚和交通离线悲观模型。因此，真正可能形成论文贡献的是：在同一 CityFlow、同一候选集、同一搜索预算下，证明这套交通动作语义和保守决策契约确实提高了**候选排序**，并稳定超过 MaxPressure；不是把已有模块拼在一起就算创新。

## 当前结果说明了什么

第一轮 5090 结果只证明了端到端链路可运行。Jinan 的一个确定性 flow 上：

| 控制器 | 平均旅行时间 | 平均队列 | 平均等待 | 吞吐量 |
|---|---:|---:|---:|---:|
| MaxPressure | 327.8168 | 155.9242 | 89.3353 | 5691 |
| 当前 World Model | 329.6685 | 159.6867 | 91.4858 | 5679 |

World Model 相对 MaxPressure 的旅行时间差约为 `+0.56%`，队列约为 `+2.41%`，等待约为 `+2.41%`，吞吐量少 12 辆。它在 1440 次决策中改变了 MaxPressure 动作 909 次（63.1%）。训练/验证 loss 下降，只能说明模型拟合了监督目标；它没有证明模型能正确比较“如果现在换成另一个灯相位，未来会怎样”。

静态代码审计给出四个已确认事实：

- `phase_movement_mask` 被轨迹文件保存并检查，但没有进入 `GraphWorldModel` 的动作条件转移；当前动作只是一个 embedding，然后广播给所有 movement。
- 邻居状态在动作注入前已经聚合，因此上游路口此刻选择哪个相位，不能显式改变下游路口的预测流入。
- 训练目标只有单步状态/奖励 MSE，部署却递归滚动 3 步。
- 规划器没有 dynamics ensemble、数据支持度、terminal value、优势阈值或 MaxPressure fallback。

这些事实可以解释为什么“验证 loss 降了，但控制没变好”是合理现象，但还不能证明哪一项是因果根源。下一轮必须用消融实验而不是故事来判断。

## 文献真正提供了哪些可迁移机制

### 1. World Model 的价值不在“会预测”，而在“预测能帮助决策”

[TD-MPC2](https://openreview.net/forum?id=Oxh5CstDJU) 同时训练多步 latent consistency、reward、termination 和 value，并在规划末端加入 Q；这说明训练目标应与规划使用方式对齐。[DreamerV3](https://www.nature.com/articles/s41586-025-08744-2) 也把 world model、reward/continuation、actor 和 critic 联合起来，但它的整个 actor-critic/JAX 系统会把当前问题改成另一条研究线，不适合第一步整体移植。

[MBPO](https://proceedings.neurips.cc/paper/2019/hash/5faf461eff3099671ad63c6f3f094f7f-Abstract.html) 的关键提醒是：模型产生的数据越多、rollout 越长，偏差越可能累积，所以应从真实状态出发并限制模型使用。[GMAN](https://ojs.aaai.org/index.php/AAAI/article/view/5477) 则说明交通预测不一定要递归单步模型，直接预测多个未来时间点也可以减少误差传播。但 GMAN 没有控制动作，所以它只能提供结构灵感，不能当作交通灯 World Model baseline。

对当前项目的直接含义是：先画出 1–5 步 free-running 误差和候选排序相关性曲线。如果 horizon 从 1 增到 3 后排序已经崩掉，继续增加搜索或模型深度没有意义。

### 2. 红绿灯动作不是普通离散编号

[ModelLight](https://arxiv.org/abs/2111.08067) 的环境模型把动作转换为相位向量，再和车道车辆数一起预测下一状态和奖励。这正好指出当前动作 embedding 的语义缺口，但 ModelLight 是单路口、预印本，而且通过递归虚拟样本训练 meta-RL，不等于我们的方法。

[PressLight](https://doi.org/10.1145/3292500.3330949) 和 MaxPressure 一类方法的竞争力来自 movement/pressure 语义，而不是网络更复杂。[Diagnosing RL for TSC](https://arxiv.org/abs/1905.04716) 进一步提醒：更复杂的状态或奖励不一定更有效，理论上合适的简洁表示可能更强。因此第一项实验必须包含“参数量相同但不使用 movement mask”的容量对照，避免把普通增容误说成动作语义的作用。

### 3. 多路口动力学必须区分“邻居现在是什么状态”和“邻居准备做什么动作”

[UniLight](https://www.ijcai.org/proceedings/2022/535) 把本站的观测压缩成对邻居有用的相位/出流预测；[MaCAR](https://www.ijcai.org/proceedings/2020/345) 更直接地预测多智能体同步动作之后的交通和动作价值。它们共同说明：只聚合邻居当前状态不够，候选联合动作也应进入跨路口传播。

但这两个工作都不是“用保守 MPC 排序 MaxPressure 候选”。因此可验证的问题应写成：显式邻居动作能否提高下游 movement 的反事实预测和候选排序，而不是“第一次考虑多路口交互”。

### 4. 不确定性不是天然可靠，必须用 CityFlow 真值校准

[PETS](https://proceedings.neurips.cc/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html) 用概率 ensemble 和 trajectory sampling 区分模型的不确定性；[MOPO](https://proceedings.neurips.cc/paper/2020/hash/a322852ce0df73e204b7e67cbbef0d0a-Abstract.html) 用 ensemble 方差惩罚 OOD rollout；[MOReL](https://proceedings.neurips.cc/paper/2020/hash/f7efa4f864ae9b88d43527f4b14f750f-Abstract.html) 把未知状态动作送入负奖励吸收态；交通领域的 [ADAC](https://doi.org/10.1145/3580305.3599459) 按离线数据中的邻居距离做悲观奖励。

另一方面，[COMBO](https://proceedings.neurips.cc/paper/2021/hash/f29a179746902e331572c483c45e5086-Abstract.html) 的出发点就是深度模型的不确定性可能不准，所以它把保守性放进 Q 值，而不是完全依赖 ensemble 方差。这意味着“加五个模型取方差”不是自动的安全方案。必须先检验 ensemble disagreement 或数据支持度是否随真实误差、真实 regret 单调增加；如果不校准，就不能把它用于 fallback。

### 5. 跨城市预训练是第二阶段，不是当前最短路径

[TrajWorld](https://proceedings.mlr.press/v267/yin25f.html) 在 80 个异构环境、超过一百万条轨迹上预训练可处理不同传感器/动作变量的 World Model。[X-Light](https://www.ijcai.org/proceedings/2024/11) 和 [CrossLight](https://doi.org/10.1145/3637528.3671927) 已经分别研究跨城市零样本和离线到在线 TSC 转移；[UniST](https://doi.org/10.1145/3637528.3671662) 也证明了跨场景时空预测预训练的价值。

这些工作让“跨城市”变成强 baseline 边界，而不是空白。当前模型连一个城市里的候选排序都尚未证明可靠，直接预训练会同时改变数据量、模型容量、路网 schema 和优化过程，很难判断性能来自哪里。因此它被降为延后方向：只有主方法在 Jinan、Hangzhou 内都通过后，才做等目标城市数据量、等计算量的预训练实验。

## 推荐的实际架构和数据流

训练阶段：

`CityFlow 独立 flow 轨迹 → 连续 H 步样本 → phase/movement/edge 语义编码 → 多个 bootstrap 图动力学模型 → H 步状态、奖励、terminal value 联合损失 → flow-disjoint 校准集上拟合不确定性和优势阈值`

决策阶段：

`当前状态 → MaxPressure 基准动作 → 固定的局部候选序列 → ensemble 预测每个候选 → 均值优势 − β×不确定性 − λ×OOD → 与阈值比较 → 通过则执行候选首动作，否则执行 MaxPressure`

核心不是让 World Model 自由搜索整个联合动作空间，而是让它在一个受控、可复现的 MaxPressure 邻域里证明自己确实比基准多看了一步。候选集合和搜索预算在所有消融中必须完全一致。

## 主路线的内部优先级

| 顺序 | 机制 | 为什么先/后做 | 继续条件 |
|---:|---|---|---|
| 1 | `phase_movement_mask` 动作门控 | 直接修复已确认的动作语义缺口，改动最小 | movement 误差和候选排序均优于容量对照 |
| 2 | 邻居联合动作消息 | 修复上游动作不能影响下游预测 | 下游反事实预测和两城市排序均改善 |
| 3 | H 步联合损失 | 对齐训练与 horizon=3 部署 | free-running 误差下降且排序不退化 |
| 4 | terminal value | 避免短 horizon 截断 | 在固定 horizon 下提高排序，而非只提高拟合 |
| 5 | ensemble＋支持度校准 | 识别模型利用和 OOD 候选 | 不确定性对误差/regret 有单调校准 |
| 6 | 优势阈值＋MaxPressure fallback | 控制最差 flow 风险 | 最差 flow 退化不超过 1%，仍保留足够覆盖率 |

如果第 1–2 步不能提高真实候选排序，主路线应立即缩小或停止；不应靠第 5–6 步把一个没有预测能力的模型包装成“安全方法”。

## 顶会故事可能在哪里，当前还缺什么

能支撑顶会的故事不能只是“World Model 用在交通灯”。现有直接工作已经覆盖环境模型、图通信、多步想象、悲观离线学习和跨城市转移。更可信的论文问题是：**交通信号的动作语义和多路口因果传播，是否能把 World Model 的预测精度转化为可靠的候选排序；一个以 MaxPressure 为锚的保守决策契约，是否能同时获得平均改善和最坏需求保护？**

这个故事成立仍需要四类证据：

1. 状态/奖励误差以外的候选排序和 regret 诊断；
2. movement mask、邻居动作、多步损失、terminal value、ensemble、fallback 的可归因消融；
3. Jinan 和 Hangzhou 独立 held-out flow 的配对统计，而不是同一 flow 换 seed；
4. 与 MaxPressure、SharedDQN、强 pressure-RL、图协调 RL 和直接 model-based TSC 方法的清晰边界。

在这些证据出现前，`gap_matrix.csv` 和 `idea_pool.csv` 中的条目只能称为“通过当前证据审计的候选”，不能写成已确认创新。
