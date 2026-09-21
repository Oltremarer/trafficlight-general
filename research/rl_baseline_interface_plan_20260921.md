# 基于当前接口接入 RL baseline 的改进计划

日期：2026-09-21。状态：待实施计划。本轮重新读取当前工程接口，结合固定版本的逐仓库源码审查制定；没有修改运行代码、安装算法依赖、运行训练或声明复现成功。

目标是继续使用本工程的 CityFlow 执行、策略调用和评估接口，让不同 baseline 接入自己的观测编码、网络和学习器。第一批完成常规相位控制，随后补动态时长、离线序列及跨场景能力。接口覆盖与具体算法实现分别交付。

**建议保留 `TrafficEnv.reset/step`、`Policy.reset/act` 和 `EpisodeRunner.run` 的现有调用形式；增加可选观测视图、策略辅助输出、整网 transition 和独立 Learner。不要把所有算法改写成当前 shared DQN 的网络与训练规则。**

本计划沿用[逐库审查总表](/Users/azure/.agent-reach/research/cityflow-baseline-adapters-20260921/interface_coverage.md)的固定提交。论文成绩不是本计划的验证结果；对已发现的上游实现问题，修正版本须记录差异，不能以“忠实复现”掩盖修正。

## 1. 当前接口能保留什么，需要补什么

| 当前代码与位置 | 已有能力 | 最小必要改进 |
|---|---|---|
| [types.py:108](</Users/azure/Documents/ChatGPT/world model traffic/src/cityflow_tsc/types.py:108>)、[simulator.py:101](</Users/azure/Documents/ChatGPT/world model traffic/src/cityflow_tsc/simulator.py:101>) | snapshot 已有 lane 数量、排队、车辆 ID/速度/距离 | 从现有 snapshot 生成 baseline 特征；需要等待累计、流入流出时，再增加 tick observer |
| [types.py:122](</Users/azure/Documents/ChatGPT/world model traffic/src/cityflow_tsc/types.py:122>)、[observations.py:30](</Users/azure/Documents/ChatGPT/world model traffic/src/cityflow_tsc/observations.py:30>) | movement 特征、mask、当前相位和邻居 | 追加可选 baseline_view；保持旧 features 的含义、顺序和 schema 不变 |
| [topology.py:73](</Users/azure/Documents/ChatGPT/world model traffic/src/cityflow_tsc/topology.py:73>) | movement 的进出 lane 集合、engine 相位映射 | 保留精确 laneLink 对、有序 lane、lane/road 长度，生成独立 graph view 和 source 相位映射 |
| [environment.py:23](</Users/azure/Documents/ChatGPT/world model traffic/src/cityflow_tsc/environment.py:23>) | 已支持注入 ObservationBuilder、RewardCalculator | 扩展构造器的可选奖励累计器；不需要换整个环境类 |
| [environment.py:126](</Users/azure/Documents/ChatGPT/world model traffic/src/cityflow_tsc/environment.py:126>) | 每个仿真 tick 都取 snapshot、统计指标 | 在相同推进点累计 pre/post reward，避免重复推进或另一套时钟 |
| [environment.py:153](</Users/azure/Documents/ChatGPT/world model traffic/src/cityflow_tsc/environment.py:153>) | `[N]` 相位动作、黄灯/全红、五元组返回值 | 固定周期入口继续使用；动态时长后续增加 step_request |
| [types.py:151](</Users/azure/Documents/ChatGPT/world model traffic/src/cityflow_tsc/types.py:151>) | Policy.reset/act、PolicyOutput.actions | baseline adapter 继续实现该协议；增加可选辅助输出与交互反馈协议 |
| [training.py:22](</Users/azure/Documents/ChatGPT/world model traffic/src/cityflow_tsc/training.py:22>) | shared DQN 网络、target、optimizer、replay、保存与恢复 | 新训练入口只管交互和调度；网络和更新规则交给具体 learner |
| [training.py:122](</Users/azure/Documents/ChatGPT/world model traffic/src/cityflow_tsc/training.py:122>) | 每路口独立的扁平样本；done 合并 terminated/truncated | 新路径先保留 joint transition，分别记录终止/截断，按 profile 生成学习 mask |
| [runtime.py:16](</Users/azure/Documents/ChatGPT/world model traffic/src/cityflow_tsc/runtime.py:16>) | 工厂当前固定选择 QueuePressure builder 和 QueueReward | 加 keyword-only profile 参数；省略时保持旧行为 |
| [runner.py:17](</Users/azure/Documents/ChatGPT/world model traffic/src/cityflow_tsc/runner.py:17>) | 所有策略共用评估、指标和轨迹输出 | 可选反馈回调、动作请求路由、writer 注入；评估仍由这一入口完成 |
| [trajectory.py:195](</Users/azure/Documents/ChatGPT/world model traffic/src/cityflow_tsc/trajectory.py:195>) | 严格读取 v1、保留整网时序和拓扑 | 新 baseline 数据使用显式 v2 writer/reader；旧 v1 继续供 World Model 使用 |

当前 `NetworkSnapshot` 已有大部分基础数据，主要缺的是稳定的算法视图与训练契约。现有 `RichTrafficObservationBuilder` 的 50/100m 统计不能直接代替 LLMLight 的 167m running 特征。当前四方向邻居也不能直接代替所有 CoLight 的图。

## 2. 比较上游实现后确定的接入方式

| 来源、固定提交前缀 | 具体利用内容 | 本地 adapter / learner | 必须保留的差异 |
|---|---|---|---|
| [LLMLight](/Users/azure/.agent-reach/research/cityflow-baseline-adapters-20260921/llmlight.md) `d5d4180` | PressLight、MPLight 及 E/A 变体；CoLight/E-CoLight/A-CoLight；规则方法 | LLMLightPolicyAdapter + RoundQLearner；局部或整网 batch | TensorFlow/Keras；按 round 采集后更新；pre-step 区间奖励；源 lane/phase 顺序；CoLight 几何 kNN |
| [LibSignal](/Users/azure/.agent-reach/research/cityflow-baseline-adapters-20260921/libsignal.md) `127af9f` | IDQN、FRAP、MPLight、CoLight、IPPO、MADDPG 等 | 按算法包装纯计算模块，或提取模型/更新核；独立与共享参数分别配置 | PressLight 的 signed pressure、MPLight 的 queue reward；道路图；PFRL observe 内部更新；不能仅看类名替换 |
| [cMALC-D](/Users/azure/.agent-reach/research/cityflow-baseline-adapters-20260921/cmalc_mappo.md) `93a9d9f` | local actor、local/global critic、GRU、episode 数据 | ActorCriticPolicyAdapter + 本地 PPOLearner | 源版本为共享 actor；MAPPO/IPPO 主要区别是 critic 输入；原路径使用 n-step return，未调用 GAE；历史 replay/旧概率问题要修正 |
| [DynamicLight](/Users/azure/.agent-reach/research/cityflow-baseline-adapters-20260921/dynamiclight.md) `e7f7cd4` | 相位 Q + 离散时长 Q、分阶段学习 | DurationPolicyAdapter + StagedDurationQLearner | 10/15/20/25/30/35/40 秒动作菜单；黄灯计入时长；各路口独立到期 |
| [FuzzyLight](/Users/azure/.agent-reach/research/cityflow-baseline-adapters-20260921/fuzzylight.md) `766e27f` | 排队规则选相位，连续 actor 学时长 | DurationPolicyAdapter + DurationActorCriticLearner | 保存连续原值与执行整数秒；不是两个动作都由 RL 学习；实际 learner 用终点 reward |
| [DiffLight](/Users/azure/.agent-reach/research/cityflow-baseline-adapters-20260921/difflight.md) `188ee2c` | lane 序列、观测缺失、扩散与动作分类 | HistoryPolicyAdapter + OfflineSequenceLearner | 序列时间对齐、reward 条件和 missing mask；动作头固定四类；论文中的 BC/CQL/DT 等不在该库中 |
| [MetaLight](/Users/azure/.agent-reach/research/cityflow-baseline-adapters-20260921/metalight.md) `72bda1a` | task 采样、任务内适应、外层更新 | TaskRunner + MetaLearner | task 不是同一城市的 agent；原实现主要是单路口；测试适应预算必须显式 |
| [DuaLight](/Users/azure/.agent-reach/research/cityflow-baseline-adapters-20260921/dualight.md)、[GESA](/Users/azure/.agent-reach/research/cityflow-baseline-adapters-20260921/gesa.md)、[X-Light](/Users/azure/.agent-reach/research/cityflow-baseline-adapters-20260921/xlight.md)、[CoSLight](/Users/azure/.agent-reach/research/cityflow-baseline-adapters-20260921/coslight.md) | 场景 embedding、多场景 PPO、历史建模、协作者选择 | 复用 joint/PPO/history 能力，分别增加任务数据 | 审查主路径为 SUMO；队列米、occupancy、相位编号不能直接等同 CityFlow；须标明移植 |
| [RobustLight](/Users/azure/.agent-reach/research/cityflow-baseline-adapters-20260921/robustlight.md)、[SLight](/Users/azure/.agent-reach/research/cityflow-baseline-adapters-20260921/slight.md) | 观测修复、分组策略模块 | 修复/分组扩展 | 权重依赖或默认学习链缺口未解决前，不计为可运行完整 baseline |

LLMLight 首轮优先包装其独立算法类和必要基类，隔离 TensorFlow 依赖，不引入其 CityFlowEnv/Pipeline，也不要求 LLM。LibSignal 不整体导入 Registry/World，而按所选算法拆清依赖。cMALC-D 借用网络和数据组织，训练收集逻辑在本地纠正。当前 PyTorch DQN 与 TensorFlow adapter 可并存；统一改写为 PyTorch 是另一个实现选择，不能混入“只接接口”的工作量或宣称原实现等价。

依赖组合尚未安装验证。实施时为所选 backend 锁定实际可用的 Python/框架/CityFlow 版本并懒加载；不为运行规则策略要求安装所有算法依赖。源码直接引入的授权边界沿用逐库报告；接口设计不以整体复制仓库为前提。

## 3. 公共接口：沿用现有签名，增加明确的可选能力

以下类、字段和方法名称都是待实现设计，不是当前已存在 API。

### 3.1 BaselineProfile：一个算法名字对应一个明确实现

小型 dataclass/配置即可，字段包括：

- `baseline_id / source_repo / source_commit / adapter_revision`。
- `observation_spec`：输入字段、顺序、单位、相位编码、图、normalizer、观测权限。
- `action_spec`：source 相位到本地动作的映射、固定周期或动态时长、合法动作。
- `reward_spec`：原始分量、局部/全局归约、系数、时间窗口、学习前缩放。
- `learning_spec`：独立/共享参数、local/joint/sequence batch、step/round/rollout 更新、bootstrap。
- `protocol`：数据文件 hash、控制周期、黄红灯、训练/验证/测试划分、交互预算与 checkpoint 选择。

source 版本与统一实验协议分开命名。主实现保留来源的网络、状态、奖励和训练规则；若改用统一奖励或控制周期，在 manifest 中标出 override。cMALC-D 的修正版明确记录行为概率和采样调度改动，默认不把历史 replay 问题带入新 learner。

不能只设 `algorithm='CoLight'` 就隐藏所有选择。例：LLMLight CoLight/E/A 共用网络类，输入分别为 phase8+vehicle12、phase8+efficient queue pressure12、再加 running12；三个 profile 就足够，无需复制三个网络。

### 3.2 观测：给 NetworkObservation 增加可选 baseline_view

保留 `features / feature_names / movement_mask / valid_mask / neighbor_index` 及旧 schema。末尾追加默认 `None` 的 `baseline_view`，旧构造器和 WorldModelPolicy 不需要改输入。

新增组合式 `BaselineObservationBuilder`：先调用原 builder 生成原观测，再用同一个 snapshot 和静态拓扑映射生成所选 baseline 的视图。`build(snapshot, current_phase, signal_stage, phase_elapsed_s)` 签名保持不变。

组合 builder 的 reset 必须同时转发给内部 movement builder 与 baseline observer，尤其不能让 rich builder 的历史跨回合残留。新旧视图分别持有 schema ID，数组保存为当前时刻的独立快照，不引用下一 tick 会原地修改的缓存。

`BaselineView` 至少包含：schema ID、有序 lane/phase 编码的 actor 输入、实体顺序、独立 graph spec、padding/validity mask。PPO critic 的全网拼接由训练 adapter 明确构造；clean 真值、离线未来标签等训练专用数据走独立 training context，不能混进 actor_view。

静态补充信息建议存 `NetworkSpec` 的可选扩展对象：laneLink 配对、lane 长度、方向和完整有向道路图。保留现有四方向表，LLMLight adapter 自建几何 kNN，LibSignal adapter 选道路图。模型节点顺序与环境 intersection_id 建立双向映射，观测/动作/奖励/下一状态共用它。

源实现的四进口、12 lane、4/8 相位假设须检查语义；对不满足的拓扑报具体不支持原因，不能通过 padding 伪装算法已支持。当前 loader 对同方向多个邻居会报错，第一批以已支持路网为范围；泛化到任意拓扑时再扩展 loader 的非四方向模式。

### 3.3 动作：PolicyOutput.actions 继续表示本地逻辑相位

所有 adapter 继续提供 `reset(seed, network)` 和 `act(observation, deterministic)`。在 adapter 内将源动作转为本地逻辑动作，由 `network.engine_phase()` 转为仿真器相位；不把上游 `action+1` 或 SUMO `action*2` 带入执行层。

PolicyOutput 末尾增加可选 `action_request` 和 `behavior`：

- 普通相位策略只有 `actions[N]`，继续调用 `env.step(output.actions)`。
- 动态策略提供 ActionRequest：phase actions、duration_s、decision_due_mask、源离散时长索引/连续原值。若同时存在 actions 与 request.phase，必须一致。
- BehaviorInfo：采样动作的 logprob、value、行为版本、采样前 recurrent state，以及算法需要的协作者/skill 信息。不要求规则策略提供。

动作 clip/round 前的输出、映射后的请求、真正执行的动作分开记录；logprob 对应真实采样的变量。recurrent state 由 PolicyAdapter 自己维护，因为当前 act 签名没有 hidden 入参；输出中的副本用于学习与审计，runner 不再推进同一 hidden。reset 只清本回合上下文，不重建网络或清空跨回合训练 replay。

历史策略实现一个可选的 `observe_context(feedback)` 协议。交互后、下一次 act 前调用一次，训练和评估都调用；feedback 只含该策略协议允许的已发生观测、动作、奖励和时间，不带未来标签或 critic 特权信息。该方法不执行优化、不追加训练 replay；学习数据只交 learner.observe。

保持当前 policy.reset 在 env.reset 前的顺序；首个 act 用 reset 返回的 observation 初始化当前回合历史，不依赖 reset 时尚未产生的环境状态。训练和评估共用同一个动作执行分派：无 action_request 调旧 step，有 request 调 step_request，禁止静默忽略 request。现有 WorldModelPolicy 明确只支持固定周期，不能被动态执行模式自动复用。

### 3.4 奖励：保留 RewardCalculator，另加时间累计

`RewardCalculator.compute(snapshot)` 保留为空间公式接口；有连续等待时间等历史量时，由单独 tick observer 累计并在 reset 清空。增加 RewardAccumulator，在 `_advance` 每个 tick 前/后读取 snapshot，按 profile 计算 mean/last/sum 或时间积分。

环境 reset 同时清空有状态的 reward observer/accumulator；每个决策窗口有明确的开始、tick 更新、结算。返回的 reward 仍为局部 `[N]` 向量；MAPPO 所需的全局 learning reward 在 learner/profile 层显式归约，不让 env.step 的 reward 随算法变成标量。

`mean`、按 tick 求和和按秒积分分别定义；若 simulator_step_s 不为 1，不能把 sum 自动当积分。缺省累计器使用现有末态 QueueReward。新字段仅在 baseline/profile 模式放入 `info['baseline']`，保留旧模式的 info 字典内容；现有 core test 会检查整个 info 字典相等。

重要实例：LLMLight 训练奖励是 t..t+29 的 pre-step 均值；当前环境奖励是 t+30 的终点值。源末条样本 next_state 还有 T−1 秒日志规则且无 done；若提供 source-compatible profile，必须记录末个 pre-step 边界供其样本 codec 使用。若采用本地 horizon/bootstrap 语义，要命名为适配版。奖励系数与 learner NORMAL_FACTOR 分开记录，避免重复缩放。

输出保留原始局部 reward 分量；learner 根据 profile 生成 local vector 或 global scalar。交通指标仍由 MetricCollector 独立计算，不受训练 reward 更换影响。

### 3.5 JointTransition：先保持同一时刻的整网对应关系

基本字段：episode/scenario ID、time、obs/next_obs、requested/executed action、raw local reward components、learning reward、dt、terminated、truncated、mask、behavior。checkpoint/profile/schema/node mapping ID 随 batch 或 manifest 保存，不必每样本重复大对象。

learner 自行选择：

| 数据布局 | 处理方式 | 目标算法 |
|---|---|---|
| local `[B,F]` | 由 joint transition 按路口展开，保留 agent/scenario 身份 | 独立/共享 DQN、PressLight、FRAP/MPLight |
| joint `[B,N,...]` | 同一时刻整网抽样，图和所有节点对齐 | CoLight、MADDPG、集中 critic |
| sequence `[B,T,N,...]` | 不跨 episode；保存历史 mask、chunk 初始 hidden 和末尾 bootstrap 状态 | recurrent PPO、DiffLight、X-Light |

terminated 与 truncated 原样保存，另由 learner 生成 bootstrap mask；不要直接复用旧 DQN 的二者合并规则。on-policy rollout 必须来自冻结的行为策略，保存采样时概率、value、末状态和行为版本。更新后轮换该 batch，不任意混入历史 replay；保存 old_logprob 本身不能让旧数据自动成为标准 PPO 样本。

### 3.6 Learner 与 runner：统一交互，不统一所有更新公式

新增 TrainingRunner，负责 reset→act→step→记录 transition→上下文反馈→learner.observe→按配置更新。Learner 持有 policy、网络、target、optimizer、normalizer 和 buffer。建议操作为 `observe / update_if_due / end_episode / end_round / save / load`。

- Q learner 可以 step 更新；LLMLight learner 在 round 结束后按原规则抽样、fit 和更新 target。
- PFRL adapter 明确标记 observe 内部负责学习，公共 update 不再做第二次优化；评估必须显式关闭其训练状态。
- PPO learner 等完整 rollout 或规定长度后更新；local/global critic 是配置差别，不能把 actor 的观测范围一并改成全网。
- offline learner 直接使用 dataset，不调用 env.step 收集训练经验；评估仍复用 EpisodeRunner。
- Meta/task runner 在外层组织 source/target、support/query 和适应预算，不扩展 env.step 来承担元学习。

先保留原 DQNTrainer 和训练命令；新增 native DQN learner 时，用现有测试对照其 loss/target/replay/checkpoint 语义。只有需要迁移旧入口时才抽公共代码，避免为接新算法重写已运行的 World Model 与 DQN 链。

## 4. 数据、保存和指标的兼容约定

1. **轨迹 v1 保留。** 现有 reader 精确检查 `cityflow-tsc-trajectory-v1`。新增 `BaselineTrajectoryWriter/Reader` 写 v2，含 baseline_view、执行动作/dt、behavior、学习 mask 和 profile。EpisodeRunner 的默认 writer 仍是 v1；选择 baseline profile 时显式使用 v2，并输出到独立 baseline run 目录。World Model 不自动消费 v2。
2. **旧数据不能补造。** v1 没有原始 lane 和行为概率，不能从 movement 特征伪造 DiffLight 数据或 PPO rollout。需要这些字段的算法只能使用真正记录了它们的新数据。
3. **checkpoint 分推理与续训。** 推理保存所有参与决策的模块、normalizer、profile、编码映射；续训另存 optimizer、target、replay/rollout、schedule、RNG。第一版保证回合/rollout 边界续训；若没有 CityFlow archive 和完整环境恢复，不承诺回合中途精确续训。
4. **旧 hash 保持。** 现有 DQN/WorldModel 的 network schema hash 显式列举旧字段；不要因为追加 baseline 拓扑数据而改写旧 hash 语义。新增 baseline_schema_hash 单独覆盖 laneLink、图、输入映射、奖励和时序。跨城市策略用明确的 schema 能力检查，而非完全相同 roadnet hash 才允许迁移。
5. **评估与训练分开。** deterministic 是否开启按协议声明；清理历史/hidden 并冻结 normalizer/optimizer。现有 ATT 来自 engine，等待/queue/throughput 来自 MetricCollector；记录指标定义与完成车辆数，不将不同统计分母的论文 ATT 当测试真值。可增报 seen/completed/active-at-horizon，保留现有指标含义。

## 5. 分阶段实施，每一步都接入具体算法

### P1：补足公共观测与奖励接口，接入 PressLight

**改动：** types/topology 增加可选 baseline 数据；runtime 接收 profile；增加 lane/phase codec、RewardAccumulator、JointTransition、最小 registry 和 TrainingRunner。保留旧调用默认。先包装一个 LLMLight PressLight 完成 act→执行→样本→真实更新→保存→统一评估的整条路径。

**交付：** 旧 FixedTime/MP/shared DQN 继续可用；新增 `presslight.llmlight` 一个具体 adapter。输入、source 相位与 reward/target 数值可与固定源码对照。不要先注册十几个空算法。

### P2：接入 MPLight 与 CoLight 核心家族

**改动：** 加相位需求/竞争编码、整网 replay、几何 kNN graph、round learner；补近停止线 running 特征。MPLight 和 CoLight 使用各自网络，不复用旧 QNetwork 冒充。

**交付：** MPLight/E-MPLight/A-MPLight、CoLight/E-CoLight/A-CoLight；E-PressLight 沿用同类网络并切换明确输入。三个 CoLight profile 共用一个网络实现。第一批相位控制的主要覆盖在此完成。

### P3：补充 LibSignal 方法与 PPO/MAPPO

此阶段有两个独立增量，均依赖 P1；无需等待全部专项方法。

- **P3a：** LibSignal FRAP、IDQN、IPPO/MADDPG 按需要逐个 adapter；复用 local/joint 数据契约。LibSignal MPLight/CoLight 作为来源不同的实现单独命名，不覆盖 LLMLight 同名 profile。接规则 SOTL/E-MP/A-MP 时复用已有观测，保留各自计时规则。
- **P3b：** 实现 rollout buffer、shared recurrent actor、local/global critic、行为概率保存和 on-policy learner。参考 cMALC-D 网络与字段，修正历史样本/old_mac 不匹配；源 n-step 与新增 GAE 作为显式不同设置。接入本地 `ippo`、`mappo`，记录与上游的差异。

**交付：** Q-learning 与 PPO 两类训练都能通过同一 Policy/TrafficEnv/EpisodeRunner 执行；训练器选择不会更换 CityFlow 环境或指标口径。IPPO 的独立/共享参数以及局部/全局 reward 必须分别声明。

### P4：接 DynamicLight/FuzzyLight 的时长控制

**改动：** 增加 ActionRequest、`TrafficEnv.step_request`、每路口信号状态机、pending transition、due/complete mask。所有推进共用同一个 backend 和 tick observer，但每路口独立管理黄灯/全红/绿灯；旧固定 step 不能直接包装成“全网推进 min(duration)”就算完成。

在事件模式，观测仍输出完整网络 `[N,...]`，只允许 due 路口重新出动作；其余动作继续执行。动作到期不是 terminated。episode 结束时记录未完成动作的实际奖励与持续时间，按 profile 闭合/截断样本。duration 总时长与纯绿灯时长明确区分，新的 DurationControlSpec 不复用固定周期必须留下正绿灯的全部验证逻辑。

**交付：** DynamicLight 的相位/时长 Q 和分阶段 schedule；FuzzyLight 的规则相位+连续 duration actor。测试 A=10s、B=25s 的交错决策，确保 B 不会在第 10 秒被错误重新采样。

### P5：接离线、历史和跨场景方法

- **P5a（依赖 v2 数据）：** episode/window dataset、独立 missing/history mask、normalizer、离线训练入口，接 DiffLight。实际数据可用性与内容仍需核实；自采数据必须标明与作者数据不同。BC/离散 CQL/DT 可以复用数据与评估接口，但它们的实现另有工作量，不能算作 DiffLight 附带。
- **P5b（依赖 history/PPO/joint）：** TaskContext、scenario/node 身份、训练/适应预算和多场景 batch；分别接 MetaLight，以及需要重新编码 CityFlow 观测/动作的 DuaLight/GESA/X-Light/CoSLight。CoSLight 须保留协作者选择的概率和损失，不能只保留交通相位动作。
- RobustLight 与 SLight 在同一接口上增加修复/分组模块，但完整算法接入依赖解决审查指出的权重或训练链缺口。

应急车辆、安全碰撞和 MOSS 算法不纳入这版普通 CityFlow 接口的完成范围；它们需要额外任务状态、动作或 cost，而不仅是一个新 policy 类。

## 6. 文件改动范围

已有文件保留原职责，新增算法代码集中在 `src/cityflow_tsc/baselines/`。下表中的新增路径尚不存在，按阶段创建，不一次搭空框架。

| 文件/目录 | 计划内容 | 阶段 |
|---|---|---|
| `types.py` | 可选 baseline_view、action_request、behavior；保留原 positional/keyword 构造兼容 | P1，duration 在 P4 |
| `topology.py` | laneLink/长度/顺序、独立 graph spec、source action codec 所需元数据 | P1–P2 |
| `baselines/profiles.py`、`baselines/registry.py`（新增） | 来源与协议配置、懒加载 factory、可运行算法列表 | P1 起逐项添加 |
| `baselines/observations.py`、`baselines/codecs.py`（新增） | 组合 builder、lane/phase/graph 编码 | P1–P2 |
| `rewards.py`、`environment.py`、`runtime.py` | 空间 reward、tick 累计器、profile 注入；老默认不变 | P1 |
| `baselines/transitions.py`、`baselines/training.py`（新增） | joint 数据、TrainingRunner、learner 协议和更新调度 | P1 |
| `baselines/adapters/`、`baselines/learners/`（新增） | 每个来源的 policy/学习器，按需加 buffer/network 文件 | P1–P5 |
| `runner.py` | 可选 writer、context feedback、dynamic request 分支；统一评估 | P1/P4 |
| `baselines/trajectory.py`、`baselines/checkpoint.py`（新增） | baseline v2 schema 和完整模型元数据；保留旧 v1 | P1–P3 |
| `train_baseline.py`（新增）、`pyproject.toml` | 通用训练/评估入口和按需依赖；旧命令继续可用 | P1 |
| `baselines/timing.py`（新增）、`environment.py` | 每路口事件与信号调度 | P4 |
| `baselines/datasets.py`、`baselines/tasks.py`（新增） | 离线序列和跨任务外层调度 | P5 |
| `tests/test_core_chain.py`、`tests/test_dqn_chain.py` 及按能力新增的测试 | 旧接口回归，编码/奖励/联合样本/PPO/时长验证 | 随对应改动 |

## 7. 实现后的检查与实际运行证据

本轮只读了测试，没有运行，也没有新增测试。实施时沿用现有 DeterministicBackend，检查与改动直接有关的行为：

| 检查 | 要证明的行为 |
|---|---|
| 旧接口兼容 | 默认 FixedTime/MP、DQN 和 WorldModel 输入 schema、v1 数据读取不变；新增可选字段不使旧 checkpoint hash 失效 |
| 观测/动作编码 | 每条 lane 使用不同计数，核对 source 顺序、phase8 放行码、pressure 公式、source→local→engine 映射；不是只检查 shape |
| 时间奖励 | 对可手算 tick 队列验证 pre/post/last/mean、黄灯计入、终点截断和缩放；旧模式 info 字典不变 |
| 学习更新 | 比较固定输入下的目标值、mask 和真正变化的参数；CoLight 节点样本不跨时刻；不把相位竞争网络替换成普通 MLP |
| PPO | 更新前重算行为概率一致，rollout 的行为版本匹配；RNN reset/chunk hidden 正确；非法动作概率为零；评估不更新参数或训练 buffer |
| 轨迹/保存 | v2 读回保留 actor view、joint 对齐和行为信息；v1 不被伪转换；保存加载后动作一致，边界续训保留必要状态 |
| 动态/离线 | 只在 due 路口重新采样；每个 pending transition 独立闭合；序列不跨 episode，actor 不读未来标签 |

确定性接口检查通过后，在用户指定的 CityFlow 运行环境执行所接算法的真实训练和冻结评估，记录软件版本、roadnet/flow hash、独立训练种子、训练预算、checkpoint 选择、指标定义和完成车辆数。多次评估同一个训练模型，不作为多次独立训练。具体长训练预算与机器由后续实施任务确定，本计划不启动远程任务或额外参数扫描。

可运行列表逐个记录“已适配、接口检查通过、真实训练/评估完成”的状态；论文协议复现另有独立证据。第一轮实施优先 P1→P2→P3，完成用户清单中的普通固定相位 baseline；P4/P5 是明确的扩展工作，不以预留字段代替算法交付。

本计划经过一名子 agent 对当前代码的独立兼容性复核，已纳入 builder/奖励状态 reset、观测快照复制、动作请求统一分派、WorldModel 固定周期限制、历史与学习分工，以及 v1/v2 writer 显式选择。复核为静态设计检查，不是实现或训练验证。
