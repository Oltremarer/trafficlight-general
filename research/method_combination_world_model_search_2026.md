# World Model x 方法组合：首轮论文检索矩阵

检索与核验截止：2026-09-03。  
范围：按方法组合而非应用领域检索；以 2024–2026 主会为主，保留少量 2020–2023 奠基基线。已有跨领域 30 篇卡片见 [cross_domain_world_model_top30_2026.md](cross_domain_world_model_top30_2026.md)。

## 0. 这轮检索回答什么

统一判断链为：

当前状态 + 候选动作/交互 → World Model 预测动作条件未来/不确定性 → 规划、评价或训练 → 决策。

因此，不把“用了 diffusion、LLM 或 GNN”本身当成方向；记录它究竟改变了状态表示、转移预测、未来采样、风险估计、候选评价、规划搜索还是策略更新。

### 纳入标记

- **核心**：正式会议来源可核验，且满足“动作条件未来 → 决策用途”。
- **相邻**：对该方法方向重要，但当前来源不足以证明它实际用于动作条件转移或决策；精读前不据此提出方法空白。
- **风险锚点**：用于界定失败模式或可靠性证据；不等同于主线候选。

## 1. 搜索日志与查询覆盖

来源优先级：正式 proceedings、OpenReview、PMLR、CVF、ACL Anthology；arXiv 仅用于定位正式版本。查询分成“方法、失败、基准”三类，避免只找到正向结果。

| 批次 | 已执行的英文查询族（均加 venue/year 复核） | 主要目的 |
|---|---|---|
| 决策层 | world model + (MPC OR MCTS OR planning OR offline RL OR imitation OR preference OR Pareto) | 找未来如何进入搜索、policy extraction、偏好学习或离线训练 |
| 可靠性层 | world model + (causal OR counterfactual OR uncertainty OR calibration OR safe OR constrained OR OOD) | 找“预测错时如何不误导决策”的实证工作 |
| 时间/结构层 | world model + (hierarchical OR skill OR graph OR relational OR multi-agent OR continual) | 找长时、关系与多任务的动力学建模 |
| 知识/仿真层 | world model + (LLM OR language OR code OR symbolic OR memory OR sim-to-real OR diffusion) | 查语言、规则、仿真残差和生成模型在链中的真实位置 |

## 2. 核心候选矩阵

### A. 规划、策略提取、偏好与离线学习

| 论文（正式来源） | 组合与方法位置 | World Model 怎样进入决策 | 论文级证据 | 应关注的风险 |
|---|---|---|---|---|
| [TD-MPC2](https://openreview.net/forum?id=Oxh5CstDJU) — ICLR 2024 | latent dynamics + MPC | 预测 reward/终止/价值并优化候选动作序列 | 多连续控制任务；此前报告已收录 | 适合作为短时 MPC 强基线，不等于离散联合控制可直接迁移 |
| [PWM: Policy Learning with Multi-Task World Models](https://proceedings.iclr.cc/paper_files/paper/2025/hash/240ea1741b205ea295721d55184ac43b-Abstract-Conference.html) — ICLR 2025 | multi-task WM + first-order policy extraction | 先以离线数据预训练 WM，再对 WM 反传提取 task policy，而非在线采样搜索 | 覆盖高动作维度与 80-task 设置 | 正则化后的模型景观更平滑是其前提；需区分“优化容易”与“真实环境最优” |
| [Hindsight PRIORs for Reward Learning from Human Preferences](https://proceedings.iclr.cc/paper_files/paper/2024/hash/fe489a28a54583ee802b8e2955c024c2-Abstract-Conference.html) — ICLR 2024 | forward dynamics + preference reward learning | 用前向模型估计轨迹状态的重要性，改善偏好标签的 credit assignment | locomotion/manipulation 中测 reward recovery 与 policy 效果 | WM 不替代人类偏好；它只帮助定位偏好应归因给轨迹的哪部分 |
| [BECAUSE](https://proceedings.neurips.cc/paper_files/paper/2024/hash/cff98e0b76e05fd1df5c9256724b3af1-Abstract-Conference.html) — NeurIPS 2024 | causal representation + offline MBRL + conservative planning | 在离线 state/action 中学习去混杂表示和不确定性，降低模型训练目标与 policy 成功之间的不一致 | 18 个不同数据质量/上下文任务及理论界 | 因果结构与混杂设定是关键假设；不能把它当作不需要行为支持的 OPE |
| [Reward-free World Models for Online Imitation Learning](https://proceedings.mlr.press/v267/li25af.html) — ICML 2025 | latent WM + inverse soft-Q imitation | 用无重建 latent dynamics 和规划，绕开 reward-policy 联合优化的不稳定性 | DMControl、MyoSuite、ManiSkill2 | 是 online imitation，不等于严格 offline RL；需单独看 expert coverage |
| [WebSynthesis](https://aclanthology.org/2026.acl-long.1157/) — ACL 2026 | LLM WM + MCTS + trajectory synthesis | 预测 accessibility-tree 转移，使搜索能在模型内回退并合成轨迹 | WebArena、WebVoyager、Mind2Web-Online | 合成网页状态若和真实网页偏离，训练数据会有系统偏差 |

**本批结论**：WM + preference 已有可信切入点，但主要是偏好 credit assignment，不是“让 WM 自己判断目标”；WM + Pareto/多目标在本轮严格主会检索中没有找到同等直接、动作条件且用于候选多目标排序的代表作。该空白只是待深查线索，不能先当作创新结论。

### B. 因果、反事实、不确定性与安全

| 论文（正式来源） | 组合与方法位置 | World Model 怎样进入决策 | 论文级证据 | 应关注的风险 |
|---|---|---|---|---|
| [Language Agents Meet Causality](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5c5bc3553815adb4d1a8a5b8701e41a9-Abstract-Conference.html) — ICLR 2025 | causal representation + LLM interface | 因果 WM 把状态、语言动作和状态转移连起来，供 LLM 查询/规划 | 因果推断与不同时间尺度的 planning | 因果变量可识别性和自然语言映射质量是核心前提 |
| [A Causal World Model Underlying Next Token Prediction](https://proceedings.mlr.press/v267/yehezkel-rohekar25a.html) — ICML 2025 | causal structure probing + language model | 用受控棋类环境检验 GPT attention 中是否有可用于合法行动预测的因果结构 | Othello/Chess 的 OOD 合法动作测试 | 主要是诊断/解释性证据，不能直接算为通用控制 WM |
| [SafeDreamer](https://openreview.net/forum?id=tsE5HLYtYg) — ICLR 2024 | Dreamer WM + Lagrangian safety planning | 在 imagined rollout 中同时预测回报与 cost，用约束优化选动作 | SafeRL 的视觉与控制任务 | 安全依赖 learned cost/dynamics；低 cost 不代表在真正 OOD 条件下已校准 |
| [STICA: Object-Centric World Models for Causality-Aware RL](https://ojs.aaai.org/index.php/AAAI/article/download/39642/43603) — AAAI 2026 | object-centric Transformer WM + causal policy/value | 将对象、动作和 reward 作为 tokens 预测对象级交互；policy/value 用 token 级因果关系选动作 | object-rich benchmark 的样本效率与最终性能 | 对象 slot 分解质量和“因果关系”估计是否随环境变化稳定，必须分开验证 |
| [Uncertainty-aware Latent Safety Filters](https://proceedings.mlr.press/v305/seo25a.html) — CoRL 2025 | WM epistemic uncertainty + conformal threshold + reachability | 把不确定性并入 latent reachability，产生 safety monitor 和 fallback action | 仿真与 Franka 视觉控制硬件实验 | 风险锚点：CoRL 适合机器人方向，但不与通用 ML 主会混为同一层级 |
| [DALI](https://proceedings.neurips.cc/paper_files/paper/2025/hash/696996164c52a52b8a162e62574fc9f9-Abstract-Conference.html) — NeurIPS 2025 | contextual WM + latent context adaptation | 从交互推断不可观测环境 context，条件化 WM 和 policy 的想象 | contextual MDP 的零样本外推 | “可干预的 latent dimension”是有趣诊断，不是已经证明真实因果发现 |

**本批结论**：可靠性方向最强的结构不是多加一个 prediction loss，而是“预测未来 + 不确定性/因果变量 + 安全过滤或保守规划”。后续精读应优先检查是否报告了校准、OOD 风险、候选排序或真实闭环安全，而不只看环境回报。

### C. 主动探索、层级时间抽象与长期想象

| 论文（正式来源） | 组合与方法位置 | World Model 怎样进入决策 | 论文级证据 | 应关注的风险 |
|---|---|---|---|---|
| [Plan2Explore](https://proceedings.mlr.press/v119/sekar20a.html) — ICML 2020 | ensemble WM + information gain | 用 imagined ensemble disagreement 规划探索动作，学习可迁移的 task-agnostic behavior | 图像观察下的 zero-/few-shot 控制 | 奠基基线；探索到的不确定区域不一定与下游安全/任务价值相关 |
| [SENSEI](https://proceedings.mlr.press/v267/sancaktar25a.html) — ICML 2025 | foundation-model semantic bias + MBRL exploration | 将语义上有意义的行为作为内在动机，提升得到多用途 WM 的数据质量 | 强调低层信息增益易发现无意义互动 | foundation model 的语义偏置可能漏掉任务关键但不显眼的 dynamics |
| [Curious Exploration via Structured World Models](https://proceedings.neurips.cc/paper_files/paper/2022/hash/98ecdc722006c2959babbdbdeb22eb75-Abstract-Conference.html) — NeurIPS 2022 | GNN WM + multi-step information gain planning | 用实体关系结构的 WM 计算未来信息增益，再零样本规划 | 结构化环境中的组合泛化 | 需要可靠 object/relationship 表示；不能假定交通状态已具备同等对象分解 |
| [Director](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a766f56d2da42cae20b5652970ec04ef-Abstract-Conference.html) — NeurIPS 2022 | latent WM + high/low-level policy | 高层在 latent WM 中选择子目标，低层去实现，缓解超长动作序列 | 像素控制、Atari、DMLab 等长程任务 | 是层级 WM 的强老基线；需对照手工子目标或 option 基线 |
| [Hieros](https://proceedings.mlr.press/v235/mattes24a.html) — ICML 2024 | multi-timescale WM + hierarchical imagination | 学习时间抽象的 world representation，在多个时间尺度想象并训练 policy | Atari 100k 与探索能力 | 更高层预测可压缩 horizon，但 error 也会在 skill 层累计 |
| [PIVOT-R](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6164b6e5352c139e9ddc1a98c09e4e4a-Abstract-Conference.html) — NeurIPS 2024 | waypoint WM + hierarchical executor | 只预测与操作任务相关的关键 waypoint，不建模全部低层轨迹 | manipulation 的时间/计算效率导向 | 关键状态是否充分，取决于任务和 waypoint 标注/诱导方式 |
| [DMWM](https://proceedings.neurips.cc/paper_files/paper/2025/hash/078bc5d384e8d3894f7bce6a34756212-Abstract-Conference.html) — NeurIPS 2025 | RSSM + logic-guided long-term imagination | System 2 的逻辑结构约束 System 1 的状态想象，服务长时 planning | DMControl 与机器人长时任务 | 逻辑模块如何从数据得到、遇到规则错误怎样退化，需全文核验 |
| [Learning Interactive World Model for Object-Centric RL](https://proceedings.neurips.cc/paper_files/paper/2025/hash/8187faaf6759ef6d4e93293339bc656e-Abstract-Conference.html) — NeurIPS 2025 | object-centric WM + interaction primitives + hierarchy | 将任务拆成可组合互动 primitive；高层决定类型/顺序，低层执行 | 高层/低层分工明确 | 不同实体数、遮挡和新交互下的 object decomposition 是风险 |

**本批结论**：层级组合解决的是“rollout 太长”，不是简单把预测 horizon 拉长。要检查的核心量应包括：高层 option/waypoint 预测是否与低层真实执行一致、是否减少真实交互、是否真的改善长程成功率。

### D. 图、多智能体、持续/多任务与结构化状态

| 论文（正式来源） | 组合与方法位置 | World Model 怎样进入决策 | 论文级证据 | 应关注的风险 |
|---|---|---|---|---|
| [Continual RL by Planning with Online World Models](https://proceedings.mlr.press/v267/liu25p.html) — ICML 2025 | online unified dynamics + reward-conditioned MPC | 用一套动态模型配任意当前 reward，避免把任务差异错学成不同物理 | Continual Bench | 任务奖励变化和真实 dynamics 改变要分开；后者未必天然不遗忘 |
| [M3W](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3a2d96d2eb2902043c2db705ca03e9a2-Abstract-Conference.html) — NeurIPS 2025 | MoE placed in WM + multi-agent planning | 在 dynamics 而非 policy 中路由专家，随后在预测 rollout 上优化 | Bi-DexHands、MA-Mujoco | 路由错误会成为长期规划误差源；需看 task-to-expert 的稳定性 |
| [ScaleZero](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4f45d2471a82b3d674f3957ef6170996-Abstract-Conference.html) — ICLR 2026 | multi-task WM + MoE + dynamic parameter scaling | 调整模型容量以保留/扩展 task-specific dynamics，供在线规划 | Atari、DMC、Jericho 多任务 | 着重多任务效率；不同 observation/action 接口统一的代价需单独记录 |
| [Mixture-of-World Models](https://proceedings.iclr.cc/paper_files/paper/2026/hash/33b47b3d2441a17b95344cd635f3dd01-Abstract-Conference.html) — ICLR 2026 | modular VAE + task-conditioned dynamics experts | 用共享骨干与任务专家建模多任务视觉 dynamics，支撑单一 agent 的控制 | Atari 100k 与 Meta-World | 专家/聚类的任务划分可能把连续变化硬离散化；应看跨任务路由稳定性 |
| [Newt: Learning Massively Multitask World Models](https://proceedings.iclr.cc/paper_files/paper/2026/hash/092359ce5cf60a80e882378944bf1be4-Abstract-Conference.html) — ICLR 2026 | language-conditioned multitask WM + offline pretraining + online RL | 先由 demonstrations 学任务表征和 action prior，再在多任务在线交互中联合优化 | 新建 200-task benchmark，报告未见任务快速适应 | 大规模 benchmark 的开放环控制不应替代长期闭环与安全证据 |
| [TMoW: Test-Time Mixture of World Models](https://iclr.cc/virtual/2026/poster/10010038) — ICLR 2026 | mixture of WMs + test-time routing/refinement | 在未见动态环境中调整模型路由、重组或少样本构造新 WM，供 embodied reasoning/action 使用 | VirtualHome、ALFWorld、RLBench | 测试时调路由的稳定性、额外计算与错误专家选择需要独立报告 |
| [Graph World Model](https://proceedings.mlr.press/v267/feng25p.html) — ICML 2025 | message passing + action node | 用图表示多模态状态，action node 连到节点/边，输出预测、生成或规划所需表示 | 含 multi-agent、RAG、planning 等 6 类任务 | 跨任务统一接口不等于每个场景都有强闭环动力学证据 |
| [Grounded Answers for Multi-agent Decision-making through Generative World Model](https://proceedings.neurips.cc/paper_files/paper/2024/hash/52c21a32429a7d6050430b606a286a75-Abstract-Conference.html) — NeurIPS 2024 | generative multi-agent simulator + language-conditioned reward | 分别学习 interaction dynamics/reward，生成交互以回答和辅助多智能体决策 | 多智能体决策问题 | 语言奖励与生成转移的各自误差会耦合；要读真实交互验证 |
| [Social World Model-Augmented Mechanism Design Policy Learning](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a21db07ccc247b6383f78939c8f894c7-Abstract-Conference.html) — NeurIPS 2025 | latent agent traits + social WM + mechanism policy | 对不同机制模拟个体反应，在 WM 中生成训练轨迹 | 税收、协作、选址任务 | trait 可辨识性和 agent policy 非平稳性是决定性假设 |

**本批结论**：图或 MoE 不是装饰性骨干；只有它们改变了“谁与谁交互、哪个 dynamics 专家负责、未来如何合成”时，才计为 WM 组合。对交通而言，这批最相关，但需要把道路拓扑、上下游传播和其他路口策略变化分别验证。

### E. LLM、语言、代码和符号约束

| 论文（正式来源） | 组合与方法位置 | World Model 怎样进入决策 | 论文级证据 | 应关注的风险 |
|---|---|---|---|---|
| [Learning to Model the World With Language (Dynalang)](https://proceedings.mlr.press/v235/lin24g.html) — ICML 2024 | language/image multimodal WM | 用语言作为未来预测的条件/目标，预测未来 text/image 表示并在 imagined rollout 中行动 | 游戏与 photorealistic navigation；可 text-only pretrain | 语言可提供规则和目标，但不保证它对当前动力学已经 grounded |
| [WorldCoder](https://proceedings.neurips.cc/paper_files/paper/2024/hash/820c61a0cd419163ccbd2c33b268816e-Abstract-Conference.html) — NeurIPS 2024 | LLM writes executable code WM | LLM 把经验更新为 Python transition/reward/goal 函数，planner 调程序做预测 | 文本/规则型环境规划 | 强依赖环境能够被离散、确定性、可执行代码近似 |
| [Generating Code World Models with LLM-guided MCTS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6f479ea488e0908ac8b1b37b27fd134c-Abstract-Conference.html) — NeurIPS 2024 | code WM generation + test/MCTS repair | 生成—测试—修复世界代码，再交给 planner | 代码 WM 基准和规划 | 单元测试覆盖不足时，程序化“可执行”仍可能错误 |
| [WALL-E](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5e772a13ccba5255331240dcd99aa38b-Abstract-Conference.html) — NeurIPS 2025 | neuro-symbolic WM + LLM MPC | 从探索轨迹归纳规则/知识图/scene graph，并把可执行模型用于 look-ahead | Mars、ALFWorld | 规则归纳对稀缺或矛盾轨迹敏感；不应把“training-free”误解为无数据 |
| [IMPLEMENT](https://aclanthology.org/2026.acl-long.827/) — ACL 2026 | object-centric symbolic WM + Monte-Carlo imagination + frozen LLM | 先采样候选动作的可能未来并形成 belief summary，再让 LLM 评估/修正 | ALFWorld 的 test-time planning | 符号状态抽取若漏掉 affordance，采样再多也无法补救 |
| [WebEvolver](https://aclanthology.org/2025.emnlp-main.454/) — EMNLP 2025 | co-evolving LLM WM + Web policy | WM 既合成网页训练轨迹，也预测候选网页动作之后的状态 | 多个真实 Web-agent 基准 | 应用真实网页回放审计合成轨迹，防止闭环自我强化偏差 |

**本批结论**：LLM 最适合处在语言规则/任务条件、候选提议、可执行模型归纳、未来摘要评价的位置。若它没有显式状态转移或不参与候选未来比较，应归为 agent memory/knowledge，不作为严格 WM 主线。

## 3. 相邻候选：保留但不提前当作主线证据

| 论文 | 为什么保留 | 当前不直接纳入核心链的原因 |
|---|---|---|
| [RAP: Reasoning via Planning](https://aclanthology.org/2023.emnlp-main.507/) — EMNLP 2023 | 把 LLM 作为 world model 与 agent，使用 MCTS 思考；是 LLM 规划的重要祖先 | 需要全文核验其“环境状态转移”是否可与外部 action-conditioned WM 等同 |
| [Agent Planning with World Knowledge Model](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d032263772946dd5026e7f3cd22bce5b-Abstract-Conference.html) — NeurIPS 2024 | 提供 global task knowledge 和 local state knowledge，针对 agent hallucination | 更像知识/状态辅助模块；需确认是否明确预测 action-conditioned future |
| [A Causal World Model Underlying Next Token Prediction](https://proceedings.mlr.press/v267/yehezkel-rohekar25a.html) — ICML 2025 | 为“LLM 内部 world model”提供受控探针 | 主要研究内部结构与合法下一步，并非闭环 environment planning 主结果 |
| [ReDRAW](https://proceedings.mlr.press/v331/lanier26a.html) — L4DC 2026 | 用 latent dynamics residual 修正 sim-to-real WM，是数字孪生残差的好风险/机制锚点 | L4DC 是领域会议，单列为支持证据，不与顶会主会卡片混用 |
| [WOMBET](https://proceedings.mlr.press/v331/kim26a.html) — L4DC 2026 | 用 uncertainty-penalized WM planning 生成 transfer offline data | 同上；还需核验 source-to-target task 的支持假设 |

## 4. 方向覆盖、基线与下一轮精读

| 方向 | 首轮判断 | 精读优先论文 | 必须提取的反例/评测 |
|---|---|---|---|
| 规划与 policy extraction | 基线成熟，创新风险最高 | TD-MPC2、PWM、WoTE | rollout horizon、model bias、真实闭环 vs model 内指标 |
| Offline/偏好/蒸馏 | 有清晰机制差异：credit assignment、去混杂、保守性 | Hindsight PRIORs、BECAUSE、Reward-free WM | behavior support、OPE、模型误差是否改善候选排序 |
| 因果/反事实 | 很有潜力，但可识别性与“causal”标签需严审 | BECAUSE、Language Agents Meet Causality、DALI | intervention vs observation、context/OOD、结构可辨识条件 |
| 不确定性/安全 | 最适合建立可靠性证据链 | SafeDreamer、latent safety filter、DALI | calibration、violation、fallback、置信度与失败相关性 |
| 主动探索 | 方法较成熟；新点应是“信息是否决策相关” | Plan2Explore、SENSEI、CEE-US | exploration coverage、semantic bias、真实数据成本 |
| 层级 WM | 机制明确：缩短有效规划 horizon | Director、Hieros、PIVOT-R、DMWM | 高层预测—低层执行一致性、长程任务成功率 |
| 图/多智能体/持续 | 适合关系动力学与拓扑系统，但 benchmark 易变 | M3W、Graph WM、Social WM、ScaleZero | non-stationarity、消息/拓扑消融、跨任务遗忘 |
| LLM/符号 | 已有多条路线，不能仅靠“加 LLM”宣称新颖 | Dynalang、WorldCoder、WALL-E、IMPLEMENT | 是否真有 WM、规则是否可执行、真实环境校验 |
| 多目标/Pareto | 首轮严格命中稀少 | 先以 Hindsight PRIORs、SafeDreamer 作相邻基线 | 需定向查“future outcome vector + Pareto selection”，避免把 multi-loss 当多目标决策 |

## 5. 面向交通信号的转译边界

这里尚未提出新方法。对每篇候选，后续只允许按以下接口转译：

路网状态 + 合法信号动作 → 多步 queue/flow/reward/risk/uncertainty future → 候选比较或 teacher choice。

下一轮精读中必须逐篇回答：

1. 论文预测的未来能否替换为交通中的 queue、spillback、throughput、switch cost，还是只适用于图像/文本？
2. 论文的“安全/因果/不确定性”是否有与候选动作排序相关的证据？
3. CityFlow 能否用同根真实分支更便宜、更准确地提供该环节？若能，学习 WM 需要证明更快、覆盖更广，或对下游 teacher/student 更有价值。
4. 评价至少分成 prediction、candidate-ranking agreement、calibration/OOD 与 closed-loop traffic result；任何一项不能替代另一项。

## 6. 下一步执行顺序

1. 对表中“精读优先”论文下载正式 PDF，逐篇抽取 输入—动作条件—未来—决策用途—基线—失败条件 六个原子证据。
2. 做方法位置矩阵：每篇只标一个主要位置（transition / uncertainty / evaluator / planner / policy update / knowledge interface），防止把多个模块的堆叠当成一个创新。
3. 对多目标/Pareto 做定向 citation chasing；若找不到“future outcome vector + Pareto selection”的正式强基线，就把它记录为待验证缺口，而非直接宣布空白。
4. 最后才建立交通转译矩阵与最小证伪实验，不从这份候选清单直接跳到方法设计。
