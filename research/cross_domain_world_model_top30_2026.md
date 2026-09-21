# 30篇跨领域 World Model 顶会论文：为什么用、怎么用、关键图怎么读

检索与核验截止：2026-08-30。

## 先给结论

World Model 的独立价值不是“再加一个预测损失”，而是把原本的

观察 → 直接动作

改成

观察/当前状态 + 候选动作 → 预测该动作对应的未来 → 比较、训练或筛选 → 执行。

它尤其适合五类问题：

1. 真实试错昂贵、危险或不可逆，例如机器人、自动驾驶、医疗和网页操作。
2. 动作后果要跨很多步才能显现，当前观察不足以选动作。
3. 希望把同一状态下的多个候选动作放进“想象环境”比较。
4. 希望从离线视频、轨迹或旧经验中学习，而不是每个新任务都重新互动。
5. LLM 会提议或解释，但不了解当前具体环境的状态转移；需要外部 World Model 补上“做了会怎样”。

入选要求：

- 2024-01-01 至 2026-08-30 可核验的主会正式论文。
- 会议限于 ICLR、ICML、NeurIPS、CVPR、ICCV、ACL、EMNLP 等领域顶会主会。
- 论文必须明确建模状态/观察到未来的变化，并让预测服务于规划、策略训练、候选筛选、适应或有边界的评估。仅仅做普通预测或只把“world”写在标题里，不入选。
- “论文作者报告的效果”与本文的跨领域解释分开写。除非论文做了闭环规划/控制实验，预测更准不等于控制必然更好。

## 读每张关键图的统一方法

先在图中找四件事：

1. 当前信息：图像、状态、轨迹、规则或网页 DOM 在哪里进入。
2. 候选条件：动作、动作序列、轨迹、治疗方案或文本回复在哪里进入。
3. 想象未来：模型预测的是图像、潜变量、符号状态、图结构、奖励还是风险。
4. 未来如何被用掉：它是训练 policy、MPC 搜索、给候选打分、合成数据，还是只辅助表征。

只有第 4 步确实改变动作或训练数据时，才可以说 World Model 参与了决策；否则它只是预测/表征模块。

## A. RL、游戏与通用控制

| # | 论文 | 动机：为什么需要 World Model | 如何使用，以及关键图怎么读 | 实验边界 |
|---:|---|---|---|---|
| 1 | [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://openreview.net/forum?id=Oxh5CstDJU) — ICLR 2024 | 连续控制若只靠真实试错，采样成本高；单步价值又看不清长时后果。 | 图中把观察编码成 latent state；候选动作序列经 latent dynamics 预测未来 reward、终止与价值，MPC 选第一步动作。它是 World Model + MPC + value bootstrapping。 | 104个连续控制任务的强结果不等于能直接处理交通信号这种异构离散联合动作。 |
| 2 | [Diffusion for World Modeling: Visual Details Matter in Atari](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6bdde0373d53d4a501249547084bed43-Abstract-Conference.html) — NeurIPS 2024 | 传统离散 latent 可能丢掉影响决策的视觉细节。 | Fig. 1 纵向是扩散去噪、横向是环境时间：policy 给动作，扩散 World Model 依次生成下一帧，policy 在这些想象轨迹里训练。 | 作者在 Atari 100k 报告平均人类归一化分数 1.46；更逼真的画面本身仍不能证明每个任务的规划更可靠。 |
| 3 | [Learning World Models for Unconstrained Goal Navigation](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6cca3481ae66707958b824d37df40177-Abstract-Conference.html) — NeurIPS 2024 | replay buffer 只记录正向邻近转移，旧模型难以把任意两个子目标连接起来。 | Fig. 1 先给出普通 model-based RL；本文扩展为“任意 key state 到任意 key state”的可达性预测，再用这些连接规划探索目标。 | 验证的是新目标下的泛化；它强调世界模型的可达性，而不是大规模视觉生成。 |
| 4 | [Accurate and Efficient World Modeling with Masked Latent Transformers](https://proceedings.mlr.press/v267/burchi25a.html) — ICML 2025 | 低维 latent 容易漏掉关键信息；像素级模型又慢，无法高效想象。 | 关键方法图应读成：图像被压到空间 latent，MaskGIT 并行补全未来 latent，actor-critic 在生成的 latent 轨迹上学习。 | Crafter 上的成绩说明 latent 精度与速度可同时改善；Fig. 1 是成就数/FPS 对比，不是因果机制图。 |
| 5 | [Continual Reinforcement Learning by Planning with Online World Models](https://proceedings.mlr.press/v267/liu25p.html) — ICML 2025 | 连续 RL 会遗忘旧任务；把每个任务绑一套 dynamics 会让同一物理世界彼此冲突。 | Fig. 1 的红箭头表示同一状态动作在不同任务语义下的冲突；作者用统一的在线 dynamics，再以任意 reward 函数作 MPC。 | 优点是动态模型按构造避免遗忘；实验结论依赖其 Continual Bench 与浅层模型设定。 |
| 6 | [iVideoGPT: Interactive VideoGPTs are Scalable World Models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7dbb5bfab324e3b86af9bd0df15498dd-Abstract-Conference.html) — NeurIPS 2024 | 视频生成模型通常不能互动，难把互联网视频规模转成可控的 agent 环境。 | 总览图把图像、动作、奖励都 token 化；next-token 模型同时可做动作条件视频预测、视觉规划和 model-based RL。 | 论文报告在人类与机器人海量轨迹预训练后可迁移；token 化产生的误差会累积，长时可靠性仍需单独测。 |
| 7 | [GenRL: Multimodal-foundation World Models for Generalization in Embodied Agents](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3076133f08b40607d00a8f48f6acd71c-Abstract-Conference.html) — NeurIPS 2024 | 每个 RL 任务手工做 reward 不可扩展，通用 VLM 又与具身 dynamics 存在域差。 | 关键图要看两条表征如何对齐：VLM 的视觉/语言任务描述进入世界模型 latent，policy 在 imagined rollout 中学习。 | 支持“语言指定任务 + World Model 想象”的组合；不是证明任意语言目标都有可靠物理 grounding。 |

## B. 机器人、具身智能与 VLA

| # | 论文 | 动机：为什么需要 World Model | 如何使用，以及关键图怎么读 | 实验边界 |
|---:|---|---|---|---|
| 8 | [DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning](https://proceedings.mlr.press/v267/zhou25t.html) — ICML 2025 | 视觉 World Model 往往需要任务专用在线数据与逆模型，离线数据难泛化。 | Fig. 1(a) 用 DINOv2 patch feature 替代重建像素；Fig. 1(b) 以目标 feature 为终点，MPC 优化动作到达它。 | 六类环境展示无需专家示范和 reward model 的零样本目标到达；目标必须能被视觉 feature 表达。 |
| 9 | [Navigation World Models](https://openaccess.thecvf.com/content/CVPR2025/html/Bar_Navigation_World_Models_CVPR_2025_paper.html) — CVPR 2025 | 固定监督导航策略无法在测试时临时加入“不左转”等新约束。 | Fig. 1 上半部是“图像 + 导航动作 → 条件扩散 Transformer → 未来视图”；下半部对多条路径生成视频，再与目标图评分。 | 既可从零规划也可给专家轨迹重排；最终指标仍由真实导航环境判断。 |
| 10 | [IRASim: A Fine-Grained World Model for Robot Manipulation](https://openaccess.thecvf.com/content/ICCV2025/html/Zhu_IRASim_A_Fine-Grained_World_Model_for_Robot_Manipulation_ICCV_2025_paper.html) — ICCV 2025 | 操作中动作与接触/物体变化必须精确对齐，粗粒度视频难用于比较候选轨迹。 | Fig. 1 左侧为历史观察与输入轨迹，中间 IRASim 是帧级动作条件扩散模型，右侧为预测/真实视频；预测用来评估或测试时规划。 | 作者报告 Push-T 规划 IoU 从 0.637 到 0.961；这不自动覆盖遮挡、长时接触和真实机器人分布外动作。 |
| 11 | [GWM: Towards Scalable Gaussian World Models for Robotic Manipulation](https://openaccess.thecvf.com/content/ICCV2025/html/Lu_GWM_Towards_Scalable_Gaussian_World_Models_for_Robotic_Manipulation_ICCV_2025_paper.html) — ICCV 2025 | 2D 图像 World Model 缺少稳定几何，难理解三维接触和物体位置。 | 关键图应沿“高斯场景 + 机器人动作 → 未来高斯 primitives → 渲染未来场景 → policy/model-based RL”读；模型同时做表征预训练和神经模拟器。 | 论文含仿真与真实操作；三维重建精度仍是动作后果可信度的前提。 |
| 12 | [ReCoRe: Regularized Contrastive Representation Learning of World Model](https://openaccess.thecvf.com/content/CVPR2024/html/Poudel_ReCoRe_Regularized_Contrastive_Representation_Learning_of_World_Model_CVPR_2024_paper.html) — CVPR 2024 | 视觉导航易把纹理/外观当成 dynamics，导致样本效率低和外观变化下失效。 | Fig. 1 上部是 contrastive loss，下部的深度等辅助任务约束 latent 对外观干预不变；World Model 用这个稳健 latent 服务 RL。 | iGibson 的 OOD 导航与 sim-to-real 感知增益证明表征更稳健，不是在线候选动作排序证据。 |
| 13 | [DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge](https://proceedings.neurips.cc/paper_files/paper/2025/hash/22d4f952efa13970f0b1ffb22170d416-Abstract-Conference.html) — NeurIPS 2025 | 直接图像预测冗余且难显式保留动态、空间和语义线索，VLA 由此难做动作推理。 | 总览图是 perception → 三类 world knowledge forecast → inverse dynamics/action diffusion；预测的紧凑知识而非整帧图像进入动作计划。 | 真实机器人和 CALVIN 实验表明该组合可用；它更接近“预测辅助动作”，不是显式候选 rollout 评估器。 |

## C. 自动驾驶与医疗

| # | 论文 | 动机：为什么需要 World Model | 如何使用，以及关键图怎么读 | 实验边界 |
|---:|---|---|---|---|
| 14 | [DriveWorld: 4D Pre-trained Scene Understanding via World Models for Autonomous Driving](https://openaccess.thecvf.com/content/CVPR2024/html/Min_DriveWorld_4D_Pre-trained_Scene_Understanding_via_World_Models_for_Autonomous_CVPR_2024_paper.html) — CVPR 2024 | 2D/3D 预训练忽略驾驶场景的时间演化，特征难同时覆盖检测、地图、跟踪和规划。 | 关键图对比 2D、3D 与其 4D 预训练；动态 memory 预测未来变化、静态传播保留场景上下文，task prompt 分流下游任务。 | 它主要证明 4D 表征预训练提升多下游指标；不能直接等同于在线模拟器。 |
| 15 | [Driving into the Future: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Driving_into_the_Future_Multiview_Visual_Forecasting_and_Planning_with_CVPR_2024_paper.html) — CVPR 2024 | 单视角视频看不全 3D surroundings，且反应式 planner 无法预先比较不同驾驶机动的风险。 | 图中多视角历史和候选 maneuver 进入 Drive-WM，输出多个一致的未来视频；图像 reward 为轨迹打分并选最优。 | 这是“候选未来评价器”最直观的驾驶范式；生成质量和奖励设计都可能成为规划失真来源。 |
| 16 | [Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a6a066fb44f2fe0d36cf740c873b8890-Abstract-Conference.html) — NeurIPS 2024 | 既有驾驶视频模型在新场景、关键细节和多层动作控制上不够可靠。 | 关键图应找三种控制入口：高层 command/goal point 和低层 trajectory/angle/speed；它们条件化长时视频 rollout，也可提供动作评价 reward。 | 论文报告 FID/FVD 与动作评价提升；视频逼真度仍非安全闭环的充分证据。 |
| 17 | [End-to-End Driving with Online Trajectory Evaluation via BEV World Model (WoTE)](https://openaccess.thecvf.com/content/ICCV2025/html/Li_End-to-End_Driving_with_Online_Trajectory_Evaluation_via_BEV_World_Model_ICCV_2025_paper.html) — ICCV 2025 | 多模态驾驶 policy 能提出多条轨迹，但若不看执行后的后果，无法可靠选出安全的一条。 | Fig. 1 上图是普通 sensor → BEV → policy → trajectory；下图加入 BEV World Model，分别预测每条轨迹的未来 BEV，再由 reward model 选择。 | 在 NAVSIM 与 Bench2Drive 闭环验证；它清楚分开“预测后果”和“用 reward 评价后果”。 |
| 18 | [Epona: Autoregressive Diffusion World Model for Autonomous Driving](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Epona_Autoregressive_Diffusion_World_Model_for_Autonomous_Driving_ICCV_2025_paper.html) — ICCV 2025 | 高分辨率、长时驾驶生成容易误差累积，传统 rollout 难同时满足清晰和连续。 | 关键图要找 autoregressive diffusion 的长序列前滚，以及 chain-of-forward training 如何把预测误差回灌训练；生成模型也作实时 motion planner。 | 作者报告 FVD 改善与 NAVSIM planner 对比；长视频仍需用真实安全/规划指标判断。 |
| 19 | [World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model](https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_World4Drive_End-to-End_Autonomous_Driving_via_Intention-aware_Physical_Latent_World_Model_ICCV_2025_paper.html) — ICCV 2025 | 端到端驾驶通常依赖昂贵的感知标注，单一 latent 又难表示不同驾驶意图。 | Fig. 1 是收敛效率图；方法图应沿“多视角 → 物理 latent + 意图 → 多个意图未来 latent → selector 排序候选轨迹”读。 | nuScenes 与闭环 NavSim 证明标注更少仍能规划；其自监督 future alignment 是否在极端长尾事件稳定仍待测。 |
| 20 | [HERMES: A Unified Self-Driving World Model for Simultaneous 3D Scene Understanding and Generation](https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_HERMES_A_Unified_Self-Driving_World_Model_for_Simultaneous_3D_Scene_ICCV_2025_paper.html) — ICCV 2025 | 传统 Driving WM 会生成未来但不理解场景；驾驶 LLM 会解释场景却不推演未来。 | Fig. 1(a–d) 从“只生成”与“只理解”过渡到统一模型：BEV 特征加 world queries，经 LLM 同时回答场景问题和生成动作条件未来。 | 证明 generation 与理解可共享表示；并未单独证明该联合模型的闭环驾驶控制优于专用 planner。 |
| 21 | [Medical World Model](https://openaccess.thecvf.com/content/ICCV2025/html/Yang_Medical_World_Model_ICCV_2025_paper.html) — ICCV 2025 | 治疗的价值取决于病灶在治疗之后如何演化，当前影像不能直接告诉医生哪种治疗后果更好。 | Fig. 1 左侧影像经 perception 成初始状态；policy 给治疗候选，progression generator 生成治疗后肿瘤，survival/inverse dynamics 再评分并反馈选治疗。 | 这是 World Model + VLM + survival analysis；应视作临床决策支持研究，不是可直接替代医生的因果疗效证明。 |

## D. LLM、代码与 Web Agent

| # | 论文 | 动机：为什么需要 World Model | 如何使用，以及关键图怎么读 | 实验边界 |
|---:|---|---|---|---|
| 22 | [WorldCoder, a Model-Based LLM Agent](https://proceedings.neurips.cc/paper_files/paper/2024/hash/820c61a0cd419163ccbd2c33b268816e-Abstract-Conference.html) — NeurIPS 2024 | LLM 直接在脑中模拟世界既难审计又要反复调用；深度 RL 学到的模型不易修改和迁移。 | Fig. 1 是 world → state/reward/goal → planner 的闭环；中间黑框是 LLM 写出的 Python transition function，planner 调代码而不是每次问 LLM。 | 在 gridworld 与任务规划上比较样本/计算效率；假设环境可被确定性的可执行代码近似。 |
| 23 | [Generating Code World Models with Large Language Models Guided by Monte Carlo Tree Search](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6f479ea488e0908ac8b1b37b27fd134c-Abstract-Conference.html) — NeurIPS 2024 | 正确的代码 World Model 需要理解规则、写精确逻辑并依据单元测试/轨迹自我修复。 | 关键图是 GIF-MCTS：生成候选代码 → 用测试和环境轨迹评估 → 改进/修复 → 让 planner 调用通过的代码模型。 | 在 CWMB 与其他基准上测规划；离散文本规则环境与连续真实物理差别很大。 |
| 24 | [WebEvolver: Enhancing Web Agent Self-Improvement with Co-evolving World Model](https://aclanthology.org/2025.emnlp-main.454/) — EMNLP 2025 | Web agent 用自己采样的数据训练会越走越窄，真实网页探索成本高且不稳定。 | Fig. 1 上半部让 world-model LLM 充当虚拟 Web server 合成训练轨迹；下半部让 policy 的多个候选动作经预测网页状态评分后选动作。 | 在 Mind2Web-Live、WebVoyager、GAIA-web 上验证；网页状态预测错误会被 policy 放大，需看真实网页回放。 |
| 25 | [Model-Based Imaginative Planning for Embodied Agents (IMPLEMENT)](https://aclanthology.org/2026.acl-long.827/) — ACL 2026 | 冻结 LLM 没见过当前房间的真实 affordance，又只能从稀疏图像观察，难可靠规划。 | Fig. 1：图像 → object-centric symbolic state → 对候选动作作 Monte-Carlo 未来状态采样 → belief summary → 冻结 LLM 反复评估/修正。 | ALFWorld 对比微调与 test-time scaling；模拟的不确定性通过采样表示，但现实机器人迁移仍未由该实验建立。 |
| 26 | [WALL-E: World Alignment by NeuroSymbolic Learning improves World Model-based LLM Agents](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5e772a13ccba5255331240dcd99aa38b-Abstract-Conference.html) — NeurIPS 2025 | LLM 的通识先验与当前环境规则脱节，把 LLM 当 WM 会预测错局部 dynamics。 | 关键图为探索轨迹 → action rule/knowledge graph/scene graph → 可执行神经符号 World Model；LLM 作为 MPC 的 look-ahead optimizer。 | Mars 与 ALFWorld 的成功强调“先校准世界知识”；训练免费不代表无需探索数据。 |
| 27 | [WebSynthesis: World Model-Guided Monte Carlo Tree Search for Efficient WebAgent Trajectory Synthesis](https://aclanthology.org/2026.acl-long.1157/) — ACL 2026 | 网页访问受网络/权限限制，且点击常不可逆，现实中很难做树搜索收集多样轨迹。 | Fig. 1 先展示合成与真实网页状态差异；系统图则是 LLM World Model 预测 accessibility-tree 状态转移，MCTS 在模型内自由回退搜索。 | 在 WebArena、WebVoyager、Mind2Web-Online 评估合成轨迹训练；预测页面结构错误时仍需真实环境校验。 |

## E. 多智能体、社会系统与图结构数字世界

| # | 论文 | 动机：为什么需要 World Model | 如何使用，以及关键图怎么读 | 实验边界 |
|---:|---|---|---|---|
| 28 | [Learning and Planning Multi-Agent Tasks via an MoE-based World Model (M3W)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3a2d96d2eb2902043c2db705ca03e9a2-Abstract-Conference.html) — NeurIPS 2025 | 多任务多智能体的最优 policy 差异很大，但有些任务的 dynamics 又可复用，单一 policy 易发生梯度冲突。 | 关键图把 MoE 放在 dynamics 而不是 policy：SoftMoE 学相近任务、SparseMoE 避免无关任务干扰，随后在预测 rollout 上优化动作。 | Bi-DexHands 与 MA-Mujoco 的结果支持任务动态模块化；专家路由错误会直接影响规划。 |
| 29 | [Social World Model-Augmented Mechanism Design Policy Learning](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a21db07ccc247b6383f78939c8f894c7-Abstract-Conference.html) — NeurIPS 2025 | 税制、设施选址、团队协作中，个体有长期隐变量偏好/能力，真实社会互动又昂贵。 | 总览图应读为：历史交互 → 推断 agent traits → 预测不同机制下的回应 → 在 social World Model 中产生训练轨迹 → 改进机制 policy。 | 论文在税收、协作与选址环境中比较 model-based/model-free RL；“社会 trait”可辨识性是关键假设。 |
| 30 | [Graph World Model](https://proceedings.mlr.press/v267/feng25p.html) — ICML 2025 | 既有 World Model 偏非结构化 token，难利用推荐、知识图、RAG、网络和多智能体任务里的显式关系。 | 关键图中状态是一张含文本/图像/embedding 的图；action node 可以直接指向节点/边，也可由相似度连接；message passing 产生预测/生成/规划结果。 | 六类任务展示统一图接口的迁移能力；它不是为单一物理控制问题设计的强闭环 simulator。 |

## 10张最值得先精读的关键图

### 1. DIAMOND，Fig. 1：World Model 怎样“替环境”训练 RL

横向是连续环境时间，纵向是扩散去噪时间。policy 先在想象中给出动作，扩散模型在动作条件下把噪声变成下一帧；生成帧和动作再成为下一时刻的条件。它说明 World Model 在这里不是辅助打分，而是一个可让 policy 反复练习的虚拟环境。

### 2. Navigation World Models，Fig. 1：同一个视频模型如何从生成变成规划

上半图是学习阶段：图像和位移/转向动作输入条件扩散模型。下半图是推理阶段：同一当前视图下生成多条候选路径的视频，把各自末帧与目标图比较，得分高的路径被保留。它完整展示了“候选动作 → 未来 → 选择”。

### 3. IRASim，Fig. 1：为什么机器人必须把动作与帧对齐

左边每条彩色曲线是关节/末端轨迹，右边每一帧都要对应这条轨迹的正确时刻。中间不是“随便生成一个看似合理的视频”，而是 frame-level action conditioning；否则夹爪早一点或晚一点接触都会让操作候选的评价失真。

### 4. WoTE，Fig. 1：最清晰的“World Model 不等于奖励函数”分工

普通端到端驾驶只产一条或多条轨迹。WoTE 先让 BEV World Model 分别回答“执行这条轨迹，未来 BEV 会怎样”，再由 Reward Model 回答“这个未来好不好”，最后选轨迹。预测、评价、选择是三件不同的事。

### 5. Medical World Model，Fig. 1：把医疗决策改成可比较的反事实后果

影像形成当前病灶状态；不同治疗计划是候选动作；肿瘤 progression generator 给出每种治疗后的病灶；survival analysis 将后果转成可比较的效用。图表达的是临床决策支持链，而不是模型已经证明治疗的真实因果效果。

### 6. WorldCoder，Fig. 1：LLM 不直接模拟，而是写一个可运行的世界

LLM 位于右侧，负责根据经验更新 Python world_model.py；planner 位于左侧，反复调用该程序预测 state、reward、goal，再把选定动作施加到真实世界。关键优点是模型可编辑、可测试、可复用。

### 7. WebEvolver，Fig. 1：一个 World Model 同时有两份工作

上框用 World Model 充当虚拟网页服务器，生成训练轨迹；下框让 agent 提出多个网页动作，World Model 预测各自后续页面，再选择动作。前者解决数据稀缺，后者解决推理时只看一步的问题。

### 8. IMPLEMENT，Fig. 1：LLM 和 World Model 的职责边界

视觉输入先变为 object-centric symbolic state。World Model 负责把候选动作展开成多个可能未来，belief summary 压缩这些未来；冻结 LLM 只读摘要做评估和修正。这样 LLM 不被要求凭语言先验发明物理 dynamics。

### 9. HERMES，Fig. 1：驾驶生成与语言理解为什么不能只是并排拼接

前三个子图依次是只生成、只理解、两个独立模块共享 feature；最后一个子图才是统一模型。它要表达的不是“加了 LLM 就能开车”，而是同一个 BEV world representation 同时支持对未来的生成和对当前场景的语言理解。

### 10. Social World Model，系统总览：把“人的差异”放入未来预测

机制 policy 改变后，各 agent 的反应不只由当前状态决定，也由未直接观测的能力/偏好决定。图中 trait inference 是关键桥梁：若 trait 估错，模拟出的社会后果与真实机制效果都会错。

## 跨领域的七种结合方式

| 结合方式 | World Model 扮演的角色 | 代表论文 | 不能混淆的点 |
|---|---|---|---|
| World Model + model-based RL | 在想象轨迹中训练 actor/critic，或作 MPC | TD-MPC2、DIAMOND、EMERALD | 预测误差小不自动保证 policy 好。 |
| World Model + video/diffusion | 生成动作条件未来画面/场景 | DIAMOND、NWM、IRASim、Vista、Epona | 视觉 FVD/FID 不是安全或控制指标。 |
| World Model + VLA/LLM | 给语言模型补当前环境的可执行 dynamics | GenRL、DreamVLA、IMPLEMENT、WALL-E | LLM 提议动作不等于 LLM 能预测后果。 |
| World Model + candidate ranking | 对同根状态的多个候选后果打分/筛选 | NWM、Drive-WM、WoTE、World4Drive | 预测、评价函数、最终选择必须分开验证。 |
| World Model + synthetic data | 先在模型里造可用轨迹，再训练真实 agent | WebEvolver、WebSynthesis、SWM-AP | 合成数据需要用真实环境回放/对齐审计。 |
| World Model + code/graph/symbolic state | 让 dynamics 可解释、可编辑、可组合 | WorldCoder、GIF-MCTS、Graph World Model | 结构化表示带来规则/接口假设，未必适合高噪声连续物理。 |
| World Model + uncertainty/adaptation | 处理域变、任务变或不完整观察 | DINO-WM、Continual WM、M3W、IMPLEMENT | 不能把 ensemble/采样不确定性直接称为已校准安全保证。 |

## 最重要的迁移结论

跨领域真正稳定的抽象是：

当前状态 + 合法候选动作
→ 动作条件未来
→ 对未来做任务相关的比较
→ 只执行当前第一步
→ 观察真实结果后重新规划。

机器人、驾驶、医疗和网页都在解决“先做错的代价太高”或“后果太晚才看得见”。RL 侧常把 World Model 当虚拟训练场/MPC；LLM 侧常把它当补足局部规则和状态变化的外部工具；视频侧常解决如何把未来生成为可控、长时、一致的表示。

但同样重要的是边界：World Model 只回答“做这个会发生什么”；reward、规则、风险约束、偏好模型或人类决策者才回答“这个未来是否值得选”。任何跨领域方案都应把预测准确性、候选排序一致性与真实闭环任务效果分开报告。
