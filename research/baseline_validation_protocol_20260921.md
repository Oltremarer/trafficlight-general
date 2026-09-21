# CityFlow baseline 下一步验证配置（2026-09-21）

状态：本文件是待执行计划；本轮没有启动新训练。所有修改和实验均针对 `rl-trafficlight`，原 `world model traffic` 不动。

## 1. 已有证据与本轮目标

5090 已完成的真实 CityFlow 运行位于：

`/mnt/pan/rl-trafficlight/runs/20260921T150511Z_cityflow_smoke`

对应源码快照：`/mnt/pan/rl-trafficlight/source_snapshots/20260921T150511Z`。

已有证据为 315 项测试通过、16 个已注册 profile 的 300 秒短训练及独立重载评估通过、4 个代表性 learner 的续训通过，以及 FixedTime/MP 运行通过。16 个 checkpoint 重载后的动作、奖励和 baseline 观测与首次评估一致。这证明当前执行链能运行，不证明 100 回合训练后的性能或论文数值已经复现。

下一步直接执行原计划第一批：**MPLight、CoLight、E-CoLight、A-CoLight + FixedTime、MaxPressure，五组流量，100 回合，5 个独立训练种子**。不另加短程试跑，不以必须超过 MP 作为继续运行的条件。

本批命名为 `core_unified_v1`：本地 PyTorch 移植、统一环境交互预算、保留各 profile 的观测与奖励。它不是“原论文超参数复现”。

## 2. 数据绑定

以下别名是本实验明确指定的文件映射，不根据城市名或文件名数字推定论文数据一致。

| 本批别名 | 路网 | 流量文件 | 路口数 | 计划发车数 |
|---|---|---|---:|---:|
| Jinan1 | roadnet_3_4.json | anon_3_4_jinan_real.json | 12 | 6295 |
| Jinan2 | roadnet_3_4.json | anon_3_4_jinan_real_2000.json | 12 | 4365 |
| Jinan3 | roadnet_3_4.json | anon_3_4_jinan_real_2500.json | 12 | 5494 |
| Hangzhou1 | roadnet_4_4.json | anon_4_4_hangzhou_real.json | 16 | 2983 |
| Hangzhou2 | roadnet_4_4.json | anon_4_4_hangzhou_real_5816.json | 16 | 6984 |

已在 5090 核对原文件：Jinan 位于 `/home/chenyuyang/C2T/data/Jinan/3_4`，Hangzhou 位于 `/home/chenyuyang/C2T/data/Hangzhou/4_4`。五个文件的每条记录均为 `startTime == endTime`，且发车时刻均在 `[0,3600)`，所以此处一条记录对应一辆计划车辆；不能把 `_5816` 当作本文件的车辆数。

正式运行时复制到拟定的 `/mnt/pan/rl-trafficlight/datasets/core-v1/{Jinan,Hangzhou}/`，核对以下 SHA256，再由所有方法共用；这一步尚未执行。

```text
roadnet_3_4.json                   55abc036ac4ac48705301f4cf46ea80d8d88a82abb9ed68404bedb96eb3cf81d
anon_3_4_jinan_real.json            233739633ef0b637125cb304dfffff9488503bac6e6861ca39243cc8ffdcebd5
anon_3_4_jinan_real_2000.json       d0931c1b759479f9e748d69c16414020bfba03d555dcc6cdf7223a0c18cb9e69
anon_3_4_jinan_real_2500.json       4245107cc7ce91b9699519f2c1258be4397e83291080a43cb0b93479557739cd
roadnet_4_4.json                   11e2fe89f632e43e81f56ea87a308d544b66f9c5af8a410e16149668d6d376c1
anon_4_4_hangzhou_real.json        595a6140e6649efe5be1b274eb8b03e8c4029e517c9ac2ad53c9ea556a414213
anon_4_4_hangzhou_real_5816.json   e9b9e31e9a9a5f59d1a668b242319c3ef69f1ab724e2f2b618ed4886cc1a7a9c
```

训练与评估使用同一个固定流量文件，各场景单独训练。这是固定需求场景的训练后控制效果比较；评估种子不同不代表获得了新需求分布，不能据此声称跨流量或跨城市泛化。

## 3. 环境与动作配置

| 配置 | 固定值及含义 |
|---|---|
| duration | 3600 秒；不在主实验末尾额外清空路网 |
| simulator_step | 1 秒 |
| decision_interval | 30 秒，共 120 次整网决策/回合 |
| green_phases | CityFlow 相位 ID `(1,2,3,4)`；策略选择本地合法动作，经过映射执行 |
| yellow_phase_id / yellow_time | 0 / 5 秒 |
| all_red_time | 0 秒，无全红阶段 |
| 保持相位 | 连续绿灯 30 秒 |
| 切换相位 | 5 秒黄灯 + 25 秒新绿灯，合计 30 秒 |
| 初始相位 | 每路口第一个合法相位；无额外 warm-up 仿真 |
| CityFlow | `thread_num=1`，`laneChange=False` |
| 等待速度阈值 | AWT 采集器 `<=0.1 m/s`；引擎队列使用引擎自身计数，不能混称同一个可调阈值 |
| horizon | `truncated=True`，本批显式使用 `bootstrap_truncated=True`，维持当前继续任务价值定义 |

黄灯占用是决策间隔的一部分，不是“30 秒绿灯之后再加 5 秒”。黄灯期间的奖励和评估统计照常累计。

FixedTime：按本地合法相位顺序循环，每 30 秒换一次，相位偏移为 0。MP：按当前实现的 movement queue pressure 求相位得分，最大值并列时取最小本地索引。这两个控制器使用与 RL 相同的数据、时序和指标采集器；这些规则也要写入结果元数据。

## 4. 第一批四个 RL 的训练配置

| CLI 参数 / 行为 | 本批值 |
|---|---|
| baseline | `mplight / colight / e-colight / a-colight` |
| episodes | 100；每次独立创建模型、优化器及 replay |
| device | `cuda` |
| hidden_dim | 64 |
| optimizer / learning_rate | Adam / 0.001 |
| gamma | 0.8，按 30 秒 transition 折扣 |
| reward_scale | 0.05，学习目标中只乘一次 |
| batch_size | 32 个整网 transition，不是 32 个单路口样本 |
| replay_capacity | 10000 个整网 transition，均匀无放回抽 batch |
| warmup_transitions | 32 |
| update_schedule / updates_per_round | round / 每回合结束更新 10 次 |
| target_update_steps | 每 100 次梯度更新同步 target；本批相当于每 10 回合一次 |
| epsilon_start / epsilon_end | 0.8 / 0.05 |
| epsilon_decay_steps | 10000 次整网环境决策，线性衰减，之后保持 0.05 |
| gradient_clip | 10 |
| TD / loss | vanilla DQN 的合法动作 max target / MSE |
| bootstrap_truncated | true；真正 terminated 才停止 bootstrap |

每个独立训练应产生 **12000 次整网环境决策、1000 次梯度更新**。它是明确的首版预算，不是已调优配置，也不预先承诺已经收敛。loss 降低不能单独证明交通控制改善。

各方法的区别继续由现有 profile 保留：

| 方法 | 观测 | 原始局部奖励 |
|---|---|---|
| MPLight | phase8 + general vehicle pressure12；相位竞争网络使用对应 L/T 需求 | `-0.25 * abs(入口排队总数 - 出口排队总数)` |
| CoLight | phase8 + vehicle count12 | `-0.25 * 入口排队总数` |
| E-CoLight | phase8 + efficient queue pressure12 | 同 CoLight |
| A-CoLight | E-CoLight 输入 + running12；距车道末端 167m 内、速度 >0.1m/s | 同 CoLight |

四者共享路口参数；reward 按每个仿真 tick 推进前的值做 30 秒区间均值（`pre_mean`）。上述奖励再乘 learner 的 0.05；不要在环境里重复缩放。CoLight 系列用自己加四个最近路口的几何 kNN（K=5），5 个 attention head，宽度 64 对应 head_dim=32，head 输出取均值。相位编码与 source/local 动作映射需随 schema 保存。

不把四个方法的训练 reward 当作统一效果指标，它们的奖励定义不同；统一比较由独立交通指标采集器完成。

## 5. 种子、checkpoint 和冻结评估

| 用途 | 固定配置 |
|---|---|
| 5 个独立训练种子 | `0,1000,2000,3000,4000` |
| 每个模型的训练 episode 种子 | 当前 runner 使用 `training_seed + episode_index`，index 为 0..99 |
| 曲线 checkpoint | 完成第 `10,20,30,40,50,60,70,80,90,100` 回合 |
| 曲线评估种子 | `9000`，同一流量，deterministic；只画曲线，不据此挑模型 |
| 最终评估种子 | `10000,10001,10002`，所有方法共用 |
| 主表选模 | 完成第 100 回合后的 checkpoint，不挑最好一轮或最好种子 |
| 规则控制器 | 每个场景运行相同三个最终评估种子；没有“训练种子” |

训练种子使用间隔 1000 的编号，是为避免当前 `seed + episode` 规则使五次训练的仿真 seed 区间重叠。每个 seed 都必须从头初始化；一次训练调用里的三个 `--eval-seeds` 不等于三个独立训练。

当前文件名使用从 0 开始的回合编号：完成第 10 回合是 `episode_0009.pt`，完成第 100 回合是 `episode_0099.pt`。以 sidecar 的 `completed_episodes` 为准，避免少跑一轮。

保留这十个完整 checkpoint，包括 optimizer、target、replay、随机数状态和 protocol/hash。学习曲线可在训练结束后，用现有 `evaluate` 命令依次加载它们生成，避免插入评估影响训练随机数状态。冻结评估禁用探索，不写入 replay、不更新参数；最终三次评估已有 train CLI 自动执行路径。

续训使用新输出目录，沿用训练 seed 和全部协议。当前 `--episodes` 在 resume 时表示**额外运行回合数**：从 completed=60 恢复到 100 应传 `--episodes 40`，不能再传 100。汇总时按模型标识、completed episode 合并原运行和续训，不重记已有评估。

统计先对一个模型的三个最终评估结果取均值，再对五个独立模型计算 mean ± sample SD（`ddof=1, n=5`）。保存全部 15 个原始结果，但不能将它们作为 15 次独立训练。当前 CLI 的 aggregate 是同一个模型跨评估 seed 的总体标准差，不能直接拿来填主表。规则控制器按三次仿真评估分别汇总，并标明重复类型；确定性流量出现零波动应如实报告。

## 6. 正式批量运行所需的最小支撑改动

这些是计划中的改动，本轮未实现；不修改算法网络或更新公式。

1. **车辆生命周期指标与耗时记录。** 在现有 snapshot/MetricCollector 路径记录计划发车、实际进入、完成、结束时在网、未进入的车辆数，区分引擎 waiting buffer 与在网车辆。核对“计划 = 完成 + 在网未完成 + 未进入”的含义与计数。当前 `throughput_vehicles` 只按活跃 ID 消失推断，不能未经语义核对直接当作精确完成车辆数。记录训练总 wall time、评估 wall time、每次整网 `act` 的耗时；CUDA 时间包含必要同步，说明计时范围。
2. **checkpoint 保存周期。** 为 train CLI 增加可选 `--checkpoint-every 10`，保证最终 checkpoint 总会保存，并保持旧默认行为。当前是每回合保存完整 replay；100 次训练会生成 10000 份完整 checkpoint，本批只需 1000 份。覆盖周期保存、最终保存、恢复后编号与 remaining episodes 的相关测试。
3. **批量启动和跨训练种子汇总。** 封装现有 train/evaluate/rule CLI，生成唯一任务目录、记录退出状态、支持跳过已完成任务及按边界 checkpoint 恢复，计算上述两级统计。验证任务矩阵无重复、seed 分类正确、缺失运行不会被当作零分或静默丢弃。

只对改动的指标、保存逻辑、任务聚合增加必要测试，使用现有 fixture；不为这些改动额外发起一轮短程 RL 训练。原有 315 项结果不作为新改动自动通过的证据。

## 7. 指标口径

主结果同时报告：`ATT_engine`、完成数/计划发车数、完成率、在网未完成数、未进入数、AWT、平均全网排队车辆数。保存原始分母和每次运行的统计，避免只有一个 ATT。

- `ATT_engine` 对应当前 `get_average_travel_time()` 返回值，元数据保留引擎二进制 hash。服务器上的 CityFlow 源码把已完成累计时长与仍在 vehiclePool 中车辆的当前累计时长共同求平均；它不是“只完成车辆的平均行程时间”。源码存在不等于已经证明安装二进制由该 checkout 编译，需保留这个来源边界。
- 完成率明确为 `N_finished / N_scheduled`；未进入车辆与在网未完成车辆分别列出。补充 `ATT_completed` 时，只在确有完成车辆时计算，否则为 null 并注明原因，不能填 0。二者不能互换命名。
- 当前 AWT 是所有观测到车辆在 3600 秒内累积的停止时长均值，未完成车辆只累计已发生的等待；不是仅完成车辆 AWT，也不包含尚未被观测到的车辆。
- 当前 queue 指每秒所有引擎 lane waiting count 之和再按时间平均，是全网总量，不是每路口均值。比较不同路网时同时列路口数，必要的归一化列另起名字。
- 学习曲线使用冻结评估的 ATT/完成率/queue，训练 loss 和 epsilon 作为诊断单独展示。不能将带探索的训练 reward 曲线作为最终效果曲线。

LLMLight 固定版本 `utils/model_test.py` 的 ATT 通过各路口车辆 enter/leave 日志汇总，缺失 leave 用 episode 截止时间补齐；它与当前直接调用 engine API 的实现路径不同。在证明两者统计集合、时间起止一致之前，不直接拿本批 ATT 对论文表计算提升。

## 8. 5090 执行配置与任务数量

继续使用已验证的 Python：`/home/chenyuyang/miniconda3/envs/c2t/bin/python`，Python 3.9.25、PyTorch 2.8.0+cu128、NumPy 1.26.2、RTX 5090。当前 CityFlow 扩展 SHA256：`8d7068117efd7efc4b3bd41ed642dfe06675e5620a4f5f8fbf855b7fb43b183a`。启动时记录实际环境，不根据旧清单假定环境未变化。

计划新建的运行根目录模板：

`/mnt/pan/rl-trafficlight/runs/<UTC>_core_unified_v1/`

这是模板，尚未创建。执行时先确认 `/mnt/pan` 已挂载且可写，创建并报告实际绝对路径，再启动；不可用就停止，不回退到系统盘。目录按 `scenario / method / train_seed / attempt` 区分，checkpoint、log、metric、trajectory、tmp 和所有缓存均放在该根目录下。原 smoke 输出保持不动。

- 建议并发上限 2 个作业；两者均 `CUDA_VISIBLE_DEVICES=0`，每作业 CityFlow thread=1。
- `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`。
- `PYTHONDONTWRITEBYTECODE=1`，`TMPDIR/XDG_CACHE_HOME/TORCH_HOME/CUDA_CACHE_PATH/TRITON_CACHE_DIR` 指向各作业目录。资源使用信息是运行诊断，不是最终性能指标。
- 在上述小改动后生成新的源码 snapshot 与文件 SHA256 manifest；目前源码含未提交实现，不能只记录初始复制仓库的 Git HEAD 就认为锁定了代码。

| 项目 | 数量 |
|---|---:|
| 完整 RL 训练 | 4 方法 × 5 流量 × 5 训练种子 = 100 次 |
| 训练 episode | 100 × 100 = 10000 回合 |
| 曲线冻结评估 | 100 模型 × 10 checkpoint × 1 seed = 1000 回合 |
| 最终冻结评估 | 100 模型 × 3 seed = 300 回合 |
| FixedTime/MP | 2 方法 × 5 流量 × 3 seed = 30 回合 |

不根据 300 秒短运行线性推算总工期；3600 秒后段拥堵、经验与轨迹写盘均可能改变耗时。

当前 CLI 支持的单任务参数示例（路径是待替换模板，非本轮执行记录）：

```bash
PYTHONPATH="$RL_SOURCE/src" "$RL_PYTHON" -m cityflow_tsc.train_baseline train \
  --baseline a-colight \
  --roadnet "$RL_DATA/Jinan/roadnet_3_4.json" \
  --flow "$RL_DATA/Jinan/anon_3_4_jinan_real.json" \
  --output "$RL_RUN/Jinan1/a-colight/train_seed_0/attempt_0" \
  --duration 3600 --episodes 100 --seed 0 --eval-seeds 10000,10001,10002 \
  --device cuda --thread-num 1 \
  --decision-interval 30 --simulator-step 1 --yellow-time 5 --all-red-time 0 \
  --yellow-phase-id 0 --green-phases 1,2,3,4 --waiting-speed-threshold 0.1 \
  --hidden-dim 64 --learning-rate 0.001 --gamma 0.8 --reward-scale 0.05 \
  --batch-size 32 --replay-capacity 10000 --warmup-transitions 32 \
  --updates-per-round 10 --target-update-steps 100 \
  --epsilon-start 0.8 --epsilon-end 0.05 --epsilon-decay-steps 10000 \
  --gradient-clip 10 --bootstrap-truncated
```

`RL_SOURCE` 是新源码快照，`RL_DATA` 是 `/mnt/pan` 下冻结的数据目录，`RL_RUN` 是实际已创建的 run root，`RL_PYTHON` 是上述 c2t Python。命令刻意只列当前支持的参数；保存周期功能完成后，批量命令再增加新参数 `--checkpoint-every 10`。当前没有 `--eval-every` 或 JSON 批配置读取入口，不应提前使用这些不存在的参数。

## 9. 何时可以下结论，以及如何扩展

运行完成与方法有效分开判断：任务实际结束、交互/更新次数符合配置、checkpoint 和数据 hash 可追溯、指标及更新数值有限、全部种子纳入汇总，才称本批执行完成。最终 ATT 与完成率是否改善、是否稳定由真实结果回答；性能差的种子也保留。出现 NaN、非法动作、数据或 checkpoint 不匹配时，停止并标记受影响任务，保留日志，不悄悄重采 seed。普通性能不佳不触发自动调参或追加训练。

之后按原计划补第二批，不把已注册 profile 数量当作计划覆盖数量：

- 第二批现有可接入的 9 项：`idqn, shared-dqn, presslight, e-presslight, frap, e-mplight, a-mplight, ippo, mappo`。加第一批为 13 项；补 `attendlight` 才到固定相位目标 14 项。
- `idqn/shared-dqn/frap` 为 step 更新：同样 100 回合预算下预期 11969 次优化事件，不能写成与 round-Q 相同的 1000 次；独立路口网络每次事件还会更新多个网络。统一的是环境交互量，同时公开优化量和 wall time。
- `presslight/e-presslight/e-mplight/a-mplight` 沿用 round-Q 配置，保留各自 profile 的观测及奖励。
- IPPO/MAPPO 另用 `learning_rate=0.0003, gamma=0.99, reward_scale=1, ppo_epochs=4, ppo_clip=0.2, entropy_coef=0.001, value_coef=0.5, return_estimator=nstep, n_steps=5, hidden_dim=64, gradient_clip=10, bootstrap_truncated=True`。每回合新鲜 120 步完整 rollout，不复用旧 replay；100 回合对应 400 次 actor 和 400 次 critic 优化。二者共享 actor，IPPO local critic、MAPPO global critic；当前 profile 使用全网平均局部负 queue 奖励。这是本地 PPO 配置，不是完整 cMALC-D 课程复现。
- SOTL/E-MP/A-MP 尚需实现，不能把现有 MP 换名字充当。`libsignal-mplight/libsignal-colight/maddpg` 虽已注册，但不自动加进用户指定的 16 项主矩阵。
- DynamicLight/FuzzyLight 尚未接入。先完成异步到期、相位与时长动作、每路口实际 elapsed/奖励归属及对应训练阶段，再定义其完整预算；不能把本批 120 步固定时长假设套过去。

论文设置表另行对齐。已核对的 LLMLight 源版本为 `d5d4180f34edb843e1d1b462d5846c75d6d4533a`，基础配置包含 batch=20、sample_size=3000、memory=12000、Keras fit epochs 上限 100 / patience=10、epsilon 每 round 乘 0.95 并以下限 0.2 截断、target 频率按 5 round 配置；其 pipeline 每 round 评估并汇总最后十轮。本地的 batch32、每 round 十次更新、线性探索率、100 次梯度同步 target、最终 checkpoint 统计均有差异。源算法的实际覆盖配置、更新调用和 ATT 还要逐项核对，不能仅改 CLI 数字就宣称等价。

第一批交付物应为：六种控制器 × 五场景的统一指标表、全部 seed 明细、四个 RL 的冻结评估学习曲线、完整配置/输入/源码 manifest、checkpoint 与日志，以及单列的论文协议差异。只在协议真正对应后计算相对论文数值的差距。
