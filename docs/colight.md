# CoLight 训练与论文结果对照

本入口只训练 CoLight：共享图注意力 Q 网络，在目标车流训练，每轮训练后关闭探索进行评估，最后汇总并生成论文参考表。无需安装或运行 LLMLight 的 LLM 部分。

代码入口：`src/cityflow_tsc/colight_experiment.py`；网络和学习器：`src/cityflow_tsc/baselines/colight.py`。

## 默认设置

| 项目 | 设置 |
|---|---|
| 输入 | 当前相位的 8 维编码＋12 条进口车道车辆数，共 20 维 |
| 网络 | 共享 MLP `[32,32]` → 邻居图注意力 → 4 个相位的 Q 值 |
| 注意力 | 5 个头，每头 32 维；按距离选择含自身的最多 5 个路口 |
| 动作 | 四个直行/左转配对相位；根据 roadnet 映射实际相位编号 |
| 奖励 | 每路口 `−0.25 × 进口排队车辆数`，决策区间内取均值；TD target 中再除以 20 |
| 优化 | Adam，学习率 `0.001`，γ=`0.8`；batch=20 |
| 回放 | 保存最多 12000 条整网 transition；每轮最多抽样 3000 条 |
| 每轮拟合 | 固定 TD target，最多 100 epochs；30% 验证集，patience=10，不回滚到最优 epoch |
| 目标网络 | 第 0 轮用初始化网络；第 r 轮读取第 `max(r−5,0)` 轮训练后的网络 |
| 探索 | 第 r 轮 `max(0.8 × 0.95^r, 0.2)`；评估 ε=0 |
| 仿真 | 每回合 3600 秒，仿真步长 1 秒，lane change 开启 |
| 控制时序 | 每次决策共 30 秒；切换包含 5 秒黄灯＋25 秒新绿灯；保持时 30 秒绿灯；全红 0 秒 |
| 训练/统计 | 100 rounds；每个种子取末 10 轮评估均值，再对训练种子求均值与样本标准差 |

这是独立 PyTorch 实现。网络与拟合参数参考 [LLMTSCS 的 CoLight 实现（固定提交）](https://github.com/usail-hkust/LLMTSCS/blob/d5d4180f34edb843e1d1b462d5846c75d6d4533a/models/colight_agent.py)，不是其历史运行的逐位复刻。它使用本项目的实际边界 next state、各路口独立探索、持续保留的 Adam 状态和独立随机种子。参数、文件 SHA256、运行库版本、逐轮训练与评估数据都会保存。

## 运行

在本项目根目录执行。运行环境需要 Python、NumPy、PyTorch 和可用的 CityFlow；本地 `.venv` 已有前面三项，但当前没有安装 CityFlow。下面的 `/path/to/...` 是需要替换的数据路径；本项目不会自动下载数据。

先运行 Jinan1，一个训练种子：

```bash
PYTHONPATH=src python -m cityflow_tsc.colight_experiment train \
  --roadnet /path/to/data/Jinan/3_4/roadnet_3_4.json \
  --flow /path/to/data/Jinan/3_4/anon_3_4_jinan_real.json \
  --output /mnt/pan/colight/jinan1-seed0 \
  --seeds 0 --rounds 100
```

`python` 应指向已安装上述依赖的环境。5090 上的全部实验输出必须放在 `/mnt/pan`；其他机器可指定自己的输出目录。输出目录须为空，避免覆盖实验。

需要五次独立训练时，将 `--seeds 0` 改为 `--seeds 0,1,2,3,4`，并使用新的输出目录。其余配置不变。单个训练种子的结果不会产生“跨种子标准差”。

每次训练自动产生：

- `protocol.json`：数据身份、训练参数、时序、种子和统计方式。
- `seed_0/run.json`：每轮训练、评估指标与网络拟合记录。
- `seed_0/latest.pt`：最近保存的网络、优化器、回放、随机状态和目标网络历史；默认每 10 轮及最后一轮保存。
- `seed_0/train/`、`seed_0/evaluation/`：每轮轨迹与指标。
- `summary.json`：训练完成后的种子汇总。
- `paper_comparison.md/.csv/.json`：我们的成绩与各论文 CoLight 参考值及设置差异。

单独评估已有 checkpoint：

```bash
PYTHONPATH=src python -m cityflow_tsc.colight_experiment evaluate \
  --checkpoint /mnt/pan/colight/jinan1-seed0/seed_0/latest.pt \
  --roadnet /path/to/data/Jinan/3_4/roadnet_3_4.json \
  --flow /path/to/data/Jinan/3_4/anon_3_4_jinan_real.json \
  --output /mnt/pan/colight/jinan1-frozen-eval --seed 20000
```

评估读取 checkpoint 同目录的 `protocol.json`，不训练、不更新回放。可换同一路网的车流；换车流后的成绩属于另一项评估。当前 CLI 没有断点续训命令，学习器的保存/加载恢复已由测试覆盖。

若不同种子分开启动，训练完成后合并同配置的结果：

```bash
PYTHONPATH=src python -m cityflow_tsc.colight_experiment compare \
  --runs /mnt/pan/colight/jinan1-seed0 /mnt/pan/colight/jinan1-seed1 \
  --output /mnt/pan/colight/jinan1-comparison
```

## 怎么与论文比较

默认先看 LLMLight 表 2 的 CoLight 行，Jinan1 参考 ATT 为 **279.60 秒**。报告还收录 FuzzyLight、Traffic-R1、Astra，以及有候选数据对应关系时的 CoLLMLight、FutureLight、AMM；每行保留论文链接、表号、训练方式与时序差异。只按文件 SHA256 自动认定已核对的 Jinan/Hangzhou 数据，未识别数据标为 `custom`。

这些值用于外部参照。默认 30 秒决策含 5 秒黄灯与 LLMLight 公布的代码设置一致，而论文文字写的是绿/黄/全红 30/3/2 秒。FuzzyLight 的训练预算、Traffic-R1 的控制时序、零样本迁移论文的训练域，以及八相位实验均不能自动视为本次同设置实验。因此报告不会将不同设置的数值差当作算法优劣结论。

同时保存两种 ATT，单位均为秒：

- `average_travel_time_s`：CityFlow 引擎原生 ATT。
- `llmtscs_att_s`：按照 LLMTSCS 的逐路口进口车道记录，累计每辆车停留时间后取均值；未记录离开的条目填仿真终点。为保持源码统计兼容，保留其“上一时刻车辆集合为空时，新进入车辆也被记录一次离开”的分支。这不是通常意义上的完整行程耗时，也不能与引擎 ATT 混用。

其他排队/等待指标沿用本项目定义，尚不声称等于各论文的 AQL/AWT。正式比较时优先检查数据、控制时序、训练域与 ATT 定义；同名 CoLight 不保证数值条件相同。

开启换道时，CityFlow 的 `get_vehicle_count()` 包含临时 shadow，而 `get_vehicles()` 已排除 shadow。完成车辆数与完成率按后者返回的真实车辆 ID 全程记账，不用原生计数推算完成数；该口径写入 `lifecycle.manifest.json`。原生 ATT 保持引擎原值，不能反推真实车辆完成数。

## 已做的检查

测试覆盖实际 PyTorch 梯度更新、跨回合目标网络、checkpoint 恢复、相位编号置换、冻结评估不改变训练状态、两种 ATT 分离，以及训练→评估→汇总命令链。测试中的交通来自确定性测试后端，不代表已跑 CityFlow，也不产生论文可比较的交通成绩。

```bash
PYTHONPATH=src python -m pytest tests/test_colight_experiment.py
```
