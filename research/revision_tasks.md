# 调研后续问题单

这是仓库内唯一的低风险问题与后续核验记录。它不改变当前主路线，也不把未完成项包装成已完成证据。

## 投稿实验前必须解决

- 为 26 篇深读论文补齐所有可访问 PDF 的逐表原始数字、seed/方差和页码。目前 `paper_result_audit.csv` 只收录本轮能够由正式页面、已审 PDF 或仓库文件安全确认的数字；其余明确标为未提取，不能进入跨论文数值排名。
- 在正式写 novelty claim 前，以投稿截止日前 30 天为界重新检索 2025–2026 年正式论文、early access 和录用列表，特别检查 traffic-signal dynamics、action-conditioned graph model 和 conservative MPC 的直接交集。
- 核实当前数据资产是否真的满足每城市 `6 train + 2 calibration + 4 test` 个内容不同的独立 flow。若不满足，先形成可审计的需求生成协议；同一确定性 flow 换 seed 不计数。
- 选定 PressLight/CoLight/UniLight 和直接 model-based TSC 的维护实现，逐项核对许可证、CityFlow 版本、黄灯/最小绿灯、reward 和旅行时间定义。无法统一契约的结果不能混表。
- 先验证 CityFlow 从 episode 起点重放再分支的逐字段确定性。若失败，采用同步平行 engine；不得用只包含观测量的手工 simulator state 冒充反事实真值。
- 在实现接口升级前冻结 trajectory schema v2、checkpoint schema v2 和候选生成器哈希，确保旧的 v1 checkpoint 只读兼容且不会被静默误载。

## 可在主体实现后集中处理

- 为表格生成脚本增加 URL 可达性缓存和 CSV 字段级 schema 校验，减少未来文献刷新时的人工检查。
- 把 movement、拥堵分位和候选支持度的诊断图统一成一个可重跑报告命令。
- 检查 5090 推理延迟、ensemble 显存峰值和候选批处理；仅在结果达到机制门槛后做性能优化。
- 当前没有独立 GPT Pro 宏观审稿器可用；现有 `audits/` 是内部敌意审计意见，不是外部专家证据。投稿前另做一次独立审稿。

## 当前不处理

- 视频生成式 World Model、视觉观测和 DreamerV3 整体迁移。
- 在单城市候选排序未成立前做大规模跨城市预训练。
- 与当前主问题无关的通用抽象层、占位模块或外部仓库复现。
