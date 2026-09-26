# PressLight：Jinan1 五种子实验

入口：`python -m cityflow_tsc.presslight_experiment train|evaluate|compare`。

已核对LLMTSCS提交 `d5d4180f34edb843e1d1b462d5846c75d6d4533a` 的 run_presslight.py、models/presslight_one.py、models/network_agent.py、utils/config.py、utils/updater.py，缓存文件Git blob hash与tree一致。run_presslight.py 的内部模型键虽名为 EfficientPressLight，实际输入是一般排队压力，展示名称为PressLight；不能把它误标为E-PressLight。

输入当前相位8维 + general queue pressure12维。每进口车道排队数减目标出口道路三车道排队总数（不除3）。奖励为−0.25×绝对值(总进口排队−总出口排队)。源网络20维输入→共享20维sigmoid；四个当前相位分支各为20维ReLU→4个Q值；根据8位当前相位完全相等选择分支，再把四个物理相位Q映射到路网相位顺序。全零黄灯编码不匹配任何分支，输出0。

Dense使用Glorot uniform、bias0；Adam lr=.001、epsilon=1e−7（tf.keras默认）。γ=.8、TD奖励/20。作者两级回放：各路口保留12000，使用共享索引最多抽3000；按路口顺序拼接，再截末12000并抽3000。因此抽满后仅末四个路口参与第二级抽样，此源代码偏置保留。batch20，最多100个优化epochs，validation30%、patience10；目标网络滞后5轮；探索max(.8*.95^round,.2)，全网一次探索硬币。五种子各100 rounds×3600秒，评估epsilon0；末十轮种子内平均后算跨种子均值/样本标准差。

时序30秒含切换黄灯5秒、全红0秒，laneChange=true。与论文文字30/3/2有差异；独立PyTorch随机数、实际回合边界next state及时间截断bootstrap等移植差异保留说明，不能宣称完全一致复现。完成量按真实ID记账，保留原生ATT和LLMTSCS口径ATT。

对照LLMLight表2 PressLight：Jinan1 291.57秒。另列Astra表1（八相位，不能公平排名）。FuzzyLight、Traffic-R1缓存表格没有同名PressLight结果行，不以其他方法填补。
