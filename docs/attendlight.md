# AttendLight：Jinan1 五种子实验

入口：`python -m cityflow_tsc.attendlight_experiment train|evaluate|compare`。复刻对象为LLMTSCS固定提交 `d5d4180f34edb843e1d1b462d5846c75d6d4533a` 的 AttendLight baseline，不等同于NeurIPS原作者完整训练平台。run_attendlight.py、models/attendlight_agent.py、utils/cityflow_env.py、utils/config.py、utils/updater.py 的缓存Git blob hash已与该提交tree核对。

输入96维：12条进口车道随后12条出口车道，每车道[停止线前0–100m运行数、100–200m运行数、200–300m运行数、整车道排队数]。运行速度严格>0.1，距离分段下界严格大于；100m边界属于第二段、200m属于第三段，300m边界排除。运行车辆跳过含shadow的ID；排队计数保留引擎原数。进口按W/E/N/S、L/T/R排序，出口按对应进口行驶方向（东/西/南/北）、车道0/1/2排序。网络不输入当前相位。

24×4 token经共享Dense32 ReLU；每相位选2条进口通行车道和6条相关出口车道，以8个token的均值作query，8个token作key/value，共享第一层四头注意力（每头8维）。四个相位token再经过四头self-attention（每头8维），共享Dense20 ReLU→20 ReLU→1得到四个Q值。注意力采用缩放点积/softmax，无残差、LayerNorm或dropout；独立Q/K/V/output，kernel采用源形状的Glorot uniform，bias0；Dense同为Glorot uniform。保留四相位PHASE_MAP，并映射回本地物理动作顺序。

奖励−0.25×abs(进口排队总数−出口排队总数)。Adam lr=.001、epsilon=1e−8；γ=.8、TD奖励/20。作者两级node-major回放与尾部截断保留（抽满时只保留末四路口），回放12000、样本3000、batch20、每round最多100拟合epochs、验证比例.3、patience10、target滞后5轮。探索逐路口独立max(.8*.95^round,.2)，测试epsilon0。Jinan1、五种子0–4、100 rounds×3600秒，每种子末十轮均值再算跨种子均值和样本标准差。

时序每30秒决策，切换包含5秒黄灯、全红0秒；laneChange=true。论文文字30/3/2不同，实际回合边界next state、时间截断bootstrap和独立PyTorch RNG也属于移植差异，不能声称历史实验逐位一致或完整复现原NeurIPS算法。报告引擎ATT、LLMTSCS口径ATT及真实ID完成量。

对照LLMLight表2 AttendLight Jinan1 291.29秒；另列Traffic-R1附录表5目标训练280.11秒和主表2零样本381.11秒，区别时序和训练域，不排公平榜单。参考值来自本地已保存论文表格。
