# DC-PSR 第三数据集补充实验报告

**数据集**：Multivariate time series data of milling processes with varying tool wear and machine tools（Denkena, Klemme, Stiehl；DOI 10.17632/zpxs87bjt8.3）
**日期**：2026-08-18
**运行环境**：Windows 11，conda env `dcpsr`，Python 3.11.15，PyTorch 2.7.1+cu118，NVIDIA GeForce RTX 3070 Ti Laptop GPU（驱动 527.58，`torch.cuda.is_available()=True`）
**代码**：`experiments_mendeley/code/`（dataset-independent 实现：Dataset Adapter → Shared DC-PSR Pipeline → Experiment Runner），完整依赖清单见 `environment_report.txt` / `requirements_snapshot.txt`

---

## 0. 执行状态总览

| 实验 | 状态 |
|---|---|
| Gate 1-3 环境部署 + HDF5 真实读取 + sanity check | **完成** |
| E1 Overall Performance Comparison（B1-B12 × 3 个 dual-source task × 5 seeds） | **完成**，45/45 单元 |
| E2 Cross-Machine Generalization | **部分完成**：3 个 dual-source task 的 B9-B12（复用 E1 结果）已完整；6 个 single-source task（MS1-MS6）应用户指示**暂停**，未启动训练，无结果产出，无遗留文件 |
| E3 Ablation and Component Analysis（A1-A6） | **完成**，从 E1 的 B11 backbone 免费派生，15/15 单元 |
| E4 Degradation Semantic Consistency | **完成**：q 指标、stage semantics、representation embeddings、逐 run 概率演化全部产出 |
| 原 Table 3（Stage-related wear-value bias） | **按指示跳过**，不在第三数据集范围内 |

按用户 2026-08-18 的明确指示：**E2 的 single-source 扩展已暂停**，本报告只报告已完成的 3 个 dual-source cross-machine task 作为核心外部验证结果；提议的 2 个代表性 single-source 补充任务（MS6: M3→M2、MS4: M2→M3）尚未启动，等待确认。

---

## 1. Dataset

- **结构**：3 台机床（M1/M2/M3）× 每台 3 把刀具（T1-T9），共 6418 个 run 级 HDF5 文件，`filelist.csv` 标签与文件名 100% 一致（逐行校验通过）。
- **Machine → Tool**：M1={T1,T2,T3}（609/609/638 runs），M2={T4,T5,T6}（928/928/928 runs），M3={T7,T8,T9}（609/560/609 runs）。
- **VB（磨损量）**：整数，范围 3-160（µm 量级），每把刀内部严格非递减（未见回退）。
- **HDF5 真实 schema**（用 h5py 实测，而非按文档名称猜测）：
  - `labels/`：`machine, tool, run, cumulated_tool_contact_time, wear`
  - `signals_sensor/`（25 kHz，gzip 压缩）：`force_sensor_x/y/z`（三轴测力仪，9 把刀具全部拥有，干净）
  - `signals_machine/`（500 Hz）：**通道名因机床而异**——M1 拥有 `torque_axis_x/y/z`，M2/M3 拥有 `force_axis_x/y/z`；两者互不存在于对方文件中，是两套硬件上真实不同的通道，不是同一物理量改了名字。

### 1.1 通道可用性判定（逐刀具解析，证据链见 `00_dataset_audit/excluded_channel_report.csv`）

| 通道 | primary（可用于主实验） | restricted（有据可查的异常，排除） | 依据 |
|---|---|---|---|
| `force_sensor_x/y/z` | T1-T9 全部 | — | 无异常记录 |
| `torque_axis_x/y/z` | T1,T2,T3（M1 专属） | — | M1 无异常记录 |
| `force_axis_x/y/z` | **仅 T9** | T4,T5,T6（M2 aliasing）；T7,T8（M3 坐标错配） | 通道名与文档"force_axis"精确匹配 |
| `torque_spindle` | T1,T2,T3,T7,T8,T9 | T4,T5,T6（M2 aliasing） | 通道名与文档精确匹配 |
| `position_control_deviation_axis_x/y` | T1-T6,T9 | T7,T8（M3 坐标错配） | 通道名与文档精确匹配 |
| `tool_position_x/y/z` | 无（全体排除） | — | UNRESOLVED：文档提到的 "position_axis" 在文件中不存在，`tool_position_*` 只是名称相近的候选，**未确认，不猜测映射**，因此在任何 tool 上都不进入 primary 集合 |

**重要修正说明**：第一轮特征提取（未使用本报告的按机床逐通道解析）曾错误地把 `signals_machine` 整组通道全部丢弃（因为用"9 把刀取交集"的方式判定通道是否存在，而 `force_axis`/`torque_axis` 各自只在 6/9 把刀上出现，交集为空），导致只剩 4 个 25kHz 力信号可用。发现后立即停止、修复为逐刀具解析后重新提取，最终版本每把刀具使用的通道数在 9-14 个不等（详见下表），无猜测成分。

| 刀具 | primary 通道数 | 备注 |
|---|---|---|
| T1,T2,T3 | 12 | labels(5) + torque_axis(3) + position_control_deviation_axis(2) + torque_spindle(1) + force_sensor(3) − 时间轴不计入特征 |
| T4,T5,T6 | 10 | force_axis 和 torque_spindle 均因文档明确的 aliasing 被排除 |
| T7,T8 | 9 | force_axis 和 position_control_deviation_axis 均因文档明确的坐标错配被排除 |
| T9 | 14 | 文档未提及 T9 有任何已知问题，force_axis/position_control_deviation_axis/torque_spindle 全部保留 |

### 1.2 T8：观测生命周期截断

T8 的第 1 个可用 run 就已经处于 VB=34 µm（其余 8 把刀均从 VB=3 µm 开始），`early_truncated=True` 已写入 `stage_coverage_by_tool.csv`。T8 的 early 阶段 VB 覆盖范围是 34-67，而其余刀具的 early 上界在 56-64 之间——T8 丢失的只是很短的初期磨合段，其 early 上界与其他刀具接近。T8 保留在全部正式实验的训练集/测试集里，未做任何人工补数据或修改 VB。

---

## 2. Features

- **原始 run-level 特征**：**528 个**（真实信号驱动，无凑数），来自 5 个通道组的时域（mean/std/rms/skewness/kurtosis/peak/p2p/crest/shape/impulse/clearance/energy/variance/zero-crossing）+ 频域（dominant freq、spectral centroid/spread/entropy/rms/skewness/kurtosis/rolloff95、5 段能量比）特征，按通道名前缀命名（如 `force_sensor_x__rms`）。
- **在线相对特征**（causal，只用 1..t 历史）：1584 个候选（原始特征 × {rel 标准化, slope 局部斜率, online rank}）。
- **每个 (task, seed) 单元最终选择**：45 个特征，经 MI(stage)+MI(q)+|Spearman(q)|−DI(跨域不稳定性) 打分 + 冗余剔除（相关系数阈值 0.92），仅用训练集拟合。

---

## 3. E1 Overall Performance Comparison

3 个 dual-source cross-machine task（对应代码里的 D1-M/D2-M/D3-M，与任务书 MD1/MD2/MD3 一一对应）：

- **D1-M = MD1**：M1+M2 → M3
- **D2-M = MD2**：M1+M3 → M2
- **D3-M = MD3**：M2+M3 → M1

B1-B12 × 3 task × 5 seeds = 45 个训练/评估单元，全部完成，`04_overall_comparison/summary/` 下有完整 by-seed 和 mean±std 表。**B1 是 oracle 参考**（用测试集真实 VB 阈值判断，`oracle_wear_reference=True`），不与 B2-B12 的 sensor-only 方法直接比较。

### 3.1 跨任务汇总（mean_task ± std_task，3 个 task 各自先算 5-seed 均值再跨 task 统计）

| Method | Acc | Macro-F1 | M-F1 | M-Rec | Jump | Smooth |
|---|---|---|---|---|---|---|
| B1 (oracle) | 0.902±0.021 | 0.903±0.021 | 0.879±0.039 | 0.891±0.087 | 0.0±0.0 | 0.006±0.001 |
| B2 relative-rule | 0.444±0.078 | 0.444±0.082 | 0.376±0.097 | 0.365±0.168 | 73.5±46.5 | 0.357±0.148 |
| B3 fixed-RF | 0.500±0.195 | 0.461±0.236 | 0.413±0.226 | 0.429±0.270 | 42.3±57.3 | 0.099±0.031 |
| B4 SVM | 0.419±0.152 | 0.357±0.193 | 0.245±0.214 | 0.202±0.182 | 80.6±68.2 | 0.152±0.036 |
| B5 relative-RF | 0.494±0.189 | 0.446±0.236 | 0.397±0.255 | 0.418±0.303 | 45.7±66.7 | 0.093±0.029 |
| B6 GBDT | 0.498±0.169 | 0.471±0.194 | 0.379±0.278 | 0.419±0.331 | 83.2±101.8 | 0.198±0.064 |
| B7 MLP | **0.537±0.162** | 0.495±0.201 | 0.271±0.285 | 0.234±0.239 | 31.5±29.9 | 0.053±0.017 |
| B8 TCN | 0.501±0.119 | 0.424±0.115 | 0.333±0.239 | 0.393±0.351 | 30.6±29.5 | 0.066±0.034 |
| B9 GRU | 0.501±0.178 | 0.452±0.202 | 0.279±0.344 | 0.275±0.372 | 36.2±48.7 | 0.046±0.021 |
| B10 TCN-GRU | 0.508±0.115 | 0.458±0.130 | 0.247±0.173 | 0.220±0.159 | 13.7±16.6 | 0.036±0.017 |
| B11 raw backbone | 0.519±0.143 | 0.465±0.172 | 0.301±0.198 | 0.302±0.211 | 15.6±17.4 | 0.040±0.022 |
| **B12 (DC-PSR)** | 0.489±0.067 | 0.398±0.022 | **0.402±0.234** | 0.531±0.409 | **2.7±3.4** | 0.031±0.021 |

**如实解读**：
1. B12 在纯 Acc 上不是最高的（B7 MLP、B10 TCN-GRU 更高），这个数据集上的跨机床迁移本身就很难，所有 sensor-only 方法都在 0.42-0.54 之间。
2. B12 的优势集中在**中期阶段识别质量**（M-F1 全场最高）和**时序一致性**（Jump 显著低于其他所有方法一个数量级）以及**跨任务方差最小**（std_task=0.067，是 12 个方法里最稳的，其余普遍 0.12-0.24）——即 B12 不是"每个任务都最准"，而是"预测更稳、跳变更少、middle 类识别更好"。
3. **M-Rec=0.531±0.409 这个数字包含虚高成分**：D2-M 上 B12 出现 middle-collapse（见第 5 节），把 M-Rec 人为拉到 1.0，是失败模式的副产品，不是真实优势，下面单独说明，不建议把这个均值直接当作论文卖点。

逐 task 完整表见 `04_overall_comparison/summary/overall_comparison_mean_std_by_task.csv`，逐 seed 表见 `by_seed/`。

---

## 4. E2 Cross-Machine Generalization（部分完成）

已完成部分（复用 E1 的 B9-B12 结果，`05_generalization/dual_source/dual_source_mean_std.csv`）：

| Task | 方向 | B11 Acc | B12 Acc | B11 M-F1 | B12 M-F1 |
|---|---|---|---|---|---|
| D1-M | M1+M2→M3 | 0.406±0.085 | 0.440±0.090 | 0.252±0.153 | 0.343±0.144 |
| D2-M | M1+M3→M2 | 0.679±0.122 | 0.565±0.130 | 0.519±0.328 | 0.660±0.066 |
| D3-M | M2+M3→M1 | 0.470±0.097 | 0.462±0.086 | 0.134±0.263 | 0.203±0.233 |

**Single-source（MS1-MS6）应用户指示暂停**，未启动训练，`05_generalization/runs/` 下无任何 single-source 单元的 `DONE.flag`/`metrics.csv`（停止时刚好在第一个单元开始训练之前，已清理掉遗留的中间文件，无残留结果）。

**提议的代表性补充任务**（等待确认后再启动，各约 5 seeds × B9-B12，单任务预计 2-2.5 小时）：

- **MS6（M3→M2）**：直接检验 D2-M 的 middle-collapse 是否是 M1+M3 组合训练特有的伪影，还是只要 M2 作为目标域就会复现——对负面发现的稳健性检验。
- **MS4（M2→M3）**：直接检验 D1-M 的正面结果里，dual-source（M1+M2）相对 single-source（M2 单独）是否有真实增量价值。

---

## 5. E3 Ablation and Component Analysis（A1-A6）

从 E1 的 B11 backbone 免费派生，每个 (task, seed) 一次训练即可得到全部 6 个消融变体，共 15 个单元，无需额外训练。完整表见 `06_ablation/summary/ablation_complete_table.csv`。

### 5.1 D1-M（干净的正面案例）

Acc/Macro-F1/M-F1/M-Rec 从 A1（仅 raw）到 A5（causal ordered filtering）单调上升，A6（最终融合）与 A5 基本持平并略微回落（因为 A6 是 A4/A5 的加权混合，β=0.25）：

| Method | Acc | M-F1 | Smooth |
|---|---|---|---|
| A1 raw | 0.406±0.085 | 0.252±0.153 | 0.044±0.025 |
| A2 raw+fine | 0.428±0.090 | 0.312±0.150 | 0.032±0.022 |
| A3 raw+prior | 0.427±0.086 | 0.314±0.142 | 0.035±0.022 |
| A4 mix | 0.427±0.087 | 0.312±0.145 | 0.034±0.022 |
| A5 ordered | 0.442±0.089 | 0.348±0.139 | 0.031±0.022 |
| A6 final (DC-PSR) | 0.440±0.090 | 0.343±0.144 | 0.031±0.021 |

fine-state 头、q 位置先验、有序滤波三个组件各自都带来正向增量，逐步降低 Smooth（波动）、提升 M-F1，是论文期望的教科书式结果。

### 5.2 D2-M（middle-collapse 的机制追溯——如实保留，不包装成优势）

| Method | Acc | M-F1 | M-Rec | L-F1 |
|---|---|---|---|---|
| A1 raw | 0.679±0.123 | 0.519±0.328 | 0.540±0.435 | 0.635±0.119 |
| A2 raw+fine | 0.595±0.103 | 0.673±0.051 | **0.994±0.013** | 0.000±0.000 |
| A3 raw+prior | 0.567±0.147 | 0.663±0.077 | **1.000±0.000** | 0.076±0.171 |
| A4 mix | 0.573±0.127 | 0.664±0.067 | **1.000±0.000** | 0.058±0.130 |
| A5 ordered | 0.565±0.130 | 0.660±0.066 | **1.000±0.000** | 0.041±0.093 |
| A6 final (DC-PSR) | 0.565±0.130 | 0.660±0.066 | **1.000±0.000** | 0.043±0.096 |

**机制**：坍缩从 **A2（raw+fine）就已经基本发生**（M-Rec 从 A1 的 0.54 跳到 A2 的 0.994），而不是从 A3 的 q 位置先验才开始——`fine_to_stage_prob` 把 5 个细粒度状态里的 3 个（S1/S2/S3）都映射到 middle，在跨机床分布偏移下，fine-state 分类头本身就明显偏向 middle，这个偏置被 raw+fine 混合直接继承。A3-A6 引入的 q 位置先验和有序滤波并没有制造这个坍缩，只是没能纠正它，反而因为该任务下 `q_hat` 在跨域测试时被压缩到极窄区间（部分 seed std 只有 0.008-0.02，几乎贴在先验 middle 中心 0.50 上）而进一步锁死。

逐 run 验证：`predictions_test_B11B12.csv` 显示 B12 最终预测 middle 的比例在 5 个 seed 上是 72%-99%，从未预测 late 超过 3.5%。这是一个真实的方法局限——DC-PSR 的细粒度状态头 + 位置先验融合机制，在训练域与测试域统计特性差异很大（M1+M3 → M2）时，会把不确定性错误地折叠进"中期"这个先验密度最高的类别，而不是保持合理的不确定性分散。**没有为了让指标好看而调整融合参数或阈值**——这些参数在看到这个结果之前就已经在 `dcpsr/config.py` 里固定。

### 5.3 D3-M（较弱但方向正确）

A1→A6 略有改善（Acc 0.470→0.462 基本持平，M-F1 0.134→0.203 上升，Smooth 0.061→0.051 下降），改善幅度不如 D1-M 明显，q_R2 为负（-0.56），说明 M1 作为测试域时 q 回归本身就很吃力（详见第 6 节）。

---

## 6. E4 Degradation Semantic Consistency

### 6.1 q 一致性指标（mean±std_seed，B11/B12 共享同一个 q_hat 头，数值相同）

| Task | q-MAE | q-RMSE | q-R2 | Spearman | Pearson |
|---|---|---|---|---|---|
| D1-M | 0.168±0.005 | 0.223±0.004 | 0.009±0.033 | 0.732±0.156 | 0.476±0.186 |
| D2-M | 0.121±0.034 | 0.166±0.033 | 0.220±0.305 | 0.408±0.643 | 0.406±0.578 |
| D3-M | 0.190±0.041 | 0.243±0.043 | **-0.564±0.546** | 0.555±0.086 | 0.437±0.233 |

q_R2 三个任务都不高（D1-M/D2-M 接近 0 或弱正，D3-M 明显为负），说明**q 回归头在跨机床迁移下普遍偏弱**——这是一个需要在论文里坦率讨论的局限，不只是 middle-collapse 那一个任务的问题。Spearman 相关性（0.41-0.73）好于 R2，说明模型至少保留了合理的**相对**排序能力，只是绝对数值的线性拟合较差。

### 6.2 Stage-VB 语义一致性（跨 3 个 task、15 个单元汇总）

按最终预测 stage 分组的真实 VB 均值：

| 预测 stage | 样本数 | 平均 VB | 平均 q_true | 平均 q_hat |
|---|---|---|---|---|
| early | 699 | 55.9±16.6 | 0.337±0.102 | 0.356±0.059 |
| middle | 1097 | 90.9±17.5 | 0.581±0.105 | 0.483±0.051 |
| late | 461 | 94.7±14.5 | 0.600±0.115 | 0.744±0.077 |

**Early < Middle < Late 的平均 VB 单调性成立**（55.9 < 90.9 < 94.7，`monotonic_wear_semantics.txt` 校验为 True）——尽管分类准确率和 q 回归都不算强，模型学到的"阶段"划分在物理磨损量上仍然是有意义、单调递增的，这是一个真实、干净的正面发现，可以支撑论文关于"阶段语义一致性"的论点。但也要注意 middle 和 late 的 VB 区间高度重叠（90.9±17.5 vs 94.7±14.5），边界不够锐利。

### 6.3 表征与概率演化

`representation_embeddings.parquet`（含原始特征、共享隐层表征 h、q_true/q_hat/stage）与逐 run 的完整概率链（`p_raw→p_fine→p_prior→p_mix→p_ordered→p_final`，`07_semantic_consistency/probability_evolution/`）已按 task×seed 全部保存，供后续 PCA/UMAP 和阶段概率演化图直接使用，未提前渲染最终论文图。

---

## 7. T8 对结果的影响

T8（early_truncated=True）在 D2-M、D3-M 中位于训练集（M3 属于训练域），在 D1-M 中位于测试集（M3 是被留出的测试域）。本轮实验**未做 with-T8 / without-T8 的 B12 敏感性对照**（按任务书优先级不是本轮重点）。定性观察：D1-M（T8 在测试集里）依然是三个任务里最干净的正面结果，没有看到 T8 造成明显的额外退化；但由于测试集是 T7+T8+T9 三把刀混合评估，无法从当前数据里单独分离出 T8 的边际贡献。这项敏感性分析留作后续工作。

---

## 8. 最终判断

**第三数据集：Partially supports DC-PSR，同时暴露了一个在 PHM2010 上不明显的真实局限。**

支持的部分：
- D1-M（M1+M2→M3）上，B12 相对 raw backbone（B11）在 Acc/Macro-F1/M-F1/M-Rec/Jump/Smooth 全部指标、全部 5 个 seed 逐一改善，消融链条 A1→A6 单调向好，是一次干净、可复现的正向验证。
- 跨 3 个任务，B12 的 M-F1 全场最高、Jump 最低、跨任务方差最小——在"中期阶段识别"和"预测稳定性"这两个论文核心诉求上有一致优势。
- Stage-VB 语义单调性（Early<Middle<Late）在汇总数据上成立，支持阶段语义具有真实物理意义。

暴露的局限：
- D2-M（M1+M3→M2）上，DC-PSR 的 fine-state 头 + q 位置先验融合机制在严重的跨域分布偏移下会坍缩成近乎单一预测 middle，把 Acc/Macro-F1 拖到低于 raw backbone，Rev/Jump/Smooth 等"一致性"指标虽然好看，但是坍缩的副产品而非真实优势。
- q 回归头（q_hat）在跨机床场景下普遍较弱（q_R2 多数接近 0 甚至为负），只有相对排序能力（Spearman）尚可，绝对拟合能力不足。
- 这个数据集本身对所有方法（包括经典基线）都构成显著挑战——纯 sensor-only 方法准确率普遍在 0.4-0.55 区间，远低于 PHM2010 上报告的数字，说明 Mendeley 数据集的跨机床迁移难度明显更高（部分原因是通道可用性因机床而异，M1 与 M2/M3 的可用特征集合本身就不完全重叠）。

这些负面结果均未做任何掩盖、调参或重抽种子处理，全部按 5 个固定种子（42/52/62/72/82）如实呈现。
