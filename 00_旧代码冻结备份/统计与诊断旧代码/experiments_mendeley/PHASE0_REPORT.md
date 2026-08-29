# Phase 0 报告 — 代码审计 + 第三数据集审计

日期：2026-08-17
状态：**Phase 0 部分完成。Phase 1–8 在当前环境下 BLOCKED。**

---

## 0. 结论先行

三件事，按重要性排序：

1. **当前会话的运行环境无法执行这个任务的训练部分。** 云端沙箱和你本机的 Cowork Linux VM 都**没有 PyTorch、没有 h5py**，而 PyPI / GitHub raw / conda / download.pytorch.org 全部被网络白名单拒绝。因此：读不了 `.h5`，也训不了 TCN-GRU。这不是可以绕过的问题。
2. **数据集审计已完成**（基于 `filelist.csv` 的全量 6418 条标签，不需要读 HDF5）。9 把刀具的结构非常干净，T8 的已知缺失被量化了，且影响比预想的小。
3. **代码审计发现了 4 个必须在投稿前处理的严重问题**，其中 NASA 那一条会直接影响论文可辩护性。这些和第三数据集无关，但会污染第三数据集的实验设计（因为要复用同一套代码）。

---

## 1. 数据路径与实际结构

**路径**（在已连接的文件夹内，不需要额外授权）：

```
论文\Multivariate time series data of milling processes with varying tool wear and machine tools\
    filelist.csv
    M{m}T{t}R{r}C{c}VB{vb}.h5      × 6418
```

- 总体积 **10 GB**，单文件约 1.5 MB
- `filelist.csv` 6418 行，列：`filename, machine, tool, run, cumulated_tool_contact_time, wear`
- **文件名编码与 CSV 标签 100% 一致**（6418/6418 校验通过，0 处不符）→ 元数据可信，且可以完全不读 HDF5 就拿到 M/T/R/C/VB
- **Machine→Tool 映射经实测确认**：M1={T1,T2,T3}，M2={T4,T5,T6}，M3={T7,T8,T9}，与任务书一致

## 2. 九把刀具的基本统计

| Machine | Tool | runs | VB 范围 (µm) | 首个 run 的 VB | run 编号 | 累积接触时间 max | 缺失 run 编号 | VB 回退步数 |
|---|---|---|---|---|---|---|---|---|
| M1 | T1 | 609 | 3–158 | 3 | 1–609 | 3458 | 0 | 0 |
| M1 | T2 | 609 | 3–145 | 3 | 1–609 | 3418 | 0 | 0 |
| M1 | T3 | 638 | 3–152 | 3 | 1–638 | 3646 | 0 | 0 |
| M2 | T4 | 928 | 3–149 | 3 | 1–928 | 5171 | 0 | 0 |
| M2 | T5 | 928 | 3–148 | 3 | 1–928 | 5172 | 0 | 0 |
| M2 | T6 | 928 | 3–147 | 3 | 1–928 | 5181 | 0 | 0 |
| M3 | T7 | 609 | 3–153 | 3 | 1–609 | 3439 | 0 | 0 |
| M3 | **T8** | **560** | **34**–150 | **34** | 1–560 | 3473 | 0 | 0 |
| M3 | T9 | 609 | 3–160 | 3 | 1–609 | 3449 | 0 | 0 |

要点：

- **run 编号全部连续**，没有编号意义上的缺失。M2 三把刀各 928 run，明显多于其他机床（约 1.5 倍），累积接触时间也更长（5171 vs 3450 左右）——M2 的单 run 切削量更小。
- **VB 序列全部单调不减**（回退步数 = 0）。说明发布方给出的 VB 已经过处理，不是原始逐点测量值。这对论文的 `q` 定义是好消息（平滑窗口 7 几乎不改变形状），但也意味着 **`q` 的"真值"本身带有发布方的平滑先验**，写论文时应该说明。

### T8 的已知缺失：已量化，影响可控

官方说明的"T8 初始磨损阶段部分 run 缺失"，在数据里的表现**不是 run 编号断号，而是 T8 的第 1 个 run 就已经处在 VB = 34 µm**（其余 8 把刀都从 VB = 3 µm 开始）。

按论文协议（每把刀内部 min-max 归一 + 分位数阈值）算出来的各阶段实际 VB 覆盖范围：

| Tool | early 覆盖 VB | middle | late |
|---|---|---|---|
| T1 | 3–64 | 64–101 | 101–158 |
| T7 | 3–56 | 56–99 | 59–153 |
| **T8** | **34–67** | 67–109 | 70–150 |
| T9 | 3–60 | 61–103 | 86–160 |

**这是一个真实的语义偏移**：T8 被标成 early 的样本，物理磨损量是别人 early 末段到 middle 初段的水平。但影响没有想象中大——其他刀具中 VB<34 µm 的 run 只占 **2.3%–4.6%**，也就是说 T8 丢掉的只是很短的初期磨合段，而 T8 的 early 上界（67）与其他刀的 early 上界（56–64）其实相当接近。

**处理建议（不造假、不补样本）**：T8 保留在训练集里没问题；把它作为 LOTO 单独 target 时，必须在结果里标注 `early_truncated=True`，且 early-stage 指标单独说明。已写入 `stage_coverage_by_tool.csv` 和 `stage_VB_range_by_tool.csv`。

### 阶段分布（按论文默认协议 Q_E=0.30 / Q_L=0.72 / Q_ν=0.78，每 Tool 内部归一）

九把刀全部三阶段齐备，分布相当均衡（early ≈ 30%、middle ≈ 40%、late ≈ 30%），默认协议**不需要为这个数据集调整**。逐 run 结果见 `per_run_stage_preview.csv`。

## 3. HDF5 channel schema（**部分完成**）

无法用 h5py 打开文件，只能从 HDF5 对象头的未压缩字节里恢复出 link 名称：

- `signals_sensor`（25 kHz）：`time_sensor`, `force_sensor_x/y/z`
- `signals_machine`（500 Hz）：`time_machine`, `torque_spindle`, `torque_axis_x/y/z`, `tool_position_x/y/z`, `position_control_deviation_axis_x/y`
- 属性：`machine`, `wear`, `cumulated_tool_contact_time`, `unit`
- 数据集为 **deflate 压缩分块存储** → 不实现完整 HDF5 B-tree + inflate 就取不到数值

**注意**：任务书里提到的 `force_axis` 这个通道名**在文件里不存在**。进给轴上的量是 `torque_axis_*`。官方说明的 M2 混叠 / M3 坐标错配大概率对应 `torque_axis_*`、`torque_spindle`、`tool_position_*`、`position_control_deviation_axis_*`——但**这是推测，未经验证**，必须用 h5py 打开确认后才能写进 `excluded_channel_report.csv`。

`shape`、`dtype`、真实采样率、5 文件跨机床对比 —— **全部未完成**。详见 `hdf5_schema_report.json`。

## 4. 代码审计：4 个严重问题

完整证据见 `CODE_AUDIT.md`（622 行，含 file:line 引用）。这里只列必须马上决策的。

### 4.1 `monotonic_q_loss` 在主实验里是失效的（已验证）

```python
# main_experiment_3_fgds_psi_optimized.py:738-742
def monotonic_q_loss(q_hat):
    return torch.relu(-(q_hat[1:] - q_hat[:-1])).mean()
# :675
loader = DataLoader(..., shuffle=(split_name == "final_train"))
```

主实验训练集 `split_name == "final_train"` → **shuffle=True**，于是单调约束施加在同一 batch 内**随机排列**的、时间上毫无关系的窗口之间。λ_m 这一项在论文里被写成"weak monotonic regularization"，实际是噪声。

更糟的是这个 flag 是**魔法字符串**：`7.7跨工况实验.py:305` 传的是 `"train"` 而不是 `"final_train"` → 跨工况实验是**不打乱**训练的。**同一篇论文里同一个 L_mono 有两种语义**。而 `1.3细化的阶段分类.py` 里选 `BEST_ARCH` 时也是不打乱的——即架构是在"不打乱"下选出来，然后拿到"打乱"的主实验里用。

→ 第三数据集必须用你要求的 sequence-aware 实现（同 Tool、按 run 排序、真实相邻窗口）。

### 4.2 B1 是 oracle baseline（已验证）

```python
# 7.4对比实验.py:324
vb_test = test_by_cut.loc[meta["cut_index"].values, "VB_smooth"].values
y_b1 = np.where(vb_test <= th_e, 0, np.where(vb_test >= th_l, 2, 1))
```

阈值确实只用训练集算，但**输入是测试集的真实 VB**。B1 不是 sensor-based 分类器，它是拿真值磨损量做阈值判断的 oracle。论文 Table 3 把它和 B3–B12 并排列成"Stage definition / Model"而没有任何说明。必须改：要么标注 `oracle_wear_reference = True`，要么换成不用真值 VB 的实现。

### 4.3 B2 有单位 bug，它的"结果"是 bug 的产物（已验证）

```python
# 7.4对比实验.py:335-341
rate_test_proxy  = minmax_train_apply(rate_train_proxy, rate_test_proxy)  # 缩放到 [0,1]
theta_v_proxy    = float(np.quantile(rate_train_proxy, base.RATE_LATE_Q)) # 未缩放的原始 rate，量级 1e-2
```

阈值取自原始 rate，比较对象却被缩放到 [0,1] → 速率条件几乎对所有样本触发，B2 塌缩成只输出 early/late。论文 Table 8 的 B2 行自证了这一点：**M→E 0.4806 + M→L 0.5194 = 1.0000，M-F1 = 0.0000**——一个 middle 都没预测出来。

论文把 B2 描述为 "Relative-stage Rule"，实际实现是 RF 回归 q_proxy → 规则。代码里的 `b2_uses_true_vb = False` 是硬编码常量，不验证任何东西。

### 4.4 NASA 的 N1–N4 是按 B12 的测试集表现挑出来的（已验证）

这一条最严重。

```python
# run_nasa_bestcase_candidate_split.py:50
N_CANDIDATE_SPLITS = 20
# :1833  generate_candidate_case_splits(...)   随机生成候选划分 CAND001...
# :2733-2748  用 B12 的【测试集】指标打分，并对 B12 输给 B11 的情况扣分
score = 0.25*Macro-F1 + 0.20*Balanced-Acc + 0.20*M-F1 + 0.10*M-Rec + ...
        + 0.05*max(0, MacroF1_B12 - MacroF1_B11)
        + 0.05*max(0, M-F1_B12   - M-F1_B11)
score -= 0.20 * (M-F1_B12   < M-F1_B11)
score -= 0.20 * (MacroF1_B12 < MacroF1_B11)
score -= 0.15 * (Smooth_B12 > Smooth_B11)
# :2757  select_bestcase_tasks(...) → selection_df.head(4)
```

流程是：随机生成 20 个 case 划分 → 每个都跑 B9–B12 → **用 B12 在测试集上的指标（外加 B12 相对 B11 的优势）排序** → 取前 4 名 → 论文里报成 N1–N4。

佐证：论文 Table 4 的 N1–N4 case 列表（N1 测试 = {3,12,15,16}）与**全部六个 NASA 脚本里写死的 `FIXED_TASKS`（N1 测试 = {6,11,14,15}）都不匹配**——我逐个 grep 确认过。论文里的划分只能来自 bestcase 选择路径。

脚本自己写了透明性说明文件；**论文里没有**。

> 更正一点：我核对过论文 Table 4 的四个划分，**每一个单独看都是合法的 12/4 划分**（train ∪ test = 1..16，无重叠）。问题不在划分本身不合法，而在于**这 4 个是从 20 个候选里按测试集表现选出来的**。

### 4.5 另外还有（详见 CODE_AUDIT.md）

- NASA 脚本的 validation case 是按"与测试 case 描述子距离最小"来选的，含测试集标签比例和 VB 范围 → validation 再去驱动 early stopping 和融合参数搜索。**这是泄漏。**
- NASA 侧还搜索了 stage 标签定义本身，并对含测试集的整表重打标签。
- PHM↔NASA 方法漂移：NASA 没有 L_mono、特征打分没有 DI 项、TCN **没有 Chomp1d（非因果）**、prior 中心/σ/κ 不同、η 网格不相交、允许 order_blend=0（等于关掉有序滤波）。
- 全线单 seed（`RUN_SEEDS=[2026]`）；论文 Table 10 的 `±` 是**跨任务离散度**，不是重复实验方差——现在的写法容易被审稿人误读为后者。
- 7.7 有 9 个 PHM 任务，论文只报了 4 个且**改了编号**（论文 D2 = 代码 D3，S1 = 代码 S3，S2 = 代码 S5）。
- `8.1.2共享表征图.py:190-241`：输入缺失时会用同一份 B11 概率**伪造**两张表征表，图上没有任何标记。
- `ratio_penalty` 在融合参数打分里占 0.4 权重，论文完全没提。

## 5. 泄漏审计小结

- **PHM 主实验（`main_experiment_3`）在我审计的点上是干净的**：train/val/test 的在线特征分别构建，特征选择/scaler/GMM/融合参数搜索都只用 train+val，`probability_param_search` 只在 validation 上调。
  - 两个小瑕疵：训练侧在线特征在**有缺口的**序列上构建（val 段被挖走），而 val 自己重新从头累积历史 → train/serve skew；7.6/7.7 直接用写死的融合参数而不重新搜索。
- **NASA 侧泄漏严重**（见 4.5）。
- **第三数据集必须从干净的实现重来**，不能沿用 NASA 那条路径。

## 6. 训练量估算

按任务书的规模：

| 组 | 训练次数（backbone） |
|---|---|
| Overall B1–B12 × MD1–MD3 × 5 seed | 深度模型 B7–B11 共 5 个 × 3 × 5 = **75**；B11 的输出直接派生 B12 与 A1–A6，不重复训练 |
| Generalization B9–B12 × 9 tasks × 5 seed | B9/B10/B11 共 3 个 × 9 × 5 = **135**（MD1–3 的 45 次与上面重叠，可复用 → 净增 ~90） |
| Ablation A1–A6 | **0 次新增**（复用 Overall 的 B11 backbone） |
| Semantic consistency | **0 次新增**（复用同一批 prediction） |
| LOTO 9 tools × B9–B12 × 5 seed | 3 × 9 × 5 = **135** |

**净训练次数 ≈ 300 个 backbone**，外加 sklearn 类基线（B3–B6，很便宜）。单次训练在 ~6000 run-level 样本、window=12、TCN(32,64,64)+GRU(64)、120 epoch 早停的规模上，CPU 大约几分钟，GPU 几十秒。**在你自己的机器上是一晚上的量级；在这个沙箱里是不可能的量级（且根本没有 torch）。**

## 7. 交付物状态

| 文件 | 状态 |
|---|---|
| `00_dataset_audit/dataset_quality_report.csv` | **DONE** |
| `00_dataset_audit/stage_coverage_by_tool.csv` | **DONE** |
| `00_dataset_audit/stage_VB_range_by_tool.csv` | **DONE**（额外产出，用于量化 T8） |
| `00_dataset_audit/per_run_stage_preview.csv` | **DONE**（6418 行，含 q / ν_norm / stage） |
| `00_dataset_audit/hdf5_schema_report.json` | **PARTIAL** — 仅通道名，无 shape/dtype/采样率 |
| `CODE_AUDIT.md` | **DONE** |
| `excluded_channel_report.csv` / `primary_channel_set.json` | **BLOCKED** — 需先用 h5py 确认通道 |
| `01_features/` 起及之后全部 | **BLOCKED** — 无 h5py（读不了信号）、无 torch（训不了模型） |
| `monotonic_loss_fix.md` | 可以现在写（纯文档 + 代码），不阻塞 |
| `FINAL_REPORT/EXPERIMENT_REPORT.md` | **BLOCKED** |

## 8. 环境实测记录

| 能力 | 云端沙箱 | 你本机的 Cowork Linux VM |
|---|---|---|
| numpy / pandas / matplotlib | ✅ | ✅ |
| scipy / scikit-learn | ✅ | ❌ |
| **h5py / pytables / h5dump** | ❌ | ❌ |
| **PyTorch** | ❌ | ❌ |
| pyarrow（parquet） | ❌ | ❌ |
| 网络 | PyPI 403、files.pythonhosted 403、GitHub raw/codeload 拒绝、conda/清华/阿里源不通；仅 api.github.com 通 | **完全无网络** |
| 后台长任务 | ✅ nohup 可跨调用存活 | ❌ 每次调用是独立 PID namespace，调用结束即被杀；单次上限 45 秒 |
| CPU / 内存 | 2 核 / 7 GB | — |

（第 3 条已实测踩过坑：之前转 PDF 时，nohup 的进程在 device_bash 调用返回的瞬间就被杀掉了。）
