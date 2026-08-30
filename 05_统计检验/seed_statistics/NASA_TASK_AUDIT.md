# NASA Milling — Task & Data Audit

日期：2026-08-30。目的：核对本轮 N1–N4 任务定义与用户新下载的 `mill` 数据是否一致，审计旧代码，确定最终冻结版本。

## 1. 数据来源核实

`C:\Users\banghai\Documents\BaiduSyncdisk\西工大\王婷\论文\mill\`：
- `Readme.pdf`（175KB）+ `mill.mat`（72.5MB）—— 经典 NASA/PHM Society Milling Data Set 单文件格式。
- `scipy.io.loadmat` 读取确认结构：`mill` 字段，shape `(1, 167)`，每条记录含
  `case, run, VB, time, DOC, feed, material, smcAC, smcDC, vib_table, vib_spindle, AE_table, AE_spindle`。
- **Case 覆盖**：case 1–16 全部存在，与本任务 N1–N4 使用的 case 全集（1–16）完全一致。
- **每 case 记录数**：`{1:17, 2:14, 3:14, 4:7, 5:6, 6:1, 7:8, 8:6, 9:9, 10:10, 11:23, 12:15, 13:15, 14:9, 15:7, 16:6}`，合计 167。
- **信号通道**：6 路，每路 9000 采样点/run（`smcAC/smcDC/vib_table/vib_spindle/AE_table/AE_spindle`）。
- **VB（磨损）标签缺失**：167 条中 **21 条缺失 VB**（`(case,run)`：`(1,2)(1,3)(1,5)(1,16)(2,6)(7,8)(8,4)(11,7)(11,14)(11,17)(12,1)(12,4)(12,10)(13,1)(13,2)(14,1)(14,5)(15,1)(16,1)(16,2)(16,4)`）。**处理方式沿用旧代码 `build_case_relative_q_and_stage_labels`（第 240 行）的既有方法**：在每个 case 内部对 VB 做线性插值（`vb.interpolate(method="linear", limit_direction="both")`）——VB（刀具后刀面磨损量）是同一 case 内随时间单调非降的物理量，case 内插值是标准、非捏造的信号处理做法，不是本轮临时决定。本轮沿用该逻辑，不重新发明剔除策略。
- 是否需要解压：否（原生 .mat）。是否有重复文件：否。官方 README 已附带（`Readme.pdf`，未逐字解析，非阻塞项）。

## 2. N1–N4 任务定义核对（关键发现：与旧脚本不一致）

**本轮采用（用户直接给定，最高优先级）：**

| Task | Train cases | Test cases |
|---|---|---|
| N1 | 1,2,4,5,6,7,8,9,10,11,13,14 | 3,12,15,16 |
| N2 | 1,3,4,5,6,7,11,12,13,14,15,16 | 2,8,9,10 |
| N3 | 1,2,4,6,7,8,9,11,13,14,15,16 | 3,5,10,12 |
| N4 | 1,2,3,4,5,6,7,9,10,11,14,15 | 8,12,13,16 |

结构性校验（程序验证，非人工核对）：4 个任务均满足 `train ∩ test = ∅`、`train ∪ test = {1..16}` 完整覆盖。**PASS**。

**旧代码 `代码/9.1nasa数据实验.py::FIXED_TASKS`（第 92-97 行）：**

| Task | Train cases | Test cases |
|---|---|---|
| N1 | 1,2,3,4,5,7,8,9,10,12,13,16 | 6,11,14,15 |
| N2 | 2,3,4,5,6,7,8,11,12,13,14,15 | 1,9,10,16 |
| N3 | 1,2,4,6,8,9,10,11,13,14,15,16 | 3,5,7,12 |
| N4 | 1,3,5,6,7,9,10,11,12,14,15,16 | 2,4,8,13 |

**两者完全不同**（例如 N1 的 test 集：本轮 `{3,12,15,16}` vs 旧脚本 `{6,11,14,15}`，仅巧合共享元素"15"）。

此外发现旧代码 `代码/` 目录下还有三个"case 选择优化"脚本：`run_nasa_bestcase_candidate_split.py`、`run_nasa_case_split_optimization.py`、`run_nasa_dcpsr_stageaware_optimization.py`，配合 `补充材料/小论文/nasa_dcpsr_results_bestcase_split/NASA_selected_bestcase_splits.json` 等产出文件——**这些脚本按候选 split 在目标任务上的表现（`Split_quality_score`/`B12_select_score`）挑选"最优" train/val/test case 划分**，这是历史阶段的探索性做法，**不符合"不能用 test 表现挑 split"的当前规范**，本轮明确不采用、不参考其选出的具体 case 组合。

**结论（按用户指定的优先级：用户直接给定 > 旧脚本）**：本轮采用用户直接给定的 N1–N4（已通过结构校验），`FIXED_TASKS`（旧脚本）与"bestcase 优化"系列脚本均仅作为审计对照，不采用其 case 划分。

## 3. 复用的核心方法代码（非 split 部分）

`代码/9.1nasa数据实验.py`（2182 行）提供了与 PHM `main_experiment_3_fgds_psi_optimized.py` 平行的完整流水线，本轮直接复用（只读方式）：
- 信号特征提取：`signal_features` / `extract_signal_features`
- 阶段/细分状态标签：`build_case_relative_q_and_stage_labels`（`QE=0.30, QL=0.72, QV=0.78`，与 PHM 的 `Q_EARLY/Q_LATE/RATE_LATE_Q` 逻辑一致）、`fit_gmm_fine_states`/`assign_fine_states`
- Train-only 特征选择：`select_features_train_only`（`N_FEATURES=45`，与 PHM 相同）
- 窗口构建：`build_sliding_windows`（`L_DEFAULT=6`——**注意与 PHM 的 L=12 不同**，因 NASA 单 case 最少仅 6 条记录（case 6），L=12 在多数 case 上不可行，L=6 是该数据集原有的、预先确定的窗口长度，非本轮临时调整）
- 模型架构：`TCNGRUMultiTaskModel`（`TCN_CHANNELS=(32,64,64)`, `GRU_HIDDEN=64`, `DROP_OUT=0.20`, `BATCH_SIZE=16`, `LR=5e-4`, `WEIGHT_DECAY=1e-5`, `EPOCHS=200`, `PATIENCE=15`——均取自脚本顶部常量，为该脚本的默认/主线配置，不使用 `MODEL_CONFIGS`/`B12_SEARCH`（架构与推理参数的探索性搜索空间，本轮不使用，不临时调参）
- DC-PHSR / DC-PSR 推理：`apply_dcpsr_inference` + 默认 `PROB_PARAMS`（第 99-107 行）：
  ```
  eta=0.75, fine_weight=0.30, temperature=1.20, mid_floor=0.12,
  late_tau=0.66, early_tau=0.38, order_blend=0.25
  ```
  **与 PHM2010 的 `B12_PARAMS` 完全一致**——确认这是该方法跨数据集共用的默认推理参数，不是本轮临时决定，本轮沿用不变。

## 4. VB 缺失分布（供参考，不剔除，见上）

各任务 test 集中原始缺失 VB 的 run 数（插值前）：N1=7（case12×3/15×1/16×3），N2=2（case2×1/8×1），N3=3（case12×3），N4=9（case8×1/12×3/13×2/16×3）。这些 run 经 case 内线性插值后正常参与建模，不从 evaluation universe 中剔除。最终 `n_test`（窗口数）以 `build_sliding_windows` 实际产出为准，**不预设固定数值**，与 D2/D3 一致的原则；另注意 case6 仅 1 条记录 < L=6，`build_sliding_windows` 会跳过该 case（不产生任何窗口）——核对后确认 case6 在 N1–N4 全部 4 个任务中都只出现在 train 集，不影响任何 test evaluation universe。

## 5. 下一步

- 基于以上确认的 N1–N4（用户版）与复用的核心方法代码，构建 `shared/reproducibility/NASA_N{1..4}_frozen_preprocess/`（source-only 构建，PREPROCESS_SEED=42，与 PHM 一致；注意剔除 VB 缺失 run）。
- Smoke test → seed0 真实验证 → 全量 seed 0–100 扫描。
