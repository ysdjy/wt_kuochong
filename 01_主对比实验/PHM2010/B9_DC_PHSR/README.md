# B9 — DC-PHSR — PHM2010 — D1

## 方法编号与全称
B9 = **DC-PHSR**（论文正文名称）。旧代码内部标识符：`DC-PSR` / `B12` / `FGDS-PSI`（未重命名，见 `EXPERIMENT_REGISTRY.md` 命名对照）。

## 方法类别
本文提出方法（proposed）。在 B3（Multi-task TCN-GRU）backbone 输出的原始概率上，做一个**确定性**的概率化阶段推理（temperature scaling → q̂-prior → fine-state 混合 → 因果有序滤波 → order_blend 最终融合），不含可学习参数。

## 原始代码来源
- Backbone 训练：`代码/main_experiment_3_fgds_psi_optimized.py::train_model`
- 推理逻辑：`代码/main_experiment_3_fgds_psi_optimized.py::apply_probability_inference`
- 推理参数 `B12_PARAMS`：`代码/7.4对比实验.py` 第 60–68 行（本轮运行时实时读取，未复制硬编码）
- 严格 seed 隔离跑法：改编自 `protocol_diagnostic_fixed_preprocess/scripts/run_diagnostic_seed.py`（原文件未改动，仅在新位置生成了输出路径改写的副本）

## 新目录中的代码位置
`code/run_seed_d1.py` — 唯一改动是把输出目录从旧的 `protocol_diagnostic_fixed_preprocess/results/` 重定向到本方法（及其 backbone B3）各自的 `results/`，并把冻结预处理源指向本项目内的镜像副本 `shared/reproducibility/PHM2010_D1_frozen_preprocess/`；训练/推理逻辑本身通过 `importlib` 只读方式实时从 `代码/` 加载，未复制或修改。

## 数据集
PHM2010，原始特征表 `baselines/htt_net/data/run_level_features_all.csv`（sha256 见 `audit/data_manifest.json`，未复制入本目录，仅 manifest 引用）。

## Train / Validation / Test 定义
- Train：C1 + C4（源域内按 stage 分层切出内部验证集）
- Validation：上述内部验证切分（源域内，非目标域）
- Test：C6，n_test = 304（`run_id` 12–315，L=12 窗口）
- 目标域（C6）**未参与**特征选择、GMM 拟合、超参数调优——仅作为最终评估集。

## Preprocessing
冻结于 `PREPROCESS_SEED=42`（2026-08-20 一次性生成，不重新计算）：45 个训练域-only MI+冗余筛选特征、StandardScaler、5-分量 GMM fine-state 分配、条件相对 E/M/L 标签、L=12 窗口。详见 `../../../shared/reproducibility/PHM2010_D1_frozen_preprocess/` 与 `audit/hash_manifest.json`。

## Seed 定义
- `PREPROCESS_SEED = 42`（固定，不参与本轮变化）
- `TRAIN_SEEDS = [42, 52, 62, 72, 82]`，仅控制权重初始化/dropout/DataLoader shuffle/优化器随机性；每个 seed 独立 Python 进程，训练前立即 reset 全部 RNG。详见 `config/seed_protocol.yaml`。

## 超参数来源
`代码/main_experiment_3_fgds_psi_optimized.py::BEST_ARCH`（channels/gru_hidden/dropout/lr），`EPOCHS=120`，`PATIENCE=18`，`WEIGHT_DECAY=1e-5`，损失权重 `LAMBDA_STAGE/FINE/Q/MONO`。跨 5 个 seed 未调整。

## 输出文件说明
每个 `results/seed{N}/`：`metrics.csv`、`metrics.json`、`predictions.csv`（含 `pred/prob_E/M/L` 与 `q_hat/q_true` 磨损量估计）、`training_log.csv`（逐 epoch 训练历史）、`run_meta.json`（含四项 hash、B12_PARAMS、best_epoch/val 指标、训练耗时）、`DONE.flag`。汇总：`results/seed_level_results.csv`、`results/summary_mean_std.csv`（ddof=1）。

## 指标定义
Acc、Macro-F1、E/M/L-F1、M-Precision/Recall、M→E/M→L（Middle 阶段误判为 Early/Late 的比例）、Rev/Jump（阶段序列违反单调性的次数）、Smooth（阶段转移平滑度）、q-MAE/RMSE/R²（磨损量回归误差，由 `predictions.csv` 中 `q_hat` vs `q_true` 计算）。5 个 seed 全部指标均有效数值，无需以 NaN 占位。

## 是否使用目标域参与调参
否（见 Train/Val/Test 定义与 `audit/data_manifest.json`）。

## 相比旧实验做了什么变化
- 相比 `final_five_seed_sweep`（混合 RNG 主对比）：本轮预处理与训练种子完全解耦（详见 `audit/OLD_VS_NEW.md`）。
- 相比 `protocol_diagnostic_fixed_preprocess`（已做过同协议诊断）：本轮是**独立重新执行**同一严格协议（非复制旧文件），用于审计复现性；结果为逐位精确复现（bit-exact），见 `audit/OLD_VS_NEW.md`。
- 新增：`q-MAE/q-RMSE/q-R2` 指标（旧 fixed-preprocess 诊断未计算）；统一到本项目要求的目录结构与 `seed_level_results.csv`/`summary_mean_std.csv` schema。

## 当前状态
**DONE**（5/5 seed 完成，协议自检通过，审计对照完成）。

## 五个 Seed 的结果（Acc / Macro-F1）

| Seed | Acc | Macro-F1 | M-F1 | M-Recall | Smooth |
|---|---|---|---|---|---|
| 42 | 0.9868 | 0.9871 | 0.9844 | 0.9767 | 0.0189 |
| 52 | 0.6875 | 0.6659 | 0.4172 | 0.2636 | 0.0397 |
| 62 | 0.8651 | 0.8698 | 0.8111 | 0.6822 | 0.0354 |
| 72 | 0.7961 | 0.7956 | 0.6837 | 0.5194 | 0.0310 |
| 82 | 0.9770 | 0.9776 | 0.9728 | 0.9690 | 0.0166 |
| **Mean±Std (ddof=1)** | **0.8625±0.1261** | **0.8592±0.1341** | **0.7738±0.2349** | **0.6822±0.3045** | **0.0283±0.0102** |

**Seed sensitivity 仍然明显**（Acc std=0.126，range 0.6875–0.9868）；最差 seed 为 **52**（Middle 阶段 Recall 仅 0.264，M-F1 仅 0.417）。无 crash / NaN。无 protocol drift（详见 `audit/protocol_check.md`）。与历史 `protocol_diagnostic_fixed_preprocess` 完全一致（详见 `audit/OLD_VS_NEW.md`）。
