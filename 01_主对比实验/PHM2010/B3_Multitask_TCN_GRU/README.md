# B3 — Multi-task TCN-GRU — PHM2010 — D1

## 方法编号与全称
B3 = Multi-task TCN-GRU（B9/DC-PHSR 的 backbone）

## 方法类别
深度学习基线 / B9 的直接前置 backbone（非独立本文方法）

## 原始代码来源
`代码/main_experiment_3_fgds_psi_optimized.py::train_model`（架构 `TCNGRUMultiTask`）

## 新目录中的代码位置
与 B9 共用 `../B9_DC_PHSR/code/run_seed_d1.py`（同一次训练产生 B3 原始输出与 B9 推理后输出，B3 不单独训练）

## 数据集 / Train-Val-Test / Preprocessing / Seed 定义
与 B9 完全一致，见 `../B9_DC_PHSR/README.md`（此方法与 B9 共享同一次训练运行）

## 输出文件说明
每个 `results/seed{N}/`：`metrics.csv`、`predictions.csv`（B3 自身原始 raw 概率，未经 B9 的确定性后处理）、`training_log.csv`、`run_meta.json`（含 checkpoint `multitask_tcn_gru_seed{N}.pth`）、`DONE.flag`。**注**：本方法未生成 `metrics.json`/`seed_level_results.csv`/`summary_mean_std.csv`（本轮任务范围仅要求 B9 的完整规范化汇总；B3 结果作为 B9 的必要中间产物一并保留、供审计使用）。

## 当前状态
**DONE**（作为 B9 训练的必然副产物，5/5 seed 完成；本身不是本轮的目标交付物）。

## 五个 Seed 的结果（B3 自身原始输出，未经 B9 后处理，供对照）

| Seed | Acc | Macro-F1 |
|---|---|---|
| 42 | 0.9868 | 0.9871 |
| 52 | 0.6743 | 0.6499 |
| 62 | 0.8487 | 0.8539 |
| 72 | 0.7796 | 0.7771 |
| 82 | 0.9868 | 0.9871 |

与旧 `protocol_diagnostic_fixed_preprocess` 的 B3(multitask_tcn_gru) 结果逐位精确复现，一致性核对同 `../B9_DC_PHSR/audit/OLD_VS_NEW.md`。
