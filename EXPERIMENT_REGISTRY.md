# 实验方法与任务注册表

## 方法统一编号（本子项目内固定使用）

| 编号 | 名称 | 类别 | 旧代码位置（只读来源） | 备注 |
|---|---|---|---|---|
| B1 | RF | 传统机器学习基线 | `代码/7.4对比实验.py`；报告见 `baselines/rf/` | |
| B2 | TCN-GRU | 深度学习基线 | `代码/7.4对比实验.py`（旧版共享 RNG，见诊断说明）；报告见 `baselines/tcn_gru/` | 旧 `main()` 中 B8→B9→B10 共享一条 RNG stream，历史 seed42 数值受污染，见 `05_统计检验` 相关诊断资产 |
| B3 | Multi-task TCN-GRU | DC-PHSR 的 backbone | `代码/7.4对比实验.py` / `代码/main_experiment_3_fgds_psi_optimized.py` 相关训练逻辑；报告见 `baselines/multitask_tcn_gru/` | B9 直接复用此 backbone 的逐 seed checkpoint，不单独调参 |
| B4 | HTT-Net (adapted) | 已发表方法复现 | `baselines/htt_net/` | |
| B5 | Multi-source Channel-Spatial Attention | 已发表方法复现 | `baselines/multi_source_attention/` | |
| B6 | MTF-AViTK | 已发表方法复现 | `baselines/mtf_avitk/` | 309M 参数 ViT-L/32，训练耗时长，历史上曾人工跑 seed72/82 |
| B7 | Dynamic GIN + TGP | 已发表方法复现 | `baselines/dynamic_gin_tgp/` | |
| B8 | DP2Net-adapted | 已发表方法复现 | `baselines/dp2net/` | |
| **B9** | **DC-PHSR**（论文新名称） | **本文方法** | `代码/main_experiment_3_fgds_psi_optimized.py` + 冻结推理逻辑；旧代码内部类名/脚本名仍为 `DC-PSR` / `B12` / `FGDS-PSI` | 在 B3 backbone 输出上做确定性概率阶段推理，不独立训练新 backbone |

> **命名对照**：论文正文一律使用 `DC-PHSR`；本目录及旧代码中出现的 `DC-PSR`、`B12`、`FGDS-PSI` 均指同一方法，历史遗留标识符，本轮不重命名底层代码。

> **新框架入口**（Round 2，本轮）：以上"旧代码位置"仅为溯源信息（只读来源，已 vendor 到各方法自己的
> `code/` 中）。实际运行统一通过 `methods/Bx_xxx/adapter.py`（`run_phm2010.py --method Bx` 调度），
> 完整清单：`methods/B1_RF/`、`methods/B2_TCN_GRU/`、`methods/B3_Multitask_TCN_GRU/`、
> `methods/B4_HTT_Net/`、`methods/B5_MultiSource_Attention/`、`methods/B6_MTF_AViTK/`、
> `methods/B7_Dynamic_GIN_TGP/`、`methods/B8_DP2Net/`、`methods/B9_DC_PHSR/`。B2/B3/B9 共享
> `methods/_internal_shared/code/pipeline.py`（避免重复约 900 行预处理逻辑）。

## 数据集

| 数据集 | 状态 |
|---|---|
| PHM2010 | 本轮启动（仅 B9 / D1） |
| NASA_Milling | NOT_STARTED |
| MTW_CM | NOT_STARTED |

## 主任务定义（PHM2010）

| 任务 | Train | Test | 含义 |
|---|---|---|---|
| **D1** | C1 + C4 | C6 | 完整寿命数据训练，未参与训练的刀具上测试（本轮唯一启动的任务） |
| D2 | C1 + C6 | C4 | 跨工况迁移（NOT_STARTED，本轮不跑） |
| D3 | C4 + C6 | C1 | 跨工况迁移（NOT_STARTED，本轮不跑） |

## Round 1 范围（历史，已完成）

**仅**运行：B9 / PHM2010 / D1 / TRAIN_SEEDS=[42,52,62,72,82] / PREPROCESS_SEED=42（固定复用冻结 preprocessing）。
结果见 `01_主对比实验/PHM2010/B9_DC_PHSR/`，保留不动。

## Round 2 范围（2026-08-30，本轮）：可移植 PHM2010 实验框架

把 B1–B9 全部 9 个方法在 PHM2010 上运行所需的代码、配置、预处理、环境、运行脚本整理进
`methods/`、`shared/`、`environment/`、`scripts/`、`run_phm2010.py`（详见根 `README.md` 目录导航
与 `MANUAL_RUN.md`）。9 个方法均已构建 adapter 并通过同进程 CPU-only smoke test（`--smoke-test`）；
B1 额外做了一次小规模真实验证（D1/D2/D3 × seed 0，非 smoke，验证完整 pipeline）。

**本轮明确不启动** B1–B9 × D1/D2/D3 × seed 0–100 的全量正式训练（9×3×101 = 2727 个 run）——
smoke test 之外的任何真实训练需用户重新明确授权。

其余数据集（NASA_Milling / MTW_CM）、消融实验、统计检验状态仍为 `NOT_STARTED`。
