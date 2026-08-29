# 00_旧代码冻结备份

历史实验资产的只读冻结拷贝。**全部视为只读历史资产，后续实验不得回写。**

## 冻结来源分支
`diagnostic/fixed-preprocess-5seed`（commit `811da09`，`main` 的严格超集，见根目录 `CHANGELOG.md`）。

## 审计并复制的历史目录

| 源目录 | 归类 | 说明 |
|---|---|---|
| `代码/` | 核心方法旧代码 | DC-PSR/DC-PHSR + 全部方法主脚本 |
| `baselines/` | 基线方法旧代码 | 8 个已发表/内部基线方法代码与报告 |
| `experiments_mendeley/` | 统计与诊断旧代码 | Mendeley 数据集审计/复现全流程 |
| `final_five_seed_sweep/` | 统计与诊断旧代码 | 冻结的 9 方法 5-seed（混合 RNG）主对比结果与协议 |
| `protocol_diagnostic_fixed_preprocess/` | 统计与诊断旧代码 | fixed-preprocessing/training-seed 隔离诊断（本轮 B9 的直接对照基准） |
| `final_statistical_evidence/` | 跨任务实验旧代码 | D1 bootstrap CI + D1/D2/D3 mean±std 统计证据 |
| `补充材料/小论文/` | 跨任务实验旧代码 | 第四/五章图表与补充实验旧代码 |

## 复制规则

- **复制**：代码（`.py`/`.ipynb`）、文档（`.md`/`.txt`/`.rst`）、配置（`.yaml`/`.yml`/`.json`/`.toml`/`.cfg`）、结果表格（`.csv`），且单文件 < 20MB。
- **仅 manifest（不复制）**：模型 checkpoint（`.pth`/`.pt`/`.ckpt`）、压缩包（`.zip`/`.7z`）、原始数组/图像转储（`.npy`）、任何路径中含 `data`/`images`/`3_models` 段的目录（原始数据集或已知的大体量数据/checkpoint 目录）、以及所有 ≥20MB 的文件。
- 完整清单见 `manifests/LEGACY_COPY_MANIFEST.csv`（`source_path,destination_path,type,size_bytes,sha256,reason,status`）。

## 统计

- 扫描文件总数：35,230
- 复制：1,473 个文件，共 353.5 MB
- 仅 manifest（未复制）：33,757 个文件

**未复制的主要大体量项**（均只在 manifest 中记录路径+哈希/大小，未复制内容）：
- `baselines/mtf_avitk/data/` 下数千个图像 `.npy` 文件（~2GB）
- `final_statistical_evidence/transfer_tasks/{D2,D3}/mtf_avitk|multi_source_attention|htt_net/` 下的大型 checkpoint（单文件最大 3.5GB，合计约 11GB）
- `protocol_diagnostic_fixed_preprocess/results/*/seed*/*.pth`（B3/TCN-GRU/HTT-Net 各 seed 的模型权重，均较小但按规则统一 manifest-only）
- `baselines/htt_net/data/run_level_features_all.csv`（权威特征表原始文件，属于"数据"类，manifest-only；其冻结派生产物已复制/镜像到 `../shared/reproducibility/PHM2010_D1_frozen_preprocess/`）

## 是否修改了任何旧文件
**否。** 所有操作均为读取源文件后 `copy2`（保留 mtime，不回写源）；本轮结束前已用 `git status` 核对全部源目录（`代码/`、`baselines/`、`experiments_mendeley/`、`final_five_seed_sweep/`、`protocol_diagnostic_fixed_preprocess/`、`final_statistical_evidence/`、`补充材料/`）均无改动。
