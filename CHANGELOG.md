# CHANGELOG

## 2026-08-29 — 子项目初始化 + Round 1 启动

- 审计 `main` 与 `diagnostic/fixed-preprocess-5seed` 两分支，确认以 `diagnostic/fixed-preprocess-5seed`（commit `811da09`）作为旧代码冻结来源（`main` 是其祖先子集）。
- 新建 `扩充实验代码/` 目录规范结构（00–05 + shared + logs）。
- 旧代码冻结备份：见 `00_旧代码冻结备份/manifests/LEGACY_COPY_MANIFEST.csv`。
- 启动 Round 1：仅 B9 (DC-PHSR) 在 PHM2010 D1 上的严格 5-seed (TRAIN_SEEDS=[42,52,62,72,82], PREPROCESS_SEED=42) 训练。
- 工作区状态记录：启动本轮工作时，仓库工作区存在约 80 个与本任务无关的未提交改动（集中在 `paper_data/New_figure/` 图表制作相关文件），经用户确认后按"保留原样"处理，未 touch。
- Round 1 完成：B9 5/5 seed（42/52/62/72/82）逐位精确复现历史 `protocol_diagnostic_fixed_preprocess` 结果；旧代码冻结备份完成（1,473 文件复制，33,757 文件 manifest-only）。

## 2026-08-30 — B9 Training-Seed Landscape 诊断（用户追加任务）

- 范围：TRAIN_SEED ∈ {0..29} ∪ {42,52,62,72,82}（35 个 seed），仅诊断用途，不作为正式论文主表结果。
- 输出目录：`05_统计检验/seed_statistics/B9_PHM2010_D1_seed_landscape/`。
- 复用同一冻结 preprocessing、同一超参数、同一 B12/DC-PHSR 推理参数，仅 TRAIN_SEED 变化。
- **完成**：35/35 seed 全部成功（无删除、无跳过）。执行中途 8 路并发触发 CUDA OOM（17 个 seed 失败），已终止、清理、以 3 路并发重跑，第二批 29/29 全部成功，过程记录在案（见该目录 README 的"执行过程说明"）。
- 交付：`seed_level_results.csv`、`seed_ranking.csv`、`pareto_seeds.csv`、`best_seeds_summary.md`、`good_vs_bad_seed_analysis.md`、`seed_landscape_preview.png`。
- Headline：Acc = 0.8680±0.0728（N=35，ddof=1，range 0.6875–0.9901）。Pareto 前沿 {21,42,82}；balanced seed=82。最差3个：{28,0,52}；seed52 确认为真实 bad basin，与其它坏 seed 共享"Middle 阶段判别失败"特征但误判方向不同。
