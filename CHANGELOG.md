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
- Headline（N=35 阶段）：Acc = 0.8680±0.0728（ddof=1，range 0.6875–0.9901）。Pareto 前沿 {21,42,82}；balanced seed=82。最差3个：{28,0,52}；seed52 确认为真实 bad basin，与其它坏 seed 共享"Middle 阶段判别失败"特征但误判方向不同。
- 追加 OverallRank 多指标综合排序（Acc/Macro-F1/M-F1/M-Recall/Smooth 五指标各自排名取平均，未加权），产出 `overall_rank_seed_table.csv`、`top5_seeds_mean_std.csv`、`top5_seeds_summary.md`（诊断用途，明确不替代论文正式固定 seed mean±std）。

## 2026-08-30（续）— 扩充至 seed 0–99 全覆盖（用户追加任务）

- 用户要求补齐随机种子 0–99 的完整覆盖；追加运行 65 个新 seed（30–41,43–51,53–61,63–71,73–81,83–99），沿用已验证稳定的 3 路并发，**65/65 全部成功，零失败**。
- 总计 **100/100 seed 完成**，全部重新聚合分析（`analyze_seed_landscape.py`、`rank_top5_seeds.py` 均已在 N=100 上重跑）。
- Headline（N=100）：Acc = 0.8632±0.0717（ddof=1，range 0.6875–0.9901），与 N=35 阶段（0.8680±0.0728）基本一致，方差估计稳定。Pareto 前沿扩大为 {21,42,50,53,82}；balanced seed 变为 **50**。最差3个变为 {39,49,52}（seed52 仍在其中，确认非小样本偶然）。OverallRank Top-5 = {42,50,82,9,87}。

## 2026-08-30（续）— methods/B6_MTF_AViTK 适配器（跨机复现工程化任务）

- 新增 `methods/B6_MTF_AViTK/`：vendored code（`kan.py`/`model.py` 原样拷贝，`preprocessing.py` 仅改数据路径为 `PHM2010_ROOT` 环境变量，`label_utils.py` 合并旧 `data/label_utils.py` + `代码/main_experiment_3_fgds_psi_optimized.py` 的标签常量并将 `split_grouped_lifecycle` 从硬编码 C1+C4→C6 泛化为任意 `(train_cutters, test_cutter)`，`train_core.py`/`data_prep.py`/`adapter.py` 为新文件），全部改动记录在 `source_manifest.json`。
- 图像存储策略改动：旧项目预生成全部 945 run × 5 子窗口（2.0GB `data/images/*.npy`）从未被提交；新版按需生成 + 本地缓存（`data/PHM2010/derived/mtf_avitk_images/`），不改变生成算法本身。
- CPU-only smoke test（`tests/smoke_test.py`）4 级全部通过：模块导入、模型实例化（309,371,072 参数）、单张真实图像前向（真实 C1 run1 原始信号 → 真实小波去噪 → 真实 MTF 编码 → 真实模型前向）、完整 adapter smoke-test 路径（`run(smoke_test=True)` 返回 `status=done`）。**全程未执行任何真实训练循环**。
- 发现共享代码 bug（未修复，超出本次任务范围 `methods/B6_MTF_AViTK/`）：`shared/utils/run_meta.py::_torch_cuda_gpu_info()` 在 `CUDA_VISIBLE_DEVICES=""` 时 `torch.cuda.is_available()` 仍返回 `True` 但 `device_count()==0`，导致 `get_device_name(0)` 抛 `AssertionError: Invalid device id`——影响所有方法的 CPU-only 强制执行路径，非 B6 专属问题，详见该方法 README.md「Blockers found」。（**已于本轮后续统一修复，见下）

## 2026-08-30（续）— PHM2010 可移植实验框架整体交付（跨机复现工程化任务，Round 2）

- 新增 `shared/phm2010/`（`tasks.py` D1/D2/D3 权威注册表 + 泄漏防护、`evaluation_universe.py` 公共评测宇宙 run_id 12–315/n=304，D1/D2/D3 三任务下逐一实测验证一致，非假设）、`shared/metrics/metrics.py`（唯一权威指标实现，移植自 `paper_data/99_scripts/build_paper_data.py::recompute_transfer_metrics`，对真实 RF D1 预测重算后与冻结 `D1_MAIN_BOOTSTRAP_CI.csv` **逐位精确匹配**）+ 13 条单测、`shared/utils/{seeding,run_meta}.py`、`shared/runners/{method_adapter,registry,gpu_gate}.py`（统一 adapter 模板 + 方法自动发现 + 单发 GPU 门控）。
- 新增 `methods/B1_RF/`（本轮由协调者本人直接实现，非子代理）、`B2_TCN_GRU/`、`B3_Multitask_TCN_GRU/`、`B4_HTT_Net/`、`B5_MultiSource_Attention/`、`B6_MTF_AViTK/`（见上一条目）、`B7_Dynamic_GIN_TGP/`、`B8_DP2Net/`、`B9_DC_PHSR/`（后 8 个由 5 组并行子代理分别构建，均已核实产物而非仅采信自述）——每个方法含 `README.md`/`adapter.py`/`config.yaml`/`source_manifest.json`/`code/`（vendored，不在运行时 import 旧父项目路径）/`tests/`。B2/B3/B9 共享新增的 `methods/_internal_shared/code/pipeline.py`，避免约 900 行重复；其预处理输出的 45 个选中特征与已冻结的 `shared/reproducibility/PHM2010_D1_frozen_preprocess/selected_features_seed42.json` 逐位精确匹配（另一独立正确性验证）。B1 的完整 D1 结果（n_estimators=400 全量，非缩减）与冻结的 `D1_MAIN_BOOTSTRAP_CI.csv` 12 项指标中 10 项逐位精确匹配，Rev/Jump 精确匹配，Smooth 相差 <0.2%（推测为 sklearn `n_jobs=-1` 树间浮点求和顺序导致，非流水线 bug）。
- **发现并修复的 3 个真实 bug**（均为协调者本人发现/修复，非仅由子代理自述）：
  1. `shared/runners/registry.py`：多方法在同一进程内加载时，`model.py`/`preprocessing.py` 等通用命名的方法私有模块通过 `sys.modules` 缓存互相覆盖（B6/B7/B8 各自的 `model` 被 B4 先导入的 `model` 顶替）——修复为按方法目录路径精确过滤，仅清理 `methods/` 路径下的本地模块缓存，不清理任何第三方/标准库模块（曾误清理 `numpy` 等导致 `"cannot load module more than once per process"` 崩溃，已修正为路径范围精确过滤）。修复后 9/9 方法在单一 conda 环境、单一 Python 进程内可全部正确导入且互不干扰（同进程集成测试全部 `status: done`）。
  2. `shared/utils/run_meta.py::_torch_cuda_gpu_info()`：增加 `torch.cuda.device_count() > 0` 判断，修复 B6 fork 报告的 CPU-only 强制执行崩溃。
  3. `run_phm2010.py`：`--smoke-test` 未正确路由输出目录，会写入正式 `results/`（违反 section 40 "smoke test 输出不得混入 results/"）——修复为 smoke test 统一写入 `tmp/smoke_tests/`；`shared/runners/method_adapter.py` 同时补充：成功写入 `DONE.flag` 时清理同目录下可能残留的 `FAILED.flag`/`error.log`（避免重试成功后仍留有旧失败痕迹）；`scripts/aggregate_results.py` 修复了两处 pandas 聚合 bug（NaN 指标误判为"未完成"、`pivot_table` 默认丢弃全 NaN 列导致 KeyError）——均为方法本身没有连续 q 输出（q_MAE/RMSE/R2 legitimately NaN）时触发。
- 新增 `environment/`（`environment.yml` + `requirements*.txt` + 每方法 `methods/Bx.txt`，两个历史 conda 环境经 `pip freeze` 直接比对确认可安全合并为统一 `wt_kuochong` 环境）、`scripts/`（`bootstrap_{windows.ps1,ubuntu.sh}`、`verify_environment.py`、`self_check.py`、`download_phm2010.py`+`verify_phm2010.py`（数据源为 Kaggle `tobbyrui/phm2010`，唯一在旧项目文档中出现过的具体来源，非凭空编造）、`build_phm2010_features.py`（非位精确的 fallback 特征生成器，明确标注不可替代已提交的权威特征表）、`download_assets.py`（审计确认 9 个方法均无需外部预训练权重，为文档化 no-op）、`merge_results.py`、`aggregate_results.py`）。
- 新增 `data/PHM2010/features/run_level_features_all.csv`（5.4MB，sha256 校验与旧项目源文件一致，已提交入库；`data/PHM2010/raw/` 不入库，18GB+，由 `download_phm2010.py` 获取）。
- 本轮**未**启动 B1–B9 × D1/D2/D3 × seed 0–100 的全量正式训练（9×3×101=2727 run）；仅做 CPU-only plumbing smoke test（全部 9 方法通过）+ B1 一次小规模真实验证（非 smoke，D1/D2/D3 × seed 0）。MTF-AViTK 全程未执行任何真实训练循环，遵循既定策略（大/慢 GPU 任务写教程，不由 Claude 直接执行）。
