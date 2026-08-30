# MTW-CM — Task & Data Audit

日期：2026-08-30。

## 1. 数据位置核实

`C:\Users\banghai\Documents\BaiduSyncdisk\西工大\王婷\论文\Multivariate time series data of milling processes with varying tool wear and machine tools\`
- 6419 个 `.h5` 文件 + `filelist.csv`，文件名编码 `M{machine}T{tool}R{run}C{cumulated_contact_time}VB{wear}.h5`。
- **3 台机床 × 3 把刀具 = 9 把刀具**：`M1={T1,T2,T3}, M2={T4,T5,T6}, M3={T7,T8,T9}`（程序核实，非人工猜测）。
- 与既有 `experiments_mendeley/00_dataset_audit/` 审计结果一致（`channel_summary.csv`/`hdf5_schema_report.json`/`stage_coverage_by_tool.csv` 等已存在，未重新生成）。
- 不重新下载；不解压（原生 .h5）；不把 ~10GB 原始数据复制进 `扩充实验代码/` 或 Git。

## 2. D1-M/D2-M/D3-M 任务映射（权威来源）

来自已冻结的 `experiments_mendeley/02_protocols/task_definitions.json`（已有真实历史运行结果佐证：`experiments_mendeley/03_sanity_check/runs/D1-M/seed42/`、`experiments_mendeley/04_overall_comparison/runs/D1-M|D2-M|D3-M/seed42/` 均存在真实 metrics/predictions）：

| Task | Train machines | Test machine | Train tools | Test tools |
|---|---|---|---|---|
| D1-M | M1, M2 | M3 | T1-T6 | T7,T8,T9 |
| D2-M | M1, M3 | M2 | T1,T2,T3,T7,T8,T9 | T4,T5,T6 |
| D3-M | M2, M3 | M1 | T4-T9 | T1,T2,T3 |

（该 JSON 中另有 `MS1-MS6` 单机床迁移与 `LOTO_T1-T9` 留一刀具任务，本轮不使用，仅 D1-M/D2-M/D3-M。）

结构校验：train machines ∩ test machine = ∅，且每个 task 的 train+test 机床覆盖全部 3 台机床。**PASS**。

## 3. 复用代码

`experiments_mendeley/code/dcpsr/`：模块化实现（`datasets/mendeley.py` 原始 HDF5 解析、`features.py`/`online_features.py` 特征工程、`selection.py` train-only 特征选择、`stages.py` 阶段标签、`splits.py` task 切分、`model.py` TCN-GRU 多任务架构、`inference.py` DC-PSR/B12 概率推理、`metrics.py` 统一指标、`runner.py` 训练/评估编排）+ `experiments_mendeley/code/scripts/00_extract_features.py`（特征提取，一次性，带缓存）+ `01_run_experiments.py`（`--phase dual --tasks D1-M,D2-M,D3-M --seeds ... --resume`，已内建断点续跑）。

**未修改任何原文件**；`raw_dir`/`out_dir` 均为显式参数（无硬编码绝对路径），因此可以安全地把 `--out-root` 指向 `扩充实验代码/` 内部新目录，不写回 `experiments_mendeley/`。

已存在的特征缓存 `experiments_mendeley/01_features/`（61MB，`00_extract_features.py` 的既有产出）**只读复制**到本项目内部，避免重新扫描 6419 个 HDF5 文件（预计耗时远长于复用缓存）。

## 4. Seed 协议

`experiments_mendeley/code/dcpsr/config.py::FINAL_SEEDS = [42,52,62,72,82]` 是旧的 5-seed 集合；本轮按用户要求扩展为 `--seeds 0,1,...,100`（101 个），`01_run_experiments.py` 原生支持 `--seeds` 覆盖参数，且天然 `--resume`。

## 5. 结果位置

`扩充实验代码/05_统计检验/seed_statistics/B9_MTW_D1M_seed_landscape/`、`B9_MTW_D2M_seed_landscape/`、`B9_MTW_D3M_seed_landscape/`，与 PHM/NASA 同构（各自 `results/seed{N}/`）；实际写出由 `01_run_experiments.py --out-root` 统一到一个中间目录，随后视需要整理/软链到上述三个 landscape 目录（详见执行记录）。

## 6. 状态

数据与协议审计：**PASS**。尚未执行 smoke test / 正式训练（详见 `OVERNIGHT_B9_EXTERNAL_STATUS.md` 当前进度）。
