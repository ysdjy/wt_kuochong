# 扩充实验代码 (wt_kuochong)

本子项目是 DC-PSR / DC-PHSR 论文实验代码的**规范化重整版本**，同时也是一个独立推送到
[`wt_kuochong`](https://github.com/ysdjy/wt_kuochong) 的**可跨 Windows / Ubuntu 多机复现实验仓库**：
克隆本仓库、一键装环境、自动准备 PHM2010 数据，即可通过一条命令启动"某个方法 × PHM2010
三个任务 × 指定 seed 范围"的实验，无需依赖本机绝对路径或旧父项目。见 `MANUAL_RUN.md`。

它建立在仓库既有历史实验（`代码/`、`baselines/`、`experiments_mendeley/`、`final_five_seed_sweep/`、`protocol_diagnostic_fixed_preprocess/`、`final_statistical_evidence/`、`补充材料/小论文/`）之上，目标是：

1. 把散乱的历史实验资产整理成一个可审计、可复现、结构统一的目录；
2. 把论文最终确定的 9 个对比方法（B1–B9）在 **PHM2010** 上跑通所需的代码、配置、公共数据处理、
   环境文件、运行脚本全部整理成一个**独立、自包含、可多机并行、可断点续跑**的实验流水线。

## 重要约定

- **不修改任何旧代码原文件**。旧目录（`代码/`、`baselines/`、`final_statistical_evidence/` 等，均在本仓库
  之外的父项目中）全部只读，本项目只复制/vendor/引用，不回写。方法代码通过 `methods/Bx_xxx/code/` 物理
  vendored 副本自包含运行，不在运行时 import 父项目路径 —— 见各方法 `source_manifest.json`。
- **DC-PHSR = 论文新名称**；旧代码内部仍使用 `DC-PSR` / `B12` / `FGDS-PSI` 作为类名、变量名、脚本名 —— 通过
  adapter 层完成新旧名称映射，不对已验证的底层代码做无谓重命名。
- 中文目录名（`扩充实验代码`、`01_主对比实验` 等）作为最外层结构保持不变；新增的可移植框架部分
  （`methods/`、`shared/`、`scripts/`、`environment/`、`data/`、`results/`）统一使用英文目录名，
  路径统一用 `pathlib` 处理，不写死任何 Windows/Linux 绝对路径。
- 历史结果目录（`00_旧代码冻结备份/`、`01_主对比实验/PHM2010/B9_DC_PHSR/results/` 等）保留不动，新框架的
  正式结果写入独立的 `results/PHM2010/` 树，不覆盖任何历史结果。

## 目录导航

| 目录 | 内容 |
|---|---|
| `run_phm2010.py` | **统一入口**：`python run_phm2010.py --method B9 --tasks all --seed-start 0 --seed-end 100 --resume` |
| `MANUAL_RUN.md` | 新电脑（Windows / Ubuntu）从 clone 到出结果的完整手册，含 B1–B9 正式运行命令 |
| `RESULTS_POLICY.md` | 统一结果目录结构 / `run_meta.json` schema / seed 约定 / 汇总规则 |
| `ENVIRONMENT.md` | 本机环境记录（conda 环境、GPU、Python/torch 版本、历史执行时长参考） |
| `EXPERIMENT_REGISTRY.md` | 方法统一编号表、任务定义、当前轮次范围 |
| `environment/` | `environment.yml` / `requirements*.txt`（含每方法额外依赖 `methods/Bx.txt`） |
| `scripts/` | 一键装环境（`bootstrap_windows.ps1` / `bootstrap_ubuntu.sh`）、数据下载/校验/生成、自检、多机结果合并/汇总 |
| `data/` | `PHM2010/features/run_level_features_all.csv`（已提交，权威特征表）+ `PHM2010/raw/`（不入 Git，见下载脚本） |
| `shared/` | 全 9 方法共用：`phm2010/`（D1/D2/D3 任务注册表、公共评测宇宙）、`metrics/`（唯一权威指标实现+单测）、`runners/`（统一 adapter 接口 + 方法自动发现 + GPU 单发门控）、`utils/`（seed/run_meta）、`reproducibility/`（冻结 preprocessing） |
| `methods/` | 9 个方法各自的 `README.md` / `adapter.py` / `config.yaml` / `source_manifest.json` / `code/` / `tests/`，见下表 |
| `results/PHM2010/` | 正式运行的结果树（按 `RESULTS_POLICY.md` schema），大部分内容 gitignore，仅 `summary/*.csv` 入库 |
| `00_旧代码冻结备份/` | 历史代码/配置/协议文档的冻结拷贝 + 复制清单（manifest），只读审计用途，不作为运行入口 |
| `01_主对比实验/` | 早期轮次（B9/PHM2010/D1 5-seed）的历史结果，保留不动 |
| `02_跨任务与鲁棒性实验/` `03_消融实验/` `04_表征与语义实验/` `05_统计检验/` | 早期轮次的历史/诊断资产，见各自 README（部分未启动，部分含正在进行中的诊断实验） |
| `logs/` | 顶层运行日志 |

## 方法统一编号

见 `EXPERIMENT_REGISTRY.md`；每个方法的详细说明（来源论文、旧代码位置、输入形式、预处理、超参数）见其
`methods/Bx_xxx/README.md`。

## 快速开始

```bash
git clone https://github.com/ysdjy/wt_kuochong.git && cd wt_kuochong/扩充实验代码
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_windows.ps1   # Windows
# 或
bash scripts/bootstrap_ubuntu.sh                                          # Ubuntu

conda run -n wt_kuochong python scripts/self_check.py
conda run -n wt_kuochong python run_phm2010.py --method B9 --tasks all --seed-start 0 --seed-end 100 --device auto --workers 1 --resume
```

完整手册（含 9 个方法的正式命令、多机分片、结果合并、已知限制）见 `MANUAL_RUN.md`。

## 当前状态（2026-08-30）

- 目录规范化：**DONE**
- 旧代码冻结备份：**DONE**（1,473 文件复制 353.5MB，33,757 文件 manifest-only，见 `00_旧代码冻结备份/README.md`）
- 早期轮次 B9 / PHM2010 / D1 5-seed：**DONE**（历史结果，详见 `01_主对比实验/PHM2010/B9_DC_PHSR/README.md`）
- **PHM2010 可移植实验框架（B1–B9 adapter + 统一 runner + 环境/数据脚本）：DONE**
  （9/9 方法 adapter 已构建并通过同进程集成 smoke test，见 `MANUAL_RUN.md` "九方法状态"表）
- B1–B9 在 D1/D2/D3 × seed 0–100 的正式大规模训练：**本轮未启动**（仅做 plumbing smoke test + B1 的小规模真实验证，未经用户重新授权不启动 9×3×101 全量实验）
- 其它数据集（NASA_Milling / MTW_CM）、消融、统计检验：**NOT_STARTED**
