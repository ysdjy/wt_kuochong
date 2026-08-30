# 运行环境

记录时间：2026-08-29，`diagnostic/fixed-preprocess-5seed` 分支，commit `811da09`。

> **2026-08-30 更新**：新框架（`environment/environment.yml` + `requirements*.txt`）把下面两个历史环境
> 统一为单一 conda 环境 `wt_kuochong`（Python 3.11.15 + torch 2.7.1+cu118）——直接对比两个环境的
> `pip freeze` 输出确认 `pub_baselines` 是 `dcpsr` 的纯子集，无冲突版本，可安全合并。9 个方法的
> adapter 已在合并后的单一环境下验证过（同进程 import + smoke test 全部通过），不再需要区分两个环境
> 手动切换。新电脑请直接用 `scripts/bootstrap_windows.ps1` / `bootstrap_ubuntu.sh` 创建 `wt_kuochong`
> 环境，无需理会下面 `dcpsr`/`pub_baselines` 的历史记录（保留仅作审计参考）。

## Conda 环境

```
conda env list
base                 *   C:\Users\banghai\miniconda3
dcpsr                    C:\Users\banghai\miniconda3\envs\dcpsr
pub_baselines            C:\Users\banghai\miniconda3\envs\pub_baselines
(其余环境与本项目无关，略)
```

- `dcpsr`：内部方法（RF / TCN-GRU / Multi-task TCN-GRU / DC-PHSR(DC-PSR) / HTT-Net）训练环境。
- `pub_baselines`：已发表方法复现环境（Multi-source Attention / MTF-AViTK；历史上也用于 Dynamic GIN+TGP / DP2Net）。
- 本机 `conda activate` 在 Bash/git-bash 工具内不总是可靠切换，改用解释器全路径调用，例如：
  `C:\Users\banghai\miniconda3\envs\dcpsr\python.exe script.py`

## GPU

```
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8192 MiB total
```

任务启动前空闲（0 MiB used）。8GB 显存上限，MTF-AViTK（309M 参数 ViT-L/32）历史上耗时最长（约 1800–2400s/seed，曾出现数小时），本轮不涉及。

## Python

Base 环境 `python --version` → 3.13.13（实际训练使用上面列出的具体 conda env，而非 base）。

## 训练执行时长参考（用于任务分级：直接跑 vs 写教程）

- RF：CPU-only，秒级。
- TCN-GRU / Multi-task TCN-GRU / DC-PHSR(DC-PSR) inference：GPU，单 seed 约 1 分钟量级（`protocol_diagnostic_fixed_preprocess/results/dc_psr/seed42/run_meta.json` 记录 `training_seconds: 57.5`）。
- HTT-Net：GPU，几分钟量级。
- Multi-source Attention / DP2Net-adapted / Dynamic GIN+TGP：GPU，几分钟量级（Dynamic GIN+TGP 历史上曾观察到 ~185s/epoch，视具体协议差异可能更长）。
- MTF-AViTK：GPU，大模型，historically 数十分钟到数小时 —— **不在本轮范围内**；若未来涉及，按约定写手动教程而非由 Claude 直接执行。
