# OLD vs NEW — B9 (DC-PHSR / legacy DC-PSR / B12) / PHM2010 / D1

## 对照的三份历史资产

1. `final_five_seed_sweep/`（2026-08-20 冻结）— 原始 9 方法 5-seed 主对比，**混合 RNG**：`代码/7.4对比实验.py::main()` 只在最开始调用一次 `set_seed`，B8→B9→B10 串行训练共享同一条 RNG stream；B12/DC-PSR 通过 `base.train_model()` 内部自带的 `set_seed` 调用做到了自隔离（未受此确认的污染，但仍与本轮"每 seed 独立进程"的隔离程度不同）。
2. `protocol_diagnostic_fixed_preprocess/`（2026-08-20）— 专门为验证"预处理/训练种子混淆假设"而做的**已严格隔离**诊断实验：固定 `PREPROCESS_SEED=42`，`TRAIN_SEED∈{42,52,62,72,82}` 通过 `run_diagnostic_seed.py` 每次训练前重置 RNG。
3. `final_statistical_evidence/`（2026-08-20）— D1 复用（未重新训练），D2/D3 新训练；不涉及本轮的 training-seed 方差问题，仅作交叉核对来源。

## Q1：新实验是否复现旧 seed42？

**是，逐位精确复现（bit-exact）。**

| 指标 | 旧 fixed-preprocess 诊断 (seed42) | 本轮新实验 (seed42) |
|---|---|---|
| Acc | 0.9868421052631579 | 0.9868421052631579 |
| Macro-F1 | 0.9871020928241414 | 0.9871020928241414 |
| feature/split/gmm/window hash | 同 `hash_manifest.json` 记录值 | 完全一致 |
| B12_PARAMS | eta=0.75, fine_weight=0.30, temperature=1.20, mid_floor=0.12, late_tau=0.66, early_tau=0.38, order_blend=0.25 | 完全一致 |

## Q2：seed52/62/72/82 与旧 fixed-preprocess diagnostic 是否一致？

**是，5/5 个 seed 全部逐位精确复现（Acc 到全部可见小数位相同）：**

| Seed | 旧 fixed-preprocess Acc | 本轮新实验 Acc | 一致 |
|---|---|---|---|
| 42 | 0.986842 | 0.9868421052631579 | ✅ |
| 52 | 0.687500 | 0.6875 | ✅ |
| 62 | 0.865132 | 0.8651315789473685 | ✅ |
| 72 | 0.796053 | 0.7960526315789473 | ✅ |
| 82 | 0.976974 | 0.9769736842105263 | ✅ |
| **Mean±Std (ddof=1)** | **0.8625±0.1261** | **0.8625±0.1261** | ✅ |

**结论**：在固定 `torch.backends.cudnn.deterministic=True` / `use_deterministic_algorithms(True, warn_only=True)` 且预处理完全冻结的前提下，本机（同一 GPU/驱动/PyTorch 版本、同一 `代码/` 版本，commit `811da096ee47bea4f65db193aa49e793dba6f47d`）对该小模型的训练是**完全确定性**的——重新执行同一协议不引入任何额外方差来源。这本身是一个有用的正向确认：本轮观测到的 seed 间方差（Acc std=0.126）**完全来自训练优化过程本身的 seed 敏感性**，不是不可控的非确定性噪声或本次执行环境的随机干扰。

## Q3：与旧 `final_five_seed_sweep`（混合 RNG，原始主对比）的差异

| Seed | 旧混合RNG sweep Acc (final_five_seed_sweep) | 本轮/fixed-preprocess Acc | 差异 |
|---|---|---|---|
| 42 | 0.9868 | 0.9868421 | ≈0（一致） |
| 52 | 0.8586 | 0.6875 | **-0.1711** |
| 62 | 0.7993 | 0.8651 | +0.0658 |
| 72 | 0.8849 | 0.7961 | -0.0888 |
| 82 | 0.9671 | 0.9770 | +0.0099 |
| Mean±Std | 0.8993±0.0776 | 0.8625±0.1261 | Std 增大 62% |

**差异来源**：不是 preprocessing、不是代码版本、不是 CUDA/PyTorch determinism 行为变化——三者均已核对一致（见下）。差异纯粹来自 **RNG 隔离方式的不同**：
- `final_five_seed_sweep` 中，即便 DC-PSR/B12 自身通过 `base.train_model()` 的内部 `set_seed()` 做到了"训练开始前重置"，但该次 sweep 的预处理（MI 特征选择、GMM 拟合）本身也用同一个 `RANDOM_SEED` 驱动，即预处理和训练种子是**混淆**的——改变 seed 同时改变了"用什么特征/GMM"和"怎么训练"两件事。
- 本轮（与 `protocol_diagnostic_fixed_preprocess` 相同）把预处理完全冻结在 `PREPROCESS_SEED=42`，只有训练种子变化——这是一个更纯净、方差更大的度量，说明**原方差里有一部分被预处理的随 seed 变化"平均掉"了**；固定预处理后，训练本身对初始化/dropout/shuffle 更敏感这件事被更清晰地暴露出来。
- 这与 2026-08-20 的诊断结论（`protocol_diagnostic_fixed_preprocess/FINAL_DIAGNOSTIC_REPORT.md`，"Conclusion C: TRAINING-SEED SENSITIVITY REMAINS"）完全吻合，本轮独立重跑再次验证了该结论，而非仅仅复制旧文件。

## Q4：代码版本 / hash / determinism 行为是否发生变化？

- **代码版本**：本轮运行时 `git_commit = 811da096ee47bea4f65db193aa49e793dba6f47d`（当前分支 `diagnostic/fixed-preprocess-5seed` HEAD）；旧 fixed-preprocess 诊断运行时 `git_commit = 916cdd6bb810820eded603ed81fd065065d3cc08`（`main` 冻结提交）。两次运行之间 `代码/` 目录**无任何改动**（`git status`/`git log -- 代码/` 均确认自 916cdd6 以来无提交触及该目录），因此代码版本号不同但代码内容相同，不构成差异来源。
- **hash**：feature/split/gmm/window 四项 hash 两次运行完全一致（见上）。
- **CUDA/PyTorch determinism 行为**：两次运行使用同一台机器、同一 GPU（RTX 3070 Ti Laptop）、同一 conda 环境 `dcpsr`，未观察到任何 determinism 相关警告差异或数值漂移的证据（bit-exact 结果本身就是最强的证据）。

## 结论

新实验不仅"复现"了旧结果，而且以**独立、结构更规范、可审计**的方式确认了旧结论：B9/DC-PHSR 在固定预处理下对 training seed 高度敏感（Acc std ≈ 0.126，seed52 明显劣化到 0.6875），这不是预处理混淆或环境噪声造成的假象，而是训练优化过程本身的真实不稳定性。
