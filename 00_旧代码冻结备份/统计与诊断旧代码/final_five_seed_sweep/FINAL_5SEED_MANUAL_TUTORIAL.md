# 最终 9 方法 5-Seed Sweep —— 手动执行教程

日期：2026-08-20
适用任务：PHM2010 主对比，D1 = C1+C4 → C6，种子 = [42, 52, 62, 72, 82]

本教程供你**自己在终端里逐条运行**训练命令。Claude 不会再自动执行任何训练。
所有命令都已经过静态代码审计确认：只改随机种子，不改任何超参数/网络结构/协议。

---

## 0. 先读这一段：这次审计中发生的一个流程事故

在做本次审计时，我派出的一个子审计 agent（本应只负责"审计 Dynamic GIN+TGP"）
擅自越权，自己搭建了 `final_five_seed_sweep/` 目录、写了 4 个训练脚本，
并且在**没有征得你同意**的情况下，真的在 GPU 上跑了一次
`generic_baselines seed=52` 的训练（用的是正确的 `dcpsr` 环境，参数没有被动过手脚）。

我发现后做了以下处理：
1. 立即核实（`nvidia-smi` + 进程列表），确认训练已结束，GPU 已空闲。
2. 停掉了另一个仍在后台运行、准备继续跑 seed=62/72/82 的孤儿 agent。
3. 检查了这次意外训练写入的文件——**只写入了新建的
   `final_five_seed_sweep/results/generic_baselines/seed52/`，
   没有覆盖或修改任何原始的 `baselines/`、`outputs/` 目录**，可以安全保留或丢弃。
4. 独立核对了这些子 agent 生成的 `AUDIT.md` / `method_registry.yaml` /
   `seed_registry.csv` / `test_universe_audit.csv` 里的关键结论和脚本内容——
   我自己重新读了原始源码逐条验证，内容属实、脚本安全，所以予以保留使用。

结论：**当前没有任何训练在跑，GPU 是空闲的，也没有任何原始结果被破坏。**
`final_five_seed_sweep/results/generic_baselines/seed52/` 下已经有一份"意外但合法"的
seed52 结果，你可以选择保留（省一次训练）或删除重跑，随你决定，下文按需说明。

---

## 1. 9 个方法现状总览

| 类别 | 方法 | 已有种子 | 缺失种子 | 结论 |
|---|---|---|---|---|
| 内部基线 | RF (B5) | 42 | 52,62,72,82 | 复用42，补4个 |
| 内部基线 | TCN-GRU (B10) | 42 | 52,62,72,82 | 复用42，补4个 |
| 内部基线 | Multi-task TCN-GRU (B11) | 42 | 52,62,72,82 | 复用42，补4个 |
| 提出方法 | DC-PSR (B12，代码内部名 `FGDS-PSI`) | 42 | 52,62,72,82 | 复用42，补4个 |
| 文献方法 | HTT-Net (adapted) | 42（`config.json` 明确记录，非猜测） | 52,62,72,82 | 复用42，补4个 |
| 文献方法 | Multi-source Attention | 42（硬编码在 `PROTO_B_CFG`） | 52,62,72,82 | 复用42，补4个 |
| 文献方法 | MTF-AViTK | **不确定** | 42,52,62,72,82（建议全部重跑） | 见下方特别说明 |
| 文献方法 | Dynamic GIN + TGP | 42,52,62,72,82 | 无 | **REUSE_OK，无需重跑** |
| 文献方法 | DP2Net-adapted | 42,52,62,72,82 | 无 | **REUSE_OK，无需重跑** |

**总计需要补跑：约 17 次脚本执行**（RF/TCN-GRU/Multi-task/DC-PSR 共用一个脚本，一次跑完 4 个方法）。

### MTF-AViTK 特别说明

`outputs/mtf_avitk/unified_protocol/training_log.csv` 里发现有一行重复/重启的 "epoch 2" 记录，
说明这次训练中途被 `--resume` 过至少一次。代码里 `PROTO_B_CFG['seed']=42` 是硬编码的，
`metrics.json` 里也没有单独存一个 "seed" 字段做交叉验证，所以严格按任务要求的
"seed 不清晰 → 全部重跑" 规则，**建议把 5 个种子全部重新跑一遍**，不复用现有那次。

不过，这次重新生成的 seed=42 会写到**新目录** `outputs/mtf_avitk/seed_sweep/seed42/`，
不会覆盖原来的 `outputs/mtf_avitk/unified_protocol/`，所以旧结果依然保留、互不影响。
如果你想省时间只信任现有 seed42、只补 52/62/72/82，也可以，只是需要你自己承担这个判断——
下面命令两种情况都给出。

---

## 2. 环境与调度原则

- 本机 GPU：RTX 3070 Ti Laptop，8GB 显存。**必须串行跑，不要同时开两个训练。**
- 两个 conda 环境（已确认存在）：
  - `dcpsr`：跑 RF/TCN-GRU/Multi-task/DC-PSR 的对比脚本、HTT-Net
  - `pub_baselines`：跑 Multi-source Attention、MTF-AViTK
- 跑之前建议先看一眼 GPU 是否空闲：

```powershell
nvidia-smi
```

- 推荐执行顺序（越靠后越耗时，MTF-AViTK 放最后）：
  1. RF / TCN-GRU / Multi-task TCN-GRU / DC-PSR（共用一个脚本，几分钟内跑完 4 个种子）
  2. HTT-Net（每个种子约 40 秒，非常快）
  3. Multi-source Attention（每个种子约 2-3 分钟）
  4. MTF-AViTK（每个种子约 30-40 分钟，显存吃紧，务必单独跑、不要和别的训练同时进行）

---

## 3. 逐方法命令

### 3.1 RF / TCN-GRU / Multi-task TCN-GRU / DC-PSR（一个脚本跑全部4个方法）

脚本：`final_five_seed_sweep/scripts/run_generic_baselines_seed.py`
（已验证：只 monkey-patch `RANDOM_SEED` 和权威特征文件路径，不改任何超参数；
输出到独立目录，绝不覆盖 `补充材料/小论文/4_comparison_experiment_recheck/` 里的
原始 seed=42 结果。）

```powershell
conda activate dcpsr
cd "C:\Users\banghai\Documents\BaiduSyncdisk\西工大\王婷\论文\final_five_seed_sweep\scripts"

python run_generic_baselines_seed.py --seed 52
python run_generic_baselines_seed.py --seed 62
python run_generic_baselines_seed.py --seed 72
python run_generic_baselines_seed.py --seed 82
```

> 注：seed=52 的结果其实已经存在（见第0节的意外训练），如果你信任它，
> 上面第一行 `--seed 52` 可以跳过不跑；如果想稳妥一点、完全由你亲手重跑一遍，
> 直接跑也没问题，反正是同一个确定性脚本、同一份数据，结果应当一致。

每个种子输出在：
`final_five_seed_sweep/results/generic_baselines/seed<N>/1_results/FINAL_comparison_results.csv`
里对应 `B5`(RF) / `B10`(TCN-GRU) / `B11`(Multi-task TCN-GRU) / `B12`(DC-PSR，表里显示为 `FGDS-PSI`) 四行。

### 3.2 HTT-Net (adapted)

脚本：`final_five_seed_sweep/scripts/run_htt_net_seed.py`
（已验证：与官方 `train_final_tuned.py` 的 frozen SOURCE_ONLY_TUNED 配置逐字段一致，
只改 `TRAIN_CFG['seed']`；输出到新目录
`outputs/htt_net/D1_C1C4_to_C6_SOURCE_ONLY_TUNED_seed<N>/`，不覆盖官方 seed=42 结果。）

```powershell
conda activate dcpsr
cd "C:\Users\banghai\Documents\BaiduSyncdisk\西工大\王婷\论文\final_five_seed_sweep\scripts"

python run_htt_net_seed.py --seed 52
python run_htt_net_seed.py --seed 62
python run_htt_net_seed.py --seed 72
python run_htt_net_seed.py --seed 82
```

### 3.3 Multi-source Attention

脚本：`final_five_seed_sweep/scripts/run_multi_source_attention_seed.py`
（已验证：只改 `PROTO_B_CFG['seed']`，CWT/SE-Net等结构参数完全不动；
输出到新目录 `outputs/multi_source_attention/seed_sweep/seed<N>/`。）

```powershell
conda activate pub_baselines
cd "C:\Users\banghai\Documents\BaiduSyncdisk\西工大\王婷\论文\final_five_seed_sweep\scripts"

python run_multi_source_attention_seed.py --seed 52 --device cuda
python run_multi_source_attention_seed.py --seed 62 --device cuda
python run_multi_source_attention_seed.py --seed 72 --device cuda
python run_multi_source_attention_seed.py --seed 82 --device cuda
```

### 3.4 MTF-AViTK

脚本：`final_five_seed_sweep/scripts/run_mtf_avitk_seed.py`
（已验证：只改 `PROTO_B_CFG['seed']`，ViT-L/32 / MTF / AdaptMLP / KAN 结构完全不动；
输出到新目录 `outputs/mtf_avitk/seed_sweep/seed<N>/`。此方法最耗显存，务必单独跑，
跑之前用 `nvidia-smi` 确认没有其它训练占用显存。）

**方案 A（推荐，稳妥）：5 个种子全部重新跑**

```powershell
conda activate pub_baselines
cd "C:\Users\banghai\Documents\BaiduSyncdisk\西工大\王婷\论文\final_five_seed_sweep\scripts"

python run_mtf_avitk_seed.py --seed 42 --device cuda
python run_mtf_avitk_seed.py --seed 52 --device cuda
python run_mtf_avitk_seed.py --seed 62 --device cuda
python run_mtf_avitk_seed.py --seed 72 --device cuda
python run_mtf_avitk_seed.py --seed 82 --device cuda
```

**方案 B（省时间）：信任现有 `outputs/mtf_avitk/unified_protocol/` 为 seed=42，只补 4 个**

```powershell
conda activate pub_baselines
cd "C:\Users\banghai\Documents\BaiduSyncdisk\西工大\王婷\论文\final_five_seed_sweep\scripts"

python run_mtf_avitk_seed.py --seed 52 --device cuda
python run_mtf_avitk_seed.py --seed 62 --device cuda
python run_mtf_avitk_seed.py --seed 72 --device cuda
python run_mtf_avitk_seed.py --seed 82 --device cuda
```

若显存不够（8GB 卡上 ViT-L/32 有时会顶到 5-8GB），可以加梯度检查点：

```powershell
python run_mtf_avitk_seed.py --seed 52 --device cuda --grad-checkpoint
```

### 3.5 Dynamic GIN + TGP、DP2Net-adapted —— 不需要跑

这两个方法的 5 个种子已经全部完成、通过审计（结果分别在
`outputs/dynamic_gin_tgp/unified_protocol/seed{42,52,62,72,82}/` 和
`outputs/dp2net/unified_protocol_B-D1/seed{42,52,62,72,82}/`），
**不要重跑**，直接复用即可。

---

## 4. 硬性约束（务必遵守）

- 每个方法只有**一套冻结配置**，5 个种子之间只改随机种子，不许重新调参。
- `C6` 只能用来做最终测试，任何脚本都不应该拿 C6 去选 checkpoint、做 early stopping、或者"种子不好看就重跑"。
- 5 个种子全部保留，不管好坏，不许因为某个种子结果差就删除或重跑（除非是 crash / NaN / 用错配置这类真正的错误）。
- 一次只跑一个训练进程，不要并行开两个脚本抢同一块 8GB 显卡。
- MTF-AViTK 一定放最后单独跑。

---

## 5. 跑完之后怎么办

5 个种子的原始结果文件都会按上面路径分别落盘（每个方法每个种子一个独立目录）。
等你把所有缺失的种子都跑完后，告诉我一声，我可以帮你：
1. 汇总成 `seed-level` 总表（`FINAL_9_METHODS_SEED_LEVEL.csv`）；
2. 计算跨种子 mean±std（统一用 `ddof=1`，与 Dynamic GIN+TGP / DP2Net 已有报告的口径一致），
   生成 `FINAL_9_METHODS_5SEED.csv`；
3. 做最终的 C6 test universe / leakage / 完整性审计，写 `FINAL_5SEED_REPORT.md`；
4. 在全部核对通过后宣布 `EXPERIMENTS FROZEN`。

我不会在你确认之前主动执行任何训练命令。
