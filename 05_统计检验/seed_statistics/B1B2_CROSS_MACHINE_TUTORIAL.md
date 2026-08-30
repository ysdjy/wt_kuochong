# B1 / B2 主对比实验 —— 另一台 Windows 电脑运行教程

在另一台 Windows 电脑上跑 **B1（RF）** 和 **B2（TCN-GRU）** 在 **PHM2010（D1/D2/D3）**、
**NASA Milling（N1-N4）**、**MTW-CM（D1-M/D2-M/D3-M）** 三个数据集、每个任务
**seed 0-100（101 个）** 的主对比实验，产出格式与本目录下已有的
`B9_PHM2010_D1_seed_landscape/` 等完全一致（同样的 `results/seedN/{metrics.csv,
metrics.json, predictions.csv, training_log.csv, confusion_matrix.csv,
run_meta.json, config_resolved.yaml, DONE.flag}` 结构）。

只需要 `git clone` 这一个仓库——B1/B2 需要的旧项目代码（`代码/
main_experiment_3_fgds_psi_optimized.py`、`代码/7.4对比实验.py`、`代码/
9.1nasa数据实验.py`、`experiments_mendeley/code/dcpsr/`）已经 vendor 进
`_shared_code/vendored_legacy/`，三个数据集的冻结预处理产物
（`shared/reproducibility/{PHM2010_D2,PHM2010_D3,NASA_N1..N4,MTW_D1-M,MTW_D2-M,
MTW_D3-M}_frozen_preprocess/`）也已随本次改动一并提交。**不需要**原始数据集
文件（mill.mat、.h5 等）——冻结预处理已经把需要的特征表、GMM、scaler 都算好并
提交了，B1/B2 全程只读取这些冻结产物，不碰原始信号。

## 0. 前置说明

- **B9/B3 的现有结果**（`B9_*_seed_landscape/`、`B3_*_seed_landscape/`）是另一个
  并发 Claude 会话跑的，本教程完全不依赖它们，只依赖同一批"冻结预处理"产物。
- 本机（RTX 3070 Ti Laptop 8GB）实测：B1（RF）每个 (task, seed) 只需几秒到十几秒
  （直接读冻结 CSV + 训练森林，不需要 GPU）；B2（TCN-GRU）需要 GPU，单个
  (task, seed) 从几十秒到十分钟以上不等，取决于该次 GPU 是否被其它任务占用
  （本机验证时正好和另一会话的训练任务抢显存，B2/PHM2010/D1/seed0 因此花了
  ~11 分钟；不抢显存时应该快很多）。
- **9 个任务 × 101 个 seed × 2 个方法 = 1818 次运行**，B2 部分建议预留数小时到
  一两天，具体取决于 GPU 独占程度。

## 1. Clone

```powershell
git clone https://github.com/ysdjy/wt_kuochong.git
cd wt_kuochong\扩充实验代码
```

## 2. 环境

跟 `MANUAL_RUN.md` 用的是同一个 conda 环境（`wt_kuochong`），本教程的脚本只需要
`numpy pandas scipy scikit-learn torch`（PyWavelets/xgboost 等用不到，但装了也
无妨）：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_windows.ps1
```

如果只想跑本教程、不想跑 `scripts\self_check.py` 检查的那一整套（因为它会检查
`data\PHM2010\raw\` 等本教程不需要的东西），也可以自己手动建一个环境：

```powershell
conda create -n wt_kuochong python=3.11.15
conda activate wt_kuochong
pip install -r environment\requirements.txt
pip install torch==2.7.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

## 3. 验证冻结预处理产物已经到位（不需要原始数据）

```powershell
conda activate wt_kuochong
cd 05_统计检验\seed_statistics\_shared_code
python -c "import json; print(json.load(open('../../../shared/reproducibility/NASA_N1_frozen_preprocess/manifest_hashes.json'))['preprocess_seed'])"
```

应该输出 `42`。如果报 `FileNotFoundError`，说明 clone 不完整或没拉到最新，重新
`git pull` 一下。

## 4. 冒烟测试（先跑这个，确认能跑通再批量跑）

```powershell
$env:PYTHONIOENCODING="utf-8"
python run_b1_seed_task.py --task D1 --train_seed 0 --results_root ..\B1_PHM2010_D1_seed_landscape\results
python run_b2_seed_task.py --task D1 --train_seed 0 --results_root ..\B2_PHM2010_D1_seed_landscape\results --backbone_root ..\B2_PHM2010_D1_seed_landscape\backbone_checkpoints
```

两条命令都应该打印 `[protocol] DONE(done): ...` 收尾，且
`..\B1_PHM2010_D1_seed_landscape\results\seed0\DONE.flag` 存在。跑通后可以放心
批量跑。

## 5. 正式批量运行

先重新生成（或直接用仓库里已经生成好的）任务-seed 列表：

```powershell
python make_b1b2_task_seed_lists.py
```

Git Bash 里跑（Windows 上装 Git 自带 Git Bash，driver 脚本是 `.sh`）：

### B1（RF，CPU，建议先跑，快）

```bash
cd /path/to/wt_kuochong/扩充实验代码/05_统计检验/seed_statistics/_shared_code
export PYTHON_EXE="/c/Users/<你的用户名>/miniconda3/envs/wt_kuochong/python.exe"   # 按实际路径改

cat b1_phm_task_seed_list.txt  | xargs -P 4 -I{} bash run_one_b1_task_seed.sh {}
cat b1_nasa_task_seed_list.txt | xargs -P 4 -I{} bash run_one_b1_nasa_task_seed.sh {}
cat b1_mtw_task_seed_list.txt  | xargs -P 4 -I{} bash run_one_b1_mtw_task_seed.sh {}
```

B1 是 CPU-only 的 RF，`-P 4`（4 路并发）对着 CPU 核数调整即可，不涉及显存风险。

### B2（TCN-GRU，GPU，务必只用 1 路并发，避免 OOM）

```bash
cat b2_phm_task_seed_list.txt  | xargs -P 1 -I{} bash run_one_b2_task_seed.sh {}
cat b2_nasa_task_seed_list.txt | xargs -P 1 -I{} bash run_one_b2_nasa_task_seed.sh {}
cat b2_mtw_task_seed_list.txt  | xargs -P 1 -I{} bash run_one_b2_mtw_task_seed.sh {}
```

**`-P 1`（单路串行）是刻意的**——本项目已经在别的地方（`05_统计检验/
seed_statistics/B9_PHM2010_D1_seed_landscape/README.md`）记录过 8 路并发触发
CUDA OOM 的教训。如果这台电脑的 GPU 显存明显大于 8GB 且确定没有别的任务在跑，
可以自己承担风险改成 `-P 2` 或更高，但不建议。

## 6. 断点续跑

所有 driver 脚本 / `run_b*_*.py` 都做了跟 `run_seed_task.py`（B9/B3）一样的
resume 检查：某个 `(task, seed)` 只要 `results/seedN/DONE.flag` 存在且
`run_meta.json` 里的 `feature_hash/split_hash/gmm_hash/window_hash/train_seed/
task` 都跟当前冻结产物匹配，就会打印 `[resume] ... skipping` 直接跳过。所以：

- 中途 Ctrl-C 打断，或电脑意外重启，重新跑同一条 `xargs` 命令就行，已完成的
  seed 不会重跑。
- 如果某个 seed 失败（`FAILED.flag` + `error.log`，没有 `DONE.flag`），同样会
  在重新跑时自动重试（因为 resume 检查只认 `DONE.flag`）。

## 7. 跑完之后：结果去哪儿看 / 怎么带回来

产出的目录（`B1_PHM2010_D1_seed_landscape/`、`B1_NASA_N1_seed_landscape/`、
`B1_MTW_D1M_seed_landscape/`……B2 同理）跟本目录已有的 `B9_*`/`B3_*` 系列是平级
的，同一套目录结构。这些目录默认不会被 git 追踪（`.gitignore` 里
`/05_统计检验/seed_statistics/` 整体被排除，只有 `_shared_code/`、`*.md`、
`summary_*/` 被显式放行）——想把结果带回这台电脑，跟 `MANUAL_RUN.md` 里
"多机结果合并"一节同样的思路：

- 直接把整个 `B1_*_seed_landscape/`、`B2_*_seed_landscape/` 目录用 U 盘/局域网
  拷回来，跟本机的 `05_统计检验/seed_statistics/` 目录合并（目录名不会跟已有
  的 B9/B3 系列冲突）。
- 或者只拷 `results/` 里的 `metrics.csv`/`metrics.json` 部分做汇总分析，
  `predictions.csv`/`training_log.csv`/`backbone_checkpoints/` 体积更大，非必要
  可以不拷回来。
- 如果想入库，参照本文件顶部"前置说明"提到的 `.gitignore` 放行规则，自己加一
  段 `!/05_统计检验/seed_statistics/B1_*_seed_landscape/summary_*` 之类的规则
  （本教程不预先做这个决定，跟 B9/B3 系列一样，入不入库由你自己决定）。

## 8. 已知限制 / 尚未验证的点

- 本机只验证了 B1（3 个数据集，真实跑通 D1/N1/D1-M 各 1 个 seed）和 B2（PHM2010/
  NASA 真实跑通 1 个 seed；MTW 只验证了 import/模型实例化，没有跑完一个完整
  seed 的真实训练——GPU 当时被另一并发会话占用，跑到很长时间还没结束，为免
  过度占用 GPU 提前收尾）。**强烈建议在这台新电脑上，先老老实实跑一遍第 4 节
  的冒烟测试（包括 B2 MTW 那一条），确认真的能跑完一整个 seed 再批量启动。**
- B2 MTW 冒烟测试命令（第 4 节没有列出，这里补充）：
  ```powershell
  python run_b2_mtw_seed_task.py --task D1-M --train_seed 0 --results_root ..\B2_MTW_D1M_seed_landscape\results --backbone_root ..\B2_MTW_D1M_seed_landscape\backbone_checkpoints
  ```
- 没有做过多机合并/去重的实测（`scripts\merge_results.py` 是给
  `methods/`/`run_phm2010.py` 那一套新框架用的，跟这里的目录结构不兼容，本教程
  第 7 节给的是手动合并思路，不是现成脚本）。
