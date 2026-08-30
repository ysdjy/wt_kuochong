# 任务：在这台电脑上跑完 B9(DC-PHSR) + B3(Multi-task TCN-GRU) 在 MTW-CM 数据集 D2-M / D3-M 上的全部 101 个随机种子

## 背景

这是一个多电脑协同跑实验的任务。主控电脑（下称"电脑A"）已经在 `05_统计检验/seed_statistics/` 下
建好了一套确定性的目录结构和 resume 协议：每个 (方法, 数据集, 任务, seed) 组合的结果都写到一个固定
路径 `Bx_数据集_任务_seed_landscape/results/seedN/`，只要代码相同、冻结预处理相同，任何一台机器算出
来的结果目录都是可以直接合并回电脑A的，不会冲突、不会覆盖已有结果。

电脑A当前正在跑 MTW-CM 数据集的 D1-M 任务（101 个 seed），**已经把 D2-M / D3-M 从电脑A的批次里停掉**，
交给这台电脑（电脑B）来跑，这样两台机器并行推进，不重复劳动。

## 你（在电脑B上执行任务的 agent）需要做什么

### 第 0 步：确认环境

- Python 环境：需要 `torch`（已安装）+ `numpy pandas scipy scikit-learn`。如果还没装全，跑：
  ```
  pip install numpy pandas scikit-learn
  ```
  （scipy 是 torch/numpy 的依赖，通常已经有）
- 不需要装 PyWavelets（那是给 B5/B6 用的，这次任务用不到）。

### 第 1 步：拉取最新代码

如果还没 clone：
```bash
git clone https://github.com/ysdjy/wt_kuochong.git
cd wt_kuochong/扩充实验代码
```
如果已经 clone 过（但可能是旧版本）：
```bash
cd <你的wt_kuochong本地路径>/扩充实验代码
git pull origin main
```
**必须确认能看到这些文件**（`git pull` 之后检查一下）：
```
05_统计检验/seed_statistics/_shared_code/run_mtw_seed_task.py
05_统计检验/seed_statistics/_shared_code/run_one_mtw_task_seed.sh
05_统计检验/seed_statistics/_shared_code/mtw_task_seed_list.txt
05_统计检验/seed_statistics/_shared_code/vendored_legacy/dcpsr/
shared/reproducibility/MTW_D2-M_frozen_preprocess/
shared/reproducibility/MTW_D3-M_frozen_preprocess/
```
如果这些文件不存在，说明 clone/pull 没拿到最新提交，**不要往下继续**，先解决这个问题（重新 `git pull`，
确认 `git log -1` 显示的 commit 是最新的）。

### 第 2 步：验证冻结预处理产物完整（不需要原始数据集）

```bash
cd 05_统计检验/seed_statistics/_shared_code
python -c "import json; print(json.load(open('../../../shared/reproducibility/MTW_D2-M_frozen_preprocess/manifest_hashes.json'))['preprocess_seed'])"
python -c "import json; print(json.load(open('../../../shared/reproducibility/MTW_D3-M_frozen_preprocess/manifest_hashes.json'))['preprocess_seed'])"
```
两次都应该输出 `42`。如果报 `FileNotFoundError`，说明 clone 不完整，回第1步。

**这一步不需要原始 MTW 数据集文件**（`.h5` 那些）——冻结预处理已经把训练/验证/测试用的特征表、
GMM、scaler 都算好并提交到仓库里了，`run_mtw_seed_task.py` 全程只读这些冻结产物。

### 第 3 步：冒烟测试（先跑 1 个 seed 确认能跑通）

```bash
export PYTHONIOENCODING=utf-8
python run_mtw_seed_task.py --task D2-M --train_seed 0 \
  --results_root ../B9_MTW_D2M_seed_landscape/results \
  --backbone_root ../B9_MTW_D2M_seed_landscape/backbone_checkpoints \
  --b3_results_root ../B3_MTW_D2M_seed_landscape/results
```
应该看到打印 `[protocol] DONE(done): ...`，并且以下文件都存在：
```
../B9_MTW_D2M_seed_landscape/results/seed0/DONE.flag
../B9_MTW_D2M_seed_landscape/results/seed0/run_meta.json
../B3_MTW_D2M_seed_landscape/results/seed0/DONE.flag
```
跑通后再跑一次同样的命令，这次应该打印 `[resume] ... skipping`（确认断点续跑逻辑正常）。

跑通了再对 D3-M 也跑一遍冒烟测试（把上面命令里的 `D2-M`/`D2M` 换成 `D3-M`/`D3M`）。

### 第 4 步：正式批量跑（优先级最高，先跑完这个再做别的任何任务）

用 Git Bash（Windows 自带 Git 会装 Git Bash）：

```bash
cd <wt_kuochong本地路径>/扩充实验代码/05_统计检验/seed_statistics/_shared_code
export PYTHONIOENCODING=utf-8

# 只挑出 D2-M / D3-M 的 (task,seed) 组合，跳过 D1-M（电脑A自己在跑）
grep -E "^D2-M,|^D3-M," mtw_task_seed_list.txt > mtw_d2m_d3m_seed_list.txt
wc -l mtw_d2m_d3m_seed_list.txt   # 应该是 202 行 (101*2)

cat mtw_d2m_d3m_seed_list.txt | xargs -P 3 -I{} bash run_one_mtw_task_seed.sh {}
```

**`-P 3`（3 路并发）是刻意的数值，不要改大**——电脑A实测过，MTW-CM 这个方法在 8GB 显卡上 3 路并发
已经能跑到 90-95% GPU 利用率（计算瓶颈，不是显存瓶颈），5 路并发实测会导致显存打满、大量任务失败
（OOM / 段错误）。如果你这台机器显卡显存明显大于 8GB（比如 16GB/24GB+）且确定没有别的任务在用，
可以自行承担风险调大，否则请保持 `-P 3`。

### 第 5 步：断点续跑 / 中途重启

- 如果中途 Ctrl-C 打断，或者电脑重启了，**直接重新执行第4步同一条命令即可**，已经跑完的 seed 会被
  自动跳过（`DONE.flag` + hash 匹配才算数），不会重复跑。
- 如果某个 seed 失败了（有 `FAILED.flag` + `error.log`，没有 `DONE.flag`），同样会在下次跑同一条命令
  时自动重试——**不要因为某个 seed 失败就跳过它或从任务列表里删掉它**，OOM/偶发崩溃只是执行失败，
  重跑就行，不是"这个种子有问题"。

### 第 6 步：跑完之后

跑完 D2-M + D3-M 各 101 个 seed（B9 + 配对的 B3，一共 202×2=404 次训练）后：

1. 确认完整性：
   ```bash
   find ../B9_MTW_D2M_seed_landscape/results -name DONE.flag | wc -l   # 应该是 101
   find ../B9_MTW_D3M_seed_landscape/results -name DONE.flag | wc -l   # 应该是 101
   find ../B3_MTW_D2M_seed_landscape/results -name DONE.flag | wc -l   # 应该是 101
   find ../B3_MTW_D3M_seed_landscape/results -name DONE.flag | wc -l   # 应该是 101
   find ../B9_MTW_D2M_seed_landscape ../B9_MTW_D3M_seed_landscape ../B3_MTW_D2M_seed_landscape ../B3_MTW_D3M_seed_landscape -name FAILED.flag | wc -l   # 应该是 0
   ```
2. 把结果告诉用户，用户会把 `B9_MTW_D2M_seed_landscape/`、`B9_MTW_D3M_seed_landscape/`、
   `B3_MTW_D2M_seed_landscape/`、`B3_MTW_D3M_seed_landscape/` 这四个整个目录拷回电脑A（U盘/局域网都
   行），跟电脑A同名目录合并——因为路径是确定性的 (方法,数据集,任务,seed) → 固定路径，两边不会有
   任何文件冲突，直接覆盖/合并即可，不需要人工去重。
   - **这几个目录本身不需要、也不应该 push 到 GitHub**（原始的 per-seed 预测/checkpoint/训练日志体
     积较大，仓库约定只提交聚合后的小 CSV，不提交这些原始结果树，`.gitignore` 已经处理好了这一点，
     git 状态应该看不到这些目录）。
3. 完成这个任务之后，**不要自行开始跑其他方法**（B1/B2/B4/B5/B6/B7/B8 或其他数据集）——等用户下一步
   安排。

## 严禁事项（协议要求，务必遵守）

- **不要修改** `run_mtw_seed_task.py`、`shared/reproducibility/MTW_*_frozen_preprocess/` 里的任何文件
  ——这些是跨机器一致性的基础，改了就会导致这台机器算出来的结果和电脑A不是同一套预处理，无法合并。
- **不要跳过或删除任何看起来"结果不好看"的 seed**——101 个 seed 必须每个都跑，不能挑着跑。
- **不要根据测试集表现调超参数**——所有超参数已经在冻结配置里定死了，这次任务不涉及调参。
- **不要把 `-P` 并发数调到 3 以上**，除非你确认这台机器的显卡显存远大于 8GB 且没有其他任务在跑。
- 完成 D2-M/D3-M 之前，不要开始任何其他任务。
