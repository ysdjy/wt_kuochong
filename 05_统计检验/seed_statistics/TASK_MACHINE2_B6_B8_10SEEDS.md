# 任务：在这台电脑上跑 B6 (MTF-AViTK) + B8 (DP2Net) 在 PHM2010 D1/D2/D3 上的 10 个抽样种子

## ⚠️ 执行顺序：先等 B9(D2-M/D3-M) 跑完，再开始这个任务

这台电脑上应该正在跑 `TASK_MACHINE2_MTW_D2M_D3M.md` 那个任务（B9+B3 在 MTW-CM D2-M/D3-M
上的101个种子）。**先确认那个任务完全跑完**（`find ../B9_MTW_D2M_seed_landscape/results
-name DONE.flag | wc -l` 和 D3M 那个都是101，`FAILED.flag` 数量是0），**再开始执行本文件**
——不要两个任务同时跑，都是重GPU负载，叠在一起会互相拖慢甚至撞显存。

如果 D2-M/D3-M 那个任务还没跑完，直接退出/挂起本文件，等它跑完再回来看这份。

## ⚠️ 这套 runner 代码是全新的，还没有独立验证过 B6/B8

`run_methods_adapter_seed_task.py`（本任务要用的脚本）目前只在电脑A上用 **B7** 实测验证中
（同一份脚本、同一套机制，B6/B7/B8 三个方法共用这一个 runner，只是各自的 `methods/Bx_xxx/
adapter.py` 不同）——B7 那边跑通了大概率说明机制没问题，但**你在电脑B上第4节的冒烟测试
这一步格外重要，务必真的跑完看到 `[protocol] DONE(done)` 再进入批量**，如果冒烟测试报错，
先截图/保留 error.log 里的内容，不要自己改代码硬跑。

## 背景

跟之前 `TASK_MACHINE2_MTW_D2M_D3M.md` 是同一套多电脑协同协议：确定性路径
(方法,数据集,任务,seed) → 固定目录，代码+冻结产物已提交到仓库，跑完的结果目录直接拷回
电脑A合并即可，不会冲突。

**这次不是跑满 0-100 全部 101 个种子**，只跑这 10 个抽样种子（用于先估计方差/耗时，
而不是正式的完整 mean±std 统计）：

```
3 13 23 33 43 53 63 73 83 93
```

B6、B7、B8 是这批方法里最重的三个（历史记录/实测：B6 约30-40分钟/seed，B7 约1小时/seed，
B8 约38分钟/seed，且**B8 没有早停机制，200个epoch是固定的**）。电脑A（这边）在跑 B2/B4/B5/B7，
电脑B（你）负责 B6 + B8。

## 0. 前置说明

- B6 的 `README.md`（`methods/B6_MTF_AViTK/README.md`）里有一条重要警告："**这个方法从不无人
  值守跑**"——它是 309M 参数、`checkpoint_best.pth` 约1.2GB（本任务默认不保存checkpoint，
  不会占这个盘）。跑之前建议先看一眼这个 README 的 "Compute cost" 一节。
- B8（`methods/B8_DP2Net/README.md`）**没有中途 checkpoint/resume**（只有整个 seed 跑完才有
  `DONE.flag`，跑到一半被打断只能从头重跑这一个 seed，不会重跑已完成的其他 seed）。
- 这两个方法都是**原始信号驱动**（不是表格特征），需要真实的 PHM2010 原始数据集文件
  （`c1/c1_wear.csv`、`c1/c1/c_1_001.csv` 这种结构），不能只靠冻结预处理产物。

## 1. Clone / Pull

```powershell
cd wt_kuochong   # 仓库根目录本身就是内容根，不需要再 cd 进"扩充实验代码"子目录
git pull origin main
```
确认能看到这些文件（`git log -1` 应该是这次改动之后的最新commit）：
```
05_统计检验/seed_statistics/_shared_code/run_methods_adapter_seed_task.py
05_统计检验/seed_statistics/_shared_code/run_one_methods_adapter_task_seed.sh
methods/B6_MTF_AViTK/adapter.py
methods/B8_DP2Net/adapter.py
```

## 2. 环境

跟 `MANUAL_RUN.md` 同一个 conda 环境（`wt_kuochong`），额外需要 PyWavelets（B6 用得到）：
```powershell
conda activate wt_kuochong
pip install PyWavelets
```
（B6 README 里提到过 `dcpsr` 环境曾经缺 PyWavelets，这台机器如果也用类似环境要留意。）

## 3. 原始数据集位置

设置 `PHM2010_ROOT` 环境变量指向你 `文档/111` 下已经放好的 PHM2010 原始数据集目录（需要包含
`c1/c1_wear.csv`、`c1/c1/c_1_001.csv` 这种结构）：

```powershell
$env:PHM2010_ROOT = "C:\path\to\your\PHM2010_raw_dir"
```

**不需要**冻结预处理产物——B6/B7/B8 都是从原始信号直接构建标签/窗口/图像（确定性算法，
非随机，不依赖 PREPROCESS_SEED 之外的东西），跟 B1/B2/B3/B4/B5/B9 用的那套表格特征冻结产物
是两回事，无需理会 `shared/reproducibility/`。

## 4. 冒烟测试（先跑这个，确认能跑通）

Git Bash 里：
```bash
cd wt_kuochong/05_统计检验/seed_statistics/_shared_code
export PYTHONIOENCODING=utf-8
export PYTHON_EXE="/c/path/to/your/envs/wt_kuochong/python.exe"   # 按实际改
export PHM2010_ROOT="/c/path/to/your/PHM2010_raw_dir"             # 按实际改

python run_methods_adapter_seed_task.py --method B6 --task D1 --train_seed 3 \
  --results_root ../B6_PHM2010_D1_seed_landscape/results
```
这一条命令会**真实跑完 B6 在 D1 任务、种子3 上的完整训练**（不是 smoke test），预计30-40分钟。
跑完应该打印 `[protocol] DONE(done): ...`，并生成
`../B6_PHM2010_D1_seed_landscape/results/seed3/{metrics.csv, run_meta.json, DONE.flag, ...}`。

**强烈建议先跑这一条，亲眼确认跑通、耗时符合预期，再进入批量**（尤其 B6 有 README 里那条
"不能无人值守"的警告——先盯着这一条完整跑一次，心里有底之后再放手批量跑）。

B8 同理：
```bash
python run_methods_adapter_seed_task.py --method B8 --task D1 --train_seed 3 \
  --results_root ../B8_PHM2010_D1_seed_landscape/results
```
预计约38分钟（200个epoch固定，无法通过早停缩短）。

## 5. 正式批量跑（10个抽样种子，不是全部101个）

先生成任务列表（或者直接用下面这段自己拼，跟仓库里 `make_b1b2_task_seed_lists.py` 是同一个
思路，这次不需要跑那个脚本，手动建两个文件更直接）：

```bash
cd wt_kuochong/05_统计检验/seed_statistics/_shared_code
for t in D1 D2 D3; do for s in 3 13 23 33 43 53 63 73 83 93; do echo "B6,$t,$s"; done; done > b6_10seed_phm_task_seed_list.txt
for t in D1 D2 D3; do for s in 3 13 23 33 43 53 63 73 83 93; do echo "B8,$t,$s"; done; done > b8_10seed_phm_task_seed_list.txt
wc -l b6_10seed_phm_task_seed_list.txt b8_10seed_phm_task_seed_list.txt   # 都应该是30行 (3任务x10种子)
```

然后批量跑（**`-P 1` 单路串行，不要并发**——B6单个就要用掉大量显存+算力，B8同理，两个方法本身
也不要同时跑，一个跑完再跑下一个，避免这台机器重蹈"显存被挤爆导致卡死"的坑）：

```bash
cat b6_10seed_phm_task_seed_list.txt | xargs -P 1 -I{} bash run_one_methods_adapter_task_seed.sh {}
cat b8_10seed_phm_task_seed_list.txt | xargs -P 1 -I{} bash run_one_methods_adapter_task_seed.sh {}
```

30个种子×30-40分钟(B6) ≈ 15-20小时；30个种子×38分钟(B8) ≈ 19小时。两个方法加起来预计
**一天半到两天**，如果你的16GB显卡比这边这台8GB笔记本GPU快很多，可能会明显更快，具体以
实测冒烟测试那一条的真实耗时为准，跑完第一条之后如果比预期快很多，可以考虑把 `-P` 调到2
（但仍然建议先用 `-P 1` 观察一段时间的显存/温度再决定要不要提速）。

## 6. 断点续跑

跟其它任务一样，靠 `DONE.flag` + `run_meta.json` 里的 `config_hash` 匹配来判断要不要跳过。
中途 Ctrl-C 或电脑重启，重新跑同一条 `xargs` 命令即可，已完成的种子自动跳过。

**但要注意 B8 没有中途 checkpoint**——如果一个 seed 训练到一半被打断（没有 DONE.flag），
重跑这个 seed 时是从 epoch 0 重新开始，不是接着断点续训，所以中途尽量不要主动打断正在跑的
B8 任务（B6 同理，也没有 epoch 级别的 resume）。

## 7. 跑完之后

结果目录 `B6_PHM2010_{D1,D2,D3}_seed_landscape/`、`B8_PHM2010_{D1,D2,D3}_seed_landscape/`
整个拷回电脑A（U盘/局域网），跟电脑A本地同名目录合并（目录名不会冲突）。这两个方法**这次只
跑了10个抽样种子，不是完整101个**——目录里 `seed{N}` 只会出现这10个，其余91个种子的目录
本来就不存在，属于预期状态，不代表"没跑完"。

跑完之后**不要自行开始其他任务或补跑剩下91个种子**，等用户根据这10个种子的耗时/方差决定
下一步（是否值得跑满101个）。

## 严禁事项

- 不要修改 `run_methods_adapter_seed_task.py`、`methods/B6_MTF_AViTK/`、`methods/B8_DP2Net/`
  下的任何文件。
- 不要跳过列表里的任何一个种子，也不要额外加跑列表之外的种子。
- 不要调超参数（`methods/B6_MTF_AViTK/adapter.py`/`methods/B8_DP2Net/adapter.py` 里的
  `DEFAULT_CFG`/`PROTO_B_CFG` 已经定死，不在本次任务范围内）。
- B6/B8 不要同时并发跑，也不要把 `-P` 提到 2 以上，除非你已经亲眼验证过冒烟测试那一条、
  确认显存/耗时都在预期范围内。
