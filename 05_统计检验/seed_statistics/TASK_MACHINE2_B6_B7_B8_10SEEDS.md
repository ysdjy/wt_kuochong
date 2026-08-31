# 任务：在这台电脑上跑 B6 (MTF-AViTK) + B7 (Dynamic GIN+TGP) 剩余种子 + B8 (DP2Net)

这份文件**取代**之前发的 `TASK_MACHINE2_B6_B8_10SEEDS.md`——内容基本一样，多加了 B7 的部分。
如果之前那份还没开始执行，以这份为准。

## ⚠️ 执行顺序：先等 B9(D2-M/D3-M) 跑完，再开始这个任务

这台电脑上应该正在跑 `TASK_MACHINE2_MTW_D2M_D3M.md` 那个任务（B9+B3 在 MTW-CM D2-M/D3-M
上的101个种子）。**先确认那个任务完全跑完**（`find ../B9_MTW_D2M_seed_landscape/results
-name DONE.flag | wc -l` 和 D3M 那个都是101，`FAILED.flag` 数量是0），**再开始执行本文件**
——不要两个任务同时跑，都是重GPU负载，叠在一起会互相拖慢甚至撞显存。

## ⚠️ 跑之前先 `git pull`

`run_methods_adapter_seed_task.py`（本任务要用的通用runner，B6/B7/B8 三个方法共用）是新加的，
之前 clone/pull 的版本可能没有。跑之前先：
```bash
cd wt_kuochong   # 仓库根目录本身就是内容根，不需要再 cd 进"扩充实验代码"子目录
git pull origin main
```
确认能看到 `05_统计检验/seed_statistics/_shared_code/run_methods_adapter_seed_task.py`。

## 关于 B7：只跑剩余的 27 个种子，不是全部 30 个

B7（Dynamic GIN+TGP）的 10 个抽样种子（3/13/23/33/43/53/63/73/83/93）× 3 个任务(D1/D2/D3)
= 30 个组合。**其中 D1 任务的种子 3、13、23 这三个已经在电脑A上跑完/正在跑，不需要你重复跑**
——你只需要跑剩下的 **27 个**：D1 的种子 33/43/53/63/73/83/93（7个）+ D2 全部10个种子 +
D3 全部10个种子。具体列表见第5节，已经帮你排除了那3个。

## 0. 前置说明

- B6 的 `README.md`（`methods/B6_MTF_AViTK/README.md`）里有一条重要警告："**这个方法从不无人
  值守跑**"——它是 309M 参数、`checkpoint_best.pth` 约1.2GB（本任务默认不保存checkpoint，
  不会占这个盘）。跑之前建议先看一眼这个 README 的 "Compute cost" 一节。
- B7 **实测**（电脑A今晚测的）：约 **76分钟/种子**（seed3, D1任务, 26个epoch才触发早停,
  4579秒），比文档记录的"约1小时"要长，按这个数字规划时间更准确。27个种子串行大概
  **34小时**。
- B8（`methods/B8_DP2Net/README.md`）**没有中途 checkpoint/resume**（只有整个 seed 跑完才有
  `DONE.flag`，跑到一半被打断只能从头重跑这一个 seed，不会重跑已完成的其他 seed），且
  **没有早停机制**，200个epoch（100预训练+100生成）是固定的，约38分钟/种子（电脑A实测）。
- 这三个方法都是**原始信号驱动**（不是表格特征），需要真实的 PHM2010 原始数据集文件
  （`c1/c1_wear.csv`、`c1/c1/c_1_001.csv` 这种结构），不能只靠冻结预处理产物。

## 1. 环境

跟 `MANUAL_RUN.md` 同一个 conda 环境（`wt_kuochong`），额外需要 PyWavelets（B6 用得到）：
```powershell
conda activate wt_kuochong
pip install PyWavelets
```

## 2. 原始数据集位置 —— 环境变量名务必用对

**这是电脑A今晚踩过的坑**：B6/B7/B8 三个方法的原始信号定位都用 `PHM2010_ROOT` 这一个环境
变量名（不是 `PHM2010_RAW_ROOT`，那个是 B5 专用的，B6/B7/B8 不认）。设置成你 `文档/111`
下已经放好的 PHM2010 原始数据集目录（需要包含 `c1/c1_wear.csv`、`c1/c1/c_1_001.csv` 这种结构）：

```powershell
$env:PHM2010_ROOT = "C:\path\to\your\PHM2010_raw_dir"
```
Git Bash 里对应：
```bash
export PHM2010_ROOT="/c/path/to/your/PHM2010_raw_dir"
```
**如果只设置了 `PHM2010_RAW_ROOT` 而没设 `PHM2010_ROOT`，B6/B7/B8 会全部秒失败**（
`FileNotFoundError`，报错信息里路径会带一个奇怪的 `methods\data\PHM2010\raw` 前缀）——
电脑A今晚就是这么栽的，9个种子瞬间全部 `exit=1`，后来才发现是环境变量名写错。

**不需要**冻结预处理产物——B6/B7/B8 都是从原始信号直接构建标签/窗口/图像（确定性算法，
非随机），跟 B1/B2/B3/B4/B5/B9 用的那套表格特征冻结产物是两回事，无需理会
`shared/reproducibility/`。

## 3. 冒烟测试（先跑这个，确认能跑通、环境变量没配错）

Git Bash 里：
```bash
cd wt_kuochong/05_统计检验/seed_statistics/_shared_code
export PYTHONIOENCODING=utf-8
export PYTHON_EXE="/c/path/to/your/envs/wt_kuochong/python.exe"   # 按实际改
export PHM2010_ROOT="/c/path/to/your/PHM2010_raw_dir"             # 按实际改，注意变量名

python run_methods_adapter_seed_task.py --method B6 --task D1 --train_seed 3 \
  --results_root ../B6_PHM2010_D1_seed_landscape/results
```
这条命令会**真实跑完 B6 在 D1 任务、种子3 上的完整训练**（不是 smoke test），预计30-40分钟。
跑完应该打印 `[protocol] DONE(done): ...`。

B7、B8 同理（B7 这条正好也是电脑A已经跑过的 D1/seed3，用来验证你这边环境是否一致）：
```bash
python run_methods_adapter_seed_task.py --method B7 --task D1 --train_seed 3 \
  --results_root ../B7_PHM2010_D1_seed_landscape/results
python run_methods_adapter_seed_task.py --method B8 --task D1 --train_seed 3 \
  --results_root ../B8_PHM2010_D1_seed_landscape/results
```

**强烈建议先跑通这三条，确认耗时符合预期（B6 30-40分钟、B7 约76分钟、B8 约38分钟），再进入
批量**——尤其 B6 有"不能无人值守"的警告，先盯着完整跑一次心里有底。

## 4. 正式批量跑

先生成任务列表：
```bash
cd wt_kuochong/05_统计检验/seed_statistics/_shared_code
for t in D1 D2 D3; do for s in 3 13 23 33 43 53 63 73 83 93; do echo "B6,$t,$s"; done; done > b6_10seed_phm_task_seed_list.txt
for t in D1 D2 D3; do for s in 3 13 23 33 43 53 63 73 83 93; do echo "B8,$t,$s"; done; done > b8_10seed_phm_task_seed_list.txt
wc -l b6_10seed_phm_task_seed_list.txt b8_10seed_phm_task_seed_list.txt   # 都应该是30行

# B7 只跑剩余27个（D1的seed 3/13/23 已经在电脑A跑了，这里排除掉）
cat > b7_10seed_remainder_task_seed_list.txt << 'EOF'
B7,D1,33
B7,D1,43
B7,D1,53
B7,D1,63
B7,D1,73
B7,D1,83
B7,D1,93
B7,D2,3
B7,D2,13
B7,D2,23
B7,D2,33
B7,D2,43
B7,D2,53
B7,D2,63
B7,D2,73
B7,D2,83
B7,D2,93
B7,D3,3
B7,D3,13
B7,D3,23
B7,D3,33
B7,D3,43
B7,D3,53
B7,D3,63
B7,D3,73
B7,D3,83
B7,D3,93
EOF
wc -l b7_10seed_remainder_task_seed_list.txt   # 应该是27行
```

然后批量跑（**`-P 1` 单路串行，不要并发，也不要三个方法同时跑**——每一个单独跑都已经很吃
显存/算力了，叠在一起大概率重演电脑A"显存被挤爆"的教训）：

```bash
cat b6_10seed_phm_task_seed_list.txt          | xargs -P 1 -I{} bash run_one_methods_adapter_task_seed.sh {}
cat b7_10seed_remainder_task_seed_list.txt    | xargs -P 1 -I{} bash run_one_methods_adapter_task_seed.sh {}
cat b8_10seed_phm_task_seed_list.txt          | xargs -P 1 -I{} bash run_one_methods_adapter_task_seed.sh {}
```

预计耗时（串行，按电脑A实测数字算，你这边显卡更好可能会快一些，但先按这个规划）：
- B6：30 × 30-40分钟 ≈ 15-20小时
- B7：27 × 76分钟 ≈ 34小时
- B8：30 × 38分钟 ≈ 19小时
- **三个加起来大约 3 天**。如果你的16GB显卡明显比这边这台8GB笔记本GPU快，跑完第一个冒烟测试
  后可以按实际耗时重新估算，也可以考虑几个方法之间用 `-P` 稍微并发（但仍建议每个方法内部
  保持 `-P 1`，方法之间要不要并发看显存余量）。

## 5. 断点续跑

跟其它任务一样，靠 `DONE.flag` + `run_meta.json` 里的 `config_hash` 匹配来判断要不要跳过。
中途 Ctrl-C 或电脑重启，重新跑同一条 `xargs` 命令即可，已完成的种子自动跳过。

**但要注意 B7/B8 都没有中途 checkpoint**（B6 也没有）——如果一个 seed 训练到一半被打断
（没有 DONE.flag），重跑这个 seed 时是从头开始，不是接着断点续训，所以中途尽量不要主动打断
正在跑的任务。

## 6. 跑完之后

结果目录 `B6_PHM2010_{D1,D2,D3}_seed_landscape/`、`B7_PHM2010_{D1,D2,D3}_seed_landscape/`、
`B8_PHM2010_{D1,D2,D3}_seed_landscape/` 整个拷回电脑A（U盘/局域网），跟电脑A本地同名目录
合并（目录名不会冲突，B7 那边电脑A自己也有 D1/seed3,13,23 三个子目录，拷过去合并成一份
完整的10个种子×3任务）。

这次**只跑了10个抽样种子，不是完整101个**——目录里 `seed{N}` 只会出现这10个，其余91个种子
的目录本来就不存在，属于预期状态，不代表"没跑完"。

跑完之后**不要自行开始其他任务或补跑剩下91个种子**，等用户根据这10个种子的耗时/方差决定
下一步。

## 严禁事项

- 不要修改 `run_methods_adapter_seed_task.py`、`methods/B6_MTF_AViTK/`、
  `methods/B7_Dynamic_GIN_TGP/`、`methods/B8_DP2Net/` 下的任何文件。
- 不要跳过列表里的任何一个种子，也不要额外加跑列表之外的种子（B7 的 D1/3,13,23 除外——那三个
  本来就不在你的列表里，属于故意排除，不是遗漏）。
- 不要调超参数（各方法 `adapter.py` 里的 `DEFAULT_CFG`/`PROTO_B_CFG` 已经定死，不在本次
  任务范围内）。
- B6/B7/B8 不要同时并发跑，也不要把单方法内部的 `-P` 提到 2 以上，除非你已经亲眼验证过冒烟
  测试那几条、确认显存/耗时都在预期范围内。
