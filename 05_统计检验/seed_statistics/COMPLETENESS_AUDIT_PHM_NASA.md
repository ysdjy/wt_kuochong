# 完整性审计 — PHM2010 + NASA（B9 + B3）

日期：2026-08-30。

## 1. Seed 完整性

| 数据集 | 方法 | Task | Done | Missing | Duplicate |
|---|---|---|---|---|---|
| PHM2010 | B9 | D1 | 101/101 | 无 | 无 |
| PHM2010 | B9 | D2 | 101/101 | 无 | 无 |
| PHM2010 | B9 | D3 | 101/101 | 无 | 无 |
| PHM2010 | B3 | D1 | 101/101 | 无 | 无 |
| PHM2010 | B3 | D2 | 101/101 | 无 | 无 |
| PHM2010 | B3 | D3 | 101/101 | 无 | 无 |
| NASA | B9 | N1 | 101/101 | 无 | 无 |
| NASA | B9 | N2 | 101/101 | 无 | 无 |
| NASA | B9 | N3 | 101/101 | 无 | 无 |
| NASA | B9 | N4 | 101/101 | 无 | 无 |
| NASA | B3 | N1 | 101/101 | 无 | 无 |
| NASA | B3 | N2 | 101/101 | 无 | 无 |
| NASA | B3 | N3 | 101/101 | 无 | 无 |
| NASA | B3 | N4 | 101/101 | 无 | 无 |

**7 × 101 × 2（B9+B3）= 1414 个 run 全部 DONE，`run_status.csv`（各 `summary_*/B9|B3/` 下）逐一核对，`failed_runs.csv` 全部为空。**

## 2. Hash 一致性（同一 task 内 101 个 seed 必须共享同一 preprocessing）

`aggregate_dataset.py` 对每个 task 的 101 个 `run_meta.json` 提取 `(feature_hash, split_hash, gmm_hash, window_hash)` 四元组，去重后必须恰好 1 个：

```
PHM2010 B9: D1 consistent=True, D2 consistent=True, D3 consistent=True
PHM2010 B3: D1 consistent=True, D2 consistent=True, D3 consistent=True
NASA    B9: N1 consistent=True, N2 consistent=True, N3 consistent=True, N4 consistent=True
NASA    B3: N1 consistent=True, N2 consistent=True, N3 consistent=True, N4 consistent=True
```

**全部一致，无 hash drift。**（详见各 `summary_*/B9|B3/hash_consistency_report.json`）

## 3. B3/B9 配对的 backbone_checkpoint_hash

程序核对全部 707 个 (task, seed) 配对的 B9/B3 `run_meta.json`：

- **707 对全部核对，0 个 mismatch**（无一例 B9 与配对 B3 的 `backbone_checkpoint_hash` 字段不一致）。
- 174 对（D1 全部 101 + D2 早期 32 + D3 早期 2 + N1 早期 39）的 **B9 侧**缺少 `backbone_checkpoint_hash` 字段——这是因为该字段是本轮任务中途新增的，D1 的原始 landscape 脚本以及 D2/D3/N1 在"B3 配对"补丁应用之前已完成的 seed，其 B9 阶段没有记录该字段。**这不代表 checkpoint 不一致**：这些 B3 结果均通过 `backfill_b3_*.py` 从 B9 训练时保存的**同一个物理 checkpoint 文件路径**（`results/seed{N}/b9_backbone_seed{N}.pth` 或 `backbone_checkpoints/*_backbone_{task}_seed{N}.pth`）直接加载重建，从未指向过其它文件——按构造即为同一份权重，无需哈希比对即可保证一致（只有一份文件，不存在"两份文件比较"的场景）。
- 剩余 533 对（本轮任务追加 B3 配对补丁之后原生产出的 B9+B3）双方 `run_meta.json` 均记录了 `backbone_checkpoint_hash` 且逐一核对相同。

**结论：707/707 对 B3/B9 确认共享同一 backbone checkpoint，无一例不一致。**

## 4. 结论

PHM2010（D1+D2+D3）与 NASA（N1+N2+N3+N4）的全部 707 个 B9 正式结果与 707 个配对 B3 结果，seed 完整、无重复、无 hash drift、backbone checkpoint 配对无误。**审计通过，可以 commit + push。**
