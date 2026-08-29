# monotonic_q_loss: what was wrong and what it is now

## Before

`main_experiment_3_fgds_psi_optimized.py:738-742`

```python
def monotonic_q_loss(q_hat):
    if q_hat.numel() <= 2:
        return torch.tensor(0.0, device=q_hat.device)
    return torch.relu(-(q_hat[1:] - q_hat[:-1])).mean()
```

`main_experiment_3_fgds_psi_optimized.py:675`

```python
loader = DataLoader(StageDataset(X, ys, yf, yq),
                    batch_size=BATCH_SIZE,
                    shuffle=(split_name == "final_train"))
```

The loss penalises `q_hat[i] > q_hat[i+1]` for rows that happen to be adjacent
**in the batch tensor**. In the main experiment `split_name == "final_train"`,
so the loader shuffles and those rows are arbitrary windows from arbitrary
positions of arbitrary conditions. The term therefore expresses nothing about
monotonic degradation; it pushes the network towards an arbitrary ordering of a
random permutation. With `LAMBDA_MONO = 0.03` the effect is small but it is
noise, not the "weak monotonic regularization" the paper describes
(`main.tex:403`).

### The same symbol has two meanings in the paper

The shuffle flag is a magic string comparison, and the downstream scripts do
not use the same string:

| script | `split_name` passed | resulting `shuffle` |
|---|---|---|
| `main_experiment_3_fgds_psi_optimized.py` (main experiment) | `"final_train"` | **True** |
| `7.7跨工况实验.py:305` (cross-condition) | `"train"` | **False** |
| `1.3细化的阶段分类.py:1245` (architecture search) | `"train"` | **False** |

So `L_mono` is noise in the main experiment and an (accidentally) valid
within-batch ordering constraint in the cross-condition experiment. `BEST_ARCH`
was additionally selected under the unshuffled regime and then reused in the
shuffled one.

## After

`dcpsr/model.py`

```python
def sequence_aware_monotonic_loss(q_hat, seq_index, order_key):
    q = q_hat.view(-1)
    same = seq_index[1:] == seq_index[:-1]
    adjacent = (order_key[1:] - order_key[:-1]) == 1
    mask = (same & adjacent).float()
    if mask.sum() < 1:
        return q.sum() * 0.0
    viol = torch.relu(-(q[1:] - q[:-1])) * mask
    return viol.sum() / mask.sum()
```

Every window carries its `(sequence_index, order_key)`. The penalty applies
only to pairs that are genuinely consecutive windows of the same degradation
sequence; everything else is masked out. The mean is taken over the number of
*valid* pairs, so the magnitude of the term does not depend on how many
cross-sequence neighbours a batch happens to contain.

To make valid pairs common rather than accidental, batches are drawn as
**shuffled contiguous blocks** (`TemporalBlockSampler`): each batch is a
contiguous run of one sequence, and the order of blocks is reshuffled every
epoch. Stochasticity is preserved; temporal adjacency inside a batch is not
destroyed. `shuffle` is now an explicit boolean argument, not a string
comparison.

`scripts/selftest.py` check 5 asserts both halves of the fix: an ordered pair
with a genuine decrease produces a non-zero loss, and the same tensor with
non-adjacent `order_key`s produces exactly `0.0`.

## Scope

This fix is applied to the **third (Mendeley) dataset only**. The published
PHM2010 and NASA numbers are left untouched, as instructed. If you later
re-run PHM2010 with the corrected loss, report it as a corrected re-run rather
than silently replacing the existing table.
