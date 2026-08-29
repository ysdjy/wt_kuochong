# MTF-AViTK D2/D3 — manual run tutorial

MTF-AViTK (ViT-L/32, 309M params) is the largest and slowest model in the
9-method comparison. Historically ~30-40+ minutes per run on this 8GB
laptop GPU (sometimes longer, and it runs closest to the VRAM ceiling of
any method here). Per your preference, this is left for you to run
manually rather than launched automatically — everything else (RF,
TCN-GRU, Multi-task TCN-GRU, DC-PSR, HTT-Net, Multi-source Attention,
DP2Net-adapted, Dynamic GIN+TGP) has already been run directly.

## Before you start

Check the GPU is idle:

```powershell
nvidia-smi
```

Close any other GPU-using program (browser with hardware video decode,
other training runs, etc.) if VRAM usage is non-trivial.

## Commands (run each one in its own terminal session, one at a time — not in parallel)

```powershell
conda activate pub_baselines
cd "C:\Users\banghai\Documents\BaiduSyncdisk\西工大\王婷\论文"

python final_statistical_evidence\scripts\methods\run_mtf_avitk_transfer_task.py --task D2 --method mtf_avitk
python final_statistical_evidence\scripts\methods\run_mtf_avitk_transfer_task.py --task D3 --method mtf_avitk
```

Each command trains once (D2: source C1+C6 → target C4; D3: source C4+C6
→ target C1), evaluates once on the held-out target, and writes:

```
final_statistical_evidence/transfer_tasks/D2/mtf_avitk/
    predictions.csv, metrics.json, config.yaml, status.json,
    best.pt, DONE.flag, _native/ (full native training artifacts)
final_statistical_evidence/transfer_tasks/D3/mtf_avitk/
    (same, for D3)
```

Nothing under `baselines/mtf_avitk/` or `outputs/mtf_avitk/` is touched —
this writes only into `final_statistical_evidence/transfer_tasks/`.

## If you hit CUDA OOM

Add either flag (or both) — these only affect training efficiency, not
the model architecture or any frozen hyperparameter:

```powershell
python final_statistical_evidence\scripts\methods\run_mtf_avitk_transfer_task.py --task D2 --method mtf_avitk --grad-checkpoint
python final_statistical_evidence\scripts\methods\run_mtf_avitk_transfer_task.py --task D2 --method mtf_avitk --batch-size 4
```

## If a run gets interrupted (Ctrl-C, terminal closed, etc.)

Just rerun the same command. Since `DONE.flag` isn't written until the run
finishes cleanly, rerunning starts that one task fresh (this script does
not do fine-grained epoch-resume like the D1 5-seed sweep did for
MTF-AViTK — each D2/D3 run is short enough, and there are only 2 of them,
that a clean restart is simpler and avoids the seed-ambiguity issue that
came up with `--resume` during the original 5-seed sweep).

## After both finish

Nothing further needed from you for this method specifically. Once **all**
8 method/task cells across every method (see
`python final_statistical_evidence\scripts\status.py`) show DONE, either:

- rerun `python final_statistical_evidence\scripts\run_transfer_tasks.py --resume`
  (it will skip everything already done and auto-run
  `aggregate_transfer_results.py` at the end), or
- run `python final_statistical_evidence\scripts\aggregate_transfer_results.py`
  directly yourself.

## Expected result shape

D1 (already frozen) MTF-AViTK: Acc≈0.9704 (bootstrap point estimate, see
`results/D1_MAIN_BOOTSTRAP_CI.csv`). D2/D3 are unseen-target transfer
tasks and are very likely to score lower than D1 — this project's other
methods have shown large across-task drops (e.g. Multi-source Attention:
D1≈0.81, D2≈0.33, D3≈0.67). A lower D2/D3 number here is an expected,
legitimate result, not a bug — report it as-is (task instruction #27: do
not drop or reframe a bad transfer-task result to make the mean look
better).
