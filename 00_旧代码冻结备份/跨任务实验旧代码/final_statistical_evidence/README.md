# final_statistical_evidence

Final statistical-collation stage for the DC-PSR paper's 9-method
comparison. Produces exactly two things:

1. **D1 fixed-model bootstrap CI** — `results/D1_MAIN_BOOTSTRAP_CI.csv`
2. **D1/D2/D3 cross-task mean±std** — `results/TRANSFER_TASKS_MEAN_STD.csv`

See `PROTOCOL.md` for the full methodology and design decisions,
`METHOD_REGISTRY.yaml` for per-method source files/configs, and
`FINAL_STATISTICAL_REPORT.md` for the manuscript-ready writeup.

## Quick status check

```powershell
python final_statistical_evidence\scripts\status.py
```

## Running remaining D2/D3 training (resumable)

```powershell
conda activate dcpsr
python final_statistical_evidence\scripts\run_transfer_tasks.py --resume
```

Safe to interrupt and rerun — already-`DONE.flag`'d cells are skipped.
Automatically runs `aggregate_transfer_results.py` once every cell is done.

MTF-AViTK (largest/slowest model) is intentionally left for manual
execution — see `MTF_AVITK_MANUAL_TUTORIAL.md`.

## Directory layout

```
final_statistical_evidence/
├── README.md                      (this file)
├── PROTOCOL.md                    full methodology
├── METHOD_REGISTRY.yaml           per-method source files / frozen configs
├── STATUS.json                    live per-(task,method) run status
├── MTF_AVITK_MANUAL_TUTORIAL.md   manual run instructions for the one method not auto-run
├── FINAL_STATISTICAL_REPORT.md    manuscript-ready writeup
├── bootstrap/<method>/            D1 moving-block bootstrap outputs
├── predictions_common_universe/   D1, 304-run common test universe, per method
├── transfer_tasks/{D2,D3}/<method>/   fresh D2/D3 training outputs (D1 is reused, not retrained)
├── results/                       final CSV tables
├── logs/                          per-(task,method) runner logs
└── scripts/                       all executable code for this stage
```

## What was reused vs. freshly trained

D1: every method's prediction file is **reused** from the existing
frozen official model (never retrained) — see the source-file table in
`PROTOCOL.md`.

D2/D3: **freshly trained** for all 9 methods, `TRAIN_SEED=42`, frozen
architecture/hyperparameters (no D2/D3-specific tuning). One pre-existing
legacy D2/D3 run for the internal window-based methods
(`补充材料/小论文/7_cross_condition_generalization/`) was found but failed
D1-consistency verification and was not reused — see `PROTOCOL.md` for
details.
