# methods/_internal_shared/

Not one of the 9 registered methods. Holds `code/pipeline.py`, the vendored,
task-parameterized data/model pipeline shared by **B2** (TCN-GRU), **B3**
(Multi-task TCN-GRU), and **B9** (DC-PHSR) — the three methods that all descend
from `代码/main_experiment_3_fgds_psi_optimized.py`. Factored out here instead
of duplicated three times (~900 lines each). B1 (RF) also uses the same
run-level feature file and task-routing logic but does not need any of this
module's PyTorch model code.

See `code/pipeline.py`'s module docstring for exact provenance (legacy source
paths + git commit) and the itemized list of what was changed vs. the original
code (only task/seed/output routing and paths — architecture/hyperparameters
untouched).

Correctness check performed 2026-08-30: `pipeline.prepare_task_data(["C1","C4"],
"C6")`'s 45 selected features match `shared/reproducibility/
PHM2010_D1_frozen_preprocess/selected_features_seed42.json` **bit-exact, in
order** — confirms this vendored port reproduces the frozen D1 preprocessing
exactly.
