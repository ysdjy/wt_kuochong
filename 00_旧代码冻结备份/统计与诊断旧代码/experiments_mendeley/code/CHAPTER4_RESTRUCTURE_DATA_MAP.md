# Chapter 4 restructure: where every number and panel comes from

All paths are relative to `experiments_mendeley/`. **CSV is the source of
truth**; `FINAL_REPORT/FINAL_TABLES.md` is a preview only. Nothing is typed by
hand — every `±` is produced by `scripts/02_aggregate_tables.py`.

Two standard deviations appear in these files and are never interchangeable:

| suffix | meaning | use it when |
|---|---|---|
| `_std_seed` | 5 fixed seeds within ONE task | a per-task cell: "MD1, B12, 0.89 ± 0.01" |
| `_std_task` | across per-task means | a cross-domain stability claim |

---

## 4.1 Datasets and Experimental Setup

| content | file |
|---|---|
| Table A dataset statistics (runs, VB range, contact time, thresholds per tool) | `00_dataset_audit/dataset_quality_report.csv` |
| Stage counts + VB range per stage per tool; T8 truncation flag | `00_dataset_audit/stage_coverage_by_tool.csv` |
| Channel inventory, sampling rates, shapes | `00_dataset_audit/hdf5_schema_report.json` |
| Which channels are primary vs restricted, and why | `00_dataset_audit/primary_channel_set.json`, `excluded_channel_report.csv` |
| Feature count / definitions | `01_features/feature_dictionary.csv` |
| Split protocol (train/val/test tools per task) | `02_protocols/task_definitions.json`, and `split` inside every `runs/*/seed*/config.json` |

**New Fig.5** (replaces old Fig.5 + Fig.6 + protocol) -> `08_plot_data/new_fig5/`
- Panel A dataset structure: `panelA_dataset_quality_report.csv`, `panelA_stage_coverage_by_tool.csv`, `panelA_channel_summary.csv`
- Panel B stage-representative signals: **not auto-generated** — pick one early / middle / late run from `panelB_per_run_stage_preview.csv` and dump the raw waveform + spectrum from the corresponding `.h5` (one short script; the run IDs are in that CSV)
- Panel C protocol: `panelC_task_definitions.json`

---

## 4.2 Overall Performance Comparison

B1–B12 on D1-M / D2-M / D3-M, 5 seeds each.

| content | file |
|---|---|
| every seed, every method, every metric | `04_overall_comparison/by_seed/overall_comparison_metrics_by_seed.csv` |
| **main comparison table** (mean ± `std_seed`, per task) | `04_overall_comparison/summary/overall_comparison_mean_std_by_task.csv` |
| Table B collapsed across tasks (mean ± `std_task`) | `04_overall_comparison/summary/overall_comparison_cross_task_summary.csv` |

Read `oracle_wear_reference` before writing this section: **B1 must be marked
as an oracle**, and B3 as using fixed-threshold *labels* derived from training
wear. Do not present either as a plain sensor-based classifier.

**New Fig.6** (replaces old Fig.8 + Fig.9 + Fig.10) -> `08_plot_data/new_fig6/`
- Panel A Acc / Macro-F1 / M-F1 / Smooth with error bars: `panelA_overall_mean_std.csv` (+ `panelA_cross_task.csv`)
- Panel B normalized confusion matrices, long format, every method/task/seed: `panelB_confusion_matrices.csv`
- Panel C middle stage (M-F1, M-Rec, M→E, M→L): `panelC_middle_stage.csv`
- Panel D consistency (Rev, Jump, Smooth): `panelD_consistency.csv`

---

## 4.3 Generalization across Different Data Scenarios

| scenario | tasks | file | suggested placement |
|---|---|---|---|
| cross-machine dual-source | D1-M / D2-M / D3-M | `05_generalization/dual_source/dual_source_mean_std.csv` | **main text** |
| cross-machine single-source | MS1–MS6 | `05_generalization/single_source/single_source_mean_std.csv` | main text (condensed) or supplementary |
| leave-one-tool-out | LOTO_T1–T9 | `05_generalization/generalization_mean_std_by_task.csv` (filter `group == leave_one_tool_out`) | **supplementary** (9 tasks is a lot of table; T8 needs its own caveat) |
| Table C, all groups, mean ± `std_task` | — | `05_generalization/summary/generalization_cross_task_summary.csv` | main text |

**New Fig.7** (replaces old Fig.11–Fig.14) -> `08_plot_data/new_fig7/`
- Panel A machine/tool generalization heatmap: `panelA_machine_generalization_source.csv`
- Panel B stage-probability evolution for a target tool: `panelB_probability_evolution/<TASK>_seed<N>.csv` (one row per run, all of `p_raw / p_fine / p_prior / p_mix / alpha / p_final`)
- Panel C cross-task mean and variability: `panelC_cross_task_variability.csv`
- Panel D radar: `panelD_radar_source.csv` (raw metrics only; normalise at plot time and say how)

---

## 4.4 Ablation and Component Analysis

A1–A6 on D1-M / D2-M / D3-M, 5 seeds. **No extra training** — all six come from the same
B11 backbone per (task, seed).

| content | file |
|---|---|
| per-seed values | `06_ablation/by_seed/ablation_metrics_by_seed.csv` |
| mean ± `std_seed` | `06_ablation/summary/ablation_mean_std.csv` |
| **Table D: merged settings + results** (replaces old Table 6 + Table 13) | `06_ablation/summary/ablation_complete_table.csv` |

`ablation_complete_table.csv` carries both the component-configuration columns
(`Raw/Fine/Prior/Ordered/Final_fusion`) and the raw numeric columns *and* the
formatted `*_display` strings — so you can re-format without re-running.

**New Fig.8** (replaces old Fig.15 + Fig.16) -> `08_plot_data/new_fig8/`
- classification panel: `ablation_task_level.csv`, `ablation_seed_level.csv`
- accuracy–smoothness tradeoff (x = Smooth, y = Acc): `tradeoff_smooth_vs_acc.csv`

---

## 4.5 Degradation Semantic Consistency Analysis

| content | file |
|---|---|
| Table E: q-MAE / q-RMSE / q-R² / Spearman / Pearson / q-Smooth | `07_semantic_consistency/q_metrics/q_metrics_summary.csv` (per-seed: `q_metrics/q_metrics_by_seed.csv`) |
| mean q_true / q_hat / VB per predicted stage | `07_semantic_consistency/stage_semantics/stage_semantics_{by_seed,summary}.csv` (+ `monotonic_wear_semantics.txt`) |
| shared hidden representation h + raw features, with machine/tool/run/stage/q/seed | `07_semantic_consistency/embeddings/representation_embeddings.parquet` |
| full probability evolution per test tool | `07_semantic_consistency/probability_evolution/` |

**New Fig.9** (replaces old Fig.17 + Fig.18 + parts of Tables 13/14) -> `08_plot_data/new_fig9/`
- Panel A raw-feature PCA/UMAP and Panel B shared-representation PCA/UMAP: both from `embeddings/` (fit the projection at plot time; the embeddings are stored un-projected on purpose)
- Panel C stage probability + q evolution: `probability_evolution/`
- Panel D semantic statistics: `panelD_semantic_statistics.csv`

---

## Not covered (by instruction)

- **Stage-related wear-estimation bias (old Table 3)** — explicitly out of
  scope for this dataset. No wear estimator is built, no numbers produced.
  See `stage_bias_experiment_status.md`.
- **PHM2010 / NASA** — untouched. Their existing single-seed numbers must not
  be placed in the same `±` column as the 5-seed Mendeley numbers; they are not
  the same quantity.
- **Leave-one-tool-out** — implemented (`--phase loto`) but not one of the four
  formal experiments. Supplementary at most.
- **T8 with/without sensitivity** — hook exists (`early_truncated` in
  `stage_coverage_by_tool.csv`); run it only after the four formal experiments.

## Cross-dataset table

The metric schema is identical for all three datasets, so once PHM2010 and NASA
are re-run through an adapter their `metrics.csv` files concatenate directly
with these, giving `Dataset | Method | Acc | Macro-F1 | M-F1 | Smooth` with
consistent mean ± std semantics. Until then, do **not** put the existing
single-seed PHM/NASA numbers in the same `±` column as the 5-seed Mendeley
numbers — they are not the same quantity.
