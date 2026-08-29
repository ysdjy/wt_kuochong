# DC-PSR Code Audit

Scope: all 13 scripts under `/mnt/user-data/uploads/论文/代码/` (read-only, unmodified).
Paper source cross-checked: `/mnt/user-data/uploads/论文/投稿版20260709/main.tex`.
Third-dataset manifest inspected: `/mnt/user-data/uploads/论文/Multivariate time series data of milling processes with varying tool wear and machine tools/filelist.csv`.

Line references are `file:line` against the staged copies. Where a claim could not be checked because the referenced artefact is not in the upload, it is marked **unverified**.

---

## 1. Script inventory

| Script | Paper artefact produced | Inputs (exact) | Outputs (exact) |
|---|---|---|---|
| `main_experiment_3_fgds_psi_optimized.py` | Main experiment (B12 row for Table 9 / `tab:8`), ablation A1–A6 table, Fig01–Fig08; also serves as the **importable library** for all 7.x scripts | `C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\1run_run_level_features\02_features\run_level_features_all.csv` (`:78-79`) | Under `$FGDS_RUN_DIR` (default `…\小论文\3_main_experiment_fgds_psi`, `:82-89`): `1_results/condition_relative_stage_thresholds.csv`, `split_and_stage_summary.csv`, `selected_features.csv`, `gmm_fine_state_mapping.csv`, `training_history.csv`, `FINAL_probability_param_ranking.csv`, `FINAL_ablation_outputs.csv`, `FINAL_classification_report_A6.csv`, `FINAL_confusion_matrix_A6.csv`, `FINAL_main_result_B12_for_Table9.csv`, `FINAL_experiment_summary.csv`; `2_models/fgds_psi_best_model.pth`; `4_predictions/FINAL_best_val_predictions.csv`, `test_C6_predictions_full_internal.csv`, `FINAL_best_test_C6_predictions.csv`; `3_figures/Fig01…Fig08*.png` (`:946,957,971,987,1006,1026,1046,1063`) |
| `7.3主实验.py` | Same as above — **duplicate** | same | same |
| `7.4对比实验.py` | Table 8 (`tab:8`, B1–B12 on C1+C4→C6), Fig8–Fig13 | `base.load_feature_table()` → same PHM CSV | `$COMPARISON_RECHECK_DIR` (default `…\小论文\4_comparison_experiment_recheck`): `1_results/FINAL_comparison_results.csv`, `FINAL_comparison_classification_reports_long.csv`, `FINAL_comparison_confusion_matrices_long.csv`, `FINAL_comparison_predictions.csv`, `DEBUG_B10_vs_B11_check.csv`, `FINAL_comparison_summary.txt`; `2_figures/Fig8…Fig13*.png`; `3_models/B8_TCN.pth`, `B9_GRU.pth`, `B10_TCN_GRU.pth`, `B11_B12_multitask_tcn_gru.pth` |
| `7.6消融实验.py` | Ablation Tables 10–12 and Fig10–Fig14 | same PHM CSV via `base` | `$FGDS_ABLATION_DIR` (default `…\小论文\6_ablation_experiment`): `Table10_ablation_summary.csv`, `Table11_ablation_classification_report.csv`, `Table12_ablation_confusion_matrix.csv`, `ablation_probabilities_test_C6.csv`, `ablation_experiment_summary.csv`, `fixed_main_probability_params.csv`, `training_history.csv`, `figures/Fig10…Fig14*.png`, `models/ablation_tcn_gru_multitask_best_model.pth` |
| `7.7跨工况实验.py` | Cross-condition Tables 12–16 (paper `tab:9` reports a 4-task subset) and Fig15–Fig20 | same PHM CSV via `base` | `$FGDS_CROSS_DIR` (default `…\小论文\7_cross_condition_generalization`): `Table12_dual_source_cross_condition_results.csv`, `Table13_single_source_cross_condition_results.csv`, `Table14_average_cross_condition_performance.csv`, `Table15_cross_condition_classification_report.csv`, `Table16_cross_condition_confusion_matrix.csv`, `cross_condition_B12_probabilities.csv`, `figures/Fig15…Fig20.png`, `models/<task>_B8_TCN.pth` … `<task>_B11_B12_multitask_tcn_gru.pth` |
| `7.9磨损估计.py` | Section 5.4 probability–wear consistency (Table 14/`tab:14` style, Fig17–Fig23) | `…\小论文\6_ablation_experiment\ablation_probabilities_test_C6.csv` (first preference, `:34-40`), fallback search in `SEARCH_DIRS` (`:40-48`); plus `…\1run_run_level_features\02_features\run_level_features_all.csv` (`:32`) for true VB | `…\小论文\9_probability_wear_consistency_analysis\`: `Data_5_4_A6_probability_wear_trajectory.csv`, `Data_5_4_all_ablation_probability_wear_trajectory.csv`, `Table_5_4_probability_wear_consistency_metrics.csv`, `Table_5_4_stagewise_degradation_statistics.csv`, `Table_5_4_probability_q_correlations.csv`, `Table_5_4_boundary_smoothness_metrics.csv`, `Fig17…Fig23*.png` |
| `7.9.1阶段偏差实验.py` | Stage-related VB bias correction analysis (not clearly mapped to a numbered paper table — **unverified**) | `D:\桌面\博士开题\公开数据\1PHM\…\run_level_features_all.csv` (`:33`); auto-discovered `FINAL_best_by_test_predictions.csv` / `FINAL_test_C6_predictions.csv` / `test_C6_predictions.csv` (`:40-45`); `ablation_probabilities_test_C6.csv` (`:47-51`); a **MICFS-TCN** VB-regression run providing `FINAL_model_ranking_by_valRMSE.csv` + `04_all_predictions/*_val_predictions.csv` (`:263-283`) — that upstream script is **not present in the upload** | `D:\桌面\博士开题\2专利\代码\小论文\12_stage_bias_analysis\`: `1_results/stage_bias_summary_base.csv`, `correction_coefficients.csv`, `correction_overall_metrics.csv`, `correction_stagewise_metrics.csv`, an `.xlsx` workbook; `2_predictions/final_predictions_with_bias_correction.csv`; `3_figures/Fig_stagewise_bias_bar.png`, `Fig_stagewise_mae_rmse.png`, `Fig_residual_vs_q_stage.png`, `Fig_wear_curve_prediction_comparison.png`, `Fig_abs_error_distribution_by_stage.png`, `Fig_stage_probability_and_correction_weight.png`; `4_logs/…` |
| `8.1.1共享表征数据.py` | Data export for the Chapter-5 representation-space figures | `…\3_main_experiment_fgds_psi\2_models\fgds_psi_best_model.pth` (`:52`), `…\1_results\selected_features.csv` (`:53`), `…\6_ablation_experiment\ablation_probabilities_test_C6.csv` (`:54`), plus the PHM feature CSV via `base` | `$CH5_VIS_OUT_ROOT/figures_representation_space/`: `repr_raw_features.csv`, `repr_hidden_hct.csv`, `repr_dcpsr_final_state.csv`, and a copy of itself as `extract_hidden_representation.py` |
| `8.1.2共享表征图.py` | `Fig5_repr_*` representation-space figures | `repr_raw_features.csv`, `repr_hidden_hct.csv`, `repr_dcpsr_final_state.csv` (`:47-49`); **fallback** `…\4_comparison_experiment_recheck\1_results\FINAL_comparison_predictions.csv` + `…\9_probability_wear_consistency_analysis\Data_5_4_A6_probability_wear_trajectory.csv` (`:198-199`) | `Fig5_repr_main_umap.*`, `Fig5_repr_main_pca.*`, `Fig5_repr_main_misclassified.*`, `Fig5_repr_dcpsr_final_umap.*`, `Fig5_repr_dcpsr_final_pca.*`, `README_representation_space.md`, and (fallback path only) `repr_raw_features_PROXY.csv`, `repr_hidden_hct_PROXY.csv` (`:239-240`) |
| `8.1.3单独优化.py` | Redraw of one figure | `figures_representation_space/repr_raw_features.csv`, `repr_hidden_hct.csv` (`:43-44`) | `Fig5_repr_main_misclassified_v2.png/.pdf`, `data_exports/Fig5_repr_main_misclassified_v2_data.csv` |
| `9.1nasa数据实验.py` | NASA "stage-aware optimized" cross-case run (`original` + `stagebalanced` splits) | `C:\Users\wangting\Desktop\博士开题\公开数据\1NASA\mill.mat` (`:50`) | `…\1NASA\nasa_dcpsr_results_stageaware_opt\`: `NASA_experiment_config_stageaware.json`, `NASA_case_summary_optimized.csv`, `NASA_run_level_features_with_labels_optimized.csv`, `NASA_online_relative_features_optimized.csv`, `NASA_cross_case_tasks_original.{csv,json}`, `NASA_cross_case_tasks_stagebalanced.{csv,json}`, `NASA_validation_case_selection.csv`, `Table_NASA_cross_case_B9_B12_metrics_{by_seed,optimized}.csv`, `Table_NASA_cross_case_mean_std_optimized.csv`, `Table_NASA_original_split_mean_std.csv`, `Table_NASA_stagebalanced_split_mean_std.csv`, `Table_NASA_q_consistency_B12_*.csv`, `NASA_experiment_diagnostics_stageaware.csv`, `NASA_B12_best_fusion_params_stageaware.csv`, `NASA_B12_fusion_search_log.csv`, `NASA_B11_best_training_configs.csv`, `NASA_stage_label_strategy_comparison.csv`, `NASA_stage_thresholds_by_task.csv`, `NASA_q_validation_quality_by_task.csv`, `Fig_NASA_B9_B12_mean_metrics.png`, `Pred_NASA_*.csv` |
| `run_nasa_bestcase_candidate_split.py` | NASA Tables `tab:11`/`tab:12` (N1–N4) — see §4.6; superset of `9.1nasa数据实验.py` | same `mill.mat` (`:60`) | `…\1NASA\nasa_dcpsr_results_bestcase_split\` (`:61`): `NASA_experiment_config_bestcase_split.json`, `NASA_candidate_splits_all.{csv,json}`, `Table_NASA_all_candidate_splits_metrics{,_by_seed}.csv`, `Table_NASA_all_candidate_splits_q_B12.csv`, `NASA_all_candidate_diagnostics.csv`, `NASA_all_candidate_B12_fusion_params.csv`, `NASA_B12_fusion_search_log_all_candidates.csv`, `NASA_B11_training_configs_all_candidates.csv`, `NASA_stage_label_strategy_comparison.csv`, `NASA_stage_thresholds_by_task.csv`, `NASA_q_validation_quality_by_task.csv`, `NASA_B12_bestcase_selection_scores.csv`, `NASA_selected_bestcase_splits.{csv,json}`, `Table_NASA_selected_bestcase_B9_B12_metrics.csv`, `Table_NASA_selected_bestcase_mean_std.csv`, `Table_NASA_selected_bestcase_q_B12{,_mean}.csv`, `NASA_bestcase_selection_summary.txt`, `Pred_NASA_*.csv` |
| `1.3细化的阶段分类.py` | Ancestor / dev version ("44run"). Produces the architecture-search evidence (`FINAL_proxy_model_ranking.csv`) that `BEST_ARCH` in the main script was frozen from | same PHM CSV (`:58-59`) | `…\PHM实验\44run_fine_state_tcn_gru_strict_no_leak\`: `00_final_results/FINAL_condition_relative_stage_thresholds.csv`, `FINAL_split_metadata.csv`, `FINAL_selected_stage_invariant_features.csv`, `FINAL_feature_count_summary.csv`, `FINAL_gmm_fine_state_mapping.csv`, `FINAL_probability_param_ranking.csv`, `FINAL_proxy_model_ranking.csv`, `FINAL_best_test_C6_predictions.csv`, `FINAL_all_predictions_val_and_test.csv`, `FINAL_classification_reports_long.csv`, `FINAL_confusion_matrices_long.csv`, `FINAL_stage_ratio_comparison.csv`, `FINAL_experiment_summary.csv`; `01_intermediate/…`; `02_models/…`; `03_figures/…`; `04_predictions/FINAL_best_val_predictions.csv` |

### 1a. Verification of the known duplication

Confirmed. Byte comparison:

- `7.3主实验.py` is CRLF (1218 CRLF pairs), `main_experiment_3_fgds_psi_optimized.py` is LF (0).
- After newline normalisation the only differences are two indentation blocks: the loss expression (`main:753-757` vs `7.3主实验.py` same lines, 4 extra spaces of continuation indent) and the closing paren of the `compact_prediction_table(...).to_csv(...)` call (`main:1143` `    )` vs `7.3` `        )`).
- Semantically identical. `7.3主实验.py` is dead weight; the 7.x scripts all `import main_experiment_3_fgds_psi_optimized as base` (`7.4对比实验.py:41`, `7.6消融实验.py:83`, `7.7跨工况实验.py:70`, `8.1.1共享表征数据.py:34`).

---

## 2. Module-level reuse map — `main_experiment_3_fgds_psi_optimized.py`

Classification key: **(a)** dataset-independent / directly reusable; **(b)** PHM2010-coupled; **(c)** needs refactor.

### (a) Dataset-independent / directly reusable as-is

| Symbol | Line | Note |
|---|---|---|
| `set_seed` | 161 | Missing `torch.backends.cudnn.deterministic`; see §4.8. |
| `savefig`, `savefig_multi` | 169, 176 | |
| `calc_q_metrics` | 239 | |
| `clf_metrics` | 254 | Assumes exactly 3 classes (`labels=[0,1,2]`) — fine for the method, not dataset-specific. |
| `Chomp1d`, `TemporalBlock`, `TCNGRUMultiTask` | 679, 687, 703 | Only coupling is `N_FINE_STATES` (module global). |
| `class_weights` | 732 | |
| `monotonic_q_loss` | 738 | Reusable, but semantically broken under shuffling — §4.1. |
| `qhat_prior` | 825 | Constants `sigma=0.17`, centres `0.18/0.50/0.84` hardcoded (`:827-831`); should become parameters. |
| `fine_to_stage_prob` | 835 | Hardcodes S0→E, S1..S3→M, S4→L (`:840`) — matches the paper, but tied to `N_FINE_STATES == 5`. |
| `temperature_scale` | 843 | |
| `causal_ordered_filter` | 850 | Transition matrix and `alpha` init are literals inside the function (`:851-853`). |
| `probability_param_search` | 907 | Grid comes from module globals. |
| `fit_train_gmm` / `assign_fine_states` | 612, 627 | Depend only on `q_true` / `rate_norm` columns. |
| `select_features_train_only` | 566 | Mostly generic **but** the DI term is hardcoded to two conditions — see (c). |

### (b) PHM2010-coupled

| Symbol | Line | Offending identifiers / lines |
|---|---|---|
| `ROOT`, `FEATURE_FILE`, `RUN_DIR` | 78-82 | `Path(r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM")`, `"PHM实验" / "1run_run_level_features" / "02_features" / "run_level_features_all.csv"` |
| `normalize_condition_name` | 184 | `if s in ["1","C1"]: return "C1"` … `["4","C4"]` … `["6","C6"]` (`:186-191`) |
| `infer_vb_column` | 195 | Candidate list `["VB","VB_max","vb","vb_max"]`, then `["vb","vb_max","vbmax"]` (`:196-203`); raises `ValueError("Cannot find VB or VB_max column.")` |
| `is_meta_or_label_col` | 206 | PHM-specific exclusions: `"flute_1","flute_2","flute_3"`, `"dominant_flute","final_dominant_flute"`, `"cut","cut_index","cutid","cut_id"` (`:210-219`); substring blacklist `"vb","flute",…` (`:224-228`) |
| `load_feature_table` | 419 | `df["condition"]`, `df["run_id"]` assumed to exist (`:425-426`); `df[df["condition"].isin(["C1","C4","C6"])]` (`:428`); sort key `["condition","run_id"]` |
| `define_condition_relative_stages` | 432 | Groups by literal column `"condition"`, orders by `"run_id"` (`:434-435`); smoothing windows 7 and 5 tuned to PHM run counts (`:436,438`) |
| `split_grouped_lifecycle` | 465 | `for cond in ["C1","C4"]` (`:467`); `test_c6 = df[df["condition"]=="C6"]` (`:483`); split names `"final_train"/"final_internal_val"/"test_C6"` (`:485`) |
| `build_online_features_for_subset` | 507 | Groups by `"condition"`, orders by `"run_id"` (`:509-510`); `meta = ["condition","run_id","VB",…]` (`:537`); merge on `["condition","run_id"]` (`:538`) |
| `feature_cols_from` | 550 | Same hardcoded meta set (`:551-553`) |
| `build_windows` | 638 | Groups by `"condition"`, contiguity test `np.all(np.diff(runs[start:end+1]) == 1)` on `run_id` (`:646`); emits `"cut_index"` (`:653`) |
| `apply_probability_inference` | 862 | `sort_values(["condition","cut_index"])` (`:863`); ordered filter applied per `"condition"` group (`:879-881`) |
| `stage_consistency_metrics` | 287 | `groupby("condition")`, `sort_values(["condition","cut_index"])` (`:298`) |
| `manuscript_metric_row` | 321 | Default `split="test_C6"` (`:321`) |
| `plot_stage_definition` | 932 | `for ax, cond in zip(axes, ["C1","C4","C6"])` (`:934`) |
| `plot_gmm_3d` | 960 | Titles `"Training space (C1+C4)"`, `"External test space (C6)"` (`:962`) |
| `plot_feature_pca` | 1049 | `assign(domain="train C1+C4")`, `"test C6"` (`:1050`, `:1055`) |
| `main` | 1080 | Split dict literal `{"final_train":…, "final_internal_val":…, "test_C6":…}` (`:1092-1096`); summary strings `"C1+C4 final_train blocks"`, `"C6"` (`:1171-1172`) |

### (c) Needs refactor to be dataset-independent (works today only by accident of the PHM setup)

1. `select_features_train_only:566` — the DI term is computed between exactly the first two conditions present:
   ```python
   conds = sorted(ft["condition"].unique())
   ...
   g0, g1 = ft[ft["condition"] == conds[0]][c].values, ft[ft["condition"] == conds[1]][c].values
   instability = abs(np.nanmean(g0) - np.nanmean(g1)) / (0.5*(np.nanstd(g0)+np.nanstd(g1)) + 1e-8)
   ```
   (`:580-586`). With 1 training group DI is silently 0 for every feature (`if len(conds) >= 2:` guard, `:582`); with ≥3 groups it silently ignores all but two. For Mendeley (up to 9 tools / 3 machines in train) this degenerates. Needs a pairwise-max or ANOVA-style generalisation.
2. `make_pack:673` — `shuffle=(split_name == "final_train")` (`:675`). Shuffling is keyed off a **magic string**, so any caller that names its training split anything else silently trains unshuffled. This is not hypothetical: `7.7跨工况实验.py:305` passes `"train"`, and `1.3细化的阶段分类.py:1245` passes `"train"`. See §4.1/§4.8.
3. `build_online_features_by_split:542` — contains dead code (`build_online_features_for_subset(sub, name) if False else build_online_features_for_subset(sub, raw_cols, name)`, `:545-546`).
4. `get_raw_numeric_sensor_cols:496` — column admissibility is decided by a name blacklist (`is_meta_or_label_col`). Any new dataset must have its label/index columns re-encoded into that blacklist or leakage occurs silently. This must become an explicit adapter-declared column list, not a regex heuristic.
5. `evaluate:899` — `col = "stage_pred_raw" if method == "raw" else f"stage_pred_{method}"`; string-keyed dispatch shared with `stage_consistency_metrics:293`. Brittle but portable.
6. Module-level side effects: directories are created at import time (`:88-89`) and `probability_param_search` / `define_condition_relative_stages` / `fit_train_gmm` / `select_features_train_only` all write CSVs into `DIR_RESULT` as a side effect (`:461`, `:487`, `:621`, `:608`, `:927`). Importing the module for reuse therefore mutates the filesystem — the 7.x scripts each work around this by reassigning `base.DIR_RESULT` after import (`7.6消融实验.py:88-92`, `7.7跨工况实验.py:75-79`) or by pre-setting `FGDS_RUN_DIR` (`7.4对比实验.py:34`).

---

## 3. Hardcoded paths, conditions, environment variables

### 3.1 Literal filesystem paths

| File:line | Literal |
|---|---|
| `main_experiment_3_fgds_psi_optimized.py:78` / `7.3主实验.py:78` | `C:\Users\wangting\Desktop\博士开题\公开数据\1PHM` |
| `main…:79` | `…\1PHM\PHM实验\1run_run_level_features\02_features\run_level_features_all.csv` |
| `main…:82` | default out `…\1PHM\PHM实验\小论文\3_main_experiment_fgds_psi` |
| `7.4对比实验.py:32` | `C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文\4_comparison_experiment_recheck` |
| `7.4对比实验.py:38` | `sys.path.append(r"C:\Users\wangting\Documents\Codex\2026-05-17\files-mentioned-by-the-user-docx")` |
| `7.6消融实验.py:47` | `…\小论文\6_ablation_experiment` |
| `7.6消融实验.py:66-67` | import fallbacks `C:\Users\wangting\Documents\Codex\2026-05-17\files-mentioned-by-the-user-docx\…`, `D:\CodeTou-Download\pythonDemo\pythonDemo\阶段信息小论文\…` |
| `7.6消融实验.py:87-88` | re-pins `base.ROOT` / `base.FEATURE_FILE` to the `C:` PHM path |
| `7.7跨工况实验.py:46` | `…\小论文\7_cross_condition_generalization` |
| `7.7跨工况实验.py:62-63`, `74-75` | same import fallbacks and `base.ROOT` re-pin |
| `7.9磨损估计.py:30` | `C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文` |
| `7.9.1阶段偏差实验.py:27` | `D:\桌面\博士开题\公开数据\1PHM` — **different machine/root from every other script** |
| `7.9.1阶段偏差实验.py:28,37` | `D:\桌面\博士开题\2专利\代码\小论文\12_stage_bias_analysis`, `D:\桌面\博士开题\2专利\代码\小论文` |
| `8.1.1共享表征数据.py:39,52-54` | `…\小论文` + `3_main_experiment_fgds_psi\2_models\fgds_psi_best_model.pth`, `…\1_results\selected_features.csv`, `6_ablation_experiment\ablation_probabilities_test_C6.csv` |
| `8.1.2共享表征图.py:37`, `8.1.3单独优化.py:29` | `…\小论文` |
| `9.1nasa数据实验.py:50-51` | `…\1NASA\mill.mat`, `…\1NASA\nasa_dcpsr_results_stageaware_opt` |
| `run_nasa_bestcase_candidate_split.py:54-55` then `60-61` | `mill.mat` and `nasa_dcpsr_results_stageaware_opt`, **immediately overwritten** by `mill.mat` and `nasa_dcpsr_results_bestcase_split` — the first `OUT_DIR.mkdir` at `:56` still creates the stale directory |
| `9.1nasa数据实验.py:147` / `run_nasa…:165` | fallback `list((Path.home()/"Desktop").rglob("mill.mat"))` — a recursive scan of the user's Desktop |
| `1.3细化的阶段分类.py:58-59, 64` | `…\1PHM`, PHM feature CSV, `…\PHM实验\44run_fine_state_tcn_gru_strict_no_leak` |

There is no config file and no CLI argument parsing anywhere.

### 3.2 Condition literals `C1` / `C4` / `C6`

- `main…/7.3主实验.py`: `:186-191` (`normalize_condition_name`), `:428` (`isin(["C1","C4","C6"])`), `:467` (`for cond in ["C1","C4"]`), `:483` (`condition == "C6"`), `:934` (`zip(axes, ["C1","C4","C6"])`), `:962`, `:1050`, `:1055`, `:1171-1172` (summary strings).
- `7.4对比实验.py`: `:3`, `:274` (titles only; logic is inherited from `base`).
- `7.6消融实验.py`: `:601-602` (summary strings).
- `7.7跨工况实验.py`: `:112-124` (`DUAL_TASKS` / `SINGLE_TASKS`), `:490-491`, `:589`, `:613` (`conds = ["C1","C4","C6"]`).
- `7.9.1阶段偏差实验.py`: `:143`, `:158`, `:218` and `:307-315` — filters everything to `condition == "C6"`, and explicitly excludes C6 from calibration.
- `1.3细化的阶段分类.py`: `:162-167`, `:352`, `:421`, `:439`, `:1290-1302` (proxy folds `C1->C4`, `C4->C1`), `:1344`, `:1484`, `:1569-1571`.
- NASA scripts use integer `case` IDs instead; condition-equivalents are `FIXED_TASKS` at `9.1nasa数据实验.py:92-97` and `run_nasa_bestcase_candidate_split.py:110-115`.

### 3.3 Environment variables

| Variable | Read at | Default |
|---|---|---|
| `FGDS_RUN_DIR` | `main…:82`, `7.3主实验.py:82` | `…\小论文\3_main_experiment_fgds_psi` |
| `FGDS_RUN_DIR` | set via `os.environ.setdefault` at `7.4对比实验.py:34`, `7.6消融实验.py:59`, `7.7跨工况实验.py:56` | points into each script's own `_base_cache` / `intermediate_main_outputs` |
| `COMPARISON_RECHECK_DIR` | `7.4对比实验.py:30-33` | `…\4_comparison_experiment_recheck` |
| `FGDS_ABLATION_DIR` | `7.6消融实验.py:45-49` | `…\6_ablation_experiment` |
| `FGDS_CROSS_DIR` | `7.7跨工况实验.py:43-48` | `…\7_cross_condition_generalization` |
| `PAPER_ROOT` | `8.1.1共享表征数据.py:37`, `8.1.2共享表征图.py:35`, `8.1.3单独优化.py:27` | `…\PHM实验\小论文` |
| `CH5_VIS_OUT_ROOT` | `8.1.1:41`, `8.1.2:39`, `8.1.3:31` | `$PAPER_ROOT\10_第五章顶刊风格可视化` |

No environment variables exist for the input data path, the NASA `mill.mat` path, or any of the `7.9*` paths.

---

## 4. Known inconsistencies and potential bugs

### 4.1 `monotonic_q_loss` is applied across shuffled batch neighbours (confirmed bug)

Definition — `main_experiment_3_fgds_psi_optimized.py:738-742`:

```python
def monotonic_q_loss(q_hat):
    if q_hat.numel() <= 2:
        return torch.tensor(0.0, device=q_hat.device)
    return torch.relu(-(q_hat[1:] - q_hat[:-1])).mean()
```

It penalises decreases between *consecutive rows of the tensor `q_hat`*, i.e. between position `i` and `i+1` **inside the mini-batch**. It is called on the model output only — no time index is passed:

`main…:753-757`
```python
loss = (
    LAMBDA_STAGE * F.cross_entropy(out["stage_logits"], ys, weight=stage_w)
    + LAMBDA_FINE * F.cross_entropy(out["fine_logits"], yf, weight=fine_w)
    + LAMBDA_Q * F.smooth_l1_loss(out["q_hat"], yq)
    + LAMBDA_MONO * monotonic_q_loss(out["q_hat"])
)
```

The DataLoader shuffles the training split — `main…:673-676`:
```python
def make_pack(df_sub, features, L, split_name):
    X, ys, yf, yq, meta = build_windows(df_sub, features, L, split_name)
    loader = DataLoader(StageDataset(X, ys, yf, yq), batch_size=BATCH_SIZE, shuffle=(split_name == "final_train"))
```

**Conclusion:** in `main_experiment_3_fgds_psi_optimized.py` / `7.3主实验.py` (`:1123`, `split_name="final_train"`), `7.4对比实验.py:309` and `7.6消融实验.py:573`, the training loader is shuffled, so the monotonic term constrains `q̂` between **arbitrary, temporally unrelated window pairs** that happen to be adjacent in a random permutation. It is not a monotonicity-in-time regulariser; in expectation it is a weak, direction-arbitrary penalty on within-batch `q̂` variance ordering. Because batches are re-permuted every epoch, the term pushes `q̂` toward a flatter, near-constant output rather than toward temporal monotonicity.

**Additional, inconsistent behaviour:** `7.7跨工况实验.py:305` calls `base.make_pack(feat_train, selected, L, "train")`. `"train" != "final_train"`, so `shuffle=False`. In the cross-condition experiment the training loader is **not** shuffled, batches are in `(condition, run_id)` order, and `monotonic_q_loss` therefore *does* act on temporally adjacent samples (except at the two boundaries per batch and at condition boundaries). The same magic-string mismatch exists in `1.3细化的阶段分类.py:1245` (`"train"`, proxy architecture search) vs `:1521` (`"final_train"`, final model). So the architecture frozen into `BEST_ARCH` (`main…:126-132`) was selected under unshuffled training, then used with shuffled training.

Net effect: the loss term named `L_mono` in the paper (`main.tex:400-403`, `:534`) means two different things in two different experiments of the same paper, and in the headline experiment it does not implement what the paper describes.

### 4.2 B1 is an oracle-input baseline (confirmed)

`7.4对比实验.py:321-327`:

```python
# B1 fixed wear-threshold reference, train thresholds only.
th_e = float(feat_train["VB_smooth"].quantile(base.Q_EARLY))
th_l = float(feat_train["VB_smooth"].quantile(base.Q_LATE))
vb_test = test_by_cut.loc[meta["cut_index"].values, "VB_smooth"].values
y_b1 = np.where(vb_test <= th_e, 0, np.where(vb_test >= th_l, 2, 1)).astype(int)
preds["B1"], probs["B1"] = y_b1, one_hot(y_b1)
method_info["B1"] = ("Fixed-stage Rule", "Fixed-stage", "wear-threshold reference")
```

The *thresholds* are train-only (that is what the comment asserts). The *input* is `feat_test["VB_smooth"]`, i.e. the smoothed **true measured flank wear of the C6 test set**, produced by `base.define_condition_relative_stages` at `main…:436` from the ground-truth `VB` column.

**State clearly: B1 is an oracle baseline.** At inference it consumes the ground-truth wear measurement that every other baseline (B3–B12) must infer from sensor signals. It is not a sensor-based method and its Table-8 row (`main.tex:756`, Acc 0.6447, Macro-F1 0.6145) is not comparable to B3–B12. It is also a *weak* oracle only because the C6 thresholds differ from the C1/C4 thresholds — which is precisely the point the paper wants to make, but the paper's Table 6 (`main.tex:600`) describes B1 only as "Fixed threshold / Rule" and does not disclose that it reads true VB at test time.

Note the same true-VB leak is *not* present in B3: `7.4对比实验.py:347-352` uses train `VB_smooth` only to derive **training labels** `ytr_fixed`, then fits on sensor features `Xtr` and predicts from `Xte`. B3 is clean.

### 4.3 B2 is a RandomForest q-proxy rule, and it contains a unit bug

`7.4对比实验.py:329-344`:

```python
# B2 sensor-derived q_proxy, no C6 true VB/q in prediction.
q_reg = RandomForestRegressor(n_estimators=300, random_state=base.RANDOM_SEED, n_jobs=-1, min_samples_leaf=2)
q_reg.fit(Xtr, feat_train["q_true"].values)
q_train_proxy = np.clip(q_reg.predict(Xtr), 0.0, 1.0)
q_test_proxy  = np.clip(q_reg.predict(Xte), 0.0, 1.0)
rate_train_proxy = causal_rate(q_train_proxy)
rate_test_proxy  = causal_rate(q_test_proxy)
rate_test_proxy  = minmax_train_apply(rate_train_proxy, rate_test_proxy)
theta_E_proxy = float(np.quantile(q_train_proxy, base.Q_EARLY))
theta_L_proxy = float(np.quantile(q_train_proxy, base.Q_LATE))
theta_v_proxy = float(np.quantile(rate_train_proxy, base.RATE_LATE_Q))
y_b2 = np.where(q_test_proxy <= theta_E_proxy, 0,
                np.where((q_test_proxy >= theta_L_proxy) | (rate_test_proxy >= theta_v_proxy), 2, 1)).astype(int)
preds["B2"], probs["B2"] = y_b2, one_hot(y_b2)
method_info["B2"] = ("Relative-stage Proxy Rule", "Relative-stage", "sensor-derived q_proxy rule")
b2_uses_true_vb = False
```

So: **B2 is a RandomForest q-proxy → threshold rule**, not a rule on true VB. `b2_uses_true_vb` is a hardcoded literal `False` (`:344`) that is then printed as if it were a check (`:464`) and written into the summary (`:485`). It verifies nothing.

**The bug:** `rate_test_proxy` is min-max rescaled into `[0,1]` against the train rate range (`minmax_train_apply`, `:97-100`), but `theta_v_proxy` is the 0.78-quantile of the **raw, unscaled** `rate_train_proxy` (`:339`). `causal_rate` (`:91-95`) is a 5-point rolling mean of `np.diff(q)`, so its values are O(10⁻²) while `rate_test_proxy` after rescaling is O(10⁻¹–10⁰). The condition `rate_test_proxy >= theta_v_proxy` is therefore true for nearly every test sample, and B2 degenerates to "early if `q_proxy <= θ_E`, else late".

This is corroborated by the paper's own numbers. `main.tex:757`, B2 row: `M→E = 0.4806`, `M→L = 0.5194`, which sum to exactly 1.0000 — **not a single true-middle sample is predicted middle** — with `M-F1 = 0.0000` and `L-F1 = 0.0000`. That degenerate signature is exactly what the unit mismatch predicts. The correct fix is either to compare in the same space (`theta_v_proxy = np.quantile(minmax_train_apply(rate_train_proxy, rate_train_proxy), RATE_LATE_Q)`) or to not rescale the test rate at all.

Consequence: the paper's headline "relative-stage rule collapses" comparison rests on a broken implementation, not on a property of the rule.

### 4.4 Leakage audit

**PHM main pipeline — clean on the points checked.** Ordering in `main…:1088-1130`:

- Features built per split independently (`build_online_features_by_split`, `:1092-1096`).
- Median imputation fitted on train only (`fill_by_train_median`, `:1103-1104`, `:1108-1109`).
- Feature selection on train only (`select_features_train_only(feat_train)`, `:1107`).
- GMM fitted on train only (`fit_train_gmm(feat_train)`, `:1111`), test only transformed (`:1114`).
- Scaler fitted on train only (`StandardScaler().fit(feat_train[selected].values)`, `:1117`).
- Early stopping uses `val_pack` only (`train_model`, `:781-806`).
- **`probability_param_search` is tuned on validation only.** `main…:1136` passes `pred_val_raw`, which comes from `predict_model(model, va_pack)` (`:1134`). The search body (`:907-930`) never touches `pred_test`. Confirmed: the fusion parameters are selected on `final_internal_val` (C1/C4 hold-out blocks), not on C6.

**Real issues found:**

1. **Train-side online features are computed on a gapped sequence.** `split_grouped_lifecycle:465-483` removes an *interior* block of each stage for validation (`start = max(0, (len(gs) - n) // 2)`, `:476`). Online features are then built per split (`build_online_features_for_subset:507`), so the expanding mean/std/rank for the training rows are computed over a sequence with holes, while the val rows restart their own expanding history from scratch (`x.expanding(min_periods=3)`, `:530-531`; `hist` list restarted per subset, `:534-539`). Train and validation features are therefore drawn from different generative processes; the test split (whole C6, contiguous) is the only one built the way deployment would build it. This is not test leakage, but it is a train/serve skew that biases validation-driven model and fusion-parameter selection.
2. **Label definition uses non-causal, whole-sequence statistics on the test condition.** `define_condition_relative_stages:436-441`: `rolling(window=7, center=True)` and `q = (vb_smooth - vb_smooth.min())/(vb_smooth.max() - vb_smooth.min())`, with per-condition quantiles `q.quantile(Q_EARLY)` etc. computed over the *entire* C6 sequence. This is a label-construction choice (targets, not inputs) and is arguably legitimate for offline evaluation, but it means `q_true` on C6 is only definable once the tool's full life is known — worth stating explicitly in the paper.
3. **`7.6消融实验.py` hardcodes the winning fusion parameters and the expected result.** `FIXED_MAIN_PROB_PARAMS` (`:118-126`) and `EXPECTED_A6 = {"Acc": 0.986842105, "Macro-F1": 0.987102093, …}` (`:128-135`), with a self-check at `:640-641` and a message at `:652-655` warning that the retrained checkpoint may not reproduce it. The ablation retrains the backbone (`:579`) but reuses the previous run's parameters, so A1–A6 are not from the same trained model as the reported B12 row unless the retrain reproduces bit-for-bit.
4. **`7.7跨工况实验.py` reuses one task's parameters for all nine tasks.** `FIXED_PROB_PARAMS` (`:102-110`) is the C1+C4→C6 optimum; `run_one_task:434` applies it unchanged to D2/D3 and all six single-source tasks. There is no per-task validation search. Not test leakage, but the paper's claim that fusion parameters are tuned on internal validation does not hold for eight of the nine tasks.
5. **NASA: validation cases are selected to resemble the test cases.** `9.1nasa数据实验.py:1591-1616` (`choose_validation_cases_similarity`) computes `test_desc = group_descriptor(df, test_cases)` (`:1599`) and then picks the 2–3 training cases whose descriptor minimises `descriptor_distance(combo_desc, test_desc, scale)` (`:1605-1610`). The descriptor (`case_descriptor:1560-1576`) includes `VB_min`, `VB_max`, `q_min`, `q_max` and the **early/middle/late label ratios** of the test cases. That validation set then drives (a) model-config selection (`train_select_method:1287-1304`, `train_select_b11:1885-1903`), (b) early stopping (`train_model:1189-1197`), and (c) the B12 fusion search (`search_b12_params_stageaware:1949-1979`). This is a direct information path from test-set statistics into every selection decision. **This is leakage.**
6. **NASA: the ground-truth stage labels themselves are searched.** `search_stage_strategy:1714-1745` sweeps `theta_E ∈ {0.25,0.30,0.35,0.40}`, `theta_L ∈ {0.60,…,0.80}`, `theta_v ∈ {0.75,…,0.90}` plus a VB-quantile strategy, calls `apply_stage_labels(df_base, …)` — which **relabels the entire dataframe including the test cases** (`:1544-1557`) — and keeps the variant maximising `label_selection_score` (`:1736`). The selection score is computed on the (test-similar) validation cases (`nearest_centroid_val_score`, `:1670-1711`). So the NASA evaluation targets are chosen to be maximally separable, per task, and differ from the paper's stated fixed `Q_E=0.30, Q_L=0.72, Q_ν=0.78` (`9.1nasa数据实验.py:69` defines `QE, QL, QV = 0.30, 0.72, 0.78`, used only for the *initial* labelling at `:247-249`, which `search_stage_strategy` then overwrites).
7. **NASA best-case run: splits are selected by test-set performance.** See §4.6.

### 4.5 Task-ID naming mismatches between code and paper

**Code — `7.7跨工况实验.py:112-124` (exact):**
```python
DUAL_TASKS = [
    ("D1_C1C4_to_C6", ["C1", "C4"], "C6"),
    ("D2_C1C6_to_C4", ["C1", "C6"], "C4"),
    ("D3_C4C6_to_C1", ["C4", "C6"], "C1"),
]
SINGLE_TASKS = [
    ("S1_C1_to_C4", ["C1"], "C4"),
    ("S2_C1_to_C6", ["C1"], "C6"),
    ("S3_C4_to_C1", ["C4"], "C1"),
    ("S4_C4_to_C6", ["C4"], "C6"),
    ("S5_C6_to_C1", ["C6"], "C1"),
    ("S6_C6_to_C4", ["C6"], "C4"),
]
```
Nine tasks: 3 dual-source + 6 single-source.

**Paper — `main.tex:630-633`:**
```
PHM2010 & D1 & C1 + C4 & C6
        & D2 & C4 + C6 & C1
        & S1 & C4      & C1
        & S2 & C6      & C1
```
Four tasks.

Mapping:

| Paper ID | Paper train→test | Code ID producing it |
|---|---|---|
| D1 | C1+C4 → C6 | `D1_C1C4_to_C6` ✔ same name |
| D2 | C4+C6 → C1 | **`D3_C4C6_to_C1`** — renamed |
| S1 | C4 → C1 | **`S3_C4_to_C1`** — renamed |
| S2 | C6 → C1 | **`S5_C6_to_C1`** — renamed |

The code's `D2_C1C6_to_C4`, `S1_C1_to_C4`, `S2_C1_to_C6`, `S4_C4_to_C6`, `S6_C6_to_C4` are computed and written to `Table12_…csv` / `Table13_…csv` but do not appear in the paper. The paper reports 4 of 9 tasks and re-indexes them, and the four reported tasks are all tasks whose **test condition is C6 or C1** — the C4-as-test tasks (code D2, S1, S6) are all dropped. Whether that selection was performance-driven is **unverified** (the underlying result CSVs are not in the upload), but the renaming makes the correspondence non-obvious and the omission is not disclosed in `main.tex`.

**NASA — `FIXED_TASKS`, identical in both NASA scripts** (`9.1nasa数据实验.py:92-97`, `run_nasa_bestcase_candidate_split.py:110-115`):
```python
FIXED_TASKS = [
    {"Task": "N1", "Train_cases": [1,2,3,4,5,7,8,9,10,12,13,16], "Test_cases": [6,11,14,15]},
    {"Task": "N2", "Train_cases": [2,3,4,5,6,7,8,11,12,13,14,15], "Test_cases": [1,9,10,16]},
    {"Task": "N3", "Train_cases": [1,2,4,6,8,9,10,11,13,14,15,16], "Test_cases": [3,5,7,12]},
    {"Task": "N4", "Train_cases": [1,3,5,6,7,9,10,11,12,14,15,16], "Test_cases": [2,4,8,13]},
]
```

### 4.6 Which NASA script produced the paper's N1–N4? (evidence)

**Paper, `main.tex:634-637`:**

| Task | Train cases | Test cases |
|---|---|---|
| N1 | 1,2,4,5,6,7,8,9,10,11,13,14 | 3,12,15,16 |
| N2 | 1,3,4,5,6,7,11,12,13,14,15,16 | 2,8,9,10 |
| N3 | 1,2,4,6,7,8,9,11,13,14,15,16 | 3,5,10,12 |
| N4 | 1,2,3,4,5,6,7,9,10,11,14,15 | 8,12,13,16 |

**None of these match `FIXED_TASKS`** (code N1 test = {6,11,14,15}; paper N1 test = {3,12,15,16}, etc.). Furthermore the paper's four test sets **overlap and are not a partition**: case 3 appears in N1 and N3; case 12 in N1, N3 and N4; case 16 in N1 and N4; case 8 in N2 and N4; case 1 is **never** a test case. A `GroupKFold`-style partition (`make_cross_case_tasks:357-364`) cannot produce that, and neither can `FIXED_TASKS`.

Overlapping, non-partitioning 4-case test groups are exactly what `run_nasa_bestcase_candidate_split.py` produces:

- `generate_candidate_case_splits:1833-1911` builds ~220+ random test-case groups of size 3 or 4 (`:1839-1841`) and keeps `N_CANDIDATE_SPLITS = 20` (`:50`).
- The effective `main` (`:2812`, the **last** of five `main` definitions in the file) evaluates B9–B12 on **all 20 candidate splits** (`:2846-2870`).
- `b12_selection_table:2725-2760` then scores each split using **B12's test-set metrics** — `Macro-F1`, `Balanced-Acc`, `M-F1`, `M-Rec`, `E-Rec`/`L-Rec`, `Smooth`, plus explicit bonuses for B12 beating B11 and penalties for B12 losing to B11:
  ```python
  score = (0.25*out["Macro-F1"] + 0.20*out["Balanced-Acc"] + 0.20*out["M-F1"]
           + 0.10*out["M-Rec"] + 0.10*np.minimum(out["E-Rec"], out["L-Rec"])
           - 0.08*out["Smooth"]
           + 0.05*np.maximum(0.0, out["Macro-F1"] - out["Macro-F1_B11"])
           + 0.05*np.maximum(0.0, out["M-F1"] - out["M-F1_B11"]))
  ...
  score = score - np.where(out["M-F1"]      < out["M-F1_B11"],      0.20, 0.0)
  score = score - np.where(out["Macro-F1"]  < out["Macro-F1_B11"],  0.20, 0.0)
  score = score - np.where(out["Smooth"]    > out["Smooth_B11"],    0.15, 0.0)
  ```
- `select_bestcase_tasks:2762-2763` returns `selection_df.head(4)["Task"]` — the top four splits by that test-metric score, with **no** constraint preventing test-case overlap (`select_final_case_splits:2739-2760`, which does deduplicate test cases, is dead code — the effective `main` never calls it).
- The script's own summary text says so: `write_bestcase_summary:2806-2807` writes `"Important transparency note: Selected results are best-case candidate split results, not an unbiased average across all splits."`

**Conclusion:** the paper's N1–N4 are almost certainly the four best-case candidate splits produced by `run_nasa_bestcase_candidate_split.py` (renamed from `CAND###` to `N1`–`N4`), **not** the `FIXED_TASKS` splits and **not** the output of `9.1nasa数据实验.py`. The evidence is: (i) the case lists match neither script's `FIXED_TASKS`; (ii) the test groups overlap and omit case 1, which only random-candidate generation produces; (iii) only `run_nasa_bestcase_candidate_split.py` selects exactly 4 splits and only it can produce overlapping selections; (iv) the script explicitly labels its own output as best-case. The paper's `main.tex:634-637` presents these as if they were a standard cross-case protocol, and the "transparency note" the code writes is not reproduced in the manuscript.

Relationship of the two NASA files: a line-level diff shows `run_nasa_bestcase_candidate_split.py` is `9.1nasa数据实验.py` plus inserted blocks (largest insertion: `run_nasa…:1699-1963`, plus the whole tail `:2561-2952`). Both files are accretions of several script generations: `9.1nasa数据实验.py` defines `main` **three** times (`:920`, `:1444`, `:2095`) and `run_nasa_bestcase_candidate_split.py` **five** times (`:938`, `:1463`, `:2458`, `:2564`, `:2812`); only the last definition in each file is live. `train_model`, `compute_stage_metrics`, `q_prior`, `q_consistency`, `mean_std_table`, `set_runtime_config`, `add_metric_context`, `diagnostic_row`, `save_prediction_detail`, `plot_optional_summary` are each defined 2–3 times per file. Any reading of the earlier definitions is misleading — e.g. `PROB_PARAMS` (`9.1:99-107`) is referenced only at `:861`, inside the **dead** first `main`.

### 4.7 Fusion-parameter inconsistency between PHM and NASA

| | PHM main | PHM 7.4 / 7.6 / 7.7 | NASA (live path) |
|---|---|---|---|
| Search | full grid, 3·3·2·3·3·3·2 = 972 combos (`main…:135-141`, `probability_param_search:907-930`) | none — fixed dict | 200 random samples + local refinement around top-20 (`search_b12_params_stageaware:1949-1979`) |
| `eta` range | `[0.55, 0.65, 0.75]` | fixed `0.75` | `[0.85, 0.90, 0.92, 0.95, 0.98]` (`9.1:81-90`) — **disjoint from the PHM grid** |
| `fine_weight` | `[0.10, 0.20, 0.30]` | `0.30` | `[0.00, 0.03, 0.05, 0.10, 0.15]` |
| `mid_floor` | `[0.04, 0.08, 0.12]` | `0.12` | `[0.00, 0.005, 0.01]` |
| `order_blend` | `[0.25, 0.50]`, `β>0` enforced (`main…:884-885`, `:925-926`) | `0.25` | `[0.00, 0.03, 0.05, 0.08, 0.10]` — **β = 0 allowed**, i.e. the ordered filter can be switched off entirely |
| `prior_sigma` | not a parameter; fixed `sigma = 0.17` (`main…:827`) | — | searched over `[0.35, 0.45, 0.55]` (`9.1:89`) |
| Prior centres | `0.18 / 0.50 / 0.84` (`main…:828-830`) | — | `0.15 / 0.50 / 0.85` (`9.1:1085`) |
| Suppression gain κ | `18.0` (`main…:143-144`) | — | `12.0` (`9.1:1089-1090`) |
| Selection objective | `1.0(1−acc)+1.0(1−F1)+1.4(1−M-Rec)+0.8·M→L+0.7·M→E+0.4·ratio_penalty+0.15·q-RMSE` (`main…:919`) | — | `b12_stageaware_score` (`9.1:1907-1932`), an 11-term weighted score with six discrete penalty cliffs |
| Search space also narrowed by q quality | no | — | yes — `random_fusion_params:1934-1947` shrinks `eta`/`fine_weight`/`prior_sigma` depending on validation Spearman/Pearson/R² |

The paper presents a single fusion formulation with one set of symbols (`main.tex:405-430`). In practice PHM and NASA use different prior centres, different σ (fixed vs searched), a different κ, non-overlapping η ranges, different objectives, and NASA permits β = 0, which disables the ordered filter that A5/A6 are supposed to demonstrate.

Additional NASA/PHM method divergences (same paper, same claimed method):

- **No monotonic loss on NASA.** `9.1nasa数据实验.py:1179-1187`:
  ```python
  loss = ce_stage(stage_logits, ys)
  if multitask:
      loss = loss + 0.35 * ce_fine(fine_logits, yf) + 0.35 * huber(q_hat, yq)
  ```
  λ_s = 1.0, λ_g = λ_q = 0.35, **λ_m absent**. PHM uses λ_s=1.00, λ_g=0.25, λ_q=0.30, λ_m=0.03 (`main…:120-123`). `main.tex:400` and `:534` both state the loss includes `λ_m L_mono`.
- **No distribution-instability term in NASA feature selection.** `9.1nasa数据实验.py:322-336`:
  ```python
  score = 1.10 * mi_s + 0.35 * mi_q + 0.55 * np.asarray(corr)
  ```
  No `− 0.35·DI` (compare `main…:587`) and no redundancy pruning (compare `main…:596-606`). `main.tex:261-273` defines the score *with* DI as a core contribution.
- **Different architecture family.** NASA `TCNBlock` uses `BatchNorm1d` and **no causal chomp** (`9.1:459-475`), i.e. the convolutions are non-causal (symmetric padding, no `Chomp1d`), whereas PHM's `TemporalBlock` chomps to enforce causality (`main…:687-701`). A non-causal TCN sees future samples inside the window. The paper claims causal, online-capable inference throughout.
- `9.1nasa数据实验.py:2170` contains a typo in a print format key, `best_macro['Macr o-F1_mean']`, which will raise `KeyError` at the end of the (dead-path) stage-aware `main`. Live path in `run_nasa_bestcase_candidate_split.py` does not hit it.

### 4.8 Other genuinely suspicious findings

**Seeds and single-seed runs.**
- `RUN_SEEDS = [2026]` — a single seed (`9.1nasa数据实验.py:46`, `run_nasa…:46`), with the multi-seed line commented out immediately below (`:47-48`). Every NASA number in the paper is a **single-seed** result. `mean_std_table:2082-2093` computes `std(ddof=1)` across the 4 tasks, not across seeds — so the `±` values in `main.tex:854` are **task-to-task spread, not run-to-run variance**.
- PHM: `RANDOM_SEED = 42`, one run, no repetition anywhere (`main…:91`; `train_model:782` re-seeds before each model). `7.4对比实验.py` trains B8/B9/B10/B11 once each with no seed re-initialisation between them (`train_stage_model:232` never calls `set_seed`), so B8/B9/B10 results depend on RNG consumption order.
- `set_seed:161-167` does not set `torch.backends.cudnn.deterministic = True` / `benchmark = False`, and `DEVICE` is CUDA when available (`:92`). Results are not bit-reproducible on GPU. `7.6消融实验.py:128-135` hardcodes the expected A6 metrics to 9 decimal places and warns at `:655` that a retrain may not match — an implicit admission that the pipeline is not reproducible.
- NASA `set_seed` (`9.1:135-139`) calls `torch.cuda.manual_seed_all` unconditionally, which is harmless but is called with **different derived seeds per config** (`cfg_seed = seed + i*100 + {"B9":9,"B10":10,"B11":11}[method]`, `9.1:1289`), so B9/B10/B11 are not trained under a common seed.

**Evaluation-metric definitions.**
- `ratio_penalty` (`main…:284`) `= Σ|true_class_ratio − pred_class_ratio|`. It is **not** reported in the paper but carries weight 0.4 in the fusion-parameter objective (`main…:919`) — a fifth of the total. It rewards fusion parameters that reproduce the validation class prior, which directly manufactures the "middle stage is not compressed" behaviour the paper attributes to the method. It is absent from the training-time model-selection score (`main…:795`), so the model and the fusion layer are selected under different objectives.
- `Rev` and `Jump` are **counts**, summed over conditions (`stage_consistency_metrics:311-320`, `main…:317-319`), while `Smooth` is a mean. So `Rev`/`Jump` scale with sequence length and are not comparable across tasks with different test sizes — the paper tabulates them side by side across D1/D2/S1/S2 (`main.tex:818-830`) and across N1–N4 (`:897-909`), which have different lengths.
- `Smooth` is defined on the **method's own** probability vector (`prob_prefix = "raw" if method=="raw" else method`, `main…:294-296`). A6 by construction blends toward the ordered filter output, so its `Smooth` is mechanically lower than A1's. The metric is not independent of the ablation axis it is used to justify.
- `M→E` and `M→L` are row-normalised confusion entries (`cm_norm[1,0]`, `cm_norm[1,2]`, `main…:281-282`), so `M-Rec + M→E + M→L = 1`. Reporting all three, as the paper does, is redundant, and the B2 row `0.4806 + 0.5194 = 1.0` (`main.tex:757`) is the tell for the §4.3 bug.
- NASA re-implements macro-F1, per-class F1 and recall by hand (`9.1:704-731`) instead of using sklearn as PHM does (`main…:257-263`). The hand-rolled version defines `f1 = 0` when a class is absent from both `y_true` and `y_pred`, whereas sklearn with `zero_division=0` does the same — behaviour matches, but the duplication is an avoidable divergence risk.
- `Predicted_class_count` (`9.1:1128`) is used only as a penalty trigger; a NASA run that predicts fewer than 3 classes is penalised in *selection*, but the metric is still reported as a normal column.

**Class weighting.**
- `class_weights:732-736` normalises by the **mean** weight (`w / w.mean()`), not by the sum. Combined with `LAMBDA_STAGE = 1.0`, this makes the effective stage-loss magnitude depend on class imbalance. Fine-state weights use the same function with `n = 5` (`main…:783`), so the fine head's effective λ drifts with the GMM's component balance rather than staying at the declared `LAMBDA_FINE = 0.25`.
- The **validation** loss is computed with **training** class weights (`train_model:784` passes `sw, fw` into both `run_epoch` calls, `:791-792`). The reported `val_loss` is therefore not a clean held-out likelihood. It does not affect early stopping (which uses `score`, `:795`), only the logged curve and Fig04.

**Dead / misleading code.**
- `B12_PARAMS` (`7.4对比实验.py:60-68`) is defined and never used; the identical dict is re-typed inline at `:401-404`. Editing one will silently not affect the run.
- `build_online_features_by_split:545-546` contains an `if False else` construct.
- `run_nasa_bestcase_candidate_split.py:55-56` creates a stale `nasa_dcpsr_results_stageaware_opt` directory before overwriting `OUT_DIR` at `:61`.
- `select_final_case_splits` (`run_nasa…:2739`) — the overlap-avoiding selector — is never called.
- `9.1nasa数据实验.py` `make_cross_case_tasks` (`:357`) and `make_cross_case_tasks_optimized` (`:345` in the second block) are unreachable from the live `main`.

**Fabricated-figure risk.**
- `8.1.2共享表征图.py:190-241` (`load_or_make_proxy`): if `repr_raw_features.csv` / `repr_hidden_hct.csv` are missing, the script silently synthesises both tables from `FINAL_comparison_predictions.csv`, setting `q_true = q_hat = np.linspace(0,1,n)` (`:207`) and `run_norm = np.linspace(0,1,len(raw))` (`:228`, `:236`), and — critically — assigns the **same B11 probability columns to both the "raw feature" row and the "shared latent" row** (`:218-220`, `:232-234`). The resulting `Fig5_repr_main_*` figures would show a "raw feature space vs shared representation space" contrast that is an artefact of identical data. The proxy flag is recorded only in `README_representation_space.md` (`:425`); the figures themselves carry no watermark (no use of `proxy_used` in any plotting function — only `:447` and `:460`). Whether the paper's Fig5 came from the real or proxy path is **unverified**.

**Other.**
- `MIN_STAGE_VAL_LEN = 8` with `n = max(MIN_STAGE_VAL_LEN, round(len(gs)*0.20))` (`main…:107`, `:474`): for a stage with fewer than 40 samples the validation block exceeds 20 %; for a stage with ≤10 samples, `n = min(n, max(len(gs)-2, 1))` leaves only 2 training samples.
- `is_meta_or_label_col:224-231` blacklists the **substring** `"vb"`, so any legitimate sensor feature whose name contains `vb` is dropped silently. For a new dataset this can silently remove real features (or, worse, fail to remove a label column with an unexpected name).
- `causal_ordered_filter:850-860` mutates the closure variable `alpha` across the loop and never re-initialises it between calls — it is re-created at `:852` on each call, so this is safe, but the transition matrix and initial state are magic numbers with no provenance (`:851-853`); the paper (`main.tex`) does not report the numeric values of A.
- `apply_probability_inference:884-885` raises if `order_blend <= 0`, but `causal_ordered_filter` is applied to `mix`, so the A5 "Ordered" ablation is *ordered-filtered mix*, not ordered-filtered raw. The paper's A1–A6 ladder (raw / raw+fine / raw+prior / mix / ordered / final) reads as if A5 isolates the filter; in code A5 = filter(A4) and A6 = (1−β)A4 + βA5. A5 and A6 therefore cannot separate the filter's contribution from the mix's.
- `7.9.1阶段偏差实验.py` depends on a `MICFS-TCN` VB-regression experiment (`:263-283`) that is **not among the audited scripts**; its `AnalysisStop` path (`:295-305`) exits with status 1 if that artefact is absent. The provenance of any numbers this script produced is **unverified**.

---

## 5. Target architecture: Dataset Adapter → Shared DC-PSR Pipeline → Experiment Runner

### 5.1 Module boundaries

```
dcpsr/
  adapters/
    base.py          # DatasetAdapter protocol (below)
    phm2010.py       # C1/C4/C6 from run_level_features_all.csv
    nasa_milling.py  # 16 cases from mill.mat
    mendeley_milling.py  # 9 tools / 3 machines from 6418 HDF5 files
  core/
    labels.py        # smooth VB -> q -> nu_norm -> E/M/L; per-sequence quantiles
    features.py      # historical z-score / first difference / online rank (causal, per sequence)
    selection.py     # MI(x,s) + MI(x,q) + |rho| - DI, generalised DI, redundancy pruning
    fine_states.py   # GMM on [q, nu_norm], q-ordered relabelling
    windows.py       # contiguity-checked sliding windows + Dataset/DataLoader
    model.py         # TCN(causal)-GRU multi-task, 3 heads
    losses.py        # stage CE, fine CE, SmoothL1, sequence-aware monotonic
    train.py         # train/early-stop, deterministic seeding
    inference.py     # prior, dual-side inhibition, fine->stage, mix, ordered filter, final
    tuning.py        # fusion-parameter search (validation only, one search policy)
    metrics.py       # single definition of Acc/F1/M->E/M->L/Rev/Jump/Smooth/q-*
  baselines/
    rules.py         # B1 (declared oracle), B2 (q-proxy rule)
    classic.py       # B3-B7
    deep.py          # B8-B10 single-task, B11 multi-task
  runners/
    run_main.py      # one adapter, one task -> B11/B12 + A1-A6
    run_compare.py   # B1-B12
    run_ablation.py  # A1-A6 from one backbone
    run_cross.py     # all tasks from adapter.tasks()
  config/
    <dataset>.yaml   # paths, hyperparameters, task definitions, seeds
```

Hard rules to enforce in the refactor:

- **No module-level I/O.** No `mkdir` at import, no CSV writes inside `core/*` functions. All persistence happens in `runners/`.
- **No magic-string control flow.** `shuffle` becomes an explicit argument of `make_pack`, never derived from a split name (fixes §4.1's second half).
- **One metric module.** Delete the hand-rolled NASA metric functions.
- **One fusion-inference implementation** with prior centres, σ, κ and the transition matrix as explicit, config-declared parameters — so PHM/NASA/Mendeley differences become visible in the config diff rather than hidden in two code paths.
- **One tuning policy**, applied per task, on validation only.

### 5.2 Dataset adapter interface

```python
@dataclass(frozen=True)
class SequenceSpec:
    sequence_id: str          # unique per independent degradation trajectory
    domain_id: str            # group used for cross-domain splitting (condition / machine / tool set)
    ordering_key: str         # monotone within-sequence index column name
    n_samples: int

class DatasetAdapter(Protocol):
    name: str

    def load(self) -> pd.DataFrame:
        """Return a long table, one row per run/cut, with at minimum:
             sequence_id : str   independent degradation trajectory
             order        : int  monotone within sequence, gap-free after load
             vb           : float ground-truth wear (adapter unit; documented)
             domain_id    : str  cross-domain grouping key
           plus signal/feature columns."""

    def feature_columns(self, df) -> list[str]:
        """EXPLICIT list of usable raw sensor/feature columns.
           Must NOT be inferred from a name blacklist."""

    def label_columns(self) -> list[str]:
        """Columns that are ground truth / progress and must never be inputs."""

    def vb_unit(self) -> str: ...          # e.g. "mm", "um", "1e-3 mm"
    def vb_is_monotone(self) -> bool: ...  # affects smoothing/interp policy

    def tasks(self) -> list[Task]:
        """Task(name, train_domains, val_policy, test_domains).
           Test domains must be disjoint from train and val."""

    def label_params(self) -> LabelParams:
        """Q_E, Q_L, Q_nu, smoothing windows. Fixed per dataset,
           never searched per task."""
```

`Task` must be a value object (`name`, `train_domains`, `test_domains`, `val_domains_or_policy`) so that task lists live in config, are printed into every output CSV, and cannot silently diverge from the paper's task table (fixes §4.5).

The validation policy must be declared as data (`"holdout_interior_block"`, `"holdout_whole_sequences"`, `"holdout_tail"`) and must **never** be allowed to consult test-domain statistics (fixes §4.4.5).

### 5.3 What each core function needs to change

| Current | Change |
|---|---|
| `load_feature_table:419` | replaced by `adapter.load()`; `infer_vb_column`/`normalize_condition_name` deleted |
| `define_condition_relative_stages:432` | takes `(df, LabelParams)`, groups by `sequence_id`, no CSV write, no `Q_EARLY` global |
| `split_grouped_lifecycle:465` | replaced by `Task` + val policy |
| `get_raw_numeric_sensor_cols:496` + `is_meta_or_label_col:206` | replaced by `adapter.feature_columns()` / `adapter.label_columns()` |
| `build_online_features_for_subset:507` | groups by `sequence_id`, ordered by `order`; **must be computed on the full sequence, then masked into splits** (fixes §4.4.1) |
| `select_features_train_only:566` | DI generalised to N training domains (max or mean pairwise); no CSV write |
| `make_pack:673` | explicit `shuffle: bool` |
| `monotonic_q_loss:738` | takes `(q_hat, sequence_id, order)` and penalises decreases only between temporally adjacent pairs *within the same sequence*; alternatively use ordered batch sampling (fixes §4.1) |
| `TemporalBlock:687` vs NASA `TCNBlock` | keep the causal (`Chomp1d`) version only |
| `qhat_prior:825` | centres, σ, κ become parameters |
| `causal_ordered_filter:850` | transition matrix and initial α become parameters, reported in the paper |
| `probability_param_search:907` | single implementation; grid/random policy in config; objective declared and reported (including whether `ratio_penalty` is in it) |

---

## 6. Integration plan for the Mendeley dataset

### 6.1 What the data actually looks like (verified from `filelist.csv`)

- 6418 rows, columns `filename, machine, tool, run, cumulated_tool_contact_time, wear`.
- 3 machines × 3 tools = 9 tools; tool IDs are **globally unique** (M1→T1–T3, M2→T4–T6, M3→T7–T9), so `tool` alone identifies a sequence.
- Runs per tool: T1 609, T2 609, T3 638, T4 928, T5 928, T6 928, T7 609, T8 560, T9 609.
- `run` is 1-based, contiguous, and matches the row count for every tool (T8 starts at run 1 but its `cumulated_tool_contact_time` starts at 177 — verify whether early runs were dropped).
- `wear` is an **integer**, range 3–160, and is **non-decreasing within every tool** (verified for all 9 tools). Distinct values per tool: 104–142. Units are almost certainly µm (VB ≈ 0.003–0.160 mm). **Confirm against the dataset documentation before publishing any absolute VB number.**
- Filenames encode everything: `M1T1R1C11VB3.h5` = machine 1, tool 1, run 1, cumulated contact time 11, VB 3.

This is a much better fit for DC-PSR than NASA: 9 long, dense, monotone sequences with per-run wear labels, and a natural two-level domain structure (machine and tool).

### 6.2 Adapter (`dcpsr/adapters/mendeley_milling.py`)

```
sequence_id   = f"M{machine}T{tool}"       # 9 independent degradation sequences
order         = run                        # contiguous, 1..N per tool
vb            = wear                       # integer, adapter documents unit (µm)
domain_id     = f"M{machine}"              # machine = the "condition" analogue
secondary_domain = sequence_id             # tool, for tool-level LOO tasks
```

`load()` must produce a **run-level feature table**, exactly analogous to PHM's `run_level_features_all.csv`. That means a one-time offline feature-extraction stage over the 6418 HDF5 files (this is the only genuinely new code):

```
mendeley_extract_features.py
  for each row in filelist.csv:
      open <root>/<filename>.h5
      for each channel:
          time-domain: mean, std, rms, skew, kurtosis, p2p, crest, shape, impulse, margin, abs-mean
          frequency:   band energies (fixed bands), spectral centroid/spread/entropy
          time-freq:   wavelet-packet band energies (same decomposition as PHM extraction)
      write one row keyed (machine, tool, run)
  -> mendeley_run_level_features_all.csv
```
Cache this CSV. Nothing downstream should ever touch the HDF5 files. Column naming must match the PHM convention (`<channel>__<stat>`) so `core/features.py` generates the same three derived families (`__rel`, `__slope`, `__online_rank`) without change.

The HDF5 internal layout (channel names, sampling rate, per-file duration) is **not visible in this upload** — only `filelist.csv` was staged. The extraction spec above must be validated against one real file before implementation.

### 6.3 Labels

Reuse `core/labels.py` unchanged, with `LabelParams(Q_E=0.30, Q_L=0.72, Q_nu=0.78, vb_smooth_window=?, rate_smooth_window=?)`:

- Per `sequence_id`: smooth `vb` → min-max → `q` → `Δq` → smooth → min-max → `nu_norm` → E/M/L by the three quantiles.
- **Smoothing windows must be rescaled.** PHM uses `window=7` on ~315-run sequences; NASA uses `window=3` on ~16-run cases. Mendeley has 560–928 runs per tool, so a window of 7 is far too short relative to sequence length. Set the window as a fraction of sequence length (e.g. `max(5, round(0.02*n))` → ~12–19) and record the chosen value in the config. Do **not** search it per task (that is the §4.4.6 mistake).
- `wear` is integer-quantised, so `Δq` will be a sparse staircase. Smooth `vb` **before** differencing (the pipeline already does) and verify `nu_norm` is not degenerate; if it is, consider fitting a monotone spline to `vb` per tool before computing `q`. Report whichever is used.
- Because `wear` is already monotone per tool, the `q` construction is well-posed without the interpolation hack NASA needs (`9.1:238-240`).

### 6.4 Tasks

Define these in `config/mendeley.yaml`, fixed before any model is trained, and report **all** of them:

- **Cross-machine (primary, the closest analogue to PHM cross-condition):**
  - `MD1`: train M1+M2 → test M3
  - `MD2`: train M1+M3 → test M2
  - `MD3`: train M2+M3 → test M1
  - Single-source: `MS1` M1→M2, `MS2` M1→M3, `MS3` M2→M1, `MS4` M2→M3, `MS5` M3→M1, `MS6` M3→M2
- **Leave-one-tool-out (secondary, 9 tasks):** train on 8 tools, test on the held-out tool.

Validation: hold out **whole tools** from the training machines (never a tool from the test machine), chosen by a fixed deterministic rule (e.g. lowest tool index per training machine) — **not** by similarity to the test set. This is the single most important change relative to the NASA code (§4.4.5).

With 9 sequences there is enough data to run **≥5 seeds per task** and report mean ± std across seeds, not across tasks. Do this; it removes the §4.8 single-seed criticism for the new dataset and gives a defensible variance estimate.

### 6.5 Reuse map — what is reused, what is new

| Component | Status for Mendeley |
|---|---|
| Signal → run-level feature extraction | **New** (HDF5 reader + feature bank). The only substantial new code. |
| `core/labels.py` | Reused; only `LabelParams` (smoothing window) differs. |
| `core/features.py` (rel / slope / online rank) | Reused unchanged once `sequence_id`/`order` are generic. |
| `core/selection.py` | Reused, **after** generalising DI to N domains (§2c-1). With 6 training tools across 2 machines, DI should be the max pairwise machine-level instability. |
| `core/fine_states.py` (GMM on `[q, nu_norm]`, 5 components) | Reused unchanged. |
| `core/windows.py` | Reused; contiguity check on `order` works as-is. Window length `L` should scale up (PHM `L=12` on ~315 runs; Mendeley has 560–928, so `L ∈ {16, 24, 32}` is the right sweep, tuned on validation). |
| `core/model.py`, `train.py` | Reused; keep the **causal** TCN block. |
| `core/losses.py` | Reused **after** the `monotonic_q_loss` fix (§4.1). With 560–928-run sequences, a sequence-aware monotonic term is finally meaningful. |
| `core/inference.py` | Reused unchanged. |
| `core/tuning.py` | Reused; use the PHM grid policy, on validation only, per task. Do not introduce a third search policy. |
| `core/metrics.py` | Reused; normalise `Rev`/`Jump` by sequence length before cross-task comparison (§4.8). |
| `baselines/*` | Reused. **Fix B2's threshold-space bug before running it** (§4.3), and label B1 as an oracle in the table (§4.2). |
| `runners/*` | Reused; only the adapter name and config file change. |

### 6.6 Ordered execution plan

1. Extract the shared pipeline out of `main_experiment_3_fgds_psi_optimized.py` into `dcpsr/core` with **no behavioural change**, and reproduce the PHM main-experiment numbers exactly. This is the regression gate.
2. Write `adapters/phm2010.py` and re-verify. Then `adapters/nasa_milling.py` on `FIXED_TASKS` — note that this will **not** reproduce the paper's N1–N4 (§4.6); that discrepancy must be resolved with the author before anything else.
3. Apply the four correctness fixes: sequence-aware `monotonic_q_loss`; explicit `shuffle`; B2 threshold space; full-sequence online-feature computation with post-hoc split masking.
4. Build `mendeley_extract_features.py`, validate on one tool, then run all 6418 files once and cache.
5. Write `adapters/mendeley_milling.py` + `config/mendeley.yaml` with the fixed task list from §6.4.
6. Run `run_cross.py --adapter mendeley --seeds 5` and report **all** tasks and **all** seeds. Do not add a best-case selection step.
