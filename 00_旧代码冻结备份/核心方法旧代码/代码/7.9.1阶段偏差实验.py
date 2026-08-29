from __future__ import annotations

import math
import os
import sys
from pathlib import Path


def _activate_bundled_runtime() -> None:
    bundled = Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "python"
    current = Path(sys.executable).resolve()
    if bundled.exists() and bundled in current.parents and str(bundled) not in sys.path:
        sys.path.insert(0, str(bundled))


_activate_bundled_runtime()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_ROOT = Path(r"D:\桌面\博士开题\公开数据\1PHM")
OUTPUT_ROOT = Path(r"D:\桌面\博士开题\2专利\代码\小论文\12_stage_bias_analysis")
RESULT_DIR = OUTPUT_ROOT / "1_results"
PRED_DIR = OUTPUT_ROOT / "2_predictions"
FIG_DIR = OUTPUT_ROOT / "3_figures"
LOG_DIR = OUTPUT_ROOT / "4_logs"
RUN_FEATURE_PATH = DATA_ROOT / "PHM实验" / "1run_run_level_features" / "02_features" / "run_level_features_all.csv"

SEARCH_ROOTS = [
    DATA_ROOT,
    Path(r"D:\桌面\博士开题\2专利\代码\小论文"),
]

PRED_PATTERNS = [
    "FINAL_best_by_test_predictions.csv",
    "FINAL_test_C6_predictions.csv",
    "test_C6_predictions.csv",
    "C6_predictions.csv",
]

PROB_PATTERNS = [
    "ablation_probabilities_test_C6.csv",
    "Data_5_4_A6_probability_wear_trajectory.csv",
    "test_C6_predictions.csv",
]

STAGE_ORDER = ["early", "middle", "late"]
METHOD_COLUMNS = {
    "Baseline": "VB_pred_base",
    "Hard-stage correction": "VB_pred_hardcorr",
    "Probability-stage correction": "VB_pred_probcorr",
}


class AnalysisStop(RuntimeError):
    pass


def ensure_dirs() -> None:
    for folder in [RESULT_DIR, PRED_DIR, FIG_DIR, LOG_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    for encoding in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def find_files(patterns: list[str]) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            name_lower = path.name.lower()
            if any(pattern.lower() in name_lower for pattern in patterns):
                resolved = path.resolve()
                if resolved not in seen:
                    found.append(path)
                    seen.add(resolved)
    return found


def choose_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def normalize_prediction(df: pd.DataFrame) -> pd.DataFrame | None:
    cols = df.columns.tolist()
    run_col = choose_column(cols, ["run_id", "cut_index", "run_id_end", "index"])
    true_col = choose_column(cols, ["VB_true", "y_true", "VB", "target", "true"])
    pred_col = choose_column(cols, ["VB_pred", "y_pred", "pred", "prediction", "VB_hat"])
    if not (run_col and true_col and pred_col):
        return None
    out = pd.DataFrame(
        {
            "run_id": to_numeric(df[run_col]),
            "VB_true": to_numeric(df[true_col]),
            "VB_pred_base": to_numeric(df[pred_col]),
        }
    )
    if "condition" in df.columns:
        out["condition"] = df["condition"].astype(str)
    out = out.dropna(subset=["run_id", "VB_true", "VB_pred_base"]).copy()
    out["run_id"] = out["run_id"].round().astype(int)
    return out


def prediction_priority(path: Path, df_norm: pd.DataFrame) -> tuple[int, int, int, float]:
    text = str(path).lower()
    score = 0
    if "3run_c1c4_to_c6_tcn" in text:
        score += 80
    if "00_final_results" in text:
        score += 30
    if "final_best_by_valrmse" in text:
        score += 25
    if "micfs" in text or "tcn" in text:
        score += 10
    if "废" in str(path):
        score -= 80
    if "12_stage_bias_analysis" in text:
        score -= 100
    n_c6 = int((df_norm.get("condition", "").astype(str).str.upper() == "C6").sum()) if "condition" in df_norm else len(df_norm)
    return (score, n_c6, len(df_norm), path.stat().st_mtime)


def load_c6_baseline() -> tuple[pd.DataFrame, Path]:
    candidates = []
    for path in find_files(PRED_PATTERNS):
        try:
            df = read_csv(path)
            norm = normalize_prediction(df)
        except Exception:
            continue
        if norm is None or norm.empty:
            continue
        if "condition" in norm.columns:
            norm = norm[norm["condition"].astype(str).str.upper() == "C6"].copy()
        if norm.empty:
            continue
        candidates.append((prediction_priority(path, norm), path, norm))
    if not candidates:
        raise AnalysisStop("No usable MICFS-TCN prediction file was found for C6.")
    candidates.sort(key=lambda item: item[0], reverse=True)
    chosen_path = candidates[0][1]
    chosen_df = candidates[0][2][["run_id", "VB_true", "VB_pred_base"]].sort_values("run_id").reset_index(drop=True)
    return chosen_df, chosen_path


def normalize_probabilities(df: pd.DataFrame) -> pd.DataFrame | None:
    cols = df.columns.tolist()
    run_col = choose_column(cols, ["run_id", "cut_index", "run_id_end", "index"])
    prob_sets = [
        ["p_final_E", "p_final_M", "p_final_L"],
        ["final_prob_E", "final_prob_M", "final_prob_L"],
        ["final_prob_early", "final_prob_middle", "final_prob_late"],
        ["prob_early", "prob_middle", "prob_late"],
        ["p_E", "p_M", "p_L"],
    ]
    chosen = None
    for names in prob_sets:
        cols_found = [choose_column(cols, [name]) for name in names]
        if all(cols_found):
            chosen = cols_found
            break
    if not run_col or not chosen:
        return None
    out = pd.DataFrame(
        {
            "run_id": to_numeric(df[run_col]),
            "p_E": to_numeric(df[chosen[0]]),
            "p_M": to_numeric(df[chosen[1]]),
            "p_L": to_numeric(df[chosen[2]]),
        }
    )
    if "condition" in df.columns:
        out["condition"] = df["condition"].astype(str)
    out = out.dropna(subset=["run_id", "p_E", "p_M", "p_L"]).copy()
    out["run_id"] = out["run_id"].round().astype(int)
    sums = out[["p_E", "p_M", "p_L"]].sum(axis=1).replace(0, np.nan)
    out[["p_E", "p_M", "p_L"]] = out[["p_E", "p_M", "p_L"]].div(sums, axis=0)
    return out.dropna(subset=["p_E", "p_M", "p_L"])


def probability_priority(path: Path, df_norm: pd.DataFrame) -> tuple[int, int, int, float]:
    text = str(path).lower()
    score = 0
    if "6_ablation_experiment" in text:
        score += 80
    if "ablation_probabilities_test_c6" in text:
        score += 60
    if "data_5_4_a6_probability_wear_trajectory" in text:
        score += 40
    if "final" in "".join(df_norm.columns).lower():
        score += 5
    if "12_stage_bias_analysis" in text:
        score -= 100
    n_c6 = int((df_norm.get("condition", "").astype(str).str.upper() == "C6").sum()) if "condition" in df_norm else len(df_norm)
    return (score, n_c6, len(df_norm), path.stat().st_mtime)


def load_c6_probabilities() -> tuple[pd.DataFrame, Path]:
    candidates = []
    for path in find_files(PROB_PATTERNS):
        try:
            df = read_csv(path)
            norm = normalize_probabilities(df)
        except Exception:
            continue
        if norm is None or norm.empty:
            continue
        if "condition" in norm.columns:
            norm = norm[norm["condition"].astype(str).str.upper() == "C6"].copy()
        if norm.empty:
            continue
        candidates.append((probability_priority(path, norm), path, norm))
    if not candidates:
        raise AnalysisStop("No usable DC-PSR probability file was found for C6.")
    candidates.sort(key=lambda item: item[0], reverse=True)
    chosen_path = candidates[0][1]
    chosen_df = candidates[0][2][["run_id", "p_E", "p_M", "p_L"]].sort_values("run_id").reset_index(drop=True)
    return chosen_df, chosen_path


def construct_true_stage(df: pd.DataFrame, true_col: str = "VB_true") -> pd.DataFrame:
    out = df.copy().sort_values("run_id").reset_index(drop=True)
    smooth = out[true_col].rolling(window=7, center=True, min_periods=1).mean()
    q_true = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-12)
    rate = q_true.diff().fillna(0)
    rate_smooth = rate.rolling(window=5, center=True, min_periods=1).mean()
    rate_norm = (rate_smooth - rate_smooth.min()) / (rate_smooth.max() - rate_smooth.min() + 1e-12)
    q30 = q_true.quantile(0.30)
    q72 = q_true.quantile(0.72)
    r78 = rate_norm.quantile(0.78)
    stage = np.where(q_true <= q30, "early", np.where((q_true >= q72) | (rate_norm >= r78), "late", "middle"))
    out["VB_smooth"] = smooth
    out["q_true"] = q_true
    out["rate_norm"] = rate_norm
    out["true_stage"] = pd.Categorical(stage, categories=STAGE_ORDER, ordered=True).astype(str)
    return out


def infer_best_config_and_val_file(baseline_path: Path) -> Path | None:
    final_dir = baseline_path.parent
    exp_root = final_dir.parent
    ranking = final_dir / "FINAL_model_ranking_by_valRMSE.csv"
    all_pred = exp_root / "04_all_predictions"
    if not ranking.exists() or not all_pred.exists():
        return None
    try:
        rank_df = read_csv(ranking)
    except Exception:
        return None
    config_col = choose_column(rank_df.columns.tolist(), ["config_name"])
    if not config_col or rank_df.empty:
        return None
    best_config = str(rank_df.iloc[0][config_col])
    candidate = all_pred / f"{best_config}_val_predictions.csv"
    if candidate.exists():
        return candidate
    matches = list(all_pred.glob(f"{best_config}*val_predictions.csv"))
    return matches[0] if matches else None


def find_calibration_file(baseline_path: Path) -> tuple[Path | None, str]:
    best_val = infer_best_config_and_val_file(baseline_path)
    if best_val is not None:
        return best_val, "Validation predictions from the best MICFS-TCN configuration selected by validation RMSE."
    for path in baseline_path.parent.parent.rglob("*val_predictions.csv"):
        if path.exists():
            return path, "Validation predictions found near the selected MICFS-TCN result."
    return None, "No non-C6 training or validation prediction file was found."


def load_calibration(baseline_path: Path) -> tuple[pd.DataFrame, Path, str]:
    calib_path, source = find_calibration_file(baseline_path)
    if calib_path is None:
        raise AnalysisStop(
            "Correction coefficients were not estimated because no non-C6 training or validation predictions were found. "
            "C6 errors were deliberately not used for calibration."
        )
    raw = read_csv(calib_path)
    norm = normalize_prediction(raw)
    if norm is None or norm.empty:
        raise AnalysisStop(f"Calibration file is unusable: {calib_path}")
    if "condition" in norm.columns:
        norm = norm[norm["condition"].astype(str).str.upper() != "C6"].copy()
    if norm.empty:
        raise AnalysisStop(
            "Calibration candidates only contained C6. C6 errors were deliberately not used for calibration."
        )
    parts = []
    for _, group in norm.groupby(norm.get("condition", pd.Series(["calibration"] * len(norm))).astype(str), sort=False):
        parts.append(construct_true_stage(group[["run_id", "VB_true", "VB_pred_base"]].copy()))
    calib = pd.concat(parts, ignore_index=True)
    return calib, calib_path, source


def learn_coefficients(calib: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage in STAGE_ORDER:
        sub = calib[calib["true_stage"] == stage]
        correction_error = sub["VB_true"] - sub["VB_pred_base"]
        rows.append(
            {
                "stage": stage,
                "n_calibration": int(len(sub)),
                "b_stage": float(correction_error.mean()) if len(sub) else 0.0,
                "std_error_to_correct": float(correction_error.std(ddof=1)) if len(sub) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    residual = y_pred - y_true
    abs_error = residual.abs()
    denom = y_true.abs().replace(0, np.nan)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "MAE": float(abs_error.mean()),
        "RMSE": float(math.sqrt(np.mean(residual**2))),
        "R2": float(1 - ss_res / (ss_tot + 1e-12)),
        "Bias": float(residual.mean()),
        "MaxAE": float(abs_error.max()),
        "MedianAE": float(abs_error.median()),
        "MeanNAPE": float((abs_error / denom).mean() * 100),
        "StdResidual": float(residual.std(ddof=1)) if len(residual) > 1 else 0.0,
    }


def make_stage_summary(df: pd.DataFrame, method_col: str, include_method: str | None = None) -> pd.DataFrame:
    rows = []
    for stage in STAGE_ORDER:
        sub = df[df["true_stage"] == stage]
        if sub.empty:
            values = {"MAE": np.nan, "RMSE": np.nan, "Bias": np.nan, "MaxAE": np.nan, "MedianAE": np.nan, "StdResidual": np.nan}
        else:
            values = metrics(sub["VB_true"], sub[method_col])
        row = {
            "Stage": stage,
            "n": int(len(sub)),
            "MAE": values["MAE"],
            "RMSE": values["RMSE"],
            "Bias": values["Bias"],
            "MaxAE": values["MaxAE"],
            "MedianAE": values["MedianAE"],
            "StdResidual": values["StdResidual"],
        }
        if include_method:
            row = {"Method": include_method, **row}
        rows.append(row)
    return pd.DataFrame(rows)


def build_outputs(c6: pd.DataFrame, coeffs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    b_map = dict(zip(coeffs["stage"], coeffs["b_stage"]))
    stage_cols = ["p_E", "p_M", "p_L"]
    prob_argmax = c6[stage_cols].idxmax(axis=1).map({"p_E": "early", "p_M": "middle", "p_L": "late"})
    c6["stage_prob_argmax"] = prob_argmax
    c6["VB_pred_hardcorr"] = c6["VB_pred_base"] + c6["stage_prob_argmax"].map(b_map).astype(float)
    c6["VB_pred_probcorr"] = c6["VB_pred_base"] + c6["p_E"] * b_map["early"] + c6["p_M"] * b_map["middle"] + c6["p_L"] * b_map["late"]
    for suffix, col in [
        ("base", "VB_pred_base"),
        ("hardcorr", "VB_pred_hardcorr"),
        ("probcorr", "VB_pred_probcorr"),
    ]:
        c6[f"residual_{suffix}"] = c6[col] - c6["VB_true"]
        c6[f"abs_error_{suffix}"] = c6[f"residual_{suffix}"].abs()

    base_summary = make_stage_summary(c6, "VB_pred_base")
    overall_rows = []
    stage_rows = []
    for method, col in METHOD_COLUMNS.items():
        overall = metrics(c6["VB_true"], c6[col])
        overall_rows.append({"Method": method, **{k: overall[k] for k in ["MAE", "RMSE", "R2", "Bias", "MaxAE", "MedianAE", "MeanNAPE"]}})
        stage_rows.append(make_stage_summary(c6, col, include_method=method))
    overall_df = pd.DataFrame(overall_rows)
    stage_df = pd.concat(stage_rows, ignore_index=True)
    return c6, base_summary, overall_df, stage_df


def save_tables(
        final_df: pd.DataFrame,
        base_summary: pd.DataFrame,
        coeffs: pd.DataFrame,
        overall_df: pd.DataFrame,
        stage_df: pd.DataFrame,
) -> list[Path]:
    saved = []
    tables = {
        "stage_bias_summary_base.csv": base_summary,
        "correction_coefficients.csv": coeffs,
        "correction_overall_metrics.csv": overall_df,
        "correction_stagewise_metrics.csv": stage_df,
        "final_predictions_with_bias_correction.csv": final_df[
            [
                "run_id",
                "VB_true",
                "q_true",
                "true_stage",
                "p_E",
                "p_M",
                "p_L",
                "stage_prob_argmax",
                "VB_pred_base",
                "VB_pred_hardcorr",
                "VB_pred_probcorr",
                "residual_base",
                "residual_hardcorr",
                "residual_probcorr",
                "abs_error_base",
                "abs_error_hardcorr",
                "abs_error_probcorr",
            ]
        ],
    }
    for name, table in tables.items():
        path = RESULT_DIR / name
        table.to_csv(path, index=False, encoding="utf-8-sig")
        saved.append(path)
    pred_path = PRED_DIR / "final_predictions_with_bias_correction.csv"
    tables["final_predictions_with_bias_correction.csv"].to_csv(pred_path, index=False, encoding="utf-8-sig")
    saved.append(pred_path)

    excel_path = RESULT_DIR / "stage_bias_analysis_results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        base_summary.to_excel(writer, sheet_name="stage_bias_base", index=False)
        coeffs.to_excel(writer, sheet_name="correction_coefficients", index=False)
        overall_df.to_excel(writer, sheet_name="overall_metrics", index=False)
        stage_df.to_excel(writer, sheet_name="stagewise_metrics", index=False)
        tables["final_predictions_with_bias_correction.csv"].to_excel(writer, sheet_name="final_predictions", index=False)
    saved.append(excel_path)
    return saved


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "savefig.dpi": 900,
        }
    )


def shade_stages(ax: plt.Axes, df: pd.DataFrame) -> None:
    colors = {"early": "#d7ebff", "middle": "#e8f5df", "late": "#ffe4df"}
    x = df["run_id"].to_numpy()
    stages = df["true_stage"].tolist()
    if len(x) == 0:
        return
    start = 0
    for i in range(1, len(stages) + 1):
        if i == len(stages) or stages[i] != stages[start]:
            ax.axvspan(x[start], x[i - 1], color=colors.get(stages[start], "#eeeeee"), alpha=0.35, lw=0)
            start = i


def save_figures(df: pd.DataFrame, stage_df: pd.DataFrame) -> list[Path]:
    setup_plot_style()
    saved = []
    palette = {
        "Baseline": "#4c78a8",
        "Hard-stage correction": "#f58518",
        "Probability-stage correction": "#54a24b",
    }
    stage_colors = {"early": "#4c78a8", "middle": "#54a24b", "late": "#e45756"}

    pivot_bias = stage_df.pivot(index="Stage", columns="Method", values="Bias").loc[STAGE_ORDER]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(STAGE_ORDER))
    width = 0.24
    for idx, method in enumerate(METHOD_COLUMNS):
        ax.bar(x + (idx - 1) * width, pivot_bias[method], width=width, label=method, color=palette[method])
    ax.axhline(0, color="#222222", linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_ORDER)
    ax.set_ylabel("Bias")
    ax.set_xlabel("True stage")
    ax.legend(loc="best")
    fig.tight_layout()
    path = FIG_DIR / "Fig_stagewise_bias_bar.png"
    fig.savefig(path, dpi=900)
    plt.close(fig)
    saved.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), sharex=True)
    for ax, metric_name, title in zip(axes, ["MAE", "RMSE"], ["(a) MAE by stage", "(b) RMSE by stage"]):
        pivot = stage_df.pivot(index="Stage", columns="Method", values=metric_name).loc[STAGE_ORDER]
        for idx, method in enumerate(METHOD_COLUMNS):
            ax.bar(x + (idx - 1) * width, pivot[method], width=width, label=method, color=palette[method])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(STAGE_ORDER)
        ax.set_xlabel("True stage")
        ax.set_ylabel(metric_name)
    axes[1].legend(loc="best")
    fig.tight_layout()
    path = FIG_DIR / "Fig_stagewise_mae_rmse.png"
    fig.savefig(path, dpi=900)
    plt.close(fig)
    saved.append(path)

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.5), sharex=True)
    residual_map = [
        ("Baseline", "residual_base"),
        ("Hard-stage correction", "residual_hardcorr"),
        ("Probability-stage correction", "residual_probcorr"),
    ]
    for ax, (method, residual_col) in zip(axes, residual_map):
        for stage in STAGE_ORDER:
            sub = df[df["true_stage"] == stage]
            ax.scatter(sub["q_true"], sub[residual_col], s=12, color=stage_colors[stage], alpha=0.82, label=stage)
        ax.axhline(0, color="#222222", linestyle="--", linewidth=0.85)
        ax.set_ylabel("Residual")
        ax.set_title(method)
    axes[-1].set_xlabel("q_true")
    axes[0].legend(loc="best", ncol=3)
    fig.tight_layout()
    path = FIG_DIR / "Fig_residual_vs_q_stage.png"
    fig.savefig(path, dpi=900)
    plt.close(fig)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    shade_stages(ax, df)
    ax.plot(df["run_id"], df["VB_true"], color="#111111", linewidth=1.8, label="True VB")
    ax.plot(df["run_id"], df["VB_pred_base"], color=palette["Baseline"], linewidth=1.4, label="Baseline")
    ax.plot(df["run_id"], df["VB_pred_hardcorr"], color=palette["Hard-stage correction"], linewidth=1.4, label="Hard-stage correction")
    ax.plot(df["run_id"], df["VB_pred_probcorr"], color=palette["Probability-stage correction"], linewidth=1.4, label="Probability-stage correction")
    ax.set_xlabel("Run index")
    ax.set_ylabel("VB")
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    path = FIG_DIR / "Fig_wear_curve_prediction_comparison.png"
    fig.savefig(path, dpi=900)
    plt.close(fig)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    positions = []
    values = []
    colors = []
    labels = []
    center = np.arange(len(STAGE_ORDER))
    offsets = [-0.25, 0.0, 0.25]
    for stage_idx, stage in enumerate(STAGE_ORDER):
        sub = df[df["true_stage"] == stage]
        for method_idx, (method, suffix) in enumerate(
                [
                    ("Baseline", "abs_error_base"),
                    ("Hard-stage correction", "abs_error_hardcorr"),
                    ("Probability-stage correction", "abs_error_probcorr"),
                ]
        ):
            positions.append(center[stage_idx] + offsets[method_idx])
            values.append(sub[suffix].to_numpy())
            colors.append(palette[method])
            labels.append(method if stage_idx == 0 else None)
    bp = ax.boxplot(values, positions=positions, widths=0.18, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
        patch.set_edgecolor("#333333")
    for median in bp["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.0)
    ax.set_xticks(center)
    ax.set_xticklabels(STAGE_ORDER)
    ax.set_xlabel("True stage")
    ax.set_ylabel("Absolute error")
    handles = [plt.Line2D([0], [0], color=palette[m], lw=6, alpha=0.72) for m in METHOD_COLUMNS]
    ax.legend(handles, list(METHOD_COLUMNS.keys()), loc="best")
    fig.tight_layout()
    path = FIG_DIR / "Fig_abs_error_distribution_by_stage.png"
    fig.savefig(path, dpi=900)
    plt.close(fig)
    saved.append(path)

    coeff_weight = (
            df["VB_pred_probcorr"] - df["VB_pred_base"]
    )
    fig, ax1 = plt.subplots(figsize=(9.0, 4.6))
    ax1.plot(df["run_id"], df["p_E"], color=stage_colors["early"], linewidth=1.3, label="p_E")
    ax1.plot(df["run_id"], df["p_M"], color=stage_colors["middle"], linewidth=1.3, label="p_M")
    ax1.plot(df["run_id"], df["p_L"], color=stage_colors["late"], linewidth=1.3, label="p_L")
    ax1.set_xlabel("Run index")
    ax1.set_ylabel("Stage probability")
    ax2 = ax1.twinx()
    ax2.plot(df["run_id"], coeff_weight, color="#111111", linewidth=1.3, linestyle="--", label="Correction weight")
    ax2.set_ylabel("Correction weight")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best", ncol=2)
    fig.tight_layout()
    path = FIG_DIR / "Fig_stage_probability_and_correction_weight.png"
    fig.savefig(path, dpi=900)
    plt.close(fig)
    saved.append(path)

    return saved


def write_log(
        baseline_path: Path | None,
        prob_path: Path | None,
        calib_path: Path | None,
        calib_source: str,
        coeffs: pd.DataFrame | None,
        overall_df: pd.DataFrame | None,
        outputs: list[Path],
        error: str | None = None,
) -> None:
    lines = []
    lines.append("Stage-related bias analysis log")
    lines.append(f"Data root: {DATA_ROOT}")
    lines.append(f"Output root: {OUTPUT_ROOT}")
    lines.append(f"Run-level feature file: {RUN_FEATURE_PATH if RUN_FEATURE_PATH.exists() else 'not found'}")
    lines.append(f"MICFS-TCN prediction file: {baseline_path if baseline_path else 'not selected'}")
    lines.append(f"DC-PSR probability file: {prob_path if prob_path else 'not selected'}")
    lines.append(f"Calibration source: {calib_source}")
    lines.append(f"Calibration file: {calib_path if calib_path else 'not selected'}")
    if coeffs is not None:
        lines.append("Correction coefficients:")
        lines.append(coeffs.to_string(index=False))
    if overall_df is not None:
        lines.append("Overall metrics:")
        lines.append(overall_df.to_string(index=False))
    if error:
        lines.append(f"Stopped reason: {error}")
    lines.append("Output files:")
    lines.extend(str(path) for path in outputs)
    (LOG_DIR / "run_log.txt").write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> None:
    ensure_dirs()
    baseline_path = None
    prob_path = None
    calib_path = None
    calib_source = "not selected"
    outputs: list[Path] = []
    coeffs = None
    overall_df = None
    try:
        c6_base, baseline_path = load_c6_baseline()
        probs, prob_path = load_c6_probabilities()
        c6 = c6_base.merge(probs, on="run_id", how="inner")
        if c6.empty:
            raise AnalysisStop("C6 baseline predictions and DC-PSR probabilities could not be matched by run_id.")
        c6 = construct_true_stage(c6)
        calib, calib_path, calib_source = load_calibration(baseline_path)
        coeffs = learn_coefficients(calib)
        if coeffs["n_calibration"].min() <= 0:
            raise AnalysisStop("At least one calibration stage has no samples. C6 errors were not used to fill missing coefficients.")
        final_df, base_summary, overall_df, stage_df = build_outputs(c6, coeffs)
        outputs.extend(save_tables(final_df, base_summary, coeffs, overall_df, stage_df))
        outputs.extend(save_figures(final_df, stage_df))
        write_log(baseline_path, prob_path, calib_path, calib_source, coeffs, overall_df, outputs)
        print("Done. Stage-related bias analysis completed.")
        print(f"Results saved to: {OUTPUT_ROOT}")
    except AnalysisStop as exc:
        write_log(baseline_path, prob_path, calib_path, calib_source, coeffs, overall_df, outputs, error=str(exc))
        print(f"Stopped. {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
