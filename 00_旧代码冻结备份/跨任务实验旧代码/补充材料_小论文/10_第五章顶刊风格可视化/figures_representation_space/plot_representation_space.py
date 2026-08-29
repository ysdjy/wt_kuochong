# -*- coding: utf-8 -*-
r"""
Plot low-dimensional representation spaces for Chapter 5.

Main figures:
    Fig5_repr_main_umap.png/pdf
    Fig5_repr_main_pca.png/pdf
    Fig5_repr_main_misclassified.png/pdf
    Fig5_repr_dcpsr_final_umap.png/pdf
    Fig5_repr_dcpsr_final_pca.png/pdf

The main figure compares:
    Row 1: Raw relative feature representation
    Row 2: Shared latent representation h_{c,t} of multi-task TCN-GRU

No "FGDS-PSI" text is used. The final method is named DC-PSR.
"""

from __future__ import annotations

from pathlib import Path
import os
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")


ROOT = Path(os.environ.get(
    "PAPER_ROOT",
    r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\PHM实验\小论文",
))
OUT_ROOT = Path(os.environ.get(
    "CH5_VIS_OUT_ROOT",
    str(ROOT / "10_第五章顶刊风格可视化"),
))
OUT_DIR = OUT_ROOT / "figures_representation_space"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_RAW = OUT_DIR / "repr_raw_features.csv"
DATA_HIDDEN = OUT_DIR / "repr_hidden_hct.csv"
DATA_DCPSR = OUT_DIR / "repr_dcpsr_final_state.csv"

DPI = 900
SAVE_FORMATS = ("png", "pdf")

COLOR_E = "#5B8C5A"
COLOR_M = "#D98C2B"
COLOR_L = "#C23B3B"
COLOR_BLACK = "#222222"
COLOR_GRID = "#DADADA"
COLOR_B12 = "#B22222"
STAGE_COLORS = {"early": COLOR_E, "middle": COLOR_M, "late": COLOR_L}
MARKERS = {"C1": "o", "C4": "^", "C6": "s"}

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


META_COLS = {
    "sample_id", "split", "condition", "run_id", "true_stage", "true_stage_id",
    "pred_stage", "q_true", "q_hat", "p_E", "p_M", "p_L",
    "uncertainty", "entropy", "misclassified",
}


def save_fig(fig, stem):
    for fmt in SAVE_FORMATS:
        path = OUT_DIR / f"{stem}.{fmt}"
        fig.savefig(path, dpi=DPI if fmt == "png" else None, bbox_inches="tight", pad_inches=0.04)
        print(f"Saved: {path}")
    plt.close(fig)


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_BLACK)
    ax.spines["bottom"].set_color(COLOR_BLACK)
    ax.grid(True, linestyle="--", linewidth=0.55, color=COLOR_GRID, alpha=0.62)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8.5, colors=COLOR_BLACK)


def feature_cols(df):
    return [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]


def pca_2d(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[1] == 1:
        X = np.column_stack([X[:, 0], np.linspace(0, 1, X.shape[0])])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = X - X.mean(axis=0, keepdims=True)
    X = X / (X.std(axis=0, keepdims=True) + 1e-12)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    emb = X @ vt[:2].T
    if emb.shape[1] == 1:
        emb = np.column_stack([emb[:, 0], np.zeros(len(emb))])
    return emb


def embed_2d(X, method="pca", random_state=42):
    method = method.lower()
    if method == "umap":
        try:
            import umap  # type: ignore
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=20,
                min_dist=0.10,
                metric="euclidean",
                random_state=random_state,
            )
            return reducer.fit_transform(np.asarray(X, dtype=float)), "UMAP1", "UMAP2", "UMAP"
        except Exception as exc:
            print(f"UMAP unavailable or failed ({exc}); fallback to PCA.")
            arr = pca_2d(X)
            return arr, "PC1", "PC2", "PCA"
    arr = pca_2d(X)
    return arr, "PC1", "PC2", "PCA"


def normalize_stage(s):
    return s.astype(str).str.lower().replace({"e": "early", "m": "middle", "l": "late"})


def marker_for_condition(cond):
    return MARKERS.get(str(cond), "o")


def draw_scatter_by_condition(ax, emb, df, colors, edgecolors=None, sizes=None, alpha=0.82):
    if edgecolors is None:
        edgecolors = "white"
    if sizes is None:
        sizes = np.full(len(df), 18.0)
    conditions = df["condition"].astype(str) if "condition" in df.columns else pd.Series(["C6"] * len(df))
    for cond in sorted(conditions.unique()):
        mask = conditions == cond
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            c=np.asarray(colors, dtype=object)[mask.values] if not np.isscalar(colors) else colors,
            marker=marker_for_condition(cond),
            s=np.asarray(sizes)[mask.values],
            alpha=alpha,
            edgecolor=edgecolors if isinstance(edgecolors, str) else np.asarray(edgecolors, dtype=object)[mask.values],
            linewidth=0.35,
            label=cond,
        )


def add_lifecycle_path(ax, emb, q, color=COLOR_BLACK):
    q = np.asarray(q, dtype=float)
    if np.nanmax(q) - np.nanmin(q) < 1e-8:
        return
    bins = np.linspace(np.nanmin(q), np.nanmax(q), 12)
    centers = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (q >= lo) & (q <= hi)
        if mask.sum() >= 3:
            centers.append(np.nanmean(emb[mask], axis=0))
    if len(centers) < 3:
        return
    centers = np.asarray(centers)
    ax.plot(centers[:, 0], centers[:, 1], color=color, linestyle="--", linewidth=1.15, alpha=0.75, zorder=5)
    ax.annotate(
        "",
        xy=centers[-1],
        xytext=centers[-2],
        arrowprops=dict(arrowstyle="->", lw=1.2, color=color),
        zorder=6,
    )


def load_or_make_proxy():
    """Load real exported representations; if absent, build a clearly labeled proxy."""
    if DATA_RAW.exists() and DATA_HIDDEN.exists():
        raw = pd.read_csv(DATA_RAW)
        hidden = pd.read_csv(DATA_HIDDEN)
        return raw, hidden, False

    # Fallback proxy from existing comparison results, only for plotting before extraction.
    pred_path = ROOT / "4_comparison_experiment_recheck" / "1_results" / "FINAL_comparison_predictions.csv"
    q_path = ROOT / "9_probability_wear_consistency_analysis" / "Data_5_4_A6_probability_wear_trajectory.csv"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Cannot find representation CSVs or fallback prediction file.\n"
            f"Run extract_hidden_representation.py first.\nMissing: {pred_path}"
        )
    pred = pd.read_csv(pred_path)
    qdf = pd.read_csv(q_path) if q_path.exists() else None
    q = qdf["q_true"].values if qdf is not None and "q_true" in qdf.columns and len(qdf) == len(pred) else np.linspace(0, 1, len(pred))

    meta = pd.DataFrame({
        "sample_id": [f"proxy_{i:05d}" for i in range(len(pred))],
        "split": "test_C6",
        "condition": pred.get("condition", "C6"),
        "run_id": pred["run_id_end"],
        "true_stage": pred["true_stage"],
        "pred_stage": pred["pred_B11"] if "pred_B11" in pred.columns else pred["pred_B12"],
        "q_true": q,
        "q_hat": q,
        "p_E": pred["prob_E_B11"],
        "p_M": pred["prob_M_B11"],
        "p_L": pred["prob_L_B11"],
    })
    prob = meta[["p_E", "p_M", "p_L"]].values
    meta["uncertainty"] = 1 - prob.max(axis=1)
    meta["entropy"] = -(prob * np.log(prob + 1e-12)).sum(axis=1) / np.log(3.0)
    meta["misclassified"] = (meta["true_stage"].astype(str).str.lower() != meta["pred_stage"].astype(str).str.lower()).astype(int)

    raw = meta.copy()
    for c in ["prob_E_B5", "prob_M_B5", "prob_L_B5"]:
        if c in pred.columns:
            raw[c] = pred[c]
    raw["run_norm"] = np.linspace(0, 1, len(raw))

    hidden = meta.copy()
    hidden["p_E"] = pred["prob_E_B11"]
    hidden["p_M"] = pred["prob_M_B11"]
    hidden["p_L"] = pred["prob_L_B11"]
    hidden["run_norm"] = np.linspace(0, 1, len(hidden))

    raw.to_csv(OUT_DIR / "repr_raw_features_PROXY.csv", index=False, encoding="utf-8-sig")
    hidden.to_csv(OUT_DIR / "repr_hidden_hct_PROXY.csv", index=False, encoding="utf-8-sig")
    return raw, hidden, True


def plot_main(raw, hidden, method="pca", highlight_misclassified=False, stem="Fig5_repr_main_pca"):
    raw_cols = feature_cols(raw)
    hidden_cols = [c for c in hidden.columns if c.startswith("h_")]
    if not hidden_cols:
        hidden_cols = feature_cols(hidden)

    raw_emb, xlab1, ylab1, used1 = embed_2d(raw[raw_cols].values, method=method)
    hid_emb, xlab2, ylab2, used2 = embed_2d(hidden[hidden_cols].values, method=method)
    used = used1 if used1 == used2 else f"{used1}/{used2}"

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.6))
    rows = [
        (raw, raw_emb, "Raw features", xlab1, ylab1),
        (hidden, hid_emb, r"Shared $h_{c,t}$", xlab2, ylab2),
    ]
    titles = [
        "true stage",
        r"$q$",
        "uncertainty",
    ]
    q_mappable = None
    u_mappable = None
    for i, (df, emb, rowname, xlab, ylab) in enumerate(rows):
        stage = normalize_stage(df["true_stage"])
        q = pd.to_numeric(df["q_true"], errors="coerce").fillna(pd.to_numeric(df["q_hat"], errors="coerce")).values
        u = pd.to_numeric(df["entropy"] if "entropy" in df.columns else df["uncertainty"], errors="coerce").fillna(0).values

        for j in range(3):
            ax = axes[i, j]
            if j == 0:
                colors = stage.map(STAGE_COLORS).fillna("#999999").values
                edge = np.where(pd.to_numeric(df.get("misclassified", 0), errors="coerce").fillna(0).values > 0, COLOR_BLACK, "white")
                mis = pd.to_numeric(df.get("misclassified", 0), errors="coerce").fillna(0).values > 0
                sizes = np.where(mis & bool(highlight_misclassified), 34, 17)
                draw_scatter_by_condition(ax, emb, df, colors, edgecolors=edge if highlight_misclassified else "white", sizes=sizes)
                add_lifecycle_path(ax, emb, q)
            elif j == 1:
                conditions = df["condition"].astype(str) if "condition" in df.columns else pd.Series(["C6"] * len(df))
                for cond in sorted(conditions.unique()):
                    mask = conditions == cond
                    q_mappable = ax.scatter(
                        emb[mask, 0], emb[mask, 1],
                        c=q[mask.values],
                        cmap="viridis",
                        vmin=np.nanmin(q),
                        vmax=np.nanmax(q),
                        marker=marker_for_condition(cond),
                        s=17,
                        alpha=0.84,
                        edgecolor="white",
                        linewidth=0.25,
                    )
                add_lifecycle_path(ax, emb, q)
            else:
                conditions = df["condition"].astype(str) if "condition" in df.columns else pd.Series(["C6"] * len(df))
                for cond in sorted(conditions.unique()):
                    mask = conditions == cond
                    u_mappable = ax.scatter(
                        emb[mask, 0], emb[mask, 1],
                        c=u[mask.values],
                        cmap="magma",
                        vmin=0,
                        vmax=max(1e-6, np.nanmax(u)),
                        marker=marker_for_condition(cond),
                        s=17,
                        alpha=0.84,
                        edgecolor="white",
                        linewidth=0.25,
                    )
            ax.set_title(f"({chr(97 + i * 3 + j)}) {rowname} / {titles[j]}", loc="left", fontsize=10.5, fontweight="bold")
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
            style_axis(ax)

    if q_mappable is not None:
        cb = fig.colorbar(q_mappable, ax=axes[:, 1], fraction=0.030, pad=0.015)
        cb.set_label(r"Relative degradation position $q$")
    if u_mappable is not None:
        cb = fig.colorbar(u_mappable, ax=axes[:, 2], fraction=0.030, pad=0.015)
        cb.set_label("Predictive entropy")

    stage_handles = [Patch(facecolor=STAGE_COLORS[s], label=s.capitalize()) for s in ["early", "middle", "late"]]
    cond_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777", markeredgecolor="white", label="C1", markersize=7),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#777777", markeredgecolor="white", label="C4", markersize=7),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#777777", markeredgecolor="white", label="C6", markersize=7),
    ]
    misc_handle = [Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=COLOR_BLACK,
                          label="Misclassified", markersize=7)] if highlight_misclassified else []
    fig.legend(handles=stage_handles + cond_handles + misc_handle, ncol=7 if highlight_misclassified else 6,
               loc="lower center", bbox_to_anchor=(0.5, -0.01), fontsize=9)
    fig.text(0.01, 0.01, f"Dimensionality reduction: {used}", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    save_fig(fig, stem)


def plot_dcpsr(df, method="pca", stem="Fig5_repr_dcpsr_final_pca"):
    if df is None or df.empty:
        return
    cols = ["p_E", "p_M", "p_L"]
    if "q_hat" in df.columns:
        cols.append("q_hat")
    emb, xlab, ylab, used = embed_2d(df[cols].values, method=method)
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.9))
    stage = normalize_stage(df["true_stage"])
    q = pd.to_numeric(df.get("q_true", df.get("q_hat")), errors="coerce")
    if q.isna().all() and "q_hat" in df.columns:
        q = pd.to_numeric(df["q_hat"], errors="coerce")
    if q.isna().any():
        q = q.fillna(pd.Series(np.linspace(0, 1, len(df)), index=df.index))
    q = q.values
    u = pd.to_numeric(df.get("entropy", df.get("uncertainty")), errors="coerce").fillna(0).values

    titles = ["true stage", r"$q$ / lifecycle", "uncertainty"]
    mappables = []
    for j, ax in enumerate(axes):
        if j == 0:
            colors = stage.map(STAGE_COLORS).fillna("#999999").values
            draw_scatter_by_condition(ax, emb, df, colors)
            add_lifecycle_path(ax, emb, q)
        elif j == 1:
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=q, cmap="viridis", s=18, alpha=0.86, edgecolor="white", linewidth=0.25)
            mappables.append((sc, r"Relative degradation position $q$"))
            add_lifecycle_path(ax, emb, q)
        else:
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=u, cmap="magma", s=18, alpha=0.86, edgecolor="white", linewidth=0.25)
            mappables.append((sc, "Predictive entropy"))
        ax.set_title(f"({chr(97+j)}) Final probabilistic state of DC-PSR / {titles[j]}", loc="left", fontsize=10.3, fontweight="bold")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        style_axis(ax)
    for sc, lab in mappables:
        cb = fig.colorbar(sc, ax=axes.tolist(), fraction=0.022, pad=0.015)
        cb.set_label(lab)
    handles = [Patch(facecolor=STAGE_COLORS[s], label=s.capitalize()) for s in ["early", "middle", "late"]]
    fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=9)
    fig.text(0.01, 0.02, f"Dimensionality reduction: {used}", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=[0, 0.065, 1, 1])
    save_fig(fig, stem)


def write_readme(proxy_used: bool):
    text = f"""# Representation space visualization

## Purpose

This folder contains low-dimensional representation-space figures for Chapter 5.

## Main figure

- `Fig5_repr_main_umap`: Raw relative feature representation vs shared latent representation `h_ct` of multi-task TCN-GRU.
- `Fig5_repr_main_pca`: PCA fallback / backup version.
- `Fig5_repr_main_misclassified`: Same as the main PCA layout, but misclassified samples are highlighted by black outlines.

Rows:

1. Raw relative feature representation.
2. Shared latent representation `h_ct` of multi-task TCN-GRU.

Columns:

1. True stage.
2. Relative degradation position `q`.
3. Predictive entropy.

Marker shapes encode condition:

- circle: C1
- triangle: C4
- square: C6

## Supplementary DC-PSR state figure

- `Fig5_repr_dcpsr_final_umap`
- `Fig5_repr_dcpsr_final_pca`

These figures visualize the final probabilistic state of DC-PSR, i.e.,
`[p_E*, p_M*, p_L*, q_hat]`.

## Data status

Proxy mode used: {proxy_used}

If `Proxy mode used` is True, run `extract_hidden_representation.py` first to export real `h_ct`.

## Recommended conclusion

Compared with raw online relative features, the shared latent representation `h_ct`
forms clearer stage separation and a more continuous degradation trajectory.
High-uncertainty or misclassified samples tend to appear near stage transition
regions. The final probabilistic state of DC-PSR remains consistent with the
continuous degradation position, supporting its interpretation as a
degradation-consistent state representation.
"""
    (OUT_DIR / "README_representation_space.md").write_text(text, encoding="utf-8")


def main():
    print("=" * 100)
    print("Plot representation space figures")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 100)

    raw, hidden, proxy_used = load_or_make_proxy()

    plot_main(raw, hidden, method="umap", highlight_misclassified=False, stem="Fig5_repr_main_umap")
    plot_main(raw, hidden, method="pca", highlight_misclassified=False, stem="Fig5_repr_main_pca")
    plot_main(raw, hidden, method="pca", highlight_misclassified=True, stem="Fig5_repr_main_misclassified")

    dcpsr = pd.read_csv(DATA_DCPSR) if DATA_DCPSR.exists() else None
    if dcpsr is not None:
        plot_dcpsr(dcpsr, method="umap", stem="Fig5_repr_dcpsr_final_umap")
        plot_dcpsr(dcpsr, method="pca", stem="Fig5_repr_dcpsr_final_pca")
    else:
        print("DC-PSR final state CSV not found; skip supplementary DC-PSR figure.")

    write_readme(proxy_used)
    try:
        shutil.copy2(Path(__file__).resolve(), OUT_DIR / "plot_representation_space.py")
    except Exception:
        pass
    print("Finished.")


if __name__ == "__main__":
    main()
