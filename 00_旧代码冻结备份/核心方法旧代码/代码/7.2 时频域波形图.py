# -*- coding: utf-8 -*-
r"""
Figure:
The time-domain and frequency-domain signals of milling force under different wear stages.

Layout:
    2 rows x 3 columns
    Row 1: time-domain milling force signal
    Row 2: frequency-domain amplitude spectrum
    Columns: Initial stage, Stable stage, Acceleration stage

Data:
    PHM 2010 raw signal files, e.g.
    C:\Users\wangting\Desktop\博士开题\公开数据\1PHM\c1\c1\c_1_001.csv

Notes:
    Raw signal CSV files have no header. This script assumes:
        column 0 = Fx
        column 1 = Fy
        column 2 = Fz
    Change FORCE_COL if you want Fy or Fz.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# =========================================================
# 0. Global configuration
# =========================================================
ROOT = Path(r"C:\Users\wangting\Desktop\博士开题\公开数据\1PHM")

# Choose one condition for this six-panel illustration.
# You can change it to "c4" or "c6".
CONDITION = "c1"
COND_ID = int(CONDITION.replace("c", ""))

SIGNAL_DIR = ROOT / CONDITION / CONDITION
WEAR_FILE = ROOT / CONDITION / f"{CONDITION}_wear.csv"

OUT_ROOT = ROOT / "PHM实验" / "小论文" / "2_force_time_frequency_stage_figure"
DIR_FIG = OUT_ROOT / "figures"
DIR_TABLE = OUT_ROOT / "tables"
DIR_DATA = OUT_ROOT / "intermediate_data"

for d in [DIR_FIG, DIR_TABLE, DIR_DATA]:
    d.mkdir(parents=True, exist_ok=True)

DPI = 600

# PHM 2010 dynamometer sampling frequency is commonly 50 kHz.
FS = 50000.0

# Plot settings
FORCE_COL = 0              # 0=Fx, 1=Fy, 2=Fz
FORCE_NAME = "Fx"
TIME_POINTS = 3000         # show 3000 samples, matching the reference style
SEGMENT_MODE = "center"    # "center" avoids the entry transient at the beginning of each cut
FREQ_MAX = 12000.0         # Hz

# Stage thresholds, consistent with your relative-stage figure logic
Q_EARLY = 0.30
Q_LATE = 0.72
RATE_LATE_Q = 0.78
EPS = 1e-12

STAGE_ORDER = ["early", "middle", "late"]
STAGE_DISPLAY = {
    "early": "Initial stage",
    "middle": "Stable stage",
    "late": "Acceleration stage",
}

LINE_COLOR = "#0072BD"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["legend.frameon"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# =========================================================
# 1. Stage construction from wear file
# =========================================================
def load_wear_table():
    if not WEAR_FILE.exists():
        raise FileNotFoundError(f"Wear file not found:\n{WEAR_FILE}")

    wear = pd.read_csv(WEAR_FILE)
    wear.columns = [str(c).strip() for c in wear.columns]
    if "cut" not in wear.columns:
        raise ValueError("Wear file must contain column: cut")

    flute_cols = [c for c in ["flute_1", "flute_2", "flute_3"] if c in wear.columns]
    if not flute_cols:
        raise ValueError("Wear file must contain at least one flute wear column.")

    wear["run_id"] = pd.to_numeric(wear["cut"], errors="coerce").astype(int)
    for c in flute_cols:
        wear[c] = pd.to_numeric(wear[c], errors="coerce")
    wear["VB"] = wear[flute_cols].max(axis=1)
    wear = wear.dropna(subset=["VB"]).sort_values("run_id").reset_index(drop=True)
    return wear[["run_id", "VB"] + flute_cols]


def smooth_series(x, window):
    return pd.Series(x, dtype=float).rolling(window=window, min_periods=1, center=True).mean().values


def build_relative_stages(wear):
    out = wear.copy().sort_values("run_id").reset_index(drop=True)
    vb_smooth = smooth_series(out["VB"].values, window=7)
    q = (vb_smooth - np.nanmin(vb_smooth)) / (np.nanmax(vb_smooth) - np.nanmin(vb_smooth) + EPS)

    rate = pd.Series(q).diff().fillna(0.0).values
    rate = smooth_series(rate, window=5)
    rate_norm = (rate - np.nanmin(rate)) / (np.nanmax(rate) - np.nanmin(rate) + EPS)

    theta_E = float(np.quantile(q, Q_EARLY))
    theta_L = float(np.quantile(q, Q_LATE))
    theta_v = float(np.quantile(rate_norm, RATE_LATE_Q))

    stage = []
    for qi, ri in zip(q, rate_norm):
        if qi <= theta_E:
            stage.append("early")
        elif (qi >= theta_L) or (ri >= theta_v):
            stage.append("late")
        else:
            stage.append("middle")

    out["VB_smooth"] = vb_smooth
    out["q_ct"] = q
    out["nu_norm_ct"] = rate_norm
    out["stage"] = stage

    summary = pd.DataFrame([{
        "condition": CONDITION.upper(),
        "theta_E": theta_E,
        "theta_L": theta_L,
        "theta_v": theta_v,
        "early_count": int((out["stage"] == "early").sum()),
        "middle_count": int((out["stage"] == "middle").sum()),
        "late_count": int((out["stage"] == "late").sum()),
    }])

    return out, summary


def select_representative_cuts(stage_df):
    """
    Select the middle cut within each stage segment/table subset.
    This avoids choosing boundary samples.
    """
    rows = []
    for st in STAGE_ORDER:
        sub = stage_df[stage_df["stage"] == st].sort_values("run_id").reset_index(drop=True)
        if sub.empty:
            raise ValueError(f"No samples found for stage: {st}")
        idx = len(sub) // 2
        row = sub.iloc[idx].copy()
        rows.append({
            "stage": st,
            "stage_display": STAGE_DISPLAY[st],
            "run_id": int(row["run_id"]),
            "VB": float(row["VB"]),
            "q_ct": float(row["q_ct"]),
            "nu_norm_ct": float(row["nu_norm_ct"]),
        })
    rep = pd.DataFrame(rows)
    rep.to_csv(DIR_TABLE / "selected_representative_cuts_for_force_signal.csv",
               index=False, encoding="utf-8-sig")
    return rep


# =========================================================
# 2. Raw signal loading and spectrum
# =========================================================
def signal_file_for_cut(run_id):
    return SIGNAL_DIR / f"c_{COND_ID}_{int(run_id):03d}.csv"


def load_force_signal(run_id, col=FORCE_COL):
    path = signal_file_for_cut(run_id)
    if not path.exists():
        raise FileNotFoundError(f"Signal file not found:\n{path}")

    sig_df = pd.read_csv(path, header=None)
    if col >= sig_df.shape[1]:
        raise ValueError(f"FORCE_COL={col} out of range. File has {sig_df.shape[1]} columns.")

    x = pd.to_numeric(sig_df.iloc[:, col], errors="coerce").dropna().values.astype(float)
    return x, path


def extract_signal_segment(x, n_points=TIME_POINTS, mode=SEGMENT_MODE):
    """
    Extract a stable segment for plotting.

    The beginning of each PHM cutting file often contains tool-entry transient,
    so using the first 3000 samples can create the artificial "small-to-large"
    trend in every wear stage. The center segment is more representative of
    steady milling.
    """
    x = np.asarray(x, dtype=float)
    if len(x) <= n_points:
        return x.copy(), 0

    if mode == "first":
        start = 0
    elif mode == "center":
        start = max(0, len(x) // 2 - n_points // 2)
    elif mode == "last":
        start = len(x) - n_points
    else:
        raise ValueError("SEGMENT_MODE must be one of: first, center, last")

    end = min(start + n_points, len(x))
    return x[start:end].copy(), start


def amplitude_spectrum(x, fs=FS):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)

    # Hann window reduces spectral leakage and makes peaks cleaner for paper plots.
    win = np.hanning(len(x))
    xw = x * win

    freq = np.fft.rfftfreq(len(xw), d=1.0 / fs)
    amp = np.abs(np.fft.rfft(xw)) * 2.0 / (np.sum(win) + EPS)
    return freq, amp


# =========================================================
# 3. Plot
# =========================================================
def plot_time_frequency_figure(rep):
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.1), constrained_layout=False)

    plot_records = []
    for j, row in rep.iterrows():
        stage = row["stage"]
        run_id = int(row["run_id"])
        display = row["stage_display"]

        sig, path = load_force_signal(run_id, FORCE_COL)
        sig_plot, start_idx = extract_signal_segment(sig, TIME_POINTS, SEGMENT_MODE)
        t = np.arange(len(sig_plot)) / FS

        freq, amp = amplitude_spectrum(sig_plot, FS)
        freq_mask = freq <= FREQ_MAX

        # Time domain
        ax_t = axes[0, j]
        ax_t.plot(t, sig_plot, color=LINE_COLOR, linewidth=0.75)
        ax_t.set_xlim(t[0], t[-1])
        ax_t.set_xlabel("t/s", fontweight="bold")
        ax_t.set_ylabel(f"{FORCE_NAME}/N", fontweight="bold")
        ax_t.set_title(f"({chr(ord('a') + j)}) {display}", fontsize=12, fontweight="bold")
        ax_t.grid(False)

        # Frequency domain
        ax_f = axes[1, j]
        ax_f.plot(freq[freq_mask], amp[freq_mask], color=LINE_COLOR, linewidth=0.80)
        ax_f.set_xlim(0, FREQ_MAX)
        ax_f.set_xlabel("f/Hz", fontweight="bold")
        ax_f.set_ylabel("Amplitude", fontweight="bold")
        ax_f.set_title(f"({chr(ord('d') + j)}) {display}", fontsize=12, fontweight="bold")
        ax_f.grid(False)

        # Bottom stage labels, close to your reference style.
        ax_f.text(
            0.5, -0.36, display,
            transform=ax_f.transAxes,
            ha="center", va="top",
            fontsize=15,
            fontweight="bold",
        )

        for ax in [ax_t, ax_f]:
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
            ax.tick_params(direction="in", top=True, right=True, width=1.0)

        plot_records.append({
            "stage": stage,
            "stage_display": display,
            "run_id": run_id,
            "signal_file": str(path),
            "n_samples": len(sig),
            "segment_mode": SEGMENT_MODE,
            "segment_start_index": int(start_idx),
            "segment_points": int(len(sig_plot)),
            "fs": FS,
            "force_column": FORCE_COL,
            "force_name": FORCE_NAME,
        })

    plt.tight_layout(rect=[0.02, 0.06, 0.995, 0.985], w_pad=2.2, h_pad=2.2)

    out_png = DIR_FIG / f"Fig_force_time_frequency_{CONDITION}_{FORCE_NAME}_wear_stages.png"
    out_pdf = DIR_FIG / f"Fig_force_time_frequency_{CONDITION}_{FORCE_NAME}_wear_stages.pdf"
    out_tif = DIR_FIG / f"Fig_force_time_frequency_{CONDITION}_{FORCE_NAME}_wear_stages.tif"
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_tif, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(plot_records).to_csv(DIR_TABLE / "force_signal_plot_records.csv",
                                      index=False, encoding="utf-8-sig")

    print(f"Saved:\n{out_png}\n{out_pdf}\n{out_tif}")


# =========================================================
# 4. Main
# =========================================================
def main():
    wear = load_wear_table()
    stage_df, summary = build_relative_stages(wear)
    rep = select_representative_cuts(stage_df)

    stage_df.to_csv(DIR_DATA / "condition_relative_stage_from_wear.csv",
                    index=False, encoding="utf-8-sig")
    summary.to_csv(DIR_TABLE / "condition_relative_stage_threshold_summary.csv",
                   index=False, encoding="utf-8-sig")

    plot_time_frequency_figure(rep)

    print("=" * 80)
    print("Figure generation finished.")
    print(f"Condition  : {CONDITION.upper()}")
    print(f"Force      : {FORCE_NAME}, column {FORCE_COL}")
    print(f"Signal dir : {SIGNAL_DIR}")
    print(f"Wear file  : {WEAR_FILE}")
    print(f"Output dir : {OUT_ROOT}")
    print("=" * 80)


if __name__ == "__main__":
    main()
