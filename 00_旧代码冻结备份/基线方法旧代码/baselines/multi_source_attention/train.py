# -*- coding: utf-8 -*-
r"""
Training + evaluation for the Multi-source Attention (Multi-Attention-CNN)
baseline. Two protocols, kept strictly separate (never mixed into one
number, never cross-contaminated):

Protocol A (original-paper sanity reproduction):
    - Labels: paper's own EM-derived fixed pass-index partition
      (preprocessing.ORIGINAL_STAGE_RANGES / metadata.csv "stage_original").
    - Split: stratified 70/30 within each of the 3 stage labels, pooled
      across C1+C4+C6 (945 total), per PAPER_SPEC.md.
    - Hyperparameters: paper's own "Scheme 5" (Fig. 9): epochs=100,
      batch_size=128, lr=0.001, Adam, step decay x0.1 at epoch 30,
      weight_decay=1e-4 (Missing in paper -> conventional default),
      CrossEntropyLoss.
    - N_SEEDS repeated trainings (paper: 12, drop best/worst, average 10;
      this reproduction: N_SEEDS=5 by default, documented adaptation --
      see README.md).
    - Output: outputs/multi_source_attention/original_protocol/

Protocol B (Unified DC-PSR comparison):
    - Labels: this project's condition-relative Early/Middle/Late
      (data/label_utils.py, reused byte-for-byte from
      代码/main_experiment_3_fgds_psi_optimized.py).
    - Split: train/val carved from C1+C4 only (data/label_utils.py
      get_unified_split), test = C6 held out entirely. Model selection
      (best epoch) uses ONLY the C1+C4 internal validation split -- C6 is
      never touched until the single final evaluation (no leakage, per
      task instruction #58/#27).
    - Metrics: reuses 代码/main_experiment_3_fgds_psi_optimized.py's own
      `manuscript_metric_row` / `stage_consistency_metrics` so Acc,
      Macro-F1, per-stage F1, M->E/M->L, Rev/Jump/Smooth are computed
      identically to every other baseline in the unified comparison.
      q-MAE/RMSE/R2 are N/A (no q-regression head in this architecture).
    - Output: outputs/multi_source_attention/unified_protocol/

Smoke-test mode (--smoke-test): a handful of epochs on a tiny subset, to
verify the full pipeline (data loading -> training -> checkpoint ->
evaluation -> CSV/JSON output) runs end to end before any real run.
Output: outputs/multi_source_attention/smoke/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.backends.cudnn.benchmark = True  # fixed input size (224x224) -> autotune is a pure speed win

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

from model import MultiAttentionCNN
import label_utils as L  # baselines/multi_source_attention/data/label_utils.py
from label_utils import dcpsr_base  # read-only reuse of DC-PSR's own metric/label functions

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent / "data"
IMG_DIR = DATA_DIR / "images"
OUT_ROOT = PROJECT_ROOT / "outputs" / "multi_source_attention"

# --- Protocol A hyperparameters (PAPER_SPEC.md "Training scheme selection (Scheme 5)") ---
PROTO_A_CFG = dict(epochs=100, batch_size=128, lr=0.001, decay_epoch=30, decay_factor=0.1,
                    weight_decay=1e-4, n_seeds=5)
# --- Protocol B: same architecture/optimizer family, early-stopped on source-only (C1+C4) val ---
PROTO_B_CFG = dict(max_epochs=100, patience=15, batch_size=64, lr=0.001, weight_decay=1e-4, seed=42)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ImagePairDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, label_col: str):
        self.rows = rows.reset_index(drop=True)
        self.label_col = label_col

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows.iloc[idx]
        force = np.load(IMG_DIR / f"{r.condition}_{int(r.run_id):03d}_force.npy").astype(np.float32) / 255.0
        vib = np.load(IMG_DIR / f"{r.condition}_{int(r.run_id):03d}_vib.npy").astype(np.float32) / 255.0
        force = torch.from_numpy(force).permute(2, 0, 1)  # [3,H,W]
        vib = torch.from_numpy(vib).permute(2, 0, 1)
        label = int(getattr(r, self.label_col))
        return force, vib, label


def run_epoch(model, loader, device, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_loss, n, correct = 0.0, 0, 0
    with torch.set_grad_enabled(train_mode):
        for force, vib, label in loader:
            force, vib, label = force.to(device), vib.to(device), label.to(device)
            logits = model(force, vib)
            loss = F.cross_entropy(logits, label)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(label)
            correct += (logits.argmax(dim=1) == label).sum().item()
            n += len(label)
    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_probs, all_true = [], []
    for force, vib, label in loader:
        force, vib = force.to(device), vib.to(device)
        logits = model(force, vib)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_true.append(label.numpy())
    return np.concatenate(all_probs), np.concatenate(all_true)


def build_pred_df(rows: pd.DataFrame, probs: np.ndarray, true_ids: np.ndarray) -> pd.DataFrame:
    """Columns matching 代码/main_experiment_3_fgds_psi_optimized.py's own
    manuscript_metric_row/stage_consistency_metrics contract."""
    pred_ids = probs.argmax(axis=1)
    df = pd.DataFrame({
        "condition": rows["condition"].values,
        "cut_index": rows["run_id"].values.astype(int),
        "stage_true_id": true_ids.astype(int),
        "stage_pred_raw": pred_ids.astype(int),
        "raw_prob_early": probs[:, 0],
        "raw_prob_middle": probs[:, 1],
        "raw_prob_late": probs[:, 2],
        "q_true_model": np.nan,  # no q-regression head -> q metrics are N/A by construction
        "q_hat": np.nan,
    })
    return df


def save_outputs(out_dir: Path, pred_df: pd.DataFrame, metric_row: dict, model, extra: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    id_to_stage = dcpsr_base.ID_TO_STAGE
    pred_table = pred_df.copy()
    pred_table["true_stage"] = pred_table["stage_true_id"].map(id_to_stage)
    pred_table["pred_stage"] = pred_table["stage_pred_raw"].map(id_to_stage)
    pred_table = pred_table.rename(columns={"cut_index": "run_id", "raw_prob_early": "p_early",
                                             "raw_prob_middle": "p_middle", "raw_prob_late": "p_late"})
    pred_table = pred_table[["condition", "run_id", "true_stage", "pred_stage", "p_early", "p_middle", "p_late"]]
    pred_table.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(pred_df["stage_true_id"], pred_df["stage_pred_raw"], labels=[0, 1, 2])
    cm_df = pd.DataFrame(cm, index=["true_early", "true_middle", "true_late"],
                          columns=["pred_early", "pred_middle", "pred_late"])
    cm_df.to_csv(out_dir / "confusion_matrix.csv", encoding="utf-8-sig")

    row = dict(metric_row)
    row["n_params"] = model.num_parameters()
    row["n_trainable_params"] = model.num_parameters(trainable_only=True)
    row.update(extra)
    pd.DataFrame([row]).to_csv(out_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in row.items()},
                   f, indent=2, ensure_ascii=False)


def load_metadata() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "metadata.csv")


# =========================================================
# Protocol A
# =========================================================
def stratified_70_30_split(meta: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for _, g in meta.groupby("stage_original"):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * 0.3)))
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        train_parts.append(meta.loc[train_idx])
        test_parts.append(meta.loc[test_idx])
    return pd.concat(train_parts).reset_index(drop=True), pd.concat(test_parts).reset_index(drop=True)


def train_one_seed_protocol_a(meta: pd.DataFrame, seed: int, device: str, cfg: dict):
    set_seed(seed)
    train_meta, test_meta = stratified_70_30_split(meta, seed)
    train_ds = ImagePairDataset(train_meta, "stage_original_id")
    test_ds = ImagePairDataset(test_meta, "stage_original_id")
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    model = MultiAttentionCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg["decay_epoch"], gamma=cfg["decay_factor"])

    for epoch in range(cfg["epochs"]):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, opt)
        sched.step()

    probs, true_ids = predict(model, test_loader, device)
    pred_df = build_pred_df(test_meta, probs, true_ids)
    metric_row = dcpsr_base.manuscript_metric_row(pred_df, method="raw", method_id="Multi-Attention-CNN (Protocol A)",
                                                   split="test_70_30")
    return model, pred_df, metric_row


def run_protocol_a(device: str, n_seeds: int | None = None, epochs: int | None = None):
    meta = load_metadata()
    cfg = dict(PROTO_A_CFG)
    if n_seeds is not None:
        cfg["n_seeds"] = n_seeds
    if epochs is not None:
        cfg["epochs"] = epochs

    out_dir = OUT_ROOT / "original_protocol"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_rows = []
    best_model, best_pred_df, best_row = None, None, None
    t0 = time.time()
    for i, seed in enumerate(range(42, 42 + cfg["n_seeds"])):
        t_seed = time.time()
        model, pred_df, row = train_one_seed_protocol_a(meta, seed, device, cfg)
        row["seed"] = seed
        row["train_time_s"] = time.time() - t_seed
        seed_rows.append(row)
        print(f"[Protocol A][seed {seed}] Acc={row['Acc']:.4f} Macro-F1={row['Macro-F1']:.4f} "
              f"({row['train_time_s']:.1f}s)")
        if best_row is None or row["Acc"] > best_row["Acc"]:
            best_model, best_pred_df, best_row = model, pred_df, row

    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(out_dir / "all_seeds_results.csv", index=False, encoding="utf-8-sig")

    mean_row = {k: seed_df[k].mean() for k in ["Acc", "Macro-F1", "E-F1", "M-F1", "L-F1", "M-Pre", "M-Rec",
                                                 "M→E", "M→L", "Rev", "Jump", "Smooth"]}
    mean_row["n_seeds"] = cfg["n_seeds"]
    mean_row["paper_reported_Acc"] = 0.982
    mean_row["gap_vs_paper"] = mean_row["Acc"] - 0.982
    with open(out_dir / "seed_average_summary.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, np.integer, float, int)) else v
                    for k, v in mean_row.items()}, f, indent=2, ensure_ascii=False)

    save_outputs(out_dir, best_pred_df, best_row, best_model,
                 extra={"total_time_s": time.time() - t0, "protocol": "A_original_7_3_split",
                        "note": "best-of-N-seeds checkpoint; see all_seeds_results.csv / seed_average_summary.json for the mean-over-seeds numbers actually reported"})
    torch.save(best_model.state_dict(), out_dir / "checkpoint_best.pth")
    print(f"[Protocol A] mean Acc over {cfg['n_seeds']} seeds = {mean_row['Acc']:.4f} "
          f"(paper reports 0.982, gap={mean_row['gap_vs_paper']:+.4f})")
    return mean_row


# =========================================================
# Protocol B (Unified)
# =========================================================
def run_protocol_b(device: str, max_epochs: int | None = None):
    cfg = dict(PROTO_B_CFG)
    if max_epochs is not None:
        cfg["max_epochs"] = max_epochs
    set_seed(cfg["seed"])

    meta = load_metadata()
    tr_df, va_df, te_df = L.get_unified_split()
    # metadata.csv already has condition/run_id/stage_unified_id; join to get image paths (condition,run_id already match)
    def attach(split_df):
        return meta.merge(split_df[["condition", "run_id"]], on=["condition", "run_id"], how="inner")

    train_meta, val_meta, test_meta = attach(tr_df), attach(va_df), attach(te_df)
    train_loader = DataLoader(ImagePairDataset(train_meta, "stage_unified_id"), batch_size=cfg["batch_size"],
                               shuffle=True, num_workers=0)
    val_loader = DataLoader(ImagePairDataset(val_meta, "stage_unified_id"), batch_size=cfg["batch_size"],
                             shuffle=False, num_workers=0)
    test_loader = DataLoader(ImagePairDataset(test_meta, "stage_unified_id"), batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=0)

    model = MultiAttentionCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    out_dir = OUT_ROOT / "unified_protocol"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc, best_epoch, patience_ctr = -1.0, -1, 0
    best_state = None
    log_rows = []
    t0 = time.time()
    for epoch in range(cfg["max_epochs"]):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, opt)
        va_loss, va_acc = run_epoch(model, val_loader, device, None)
        log_rows.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                          "val_loss": va_loss, "val_acc": va_acc})
        improved = va_acc > best_val_acc
        if improved:
            best_val_acc, best_epoch = va_acc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        print(f"[Protocol B] epoch {epoch}: train_acc={tr_acc:.4f} val_acc={va_acc:.4f} "
              f"(best={best_val_acc:.4f}@{best_epoch})")
        if patience_ctr >= cfg["patience"]:
            print(f"[Protocol B] early stopping at epoch {epoch} (patience={cfg['patience']})")
            break

    pd.DataFrame(log_rows).to_csv(out_dir / "training_log.csv", index=False, encoding="utf-8-sig")
    model.load_state_dict(best_state)  # best-on-(C1+C4)-val checkpoint, C6 never touched until here
    train_time = time.time() - t0

    # Inference timing benchmark on C6 (warm-up + averaged)
    model.eval()
    with torch.no_grad():
        warm_batch = next(iter(test_loader))
        f0, v0, _ = warm_batch
        f0, v0 = f0.to(device), v0.to(device)
        for _ in range(3):
            model(f0, v0)
        if device == "cuda":
            torch.cuda.synchronize()
        t_inf0 = time.time()
        n_samples = 0
        for _ in range(5):
            for force, vib, _ in test_loader:
                force, vib = force.to(device), vib.to(device)
                model(force, vib)
                n_samples += len(force)
        if device == "cuda":
            torch.cuda.synchronize()
        inf_time = time.time() - t_inf0
    ms_per_sample = 1000.0 * inf_time / max(n_samples, 1)

    probs, true_ids = predict(model, test_loader, device)
    pred_df = build_pred_df(test_meta, probs, true_ids)
    metric_row = dcpsr_base.manuscript_metric_row(pred_df, method="raw",
                                                   method_id="Multi-Attention-CNN (Protocol B, Unified)",
                                                   split="test_C6")
    save_outputs(out_dir, pred_df, metric_row, model,
                 extra={"protocol": "B_unified_C1C4_to_C6", "best_epoch": best_epoch,
                        "best_val_acc_C1C4_source_only": best_val_acc,
                        "train_time_s": train_time, "inference_ms_per_sample": ms_per_sample,
                        "inference_samples_per_s": 1000.0 / ms_per_sample if ms_per_sample > 0 else None,
                        "leakage_audit": "C6 used only for this single final evaluation; "
                                         "model/epoch selection used only the C1+C4 internal validation split"})
    torch.save(best_state, out_dir / "checkpoint_best.pth")
    print(f"[Protocol B] test C6: Acc={metric_row['Acc']:.4f} Macro-F1={metric_row['Macro-F1']:.4f}")
    return metric_row


# =========================================================
# Smoke test
# =========================================================
def run_smoke_test(device: str):
    meta = load_metadata().groupby("stage_unified").head(4).reset_index(drop=True)  # ~12 samples
    ds = ImagePairDataset(meta, "stage_unified_id")
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    model = MultiAttentionCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    out_dir = OUT_ROOT / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(3):
        loss, acc = run_epoch(model, loader, device, opt)
        print(f"[smoke] epoch {epoch}: loss={loss:.4f} acc={acc:.4f}")
    torch.save(model.state_dict(), out_dir / "smoke_checkpoint.pth")
    probs, true_ids = predict(model, loader, device)
    pred_df = build_pred_df(meta, probs, true_ids)
    metric_row = dcpsr_base.manuscript_metric_row(pred_df, method="raw", method_id="smoke", split="smoke")
    save_outputs(out_dir, pred_df, metric_row, model, extra={"protocol": "smoke_test"})
    print("[smoke] OK -- pipeline runs end to end.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["A", "B", "smoke"], required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    if args.protocol == "smoke":
        run_smoke_test(args.device)
    elif args.protocol == "A":
        run_protocol_a(args.device, n_seeds=args.n_seeds, epochs=args.epochs)
    elif args.protocol == "B":
        run_protocol_b(args.device, max_epochs=args.epochs)
