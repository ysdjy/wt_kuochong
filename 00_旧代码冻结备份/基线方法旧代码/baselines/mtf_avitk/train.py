# -*- coding: utf-8 -*-
r"""
Training + evaluation for the MTF-AViTK baseline. Two protocols, kept
strictly separate:

Protocol A (original-paper sanity reproduction, B1 split):
    - Labels: paper's own fixed cut-index thresholds (Initial 1-50 /
      Normal 51-175 / Severe 176-315), metadata.csv "stage_original".
    - Split: B1 = train {C1,C4}, test {C6} (Table 4), at SUB-WINDOW
      granularity (5 images/run, matching how the paper's own classifier
      is trained/evaluated -- see PAPER_SPEC.md).
    - Hyperparameters: paper's own Table 5 -- SGD, lr=0.0006, wd=2e-4,
      momentum=0.9, epochs=50, batch_size=8.
    - Output: outputs/mtf_avitk/original_protocol/

Protocol B (Unified DC-PSR comparison, D1 = C1+C4 -> C6):
    - Labels: this project's condition-relative Early/Middle/Late
      (data/label_utils.py, reused byte-for-byte from
      代码/main_experiment_3_fgds_psi_optimized.py).
    - Split: train/val carved from C1+C4 only, test = C6, at sub-window
      level for training; EVALUATION aggregates each run's 5 sub-window
      probability vectors (mean) back to one run-level prediction before
      computing metrics, so results are directly comparable (945 total
      test predictions -> 315 C6-run-level predictions) to every other
      run-level DC-PSR baseline. This aggregation is a reimplementation
      choice (the paper evaluates at image level, not run level) --
      documented in README.md, not a paper-specified step.
    - Model selection (best epoch) uses ONLY the C1+C4 internal
      validation split -- C6 is never touched until the single final
      evaluation (no leakage, task instruction #58/#27).
    - Metrics: reuses 代码/main_experiment_3_fgds_psi_optimized.py's own
      `manuscript_metric_row` / `stage_consistency_metrics`. q-MAE/RMSE/R2
      are N/A (no q-regression head).
    - Output: outputs/mtf_avitk/unified_protocol/

GPU memory (task instruction #68): ViT-L/32 (309M params) on an 8GB card.
AMP (mixed precision) is on by default; --grad-checkpoint and
--grad-accum-steps are available if AMP alone is insufficient. These are
pure memory/compute trade-offs, not changes to the model architecture,
and are recorded in the run's config.json / README.md.

Smoke-test mode (--protocol smoke): a handful of epochs on a tiny subset.
Output: outputs/mtf_avitk/smoke/
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

torch.backends.cudnn.benchmark = True  # fixed input size (384x384) -> autotune is a pure speed win

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))


def safe_torch_save(obj, path, retries: int = 8, delay: float = 3.0):
    """torch.save with retry-on-PermissionError.

    The outputs/ directory lives inside a BaiduSyncdisk-synced folder;
    BaiduSyncdisk briefly locks large files (checkpoints here are 2-4GB)
    while uploading them, which raises a transient
    PermissionError/OSError on the next overwrite. Writes to a temp file
    first and atomically replaces the target, retrying on lock
    collisions -- not a training bug, purely a cloud-sync race condition.
    """
    import os
    import time as _time
    path = str(path)
    tmp_path = path + ".tmp_write"
    last_err = None
    for attempt in range(retries):
        try:
            torch.save(obj, tmp_path)
            os.replace(tmp_path, path)
            return
        except (PermissionError, OSError) as e:
            last_err = e
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            print(f"[safe_torch_save] attempt {attempt+1}/{retries} failed ({e}); "
                  f"retrying in {delay}s (likely BaiduSyncdisk file lock)", flush=True)
            _time.sleep(delay)
    raise RuntimeError(f"safe_torch_save: giving up after {retries} attempts on {path}") from last_err

from model import MTF_AViTK
import label_utils as L
from label_utils import dcpsr_base

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent / "data"
IMG_DIR = DATA_DIR / "images"
OUT_ROOT = PROJECT_ROOT / "outputs" / "mtf_avitk"

# Explicit: PAPER_SPEC.md Table 5
PROTO_A_CFG = dict(epochs=50, batch_size=8, lr=0.0006, weight_decay=2e-4, momentum=0.9, seed=42)
PROTO_B_CFG = dict(max_epochs=50, patience=10, batch_size=8, lr=0.0006, weight_decay=2e-4, momentum=0.9, seed=42)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MTFImageDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, label_col: str):
        self.rows = rows.reset_index(drop=True)
        self.label_col = label_col

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows.iloc[idx]
        img = np.load(IMG_DIR / f"{r.condition}_{int(r.run_id):03d}_{int(r.subwindow)}.npy").astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # [3,H,W]
        label = int(getattr(r, self.label_col))
        return img, label


def build_model(device: str, grad_checkpoint: bool) -> MTF_AViTK:
    model = MTF_AViTK().to(device)
    if grad_checkpoint:
        model.backbone.grad_checkpointing = True
    return model


def run_epoch(model, loader, device, optimizer=None, scaler=None, grad_accum_steps=1):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_loss, n, correct = 0.0, 0, 0
    use_amp = device == "cuda"
    if train_mode:
        optimizer.zero_grad()
    with torch.set_grad_enabled(train_mode):
        for i, (img, label) in enumerate(loader):
            img, label = img.to(device), label.to(device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(img)
                loss = F.cross_entropy(logits, label)
            if train_mode:
                loss_scaled = loss / grad_accum_steps
                if scaler is not None:
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()
                if (i + 1) % grad_accum_steps == 0:
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
            total_loss += loss.item() * len(label)
            correct += (logits.argmax(dim=1) == label).sum().item()
            n += len(label)
    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_probs, all_true = [], []
    use_amp = device == "cuda"
    for img, label in loader:
        img = img.to(device)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            logits = model(img)
        probs = F.softmax(logits.float(), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_true.append(label.numpy())
    return np.concatenate(all_probs), np.concatenate(all_true)


def build_pred_df_subwindow(rows: pd.DataFrame, probs: np.ndarray, true_ids: np.ndarray) -> pd.DataFrame:
    pred_ids = probs.argmax(axis=1)
    return pd.DataFrame({
        "condition": rows["condition"].values,
        "run_id": rows["run_id"].values.astype(int),
        "subwindow": rows["subwindow"].values.astype(int),
        "stage_true_id": true_ids.astype(int),
        "prob_early": probs[:, 0], "prob_middle": probs[:, 1], "prob_late": probs[:, 2],
    })


def aggregate_to_run_level(sw_df: pd.DataFrame) -> pd.DataFrame:
    """Mean-probability aggregation of the 5 sub-window predictions per run
    -> one run-level prediction (reimplementation choice, see module
    docstring)."""
    agg = sw_df.groupby(["condition", "run_id"]).agg(
        stage_true_id=("stage_true_id", "first"),
        prob_early=("prob_early", "mean"),
        prob_middle=("prob_middle", "mean"),
        prob_late=("prob_late", "mean"),
    ).reset_index()
    probs = agg[["prob_early", "prob_middle", "prob_late"]].values
    agg["pred_id"] = probs.argmax(axis=1)
    return agg


def build_pred_df_for_metrics(run_level_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "condition": run_level_df["condition"].values,
        "cut_index": run_level_df["run_id"].values.astype(int),
        "stage_true_id": run_level_df["stage_true_id"].values.astype(int),
        "stage_pred_raw": run_level_df["pred_id"].values.astype(int),
        "raw_prob_early": run_level_df["prob_early"].values,
        "raw_prob_middle": run_level_df["prob_middle"].values,
        "raw_prob_late": run_level_df["prob_late"].values,
        "q_true_model": np.nan,
        "q_hat": np.nan,
    })


def save_outputs(out_dir: Path, pred_df_for_metrics: pd.DataFrame, metric_row: dict, model, extra: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    id_to_stage = dcpsr_base.ID_TO_STAGE
    pred_table = pred_df_for_metrics.copy()
    pred_table["true_stage"] = pred_table["stage_true_id"].map(id_to_stage)
    pred_table["pred_stage"] = pred_table["stage_pred_raw"].map(id_to_stage)
    pred_table = pred_table.rename(columns={"cut_index": "run_id", "raw_prob_early": "p_early",
                                             "raw_prob_middle": "p_middle", "raw_prob_late": "p_late"})
    pred_table = pred_table[["condition", "run_id", "true_stage", "pred_stage", "p_early", "p_middle", "p_late"]]
    pred_table.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(pred_df_for_metrics["stage_true_id"], pred_df_for_metrics["stage_pred_raw"], labels=[0, 1, 2])
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
# Protocol A (B1 split, sub-window granularity)
# =========================================================
def run_protocol_a(device: str, epochs: int | None = None, grad_checkpoint: bool = False,
                    grad_accum_steps: int = 1, batch_size: int | None = None, resume: bool = False):
    cfg = dict(PROTO_A_CFG)
    if epochs is not None:
        cfg["epochs"] = epochs
    if batch_size is not None:
        cfg["batch_size"] = batch_size
    set_seed(cfg["seed"])

    meta = load_metadata()
    train_meta = meta[meta["condition"].isin(["C1", "C4"])].reset_index(drop=True)
    test_meta = meta[meta["condition"] == "C6"].reset_index(drop=True)

    train_loader = DataLoader(MTFImageDataset(train_meta, "stage_original_id"), batch_size=cfg["batch_size"],
                               shuffle=True, num_workers=0)
    test_loader = DataLoader(MTFImageDataset(test_meta, "stage_original_id"), batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=0)

    model = build_model(device, grad_checkpoint)
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    out_dir = OUT_ROOT / "original_protocol"
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_path = out_dir / "epoch_checkpoint.pth"
    log_path = out_dir / "training_log.csv"
    log_rows = []
    start_epoch = 0
    if resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        if log_path.exists():
            log_rows = pd.read_csv(log_path).to_dict("records")
        print(f"[Protocol A/B1] resumed from epoch {start_epoch} ({resume_path})", flush=True)

    t0 = time.time()
    for epoch in range(start_epoch, cfg["epochs"]):
        t_ep = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, device, opt, scaler, grad_accum_steps)
        log_rows.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "time_s": time.time() - t_ep})
        print(f"[Protocol A/B1] epoch {epoch}: train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"({time.time()-t_ep:.1f}s)", flush=True)
        pd.DataFrame(log_rows).to_csv(log_path, index=False, encoding="utf-8-sig")
        cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        safe_torch_save({"model": cpu_state, "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch}, resume_path)
        del cpu_state
        if device == "cuda":
            torch.cuda.empty_cache()
    train_time = time.time() - t0

    probs, true_ids = predict(model, test_loader, device)
    sw_pred_df = build_pred_df_subwindow(test_meta, probs, true_ids)
    sw_pred_df.to_csv(out_dir / "B1_subwindow_predictions.csv", index=False, encoding="utf-8-sig")
    # Paper evaluates per-image; report image-level metrics for the sanity check.
    pred_df = pd.DataFrame({
        "condition": sw_pred_df["condition"], "cut_index": sw_pred_df["run_id"],
        "stage_true_id": sw_pred_df["stage_true_id"], "stage_pred_raw": sw_pred_df[["prob_early","prob_middle","prob_late"]].values.argmax(axis=1),
        "raw_prob_early": sw_pred_df["prob_early"], "raw_prob_middle": sw_pred_df["prob_middle"], "raw_prob_late": sw_pred_df["prob_late"],
        "q_true_model": np.nan, "q_hat": np.nan,
    })
    metric_row = dcpsr_base.manuscript_metric_row(pred_df, method="raw", method_id="MTF-AViTK (Protocol A, B1)",
                                                   split="B1_image_level")
    metric_row["paper_reported_B1_Acc"] = 0.9538
    metric_row["gap_vs_paper"] = metric_row["Acc"] - 0.9538
    save_outputs(out_dir, pred_df, metric_row, model,
                 extra={"protocol": "A_original_B1_split_image_level", "train_time_s": train_time,
                        "grad_checkpoint": grad_checkpoint, "grad_accum_steps": grad_accum_steps,
                        "batch_size": cfg["batch_size"], "amp": device == "cuda"})
    safe_torch_save(model.state_dict(), out_dir / "checkpoint_best.pth")
    print(f"[Protocol A/B1] image-level Acc={metric_row['Acc']:.4f} (paper reports 0.9538, "
          f"gap={metric_row['gap_vs_paper']:+.4f})")
    return metric_row


# =========================================================
# Protocol B (Unified, run-level aggregation)
# =========================================================
def run_protocol_b(device: str, epochs: int | None = None, grad_checkpoint: bool = False,
                    grad_accum_steps: int = 1, batch_size: int | None = None, resume: bool = False):
    cfg = dict(PROTO_B_CFG)
    if epochs is not None:
        cfg["max_epochs"] = epochs
    if batch_size is not None:
        cfg["batch_size"] = batch_size
    set_seed(cfg["seed"])

    meta = load_metadata()
    tr_df, va_df, te_df = L.get_unified_split()

    def attach(split_df):
        return meta.merge(split_df[["condition", "run_id"]], on=["condition", "run_id"], how="inner")

    train_meta, val_meta, test_meta = attach(tr_df), attach(va_df), attach(te_df)
    train_loader = DataLoader(MTFImageDataset(train_meta, "stage_unified_id"), batch_size=cfg["batch_size"],
                               shuffle=True, num_workers=0)
    val_loader = DataLoader(MTFImageDataset(val_meta, "stage_unified_id"), batch_size=cfg["batch_size"],
                             shuffle=False, num_workers=0)
    test_loader = DataLoader(MTFImageDataset(test_meta, "stage_unified_id"), batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=0)

    model = build_model(device, grad_checkpoint)
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    out_dir = OUT_ROOT / "unified_protocol"
    out_dir.mkdir(parents=True, exist_ok=True)

    def run_level_val_acc():
        probs, true_ids = predict(model, val_loader, device)
        sw = build_pred_df_subwindow(val_meta, probs, true_ids)
        run_lvl = aggregate_to_run_level(sw)
        return (run_lvl["pred_id"] == run_lvl["stage_true_id"]).mean()

    resume_path = out_dir / "epoch_checkpoint.pth"
    log_path = out_dir / "training_log.csv"
    best_val_acc, best_epoch, patience_ctr = -1.0, -1, 0
    best_state = None
    log_rows = []
    start_epoch = 0
    if resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc, best_epoch, patience_ctr = ckpt["best_val_acc"], ckpt["best_epoch"], ckpt["patience_ctr"]
        best_state = ckpt["best_state"]
        if log_path.exists():
            log_rows = pd.read_csv(log_path).to_dict("records")
        print(f"[Protocol B] resumed from epoch {start_epoch} (best={best_val_acc:.4f}@{best_epoch})", flush=True)

    t0 = time.time()
    for epoch in range(start_epoch, cfg["max_epochs"]):
        t_ep = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, device, opt, scaler, grad_accum_steps)
        va_acc = run_level_val_acc()
        log_rows.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                          "val_run_level_acc": va_acc, "time_s": time.time() - t_ep})
        improved = va_acc > best_val_acc
        if improved:
            best_val_acc, best_epoch = va_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        print(f"[Protocol B] epoch {epoch}: train_acc={tr_acc:.4f} val_run_acc={va_acc:.4f} "
              f"(best={best_val_acc:.4f}@{best_epoch}) [{time.time()-t_ep:.1f}s]", flush=True)
        pd.DataFrame(log_rows).to_csv(log_path, index=False, encoding="utf-8-sig")
        cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        safe_torch_save({"model": cpu_state, "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_val_acc": best_val_acc, "best_epoch": best_epoch,
                    "patience_ctr": patience_ctr, "best_state": best_state}, resume_path)
        del cpu_state
        if device == "cuda":
            torch.cuda.empty_cache()
        if patience_ctr >= cfg["patience"]:
            print(f"[Protocol B] early stopping at epoch {epoch}", flush=True)
            break
    model.load_state_dict(best_state)
    model.to(device)
    train_time = time.time() - t0

    # Inference timing on C6 (warm-up + averaged), sub-window level
    model.eval()
    with torch.no_grad():
        warm_img, _ = next(iter(test_loader))
        warm_img = warm_img.to(device)
        for _ in range(3):
            model(warm_img)
        if device == "cuda":
            torch.cuda.synchronize()
        t_inf0 = time.time()
        n_samples = 0
        for _ in range(3):
            for img, _ in test_loader:
                img = img.to(device)
                model(img)
                n_samples += len(img)
        if device == "cuda":
            torch.cuda.synchronize()
        inf_time = time.time() - t_inf0
    ms_per_sample = 1000.0 * inf_time / max(n_samples, 1)

    probs, true_ids = predict(model, test_loader, device)
    sw_pred_df = build_pred_df_subwindow(test_meta, probs, true_ids)
    sw_pred_df.to_csv(out_dir / "test_subwindow_predictions.csv", index=False, encoding="utf-8-sig")
    run_lvl = aggregate_to_run_level(sw_pred_df)
    pred_df = build_pred_df_for_metrics(run_lvl)
    metric_row = dcpsr_base.manuscript_metric_row(pred_df, method="raw",
                                                   method_id="MTF-AViTK (Protocol B, Unified)", split="test_C6")
    save_outputs(out_dir, pred_df, metric_row, model,
                 extra={"protocol": "B_unified_C1C4_to_C6_run_level_aggregated", "best_epoch": best_epoch,
                        "best_val_acc_C1C4_source_only_run_level": best_val_acc, "train_time_s": train_time,
                        "inference_ms_per_sample_subwindow": ms_per_sample,
                        "inference_samples_per_s_subwindow": 1000.0 / ms_per_sample if ms_per_sample > 0 else None,
                        "grad_checkpoint": grad_checkpoint, "grad_accum_steps": grad_accum_steps,
                        "batch_size": cfg["batch_size"], "amp": device == "cuda",
                        "leakage_audit": "C6 used only for this single final evaluation; "
                                         "model/epoch selection used only the C1+C4 internal validation split "
                                         "(run-level aggregated)"})
    safe_torch_save(best_state, out_dir / "checkpoint_best.pth")
    print(f"[Protocol B] test C6 (run-level): Acc={metric_row['Acc']:.4f} Macro-F1={metric_row['Macro-F1']:.4f}")
    return metric_row


# =========================================================
# Smoke test
# =========================================================
def run_smoke_test(device: str, grad_checkpoint: bool = False):
    meta = load_metadata()
    small = meta.groupby(["condition", "stage_unified"]).head(1).reset_index(drop=True)  # small subset
    ds = MTFImageDataset(small, "stage_unified_id")
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    model = build_model(device, grad_checkpoint)
    opt = torch.optim.SGD(model.parameters(), lr=0.0006, momentum=0.9, weight_decay=2e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    out_dir = OUT_ROOT / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(2):
        loss, acc = run_epoch(model, loader, device, opt, scaler)
        print(f"[smoke] epoch {epoch}: loss={loss:.4f} acc={acc:.4f}")
    safe_torch_save(model.state_dict(), out_dir / "smoke_checkpoint.pth")
    probs, true_ids = predict(model, loader, device)
    sw_pred_df = build_pred_df_subwindow(small, probs, true_ids)
    run_lvl = aggregate_to_run_level(sw_pred_df)
    pred_df = build_pred_df_for_metrics(run_lvl)
    metric_row = dcpsr_base.manuscript_metric_row(pred_df, method="raw", method_id="smoke", split="smoke")
    save_outputs(out_dir, pred_df, metric_row, model, extra={"protocol": "smoke_test"})
    print("[smoke] OK -- pipeline runs end to end.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["A", "B", "smoke"], required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="resume from outputs/.../epoch_checkpoint.pth if present")
    args = parser.parse_args()

    if args.protocol == "smoke":
        run_smoke_test(args.device, args.grad_checkpoint)
    elif args.protocol == "A":
        run_protocol_a(args.device, epochs=args.epochs, grad_checkpoint=args.grad_checkpoint,
                        grad_accum_steps=args.grad_accum_steps, batch_size=args.batch_size, resume=args.resume)
    elif args.protocol == "B":
        run_protocol_b(args.device, epochs=args.epochs, grad_checkpoint=args.grad_checkpoint,
                        grad_accum_steps=args.grad_accum_steps, batch_size=args.batch_size, resume=args.resume)
