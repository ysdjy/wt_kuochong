# -*- coding: utf-8 -*-
r"""
Training + evaluation for the Dynamic GIN + TGP baseline. Two protocols,
kept strictly separate (see PAPER_SPEC.md and README.md):

Protocol A (original-paper sanity reproduction, D1 split):
    - Labels: paper's own fixed cut-index thresholds (Initial 1-50 /
      Normal 51-210 / Severe 211-315), first 300 passes only.
    - Split: D1 = train/val {C1,C4} (0.7:0.3), test {C6} (Table 3).
    - Sampling: stratified 5/2/3 samples per run per stage (Sec 3.2).
    - Hyperparameters: paper's own Sec 3.4 Scheme 1 -- Adam, lr=1e-4,
      weight_decay(L2)=0.1, batch=4, epochs=50,
      ReduceLROnPlateau(factor=0.5, patience=10).
    - Output: outputs/dynamic_gin_tgp/original_protocol/

Protocol B (Unified DC-PSR comparison, D1 = C1+C4 -> C6):
    - Labels: condition-relative Early/Middle/Late
      (data/label_utils.py, reused byte-for-byte from
      代码/main_experiment_3_fgds_psi_optimized.py).
    - Split: train/val carved from C1+C4 only (label_utils split), test
      = C6 (all runs in the common DC-PSR universe, up to 315 -- NOT
      truncated to 300, per task instruction #39: Protocol A's own
      300-pass truncation is paper-native and must not silently become
      the unified benchmark's test universe).
    - Segment granularity: every one of a run's 10 stable-region
      portions is used as a training/eval segment (see preprocessing.py);
      evaluation aggregates each run's 10 segment-level probability
      vectors (mean) back to one run-level prediction before computing
      metrics (task instruction #14/#40).
    - Model selection (best epoch) uses ONLY the C1+C4 internal
      validation split -- C6 is never touched until the single final
      evaluation (no leakage, task instructions #39/#68).
    - Metrics: reuses 代码/main_experiment_3_fgds_psi_optimized.py's own
      `manuscript_metric_row` / `stage_consistency_metrics` via
      data/label_utils.py's `dcpsr_base`. q-MAE/RMSE/R2 are N/A (no
      q-regression head).
    - Output: outputs/dynamic_gin_tgp/unified_protocol/

GPU memory: this architecture's Conv2d_4 (288 output channels on a
~282x282 GASF-derived spatial map, Table 1) is memory-heavy even at the
paper's own batch=4 -- confirmed OOM at batch=16 on an 8GB card during
CPU-side development. AMP is on by default when running on CUDA.

Smoke-test mode (--protocol smoke): a handful of epochs on a tiny subset.
Output: outputs/dynamic_gin_tgp/smoke/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

import preprocessing as P  # noqa: E402
from model import DynamicGIN_TGP  # noqa: E402
import label_utils as L  # noqa: E402
from label_utils import dcpsr_base  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
OUT_ROOT = PROJECT_ROOT / "outputs" / "dynamic_gin_tgp"

# Explicit: PAPER_SPEC.md sec 8 (Sec 3.4 Scheme 1)
PROTO_A_CFG = dict(epochs=50, batch_size=4, lr=1e-4, weight_decay=0.1, seed=42, plateau_patience=10)
PROTO_B_CFG = dict(max_epochs=50, patience=15, batch_size=4, lr=1e-4, weight_decay=0.1, seed=42,
                    plateau_patience=10)


def safe_torch_save(obj, path, retries: int = 8, delay: float = 3.0):
    """torch.save with retry-on-PermissionError (BaiduSyncdisk file-lock
    races on large checkpoint files -- same pattern as
    baselines/mtf_avitk/train.py)."""
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
            print(f"[safe_torch_save] attempt {attempt+1}/{retries} failed ({e}); retrying in {delay}s",
                  flush=True)
            _time.sleep(delay)
    raise RuntimeError(f"safe_torch_save: giving up after {retries} attempts on {path}") from last_err


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class ProtocolADataset(Dataset):
    """Rows: condition, run_id, stage_original, portion_idx.

    IMPORTANT: rows are shuffled (fixed seed) at construction time,
    regardless of the input manifest's own order. Reason: the model's
    static graph (Eq.7-9, model.py::build_static_graph) is built by
    CONCATENATING every sample in a batch before computing cosine
    similarity -- i.e. a sample's prediction depends on which other
    samples share its batch (confirmed directly: feeding the same
    sample with two different sets of "batch-mates" gives two different
    predictions). Any manifest/loader that groups a run's 10 portions
    into consecutive rows (e.g. build_protocol_a_manifest's per-run
    ordering) combined with shuffle=False (needed at eval time so
    predictions align with their metadata row-for-row) would make every
    evaluation batch homogeneous -- all 4 samples from the SAME run,
    hence the SAME true label. That lets the static-graph mechanism leak
    the true label into each sample's own prediction through its
    batch-mates, producing artificially inflated (meaningless) eval
    accuracy. This was caught from a real training run: Protocol B's
    val_run_acc jumped to 100% by epoch 1 while train_acc was still
    ~67% -- far faster than every other baseline in this project using
    the identical data split (compare outputs/mtf_avitk/unified_protocol/
    training_log.csv's much more gradual climb). Shuffling once here
    (not per-DataLoader-epoch, so evaluation order/results stay
    reproducible across repeated calls) breaks the run-homogeneity
    shortcut for both Protocol A's val/test loaders and Protocol B's."""

    def __init__(self, manifest: pd.DataFrame, shuffle_seed: int = 12345):
        self.manifest = manifest.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        r = self.manifest.iloc[idx]
        windows = P.load_windows(r.condition, int(r.run_id))  # [10,6,288]
        x = windows[int(r.portion_idx)]
        y = P.STAGE_A_TO_ID[r.stage_original]
        return torch.from_numpy(x), y


class ProtocolBSegmentDataset(Dataset):
    """One row per (condition, run_id) x 10 portions. `label_df` supplies
    condition/run_id/stage_id for the runs in this split.

    Rows are shuffled (fixed seed) at construction time -- see
    ProtocolADataset's docstring for why this matters: without it, every
    eval-time batch (built with shuffle=False so predictions stay
    row-aligned with their metadata) would contain only consecutive
    portions of a single run, letting the batch-dependent static-graph
    mechanism (Eq.7-9) leak the true label into each sample's own
    prediction via its batch-mates."""

    def __init__(self, label_df: pd.DataFrame, shuffle_seed: int = 12345):
        rows = []
        for _, r in label_df.iterrows():
            for portion_idx in range(10):
                rows.append({"condition": r["condition"], "run_id": int(r["run_id"]),
                             "stage_id": int(r["stage_id"]), "portion_idx": portion_idx})
        self.rows = pd.DataFrame(rows).sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows.iloc[idx]
        windows = P.load_windows(r.condition, int(r.run_id))
        x = windows[int(r.portion_idx)]
        return torch.from_numpy(x), int(r.stage_id)


def make_loader(ds, batch_size, shuffle):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def run_epoch(model, loader, device, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_loss, n, correct = 0.0, 0, 0
    with torch.set_grad_enabled(train_mode):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train_mode:
                optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            if train_mode:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, trues = [], []
    for x, y in loader:
        x = x.to(device)
        p = F.softmax(model(x), dim=1).cpu().numpy()
        probs.append(p)
        trues.append(np.asarray(y))
    return np.concatenate(probs), np.concatenate(trues)


def aggregate_run_level(rows_df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    df = rows_df.copy().reset_index(drop=True)
    df["p_early"], df["p_middle"], df["p_late"] = probs[:, 0], probs[:, 1], probs[:, 2]
    agg = df.groupby(["condition", "run_id"]).agg(
        stage_id=("stage_id", "first"),
        p_early=("p_early", "mean"), p_middle=("p_middle", "mean"), p_late=("p_late", "mean"),
    ).reset_index()
    agg["pred_id"] = agg[["p_early", "p_middle", "p_late"]].values.argmax(axis=1)
    return agg


def build_pred_df_for_metrics(run_level_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "condition": run_level_df["condition"].values,
        "cut_index": run_level_df["run_id"].values.astype(int),
        "stage_true_id": run_level_df["stage_id"].values.astype(int),
        "stage_pred_raw": run_level_df["pred_id"].values.astype(int),
        "raw_prob_early": run_level_df["p_early"].values,
        "raw_prob_middle": run_level_df["p_middle"].values,
        "raw_prob_late": run_level_df["p_late"].values,
        "q_true_model": np.nan, "q_hat": np.nan,
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
    pred_table.to_csv(out_dir / "run_predictions.csv", index=False, encoding="utf-8-sig")

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(pred_df_for_metrics["stage_true_id"], pred_df_for_metrics["stage_pred_raw"], labels=[0, 1, 2])
    cm_df = pd.DataFrame(cm, index=["true_early", "true_middle", "true_late"],
                          columns=["pred_early", "pred_middle", "pred_late"])
    cm_df.to_csv(out_dir / "confusion_matrix.csv", encoding="utf-8-sig")

    row = dict(metric_row)
    row["n_params"] = model.num_parameters()
    row["n_trainable_params"] = model.num_parameters(trainable_only=True)
    row.update(extra)
    pd.DataFrame([row]).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in row.items()},
                   f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Protocol A: D1 split (paper-native)
# ---------------------------------------------------------------------------
def run_protocol_a(device: str, epochs: int | None = None, batch_size: int | None = None, resume: bool = False):
    cfg = dict(PROTO_A_CFG)
    if epochs is not None:
        cfg["epochs"] = epochs
    if batch_size is not None:
        cfg["batch_size"] = batch_size
    set_seed(cfg["seed"])

    manifest_c1 = P.build_protocol_a_manifest("C1")
    manifest_c4 = P.build_protocol_a_manifest("C4")
    manifest_c6 = P.build_protocol_a_manifest("C6")
    source = pd.concat([manifest_c1, manifest_c4], ignore_index=True)

    rng = np.random.default_rng(cfg["seed"])
    idx = rng.permutation(len(source))
    n_val = int(round(0.3 * len(source)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_manifest = source.iloc[train_idx].reset_index(drop=True)
    val_manifest = source.iloc[val_idx].reset_index(drop=True)

    train_loader = make_loader(ProtocolADataset(train_manifest), cfg["batch_size"], True)
    val_loader = make_loader(ProtocolADataset(val_manifest), cfg["batch_size"], False)
    test_loader = make_loader(ProtocolADataset(manifest_c6), cfg["batch_size"], False)

    model = DynamicGIN_TGP(topk=144).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                             patience=cfg["plateau_patience"])

    out_dir = OUT_ROOT / "original_protocol"
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_path = out_dir / "epoch_checkpoint.pth"
    log_path = out_dir / "training_log.csv"
    log_rows, start_epoch = [], 0
    best_val_acc, best_state = -1.0, None
    if resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc, best_state = ckpt["best_val_acc"], ckpt["best_state"]
        if log_path.exists():
            log_rows = pd.read_csv(log_path).to_dict("records")
        print(f"[Protocol A/D1] resumed from epoch {start_epoch} (best_val_acc={best_val_acc:.4f})", flush=True)

    t0 = time.time()
    for epoch in range(start_epoch, cfg["epochs"]):
        t_ep = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, device, opt)
        va_loss, va_acc = run_epoch(model, val_loader, device, None)
        scheduler.step(va_acc)
        improved = va_acc > best_val_acc
        if improved:
            best_val_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log_rows.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                          "val_loss": va_loss, "val_acc": va_acc, "time_s": time.time() - t_ep,
                          "lr": opt.param_groups[0]["lr"]})
        print(f"[Protocol A/D1] epoch {epoch}: train_acc={tr_acc:.4f} val_acc={va_acc:.4f} "
              f"(best={best_val_acc:.4f}) [{time.time()-t_ep:.1f}s]", flush=True)
        pd.DataFrame(log_rows).to_csv(log_path, index=False, encoding="utf-8-sig")
        safe_torch_save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                          "optimizer": opt.state_dict(), "scheduler": scheduler.state_dict(),
                          "epoch": epoch, "best_val_acc": best_val_acc, "best_state": best_state}, resume_path)
    model.load_state_dict(best_state)
    model.to(device)
    train_time = time.time() - t0

    probs, trues = predict(model, test_loader, device)
    pred_ids = probs.argmax(axis=1)
    pred_df = pd.DataFrame({
        "condition": "C6", "cut_index": np.arange(len(trues)),
        "stage_true_id": trues, "stage_pred_raw": pred_ids,
        "raw_prob_early": probs[:, 0], "raw_prob_middle": probs[:, 1], "raw_prob_late": probs[:, 2],
        "q_true_model": np.nan, "q_hat": np.nan,
    })
    metric_row = dcpsr_base.manuscript_metric_row(pred_df, method="raw",
                                                   method_id="Dynamic GIN+TGP (Protocol A, D1)", split="D1_C6")
    metric_row["paper_reported_D1_Acc"] = 0.9571
    metric_row["gap_vs_paper"] = metric_row["Acc"] - 0.9571
    save_outputs(out_dir, pred_df, metric_row, model,
                 extra={"protocol": "A_original_D1_split_sample_level", "train_time_s": train_time,
                        "batch_size": cfg["batch_size"], "topk": 144})
    safe_torch_save(model.state_dict(), out_dir / "checkpoint_best.pth")
    (out_dir / "DONE.flag").write_text(f"done at epoch {cfg['epochs']}, Acc={metric_row['Acc']:.4f}\n")
    print(f"[Protocol A/D1] sample-level Acc={metric_row['Acc']:.4f} "
          f"(paper reports 0.9571, gap={metric_row['gap_vs_paper']:+.4f})")
    return metric_row


# ---------------------------------------------------------------------------
# Protocol B: Unified (C1+C4 -> C6), run-level aggregation
# ---------------------------------------------------------------------------
def run_protocol_b(device: str, epochs: int | None = None, batch_size: int | None = None, resume: bool = False,
                    seed: int | None = None):
    cfg = dict(PROTO_B_CFG)
    if epochs is not None:
        cfg["max_epochs"] = epochs
    if batch_size is not None:
        cfg["batch_size"] = batch_size
    if seed is not None:
        cfg["seed"] = seed
    set_seed(cfg["seed"])

    tr_df, va_df, te_df = L.get_unified_split()

    train_loader = make_loader(ProtocolBSegmentDataset(tr_df), cfg["batch_size"], True)
    val_ds = ProtocolBSegmentDataset(va_df)
    val_loader = make_loader(val_ds, cfg["batch_size"], False)
    test_ds = ProtocolBSegmentDataset(te_df)
    test_loader = make_loader(test_ds, cfg["batch_size"], False)

    model = DynamicGIN_TGP(topk=144).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                             patience=cfg["plateau_patience"])

    out_dir = OUT_ROOT / "unified_protocol" / f"seed{cfg['seed']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_path = out_dir / "epoch_checkpoint.pth"
    log_path = out_dir / "training_log.csv"
    log_rows, start_epoch, patience_ctr = [], 0, 0
    best_val_acc, best_epoch, best_state = -1.0, -1, None
    if resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc, best_epoch, patience_ctr = ckpt["best_val_acc"], ckpt["best_epoch"], ckpt["patience_ctr"]
        best_state = ckpt["best_state"]
        if log_path.exists():
            log_rows = pd.read_csv(log_path).to_dict("records")
        print(f"[Protocol B] resumed from epoch {start_epoch} (best={best_val_acc:.4f}@{best_epoch})", flush=True)

    def run_level_val_acc():
        probs, _ = predict(model, val_loader, device)
        agg = aggregate_run_level(val_ds.rows, probs)
        return (agg["pred_id"] == agg["stage_id"]).mean()

    t0 = time.time()
    for epoch in range(start_epoch, cfg["max_epochs"]):
        t_ep = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, device, opt)
        va_acc = run_level_val_acc()
        scheduler.step(va_acc)
        improved = va_acc > best_val_acc
        if improved:
            best_val_acc, best_epoch, patience_ctr = va_acc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
        log_rows.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                          "val_run_level_acc": va_acc, "time_s": time.time() - t_ep,
                          "lr": opt.param_groups[0]["lr"]})
        print(f"[Protocol B] epoch {epoch}: train_acc={tr_acc:.4f} val_run_acc={va_acc:.4f} "
              f"(best={best_val_acc:.4f}@{best_epoch}) [{time.time()-t_ep:.1f}s]", flush=True)
        pd.DataFrame(log_rows).to_csv(log_path, index=False, encoding="utf-8-sig")
        safe_torch_save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                          "optimizer": opt.state_dict(), "scheduler": scheduler.state_dict(),
                          "epoch": epoch, "best_val_acc": best_val_acc, "best_epoch": best_epoch,
                          "patience_ctr": patience_ctr, "best_state": best_state}, resume_path)
        if patience_ctr >= cfg["patience"]:
            print(f"[Protocol B] early stopping at epoch {epoch}", flush=True)
            break
    model.load_state_dict(best_state)
    model.to(device)
    train_time = time.time() - t0

    probs, _ = predict(model, test_loader, device)
    run_lvl = aggregate_run_level(test_ds.rows, probs)
    pred_df = build_pred_df_for_metrics(run_lvl)
    metric_row = dcpsr_base.manuscript_metric_row(pred_df, method="raw",
                                                   method_id="Dynamic GIN+TGP (Protocol B, Unified)", split="test_C6")
    save_outputs(out_dir, pred_df, metric_row, model,
                 extra={"protocol": "B_unified_C1C4_to_C6_run_level_aggregated", "best_epoch": best_epoch,
                        "best_val_acc_C1C4_source_only_run_level": best_val_acc, "train_time_s": train_time,
                        "seed": cfg["seed"], "topk": 144,
                        "leakage_audit": "C6 used only for this single final evaluation; model/epoch "
                                         "selection used only the C1+C4 internal validation split "
                                         "(run-level aggregated)"})
    safe_torch_save(best_state, out_dir / "checkpoint_best.pth")
    (out_dir / "DONE.flag").write_text(f"done, best_epoch={best_epoch}, Acc={metric_row['Acc']:.4f}\n")
    print(f"[Protocol B] test C6 (run-level, seed={cfg['seed']}): Acc={metric_row['Acc']:.4f} "
          f"Macro-F1={metric_row['Macro-F1']:.4f}")
    return metric_row


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def run_smoke_test(device: str):
    manifest = P.build_protocol_a_manifest("C1").groupby("stage_original").head(4).reset_index(drop=True)
    loader = make_loader(ProtocolADataset(manifest), batch_size=4, shuffle=True)
    model = DynamicGIN_TGP(topk=144).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.1)
    out_dir = OUT_ROOT / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(2):
        loss, acc = run_epoch(model, loader, device, opt)
        print(f"[smoke] epoch {epoch}: loss={loss:.4f} acc={acc:.4f}")
    safe_torch_save(model.state_dict(), out_dir / "smoke_checkpoint.pth")
    probs, trues = predict(model, loader, device)
    pred_df = pd.DataFrame({
        "condition": "C1", "cut_index": np.arange(len(trues)), "stage_true_id": trues,
        "stage_pred_raw": probs.argmax(1), "raw_prob_early": probs[:, 0],
        "raw_prob_middle": probs[:, 1], "raw_prob_late": probs[:, 2],
        "q_true_model": np.nan, "q_hat": np.nan,
    })
    metric_row = dcpsr_base.manuscript_metric_row(pred_df, method="raw", method_id="smoke", split="smoke")
    save_outputs(out_dir, pred_df, metric_row, model, extra={"protocol": "smoke_test"})
    print("[smoke] OK -- pipeline runs end to end.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["A", "B", "smoke"], required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.protocol == "smoke":
        run_smoke_test(args.device)
    elif args.protocol == "A":
        run_protocol_a(args.device, epochs=args.epochs, batch_size=args.batch_size, resume=args.resume)
    elif args.protocol == "B":
        run_protocol_b(args.device, epochs=args.epochs, batch_size=args.batch_size, resume=args.resume,
                        seed=args.seed)
