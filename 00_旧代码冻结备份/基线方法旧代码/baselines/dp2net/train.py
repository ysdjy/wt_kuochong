# -*- coding: utf-8 -*-
r"""
Training + evaluation for the DP2Net baseline. Algorithm 1's two-stage
protocol (pretrain S+F, then train G+F with S frozen), run for three
distinct splits (see PAPER_SPEC.md and README.md):

Protocol A (original-paper sanity reproduction):
    - Labels: paper-native stage4 (I/II/III; IV empirically absent for
      C1/C4/C6 in this project's archive -- PAPER_SPEC.md sec 6b).
    - Split: Source=C1 (70/30 train/val), Target test = C4 and C6
      separately (paper's own Task 1/Task 2, Table 2).
    - Hyperparameters: paper's own Sec 4.3 -- batch=64, Adam lr=1e-3 for
      S/F (F's lr cosine-annealed, period 20 epochs), Adam lr=1e-5 for G,
      alpha=20, kpool=4, k=25. 100+100 epochs (Algorithm 1).
    - Output: outputs/dp2net/original_protocol/

Protocol B-S (Unified, native single-source, C1 -> C6):
    - Labels: DC-PSR condition-relative E/M/L (data/label_utils.py).
    - Split: get_single_source_split("C1","C6") -- preserves DP2Net's
      own SSDG character (source-only training, target never seen).
    - Output: outputs/dp2net/unified_protocol_B-S/

Protocol B-D1 (Unified, pooled-source adapted, C1+C4 -> C6):
    - Labels: DC-PSR condition-relative E/M/L.
    - Split: get_unified_split() (pooled C1+C4 source, C6 target/test) --
      named "DP2Net-adapted (pooled source)", NOT "original DP2Net"
      (task instruction #64). This is what enters the DC-PSR D1 main
      comparison table.
    - Output: outputs/dp2net/unified_protocol_B-D1/

Both B-S and B-D1 use `data/unified_windows` (preprocessing.py's
`build_unified_windows_cache`, 8 windows/run) and aggregate each run's
window-level probabilities (mean) to one run-level prediction before
computing metrics, exactly like every other baseline in this project.

Smoke-test mode (--protocol smoke): 2+2 epochs on a tiny subset.
Output: outputs/dp2net/smoke/
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

import preprocessing as PP  # noqa: E402
from model import SpatialAttention, Generator, WDCNN, mmd_loss  # noqa: E402
import label_utils as L  # noqa: E402
from label_utils import dcpsr_base  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
OUT_ROOT = PROJECT_ROOT / "outputs" / "dp2net"

# Explicit: PAPER_SPEC.md sec 5 (Sec 4.3)
CFG = dict(pretrain_epochs=100, gen_epochs=100, batch_size=64, lr_s=1e-3, lr_f=1e-3,
           lr_g=1e-5, cosine_period=20, alpha=20.0, seed=42)


def safe_torch_save(obj, path, retries: int = 8, delay: float = 3.0):
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
class ProtocolAWindowDataset(Dataset):
    """Rows from data/sample_manifest.csv (condition,run_id,class,class_id,npy_path)."""

    def __init__(self, manifest: pd.DataFrame):
        self.manifest = manifest.reset_index(drop=True)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        r = self.manifest.iloc[idx]
        x = np.load(THIS_DIR / r.npy_path).astype(np.float32)
        return torch.from_numpy(x).unsqueeze(0), int(r.class_id)


class UnifiedWindowDataset(Dataset):
    """Rows from data/unified_manifest.csv, joined with a label table
    (condition,run_id,stage_id) built by label_utils for the given split."""

    def __init__(self, unified_manifest: pd.DataFrame, label_df: pd.DataFrame):
        merged = unified_manifest.merge(
            label_df[["condition", "run_id", "stage_id"]], on=["condition", "run_id"], how="inner")
        self.rows = merged.reset_index(drop=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows.iloc[idx]
        x = np.load(THIS_DIR / r.npy_path).astype(np.float32)
        return torch.from_numpy(x).unsqueeze(0), int(r.stage_id)


def make_loader(ds, batch_size, shuffle):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# ---------------------------------------------------------------------------
# Algorithm 1: two-stage training
# ---------------------------------------------------------------------------
def pretrain_s_f(s, f, train_loader, val_loader, device, epochs, lr_s, lr_f, cosine_period,
                  log_rows, resume_state=None):
    s.to(device); f.to(device)
    opt = torch.optim.Adam(list(s.parameters()) + list(f.parameters()), lr=lr_s)
    # F's own lr cosine-annealed (paper: "the learning rate's cosine attenuation strategy
    # was applied to the training process of the feature extraction module"); S shares
    # the same optimizer/lr per Sec 4.3's single "S/F Adam learning rate = 0.001" line.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cosine_period)
    start_epoch = 0
    if resume_state is not None:
        s.load_state_dict(resume_state["s"])
        f.load_state_dict(resume_state["f"])
        opt.load_state_dict(resume_state["opt"])
        scheduler.load_state_dict(resume_state["scheduler"])
        start_epoch = resume_state["epoch"] + 1
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        s.train(); f.train()
        total_loss, n, correct = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            fa, _ = s(x)
            logits = f(fa)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        scheduler.step()
        va_acc = eval_s_f_acc(s, f, val_loader, device) if val_loader is not None else float("nan")
        log_rows.append({"stage": "pretrain", "epoch": epoch, "train_loss": total_loss / max(n, 1),
                          "train_acc": correct / max(n, 1), "val_acc": va_acc, "time_s": time.time() - t0})
        print(f"[Stage1 pretrain S+F] epoch {epoch}: loss={total_loss/max(n,1):.4f} "
              f"train_acc={correct/max(n,1):.4f} val_acc={va_acc:.4f} [{time.time()-t0:.1f}s]", flush=True)
        yield {"s": {k: v.cpu() for k, v in s.state_dict().items()},
               "f": {k: v.cpu() for k, v in f.state_dict().items()},
               "opt": opt.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch}


@torch.no_grad()
def eval_s_f_acc(s, f, loader, device):
    s.eval(); f.eval()
    correct, n = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        fa, _ = s(x)
        pred = f(fa).argmax(1)
        correct += (pred == y).sum().item()
        n += len(y)
    return correct / max(n, 1)


def train_g_f(s, g, f, train_loader, val_loader, device, epochs, lr_g, lr_f, alpha,
               log_rows, resume_state=None):
    s.to(device); g.to(device); f.to(device)
    for p in s.parameters():
        p.requires_grad_(False)
    vst = torch.from_numpy(PP.build_vst()).float().unsqueeze(0).to(device)  # [1,4608]
    opt_g = torch.optim.Adam(g.parameters(), lr=lr_g)
    opt_f = torch.optim.Adam(f.parameters(), lr=lr_f)
    start_epoch = 0
    if resume_state is not None:
        g.load_state_dict(resume_state["g"])
        f.load_state_dict(resume_state["f"])
        opt_g.load_state_dict(resume_state["opt_g"])
        opt_f.load_state_dict(resume_state["opt_f"])
        start_epoch = resume_state["epoch"] + 1
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        g.train(); f.train(); s.eval()
        total_lg, total_ltask, n = 0.0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                fa, _ = s(x)
            # --- optimize G: L_G = L_MSE - alpha*L_MMD (Eq.10) ---
            wg = g(fa)
            l_mse = F.mse_loss(wg, vst.expand_as(wg))
            fg = wg * x  # Fg = Wg * raw input (Eq.7 surrounding text, Sec 3.2)
            with torch.no_grad():
                _, emb_s = f(fa, return_features=True)
            _, emb_g = f(fg, return_features=True)
            l_mmd = mmd_loss(emb_s, emb_g)
            l_g = l_mse - alpha * l_mmd
            opt_g.zero_grad()
            l_g.backward()
            opt_g.step()

            # --- optimize F: L_task = CE(source) + CE(generated) (Eq.11) ---
            with torch.no_grad():
                fa2, _ = s(x)
                wg2 = g(fa2)
                fg2 = wg2 * x
            logits_s = f(fa2)
            logits_g = f(fg2)
            l_task = F.cross_entropy(logits_s, y) + F.cross_entropy(logits_g, y)
            opt_f.zero_grad()
            l_task.backward()
            opt_f.step()

            total_lg += l_g.item() * len(y)
            total_ltask += l_task.item() * len(y)
            n += len(y)
        va_acc = eval_s_f_acc(s, f, val_loader, device) if val_loader is not None else float("nan")
        log_rows.append({"stage": "generalize", "epoch": epoch, "l_g": total_lg / max(n, 1),
                          "l_task": total_ltask / max(n, 1), "val_acc": va_acc, "time_s": time.time() - t0})
        print(f"[Stage2 train G+F] epoch {epoch}: l_g={total_lg/max(n,1):.4f} "
              f"l_task={total_ltask/max(n,1):.4f} val_acc={va_acc:.4f} [{time.time()-t0:.1f}s]", flush=True)
        yield {"g": {k: v.cpu() for k, v in g.state_dict().items()},
               "f": {k: v.cpu() for k, v in f.state_dict().items()},
               "opt_g": opt_g.state_dict(), "opt_f": opt_f.state_dict(), "epoch": epoch}


@torch.no_grad()
def predict_sample_level(s, f, loader, device):
    """Inference per Algorithm 1: trained S and F only (G is training-time-only)."""
    s.eval(); f.eval()
    probs, trues = [], []
    for x, y in loader:
        x = x.to(device)
        fa, _ = s(x)
        p = F.softmax(f(fa), dim=1).cpu().numpy()
        probs.append(p)
        trues.append(np.asarray(y))
    return np.concatenate(probs), np.concatenate(trues)


def aggregate_run_level(rows_df: pd.DataFrame, probs: np.ndarray, label_col: str) -> pd.DataFrame:
    df = rows_df.copy().reset_index(drop=True)
    df["p_early"], df["p_middle"], df["p_late"] = probs[:, 0], probs[:, 1], probs[:, 2]
    agg = df.groupby(["condition", "run_id"]).agg(
        stage_id=(label_col, "first"),
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


def save_outputs(out_dir: Path, pred_df_for_metrics: pd.DataFrame, metric_row: dict, s, g, f, extra: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    id_to_stage = dcpsr_base.ID_TO_STAGE
    pred_table = pred_df_for_metrics.copy()
    pred_table["true_stage"] = pred_table["stage_true_id"].map(lambda i: id_to_stage.get(i, str(i)))
    pred_table["pred_stage"] = pred_table["stage_pred_raw"].map(lambda i: id_to_stage.get(i, str(i)))
    pred_table = pred_table.rename(columns={"cut_index": "run_id", "raw_prob_early": "p_early",
                                             "raw_prob_middle": "p_middle", "raw_prob_late": "p_late"})
    pred_table = pred_table[["condition", "run_id", "true_stage", "pred_stage", "p_early", "p_middle", "p_late"]]
    pred_table.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    n_params = sum(p.numel() for p in s.parameters()) + sum(p.numel() for p in g.parameters()) + \
        sum(p.numel() for p in f.parameters())
    row = dict(metric_row)
    row["n_params_total_SGF"] = n_params
    row.update(extra)
    pd.DataFrame([row]).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in row.items()},
                   fp, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Protocol A
# ---------------------------------------------------------------------------
def run_protocol_a(device: str, pretrain_epochs=None, gen_epochs=None, batch_size=None, resume=False):
    cfg = dict(CFG)
    if pretrain_epochs is not None:
        cfg["pretrain_epochs"] = pretrain_epochs
    if gen_epochs is not None:
        cfg["gen_epochs"] = gen_epochs
    if batch_size is not None:
        cfg["batch_size"] = batch_size
    set_seed(cfg["seed"])

    manifest = pd.read_csv(THIS_DIR / "data" / "sample_manifest.csv")
    c1 = manifest[manifest.condition == "C1"].reset_index(drop=True)
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.permutation(len(c1))
    n_val = int(round(0.3 * len(c1)))
    val_m, train_m = c1.iloc[idx[:n_val]], c1.iloc[idx[n_val:]]
    c4_m = manifest[manifest.condition == "C4"].reset_index(drop=True)
    c6_m = manifest[manifest.condition == "C6"].reset_index(drop=True)

    train_loader = make_loader(ProtocolAWindowDataset(train_m), cfg["batch_size"], True)
    val_loader = make_loader(ProtocolAWindowDataset(val_m), cfg["batch_size"], False)
    c4_loader = make_loader(ProtocolAWindowDataset(c4_m), cfg["batch_size"], False)
    c6_loader = make_loader(ProtocolAWindowDataset(c6_m), cfg["batch_size"], False)

    s, g, f = SpatialAttention(), Generator(), WDCNN(num_classes=3)
    out_dir = OUT_ROOT / "original_protocol"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.csv"
    log_rows = pd.read_csv(log_path).to_dict("records") if (resume and log_path.exists()) else []

    ckpt1_path = out_dir / "stage1_checkpoint.pth"
    resume_state1 = None
    if resume and ckpt1_path.exists():
        ck = torch.load(ckpt1_path, map_location=device, weights_only=False)
        if ck["epoch"] < cfg["pretrain_epochs"] - 1:
            resume_state1 = ck
        else:
            s.load_state_dict(ck["s"]); f.load_state_dict(ck["f"])
    if resume_state1 is not None or not (resume and ckpt1_path.exists()):
        for state in pretrain_s_f(s, f, train_loader, val_loader, device, cfg["pretrain_epochs"],
                                    cfg["lr_s"], cfg["lr_f"], cfg["cosine_period"], log_rows,
                                    resume_state=resume_state1):
            safe_torch_save(state, ckpt1_path)
            pd.DataFrame(log_rows).to_csv(log_path, index=False, encoding="utf-8-sig")

    ckpt2_path = out_dir / "stage2_checkpoint.pth"
    resume_state2 = None
    if resume and ckpt2_path.exists():
        ck = torch.load(ckpt2_path, map_location=device, weights_only=False)
        if ck["epoch"] < cfg["gen_epochs"] - 1:
            resume_state2 = ck
        else:
            g.load_state_dict(ck["g"]); f.load_state_dict(ck["f"])
    if resume_state2 is not None or not (resume and ckpt2_path.exists()):
        for state in train_g_f(s, g, f, train_loader, val_loader, device, cfg["gen_epochs"],
                                 cfg["lr_g"], cfg["lr_f"], cfg["alpha"], log_rows,
                                 resume_state=resume_state2):
            safe_torch_save(state, ckpt2_path)
            pd.DataFrame(log_rows).to_csv(log_path, index=False, encoding="utf-8-sig")

    results = {}
    for name, loader, paper_acc in [("C4", c4_loader, 0.9091), ("C6", c6_loader, 0.8766)]:
        probs, trues = predict_sample_level(s, f, loader, device)
        pred_df = pd.DataFrame({
            "condition": name, "cut_index": np.arange(len(trues)), "stage_true_id": trues,
            "stage_pred_raw": probs.argmax(1), "raw_prob_early": probs[:, 0],
            "raw_prob_middle": probs[:, 1], "raw_prob_late": probs[:, 2],
            "q_true_model": np.nan, "q_hat": np.nan,
        })
        metric_row = dcpsr_base.manuscript_metric_row(pred_df, method="raw",
                                                        method_id=f"DP2Net (Protocol A, Target={name})",
                                                        split=f"target_{name}")
        metric_row["paper_reported_Acc"] = paper_acc
        metric_row["gap_vs_paper"] = metric_row["Acc"] - paper_acc
        save_outputs(out_dir / f"target_{name}", pred_df, metric_row, s, g, f,
                     extra={"protocol": f"A_original_source_C1_target_{name}_sample_level"})
        print(f"[Protocol A] Target={name}: Acc={metric_row['Acc']:.4f} "
              f"(paper reports {paper_acc}, gap={metric_row['gap_vs_paper']:+.4f})")
        results[name] = metric_row
    safe_torch_save({"s": s.state_dict(), "g": g.state_dict(), "f": f.state_dict()}, out_dir / "checkpoint_final.pth")
    (out_dir / "DONE.flag").write_text(f"done: C4 Acc={results['C4']['Acc']:.4f}, C6 Acc={results['C6']['Acc']:.4f}\n")
    return results


# ---------------------------------------------------------------------------
# Protocol B (B-S: native single-source, B-D1: pooled-source adapted)
# ---------------------------------------------------------------------------
def run_protocol_b(variant: str, device: str, pretrain_epochs=None, gen_epochs=None, batch_size=None,
                    resume=False, seed=None):
    assert variant in ("B-S", "B-D1")
    cfg = dict(CFG)
    if pretrain_epochs is not None:
        cfg["pretrain_epochs"] = pretrain_epochs
    if gen_epochs is not None:
        cfg["gen_epochs"] = gen_epochs
    if batch_size is not None:
        cfg["batch_size"] = batch_size
    if seed is not None:
        cfg["seed"] = seed
    set_seed(cfg["seed"])

    unified_manifest = pd.read_csv(THIS_DIR / "data" / "unified_manifest.csv")
    if variant == "B-S":
        tr_df, va_df, te_df = L.get_single_source_split("C1", "C6")
        method_name = "DP2Net (Protocol B-S, native single-source, C1->C6)"
    else:
        tr_df, va_df, te_df = L.get_unified_split()
        method_name = "DP2Net-adapted (pooled source) (Protocol B-D1, C1+C4->C6)"

    train_loader = make_loader(UnifiedWindowDataset(unified_manifest, tr_df), cfg["batch_size"], True)
    val_ds = UnifiedWindowDataset(unified_manifest, va_df)
    val_loader = make_loader(val_ds, cfg["batch_size"], False)
    test_ds = UnifiedWindowDataset(unified_manifest, te_df)
    test_loader = make_loader(test_ds, cfg["batch_size"], False)

    s, g, f = SpatialAttention(), Generator(), WDCNN(num_classes=3)
    out_dir = OUT_ROOT / f"unified_protocol_{variant}" / f"seed{cfg['seed']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.csv"
    log_rows = pd.read_csv(log_path).to_dict("records") if (resume and log_path.exists()) else []

    ckpt1_path = out_dir / "stage1_checkpoint.pth"
    resume_state1 = None
    if resume and ckpt1_path.exists():
        ck = torch.load(ckpt1_path, map_location=device, weights_only=False)
        if ck["epoch"] < cfg["pretrain_epochs"] - 1:
            resume_state1 = ck
        else:
            s.load_state_dict(ck["s"]); f.load_state_dict(ck["f"])
    if resume_state1 is not None or not (resume and ckpt1_path.exists()):
        for state in pretrain_s_f(s, f, train_loader, val_loader, device, cfg["pretrain_epochs"],
                                    cfg["lr_s"], cfg["lr_f"], cfg["cosine_period"], log_rows,
                                    resume_state=resume_state1):
            safe_torch_save(state, ckpt1_path)
            pd.DataFrame(log_rows).to_csv(log_path, index=False, encoding="utf-8-sig")

    ckpt2_path = out_dir / "stage2_checkpoint.pth"
    resume_state2 = None
    if resume and ckpt2_path.exists():
        ck = torch.load(ckpt2_path, map_location=device, weights_only=False)
        if ck["epoch"] < cfg["gen_epochs"] - 1:
            resume_state2 = ck
        else:
            g.load_state_dict(ck["g"]); f.load_state_dict(ck["f"])
    if resume_state2 is not None or not (resume and ckpt2_path.exists()):
        for state in train_g_f(s, g, f, train_loader, val_loader, device, cfg["gen_epochs"],
                                 cfg["lr_g"], cfg["lr_f"], cfg["alpha"], log_rows,
                                 resume_state=resume_state2):
            safe_torch_save(state, ckpt2_path)
            pd.DataFrame(log_rows).to_csv(log_path, index=False, encoding="utf-8-sig")

    probs, _ = predict_sample_level(s, f, test_loader, device)
    run_lvl = aggregate_run_level(test_ds.rows, probs, "stage_id")
    pred_df = build_pred_df_for_metrics(run_lvl)
    metric_row = dcpsr_base.manuscript_metric_row(pred_df, method="raw", method_id=method_name, split="test_C6")
    save_outputs(out_dir, pred_df, metric_row, s, g, f,
                 extra={"protocol": f"{variant}_run_level_aggregated", "seed": cfg["seed"],
                        "leakage_audit": "C6 used only for this single final evaluation; model selection "
                                         "used only the source-domain internal validation split"})
    safe_torch_save({"s": s.state_dict(), "g": g.state_dict(), "f": f.state_dict()},
                     out_dir / "checkpoint_final.pth")
    (out_dir / "DONE.flag").write_text(f"done, Acc={metric_row['Acc']:.4f}\n")
    print(f"[Protocol {variant}] test C6 (run-level, seed={cfg['seed']}): Acc={metric_row['Acc']:.4f} "
          f"Macro-F1={metric_row['Macro-F1']:.4f}")
    return metric_row


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def run_smoke_test(device: str):
    manifest = pd.read_csv(THIS_DIR / "data" / "sample_manifest.csv")
    small = manifest[manifest.condition == "C1"].groupby("class").head(4).reset_index(drop=True)
    if len(small) == 0:
        raise RuntimeError("data/sample_manifest.csv is empty or missing -- run "
                            "`python preprocessing.py --quick` first")
    loader = make_loader(ProtocolAWindowDataset(small), batch_size=4, shuffle=True)
    s, g, f = SpatialAttention(), Generator(), WDCNN(num_classes=3)
    out_dir = OUT_ROOT / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_rows = []
    for _ in pretrain_s_f(s, f, loader, loader, device, 2, 1e-3, 1e-3, 20, log_rows):
        pass
    for _ in train_g_f(s, g, f, loader, loader, device, 2, 1e-5, 1e-3, 20.0, log_rows):
        pass
    safe_torch_save({"s": s.state_dict(), "g": g.state_dict(), "f": f.state_dict()},
                     out_dir / "smoke_checkpoint.pth")
    probs, trues = predict_sample_level(s, f, loader, device)
    pred_df = pd.DataFrame({
        "condition": "C1", "cut_index": np.arange(len(trues)), "stage_true_id": trues,
        "stage_pred_raw": probs.argmax(1), "raw_prob_early": probs[:, 0],
        "raw_prob_middle": probs[:, 1], "raw_prob_late": probs[:, 2],
        "q_true_model": np.nan, "q_hat": np.nan,
    })
    metric_row = dcpsr_base.manuscript_metric_row(pred_df, method="raw", method_id="smoke", split="smoke")
    save_outputs(out_dir, pred_df, metric_row, s, g, f, extra={"protocol": "smoke_test"})
    print("[smoke] OK -- pipeline runs end to end.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["A", "B-S", "B-D1", "smoke"], required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--gen-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.protocol == "smoke":
        run_smoke_test(args.device)
    elif args.protocol == "A":
        run_protocol_a(args.device, pretrain_epochs=args.pretrain_epochs, gen_epochs=args.gen_epochs,
                        batch_size=args.batch_size, resume=args.resume)
    else:
        run_protocol_b(args.protocol, args.device, pretrain_epochs=args.pretrain_epochs,
                        gen_epochs=args.gen_epochs, batch_size=args.batch_size, resume=args.resume,
                        seed=args.seed)
