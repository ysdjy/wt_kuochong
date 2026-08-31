# -*- coding: utf-8 -*-
r"""
B4 (HTT-Net), NASA Milling N1-N4, strict TRAIN_SEED isolation.

Sibling to run_b4_seed_task.py (PHM2010) and run_nasa_seed_task.py (B9+B3
NASA) -- reuses the same NASA frozen-preprocess artifacts and
`nasa.build_sliding_windows`/`nasa.StageDataset` windowing (via the same
in-memory source-patch import as run_nasa_seed_task.py) that the NASA B9/B3
runner already uses, swapping only the model/training-loop for HTTNet
(vendored_legacy/htt_net_model.py, same architecture/hyperparameters as the
PHM2010 B4 runner). No probability post-processing (B9-only step);
q-MAE/q-RMSE/q-R2/Spearman legitimately NaN (HTT-Net has no continuous
degradation-index head).

Usage:
    python run_b4_nasa_seed_task.py --task N1 --train_seed 3 \
        --results_root ../B4_NASA_N1_seed_landscape/results \
        --backbone_root ../B4_NASA_N1_seed_landscape/backbone_checkpoints
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import subprocess
import sys
import tempfile
import time
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

warnings.filterwarnings("ignore")

THIS_DIR = Path(__file__).resolve().parent
SEED_STATS_DIR = THIS_DIR.parent
EXPAND_ROOT = SEED_STATS_DIR.parents[1]
PROJECT_ROOT = EXPAND_ROOT.parent
CODE_DIR = PROJECT_ROOT / "代码"
NASA_SCRIPT = CODE_DIR / "9.1nasa数据实验.py"
MILL_MAT = PROJECT_ROOT / "mill" / "mill.mat"
VENDORED_DIR = THIS_DIR / "vendored_legacy"

PREPROCESS_SEED = 42
STAGE_ORDER = ["E", "M", "L"]
HTT_ARCH = {"embed_dim": 32, "depths": (2, 2, 2, 2), "num_heads": 4, "window_size": 3, "dropout": 0.20}
TRAIN_CFG = {"lr": 5e-4, "weight_decay": 1e-5, "epochs": 120, "patience": 18, "grad_clip": 1.0}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_dir_for(task: str) -> Path:
    return EXPAND_ROOT / "shared" / "reproducibility" / f"NASA_{task}_frozen_preprocess"


def assert_frozen_artifacts_unchanged(frozen_dir: Path):
    with open(frozen_dir / "manifest_hashes.json", "r", encoding="utf-8") as f:
        recorded = json.load(f)
    mismatches = []
    for name, expected in recorded["files"].items():
        actual = sha256_file(frozen_dir / name)
        if actual != expected:
            mismatches.append((name, expected, actual))
    if mismatches:
        lines = [f"  {n}: expected {e[:16]}... got {a[:16]}..." for n, e, a in mismatches]
        raise RuntimeError("FROZEN PREPROCESSING ARTIFACTS CHANGED -- protocol violated:\n" + "\n".join(lines))
    return recorded


def set_train_seed(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"[set_train_seed] WARNING: use_deterministic_algorithms unavailable: {e}")


def import_nasa_module():
    src = NASA_SCRIPT.read_text(encoding="utf-8")
    safe_out_dir = Path(tempfile.gettempdir()) / "nasa_b4_runner_side_outputs"
    old_mat_line = 'MAT_FILE = Path(r"C:\\Users\\wangting\\Desktop\\博士开题\\公开数据\\1NASA\\mill.mat")'
    old_out_line = 'OUT_DIR = Path(r"C:\\Users\\wangting\\Desktop\\博士开题\\公开数据\\1NASA\\nasa_dcpsr_results_stageaware_opt")'
    assert old_mat_line in src, "MAT_FILE line not found -- old script changed unexpectedly, aborting"
    assert old_out_line in src, "OUT_DIR line not found -- old script changed unexpectedly, aborting"
    src = src.replace(old_mat_line, f'MAT_FILE = Path(r"{MILL_MAT}")')
    src = src.replace(old_out_line, f'OUT_DIR = Path(r"{safe_out_dir}")')
    mod = types.ModuleType("nasa_b4_runner_9_1")
    mod.__file__ = str(NASA_SCRIPT)
    sys.modules["nasa_b4_runner_9_1"] = mod
    exec(compile(src, str(NASA_SCRIPT), "exec"), mod.__dict__)
    return mod


def load_frozen(frozen_dir: Path):
    with open(frozen_dir / "selected_features_seed42.json", "r", encoding="utf-8") as f:
        sel_info = json.load(f)
    selected = sel_info["selected_features_in_order"]
    feat_train = pd.read_csv(frozen_dir / "feat_train_frozen.csv")
    feat_val = pd.read_csv(frozen_dir / "feat_val_frozen.csv")
    feat_test = pd.read_csv(frozen_dir / "feat_test_frozen.csv")
    return selected, feat_train, feat_val, feat_test


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


def class_weights(y: np.ndarray, device: str):
    import torch
    cnt = np.bincount(y, minlength=3).astype(float)
    w = cnt.sum() / (3 * np.maximum(cnt, 1.0))
    return torch.tensor(w / w.mean(), dtype=torch.float32, device=device)


def predict_stage_model(model, loader, device: str):
    import torch
    import torch.nn.functional as F
    model.eval()
    probs, preds = [], []
    with torch.no_grad():
        for X, ys, _, _ in loader:
            p = F.softmax(model(X.to(device)), dim=1).detach().cpu().numpy()
            probs.append(p)
            preds.append(np.argmax(p, axis=1))
    return np.concatenate(preds), np.concatenate(probs)


def train_htt_model(model, train_loader, val_loader, ys_val: np.ndarray, device: str):
    import torch
    import torch.nn.functional as F
    ys_train_all = np.concatenate([b[1].numpy() for b in train_loader])
    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN_CFG["lr"], weight_decay=TRAIN_CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, min_lr=1e-5)
    w = class_weights(ys_train_all, device)
    best_state, best_score, wait, best_epoch = None, np.inf, 0, 0
    epoch_log = []

    for epoch in range(1, TRAIN_CFG["epochs"] + 1):
        model.train()
        epoch_losses = []
        for X, ys, _, _ in train_loader:
            X, ys = X.to(device), ys.to(device)
            logits = model(X)
            loss = F.cross_entropy(logits, ys, weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CFG["grad_clip"])
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))

        yv, pv = predict_stage_model(model, val_loader, device)
        p_each, r_each, f1_each, _ = precision_recall_fscore_support(ys_val, yv, labels=[0, 1, 2], average=None, zero_division=0)
        _, _, macro_f1, _ = precision_recall_fscore_support(ys_val, yv, labels=[0, 1, 2], average="macro", zero_division=0)
        val_loss = -np.mean(np.log(np.clip(pv[np.arange(len(yv)), ys_val], 1e-12, 1.0)))
        score = 0.7 * (1 - macro_f1) + 1.0 * (1 - r_each[1]) + 0.15 * val_loss
        scheduler.step(score)

        epoch_log.append({
            "epoch": epoch, "train_loss": float(np.mean(epoch_losses)), "val_loss": float(val_loss),
            "val_acc": float(accuracy_score(ys_val, yv)), "val_macro_f1": float(macro_f1),
            "val_middle_recall": float(r_each[1]), "score": float(score),
        })
        if score < best_score:
            best_score, best_state, wait, best_epoch = score, copy.deepcopy(model.state_dict()), 0, epoch
        else:
            wait += 1
        if wait >= TRAIN_CFG["patience"]:
            break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "n_epochs_run": epoch, "best_score": float(best_score)}, epoch_log


def run_one(task: str, train_seed: int, results_root: Path, backbone_root: Path):
    frozen_dir = frozen_dir_for(task)
    hashes = assert_frozen_artifacts_unchanged(frozen_dir)
    print(f"[protocol] frozen artifacts verified OK task={task} preprocess_seed={hashes['preprocess_seed']}")

    out_dir = results_root / f"seed{train_seed}"
    if (out_dir / "DONE.flag").exists():
        try:
            with open(out_dir / "run_meta.json", "r", encoding="utf-8") as f:
                prior = json.load(f)
            if (prior.get("gmm_hash") == hashes["files"]["gmm_seed42.pkl"]
                    and prior.get("split_hash") == hashes["files"]["split_manifest.csv"]
                    and prior.get("feature_hash") == hashes["files"]["selected_features_seed42.json"]
                    and prior.get("window_hash") == hashes["files"]["window_manifest.csv"]
                    and prior.get("train_seed") == train_seed and prior.get("task") == task):
                print(f"[resume] task={task} seed={train_seed} already DONE with matching hashes -- skipping")
                return "skipped"
        except Exception:
            pass

    nasa = import_nasa_module()
    sys.path.insert(0, str(VENDORED_DIR))
    from htt_net_model import HTTNet
    selected, feat_train, feat_val, feat_test = load_frozen(frozen_dir)
    L = hashes["window_length_L"]

    set_train_seed(train_seed)
    from torch.utils.data import DataLoader
    tr_pack = nasa.build_sliding_windows(feat_train, selected, L)
    va_pack = nasa.build_sliding_windows(feat_val, selected, L)
    te_pack = nasa.build_sliding_windows(feat_test, selected, L)
    train_loader = DataLoader(nasa.StageDataset(tr_pack), batch_size=nasa.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(nasa.StageDataset(va_pack), batch_size=nasa.BATCH_SIZE, shuffle=False)
    te_loader = DataLoader(nasa.StageDataset(te_pack), batch_size=nasa.BATCH_SIZE, shuffle=False)
    meta = te_pack["meta"].reset_index(drop=True)
    y_true = te_pack["ys"].astype(int)
    input_dim = len(selected)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    model = HTTNet(input_dim=input_dim, num_classes=3, **HTT_ARCH).to(device)
    model, best_info, epoch_log = train_htt_model(model, train_loader, val_loader, va_pack["ys"].astype(int), device)
    t1 = time.time()
    backbone_root.mkdir(parents=True, exist_ok=True)
    ckpt_path = backbone_root / f"b4_backbone_{task}_seed{train_seed}.pth"
    torch.save(model.state_dict(), ckpt_path)
    backbone_checkpoint_hash = sha256_file(ckpt_path)

    y_pred, prob = predict_stage_model(model, te_loader, device)

    p_each, r_each, f1_each, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cmn = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    dy = np.diff(y_pred.astype(int))
    rev = int(np.sum(dy < 0))
    jump = int(np.sum(np.abs(dy) >= 2))
    smooth = float(np.mean(np.sum(np.abs(np.diff(prob, axis=0)), axis=1))) if len(prob) > 1 else float("nan")

    seed_summary = {
        "Method": "B4_HTT_Net", "Dataset": "NASA_Milling", "Task": task, "Seed": train_seed,
        "Acc": float(accuracy_score(y_true, y_pred)), "Macro-F1": float(macro_f1),
        "E-F1": float(f1_each[0]), "M-F1": float(f1_each[1]), "L-F1": float(f1_each[2]),
        "M-Precision": float(p_each[1]), "M-Recall": float(r_each[1]),
        "M_to_E": float(cmn[1, 0]), "M_to_L": float(cmn[1, 2]),
        "Rev": rev, "Jump": jump, "Smooth": smooth,
        "q-MAE": np.nan, "q-RMSE": np.nan, "q-R2": np.nan, "Spearman": np.nan,
        "best_epoch": int(best_info["best_epoch"]), "n_epochs_run": int(best_info["n_epochs_run"]),
        "training_seconds": t1 - t0,
    }

    pred_df = meta[["case", "run"]].copy()
    pred_df["true_stage"] = [STAGE_ORDER[i] for i in y_true]
    pred_df["pred"] = [STAGE_ORDER[i] for i in y_pred]
    pred_df["prob_E"], pred_df["prob_M"], pred_df["prob_L"] = prob[:, 0], prob[:, 1], prob[:, 2]

    cm_df = pd.DataFrame(cm, index=[f"true_{s}" for s in STAGE_ORDER], columns=[f"pred_{s}" for s in STAGE_ORDER])

    out_dir.mkdir(parents=True, exist_ok=True)
    config_resolved = {
        "method": "B4_HTT_Net", "dataset": "NASA_Milling", "task": task,
        "train_cases": hashes["train_cases"], "test_cases": hashes["test_cases"],
        "preprocess_seed": PREPROCESS_SEED, "train_seed": train_seed,
        "architecture": {**HTT_ARCH, "L": L}, "training": TRAIN_CFG,
    }
    config_hash = hashlib.sha256(json.dumps(config_resolved, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    with open(out_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        for k, v in config_resolved.items():
            f.write(f"{k}: {json.dumps(v, ensure_ascii=False)}\n")

    meta_json = {
        "method": "B4_HTT_Net", "dataset": "NASA_Milling", "task": task,
        "train_cases": hashes["train_cases"], "test_cases": hashes["test_cases"],
        "preprocess_seed": PREPROCESS_SEED, "train_seed": train_seed, "git_commit": git_commit(),
        "feature_hash": hashes["files"]["selected_features_seed42.json"],
        "split_hash": hashes["files"]["split_manifest.csv"],
        "gmm_hash": hashes["files"]["gmm_seed42.pkl"],
        "window_hash": hashes["files"]["window_manifest.csv"],
        "start_time": t0, "end_time": t1, "training_seconds": t1 - t0,
        "best_epoch": int(best_info["best_epoch"]), "n_test": len(meta),
        "backbone_checkpoint_hash": backbone_checkpoint_hash, "backbone_checkpoint_path": str(ckpt_path),
        "config_hash": config_hash,
    }
    pd.DataFrame([seed_summary]).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    json_row = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in seed_summary.items()}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump([json_row], f, indent=2, default=str)
    pred_df.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(epoch_log).to_csv(out_dir / "training_log.csv", index=False, encoding="utf-8-sig")
    cm_df.to_csv(out_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_json, f, indent=2, default=str)
    (out_dir / "DONE.flag").write_text(f"done at {t1}\n", encoding="utf-8")
    print(f"[save_run] task={task} seed={train_seed} Acc={seed_summary['Acc']:.4f} "
          f"MacroF1={seed_summary['Macro-F1']:.4f} MF1={seed_summary['M-F1']:.4f} "
          f"MRec={seed_summary['M-Recall']:.4f} Smooth={seed_summary['Smooth']:.4f} n_test={len(meta)}")
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["N1", "N2", "N3", "N4"])
    ap.add_argument("--train_seed", type=int, required=True)
    ap.add_argument("--results_root", required=True, type=Path)
    ap.add_argument("--backbone_root", required=True, type=Path)
    args = ap.parse_args()

    out_dir = args.results_root / f"seed{args.train_seed}"
    try:
        status = run_one(args.task, args.train_seed, args.results_root, args.backbone_root)
        print(f"[protocol] DONE({status}): task={args.task} train_seed={args.train_seed}")
    except Exception as e:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "error.log", "w", encoding="utf-8") as f:
            import traceback
            f.write(f"task={args.task} seed={args.train_seed}\n")
            traceback.print_exc(file=f)
        (out_dir / "FAILED.flag").write_text(f"failed: {e}\n", encoding="utf-8")
        print(f"[protocol] FAILED: task={args.task} train_seed={args.train_seed}: {e}")
        raise


if __name__ == "__main__":
    main()
