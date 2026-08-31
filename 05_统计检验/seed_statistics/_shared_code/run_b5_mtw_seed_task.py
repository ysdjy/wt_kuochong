# -*- coding: utf-8 -*-
r"""
B5 (Multi-source Channel-Spatial Attention CNN, raw-signal CWT images),
MTW-CM (D1-M/D2-M/D3-M), strict TRAIN_SEED isolation.

NEW adaptation (not a port of an existing baseline) -- see
vendored_legacy/b5_mtw_signal_preprocessing.py's module docstring for the
channel choice: `signals_sensor/force_sensor_{x,y,z}` (genuine 3-axis
force, common to all 3 machines) and `signals_machine/tool_position_{x,y,z}`
(common second modality; MTW-CM's raw h5 has no vibration/accelerometer
channel).

Labels/split come from the same MTW-CM frozen-preprocess artifacts every
other MTW-CM method uses (feat_{train,val,test}_frozen.csv's sequence_id/
order_key/source_file/stage_id columns) -- deterministic, no re-derivation.
Raw signal itself is read directly from the per-run .h5 file named in each
row's own `source_file` column.

Evaluation universe: like PHM2010/NASA, this raw-signal method's native
test coverage (every run in the test sequences) is wider than the windowed
methods' (B1/B4/B9's dcpsr.model.build_windows can only emit a prediction
from order_key L onward within a sequence) -- restricted to order_key >= L
per test sequence before scoring, matching the windowed methods' coverage.

Usage:
    python run_b5_mtw_seed_task.py --task D1-M --train_seed 3 \
        --results_root ../B5_MTW_D1M_seed_landscape/results \
        --backbone_root ../B5_MTW_D1M_seed_landscape/backbone_checkpoints
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
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
CODE_DIR = THIS_DIR / "vendored_legacy"
DCPSR_CODE_DIR = PROJECT_ROOT / "experiments_mendeley" / "code"

STAGE_ORDER = ["E", "M", "L"]
ID_TO_STAGE = {0: "early", 1: "middle", 2: "late"}
DEFAULT_CFG = dict(max_epochs=100, patience=15, batch_size=64, lr=0.001, weight_decay=1e-4)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_dir_for(task: str) -> Path:
    return EXPAND_ROOT / "shared" / "reproducibility" / f"MTW_{task}_frozen_preprocess"


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


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


def run_epoch(model, loader, device, optimizer):
    import torch
    import torch.nn.functional as F
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_loss, n, correct = 0.0, 0, 0
    with torch.set_grad_enabled(train_mode):
        for a, b, label in loader:
            a, b, label = a.to(device), b.to(device), label.to(device)
            logits = model(a, b)
            loss = F.cross_entropy(logits, label)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(label)
            correct += (logits.argmax(dim=1) == label).sum().item()
            n += len(label)
    return total_loss / max(n, 1), correct / max(n, 1)


def run_one(task: str, train_seed: int, results_root: Path, backbone_root: Path):
    sys.path.insert(0, str(DCPSR_CODE_DIR))
    from dcpsr import config as C

    frozen_dir = frozen_dir_for(task)
    hashes = assert_frozen_artifacts_unchanged(frozen_dir)
    print(f"[protocol] frozen artifacts verified OK task={task} preprocess_seed={hashes['preprocess_seed']}")
    L = C.ARCH["L"]

    out_dir = results_root / f"seed{train_seed}"
    config_id = {"method": "B5_MultiSource_Attention", "dataset": "MTW_CM", "task": task,
                 "train_sequences": hashes["train_sequences"], "test_sequences": hashes["test_sequences"], "train_seed": train_seed}
    config_hash = hashlib.sha256(json.dumps(config_id, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    if (out_dir / "DONE.flag").exists():
        try:
            with open(out_dir / "run_meta.json", "r", encoding="utf-8") as f:
                prior = json.load(f)
            if prior.get("config_hash") == config_hash and prior.get("train_seed") == train_seed and prior.get("task") == task:
                print(f"[resume] task={task} seed={train_seed} already DONE with matching config -- skipping")
                return "skipped"
        except Exception:
            pass

    sys.path.insert(0, str(CODE_DIR))
    from b5_mtw_dataset import ImagePairDataset
    from b5_model import MultiAttentionCNN

    feat_train = pd.read_csv(frozen_dir / "feat_train_frozen.csv")
    feat_val = pd.read_csv(frozen_dir / "feat_val_frozen.csv")
    feat_test = pd.read_csv(frozen_dir / "feat_test_frozen.csv")
    for df in (feat_train, feat_val, feat_test):
        df["label"] = df["stage_id"]

    set_train_seed(train_seed)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from torch.utils.data import DataLoader
    train_loader = DataLoader(ImagePairDataset(feat_train, "label"), batch_size=DEFAULT_CFG["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(ImagePairDataset(feat_val, "label"), batch_size=DEFAULT_CFG["batch_size"], shuffle=False, num_workers=0)
    test_loader_full = DataLoader(ImagePairDataset(feat_test, "label"), batch_size=DEFAULT_CFG["batch_size"], shuffle=False, num_workers=0)

    t0 = time.time()
    model = MultiAttentionCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=DEFAULT_CFG["lr"], weight_decay=DEFAULT_CFG["weight_decay"])

    best_val_acc, best_epoch, patience_ctr, best_state = -1.0, -1, 0, None
    epoch_log = []
    n_epochs_run = 0
    for epoch in range(DEFAULT_CFG["max_epochs"]):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, opt)
        va_loss, va_acc = run_epoch(model, val_loader, device, None)
        epoch_log.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
        n_epochs_run = epoch + 1
        if va_acc > best_val_acc:
            best_val_acc, best_epoch = va_acc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= DEFAULT_CFG["patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    t1 = time.time()

    backbone_root.mkdir(parents=True, exist_ok=True)
    ckpt_path = backbone_root / f"b5_backbone_{task}_seed{train_seed}.pth"
    torch.save(model.state_dict(), ckpt_path)
    backbone_checkpoint_hash = sha256_file(ckpt_path)

    model.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for a, b, label in test_loader_full:
            a, b = a.to(device), b.to(device)
            logits = model(a, b)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_true.append(label.numpy())
    probs = np.concatenate(all_probs) if all_probs else np.zeros((0, 3))
    true_ids = np.concatenate(all_true) if all_true else np.zeros((0,), dtype=int)
    pred_ids = probs.argmax(axis=1) if len(probs) else np.zeros((0,), dtype=int)

    full_pred_df = pd.DataFrame({
        "sequence_id": feat_test["sequence_id"].values, "order_key": feat_test["order_key"].values.astype(int),
        "true_stage": [ID_TO_STAGE[i] for i in true_ids],
        "pred_stage": [ID_TO_STAGE[i] for i in pred_ids],
        "prob_E": probs[:, 0] if len(probs) else [], "prob_M": probs[:, 1] if len(probs) else [], "prob_L": probs[:, 2] if len(probs) else [],
    })
    # Restrict to order_key >= L per test sequence, matching the windowed
    # methods' (B1/B4/B9) coverage.
    pred_df = full_pred_df[full_pred_df["order_key"] >= L].sort_values(["sequence_id", "order_key"]).reset_index(drop=True)
    if len(pred_df) == 0:
        raise RuntimeError(f"PROTOCOL_FAILED: restricting to order_key>={L} left 0 test rows")

    stage_to_id = {"E": 0, "M": 1, "L": 2}
    id_map = {"early": "E", "middle": "M", "late": "L"}
    y_true = np.array([stage_to_id[id_map[s]] for s in pred_df["true_stage"]])
    y_pred = np.array([stage_to_id[id_map[s]] for s in pred_df["pred_stage"]])
    prob_cu = pred_df[["prob_E", "prob_M", "prob_L"]].values

    p_each, r_each, f1_each, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cmn = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    dy = np.diff(y_pred.astype(int))
    rev = int(np.sum(dy < 0))
    jump = int(np.sum(np.abs(dy) >= 2))
    smooth = float(np.mean(np.sum(np.abs(np.diff(prob_cu, axis=0)), axis=1))) if len(prob_cu) > 1 else float("nan")

    seed_summary = {
        "Method": "B5_MultiSource_Attention", "Dataset": "MTW_CM", "Task": task, "Seed": train_seed,
        "Acc": float(accuracy_score(y_true, y_pred)), "Macro-F1": float(macro_f1),
        "E-F1": float(f1_each[0]), "M-F1": float(f1_each[1]), "L-F1": float(f1_each[2]),
        "M-Precision": float(p_each[1]), "M-Recall": float(r_each[1]),
        "M_to_E": float(cmn[1, 0]), "M_to_L": float(cmn[1, 2]),
        "Rev": rev, "Jump": jump, "Smooth": smooth,
        "q-MAE": np.nan, "q-RMSE": np.nan, "q-R2": np.nan, "Spearman": np.nan,
        "best_epoch": int(best_epoch), "n_epochs_run": int(n_epochs_run),
        "training_seconds": t1 - t0,
    }

    cm_df = pd.DataFrame(cm, index=[f"true_{s}" for s in STAGE_ORDER], columns=[f"pred_{s}" for s in STAGE_ORDER])

    out_dir.mkdir(parents=True, exist_ok=True)
    config_resolved = dict(config_id, architecture={"reduction_ratio": 16, "dropout": 0.5, "image_size": 224}, training=DEFAULT_CFG)
    with open(out_dir / "config_resolved.yaml", "w", encoding="utf-8") as f:
        for k, v in config_resolved.items():
            f.write(f"{k}: {json.dumps(v, ensure_ascii=False)}\n")

    meta_json = {
        **config_id, "git_commit": git_commit(), "window_length_L": L,
        "start_time": t0, "end_time": t1, "training_seconds": t1 - t0,
        "best_epoch": int(best_epoch), "n_test": len(pred_df), "n_test_native": len(full_pred_df),
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
          f"MRec={seed_summary['M-Recall']:.4f} Smooth={seed_summary['Smooth']:.4f} n_test={len(pred_df)}")
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["D1-M", "D2-M", "D3-M"])
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
