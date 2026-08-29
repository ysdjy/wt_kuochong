# -*- coding: utf-8 -*-
"""
D2/D3 transfer-task training for the internal window-based methods:
RF, TCN-GRU, Multi-task TCN-GRU (+DC-PSR post-processing), HTT-Net.

Each method writes into
    final_statistical_evidence/transfer_tasks/<task>/<method>/
        config.yaml, status.json, run.log, best.pt (or model.pkl for RF),
        metrics.json, predictions.csv, DONE.flag

TRAIN_SEED is fixed at 42 for every job (task instruction: no seed study
in this stage). Architecture/hyperparameters are the frozen D1 config,
unchanged -- only which two conditions are pooled as source and which one
is held out as target changes between D2 and D3.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import common_pipeline as cp

TRAIN_SEED = 42
TASKS = {
    "D2": (["C1", "C6"], "C4"),
    "D3": (["C4", "C6"], "C1"),
}
TRANSFER_ROOT = Path(__file__).resolve().parents[2] / "transfer_tasks"


def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(out_dir))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    logger.addHandler(sh)
    return logger


def write_status(out_dir: Path, state: str, extra: dict | None = None):
    payload = {"state": state, "timestamp": time.time()}
    if extra:
        payload.update(extra)
    (out_dir / "status.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def already_done(out_dir: Path) -> bool:
    return (out_dir / "DONE.flag").exists()


# ---------------------------------------------------------------------------
# TCN/GRU model defs -- verbatim from 代码/7.7跨工况实验.py section 2
# ---------------------------------------------------------------------------
class TCNGRUStageOnly(nn.Module):
    def __init__(self, input_dim, channels=(32, 64, 64), hidden=64, dropout=0.2):
        super().__init__()
        base = cp.import_base()
        layers, ch = [], input_dim
        for i, out_ch in enumerate(channels):
            layers.append(base.TemporalBlock(ch, out_ch, 3, 2 ** i, dropout))
            ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.gru = nn.GRU(ch, hidden, batch_first=True)
        self.shared = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout))
        self.stage_head = nn.Linear(64, 3)

    def forward(self, x):
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(h)
        return self.stage_head(self.shared(h[:, -1, :]))


def class_weights(y, base):
    cnt = np.bincount(y, minlength=3).astype(float)
    w = cnt.sum() / (3 * np.maximum(cnt, 1.0))
    return torch.tensor(w / w.mean(), dtype=torch.float32, device=base.DEVICE)


def predict_stage_model(model, pack, base):
    model.eval()
    probs, preds = [], []
    with torch.no_grad():
        for X, _, _, _ in pack["loader"]:
            p = F.softmax(model(X.to(base.DEVICE)), dim=1).detach().cpu().numpy()
            probs.append(p)
            preds.append(np.argmax(p, axis=1))
    return np.concatenate(preds), np.concatenate(probs)


def train_stage_model(model, tr_pack, va_pack, base, epochs=120, patience=18, logger=None):
    model = model.to(base.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=base.BEST_ARCH["lr"], weight_decay=base.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, min_lr=1e-5)
    w = class_weights(tr_pack["ys"], base)
    best_state, best_score, wait = None, np.inf, 0
    best_info = {"best_epoch": 0, "best_val_acc": np.nan, "best_val_macro_f1": np.nan, "best_val_MRec": np.nan}
    for epoch in range(1, epochs + 1):
        model.train()
        for X, ys, _, _ in tr_pack["loader"]:
            X, ys = X.to(base.DEVICE), ys.to(base.DEVICE)
            logits = model(X)
            loss = F.cross_entropy(logits, ys, weight=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), base.GRAD_CLIP)
            opt.step()
        yv, pv = predict_stage_model(model, va_pack, base)
        m = base.clf_metrics(va_pack["ys"], yv)
        val_loss = -np.mean(np.log(np.clip(pv[np.arange(len(yv)), va_pack["ys"]], 1e-12, 1.0)))
        score = 0.7 * (1 - m["f1"]) + 1.0 * (1 - m["middle_recall"]) + 0.15 * val_loss
        scheduler.step(score)
        if score < best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
            best_info = {"best_epoch": epoch, "best_val_acc": m["acc"], "best_val_macro_f1": m["f1"], "best_val_MRec": m["middle_recall"]}
        else:
            wait += 1
        if logger and epoch % 10 == 0:
            logger.info(f"epoch {epoch} val_acc={m['acc']:.4f} val_macro_f1={m['f1']:.4f} wait={wait}")
        if wait >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_info


def pred_df_from(meta, y_pred, prob, base):
    df = meta[["condition", "run_id"]].copy() if "run_id" in meta.columns else meta[["condition", "cut_index"]].rename(columns={"cut_index": "run_id"})
    df["true_stage"] = meta["stage_true"].values
    df["pred_stage"] = [base.ID_TO_STAGE[int(v)] for v in y_pred]
    df["p_early"], df["p_middle"], df["p_late"] = prob[:, 0], prob[:, 1], prob[:, 2]
    return df


def run_rf(task: str, train_conditions, test_condition, force: bool = False):
    out_dir = TRANSFER_ROOT / task / "rf"
    if already_done(out_dir) and not force:
        print(f"[rf/{task}] DONE.flag present, skip")
        return
    logger = setup_logger(out_dir)
    write_status(out_dir, "running")
    logger.info(f"rf {task}: train={train_conditions} test={test_condition} seed={TRAIN_SEED}")

    base = cp.import_base()
    cp.set_train_seed(TRAIN_SEED)
    tr_pack, va_pack, te_pack, selected, feat_train, feat_test = cp.prepare_task_data(train_conditions, test_condition)
    meta = te_pack["meta"].copy().reset_index(drop=True)
    y_true = te_pack["ys"].astype(int)

    # RF uses the per-run selected-feature snapshot directly (not the L=12
    # window tensor) -- matches protocol_diagnostic_fixed_preprocess's
    # run_rf() exactly: train on ALL feasible train rows (unwindowed),
    # evaluate only on the windowed-feasible test run_ids (meta["cut_index"])
    # so the RF test universe matches the other window-based methods'.
    Xtr = feat_train[selected].values
    ytr = feat_train["stage_id"].values.astype(int)
    test_by_run = feat_test.set_index("run_id")
    Xte = test_by_run.loc[meta["cut_index"].values, selected].values

    from sklearn.ensemble import RandomForestClassifier
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=400, random_state=TRAIN_SEED, class_weight="balanced_subsample", n_jobs=-1)
    rf.fit(Xtr, ytr)
    y_pred = rf.predict(Xte)
    prob = rf.predict_proba(Xte)
    t1 = time.time()

    metrics = cp.full_metrics(y_true, y_pred, prob)
    pred_df = pred_df_from(meta, y_pred, prob, base)

    import pickle
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump(rf, f)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (out_dir / "config.yaml").write_text(
        f"method: rf\ntask: {task}\ntrain_conditions: {train_conditions}\ntest_condition: {test_condition}\n"
        f"train_seed: {TRAIN_SEED}\nn_estimators: 400\nclass_weight: balanced_subsample\ntrain_seconds: {t1 - t0:.2f}\n",
        encoding="utf-8",
    )
    write_status(out_dir, "done", {"acc": metrics["Acc"]})
    (out_dir / "DONE.flag").write_text(f"done at {time.time()}\n", encoding="utf-8")
    logger.info(f"rf {task} DONE acc={metrics['Acc']:.4f}")


def run_tcn_gru(task: str, train_conditions, test_condition, force: bool = False):
    out_dir = TRANSFER_ROOT / task / "tcn_gru"
    if already_done(out_dir) and not force:
        print(f"[tcn_gru/{task}] DONE.flag present, skip")
        return
    logger = setup_logger(out_dir)
    write_status(out_dir, "running")
    logger.info(f"tcn_gru {task}: train={train_conditions} test={test_condition} seed={TRAIN_SEED}")

    base = cp.import_base()
    cp.set_train_seed(TRAIN_SEED)
    tr_pack, va_pack, te_pack, selected, feat_train, feat_test = cp.prepare_task_data(train_conditions, test_condition)
    meta = te_pack["meta"].copy().reset_index(drop=True)
    y_true = te_pack["ys"].astype(int)
    input_dim = len(selected)

    t0 = time.time()
    model = TCNGRUStageOnly(input_dim, base.BEST_ARCH["channels"], base.BEST_ARCH["gru_hidden"], base.BEST_ARCH["dropout"])
    model, best_info = train_stage_model(model, tr_pack, va_pack, base, epochs=base.EPOCHS, patience=base.PATIENCE, logger=logger)
    t1 = time.time()
    torch.save(model.state_dict(), out_dir / "best.pt")

    y_pred, prob = predict_stage_model(model, te_pack, base)
    metrics = cp.full_metrics(y_true, y_pred, prob)
    pred_df = pred_df_from(meta, y_pred, prob, base)

    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (out_dir / "config.yaml").write_text(
        f"method: tcn_gru\ntask: {task}\ntrain_conditions: {train_conditions}\ntest_condition: {test_condition}\n"
        f"train_seed: {TRAIN_SEED}\nBEST_ARCH: {base.BEST_ARCH}\nepochs: {base.EPOCHS}\npatience: {base.PATIENCE}\n"
        f"train_seconds: {t1 - t0:.2f}\nbest_info: {best_info}\n",
        encoding="utf-8",
    )
    write_status(out_dir, "done", {"acc": metrics["Acc"]})
    (out_dir / "DONE.flag").write_text(f"done at {time.time()}\n", encoding="utf-8")
    logger.info(f"tcn_gru {task} DONE acc={metrics['Acc']:.4f}")


def run_multitask_and_dcpsr(task: str, train_conditions, test_condition, force: bool = False):
    out_dir_mt = TRANSFER_ROOT / task / "multitask_tcn_gru"
    out_dir_dc = TRANSFER_ROOT / task / "dc_psr"
    if already_done(out_dir_mt) and already_done(out_dir_dc) and not force:
        print(f"[multitask_tcn_gru+dc_psr/{task}] DONE.flag present for both, skip")
        return
    logger = setup_logger(out_dir_mt)
    out_dir_dc.mkdir(parents=True, exist_ok=True)
    write_status(out_dir_mt, "running")
    write_status(out_dir_dc, "running")
    logger.info(f"multitask_tcn_gru+dc_psr {task}: train={train_conditions} test={test_condition} seed={TRAIN_SEED}")

    base = cp.import_base()
    cp.set_train_seed(TRAIN_SEED)
    base.RANDOM_SEED = TRAIN_SEED
    tr_pack, va_pack, te_pack, selected, feat_train, feat_test = cp.prepare_task_data(train_conditions, test_condition)
    meta = te_pack["meta"].copy().reset_index(drop=True)
    y_true = te_pack["ys"].astype(int)
    input_dim = len(selected)

    t0 = time.time()
    mt_model, hist, best_score, best_epoch = base.train_model(tr_pack, va_pack, input_dim)
    t1 = time.time()
    out_dir_mt.mkdir(parents=True, exist_ok=True)
    torch.save(mt_model.state_dict(), out_dir_mt / "best.pt")

    pred_test_raw = base.predict_model(mt_model, te_pack)
    B12_PARAMS = {
        "eta": 0.75, "fine_weight": 0.30, "temperature": 1.20, "mid_floor": 0.12,
        "late_tau": 0.66, "early_tau": 0.38, "order_blend": 0.25,
    }
    pred_test = base.apply_probability_inference(pred_test_raw, B12_PARAMS)

    y_pred_mt = pred_test["stage_pred_raw"].values.astype(int)
    prob_mt = pred_test[[f"raw_prob_{s}" for s in base.STAGE_NAMES]].values
    y_pred_dc = pred_test["stage_pred_final"].values.astype(int)
    prob_dc = pred_test[[f"final_prob_{s}" for s in base.STAGE_NAMES]].values

    metrics_mt = cp.full_metrics(y_true, y_pred_mt, prob_mt)
    metrics_dc = cp.full_metrics(y_true, y_pred_dc, prob_dc)
    pred_df_mt = pred_df_from(meta, y_pred_mt, prob_mt, base)
    pred_df_dc = pred_df_from(meta, y_pred_dc, prob_dc, base)

    pred_df_mt.to_csv(out_dir_mt / "predictions.csv", index=False)
    (out_dir_mt / "metrics.json").write_text(json.dumps(metrics_mt, indent=2, default=str), encoding="utf-8")
    (out_dir_mt / "config.yaml").write_text(
        f"method: multitask_tcn_gru\ntask: {task}\ntrain_conditions: {train_conditions}\ntest_condition: {test_condition}\n"
        f"train_seed: {TRAIN_SEED}\nBEST_ARCH: {base.BEST_ARCH}\ntrain_seconds: {t1 - t0:.2f}\nbest_epoch: {best_epoch}\n",
        encoding="utf-8",
    )
    write_status(out_dir_mt, "done", {"acc": metrics_mt["Acc"]})
    (out_dir_mt / "DONE.flag").write_text(f"done at {time.time()}\n", encoding="utf-8")

    out_dir_dc.mkdir(parents=True, exist_ok=True)
    pred_df_dc.to_csv(out_dir_dc / "predictions.csv", index=False)
    (out_dir_dc / "metrics.json").write_text(json.dumps(metrics_dc, indent=2, default=str), encoding="utf-8")
    (out_dir_dc / "config.yaml").write_text(
        f"method: dc_psr\ntask: {task}\ntrain_conditions: {train_conditions}\ntest_condition: {test_condition}\n"
        f"train_seed: {TRAIN_SEED}\nB12_PARAMS: {B12_PARAMS}\nshared_checkpoint: {str((out_dir_mt / 'best.pt').relative_to(TRANSFER_ROOT.parent))}\n",
        encoding="utf-8",
    )
    write_status(out_dir_dc, "done", {"acc": metrics_dc["Acc"]})
    (out_dir_dc / "DONE.flag").write_text(f"done at {time.time()}\n", encoding="utf-8")
    logger.info(f"multitask_tcn_gru {task} DONE acc={metrics_mt['Acc']:.4f}; dc_psr DONE acc={metrics_dc['Acc']:.4f}")


def run_htt_net(task: str, train_conditions, test_condition, force: bool = False):
    out_dir = TRANSFER_ROOT / task / "htt_net"
    if already_done(out_dir) and not force:
        print(f"[htt_net/{task}] DONE.flag present, skip")
        return
    logger = setup_logger(out_dir)
    write_status(out_dir, "running")
    logger.info(f"htt_net {task}: train={train_conditions} test={test_condition} seed={TRAIN_SEED}")

    base = cp.import_base()
    cp.set_train_seed(TRAIN_SEED)
    tr_pack, va_pack, te_pack, selected, feat_train, feat_test = cp.prepare_task_data(train_conditions, test_condition)
    meta = te_pack["meta"].copy().reset_index(drop=True)
    y_true = te_pack["ys"].astype(int)
    input_dim = len(selected)

    import sys
    HTT_DIR = TRANSFER_ROOT.parents[1] / "baselines" / "htt_net"
    sys.path.insert(0, str(HTT_DIR))
    if "train" in sys.modules:
        del sys.modules["train"]
    import train as T  # noqa

    T.HTT_ARCH = {"embed_dim": 64, "depths": (2, 2, 2, 2), "num_heads": 4, "window_size": 3, "dropout": 0.20}
    T.TRAIN_CFG["lr"] = 5e-4

    t0 = time.time()
    model = T.HTTNet(input_dim=input_dim, num_classes=3, **T.HTT_ARCH)
    model, hist, best_info = T.train_htt_model(model, tr_pack, va_pack, T.TRAIN_CFG["epochs"], T.TRAIN_CFG["patience"])
    t1 = time.time()
    torch.save(model.state_dict(), out_dir / "best.pt")

    y_pred, prob = T.predict_stage_model(model, te_pack)
    metrics = cp.full_metrics(y_true, y_pred, prob)
    pred_df = pred_df_from(meta, y_pred, prob, base)

    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (out_dir / "config.yaml").write_text(
        f"method: htt_net\ntask: {task}\ntrain_conditions: {train_conditions}\ntest_condition: {test_condition}\n"
        f"train_seed: {TRAIN_SEED}\nHTT_ARCH: {T.HTT_ARCH}\nlr: {T.TRAIN_CFG['lr']}\nepochs: {T.TRAIN_CFG['epochs']}\n"
        f"patience: {T.TRAIN_CFG['patience']}\ntrain_seconds: {t1 - t0:.2f}\nn_parameters: {model.num_parameters()}\n",
        encoding="utf-8",
    )
    write_status(out_dir, "done", {"acc": metrics["Acc"]})
    (out_dir / "DONE.flag").write_text(f"done at {time.time()}\n", encoding="utf-8")
    logger.info(f"htt_net {task} DONE acc={metrics['Acc']:.4f}")


DISPATCH = {
    "rf": run_rf,
    "tcn_gru": run_tcn_gru,
    "multitask_tcn_gru": run_multitask_and_dcpsr,
    "dc_psr": run_multitask_and_dcpsr,
    "htt_net": run_htt_net,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASKS.keys()))
    parser.add_argument("--method", required=True, choices=list(DISPATCH.keys()))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    train_conditions, test_condition = TASKS[args.task]
    DISPATCH[args.method](args.task, train_conditions, test_condition, force=args.force)


if __name__ == "__main__":
    main()
