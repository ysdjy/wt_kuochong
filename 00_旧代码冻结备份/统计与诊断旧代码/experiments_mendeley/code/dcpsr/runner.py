"""Experiment runner. One (task, seed) unit at a time, resumable.

For each unit exactly ONE multi-task backbone (B11) is trained; B12 and the
whole ablation A1..A6 are derived from its outputs. Single-task deep baselines
B7..B10 are trained separately; B1..B6 are fitted on the same features.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .baselines import BASELINE_INFO, DETERMINISTIC_METHODS, run_b1_b6
from .datasets.base import Task
from .inference import ABLATION_OF, apply_inference, fusion_grid
from .metrics import METRIC_COLS, Q_METRIC_COLS, confusion_rows, evaluate
from .selection import select_features
from .splits import make_splits
from .stages import FineStateModel

DEEP_SINGLE = {"B7": "mlp", "B8": "tcn", "B9": "gru", "B10": "tcngru"}


# --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    import random
    import torch
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def code_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=Path(__file__).parent, text=True).strip()
    except Exception:                                             # noqa: BLE001
        import hashlib
        h = hashlib.sha256()
        for p in sorted(Path(__file__).parent.rglob("*.py")):
            h.update(p.read_bytes())
        return "sha256:" + h.hexdigest()[:16]


class ExperimentRunner:
    def __init__(self, labelled: pd.DataFrame, online: pd.DataFrame,
                 candidate_features: list[str], out_root: str | Path,
                 channel_set: str = "primary", dataset_name: str = "mendeley"):
        self.labelled = labelled
        self.online = online
        self.candidates = candidate_features
        self.root = Path(out_root)
        self.channel_set = channel_set
        self.dataset = dataset_name
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- plumbing
    def unit_dir(self, task: Task, seed: int) -> Path:
        return self.root / "runs" / f"{task.name}" / f"seed{seed}"

    def is_done(self, task: Task, seed: int) -> bool:
        d = self.unit_dir(task, seed)
        return (d / "DONE.flag").exists() and (d / "metrics.json").exists()

    # ------------------------------------------------------------------ run
    def run_unit(self, task: Task, seed: int, methods: list[str],
                 fusion_search: bool = True, save_embeddings: bool = True) -> dict:
        import torch
        from .model import (StageOnlyNet, TCNGRUMultiTask, build_windows,
                            class_weights, make_loader,
                            sequence_aware_monotonic_loss)
        t0 = time.time()
        d = self.unit_dir(task, seed); d.mkdir(parents=True, exist_ok=True)
        set_seed(seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  task={task.name}  seed={seed}  device={device}  methods={methods}", flush=True)

        # 1. splits on the LABELLED table (labels are evaluation truth only)
        tr_lab, va_lab, te_lab, split_info = make_splits(self.labelled, task, seed)

        # 2. attach online features (already causal, built per whole sequence)
        key = ["sequence_id", "order_key"]
        on = self.online
        tr = tr_lab[key].merge(on, on=key, how="left")
        va = va_lab[key].merge(on, on=key, how="left")
        te = te_lab[key].merge(on, on=key, how="left")

        # 3. impute with TRAIN medians only
        med = tr[self.candidates].median()
        for f in (tr, va, te):
            f[self.candidates] = f[self.candidates].fillna(med).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # 4. feature selection -- TRAIN ONLY
        selected, sel_table = select_features(tr, self.candidates, seed=seed)
        sel_table.to_csv(d / "feature_selection.csv", index=False, encoding="utf-8-sig")

        # 5. scaler -- TRAIN ONLY
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(tr[selected].values)
        for f in (tr, va, te):
            f[selected] = np.nan_to_num(scaler.transform(f[selected].values),
                                        nan=0.0, posinf=0.0, neginf=0.0)

        # 6. fine states -- GMM on TRAIN ONLY
        fsm = FineStateModel(seed=seed).fit(tr)
        tr, va, te = fsm.assign(tr), fsm.assign(va), fsm.assign(te)
        fsm.mapping_table.to_csv(d / "gmm_fine_state_mapping.csv", index=False, encoding="utf-8-sig")

        # 7. windows
        L = C.ARCH["L"]
        packs = {}
        for nm, f in (("train", tr), ("val", va), ("test", te)):
            X, ys, yf, yq, si, ok, meta = build_windows(f, selected, L, nm)
            packs[nm] = dict(X=X, ys=ys, yf=yf, yq=yq, seq_index=si, order_key=ok, meta=meta)

        print(f"    train {len(packs['train']['X'])} / val {len(packs['val']['X'])} / "
              f"test {len(packs['test']['X'])} windows | "
              f"train tools {split_info['train_sequences']} | "
              f"val tools {split_info['validation_sequences']} | "
              f"test tools {split_info['test_sequences']}", flush=True)
        self._dump_diagnostics(d, tr, va, te, packs, selected, split_info)

        results, preds = {}, {}

        # 8. B11 multi-task backbone (shared with B12 and A1..A6)
        need_b11 = any(m in methods for m in ("B11", "B12")) or any(m.startswith("A") for m in methods)
        best_params = dict(C.FUSION_DEFAULT)
        hist = None
        if need_b11:
            model, hist, best_epoch = self._train_multitask(packs, len(selected), seed, device)
            torch.save(model.state_dict(), d / "b11_backbone.pth")
            pred_val = self._predict_multitask(model, packs["val"], device)
            pred_test = self._predict_multitask(model, packs["test"], device, want_hidden=save_embeddings)

            # 9. fusion parameters -- VALIDATION ONLY
            if fusion_search:
                best_params, rank = self._search_fusion(pred_val)
                rank.to_csv(d / "fusion_param_ranking.csv", index=False, encoding="utf-8-sig")
            with open(d / "fusion_params.json", "w") as fh:
                json.dump(best_params, fh, indent=2)

            inf_val = apply_inference(pred_val, best_params)
            inf_test = apply_inference(pred_test, best_params)
            inf_val.to_csv(d / "predictions_val_B11B12.csv", index=False, encoding="utf-8-sig")
            inf_test.to_csv(d / "predictions_test_B11B12.csv", index=False, encoding="utf-8-sig")
            preds["B11B12"] = inf_test

            for variant, aid in ABLATION_OF.items():
                m = evaluate(inf_test, variant)
                m.update(dict(Method=aid, kind="ablation", variant=variant))
                results[aid] = m
            results["B11"] = {**evaluate(inf_test, "raw"), "Method": "B11", "kind": "baseline", "variant": "raw"}
            results["B12"] = {**evaluate(inf_test, "final"), "Method": "B12", "kind": "proposed", "variant": "final"}
            if save_embeddings:
                self._save_embeddings(d, inf_test, pred_test, seed, task)

        # 10. single-task deep baselines
        for mid, kind in DEEP_SINGLE.items():
            if mid not in methods:
                continue
            set_seed(seed)
            net, _ = self._train_stage_only(packs, len(selected), kind, seed, device)
            p = self._predict_stage_only(net, packs["test"], device, mid)
            p.to_csv(d / f"predictions_test_{mid}.csv", index=False, encoding="utf-8-sig")
            results[mid] = {**evaluate(p, mid.lower(), with_q=False),
                            "Method": mid, "kind": "baseline", "variant": mid.lower()}
            preds[mid] = p

        # 11. classical baselines
        if any(m in methods for m in ("B1", "B2", "B3", "B4", "B5", "B6")):
            tem, trm = packs["test"]["meta"], packs["train"]["meta"]
            tr_w = trm[["sequence_id", "order_key"]].merge(tr, on=["sequence_id", "order_key"], how="left")
            te_w = tem[["sequence_id", "order_key"]].merge(te, on=["sequence_id", "order_key"], how="left")
            b = run_b1_b6(tr_w, te_w, selected, seed=seed)
            for mid, (y, prob, info) in b.items():
                if mid not in methods:
                    continue
                p = tem.copy()
                p["stage_pred_" + mid.lower()] = np.asarray(y, int)
                for i, s in enumerate(C.STAGE_NAMES):
                    p[f"{mid.lower()}_prob_{s}"] = prob[:, i]
                p.to_csv(d / f"predictions_test_{mid}.csv", index=False, encoding="utf-8-sig")
                results[mid] = {**evaluate(p, mid.lower(), with_q=False),
                                "Method": mid, "kind": "baseline", "variant": mid.lower(),
                                **{k: v for k, v in info.items() if k == "oracle_wear_reference"}}
                preds[mid] = p

        # 12. persist -- MERGE with any metrics already on disk for this unit
        # (methods not recomputed this call, e.g. A1-A6/B11/B12 from an earlier
        # run that only asked for a subset of methods, are preserved rather
        # than silently dropped).
        cfg = self._config(task, seed, split_info, selected, best_params, device, t0)
        with open(d / "config.json", "w") as fh:
            json.dump(cfg, fh, indent=2, default=str)
        rows = []
        for mid, m in results.items():
            rows.append(dict(dataset=self.dataset, task=task.name, group=task.group,
                             channel_set=self.channel_set, seed=seed, **m))
        mdf = pd.DataFrame(rows)
        prior_path = d / "metrics.csv"
        if prior_path.exists() and len(mdf):
            prior = pd.read_csv(prior_path)
            prior = prior[~prior["Method"].isin(mdf["Method"])]
            mdf = pd.concat([prior, mdf], ignore_index=True) if len(prior) else mdf
        mdf.to_csv(d / "metrics.csv", index=False, encoding="utf-8-sig")
        rows = mdf.to_dict("records")
        with open(d / "metrics.json", "w") as fh:
            json.dump(rows, fh, indent=2, default=str)
        if hist is not None:
            hist.to_csv(d / "training_history.csv", index=False, encoding="utf-8-sig")
        (d / "DONE.flag").write_text(f"{time.time()}\n")
        return dict(metrics=mdf, config=cfg)

    # ---------------------------------------------------------- sub-routines
    def _train_multitask(self, packs, input_dim, seed, device):
        import torch
        import torch.nn.functional as F
        from .model import (TCNGRUMultiTask, class_weights, make_loader,
                            sequence_aware_monotonic_loss)
        model = TCNGRUMultiTask(input_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=C.ARCH["lr"], weight_decay=C.WEIGHT_DECAY)
        sw = class_weights(packs["train"]["ys"], 3, device)
        fw = class_weights(packs["train"]["yf"], C.N_FINE_STATES, device)
        tr_loader, tr_sampler = make_loader(packs["train"], seed=seed, shuffle=True)
        va_loader, _ = make_loader(packs["val"], seed=seed, shuffle=False)
        best, best_ep, best_state, wait, rows = np.inf, -1, None, 0, []
        for ep in range(C.EPOCHS):
            tr_sampler.set_epoch(ep)
            trl, st = self._epoch(model, tr_loader, opt, sw, fw, device)
            val, _ = self._epoch(model, va_loader, None, sw, fw, device)
            rows.append(dict(epoch=ep, train_loss=trl, val_loss=val, **st))
            improved = val < best - 1e-6
            if improved:
                best, best_ep, wait = val, ep, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
            if ep % 5 == 0 or improved or wait >= C.PATIENCE:
                print(f"      [Epoch {ep+1}/{C.EPOCHS}] train={trl:.4f} val={val:.4f} "
                      f"best={best:.4f}@{best_ep+1} pairs={st['mono_valid_pairs']} "
                      f"viol={st['mono_violation_rate']:.3f}"
                      + ("  <- best" if improved else f"  patience {wait}/{C.PATIENCE}"),
                      flush=True)
            if wait >= C.PATIENCE:
                print(f"      early stop at epoch {ep+1}, best epoch {best_ep+1}", flush=True)
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, pd.DataFrame(rows), best_ep

    @staticmethod
    def _epoch(model, loader, opt, sw, fw, device):
        import torch
        import torch.nn.functional as F
        from .model import sequence_aware_monotonic_loss
        train = opt is not None
        model.train() if train else model.eval()
        tot, n = 0.0, 0
        pairs = viols = 0
        mono_sum = 0.0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for X, ys, yf, yq, si, ok in loader:
                X, ys, yf, yq = X.to(device), ys.to(device), yf.to(device), yq.to(device)
                si, ok = si.to(device), ok.to(device)
                out = model(X)
                mono, st = sequence_aware_monotonic_loss(out["q_hat"], si, ok, return_stats=True)
                loss = (C.LAMBDA_STAGE * F.cross_entropy(out["stage_logits"], ys, weight=sw)
                        + C.LAMBDA_FINE * F.cross_entropy(out["fine_logits"], yf, weight=fw)
                        + C.LAMBDA_Q * F.smooth_l1_loss(out["q_hat"], yq)
                        + C.LAMBDA_MONO * mono)
                if train:
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
                    opt.step()
                tot += float(loss) * len(X); n += len(X)
                pairs += st["n_valid_pairs"]; viols += st["n_violations"]
                mono_sum += float(mono) * len(X)
        return (tot / max(n, 1),
                dict(mono_valid_pairs=pairs, mono_violations=viols,
                     mono_violation_rate=viols / max(pairs, 1),
                     mono_loss=mono_sum / max(n, 1)))

    @staticmethod
    def _predict_multitask(model, pack, device, want_hidden=False):
        import torch
        model.eval()
        X = torch.tensor(pack["X"], dtype=torch.float32)
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                outs.append(model(X[i:i + 256].to(device), return_hidden=want_hidden))
        cat = lambda k: torch.cat([o[k] for o in outs]).cpu().numpy()
        df = pack["meta"].copy().reset_index(drop=True)
        sp, fp, qh = cat("stage_prob"), cat("fine_prob"), cat("q_hat").ravel()
        for i, s in enumerate(C.STAGE_NAMES):
            df[f"raw_prob_{s}"] = sp[:, i]
        for i in range(fp.shape[1]):
            df[f"fine_prob_{i}"] = fp[:, i]
        df["q_hat"] = qh
        if want_hidden:
            H = cat("h")
            for j in range(H.shape[1]):
                df[f"h{j}"] = H[:, j]
        return df

    def _train_stage_only(self, packs, input_dim, kind, seed, device):
        import torch
        import torch.nn.functional as F
        from .model import StageOnlyNet, class_weights, make_loader
        net = StageOnlyNet(input_dim, kind=kind).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=C.ARCH["lr"], weight_decay=C.WEIGHT_DECAY)
        sw = class_weights(packs["train"]["ys"], 3, device)
        tr, sampler = make_loader(packs["train"], seed=seed, shuffle=True)
        va, _ = make_loader(packs["val"], seed=seed, shuffle=False)
        best, best_state, wait = np.inf, None, 0
        for ep in range(C.EPOCHS):
            sampler.set_epoch(ep)
            for phase, loader, o in (("tr", tr, opt), ("va", va, None)):
                net.train() if o else net.eval()
                tot, n = 0.0, 0
                with (torch.enable_grad() if o else torch.no_grad()):
                    for X, ys, *_ in loader:
                        X, ys = X.to(device), ys.to(device)
                        loss = F.cross_entropy(net(X)["stage_logits"], ys, weight=sw)
                        if o:
                            o.zero_grad(); loss.backward()
                            torch.nn.utils.clip_grad_norm_(net.parameters(), C.GRAD_CLIP)
                            o.step()
                        tot += float(loss) * len(X); n += len(X)
                if phase == "va":
                    v = tot / max(n, 1)
            if v < best - 1e-6:
                best, wait = v, 0
                best_state = {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
            else:
                wait += 1
                if wait >= C.PATIENCE:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        return net, best

    @staticmethod
    def _predict_stage_only(net, pack, device, mid):
        import torch
        import torch.nn.functional as F
        net.eval()
        X = torch.tensor(pack["X"], dtype=torch.float32)
        ps = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                ps.append(F.softmax(net(X[i:i + 256].to(device))["stage_logits"], 1).cpu().numpy())
        P = np.concatenate(ps)
        df = pack["meta"].copy().reset_index(drop=True)
        v = mid.lower()
        for i, s in enumerate(C.STAGE_NAMES):
            df[f"{v}_prob_{s}"] = P[:, i]
        df[f"stage_pred_{v}"] = P.argmax(1)
        return df

    @staticmethod
    def _search_fusion(pred_val: pd.DataFrame):
        """Grid is fixed in config.py and evaluated ONLY on validation."""
        rows = []
        for p in fusion_grid():
            m = evaluate(apply_inference(pred_val, p), "final")
            score = (1.0 * (1 - m["Acc"]) + 1.0 * (1 - m["Macro_F1"])
                     + 1.4 * (1 - m["M_Rec"]) + 0.8 * _nz(m["M_to_L"])
                     + 0.7 * _nz(m["M_to_E"]) + 0.15 * _nz(m["q_RMSE"]))
            rows.append({**p, **{f"val_{k}": v for k, v in m.items()
                                 if isinstance(v, (int, float))}, "val_score": score})
        rank = pd.DataFrame(rows).sort_values("val_score").reset_index(drop=True)
        best = {k: float(rank.iloc[0][k]) for k in C.FUSION_GRID}
        return best, rank

    @staticmethod
    def _save_embeddings(d: Path, inf_test: pd.DataFrame, pred_test: pd.DataFrame,
                         seed: int, task: Task) -> None:
        hcols = [c for c in pred_test.columns if c.startswith("h") and c[1:].isdigit()]
        if not hcols:
            return
        keep = ["sequence_id", "domain_id", "order_key", "VB", "q_true", "q_hat",
                "stage_true", "stage_true_id"]
        emb = pred_test[keep + hcols].copy()
        emb["seed"] = seed; emb["task"] = task.name
        emb["stage_pred_final"] = inf_test["stage_pred_final_name"].values
        _write_table(emb, d / "representation_embeddings")

    @staticmethod
    def _dump_diagnostics(d: Path, tr, va, te, packs, selected, split_info) -> None:
        rows = []
        for nm, f in (("train", tr), ("val", va), ("test", te)):
            sc = f["stage"].value_counts().reindex(C.STAGE_NAMES, fill_value=0)
            fc = f["fine_state_true"].value_counts().reindex(range(C.N_FINE_STATES), fill_value=0)
            rows.append(dict(split=nm, n_runs=len(f), n_windows=len(packs[nm]["X"]),
                             tools=",".join(sorted(f.sequence_id.unique())),
                             **{f"stage_{k}": int(v) for k, v in sc.items()},
                             **{f"S{k}": int(v) for k, v in fc.items()},
                             q_min=float(f.q_true.min()), q_max=float(f.q_true.max()),
                             VB_min=float(f.VB.min()), VB_max=float(f.VB.max())))
        pd.DataFrame(rows).to_csv(d / "split_diagnostics.csv", index=False, encoding="utf-8-sig")

    def _config(self, task, seed, split_info, selected, fusion, device, t0) -> dict:
        return dict(
            dataset=self.dataset, channel_set=self.channel_set,
            task=task.to_dict(), seed=seed, split=split_info,
            n_features_candidate=len(self.candidates), n_features_selected=len(selected),
            selected_features=selected,
            window_length=C.ARCH["L"], dropout=C.ARCH["dropout"], learning_rate=C.ARCH["lr"],
            tcn_channels=list(C.ARCH["tcn_channels"]), gru_hidden=C.ARCH["gru_hidden"],
            batch_size=C.BATCH_SIZE, epochs=C.EPOCHS, patience=C.PATIENCE,
            loss_weights=dict(stage=C.LAMBDA_STAGE, fine=C.LAMBDA_FINE,
                              q=C.LAMBDA_Q, mono=C.LAMBDA_MONO),
            monotonic_loss="sequence_aware (same sequence, consecutive order_key)",
            gmm=dict(n_states=C.N_FINE_STATES, fitted_on="train split only"),
            stage_thresholds=dict(Q_E=C.Q_EARLY, Q_L=C.Q_LATE, Q_nu=C.RATE_LATE_Q,
                                  normalised_within="sequence_id"),
            fusion_parameters=fusion, fusion_tuned_on="internal validation split",
            code_hash=code_hash(), device=device, python=platform.python_version(),
            start_time=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
            end_time=time.strftime("%Y-%m-%dT%H:%M:%S"), runtime_sec=round(time.time() - t0, 2))


def _nz(v):
    return 0.0 if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)


def _write_table(df: pd.DataFrame, stem: Path) -> None:
    try:
        df.to_parquet(stem.with_suffix(".parquet"), index=False)
    except Exception:                                             # noqa: BLE001
        df.to_csv(stem.with_suffix(".csv"), index=False, encoding="utf-8-sig")
