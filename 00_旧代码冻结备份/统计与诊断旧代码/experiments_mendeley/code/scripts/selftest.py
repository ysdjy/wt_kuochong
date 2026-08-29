#!/usr/bin/env python3
"""Self-test on synthetic data. Run this FIRST on the target machine.

    python scripts/selftest.py

Checks, in order:
  1. every module imports (torch present?)
  2. stage construction, per-sequence q normalisation, T8-style truncation flag
  3. causal online features -- verified by a prefix-invariance test
  4. feature selection / split construction / no-leak assertions
  5. sequence-aware monotonic loss actually distinguishes shuffled from
     temporally adjacent rows  (the bug this package fixes)
  6. one tiny end-to-end (task, seed) unit with the real runner
"""
from __future__ import annotations
import sys, tempfile, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd

OK, FAIL = "  ok  ", " FAIL "
results = []


def check(name):
    def deco(fn):
        try:
            fn(); results.append((OK, name, ""))
        except Exception as e:                                    # noqa: BLE001
            results.append((FAIL, name, f"{type(e).__name__}: {e}"))
            traceback.print_exc()
    return deco


def synth(n_per=140, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed); rows = []
    for m in (1, 2, 3):
        for k in range(3):
            t = (m - 1) * 3 + k + 1
            vb0 = 34.0 if t == 8 else 3.0
            vb = np.sort(vb0 + np.cumsum(rng.gamma(2.0, 0.5, n_per)))
            for i in range(n_per):
                r = dict(sequence_id=f"T{t}", domain_id=f"M{m}", order_key=i + 1,
                         VB=float(vb[i]), ctime=4.0 * (i + 1))
                for j in range(1, 13):
                    r[f"ch{j}__rms"] = vb[i] * (0.05 * j) + rng.normal(m * 0.4, 0.6)
                rows.append(r)
    return pd.DataFrame(rows)


def main() -> int:
    df = synth()

    @check("1. imports")
    def _():
        import dcpsr.config, dcpsr.stages, dcpsr.selection, dcpsr.splits
        import dcpsr.inference, dcpsr.metrics, dcpsr.baselines, dcpsr.aggregate
        import torch  # noqa: F401
        print("   torch", torch.__version__, "cuda", torch.cuda.is_available())

    from dcpsr.stages import build_stage_labels, stage_coverage
    lab, summ = build_stage_labels(df)

    @check("2. per-sequence stage construction + truncation flag")
    def _():
        assert set(lab.stage.unique()) == {"early", "middle", "late"}
        for _, s in lab.groupby("sequence_id"):
            assert abs(s.q_true.min()) < 1e-9 and abs(s.q_true.max() - 1) < 1e-9, \
                "q must be min-max normalised inside each sequence"
        cov = stage_coverage(lab)
        assert bool(cov.loc[cov.sequence_id == "T8", "early_truncated"].iloc[0]), \
            "T8-style run-in truncation not detected"

    from dcpsr.online_features import build_online_features, online_feature_columns

    @check("3. online features are causal (prefix invariance)")
    def _():
        raw = [c for c in df.columns if c.endswith("__rms")]
        full = build_online_features(lab, raw)
        cut = lab.sort_values(["sequence_id", "order_key"])
        cut = cut[cut.groupby("sequence_id")["order_key"].rank(method="first") <= 40].copy()
        part = build_online_features(cut, raw)
        cols = online_feature_columns(full)
        a = full.merge(part[["sequence_id", "order_key"]], on=["sequence_id", "order_key"])
        a = a.sort_values(["sequence_id", "order_key"])[cols].values
        b = part.sort_values(["sequence_id", "order_key"])[cols].values
        assert np.allclose(a, b, atol=1e-8), \
            "online features changed when future runs were removed -> NOT causal"

    from dcpsr.datasets.mendeley import MendeleyMachineToolWear
    from dcpsr.selection import select_features
    from dcpsr.splits import make_splits
    tasks = {t.name: t for t in MendeleyMachineToolWear(".", ".").task_definitions()}

    @check("4. splits + no-leak assertions + feature selection")
    def _():
        for tn in ("D1-M", "MS1", "LOTO_T8"):
            tr, va, te, info = make_splits(lab, tasks[tn], 42)
            assert not (set(tr.sequence_id) & set(te.sequence_id))
            assert not (set(va.sequence_id) & set(te.sequence_id))
            assert info["validation_mode"] in ("tool_holdout", "contiguous_tail")
        tr, va, te, _ = make_splits(lab, tasks["D1-M"], 42)
        sel, rank = select_features(tr, [c for c in df.columns if c.endswith("__rms")],
                                    seed=42, n_selected=6)
        assert 1 <= len(sel) <= 6 and rank.selected.sum() == len(sel)

    @check("5. sequence-aware monotonic loss rejects shuffled neighbours")
    def _():
        import torch
        from dcpsr.model import sequence_aware_monotonic_loss
        q = torch.tensor([[0.1], [0.2], [0.3], [0.9], [0.8]], dtype=torch.float32)
        si = torch.tensor([0, 0, 0, 1, 1]); ok = torch.tensor([1, 2, 3, 7, 8])
        loss_ordered = float(sequence_aware_monotonic_loss(q, si, ok))
        assert loss_ordered > 0, "a genuine decrease inside sequence 1 must be penalised"
        # rows from different sequences / non-adjacent keys must contribute nothing
        si2 = torch.tensor([0, 1, 2, 3, 4]); ok2 = torch.tensor([1, 1, 1, 1, 1])
        assert float(sequence_aware_monotonic_loss(q, si2, ok2)) == 0.0, \
            "non-adjacent rows must be masked out"
        print(f"   ordered-pair loss={loss_ordered:.4f}, shuffled-pair loss=0.0000")

    @check("6. end-to-end unit (tiny)")
    def _():
        from dcpsr.runner import ExperimentRunner
        raw = [c for c in df.columns if c.endswith("__rms")]
        online = build_online_features(lab, raw)
        cand = online_feature_columns(online)
        import dcpsr.config as C
        ep, pa = C.EPOCHS, C.PATIENCE
        C.EPOCHS, C.PATIENCE = 3, 2
        grid = dict(C.FUSION_GRID)
        C.FUSION_GRID = {k: v[:1] for k, v in grid.items()}
        try:
            with tempfile.TemporaryDirectory() as td:
                r = ExperimentRunner(lab, online, cand, td, "primary", "selftest")
                res = r.run_unit(tasks["D1-M"], 42,
                                 ["B1", "B2", "B5", "B10", "B11", "B12"])
                m = res["metrics"]
                print("  ", m[["Method", "Acc", "Macro_F1", "M_F1", "Smooth"]]
                      .round(4).to_string(index=False).replace("\n", "\n   "))
                assert {"A1", "A4", "A5", "A6", "B11", "B12"} <= set(m.Method)
                assert Path(td, "runs", "D1-M", "seed42", "DONE.flag").exists()
        finally:
            C.EPOCHS, C.PATIENCE, C.FUSION_GRID = ep, pa, grid

    print()
    for st, name, err in results:
        print(f"[{st}] {name}" + (f"   {err}" if err else ""))
    n_fail = sum(1 for s, *_ in results if s == FAIL)
    print(f"\n{len(results)-n_fail}/{len(results)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
