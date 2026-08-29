#!/usr/bin/env python3
"""Phase 0+1: dataset audit and run-level feature extraction (resumable).

    python scripts/00_extract_features.py --raw-dir <dir> --out-root experiments_mendeley \
        [--channel-set primary|all] [--audit-only] [--limit-per-shard 0]

Shards by tool so an interrupted run resumes from the next unfinished tool.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from dcpsr.datasets.mendeley import MendeleyMachineToolWear
from dcpsr.stages import build_stage_labels, stage_coverage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-root", default="experiments_mendeley")
    ap.add_argument("--channel-set", default="primary", choices=["primary", "all"])
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--tools", default="", help="comma list, e.g. T1,T2 (default: all)")
    args = ap.parse_args()

    root = Path(args.out_root)
    d_audit, d_feat = root / "00_dataset_audit", root / "01_features"
    for d in (d_audit, d_feat, d_feat / "shards"):
        d.mkdir(parents=True, exist_ok=True)

    ad = MendeleyMachineToolWear(args.raw_dir, d_feat, channel_set=args.channel_set)
    meta = ad.read_filelist()
    print(f"filelist: {len(meta)} runs, {meta.sequence_id.nunique()} sequences, "
          f"{meta.domain_id.nunique()} domains")

    # ---- audit ------------------------------------------------------------
    schema = ad.scan_schema()
    (d_audit / "hdf5_schema_report.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    chan_sum = ad.channel_summary(schema)
    chan_sum.to_csv(d_audit / "channel_summary.csv", index=False, encoding="utf-8-sig")
    print("\n--- real HDF5 channels (from h5py) ---")
    print(chan_sum[["file", "tool", "group", "channel", "shape", "dtype", "compression",
                    "fs_measured_hz", "status"]].to_string(index=False))
    for grp, fs in (next(iter(schema["files"].values()))).get("fs_measured", {}).items():
        ad.measured_fs[grp] = fs
    by_domain = ad.discover_channels_by_domain(schema)
    print("\n--- channels physically present, per machine (post per-machine discovery) ---")
    for dom, groups in sorted(by_domain.items()):
        for grp, chans in groups.items():
            print(f"   {dom} {grp}: {sorted(chans)}")
    (d_audit / "primary_channel_set.json").write_text(
        json.dumps(ad.primary_channel_set(by_domain), indent=2, ensure_ascii=False))
    exc = ad.excluded_channel_report(by_domain)
    exc.to_csv(d_audit / "excluded_channel_report.csv", index=False, encoding="utf-8-sig")
    print("\n--- channel status, resolved PER TOOL (evidence-based; nothing inferred "
          "from name similarity) ---")
    print(exc[["channel", "group", "status", "present_on_tools", "tools_affected",
               "tools_primary", "reason", "used_in_primary_experiment"]].to_string(index=False))
    un = exc[exc.status == "unresolved"]
    if len(un):
        print(f"\n!! {len(un)} channel(s) UNRESOLVED -- excluded from the primary set for "
              f"every tool, NOT mapped onto a documented name by guesswork:")
        print(un[["channel", "evidence"]].to_string(index=False))

    lab, summ = build_stage_labels(meta)
    summ.to_csv(d_audit / "dataset_quality_report.csv", index=False, encoding="utf-8-sig")
    stage_coverage(lab).to_csv(d_audit / "stage_coverage_by_tool.csv", index=False, encoding="utf-8-sig")
    lab.to_csv(d_audit / "per_run_stage_preview.csv", index=False, encoding="utf-8-sig")
    print(summ.to_string(index=False))
    if args.audit_only:
        print("audit only -- stopping"); return 0

    # ---- features (resumable, sharded by tool) -----------------------------
    # Channel list is resolved PER TOOL: a machine-specific channel (e.g.
    # torque_axis_* on M1, force_axis_* on M2/M3) is never dropped just
    # because a different machine's tools lack it under that name.
    tools = args.tools.split(",") if args.tools else sorted(meta.sequence_id.unique())
    for tool in tools:
        shard = d_feat / "shards" / f"{args.channel_set}_{tool}.csv"
        if shard.exists():
            print(f"[skip] {tool} -> {shard.name}"); continue
        chans = ad.channels_for_tool(tool, by_domain, args.channel_set)
        print(f"\n{tool}: channel set '{args.channel_set}' -> {len(chans)} channels")
        for g, c, fs in chans:
            print(f"   {g}/{c}  fs={fs}")
        sub = meta[meta.sequence_id == tool]
        t0 = time.time()
        df = ad.extract_features(chans, sub,
                                 progress=lambda i, n: print(f"   {tool} {i}/{n}", flush=True))
        df.to_csv(shard, index=False, encoding="utf-8-sig")
        bad = int((df["_read_error"].astype(str) != "").sum())
        print(f"[done] {tool}: {len(df)} runs, {df.shape[1]} cols, {bad} read errors, "
              f"{time.time()-t0:.1f}s")

    shards = sorted((d_feat / "shards").glob(f"{args.channel_set}_*.csv"))
    if len(shards) < len(tools):
        print(f"WARNING: {len(shards)}/{len(tools)} shards present"); return 1
    all_df = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
    all_df = all_df.sort_values(["sequence_id", "order_key"]).reset_index(drop=True)
    out = d_feat / f"run_level_features_{args.channel_set}.csv"
    all_df.to_csv(out, index=False, encoding="utf-8-sig")
    try:
        all_df.to_parquet(d_feat / f"run_level_features_{args.channel_set}.parquet", index=False)
    except Exception as e:                                        # noqa: BLE001
        print(f"(parquet unavailable: {e}; CSV written)")
    feat_cols = MendeleyMachineToolWear.feature_columns(all_df)
    pd.DataFrame(dict(feature=feat_cols,
                      channel=[c.split("__")[0] for c in feat_cols],
                      statistic=[c.split("__")[-1] for c in feat_cols])
                 ).to_csv(d_feat / "feature_dictionary.csv", index=False, encoding="utf-8-sig")
    print(f"\nwrote {out}  ({len(all_df)} rows x {len(feat_cols)} features)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
