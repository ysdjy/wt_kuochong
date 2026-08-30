# data/

```
data/
└─ PHM2010/
   ├─ raw/          # NOT committed (18GB+) -- populated by scripts/download_phm2010.py
   │  └─ c{1..6}/c{1..6}/c_{cond}_{run:03d}.csv   (7-channel signal, no header)
   │  └─ c{1..6}/c{cond}_wear.csv                  (columns: cut, flute_1, flute_2, flute_3)
   └─ features/
      └─ run_level_features_all.csv   # COMMITTED (5.4MB) -- see provenance below
```

## `features/run_level_features_all.csv`

- **sha256**: `6e8affeb681d0b386e453421a0df7a66932138199eb236403d27b797c11eeb88`
- **Size**: 5,458,199 bytes
- **Provenance**: this exact file is copied byte-for-byte from the old parent
  project's `baselines/htt_net/data/run_level_features_all.csv` (verified via
  sha256 match at copy time, 2026-08-30). It is the authoritative run-level
  feature table used by every published D1/D2/D3 result for the window-based
  methods (B1 RF, B2 TCN-GRU, B3 Multi-task TCN-GRU, B9 DC-PHSR, B4 HTT-Net).
- **Why it is committed directly rather than regenerated**: the original raw-signal
  feature-extraction script that produced this file was lost — see the old
  project's `baselines/htt_net/README.md` "Data availability" section. This file
  was itself recovered from a cached intermediate output
  (`补充材料/小论文/阶段分类前传/1.1阶段分类/01_intermediate/loaded_feature_table_with_condition_relative_stage.csv`),
  not regenerated from raw signal. A best-effort regenerator does exist
  (`scripts/build_phm2010_features.py`, itself ported from the old project's
  `baselines/htt_net/data/build_run_level_features.py`), but its own docstring
  states its output is **not bit-identical** to this file — kept only as an
  extensibility fallback for a future dataset the original extractor is
  unavailable for, never as a substitute for this CSV on PHM2010 itself.
- **Columns**: `condition, run_id, file_name, signal_len, n_channels, flute_1,
  flute_2, flute_3, VB, VB_mean, dominant_flute`, plus per-channel (ch1..ch7)
  statistical/spectral features (`ch{n}_mean, ch{n}_std, ..., ch{n}_wp_ener...`).
  5.4MB total across 315 runs x 3 conditions.
- No license restriction: PHM2010 is a public benchmark dataset and this is a
  small derived feature table, not a redistribution of the raw signal archive.

## `raw/` (not committed)

Raw PHM2010 signal data. Source: Kaggle dataset **`tobbyrui/phm2010`** — this is
the only concretely-named public source of this data anywhere in the old parent
project (`baselines/htt_net/data/build_run_level_features.py` docstring and
`baselines/htt_net/README.md`, both verbatim). No other URL/handle for this
dataset appears in the old project's code or docs — do not substitute a
different or invented source.

Fetch it with:

```bash
python scripts/download_phm2010.py
python scripts/verify_phm2010.py
```

See `MANUAL_RUN.md` for one-time Kaggle credential setup (API token; never
committed to git — see `.gitignore`).
