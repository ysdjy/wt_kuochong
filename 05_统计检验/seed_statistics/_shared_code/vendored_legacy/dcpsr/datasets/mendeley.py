"""Adapter for the Mendeley / LUH-IFW dataset

    "Multivariate time series data of milling processes with varying tool wear
     and machine tools"   (Denkena, Klemme, Stiehl; DOI 10.17632/zpxs87bjt8.3)

6418 run-level HDF5 files, 3 machines x 3 tools.  Filenames encode
M{machine}T{tool}R{run}C{cumulated_contact_time}VB{wear}.h5 and `filelist.csv`
carries the same labels (verified identical for all 6418 rows).

SEMANTICS
---------
    sequence_id = Tool   (T1..T9)  -- one independent degradation sequence,
                                      q is normalised inside it
    domain_id   = Machine(M1..M3)  -- the transfer domain

KNOWN DATA ISSUES (from the dataset documentation).  These drive the
primary/restricted channel split; they are never silently ignored.
    M2 (T4,T5,T6): aliasing on the feed-drive force and spindle torque
    M3 (T7,T8)   : machine/workpiece coordinate mismatch affecting position,
                   position control deviation and feed-drive force
    T8           : runs from the initial wear phase are absent -- the first
                   available run is already at VB = 34 um

CHANNEL SCHEMA DIFFERS BY MACHINE (verified with h5py, 2026-08-18)
--------------------------------------------------------------------------
M1 files carry a `signals_machine/torque_axis_{x,y,z}` channel that M2/M3
files do not have. M2 and M3 files instead carry `signals_machine/
force_axis_{x,y,z}`, which M1 does not have. These are NOT the same
physical quantity under two names -- they are genuinely different channels
present on different hardware, and the dataset documentation's "force_axis"
artefact name matches the M2/M3 channel EXACTLY (byte-for-byte), which makes
it a CONFIRMED match, not a guess:
    - M2 (T4,T5,T6): force_axis is the documented aliasing artefact.
    - M3 (T7,T8)   : force_axis is the documented coordinate-mismatch
                     artefact (T9 is not named in the documentation and is
                     kept primary).
    - M1 (T1,T2,T3): torque_axis_{x,y,z} carries no documented artefact and
                     is NOT related to "force_axis" by anything other than a
                     superficial name resemblance the earlier (h5py-less)
                     audit guessed at. It is treated as an ordinary primary
                     channel.
An earlier pass (before h5py was available in this environment) discovered
channels by intersecting schemas across ALL nine tools. Because force_axis
and torque_axis are each present on only 6 of 9 tools, neither survived that
intersection and the entire signals_machine group was silently dropped for
every tool. Channel discovery here is done PER MACHINE instead, so a channel
present on a machine's own files is offered to that machine's tools even
when a different machine lacks it under that name.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .base import DatasetAdapter, Task
from ..features import channel_features

FNAME_RE = re.compile(r"^M(\d+)T(\d+)R(\d+)C(\d+)VB(\d+)\.h5$")

# --- channel registry -------------------------------------------------------
GROUP_FS = {"signals_sensor": 25000.0, "signals_machine": 500.0}
TIME_CHANNELS = {"time_sensor", "time_machine"}

MACHINE_TOOLS = {"M1": ["T1", "T2", "T3"], "M2": ["T4", "T5", "T6"], "M3": ["T7", "T8", "T9"]}

# ---------------------------------------------------------------------------
# Channels with a CONFIRMED documented artefact, restricted for the tools the
# dataset documentation actually names -- never for a tool it does not name,
# and never inferred from name similarity to a channel that isn't there.
# ---------------------------------------------------------------------------
CONFIRMED_ISSUES = {
    "torque_spindle": dict(issue="aliasing (feed-drive spindle torque)",
                           scope=["T4", "T5", "T6"],
                           evidence="channel name matches the documented 'torque_spindle' "
                                    "exactly (M2 aliasing)"),
}
CONFIRMED_PREFIX_ISSUES = {
    "position_control_deviation_axis": dict(
        issue="machine/workpiece coordinate mismatch", scope=["T7", "T8"],
        evidence="channel name matches the documented 'position_control_deviation_axis' "
                 "exactly (M3 coordinate mismatch)"),
    "force_axis": dict(
        issue="aliasing (T4-T6) / machine-workpiece coordinate mismatch (T7-T8)",
        scope=["T4", "T5", "T6", "T7", "T8"],
        evidence="channel name matches the documented 'force_axis' exactly. Present on M2 "
                 "and M3 files only (M1 instead carries torque_axis_*, a separate, "
                 "undocumented channel kept primary). Restricted for T4-T6 (M2 aliasing) "
                 "and T7-T8 (M3 coordinate mismatch); T9 is not named in the documentation "
                 "and stays primary."),
}
# Documented names with NO corresponding channel anywhere in the files. Kept
# unresolved/excluded for every tool -- never mapped onto a real channel by
# guesswork.
UNRESOLVED_DOC_NAMES = {
    "position_axis": dict(
        documented_scope_coordinate=["T7", "T8"],
        candidates_in_file=["tool_position_x", "tool_position_y", "tool_position_z"],
        note=("No channel named 'position_axis' exists. tool_position_* is the only "
              "plausible candidate. NOT confirmed -- excluded from the primary set "
              "for every tool.")),
}
UNRESOLVED_CHANNELS = sorted({c for v in UNRESOLVED_DOC_NAMES.values()
                              for c in v["candidates_in_file"]})


def channel_status(ch: str) -> tuple[str, dict]:
    """Global status used for the human-readable summary report: 'restricted'
    if the channel has a confirmed artefact on ANY tool. Use
    channel_status_for_tool for the per-tool decision that actually drives
    feature extraction."""
    if ch in CONFIRMED_ISSUES:
        return "restricted", CONFIRMED_ISSUES[ch]
    for pref, meta in CONFIRMED_PREFIX_ISSUES.items():
        if ch.startswith(pref):
            return "restricted", meta
    if ch in UNRESOLVED_CHANNELS:
        which = [k for k, v in UNRESOLVED_DOC_NAMES.items() if ch in v["candidates_in_file"]]
        return "unresolved", dict(issue="UNRESOLVED", scope=[],
                                  evidence=f"possible match for documented name(s) {which}; not confirmed")
    return "primary", dict(issue="", scope=[], evidence="no documented artefact matches this channel name")


def channel_status_for_tool(ch: str, tool: str) -> tuple[str, dict]:
    """Per-tool status. A channel restricted dataset-wide by channel_status()
    is only actually restricted for the tools named in its documented scope;
    it is primary for every other tool that has it."""
    status, meta = channel_status(ch)
    if status == "restricted" and tool not in meta.get("scope", []):
        return "primary", dict(issue="", scope=[],
                               evidence=f"documented artefact scope is {meta.get('scope')}; "
                                        f"does not name {tool}")
    return status, meta


def _tool2machine(t: str) -> str:
    n = int(t[1:])
    return f"M{(n - 1) // 3 + 1}"


def _jsonable(v):
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    return v


def _measure_fs(path, datasets: dict) -> dict:
    """Sampling rate measured from the time channel actually stored in the file."""
    import h5py
    out = {}
    with h5py.File(path, "r") as h:
        for name in datasets:
            grp, _, leaf = name.rpartition("/")
            if leaf not in TIME_CHANNELS:
                continue
            try:
                t = np.asarray(h[name][()], dtype=np.float64).ravel()
                if t.size > 2:
                    dt = float(np.median(np.diff(t)))
                    out[grp or "/"] = round(1.0 / dt, 4) if dt > 0 else None
            except Exception:                                     # noqa: BLE001
                out[grp or "/"] = None
    return out


class MendeleyMachineToolWear(DatasetAdapter):
    name = "mendeley_machine_tool_wear"

    def __init__(self, raw_dir: str | Path, out_dir: str | Path,
                 channel_set: str = "primary", n_bands: int = 5):
        self.raw_dir = Path(raw_dir)
        self.out_dir = Path(out_dir)
        self.channel_set = channel_set     # "primary" | "all"
        self.n_bands = n_bands
        self.measured_fs: dict[str, float] = {}
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ meta
    def read_filelist(self) -> pd.DataFrame:
        fl = self.raw_dir / "filelist.csv"
        if not fl.exists():
            raise FileNotFoundError(fl)
        df = pd.read_csv(fl)
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={"cumulated_tool_contact_time": "ctime", "wear": "VB"})
        p = df["filename"].str.extract(FNAME_RE).astype(float)
        p.columns = ["m", "t", "r", "c", "vb"]
        bad = ((p["m"] != df["machine"]) | (p["t"] != df["tool"]) |
               (p["r"] != df["run"]) | (p["c"] != df["ctime"]) | (p["vb"] != df["VB"]))
        if bad.any():
            raise ValueError(f"{int(bad.sum())} rows where filename and filelist.csv disagree")
        df["domain_id"] = "M" + df["machine"].astype(int).astype(str)
        df["sequence_id"] = "T" + df["tool"].astype(int).astype(str)
        df["order_key"] = df["run"].astype(int)
        df["source_file"] = df["filename"]
        return df[["sequence_id", "domain_id", "order_key", "VB", "ctime",
                   "source_file"]].sort_values(["sequence_id", "order_key"]).reset_index(drop=True)

    # --------------------------------------------------------------- schema
    def scan_schema(self, probe_files: list[str] | None = None) -> dict:
        """Open a few files and record the true HDF5 tree. Requires h5py."""
        import h5py
        meta = self.read_filelist()
        if probe_files is None:
            probe_files = []
            for tool in ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]:
                sub = meta[meta.sequence_id == tool]
                if len(sub):
                    for frac in (0.0, 0.5, 0.95):
                        probe_files.append(sub.iloc[int(frac * (len(sub) - 1))]["source_file"])
        report: dict = {"dataset": self.name, "files": {},
                        "unresolved_documented_names": UNRESOLVED_DOC_NAMES}
        for fn in probe_files:
            path = self.raw_dir / fn
            entry: dict = {"root_attrs": {}, "datasets": {}}
            with h5py.File(path, "r") as h:
                entry["root_attrs"] = {k: _jsonable(v) for k, v in h.attrs.items()}

                def visit(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        entry["datasets"][name] = dict(
                            shape=list(obj.shape), dtype=str(obj.dtype),
                            compression=obj.compression,
                            attrs={k: _jsonable(v) for k, v in obj.attrs.items()})
                h.visititems(visit)
            entry["fs_measured"] = _measure_fs(path, entry["datasets"])
            report["files"][fn] = entry
        report["channels"] = sorted({n.split("/")[-1]
                                     for e in report["files"].values()
                                     for n in e["datasets"]})
        return report

    def discover_channels_by_domain(self, schema: dict | None = None
                                    ) -> dict[str, dict[str, list[str]]]:
        """{domain: {group: [channel,...]}}, time axes removed. A channel is
        offered to a machine's tools when it is common across THAT machine's
        own probed files -- it is never dropped just because a different
        machine's files lack it under the same name."""
        rep = schema or self.scan_schema()
        by_domain: dict[str, list[set]] = {}
        for fn, e in rep["files"].items():
            m = re.match(r"^M(\d+)", fn)
            dom = f"M{m.group(1)}"
            by_domain.setdefault(dom, []).append({n for n in e["datasets"]})
        out: dict[str, dict[str, list[str]]] = {}
        for dom, sets in by_domain.items():
            common = set.intersection(*sets) if sets else set()
            groups: dict[str, list[str]] = {}
            for full in sorted(common):
                grp, _, leaf = full.rpartition("/")
                if leaf in TIME_CHANNELS:
                    continue
                groups.setdefault(grp or "/", []).append(leaf)
            out[dom] = groups
        return out

    # kept for backward compatibility with callers expecting a flat map:
    # channels common across every probed tool (used only for the printed
    # cross-machine summary, never for feature selection).
    def discover_channels(self, schema: dict | None = None) -> dict[str, list[str]]:
        by_dom = self.discover_channels_by_domain(schema)
        per_dom_sets = [{(g, c) for g, cs in groups.items() for c in cs} for groups in by_dom.values()]
        common = set.intersection(*per_dom_sets) if per_dom_sets else set()
        groups: dict[str, list[str]] = {}
        for g, c in sorted(common):
            groups.setdefault(g, []).append(c)
        return groups

    def primary_channel_set(self, by_domain: dict[str, dict[str, list[str]]]) -> dict:
        """Per-tool resolved buckets: {tool: {"primary": [...], "restricted": [...],
        "unresolved": [...]}}, plus a flat 'primary' dataset-wide bucket used only
        for reporting which channel NAMES appear in the primary set anywhere."""
        per_tool: dict[str, dict[str, list[tuple[str, str]]]] = {}
        for dom, groups in by_domain.items():
            for tool in MACHINE_TOOLS[dom]:
                buckets: dict[str, list[tuple[str, str]]] = {"primary": [], "restricted": [], "unresolved": []}
                for grp, chans in groups.items():
                    for ch in chans:
                        st, _ = channel_status_for_tool(ch, tool)
                        buckets[st].append((grp, ch))
                per_tool[tool] = buckets
        flat_primary: dict[str, list[str]] = {}
        for tool, b in per_tool.items():
            for grp, ch in b["primary"]:
                flat_primary.setdefault(grp, [])
                if ch not in flat_primary[grp]:
                    flat_primary[grp].append(ch)
        return {"by_tool": per_tool, "primary": flat_primary,
                "policy": ("Version A (primary) is used for the main experiments. Restriction "
                           "is resolved PER TOOL against the documented scope (e.g. force_axis "
                           "is restricted for T4-T8 but primary for T9). Restricted channels have "
                           "a documented artefact for that specific tool. UNRESOLVED channels are "
                           "documented-name candidates that could NOT be confirmed against the "
                           "files; they are kept out of the primary set for every tool."),
                "unresolved_documented_names": UNRESOLVED_DOC_NAMES}

    def excluded_channel_report(self, by_domain: dict[str, dict[str, list[str]]]) -> pd.DataFrame:
        """One row per (group, channel): which tools see it as primary vs
        restricted vs unresolved, and why. Nothing is hidden behind a single
        dataset-wide flag."""
        all_chans: dict[tuple[str, str], set[str]] = {}
        for dom, groups in by_domain.items():
            for grp, chans in groups.items():
                for ch in chans:
                    all_chans.setdefault((grp, ch), set()).update(MACHINE_TOOLS[dom])
        rows = []
        for (grp, ch), tools_with_channel in all_chans.items():
            restricted_for, primary_for = [], []
            reason = evidence = ""
            for t in sorted(tools_with_channel, key=lambda x: int(x[1:])):
                st, meta = channel_status_for_tool(ch, t)
                if st == "restricted":
                    restricted_for.append(t)
                    reason, evidence = meta.get("issue", ""), meta.get("evidence", "")
                elif st == "unresolved":
                    reason, evidence = meta.get("issue", ""), meta.get("evidence", "")
                else:
                    primary_for.append(t)
            global_status, _ = channel_status(ch)
            rows.append(dict(channel=ch, group=grp, status=global_status,
                             present_on_tools=",".join(sorted(tools_with_channel, key=lambda x: int(x[1:]))),
                             tools_affected=",".join(restricted_for),
                             tools_primary=",".join(primary_for),
                             reason=reason, evidence=evidence,
                             used_in_primary_experiment=bool(primary_for),
                             used_for_every_tool_with_this_channel=(len(primary_for) == len(tools_with_channel)),
                             retained_in_raw_data=True))
        return pd.DataFrame(rows).sort_values(["status", "group", "channel"]).reset_index(drop=True)

    def channel_summary(self, report: dict) -> pd.DataFrame:
        """One row per (file, dataset) from the REAL h5 scan -- shape, dtype,
        compression, attrs, and the fs actually implied by the time channel."""
        rows = []
        for fn, e in report["files"].items():
            tool_num = int(re.match(r"^M\d+T(\d+)", fn).group(1))
            tool = f"T{tool_num}"
            for name, meta in e["datasets"].items():
                grp, _, leaf = name.rpartition("/")
                st, ev = channel_status_for_tool(leaf, tool)
                rows.append(dict(file=fn, tool=tool, group=grp or "/", channel=leaf, path=name,
                                 shape="x".join(map(str, meta["shape"])),
                                 n_samples=meta["shape"][0] if meta["shape"] else 0,
                                 dtype=meta["dtype"], compression=meta.get("compression"),
                                 attrs=json.dumps(meta.get("attrs", {}), ensure_ascii=False),
                                 fs_measured_hz=e.get("fs_measured", {}).get(grp or "/"),
                                 status=st, status_evidence=ev.get("evidence", "")))
        return pd.DataFrame(rows)

    # ------------------------------------------------------- feature build
    def channels_for_tool(self, tool: str, by_domain: dict[str, dict[str, list[str]]],
                          channel_set: str | None = None) -> list[tuple[str, str, float]]:
        """Resolved (group, channel, fs) list for ONE tool. channel_set='primary'
        excludes anything restricted or unresolved for THIS tool; 'all' includes
        every channel physically present on this tool's machine."""
        cs = channel_set or self.channel_set
        dom = _tool2machine(tool)
        groups = by_domain.get(dom, {})
        out = []
        for grp, chans in groups.items():
            fs = self.measured_fs.get(grp, GROUP_FS.get(grp.strip("/"), np.nan))
            for ch in chans:
                if cs == "primary":
                    st, _ = channel_status_for_tool(ch, tool)
                    if st != "primary":
                        continue
                out.append((grp, ch, fs))
        return sorted(out, key=lambda t: (t[0], t[1]))

    def channels_for(self, groups: dict[str, list[str]]) -> list[tuple[str, str, float]]:
        """Deprecated flat variant retained for callers that want a single
        dataset-wide list (reporting only). Feature extraction must use
        channels_for_tool so machine-specific channels are not dropped."""
        out = []
        for grp, chans in groups.items():
            fs = self.measured_fs.get(grp, GROUP_FS.get(grp.strip("/"), np.nan))
            for ch in chans:
                out.append((grp, ch, fs))
        return sorted(out, key=lambda t: (t[0], t[1]))

    def extract_features(self, channels, files: pd.DataFrame,
                         progress=None) -> pd.DataFrame:
        """Feature rows for the given subset of files. Requires h5py.
        `channels` is the (group, channel, fs) list for THIS subset's tool
        (from channels_for_tool) -- callers must not mix tools in one call."""
        import h5py
        rows = []
        for i, r in enumerate(files.itertuples(index=False)):
            path = self.raw_dir / r.source_file
            row = dict(sequence_id=r.sequence_id, domain_id=r.domain_id,
                       order_key=int(r.order_key), VB=float(r.VB),
                       ctime=float(r.ctime), source_file=r.source_file)
            try:
                with h5py.File(path, "r") as h:
                    for grp, ch, fs in channels:
                        key = f"{grp}/{ch}" if grp not in ("", "/") else ch
                        if key not in h:
                            row.update({k: np.nan for k in
                                        channel_features(np.array([]), fs, ch, self.n_bands)})
                            continue
                        x = np.asarray(h[key][()], dtype=np.float64).ravel()
                        row.update(channel_features(x, fs, ch, self.n_bands))
                row["_read_error"] = ""
            except Exception as exc:                       # noqa: BLE001
                row["_read_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            if progress and (i + 1) % 50 == 0:
                progress(i + 1, len(files))
        return pd.DataFrame(rows)

    def load_run_level_table(self) -> pd.DataFrame:
        f = self.out_dir / f"run_level_features_{self.channel_set}.csv"
        if not f.exists():
            raise FileNotFoundError(
                f"{f} not found -- run scripts/00_extract_features.py first")
        df = pd.read_csv(f)
        self.validate_table(df)
        return df

    # ------------------------------------------------------------- protocol
    def task_definitions(self) -> list[Task]:
        M = MACHINE_TOOLS
        tasks: list[Task] = []
        # Group A -- leave-one-machine-out, dual source
        for i, (name, held) in enumerate([("D1-M", "M3"), ("D2-M", "M2"), ("D3-M", "M1")], 1):
            tr = [m for m in M if m != held]
            tasks.append(Task(name=name, group="cross_machine_dual",
                              train_domains=tr, test_domains=[held],
                              train_sequences=[t for m in tr for t in M[m]],
                              test_sequences=M[held],
                              notes=f"{'+'.join(tr)} -> {held}"))
        # Group B -- single source, all six directions
        k = 0
        for src in ["M1", "M2", "M3"]:
            for dst in ["M1", "M2", "M3"]:
                if src == dst:
                    continue
                k += 1
                tasks.append(Task(name=f"MS{k}", group="cross_machine_single",
                                  train_domains=[src], test_domains=[dst],
                                  train_sequences=M[src], test_sequences=M[dst],
                                  notes=f"{src} -> {dst}"))
        # Group C -- leave-one-tool-out
        all_tools = [t for m in M for t in M[m]]
        for t in all_tools:
            home = _tool2machine(t)
            tasks.append(Task(
                name=f"LOTO_{t}", group="leave_one_tool_out",
                train_domains=list(M), test_domains=[home],
                train_sequences=[x for x in all_tools if x != t], test_sequences=[t],
                notes=(f"held-out tool {t} (machine {home}); same-machine tools in "
                       f"train: {[x for x in M[home] if x != t]}"
                       + ("; WARNING early stage truncated (first run already at VB=34um)"
                          if t == "T8" else ""))))
        for t in tasks:
            t.validate()
        return tasks
