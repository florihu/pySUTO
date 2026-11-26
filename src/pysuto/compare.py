"""
compare.py
==========
``python -m pysuto compare``: compare two chains of runs, array by array, and say
where they first differ. A chain is named by its leaf run id(s); its lineage is walked
through ``parents.upstream`` in each run's ``metadata.json``::

    python -m pysuto compare --data-dir data --base psut_<ref> --new psut_<check>

A leaf names the chain it implies: itself and its ancestors. Siblings are not ancestors,
so a chain whose last stage has two children (psut and uncprop both hang off the opt)
takes **both** leaves: ``--base psut_<a> uncprop_<b>``. Two different runs of one stage in
one chain are refused.

The two chains are paired stage by stage (feed → imp → const → opt → uncprop → psut; feed
runs by their feed name). A stage only one chain has is *skipped* and listed, not a
difference. Per paired run the command compares

* every stored array of every ``.npy`` / ``.npz`` file and every column (and the index) of
  every ``.parquet`` file (the imputation log among them): dtype, shape, NaN positions and
  values, exactly (no tolerance); a file only one run has is a difference;
* ``parents.inputs`` of the two sidecars, as stored.

Stage 2 stores its per-feed rows (``F``, ``Sigma``, ``Omega``) in the order of its
``--feed-id`` arguments, recorded in ``feed_ids``. When two files hold the same feeds in
another order, the new file's rows are aligned to the base's ``feed_ids`` before they are
compared, and the run carries the note "feed order differs": a reordering alone is
no difference.

Not compared: the sidecars' run ids, timestamps and ``stage_config``, ``diag/`` and the
feed's ``feed.pkl`` (a pickle, whose bytes are no array). The command reads and writes
nothing under the data directory (no log file, no lock, no registry).

Exit code: 0 when every paired run is identical, 1 on any difference, 2 on a usage error
(an unknown run id, no stage in common, a chain with two runs of one stage).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

STAGES = ("feed", "imp", "const", "opt", "uncprop", "psut")

# parents.upstream key → stage of the run it names (a list for the feeds)
_UPSTREAM_STAGE = {"feed_run_ids": "feed", "imp_run_id": "imp", "const_run_id": "const",
                   "opt_run_id": "opt"}
_SKIP_DIRS  = ("diag",)
_SKIP_FILES = ("metadata.json", "feed.pkl")
# Stage 2's per-feed rows, stored in --feed-id order and aligned by feed_ids
_FEED_IDS   = "feed_ids"
_PER_FEED   = ("F", "Sigma", "Omega", _FEED_IDS)


class CompareError(ValueError):
    """A usage error: an unknown run, an empty overlap, two runs of one stage."""


@dataclass
class RunDiff:
    """The comparison of one paired run."""

    stage:       str
    base:        str
    new:         str
    n_arrays:    int  = 0
    differences: list = field(default_factory=list)   # one line each
    notes:       list = field(default_factory=list)   # one line each, no difference


@dataclass
class Report:
    """The comparison of two chains."""

    runs:    list = field(default_factory=list)       # RunDiff, stage order
    skipped: list = field(default_factory=list)       # (stage, label, side)

    @property
    def identical(self) -> bool:
        return not any(r.differences for r in self.runs)


# =============================================================================
# Lineage
# =============================================================================

def run_stage(run_id: str) -> str:
    """The stage a run id names (its first token); ``CompareError`` if unknown."""
    stage = run_id.split("_", 1)[0]
    if stage not in STAGES:
        raise CompareError(f"{run_id!r} is not a run id of a stage in {STAGES}")
    return stage


def run_dir(data_dir: str, run_id: str) -> str:
    """The run's directory under ``<data_dir>/proc``; ``CompareError`` if it is absent."""
    stage = run_stage(run_id)
    for sub in ((stage, "registry"), (stage,)):
        path = os.path.join(data_dir, "proc", *sub, run_id)
        if os.path.isfile(os.path.join(path, "metadata.json")):
            return path
    raise CompareError(f"run {run_id!r} has no metadata.json under {data_dir}/proc")


def _sidecar(data_dir: str, run_id: str) -> dict:
    with open(os.path.join(run_dir(data_dir, run_id), "metadata.json"),
              encoding="utf-8") as f:
        return json.load(f)


def feed_name(run_id: str) -> str:
    """``feed_<name>_<date>_<time>_<hex>`` → ``<name>`` (a feed name has no date part)."""
    parts = run_id.split("_")
    return "_".join(parts[1:-3])


def lineage(data_dir: str, leaves) -> dict:
    """The chain a set of leaf run ids implies, as ``{stage: [run ids]}``.

    Walks ``parents.upstream`` from each leaf to its feeds. Every stage but ``feed`` holds
    one run; two different runs of one stage raise :class:`CompareError`.
    """
    chain: dict = {}
    todo = list(leaves)
    while todo:
        run_id = todo.pop()
        stage = run_stage(run_id)
        held = chain.setdefault(stage, [])
        if run_id in held:
            continue
        if stage != "feed" and held:
            raise CompareError(f"two runs of stage {stage!r} in one chain: "
                               f"{held[0]} and {run_id}")
        held.append(run_id)
        upstream = _sidecar(data_dir, run_id).get("parents", {}).get("upstream", {})
        for key, up_stage in _UPSTREAM_STAGE.items():
            ids = upstream.get(key)
            if ids is None:
                continue
            todo.extend([ids] if isinstance(ids, str) else ids)
            for i in ([ids] if isinstance(ids, str) else ids):
                if run_stage(i) != up_stage:
                    raise CompareError(f"{run_id}: upstream {key} names {i!r}")
    return chain


# =============================================================================
# Arrays of a run
# =============================================================================

def _walk(path: str) -> list:
    """Relative paths of the compared files of a run, sorted."""
    found = []
    for top, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for name in files:
            if name not in _SKIP_FILES:
                rel = os.path.relpath(os.path.join(top, name), path)
                found.append(rel.replace(os.sep, "/"))    # one key on every platform
    return sorted(found)


def _arrays(path: str):
    """``(name, array)`` of every array of one file; the file is read lazily per key."""
    low = path.lower()
    if low.endswith(".npy"):
        yield "", np.load(path, allow_pickle=True)
    elif low.endswith(".npz"):
        with np.load(path, allow_pickle=True) as z:   # our own run files
            for key in z.files:
                yield key, z[key]
    elif low.endswith(".parquet"):
        frame = pd.read_parquet(path)
        yield "[index]", np.asarray(frame.index)
        for col in frame.columns:
            yield f"[{col}]", frame[col].to_numpy()


def _mismatch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Boolean mask of the elements that differ; NaN equals NaN, a field at a time."""
    if a.dtype.names:
        mask = np.zeros(a.shape, dtype=bool)
        for name in a.dtype.names:
            mask |= _mismatch(a[name], b[name])
        return mask
    if a.dtype == object:
        na, nb = pd.isna(a), pd.isna(b)
        mask = (na != nb).ravel()
        fa, fb = a.ravel(), b.ravel()
        for i in np.flatnonzero((~na & ~nb).ravel()):
            mask[i] = not np.array_equal(fa[i], fb[i])
        return mask.reshape(a.shape)
    if a.dtype.kind in "mM":
        na, nb = np.isnat(a), np.isnat(b)
        return (na != nb) | (~na & ~nb & (a != b))
    if a.dtype.kind in "fc":
        na, nb = np.isnan(a), np.isnan(b)
        return (na != nb) | (~na & ~nb & (a != b))
    return a != b


def first_difference(a: np.ndarray, b: np.ndarray):
    """``None`` when identical, else a one-line reason (dtype, shape or first index)."""
    if a.dtype != b.dtype:
        return f"dtype {a.dtype} vs {b.dtype}"
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}"
    if a.size == 0:
        return None
    mask = _mismatch(a, b)
    if not mask.any():
        return None
    flat = int(np.argmax(mask.ravel()))
    idx  = tuple(int(i) for i in np.unravel_index(flat, a.shape))
    return (f"first at index {idx}: {_short(a.ravel()[flat])} vs "
            f"{_short(b.ravel()[flat])} ({int(mask.sum())} of {a.size} differ)")


def _feed_ids(path: str):
    """The ``feed_ids`` of an ``.npz`` file as a list, or ``None`` if it has none."""
    if not path.lower().endswith(".npz"):
        return None
    with np.load(path, allow_pickle=True) as z:
        return [str(f) for f in z[_FEED_IDS]] if _FEED_IDS in z.files else None


def feed_permutation(base_ids, new_ids):
    """Row indices that put the new file's per-feed rows in the base's feed order.

    ``None`` when the orders agree, or when the two files do not hold the same distinct
    feeds (their ``feed_ids`` then differ as an array).
    """
    if base_ids is None or new_ids is None or base_ids == new_ids:
        return None
    if sorted(base_ids) != sorted(new_ids) or len(set(base_ids)) != len(base_ids):
        return None
    return np.array([new_ids.index(f) for f in base_ids], dtype=np.int64)


def _label(rel: str, name: str) -> str:
    return f"{rel}::{name}" if name else rel


def _short(x) -> str:
    return repr(x.item() if isinstance(x, np.generic) and x.dtype.kind != "V" else x)[:60]


# =============================================================================
# Runs and chains
# =============================================================================

def compare_runs(stage: str, base_dir: str, new_dir: str, base_id: str,
                 new_id: str) -> RunDiff:
    """Compare two runs of one stage: their arrays, then their ``parents.inputs``."""
    out = RunDiff(stage=stage, base=base_id, new=new_id)
    base_files, new_files = _walk(base_dir), _walk(new_dir)
    for rel in sorted(set(base_files) | set(new_files)):
        if rel not in new_files:
            out.differences.append(f"{rel}: only in base")
            continue
        if rel not in base_files:
            out.differences.append(f"{rel}: only in new")
            continue
        base_path, new_path = os.path.join(base_dir, rel), os.path.join(new_dir, rel)
        base_ids, new_ids = _feed_ids(base_path), _feed_ids(new_path)
        perm = feed_permutation(base_ids, new_ids)
        if perm is not None:
            out.notes.append(f"{rel}: feed order differs: base {base_ids} vs new "
                             f"{new_ids}; per-feed rows aligned by {_FEED_IDS}")
        base_it, new_it = _arrays(base_path), _arrays(new_path)
        seen_base = {}
        for name, arr in base_it:
            seen_base[name] = arr
        seen_new = set()
        for name, arr in new_it:
            seen_new.add(name)
            if name not in seen_base:
                out.differences.append(f"{_label(rel, name)}: only in new")
                continue
            if (perm is not None and name in _PER_FEED and arr.ndim
                    and len(arr) == len(perm)):
                arr = arr[perm]
            out.n_arrays += 1
            why = first_difference(seen_base.pop(name), arr)
            if why:
                out.differences.append(f"{_label(rel, name)}: {why}")
        for name in seen_base:
            out.differences.append(f"{_label(rel, name)}: only in base")
    in_base = _inputs(base_dir)
    in_new  = _inputs(new_dir)
    if in_base != in_new:
        out.differences.append(f"parents.inputs: base {in_base} vs new {in_new}")
    return out


def _inputs(path: str):
    with open(os.path.join(path, "metadata.json"), encoding="utf-8") as f:
        return json.load(f).get("parents", {}).get("inputs", {})


def _by_feed_name(run_ids) -> dict:
    named = {}
    for run_id in run_ids:
        if feed_name(run_id) in named:
            raise CompareError(f"two feed runs named {feed_name(run_id)!r} in one chain: "
                               f"{named[feed_name(run_id)]} and {run_id}")
        named[feed_name(run_id)] = run_id
    return named


def compare_chains(data_dir: str, base, new) -> Report:
    """Compare the chains implied by the leaf run ids ``base`` and ``new``."""
    chain_b, chain_n = lineage(data_dir, base), lineage(data_dir, new)
    report = Report()
    for stage in STAGES:
        runs_b, runs_n = chain_b.get(stage, []), chain_n.get(stage, [])
        if stage == "feed":
            by_b, by_n = _by_feed_name(runs_b), _by_feed_name(runs_n)
            pairs = [(by_b[k], by_n[k]) for k in by_b if k in by_n]
            for k in by_b:
                if k not in by_n:
                    report.skipped.append((stage, by_b[k], "new has no such feed"))
            for k in by_n:
                if k not in by_b:
                    report.skipped.append((stage, by_n[k], "base has no such feed"))
        elif runs_b and runs_n:
            pairs = [(runs_b[0], runs_n[0])]
        else:
            pairs = []
            if runs_b or runs_n:
                report.skipped.append((stage, (runs_b or runs_n)[0],
                                       "new has none" if runs_b else "base has none"))
        for rb, rn in pairs:
            report.runs.append(compare_runs(stage, run_dir(data_dir, rb),
                                            run_dir(data_dir, rn), rb, rn))
    if not report.runs:
        raise CompareError("the two chains have no stage in common")
    return report


def format_report(report: Report) -> str:
    """The report as text: one line per paired run, its differences indented."""
    lines = []
    for r in report.runs:
        verdict = "DIFFERENT" if r.differences else "identical"
        lines.append(f"{r.stage:8s}{verdict:10s}{r.n_arrays:5d} arrays  "
                     f"{r.base}  {r.new}")
        lines.extend(f"    note: {n}" for n in r.notes)
        lines.extend(f"    {d}" for d in r.differences)
    for stage, run_id, why in report.skipped:
        lines.append(f"{stage:8s}skipped   {run_id}: {why}")
    lines.append("chains identical" if report.identical else "chains DIFFER")
    return "\n".join(lines)
