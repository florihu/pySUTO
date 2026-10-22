"""
issue106_driver_gate.py
=======================
#106 measurement: the knob surface (#100, #104), the stage lock (#101), the pipeline
(#103) and the scratch-store flags (#108) moved nothing. Two chains are run from the
reference chain's feed runs through ``pysuto.pipeline`` and compared with it:

    chain 2   ``Pipeline(("imp", "const", "opt"), base).run({})`` — no overrides, every
              run written to ``data/proc`` as a hand-run driver writes it;
    chain 3   every knob of every driver passed explicitly through ``--set`` at its
              default (read from each driver's ``KNOBS``), its imp and const runs
              written to a scratch store via ``--imp-dir`` / ``--const-dir`` (the
              sweep's path, #108).

Decision rule, fixed in this docstring before either chain ran:

    1  chain 2: every array stored by its imp, const and opt runs — all files but
       ``metadata.json`` and ``diag/``; npz/npy arrays by ``np.array_equal`` with NaN
       equal to NaN, parquet frames by ``DataFrame.equals`` — equals the reference's
       bit for bit, ``F_opt`` in all 30 years included;
    2  chain 3: the same;
    3  sidecars: ``stage_config["overrides"]`` is ``{}`` on chain 2's three runs and,
       on chain 3's, the full default dict of that driver (compared as JSON);
       ``feasibility_check`` reads ``"first"`` on both opt runs;
    4  no ``data/proc/.stage.lock`` is left after either chain.

The parallel solve is bit-identical across runs (#71), so equality is the criterion,
not a tolerance; any difference is a bug in the build tickets.

    python scripts/explo/issue106_driver_gate.py run [--ref <opt_run_id>]
    python scripts/explo/issue106_driver_gate.py compare <opt_chain2> <opt_chain3>
                                                         [--ref <opt_run_id>]

``run`` runs both chains, one after another (≈ 40 min; in the background), then
compares; ``compare`` re-reads finished chains. Where an imputation log differs, a
diagnostic (not part of the rule) compares it as a multiset: both frames sorted by
every column. The per-year F_opt table goes to
``docs/rep/figs/issue106/``. Chain 3's scratch runs are removed after a passing compare.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from pysuto.pipeline import Chain, Pipeline                       # noqa: E402

REF      = "opt_20261021_131744_58504ad3"      # the final reference chain (#110)
PROC     = os.path.join("data", "proc")
SCRATCH  = "/dev/shm/pysuto-gate"
FIGS     = os.path.join("docs", "rep", "figs", "issue106")
DRIVERS  = {"imp": "collect_aggregate_impute.py", "const": "constraints.py",
            "opt": "run_solver.py"}
SKIP     = {"metadata.json", "diag"}


# ---------------------------------------------------------------------------
# chains
# ---------------------------------------------------------------------------

def driver_defaults() -> dict:
    """``<stage>.<KNOB>`` → the driver's default, for every knob of every driver."""
    sys.path.insert(0, "scripts")
    out = {}
    for stage, script in DRIVERS.items():
        spec = importlib.util.spec_from_file_location(f"_gate_{stage}",
                                                      os.path.join("scripts", script))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out.update({f"{stage}.{k}": getattr(mod, k) for k in mod.KNOBS})
    return out


def run_chains(ref: str) -> tuple:
    base = Chain.from_opt_run(ref)
    two = Pipeline(base=base, cwd=os.getcwd()).run({}, label="r106 zero overrides")
    print(f"chain 2: {two}", flush=True)
    lock_after = [os.path.exists(os.path.join(PROC, ".stage.lock"))]
    dirs = {s: os.path.join(SCRATCH, s) for s in ("imp", "const")}
    three = Pipeline(base=base, cwd=os.getcwd(), run_dirs=dirs).run(
        driver_defaults(), label="r106 explicit defaults")
    print(f"chain 3: {three}", flush=True)
    lock_after.append(os.path.exists(os.path.join(PROC, ".stage.lock")))
    return two, three, lock_after


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

def _equal(a, b) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype.kind == "f":
        return bool(np.array_equal(a, b, equal_nan=True))
    if a.dtype.names:                                   # structured (iter_history)
        return all(_equal(a[n], b[n]) for n in a.dtype.names)
    if a.dtype.kind == "O":
        return pd.Series(a.ravel()).equals(pd.Series(b.ravel()))
    return bool(np.array_equal(a, b))


def compare_dirs(new: str, ref: str) -> list:
    """(relative path, key, equal) for every stored array under ``ref``; a file or key
    missing on either side is unequal."""
    rows = []
    for root, dirs, files in os.walk(ref):
        dirs[:] = sorted(d for d in dirs if d not in SKIP)
        for f in sorted(files):
            if f in SKIP:
                continue
            rel = os.path.relpath(os.path.join(root, f), ref)
            p, q = os.path.join(new, rel), os.path.join(ref, rel)
            if not os.path.exists(p):
                rows.append((rel, "<file>", False))
            elif f.endswith(".npz"):
                a, b = np.load(p, allow_pickle=True), np.load(q, allow_pickle=True)
                for k in sorted(set(a.files) | set(b.files)):
                    rows.append((rel, k, k in a.files and k in b.files
                                 and _equal(a[k], b[k])))
            elif f.endswith(".npy"):
                rows.append((rel, "", _equal(np.load(p, allow_pickle=True),
                                             np.load(q, allow_pickle=True))))
            elif f.endswith(".parquet"):
                rows.append((rel, "", pd.read_parquet(p).equals(pd.read_parquet(q))))
            else:
                with open(p, "rb") as fa, open(q, "rb") as fb:
                    rows.append((rel, "<bytes>", fa.read() == fb.read()))
    extra = []
    for root, dirs, files in os.walk(new):
        dirs[:] = [d for d in dirs if d not in SKIP]
        extra += [os.path.relpath(os.path.join(root, f), new) for f in files
                  if f not in SKIP and not os.path.exists(
                      os.path.join(ref, os.path.relpath(os.path.join(root, f), new)))]
    rows += [(rel, "<only in new>", False) for rel in extra]
    return rows


def log_as_multiset(new: str, ref: str) -> tuple:
    """(equal once sorted by every column, rows out of place) of two imputation logs."""
    a, b = (pd.read_parquet(os.path.join(d, "imputation_log.parquet")) for d in (new, ref))

    def canon(df):
        df = df.copy()
        for c in ("source_edges", "coefficients"):
            df[c] = df[c].map(lambda v: tuple(np.asarray(v).tolist()))
        return df.sort_values(list(df.columns), kind="mergesort").reset_index(drop=True)
    moved = int((a["edge_id"].to_numpy() != b["edge_id"].to_numpy()).sum()) \
        if len(a) == len(b) else -1
    return canon(a).equals(canon(b)), moved


def _sidecar(d: str) -> dict:
    with open(os.path.join(d, "metadata.json"), encoding="utf-8") as f:
        return json.load(f)


def _scratch_chain(opt: str) -> Chain:
    """Chain 3: its opt run in data/proc, its const and imp runs in the scratch store."""
    const = _sidecar(os.path.join(PROC, "opt", opt))["parents"]["upstream"]["const_run_id"]
    up = _sidecar(os.path.join(SCRATCH, "const", const))["parents"]["upstream"]
    imp = up["imp_run_id"]
    feeds = _sidecar(os.path.join(SCRATCH, "imp", imp))["parents"]["upstream"]["feed_run_ids"]
    return Chain(feed_run_ids=tuple(feeds), imp=imp, const=const, opt=opt)


def _plain(v):
    return json.loads(json.dumps(v))


def compare(two: Chain, three: Chain, ref: Chain, defaults: dict) -> bool:
    where = {"two":   {s: os.path.join(PROC, s) for s in ("imp", "const", "opt")},
             "three": {"imp": os.path.join(SCRATCH, "imp"),
                       "const": os.path.join(SCRATCH, "const"),
                       "opt": os.path.join(PROC, "opt")}}
    ok, years = True, []
    for name, chain in (("two", two), ("three", three)):
        print(f"\n== chain {name}: {chain.imp} → {chain.const} → {chain.opt}")
        for stage in ("imp", "const", "opt"):
            new = os.path.join(where[name][stage], getattr(chain, stage))
            rows = compare_dirs(new, os.path.join(PROC, stage, getattr(ref, stage)))
            bad = [r for r in rows if not r[2]]
            print(f"   {stage}: {len(rows) - len(bad)}/{len(rows)} arrays identical"
                  + ("" if not bad else f"; DIFFER: {bad[:8]}"))
            ok &= not bad
            if stage == "imp" and any(r[0] == "imputation_log.parquet" for r in bad):
                same, moved = log_as_multiset(new, os.path.join(PROC, "imp", ref.imp))
                print(f"   imp log as a multiset (diagnostic): "
                      f"{'identical' if same else 'DIFFERENT'}; {moved} rows out of place")
            if stage == "opt":
                years += [(name, r[0].split("_")[-1].split(".")[0], r[2])
                          for r in rows if r[1] == "F_opt"]
            sc = _sidecar(new)["stage_config"]
            want = ({} if name == "two" else
                    _plain({k.split(".", 1)[1]: v for k, v in defaults.items()
                            if k.startswith(stage + ".")}))
            got = sc.get("overrides")
            print(f"   {stage} overrides: {'as expected' if got == want else got}")
            ok &= got == want
            if stage == "opt":
                print(f"   opt feasibility_check: {sc.get('feasibility_check')!r}")
                ok &= sc.get("feasibility_check") == "first"
    os.makedirs(FIGS, exist_ok=True)
    t = pd.DataFrame(years, columns=["chain", "year", "F_opt_identical"]).pivot(
        index="year", columns="chain", values="F_opt_identical")
    t.columns = [f"chain_{c}" for c in t.columns]
    path = os.path.join(FIGS, f"f_opt_equality_{two.opt}_{three.opt}.csv")
    t.to_csv(path)
    print(f"\n   F_opt identical per year: chain 2 {int(t['chain_two'].sum())}/{len(t)}, "
          f"chain 3 {int(t['chain_three'].sum())}/{len(t)} → {path}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="#106 bit-identical driver gate.")
    ap.add_argument("mode", choices=("run", "compare"))
    ap.add_argument("opts", nargs="*", help="compare: the chain 2 and chain 3 opt runs")
    ap.add_argument("--ref", default=REF)
    a = ap.parse_args()
    ref = Chain.from_opt_run(a.ref)
    defaults = driver_defaults()
    if a.mode == "run":
        two, three, lock_after = run_chains(a.ref)
        print(f"   lock left after chain 2 / chain 3: {lock_after}")
        ok = not any(lock_after)
    else:
        two, three, ok = Chain.from_opt_run(a.opts[0]), _scratch_chain(a.opts[1]), True
    ok &= compare(two, three, ref, defaults)
    print(f"\n== VERDICT: {'PASS' if ok else 'FAIL'} — {'every' if ok else 'not every'} "
          f"item of the decision rule holds")
    if ok:
        shutil.rmtree(SCRATCH, ignore_errors=True)


if __name__ == "__main__":
    main()
