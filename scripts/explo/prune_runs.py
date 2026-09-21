"""Delete superseded const/opt run directories and prune them from the registries.

Keeps the runs Round P needs simultaneously (base, Round O and their parents) and
every run a shipped round is measured on. Prints what it will do, then does it.

    python scripts/explo/prune_runs.py --dry-run
    python scripts/explo/prune_runs.py
"""
import json
import os
import shutil
import sys

ROOT = "data/proc"
KEEP_OPT = {
    "opt_20260916_124721_c4d1d944",  # Round P base (roundM_revert_verify)
    "opt_20260917_121759_c57f252b",  # Round O
    "opt_20260908_114321_279e7979",  # shipped Rounds A-F
    "opt_20260909_094009_49eab1dc",  # shipped Rounds G+H
    "opt_20260910_133212_45f034f1",  # shipped Round J.1
    "opt_20260914_090327_c21a5c48",  # shipped Round K
    "opt_20260915_131115_b8bf05a9",  # shipped Round L
}


def _upstream(stage, rid):
    try:
        with open(f"{ROOT}/{stage}/{rid}/metadata.json") as fh:
            return json.load(fh)["parents"]["upstream"]
    except Exception:
        return {}


def _keep_sets():
    const, imp = set(), set()
    for o in KEEP_OPT:
        c = _upstream("opt", o).get("const_run_id")
        if c:
            const.add(c)
            i = _upstream("const", c).get("imp_run_id")
            if i:
                imp.add(i)
    return const, imp


def _size_gb(path):
    return sum(os.path.getsize(os.path.join(d, f))
               for d, _, fs in os.walk(path) for f in fs) / 1e9


def main(dry: bool) -> None:
    keep_const, keep_imp = _keep_sets()
    keep = {"opt": KEEP_OPT, "const": keep_const, "imp": keep_imp}
    doomed = {}
    for stage in ("opt", "const"):
        doomed[stage] = sorted(
            r for r in os.listdir(f"{ROOT}/{stage}")
            if os.path.isdir(f"{ROOT}/{stage}/{r}") and r not in keep[stage]
        )
    total = sum(_size_gb(f"{ROOT}/{s}/{r}")
                for s in doomed for r in doomed[s])
    print(f"keep: {len(KEEP_OPT)} opt, {len(keep_const)} const, {len(keep_imp)} imp")
    for s in ("opt", "const"):
        print(f"delete {len(doomed[s]):3d} {s} run dirs")
    print(f"frees ~{total:.1f} GB")
    if dry:
        print("\n--dry-run: nothing removed")
        return
    for stage in ("opt", "const"):
        for rid in doomed[stage]:
            shutil.rmtree(f"{ROOT}/{stage}/{rid}")
        path = f"{ROOT}/{stage}/registry.json"
        with open(path) as fh:
            reg = json.load(fh)
        runs = reg["runs"] if isinstance(reg.get("runs"), dict) else reg
        gone = [r for r in doomed[stage] if r in runs]
        for r in gone:
            del runs[r]
        with open(path, "w") as fh:
            json.dump(reg, fh, indent=2)
        print(f"{stage}: removed {len(doomed[stage])} dirs, "
              f"pruned {len(gone)} registry entries, {len(runs)} remain")


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
