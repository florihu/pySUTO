"""Delete superseded const/opt run directories and prune them from the registries.

Keeps the runs Round P needs simultaneously (base, Round O and their parents) and
every run a shipped round is measured on. Prints what it will do, then does it.

    python scripts/explo/prune_runs.py --dry-run
    python scripts/explo/prune_runs.py
    python scripts/explo/prune_runs.py --keep-recent 5 opt_20260921_084333_d12ba597

The newest `--keep-recent` (default 3) opt runs are ALWAYS kept, whatever the
allowlist says: the allowlist is static and so is always stale with respect to
the run someone most likely still needs. Extra run ids may be named on the CLI.
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


def _recent_opt(keep_recent: int) -> set:
    """Always keep the newest `keep_recent` opt runs, whatever the allowlist says.

    The allowlist is static and is therefore always out of date with respect to
    a run made after it was last edited — which is exactly the run someone is
    most likely to still need. This is the guard that makes the script safe to
    run at any moment.
    """
    d = f"{ROOT}/opt"
    runs = [r for r in os.listdir(d) if os.path.isdir(f"{d}/{r}")]
    return set(sorted(runs, key=lambda r: os.path.getmtime(f"{d}/{r}"),
                      reverse=True)[:keep_recent])


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


def main(dry: bool, keep_recent: int = 3, extra: set = frozenset()) -> None:
    recent = _recent_opt(keep_recent)
    KEEP_OPT.update(recent)
    KEEP_OPT.update(extra)
    print(f"protecting the {len(recent)} newest opt runs regardless of the allowlist: "
          + ", ".join(sorted(recent)))
    if extra:
        print(f"protecting --keep: {', '.join(sorted(extra))}")
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
    args = sys.argv[1:]
    n = 3
    if "--keep-recent" in args:
        n = int(args[args.index("--keep-recent") + 1])
    keep = {a for a in args if a.startswith(("opt_", "const_"))}
    main("--dry-run" in args, keep_recent=n, extra=keep)
