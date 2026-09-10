"""
restore_ore_grade_versions.py
=============================
Restore data/input/params/ore_grade/v2/params.csv and v3/params.csv, which the
registry's inode-sharing defect overwrote (see ADR 0023 §"Discovered alongside",
scripts/explo/registry_audit.py). Both versions regenerate exactly from the
generator as it stood when they were registered — v2 from commit 696027c
(ADR 0015, no tailings rows), v3 from 8ac5720 (Round A tailings amendment) —
because the feed fields the generator reads did not change afterwards.

Nothing is written unless the regenerated file's sha256 equals the hash the
registry recorded for that version. Every params.csv under ore_grade/ is first
given a private inode so the write cannot reach any other version.

    python scripts/explo/restore_ore_grade_versions.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from pysuto.acc import AccountingStructure          # noqa: E402
from pysuto.config import load_config               # noqa: E402

BASE = Path("data/input/params/ore_grade")
GENERATORS = {2: "696027c", 3: "8ac5720"}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _generator_at(commit: str):
    src = subprocess.check_output(["git", "show", f"{commit}:scripts/build_ore_grade.py"], text=True)
    tmp = Path(tempfile.mkdtemp()) / f"bog_{commit}.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location(f"bog_{commit}", tmp)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, "scripts")
    spec.loader.exec_module(mod)
    return mod


def regenerate(version: int) -> pd.DataFrame:
    b = _generator_at(GENERATORS[version])
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id, storage_dir=b.ACC_DIR)
    eff_mf, eff_sxew = b._eff_nominals()
    prod = b.build_production_frame(acc, b._latest_feed_froot("icsg"), b._latest_feed_froot("crude_ore"))
    grades = b.compute_insitu_grades(prod, b.load_factor_fallback(), eff_mf, eff_sxew)
    frames = [b.build_param_rows(grades, b.region_bands(grades))]
    if hasattr(b, "compute_tailings_grades"):
        cc, cs = b._conc_nominals()
        tails = b.compute_tailings_grades(prod, grades, eff_mf, eff_sxew, cc, cs)
        frames.append(b.build_tailings_rows(tails, b.tailings_bands(tails)))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    recorded = {v["version"]: v["hashes"]["params.csv"]
                for v in json.load(open("data/input/params/registry.json"))["ore_grade"]["versions"]}
    for p in sorted(BASE.glob("*/params.csv")):
        if os.stat(p).st_nlink > 1:
            tmp = p.with_name("params.csv.private~")
            shutil.copy2(p, tmp)
            os.replace(tmp, p)
            print(f"   {p}: private inode")
    for version in sorted(GENERATORS):
        target = BASE / f"v{version}" / "params.csv"
        if _sha(target) == recorded[version]:
            print(f"   v{version}: already matches registry — skipped")
            continue
        out = Path(tempfile.mkdtemp()) / "params.csv"
        regenerate(version).to_csv(out, index=False)
        if _sha(out) != recorded[version]:
            print(f"   v{version}: regeneration sha {_sha(out)[:16]} != recorded {recorded[version][:16]} — NOT written")
            continue
        shutil.copy2(out, target)
        print(f"   v{version}: restored, sha {_sha(target)[:16]} == registry")


if __name__ == "__main__":
    main()
