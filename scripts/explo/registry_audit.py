"""
registry_audit.py
=================
Check every snapshot file under data/input/<kind>/<id>/v<N>/ against the
sha256 the registry recorded when that version was materialised, and report
the inode sharing that lets an in-place write to current/ reach a snapshot.

The recorded hashes are authoritative for what each run consumed (they are
copied into every run's metadata.json); the retained file is what this audit
verifies. A mismatch means the retained snapshot is NOT the input that run
used. Found 2026-09-10 while regenerating ore_grade (28 of 93 mismatched):
the 2026 migration hardlinked v1/ back to current/, so every generator that
rewrote current/ in place (to_csv) rewrote the history too.

    python scripts/explo/registry_audit.py
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path("data/input")
KINDS = ("params", "feed", "acc", "mask")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def audit() -> list[tuple]:
    bad = []
    n = 0
    for kind in KINDS:
        reg = ROOT / kind / "registry.json"
        if not reg.is_file():
            continue
        for id_, rec in json.load(open(reg)).items():
            for v in rec["versions"]:
                for name, recorded in v["hashes"].items():
                    p = ROOT / kind / id_ / f"v{v['version']}" / name
                    n += 1
                    on_disk = _sha(p) if p.is_file() else ""
                    if on_disk != recorded:
                        bad.append((kind, id_, v["version"], name, recorded[:8], on_disk[:8] or "MISSING"))
    print(f"checked {n} version-files, {len(bad)} mismatch(es)")
    for b in bad:
        print("  %-6s %-12s v%-2d %-28s recorded %s  on disk %s" % b)
    return bad


def shared_current() -> None:
    print("\ncurrent/ files still sharing an inode with a snapshot:")
    for kind in KINDS:
        for cur in (ROOT / kind).glob("*/current"):
            for f in cur.rglob("*"):
                if f.is_file() and os.stat(f).st_nlink > 1:
                    print(f"  {f}  nlink={os.stat(f).st_nlink}")


if __name__ == "__main__":
    audit()
    shared_current()
