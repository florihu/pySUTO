"""
migrate_registries.py
=====================
One-shot, idempotent migration from the legacy registry layout to the
harmonized layout described in the registry plan.

Input side (data/input/)
------------------------
* feed/<id>/v0/                → feed/<id>/v1/  (rename, build sidecar
                                  registry.json, materialise current/
                                  from v1 contents)
* acc/acc_v0/                  → acc/copper/v1/ + current/ (id renamed
                                  to ``copper``, ``--acc-id`` overrides)
* params/<id>/                 → params/<id>/v1/ snapshot + current/
                                  (existing files stay as current/)

In all cases ``v1/`` is materialised with hardlinks back to ``current/``
where the filesystem supports it (so disk usage stays flat).

Output side (data/proc/)
------------------------
* feed/registry/registry.json  rewritten to the 4-field harmonized
                                index ({run_id, stage, label, created_at}).
                                For each surviving run directory whose
                                ``metadata.json`` lacks the new schema,
                                back-fill a sidecar from the legacy index
                                entry (best-effort).
* imp/registry.json            same treatment.
* const/registry.json          same treatment + back-fill sidecar.
* opt/registry.json            same treatment + back-fill sidecar.

Existing run directories are NOT renamed: their names are opaque labels
once the convention is "label only in registry".  Legacy run-ids stay in
their original directory names for REP-trace continuity; the index
entries point at them by run_id.

Run::

    python scripts/migrate_registries.py --dry-run     # prints the plan
    python scripts/migrate_registries.py               # executes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT  = _SCRIPT_DIR.parent

INPUT_ROOT  = _REPO_ROOT / "data" / "input"
PROC_ROOT   = _REPO_ROOT / "data" / "proc"

OUTPUT_KINDS = ("feed", "imp", "const", "opt")
INPUT_KINDS  = ("feed", "acc", "params")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("migrate_registries")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _short_code() -> str:
    return uuid.uuid4().hex[:8]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_bearing(kind: str, current: Path) -> list[str]:
    if kind == "params":
        return ["params.csv"]
    if kind == "acc":
        return ["root.ods", "bounds.ods", "rules.ods"]
    # feed: anything not *.bib / *.md
    return sorted(
        p.name for p in current.iterdir()
        if p.is_file() and not p.name.lower().endswith((".bib", ".md"))
    )


def _hash_loadbearing(kind: str, current: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in _load_bearing(kind, current):
        p = current / name
        out[name] = _sha256_file(p) if p.is_file() else ""
    return out


def _ensure_dir(p: Path, dry: bool) -> None:
    if dry:
        log.info("  (dry) would mkdir -p %s", p)
    else:
        p.mkdir(parents=True, exist_ok=True)


def _copy_tree(src: Path, dst: Path, dry: bool, use_links: bool = False) -> None:
    """Copy every file under *src* into *dst*; hardlink when *use_links*."""
    if not src.is_dir():
        return
    for child in src.rglob("*"):
        if not child.is_file():
            continue
        target = dst / child.relative_to(src)
        if dry:
            verb = "hardlink" if use_links else "copy"
            log.info("  (dry) would %s %s → %s", verb, child, target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if use_links:
            try:
                os.link(child, target)
                continue
            except OSError:
                pass
        shutil.copy2(child, target)


def _rename(src: Path, dst: Path, dry: bool) -> None:
    if dry:
        log.info("  (dry) would mv %s → %s", src, dst)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)


# ---------------------------------------------------------------------------
# Input-side migration
# ---------------------------------------------------------------------------

def migrate_input_feed(dry: bool) -> None:
    """data/input/feed/<id>/v0/ → /v1/ + current/.  Build registry.json."""
    root = INPUT_ROOT / "feed"
    if not root.is_dir():
        log.info("[feed] %s does not exist — skipping", root)
        return

    registry: dict[str, dict] = {}
    if (root / "registry.json").is_file():
        registry = json.loads((root / "registry.json").read_text())

    for feed_dir in sorted(root.iterdir()):
        if not feed_dir.is_dir() or feed_dir.name == "registry.json":
            continue
        fid = feed_dir.name
        v0  = feed_dir / "v0"
        v1  = feed_dir / "v1"
        current = feed_dir / "current"

        if v0.is_dir() and not v1.is_dir():
            log.info("[feed/%s] v0 → v1", fid)
            _rename(v0, v1, dry)

        if not (feed_dir / "v1").is_dir():
            log.info("[feed/%s] no v1 found — skipping", fid)
            continue

        if not current.is_dir():
            log.info("[feed/%s] materialising current/ from v1/", fid)
            _copy_tree(feed_dir / "v1", current, dry, use_links=True)

        if not dry and fid not in registry:
            hashes = _hash_loadbearing("feed", current)
            registry[fid] = {
                "label":    fid,
                "versions": [{
                    "version": 1,
                    "date":    _now_iso(),
                    "code":    _short_code(),
                    "hashes":  hashes,
                }],
            }

    if not dry and registry:
        (root / "registry.json").write_text(json.dumps(registry, indent=2, sort_keys=True))
        log.info("[feed] wrote %s (%d ids)", root / "registry.json", len(registry))


def migrate_input_acc(dry: bool, target_id: str = "copper") -> None:
    """data/input/acc/acc_v0/ → /<target_id>/v1 + current/."""
    root    = INPUT_ROOT / "acc"
    legacy  = root / "acc_v0"
    target  = root / target_id

    if not legacy.is_dir() and not target.is_dir():
        log.info("[acc] nothing to migrate at %s", root)
        return

    registry: dict[str, dict] = {}
    if (root / "registry.json").is_file():
        registry = json.loads((root / "registry.json").read_text())

    if legacy.is_dir():
        v1 = target / "v1"
        if not v1.is_dir():
            log.info("[acc] %s → %s/v1", legacy, target)
            _ensure_dir(target, dry)
            _rename(legacy, v1, dry)

    current = target / "current"
    if (target / "v1").is_dir() and not current.is_dir():
        log.info("[acc] materialising current/ from v1/")
        _copy_tree(target / "v1", current, dry, use_links=True)

    if not dry and target_id not in registry and current.is_dir():
        hashes = _hash_loadbearing("acc", current)
        registry[target_id] = {
            "label":    target_id,
            "versions": [{
                "version": 1,
                "date":    _now_iso(),
                "code":    _short_code(),
                "hashes":  hashes,
            }],
        }

    if not dry and registry:
        (root / "registry.json").write_text(json.dumps(registry, indent=2, sort_keys=True))
        log.info("[acc] wrote %s (%d ids)", root / "registry.json", len(registry))


def migrate_input_params(dry: bool) -> None:
    """Snapshot existing params/<id>/params.csv (and friends) as v1/ +
    leave the working copy in current/."""
    root = INPUT_ROOT / "params"
    if not root.is_dir():
        log.info("[params] %s does not exist — skipping", root)
        return

    registry: dict[str, dict] = {}
    if (root / "registry.json").is_file():
        registry = json.loads((root / "registry.json").read_text())

    for pdir in sorted(root.iterdir()):
        if not pdir.is_dir():
            continue
        pid = pdir.name
        current = pdir / "current"
        v1      = pdir / "v1"

        # The legacy params layout puts files directly under pdir/ with no
        # subfolder.  We re-home everything that isn't a versioned snapshot
        # into current/.  Existing v1/ stays untouched.
        if not current.is_dir():
            log.info("[params/%s] re-homing files into current/", pid)
            if not dry:
                current.mkdir(parents=True)
            for child in sorted(pdir.iterdir()):
                if child.name in ("current", "registry.json") or child.name.startswith("v"):
                    continue
                if child.is_file():
                    target = current / child.name
                    if dry:
                        log.info("  (dry) would mv %s → %s", child, target)
                    else:
                        shutil.move(str(child), str(target))

        if current.is_dir() and not v1.is_dir():
            log.info("[params/%s] snapshotting current/ → v1/", pid)
            _copy_tree(current, v1, dry, use_links=True)

        if not dry and pid not in registry and current.is_dir():
            hashes = _hash_loadbearing("params", current)
            registry[pid] = {
                "label":    pid,
                "versions": [{
                    "version": 1,
                    "date":    _now_iso(),
                    "code":    _short_code(),
                    "hashes":  hashes,
                }],
            }

    if not dry and registry:
        (root / "registry.json").write_text(json.dumps(registry, indent=2, sort_keys=True))
        log.info("[params] wrote %s (%d ids)", root / "registry.json", len(registry))


# ---------------------------------------------------------------------------
# Output-side migration: trim registry.json + back-fill sidecars
# ---------------------------------------------------------------------------

def _legacy_label_from(stage: str, entry: dict, run_id: str) -> str:
    if entry.get("label"):
        return str(entry["label"])
    if stage == "feed":
        return str(entry.get("feed_id", run_id))
    if stage == "imp":
        return str(entry.get("label") or run_id)
    return run_id


def _back_fill_sidecar(stage: str, run_dir: Path, entry: dict) -> None:
    """If the legacy entry already contains parent info but no sidecar
    exists at run_dir/metadata.json, write a sidecar reconstructed from
    the legacy entry's fields."""
    sidecar_path = run_dir / "metadata.json"
    if sidecar_path.is_file():
        # Already a sidecar; do not overwrite (treat existing as canonical).
        return
    run_id = entry.get("run_id") or entry.get("feed_run_id") or entry.get("optim_run_id") \
             or run_dir.name

    if stage == "feed":
        parents = {
            "inputs": {
                "acc":  {"id": entry.get("acc_id", "")},
                "feed": {"id": entry.get("feed_id", "")},
            },
            "upstream": {},
        }
    elif stage == "imp":
        feeds = entry.get("feed_ids", [])
        parents = {
            "inputs":   {"acc": {"id": entry.get("acc_id", "")}},
            "upstream": {"feed_run_ids": list(feeds)},
        }
    elif stage == "const":
        parents = {
            "inputs":   {"acc": {"id": entry.get("acc_id", "")}},
            "upstream": {"imp_run_id": entry.get("agg_id", "")},
        }
    elif stage == "opt":
        parents = {
            "inputs":   {},
            "upstream": {"const_run_id": entry.get("const_run_id", "")},
        }
    else:
        parents = {"inputs": {}, "upstream": {}}

    sidecar = {
        "run_id":     run_id,
        "stage":      stage,
        "label":      _legacy_label_from(stage, entry, run_id),
        "created_at": entry.get("created_at") or _now_iso(),
        "parents":    parents,
        "stage_config":  {"p": entry["p"]} if "p" in entry else {},
        "stage_summary": {
            k: v for k, v in entry.items()
            if k not in {"acc_id", "label", "agg_id", "created_at",
                         "n_rows", "n_edges", "timesteps", "p",
                         "feed_id", "feed_run_id", "feed_class", "odir",
                         "metadata_path", "feed_ids", "optim_run_id",
                         "const_run_id", "per_timestep"}
        },
    }
    if "timesteps" in entry:
        sidecar["stage_summary"]["timesteps"] = entry["timesteps"]
    if "per_timestep" in entry:
        sidecar["stage_summary"]["per_timestep"] = entry["per_timestep"]

    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    log.info("[%s] back-filled sidecar at %s", stage, sidecar_path)


def migrate_output_registry(stage: str, dry: bool) -> None:
    if stage == "feed":
        reg_path = PROC_ROOT / "feed" / "registry" / "registry.json"
    else:
        reg_path = PROC_ROOT / stage / "registry.json"

    if not reg_path.is_file():
        log.info("[%s] no registry at %s — skipping", stage, reg_path)
        return

    legacy = json.loads(reg_path.read_text())
    trimmed: dict[str, dict] = {}
    for key, entry in legacy.items():
        if isinstance(entry, str):
            continue
        run_id = entry.get("run_id") or entry.get("feed_run_id") \
                  or entry.get("optim_run_id") or key
        created_at = entry.get("created_at") or _now_iso()
        label = _legacy_label_from(stage, entry, run_id)
        trimmed[run_id] = {
            "run_id":     run_id,
            "stage":      stage,
            "label":      label,
            "created_at": created_at,
        }

        # Back-fill sidecar if necessary.
        run_dir = reg_path.parent / run_id
        if run_dir.is_dir():
            if dry:
                log.info("[%s] (dry) would back-fill sidecar under %s", stage, run_dir)
            else:
                _back_fill_sidecar(stage, run_dir, entry)

    if dry:
        log.info("[%s] (dry) would rewrite %s with %d trimmed entries",
                 stage, reg_path, len(trimmed))
    else:
        reg_path.write_text(json.dumps(trimmed, indent=2))
        log.info("[%s] rewrote %s with %d trimmed entries",
                 stage, reg_path, len(trimmed))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="One-shot pySUTO registry migration.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned actions without touching disk.")
    p.add_argument("--acc-id", default="copper",
                   help="Target id under data/input/acc/ for the legacy "
                        "acc_v0/ folder (default: copper).")
    p.add_argument("--skip-input", action="store_true")
    p.add_argument("--skip-output", action="store_true")
    args = p.parse_args()

    if not args.skip_input:
        log.info("── Migrating input registries ──")
        migrate_input_feed(args.dry_run)
        migrate_input_acc(args.dry_run, target_id=args.acc_id)
        migrate_input_params(args.dry_run)

    if not args.skip_output:
        log.info("── Migrating output registries ──")
        for stage in OUTPUT_KINDS:
            migrate_output_registry(stage, args.dry_run)

    log.info("Done.")


if __name__ == "__main__":
    main()
