"""
input_registry.py
=================
Per-kind versioned snapshot store for pySUTO inputs (``feed`` raw sources,
``acc`` definitions, ``params`` CSVs).

On-disk layout per kind::

    data/input/<kind>/
    ├── registry.json
    └── <id>/
        ├── current/       ← live working copy (user edits here)
        ├── v1/            ← snapshot; unchanged files = hardlinks back
        └── v2/

When :meth:`InputRegistry.initialize` runs, the registry hashes the
*load-bearing* files in each ``<id>/current/`` and compares them
against the latest snapshot.  If anything changed, ``v_{N+1}/`` is
materialised: every file in ``current/`` is either hardlinked from
the prior version (if unchanged) or copied fresh.

Output runs reference inputs by ``(id, version)`` tuples in their
per-run ``metadata.json``, so this is the authoritative source for
assumption-change auditing.

``current/`` must never share an inode with a snapshot: a truncating
write to the live copy (``to_csv``, ``open(..., "w")``) would then rewrite
every snapshot linked to it and silently destroy the history the recorded
hashes vouch for.  The 2026 migration materialised ``v1/`` as hardlinks
*back to* ``current/`` and every later snapshot linked "unchanged" files
onward, so :meth:`initialize` first gives each file under ``current/`` a
private inode (copy-and-replace when its link count exceeds one).
Snapshots may still share inodes among themselves — nothing writes there.

Load-bearing files (changes trigger a version bump):

* ``params/<id>/`` — ``params.csv``
* ``acc/<id>/``    — ``root.ods``, ``bounds.ods``, ``rules.ods``, ``masks.csv``
                     (``masks.csv`` optional; absent file hashes to empty)
* ``mask/<id>/``   — ``rules.csv`` (separate-kind mask overlay)
* ``feed/<id>/``   — per-feed declared list, or default = "every file in
  ``current/`` whose suffix is not ``.bib`` / ``.md``".
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pysuto.logging import get_logger

_log = get_logger("input_registry")


_KINDS = ("feed", "acc", "params", "mask")

_DEFAULT_LOAD_BEARING: dict[str, tuple[str, ...]] = {
    "params": ("params.csv",),
    "acc":    ("root.ods", "bounds.ods", "rules.ods", "masks.csv"),
    "mask":   ("rules.csv",),
    # "feed" handled in InputRegistry._load_bearing_files
}

_FEED_IGNORE_SUFFIXES = (".bib", ".md")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_files(root: Path, names: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in names:
        p = root / name
        out[name] = _sha256_file(p) if p.is_file() else ""
    return out


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _short_code() -> str:
    return uuid.uuid4().hex[:8]


# ──────────────────────────────────────────────────────────────────────
# Single-kind registry
# ──────────────────────────────────────────────────────────────────────

class InputRegistry:
    """Versioned snapshot store for one input kind.

    Parameters
    ----------
    kind
        ``"feed"``, ``"acc"`` or ``"params"``.
    root
        Directory holding ``<kind>/`` subtree (default ``data/input``).
    feed_load_bearing
        Optional ``{feed_id: [filenames]}`` mapping.  Only consulted when
        ``kind == "feed"``.  Ids absent from this mapping fall back to
        the "everything except *.bib/*.md" convention.
    """

    def __init__(
        self,
        kind: str,
        root: Path | str = Path("data/input"),
        feed_load_bearing: dict[str, list[str]] | None = None,
    ) -> None:
        if kind not in _KINDS:
            raise ValueError(f"Unknown input kind: {kind!r}; expected one of {_KINDS}")
        self.kind = kind
        self.root = Path(root) / kind
        self._feed_load_bearing = feed_load_bearing or {}
        self._index_path = self.root / "registry.json"
        self._index: dict[str, dict] = {}
        self._pins: dict[str, int] = {}
        self._load_index()

    # ── public API ────────────────────────────────────────────────────

    def initialize(
        self,
        label_overrides: dict[str, str] | None = None,
        pins: dict[str, int] | None = None,
    ) -> "InputRegistry":
        """Scan every ``<id>/current/`` and bump versions where load-bearing
        files changed.  ``pins`` are stored on the instance so subsequent
        :meth:`resolve` / :meth:`entry` calls honor them (pins do not
        affect bump decisions)."""
        self._pins = dict(pins or {})
        label_overrides = label_overrides or {}

        self.root.mkdir(parents=True, exist_ok=True)
        for child in sorted(self.root.iterdir()):
            if not child.is_dir():
                continue
            if not (child / "current").is_dir():
                continue
            self._maybe_bump(child.name, label_overrides.get(child.name))

        self._save_index()
        return self

    def resolve(self, id: str, version: int | None = None) -> Path:
        """Absolute path to ``<root>/<id>/v{version-or-latest}/``."""
        rec = self._require_entry(id)
        v = self._effective_version(id, version, rec)
        self._require_version(id, v)
        return (self.root / id / f"v{v}").resolve()

    def entry(self, id: str, version: int | None = None) -> dict:
        """Public registry entry ``{id, version, date, code, label}`` —
        the form embedded into downstream runs' ``metadata.json``."""
        rec = self._require_entry(id)
        v = self._effective_version(id, version, rec)
        v_rec = self._require_version(id, v)
        return {
            "id":      id,
            "version": v,
            "date":    v_rec["date"],
            "code":    v_rec["code"],
            "label":   rec["label"],
        }

    def latest_version(self, id: str) -> int:
        return self._require_entry(id)["versions"][-1]["version"]

    def ids(self) -> list[str]:
        return sorted(self._index.keys())

    # ── internal: bump logic ──────────────────────────────────────────

    def _maybe_bump(self, id: str, label_override: Optional[str]) -> None:
        current = self.root / id / "current"
        n_shared = self._privatise_current(current)
        if n_shared:
            _log.warning(
                "input/%s/%s: %d file(s) under current/ shared an inode with a "
                "snapshot — given a private copy (an in-place write would have "
                "rewritten history)", self.kind, id, n_shared)
        load_bearing = self._load_bearing_files(id, current)
        new_hashes = _hash_files(current, load_bearing)

        rec = self._index.get(id, {"label": "", "versions": []})
        prior_versions = rec["versions"]
        prior_hashes = prior_versions[-1]["hashes"] if prior_versions else None

        if prior_hashes == new_hashes and prior_versions:
            # No load-bearing change — possibly refresh label.
            if label_override and rec["label"] != label_override:
                rec["label"] = label_override
                self._index[id] = rec
            return

        new_version = prior_versions[-1]["version"] + 1 if prior_versions else 1
        v_dir = self.root / id / f"v{new_version}"
        v_dir.mkdir(parents=True, exist_ok=False)
        self._snapshot(current, v_dir, self._latest_v_dir(id, prior_versions))

        label = self._resolve_label(id, current, label_override, rec.get("label", ""))
        rec["label"] = label
        rec["versions"].append({
            "version": new_version,
            "date":    _now_iso(),
            "code":    _short_code(),
            "hashes":  new_hashes,
        })
        self._index[id] = rec
        _log.info("input/%s/%s: bumped to v%d (label=%r)", self.kind, id, new_version, label)

    @staticmethod
    def _privatise_current(current: Path) -> int:
        """Give every file under *current* its own inode; return how many
        were shared (link count > 1) and replaced by a private copy."""
        n = 0
        for child in current.rglob("*"):
            if child.is_file() and os.stat(child).st_nlink > 1:
                tmp = child.with_name(child.name + ".private~")
                shutil.copy2(child, tmp)
                os.replace(tmp, child)
                n += 1
        return n

    def _snapshot(
        self,
        src: Path,
        dst: Path,
        prior_version_dir: Optional[Path],
    ) -> None:
        """Copy every file under *src* into *dst*, hardlinking files whose
        content matches the prior version (if present)."""
        for child in src.rglob("*"):
            if not child.is_file():
                continue
            rel = child.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)

            linked = False
            if prior_version_dir is not None:
                prior_file = prior_version_dir / rel
                if prior_file.is_file() and _sha256_file(prior_file) == _sha256_file(child):
                    try:
                        os.link(prior_file, target)
                        linked = True
                    except OSError:
                        linked = False
            if not linked:
                shutil.copy2(child, target)

    def _latest_v_dir(self, id: str, prior_versions: list[dict]) -> Optional[Path]:
        if not prior_versions:
            return None
        v = prior_versions[-1]["version"]
        p = self.root / id / f"v{v}"
        return p if p.is_dir() else None

    def _load_bearing_files(self, id: str, current: Path) -> list[str]:
        if self.kind == "feed":
            declared = self._feed_load_bearing.get(id)
            if declared:
                return list(declared)
            if not current.is_dir():
                return []
            return sorted(
                p.name for p in current.iterdir()
                if p.is_file() and not p.name.lower().endswith(_FEED_IGNORE_SUFFIXES)
            )
        return list(_DEFAULT_LOAD_BEARING[self.kind])

    def _resolve_label(
        self,
        id: str,
        current: Path,
        override: Optional[str],
        carry_forward: str,
    ) -> str:
        if override:
            return override
        if self.kind == "feed":
            label = self._feed_meta_label(current)
            if label:
                return label
        if carry_forward:
            return carry_forward
        return id

    @staticmethod
    def _feed_meta_label(current: Path) -> Optional[str]:
        try:
            from pysuto.meta import DataFeedMetadata
        except ImportError:
            return None
        if not current.is_dir():
            return None
        for p in sorted(current.iterdir()):
            if p.suffix.lower() == ".xlsx":
                try:
                    meta = DataFeedMetadata.from_excel(str(p))
                    name = getattr(meta, "feed_name", "") or ""
                    return name.strip() or None
                except Exception:
                    return None
        return None

    # ── internal: index IO + lookups ──────────────────────────────────

    def _load_index(self) -> None:
        if self._index_path.is_file():
            with open(self._index_path, encoding="utf-8") as f:
                self._index = json.load(f)

    def _save_index(self) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2, sort_keys=True)

    def _require_entry(self, id: str) -> dict:
        if id not in self._index:
            raise KeyError(
                f"input/{self.kind}: no id {id!r} registered. "
                f"Add files under data/input/{self.kind}/{id}/current/ and call initialize()."
            )
        return self._index[id]

    def _require_version(self, id: str, version: int) -> dict:
        for v in self._index[id]["versions"]:
            if v["version"] == version:
                return v
        raise KeyError(f"input/{self.kind}/{id}: version {version} not registered")

    def _effective_version(
        self,
        id: str,
        explicit: int | None,
        rec: dict,
    ) -> int:
        if explicit is not None:
            return explicit
        if id in self._pins:
            return self._pins[id]
        return rec["versions"][-1]["version"]


# ──────────────────────────────────────────────────────────────────────
# Convenience wrapper across the three kinds
# ──────────────────────────────────────────────────────────────────────

@dataclass
class InputRegistries:
    feed:   InputRegistry
    acc:    InputRegistry
    params: InputRegistry
    mask:   InputRegistry

    @classmethod
    def initialize(
        cls,
        root: Path | str = Path("data/input"),
        label_overrides: dict[str, dict[str, str]] | None = None,
        pins: dict[str, dict[str, int]] | None = None,
        feed_load_bearing: dict[str, list[str]] | None = None,
    ) -> "InputRegistries":
        label_overrides = label_overrides or {}
        pins = pins or {}
        return cls(
            feed=InputRegistry("feed", root, feed_load_bearing=feed_load_bearing).initialize(
                label_overrides.get("feed"), pins.get("feed")),
            acc=InputRegistry("acc", root).initialize(
                label_overrides.get("acc"), pins.get("acc")),
            params=InputRegistry("params", root).initialize(
                label_overrides.get("params"), pins.get("params")),
            mask=InputRegistry("mask", root).initialize(
                label_overrides.get("mask"), pins.get("mask")),
        )
