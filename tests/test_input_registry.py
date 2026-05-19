"""
test_input_registry.py
======================
Unit tests for pysuto.input_registry:

* version-bump trigger driven by load-bearing files only
* hardlink dedup for unchanged files across versions
* fresh copy for files that changed
* label fallback chain (override → meta sidecar → carry-forward → id)
* per-id pin override on resolve / entry
* idempotent re-initialize when nothing changed
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from pysuto.input_registry import InputRegistries, InputRegistry


# =============================================================================
# Helpers
# =============================================================================


def _write(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _params_id(root: Path, id: str, params_csv: str, extras: dict[str, str] | None = None) -> None:
    """Materialise data/input/params/<id>/current/{params.csv, *extras*}."""
    current = root / "params" / id / "current"
    _write(current / "params.csv", params_csv)
    for name, body in (extras or {}).items():
        _write(current / name, body)


def _acc_id(root: Path, id: str, ods_bytes: dict[str, bytes]) -> None:
    current = root / "acc" / id / "current"
    current.mkdir(parents=True, exist_ok=True)
    for name, body in ods_bytes.items():
        (current / name).write_bytes(body)


def _feed_id(root: Path, id: str, files: dict[str, str]) -> None:
    current = root / "feed" / id / "current"
    for name, body in files.items():
        _write(current / name, body)


# =============================================================================
# Params kind
# =============================================================================


def test_params_first_bump_creates_v1(tmp_path: Path) -> None:
    _params_id(tmp_path, "conc", "param_id;value_nominal\nfoo;0.5\n")

    reg = InputRegistry("params", root=tmp_path).initialize()

    assert reg.latest_version("conc") == 1
    snap = tmp_path / "params" / "conc" / "v1"
    assert (snap / "params.csv").read_text() == "param_id;value_nominal\nfoo;0.5\n"

    e = reg.entry("conc")
    assert e["id"] == "conc"
    assert e["version"] == 1
    assert e["label"] == "conc"            # default fallback
    assert len(e["code"]) == 8


def test_params_no_change_no_bump(tmp_path: Path) -> None:
    _params_id(tmp_path, "conc", "param_id;value\nx;1\n")
    InputRegistry("params", root=tmp_path).initialize()

    # Re-initialize without changing anything.
    reg = InputRegistry("params", root=tmp_path).initialize()
    assert reg.latest_version("conc") == 1
    assert not (tmp_path / "params" / "conc" / "v2").exists()


def test_params_load_bearing_change_bumps(tmp_path: Path) -> None:
    _params_id(tmp_path, "conc", "param_id;value\nx;1\n")
    InputRegistry("params", root=tmp_path).initialize()

    # Edit params.csv → bump.
    _write(tmp_path / "params" / "conc" / "current" / "params.csv",
           "param_id;value\nx;2\n")
    reg = InputRegistry("params", root=tmp_path).initialize()

    assert reg.latest_version("conc") == 2
    assert (tmp_path / "params" / "conc" / "v2" / "params.csv").read_text() \
        == "param_id;value\nx;2\n"
    # v1 contents unchanged
    assert (tmp_path / "params" / "conc" / "v1" / "params.csv").read_text() \
        == "param_id;value\nx;1\n"


def test_params_non_load_bearing_change_does_not_bump(tmp_path: Path) -> None:
    _params_id(
        tmp_path, "conc",
        "param_id;value\nx;1\n",
        extras={"copper_conc.bib": "@article{foo,year=2022}"},
    )
    InputRegistry("params", root=tmp_path).initialize()

    # Edit only the non-load-bearing .bib file.
    _write(
        tmp_path / "params" / "conc" / "current" / "copper_conc.bib",
        "@article{foo,year=2024}",
    )
    reg = InputRegistry("params", root=tmp_path).initialize()

    assert reg.latest_version("conc") == 1
    assert not (tmp_path / "params" / "conc" / "v2").exists()


def test_unchanged_files_are_hardlinked_in_new_version(tmp_path: Path) -> None:
    _params_id(
        tmp_path, "conc",
        "param_id;value\nx;1\n",
        extras={"sidecar.txt": "unchanged across versions"},
    )
    InputRegistry("params", root=tmp_path).initialize()

    # Change params.csv (load-bearing) but leave sidecar.txt alone.
    _write(tmp_path / "params" / "conc" / "current" / "params.csv",
           "param_id;value\nx;2\n")
    InputRegistry("params", root=tmp_path).initialize()

    sidecar_v1 = tmp_path / "params" / "conc" / "v1" / "sidecar.txt"
    sidecar_v2 = tmp_path / "params" / "conc" / "v2" / "sidecar.txt"
    csv_v1 = tmp_path / "params" / "conc" / "v1" / "params.csv"
    csv_v2 = tmp_path / "params" / "conc" / "v2" / "params.csv"

    # Hardlink → same inode → link count >= 2 and identical st_ino.
    assert sidecar_v1.stat().st_ino == sidecar_v2.stat().st_ino
    assert sidecar_v1.stat().st_nlink >= 2

    # Changed file → fresh copy → different inodes.
    assert csv_v1.stat().st_ino != csv_v2.stat().st_ino
    assert csv_v1.read_text() != csv_v2.read_text()


def test_resolve_pin_overrides_latest(tmp_path: Path) -> None:
    _params_id(tmp_path, "conc", "v;1\n")
    InputRegistry("params", root=tmp_path).initialize()
    _write(tmp_path / "params" / "conc" / "current" / "params.csv", "v;2\n")
    InputRegistry("params", root=tmp_path).initialize()
    _write(tmp_path / "params" / "conc" / "current" / "params.csv", "v;3\n")
    InputRegistry("params", root=tmp_path).initialize()

    reg = InputRegistry("params", root=tmp_path).initialize(pins={"conc": 1})

    assert reg.resolve("conc") == (tmp_path / "params" / "conc" / "v1").resolve()
    assert reg.entry("conc")["version"] == 1
    # Explicit version arg overrides pin
    assert reg.entry("conc", version=3)["version"] == 3


def test_label_override_applied(tmp_path: Path) -> None:
    _params_id(tmp_path, "conc", "x\n")
    reg = InputRegistry("params", root=tmp_path).initialize(
        label_overrides={"conc": "Copper concentration (Schlesinger 2022)"})
    assert reg.entry("conc")["label"] == "Copper concentration (Schlesinger 2022)"


def test_label_carries_forward_across_bumps(tmp_path: Path) -> None:
    _params_id(tmp_path, "conc", "x\n")
    InputRegistry("params", root=tmp_path).initialize(
        label_overrides={"conc": "Original label"})

    # Change file → bump → new entry should keep the label without re-supplying it.
    _write(tmp_path / "params" / "conc" / "current" / "params.csv", "y\n")
    reg = InputRegistry("params", root=tmp_path).initialize()

    assert reg.entry("conc")["label"] == "Original label"


def test_label_override_updates_without_bump(tmp_path: Path) -> None:
    _params_id(tmp_path, "conc", "x\n")
    InputRegistry("params", root=tmp_path).initialize(label_overrides={"conc": "Old"})

    # Re-initialize with new label override but no file change.
    reg = InputRegistry("params", root=tmp_path).initialize(label_overrides={"conc": "New"})
    assert reg.latest_version("conc") == 1                       # no bump
    assert reg.entry("conc")["label"] == "New"


# =============================================================================
# Acc kind
# =============================================================================


def test_acc_default_load_bearing_set(tmp_path: Path) -> None:
    _acc_id(tmp_path, "copper", {
        "root.ods":   b"\x01root",
        "bounds.ods": b"\x01bounds",
        "rules.ods":  b"\x01rules",
        "notes.md":   b"# ignored",
    })

    InputRegistry("acc", root=tmp_path).initialize()

    # Change notes.md only → no bump.
    (tmp_path / "acc" / "copper" / "current" / "notes.md").write_bytes(b"# changed")
    reg = InputRegistry("acc", root=tmp_path).initialize()
    assert reg.latest_version("copper") == 1

    # Change root.ods → bump.
    (tmp_path / "acc" / "copper" / "current" / "root.ods").write_bytes(b"\x02root")
    reg = InputRegistry("acc", root=tmp_path).initialize()
    assert reg.latest_version("copper") == 2


# =============================================================================
# Feed kind
# =============================================================================


def test_feed_default_load_bearing_excludes_bib_md(tmp_path: Path) -> None:
    _feed_id(tmp_path, "baci", {
        "data.csv":      "year,value\n2020,1\n",
        "meta.xlsx":     "(binary surrogate)",
        "citation.bib":  "@article{baci}",
        "README.md":     "# baci feed",
    })
    InputRegistry("feed", root=tmp_path).initialize()

    # Edit a non-load-bearing file → no bump.
    _write(tmp_path / "feed" / "baci" / "current" / "citation.bib", "@article{baci2}")
    _write(tmp_path / "feed" / "baci" / "current" / "README.md", "# baci feed (rev)")
    reg = InputRegistry("feed", root=tmp_path).initialize()
    assert reg.latest_version("baci") == 1

    # Edit data.csv → bump.
    _write(tmp_path / "feed" / "baci" / "current" / "data.csv",
           "year,value\n2020,1\n2021,2\n")
    reg = InputRegistry("feed", root=tmp_path).initialize()
    assert reg.latest_version("baci") == 2


def test_feed_explicit_load_bearing_list(tmp_path: Path) -> None:
    _feed_id(tmp_path, "icsg", {
        "primary.csv":   "x,1\n",
        "secondary.csv": "y,2\n",
    })

    # Declare only primary.csv as load-bearing.
    reg = InputRegistry(
        "feed",
        root=tmp_path,
        feed_load_bearing={"icsg": ["primary.csv"]},
    ).initialize()
    assert reg.latest_version("icsg") == 1

    # Changing secondary.csv does NOT bump.
    _write(tmp_path / "feed" / "icsg" / "current" / "secondary.csv", "y,99\n")
    reg = InputRegistry(
        "feed", root=tmp_path,
        feed_load_bearing={"icsg": ["primary.csv"]},
    ).initialize()
    assert reg.latest_version("icsg") == 1

    # Changing primary.csv DOES bump.
    _write(tmp_path / "feed" / "icsg" / "current" / "primary.csv", "x,42\n")
    reg = InputRegistry(
        "feed", root=tmp_path,
        feed_load_bearing={"icsg": ["primary.csv"]},
    ).initialize()
    assert reg.latest_version("icsg") == 2


# =============================================================================
# InputRegistries convenience wrapper
# =============================================================================


def test_input_registries_initializes_all_three_kinds(tmp_path: Path) -> None:
    _params_id(tmp_path, "conc", "x\n")
    _acc_id(tmp_path, "copper", {"root.ods": b"r", "bounds.ods": b"b", "rules.ods": b"u"})
    _feed_id(tmp_path, "baci", {"data.csv": "x,1\n"})

    inputs = InputRegistries.initialize(root=tmp_path)

    assert inputs.params.entry("conc")["version"] == 1
    assert inputs.acc.entry("copper")["version"] == 1
    assert inputs.feed.entry("baci")["version"] == 1

    # Per-kind registry files exist.
    assert (tmp_path / "params" / "registry.json").is_file()
    assert (tmp_path / "acc" / "registry.json").is_file()
    assert (tmp_path / "feed" / "registry.json").is_file()


def test_input_registries_pins_per_kind(tmp_path: Path) -> None:
    _params_id(tmp_path, "conc", "v;1\n")
    InputRegistries.initialize(root=tmp_path)
    _write(tmp_path / "params" / "conc" / "current" / "params.csv", "v;2\n")
    InputRegistries.initialize(root=tmp_path)

    inputs = InputRegistries.initialize(
        root=tmp_path,
        pins={"params": {"conc": 1}},
    )
    assert inputs.params.entry("conc")["version"] == 1


# =============================================================================
# Registry JSON shape
# =============================================================================


def test_registry_json_shape(tmp_path: Path) -> None:
    _params_id(tmp_path, "conc", "x\n")
    InputRegistry("params", root=tmp_path).initialize(
        label_overrides={"conc": "Cu concentration params"})

    payload = json.loads((tmp_path / "params" / "registry.json").read_text())
    assert "conc" in payload
    rec = payload["conc"]
    assert rec["label"] == "Cu concentration params"
    assert isinstance(rec["versions"], list) and len(rec["versions"]) == 1
    v1 = rec["versions"][0]
    assert v1["version"] == 1
    assert set(v1.keys()) == {"version", "date", "code", "hashes"}
    assert "params.csv" in v1["hashes"]
    assert len(v1["hashes"]["params.csv"]) == 64                  # sha256 hex
