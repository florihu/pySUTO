"""The Stage 1 build scripts and the BACI / crude-ore parses, on toy inputs in ``tmp_path`` (#281).

``build_trade_shares`` (the #53 record, the BACI-feed-in-force guard), ``build_baci_reexport``
(the snapshot's sha256 refusal, the quantity sidecars), ``build_ore_grade`` (its parents
record, the factor fallback), the stage lock a hand run takes, ``baci_parse_feed``'s σ
sidecar and ``main`` (ADR 0033), and the crude-ore parse. No script's ``__main__`` runs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import copper_case  # noqa: F401  (puts scripts/ on sys.path, #189)
import copper_feed
import build_baci_reexport as R
import build_ore_grade as O
import build_trade_shares as T
from pysuto.registry import stage_lock
from pysuto.registry.lock import HOLDER_ENV
from pysuto.input_registry import InputRegistries
from pysuto.registry import LockHeld

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS / "parse" / "baci_parse"))
sys.path.insert(0, str(_SCRIPTS / "parse"))
import baci_parse_feed as B  # noqa: E402
import crude_ore_parse_feed as C  # noqa: E402


def _registered(root, feeds=("baci", "icsg"), params=("conc", "eff")):
    for kind, ids, name in (("feed", feeds, "values.csv"), ("params", params, "params.csv")):
        for id_ in ids:
            f = root / kind / id_ / "current" / name
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("x\n1\n")
    return InputRegistries.initialize(root=root)


def _feed_run(proc, label, stamp, baci_version=None):
    run = proc / "feed" / "registry" / f"feed_{label}_{stamp}"
    run.mkdir(parents=True)
    if baci_version is not None:
        (run / "metadata.json").write_text(json.dumps(
            {"parents": {"inputs": {"feed": {"id": "baci", "version": baci_version}}}}))
    return run


# ---------------------------------------------------------------------------
# The stage lock of a hand run (U1b-G4-6)
# ---------------------------------------------------------------------------

def test_a_hand_run_holds_the_stage_lock_and_a_second_one_is_refused(tmp_path):
    with stage_lock("build_trade_shares", root=str(tmp_path / "proc")):
        assert (tmp_path / "proc" / ".stage.lock").is_file()
        with pytest.raises(LockHeld, match="build_trade_shares"):
            with stage_lock("build_ore_grade", root=str(tmp_path / "proc")):
                pass
    assert not (tmp_path / "proc" / ".stage.lock").exists()


def test_a_build_the_hook_runs_joins_the_lock_its_pipeline_holds(tmp_path, monkeypatch):
    """The pipeline holds the lock for its feed step and names itself as holder in the
    child's environment (``HOLDER_ENV``); a builder started there neither refuses nor
    removes the lock (#279 keeps this one mechanism for U1b-G4-6)."""
    proc = str(tmp_path / "proc")
    with stage_lock("pipeline_feed", root=proc):
        monkeypatch.setenv(HOLDER_ENV, str(os.getpid()))
        with stage_lock("build_trade_shares", root=proc):
            pass
        assert (tmp_path / "proc" / ".stage.lock").is_file()


def test_the_hook_s_script_children_inherit_the_holder_marker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(copper_feed, "_SCRIPTS_DIR", tmp_path)
    monkeypatch.setenv(HOLDER_ENV, "4242")
    (tmp_path / "probe.py").write_text(
        f"import os\nprint('holder=' + os.environ.get({HOLDER_ENV!r}, 'no'))\n")
    copper_feed._run_script("probe.py", data_dir=str(tmp_path))
    assert "holder=4242" in (tmp_path / "logs" / "probe.log").read_text()


# ---------------------------------------------------------------------------
# build_trade_shares: the #53 record and the BACI feed in force (U1b-G4-5, G6-2)
# ---------------------------------------------------------------------------

def test_the_weights_record_the_baci_run_and_the_versions_they_read(tmp_path, monkeypatch):
    _registered(tmp_path / "input")
    _feed_run(tmp_path / "proc", "baci", "20261019_100000_aaaaaaaa", 1)
    monkeypatch.setattr(T, "_INPUT_ROOT", str(tmp_path / "input"))
    monkeypatch.setattr(T, "FEED_REG", str(tmp_path / "proc" / "feed" / "registry"))
    out = tmp_path / "out"
    out.mkdir()
    T.write_parents(str(out), T._input_registries())
    assert json.loads((out / "parents.json").read_text()) == {
        "upstream": {"feed_run_ids": ["feed_baci_20261019_100000_aaaaaaaa"]},
        "inputs": {"feed": {"baci": 1, "icsg": 1}, "params": {"conc": 1, "eff": 1}}}


def test_a_baci_feed_run_built_from_the_input_in_force_passes(tmp_path):
    inputs = _registered(tmp_path / "input")
    _feed_run(tmp_path / "proc", "baci", "20261019_100000_aaaaaaaa", 1)
    T.check_baci_feed_current(inputs, str(tmp_path / "proc" / "feed" / "registry"))


def test_a_baci_input_registered_after_baci_feed_ran_is_refused(tmp_path):
    """The weights would name a feed run that read the older matrix: the guard of #53
    compares the run's id, not the input version that run was built from."""
    root = tmp_path / "input"
    _registered(root)
    _feed_run(tmp_path / "proc", "baci", "20261019_100000_aaaaaaaa", 1)
    (root / "feed" / "baci" / "current" / "values.csv").write_text("x\n2\n")
    inputs = InputRegistries.initialize(root=root)               # baci v2
    with pytest.raises(ValueError, match="rerun baci_feed first"):
        T.check_baci_feed_current(inputs, str(tmp_path / "proc" / "feed" / "registry"))


# ---------------------------------------------------------------------------
# build_baci_reexport (U1b-G4-9, G3-14, G6-2)
# ---------------------------------------------------------------------------

def _gross_baci_registered(tmp_path, monkeypatch):
    root = tmp_path / "input"
    _registered(root, feeds=("baci",), params=())
    monkeypatch.setattr(R, "_INPUT_ROOT", str(root))
    return root


def test_the_netting_reads_the_registered_snapshot_it_was_given(tmp_path, monkeypatch):
    root = _gross_baci_registered(tmp_path, monkeypatch)
    assert R._source_dir(1) == str(root / "feed" / "baci" / "v1")


def test_a_modified_snapshot_is_refused(tmp_path, monkeypatch):
    root = _gross_baci_registered(tmp_path, monkeypatch)
    (root / "feed" / "baci" / "v1" / "values.csv").write_text("x\n999\n")
    with pytest.raises(ValueError, match="snapshot was modified"):
        R._source_dir(1)


def test_an_unregistered_snapshot_is_refused(tmp_path, monkeypatch):
    _gross_baci_registered(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="not in the input registry"):
        R._source_dir(9)


def test_the_source_version_has_no_default():
    """A stale default (5; the reference chain nets 33) once netted whatever was typed."""
    r = subprocess.run([sys.executable, str(_SCRIPTS / "build_baci_reexport.py")],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "--source-version" in r.stderr


def test_the_quantity_sidecars_travel_with_the_netted_feed(tmp_path):
    src, out = tmp_path / "v1", tmp_path / "current"
    src.mkdir(), out.mkdir()
    (src / "quantity_cv_scale.csv").write_text("new\n")
    (src / "quantity_recon.csv").write_text("recon\n")
    (out / "quantity_cv_scale.csv").write_text("old\n")
    R.carry_quantity_sidecars(str(src), str(out))
    assert (out / "quantity_cv_scale.csv").read_text() == "new\n"
    assert (out / "quantity_recon.csv").read_text() == "recon\n"


def test_a_source_without_the_sidecar_leaves_no_older_copy_to_scale_sigma(tmp_path):
    src, out = tmp_path / "v1", tmp_path / "current"
    src.mkdir(), out.mkdir()
    (out / "quantity_cv_scale.csv").write_text("old\n")
    R.carry_quantity_sidecars(str(src), str(out))
    assert not (out / "quantity_cv_scale.csv").exists()


# ---------------------------------------------------------------------------
# build_ore_grade (U1b-G3-8, G3-18, G4-1, S1-4)
# ---------------------------------------------------------------------------

def test_the_ore_grade_builder_follows_the_data_dir(tmp_path, monkeypatch):
    for n in ("_INPUT_ROOT", "ACC_DIR", "FEED_REG", "OG_DIR", "OUT_CSV"):
        monkeypatch.setattr(O, n, getattr(O, n))        # restore the module's paths afterwards
    O.set_data_dir(str(tmp_path))
    assert O._INPUT_ROOT == str(tmp_path / "input")
    assert O.FEED_REG == str(tmp_path / "proc" / "feed" / "registry")
    assert O.OUT_CSV == str(tmp_path / "input" / "params" / "ore_grade" / "current" / "params.csv")


def test_the_newest_feed_run_supplies_the_ore_grade_production(tmp_path, monkeypatch):
    reg = tmp_path / "proc" / "feed" / "registry"
    for stamp, v in (("20261101_100000_aaaaaaaa", 1.0), ("20261102_100000_bbbbbbbb", 2.0)):
        run = reg / f"feed_icsg_{stamp}"
        run.mkdir(parents=True)
        np.savez(run / "result.npz", F_root=np.array([v]))
    monkeypatch.setattr(O, "FEED_REG", str(reg))
    run_id, froot = O._latest_feed_froot("icsg")
    assert run_id == "feed_icsg_20261102_100000_bbbbbbbb" and froot.tolist() == [2.0]
    with pytest.raises(FileNotFoundError, match="feed_crude_ore_"):
        O._latest_feed_froot("crude_ore")


def test_stale_ore_grade_params_can_be_told_by_their_recorded_parents(tmp_path, monkeypatch):
    """Like trade_shares (#53): the runs and the eff / conc versions the grades were built
    from sit beside params.csv, where the params registry hashes them."""
    root = tmp_path / "input"
    _registered(root, feeds=(), params=("conc", "eff"))
    monkeypatch.setattr(O, "_INPUT_ROOT", str(root))
    out = root / "params" / "ore_grade" / "current"
    out.mkdir(parents=True)
    (out / "params.csv").write_text("x\n1\n")
    built = O.write_parents(str(out), ["feed_icsg_a", "feed_crude_ore_b"])
    assert built == {"upstream": {"feed_run_ids": ["feed_icsg_a", "feed_crude_ore_b"]},
                     "inputs": {"params": {"conc": 1, "eff": 1}}}
    assert json.loads((out / "parents.json").read_text()) == built
    before = InputRegistries.initialize(root=root).params.entry("ore_grade")["version"]
    O.write_parents(str(out), ["feed_icsg_c", "feed_crude_ore_b"])
    assert InputRegistries.initialize(root=root).params.entry("ore_grade")["version"] == before + 1


def test_the_factor_fallback_averages_the_members_of_row(tmp_path, monkeypatch):
    og = tmp_path / "og"
    og.mkdir()
    pd.DataFrame({"material_id": ["Me.Cu"] * 4 + ["Me.Au"],
                  "alphanumiso": ["CHL", "AAA", "BBB", "AAA", "CHL"],
                  "year": [2000, 2000, 2000, 2001, 2000],
                  "factor": [150.0, 100.0, 300.0, 50.0, 9.0]}
                 ).to_csv(og / "all_integrated_est_fac_2024.csv", index=False)
    with pd.ExcelWriter(og / "ore_grade_map.xlsx") as w:
        pd.DataFrame({"alphanumiso": ["CHL", "AAA", "BBB"],
                      "root_region": ["Chile", "ROW", "ROW"]}
                     ).to_excel(w, sheet_name="alphanumiso_map", index=False)
    monkeypatch.setattr(O, "OG_DIR", str(og))
    out = O.load_factor_fallback().set_index(["region", "time"])["factor"]
    assert out.to_dict() == {("Chile", "2000"): 150.0, ("ROW", "2000"): 200.0,
                             ("ROW", "2001"): 50.0}


# ---------------------------------------------------------------------------
# The BACI parse (U1b-S1-1): the σ sidecar and main
# ---------------------------------------------------------------------------

def _baci_concordance(root):
    conc = root / "feed" / "baci" / "current" / "conc.xlsx"
    conc.parent.mkdir(parents=True)
    with pd.ExcelWriter(conc) as w:
        pd.DataFrame({"src_sector": [260300, 740311], "root_sector": ["Concentrate", "Cathode"]}
                     ).to_excel(w, sheet_name="sector_map", index=False)
        pd.DataFrame({"src_region": [152, 604, 156, 276, 40, 4, 8],
                      "root_region": ["Chile", "Peru", "China", "Germany", "Austria",
                                      "ROW", "ROW"]}
                     ).to_excel(w, sheet_name="region_map", index=False)


def _baci_paths(tmp_path, monkeypatch):
    out = tmp_path / "v0"
    monkeypatch.setattr(B, "_INPUT_ROOT", str(tmp_path / "input"))
    monkeypatch.setattr(B, "OUT_DIR", str(out))
    monkeypatch.setattr(B, "OUT_PATH", str(out / "values.csv"))
    monkeypatch.setattr(B, "RECON_PATH", str(out / "quantity_recon.csv"))
    monkeypatch.setattr(B, "CV_SCALE_PATH", str(out / "quantity_cv_scale.csv"))
    _baci_concordance(tmp_path / "input")
    return out


def test_flagged_legs_folded_onto_one_root_leg_are_weighted_by_the_mass_they_carry(
        tmp_path, monkeypatch):
    """Two BACI codes (4, 8) both fold onto ROW; the root leg's σ multiplier is the
    mass-weighted mean over the flagged source legs, an unflagged leg is no root leg."""
    out = _baci_paths(tmp_path, monkeypatch)
    out.mkdir()
    rec = pd.DataFrame({
        "Year": 2020, "Sector": 260300, "Region_from": 152, "Region_to": [4, 8, 156],
        "side": ["above", "above", "none"], "v": 1.0, "q_obs": [100.0, 50.0, 70.0],
        "q_new": [200.0, 300.0, 70.0], "cu_frac": 0.6, "year_frac": 0.6, "g_ref": 0.275,
        "cv_scale": [2.0, 4.0, 1.0], "flagged": [True, True, False]})
    B._write_sidecars(rec, None)
    scale = pd.read_csv(out / "quantity_cv_scale.csv")
    assert scale[["sector", "region_from", "region_to", "time"]].values.tolist() == [
        ["Concentrate", "Chile", "ROW", 2020]]
    assert scale["q_new"].iloc[0] == 500.0
    assert scale["cv_scale"].iloc[0] == pytest.approx((2.0 * 200 + 4.0 * 300) / 500, rel=1e-12)
    assert len(pd.read_csv(out / "quantity_recon.csv")) == 2          # the flagged legs only


def _baci_year(path, rows):
    pd.DataFrame(rows, columns=["t", "i", "j", "k", "v", "q"]).to_csv(path, index=False)


def test_the_parse_converts_tonnes_to_kt_and_rebuilds_a_leg_whose_value_contradicts_it(
        tmp_path, monkeypatch):
    out = _baci_paths(tmp_path, monkeypatch)
    raw = tmp_path / "raw"
    raw.mkdir()
    # cathode: 6,000 USD/t. Concentrate at 0.275 of it is in band (Peru); Chile's carries
    # 0.65, kilotonnes of copper's worth of value on too little mass (ADR 0033, above).
    _baci_year(raw / "BACI_HS92_Y2000_V202601.csv", [
        (2000, 276, 40, 740311, 6000.0, 1000.0),
        (2000, 604, 156, 260300, 16500.0, 10000.0),
        (2000, 152, 156, 260300, 39000.0, 10000.0),
        (2000, 180, 834, 740311, 600.0, 100.0),                 # a transit declaration (#95)
        (2000, 152, 156, 740110, 500.0, np.nan)])               # no quantity: dropped
    _baci_year(raw / "BACI_HS92_Y2001_V202601.csv", [(2001, 276, 40, 740311, 6000.0, 1000.0)])
    monkeypatch.setattr(B, "BACI_RAW_DIR", str(raw))
    monkeypatch.setattr(B, "TRANSIT_DECLARATIONS", {180: (834,)})
    monkeypatch.setattr(B, "MIRROR_DUPLICATES", {})

    B.main()

    vals = pd.read_csv(out / "values.csv")
    assert list(vals.columns) == ["Year", "Sector", "Region_from", "Region_to", "Value", "Unit"]
    assert (vals["Unit"] == "kt").all()
    by = vals.set_index(["Year", "Sector", "Region_from"])["Value"]
    assert by[(2000, 740311, 276)] == pytest.approx(1.0, rel=1e-12)          # 1,000 t
    assert by[(2000, 260300, 604)] == pytest.approx(10.0, rel=1e-12)         # in band: as reported
    # q_new = v / (P_cu * g_ref): 39,000 kUSD / (6,000 USD/t * 0.275 of the conc nominal)
    assert by[(2000, 260300, 152)] == pytest.approx(39000.0 * 1000 / (6000.0 * B.G_NOM) / 1000,
                                                    rel=1e-12)
    assert (2000, 740110, 152) not in by.index
    assert (2000, 740311, 180) not in by.index                  # transit dropped before the price
    scale = pd.read_csv(out / "quantity_cv_scale.csv")
    assert scale[["sector", "region_from", "region_to"]].values.tolist() == [
        ["Concentrate", "Chile", "China"]]
    assert scale["cv_scale"].iloc[0] == B.MAX_CV_SCALE        # one in-band year: ignorance


# ---------------------------------------------------------------------------
# The crude-ore parse (U1b-S1-5)
# ---------------------------------------------------------------------------

def test_the_crude_ore_parse_keeps_copper_ore_in_kt_from_1995(tmp_path, monkeypatch):
    raw = tmp_path / "kroiss.xlsx"
    pd.DataFrame({
        "material_name": ["Copper - associated ore"] * 5 + ["Gold - associated ore"],
        "country_name":  ["Chile", "Chile", "Peru", "Peru", "Peru", "Chile"],
        "year":          [2000, 1990, 2000, 2001, 2002, 2000],
        "value":         [5000.0, 7000.0, 2500.0, 0.0, np.nan, 8000.0]}
    ).to_excel(raw, sheet_name=C.RAW_SHEET, index=False)
    monkeypatch.setattr(C, "RAW_FILE", str(raw))
    monkeypatch.setattr(C, "OUT_PATH", str(tmp_path / "v0" / "values.csv"))
    monkeypatch.setattr(C, "start_run_log", lambda *a, **k: None)

    C.main()

    out = pd.read_csv(tmp_path / "v0" / "values.csv")
    assert list(out.columns) == ["Year", "Material", "Region", "Value", "Unit"]
    assert out.values.tolist() == [
        [2000, "Copper - associated ore", "Chile", 5.0, "kt"],
        [2000, "Copper - associated ore", "Peru", 2.5, "kt"]]
