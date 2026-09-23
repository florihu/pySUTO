"""
test_psut.py
============
Stage 6 · Slices 1–2 — the tidy-canonical PSUT (issues #18, #19; ADR 0037).

Fixture topology (psut_toy_acc)
-------------------------------
Regions A and B carry a two-process chain; region C has nodes but no edges
(labels that must still appear in every block view).

  Nature/SB/r  →  Mining/Process/r      (E)
  Mining/Process/r  →  Ore/Flow/r       (S)
  Ore/Flow/r  →  Smelting/Process/r     (U, domestic)
  Ore/Flow/B  →  Smelting/Process/A     (U, cross-region)
  Scrap/SB/A  →  Smelting/Process/A     (O)
  Smelting/Process/r  →  Cathode/Flow/r (S)
  Smelting/Process/r  →  Slag/SB/r      (P)
  Cathode/Flow/r  →  Use/SB/r           (Y)

14 edges per (period, valuation) × periods 2020, 2021 × valuations Mass, CC
= 56 edges. CC values (2020; 2021 = ×1.1; Mass = ×3) are chosen so every
Process and Flow node balances exactly from the table alone.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

PERIODS    = ["2020", "2021"]
VALUATIONS = ["Mass", "CC"]

NODES = [
    # sector      entity     region  entity_sub
    ("Nature",   "SB",      "A", "Nature"),    # 0
    ("Mining",   "Process", "A", None),        # 1
    ("Ore",      "Flow",    "A", None),        # 2
    ("Smelting", "Process", "A", None),        # 3
    ("Cathode",  "Flow",    "A", None),        # 4
    ("Slag",     "SB",      "A", "SEM"),       # 5
    ("Use",      "SB",      "A", "FD"),        # 6
    ("Nature",   "SB",      "B", "Nature"),    # 7
    ("Mining",   "Process", "B", None),        # 8
    ("Ore",      "Flow",    "B", None),        # 9
    ("Smelting", "Process", "B", None),        # 10
    ("Cathode",  "Flow",    "B", None),        # 11
    ("Slag",     "SB",      "B", "SEM"),       # 12
    ("Use",      "SB",      "B", "FD"),        # 13
    ("Mining",   "Process", "C", None),        # 14  no edges
    ("Ore",      "Flow",    "C", None),        # 15  no edges
    ("Smelting", "Process", "C", None),        # 16  no edges
    ("Cathode",  "Flow",    "C", None),        # 17  no edges
    ("Scrap",    "SB",      "A", "SEM"),       # 18
]

# (from_id, to_id, block, CC value in 2020)
EDGES = [
    (0, 1,   "E", 100.0),
    (1, 2,   "S", 100.0),
    (2, 3,   "U", 100.0),
    (9, 3,   "U",  20.0),      # cross-region use B → A
    (18, 3,  "O",   5.0),      # scrap into Smelting/A
    (3, 4,   "S", 115.0),
    (3, 5,   "P",  10.0),
    (4, 6,   "Y", 115.0),
    (7, 8,   "E",  60.0),
    (8, 9,   "S",  60.0),
    (9, 10,  "U",  40.0),
    (10, 11, "S",  36.0),
    (10, 12, "P",   4.0),
    (11, 13, "Y",  36.0),
]
N_PER_SLICE = len(EDGES)
BLOCK_SIZES = {"S": 4, "U": 3, "Y": 2, "E": 2, "O": 1, "P": 2}
CC_TOTALS_2020 = {"S": 311.0, "U": 160.0, "Y": 151.0, "E": 160.0, "O": 5.0, "P": 14.0}


def _scale(t: str, val: str) -> float:
    return (1.1 if t == "2021" else 1.0) * (3.0 if val == "Mass" else 1.0)


def _make_toy_acc():
    """Minimal AccountingStructure for the PSUT — no file I/O, no rules."""
    from pysuto.acc import AccountingStructure

    acc = AccountingStructure.__new__(AccountingStructure)
    acc.logger     = logging.getLogger("test_psut_acc")
    acc.id         = "toy"
    acc.node_index = pd.DataFrame(NODES, columns=["sector", "entity", "region", "entity_sub"])
    acc.n_nodes    = len(NODES)
    acc.time_range = list(PERIODS)
    acc.valuation  = list(VALUATIONS)

    rows, eid = [], 0
    for t in PERIODS:
        for val in VALUATIONS:
            for fi, ti, _, _ in EDGES:
                rows.append({"edge_id": eid, "from_id": fi, "to_id": ti,
                             "time": t, "valuation": val})
                eid += 1
    acc.edge_index = pd.DataFrame(rows)
    acc.n_edges    = len(acc.edge_index)
    return acc


def _full_vector(acc) -> np.ndarray:
    """The complete a* over every edge of the toy structure."""
    F = np.zeros(acc.n_edges)
    for eid, (_, row) in enumerate(acc.edge_index.iterrows()):
        base = EDGES[eid % N_PER_SLICE][3]
        F[eid] = base * _scale(row["time"], row["valuation"])
    return F


def _legacy_results(acc, F):
    """Legacy full-vector path: every timestep carries the whole vector."""
    return {t: SimpleNamespace(timestep=t, F_opt=F.copy(), cols_t=None) for t in PERIODS}


def _block_results(acc, F, seed=0):
    """Block path: year-local F_opt over a *shuffled* cols_t, so a positional
    read would land on the wrong edges."""
    rng = np.random.default_rng(seed)
    out = {}
    for t in PERIODS:
        cols = rng.permutation(np.flatnonzero(acc.edge_index["time"].to_numpy() == t))
        out[t] = SimpleNamespace(timestep=t, F_opt=F[cols], cols_t=cols)
    return out


def _optim_results(acc, F):
    """Real OptimResult objects on the block path, for registry-backed tests."""
    from pysuto.solver import OptimResult
    out = {}
    for t, r in _block_results(acc, F).items():
        out[t] = OptimResult(
            timestep=t, F_opt=r.F_opt, eps_opt=np.zeros(0), c_star=np.zeros(0),
            residuals=np.zeros(0), objective=0.0, iterations=0, solve_time_s=0.0,
            solver_status="Solved", convergence={}, cols_t=r.cols_t,
            rows_t=np.zeros(0, dtype=np.int64),
        )
    return out


@pytest.fixture
def toy_acc():
    return _make_toy_acc()


@pytest.fixture
def F(toy_acc):
    return _full_vector(toy_acc)


@pytest.fixture
def psut(toy_acc, F):
    from pysuto.psut import PSUT
    return PSUT.from_results(_legacy_results(toy_acc, F), toy_acc)


# ---------------------------------------------------------------------------
# 1) Block decomposition
# ---------------------------------------------------------------------------

def test_every_edge_lands_in_exactly_one_block(toy_acc, psut):
    assert set(psut.table["edge_id"]) == set(toy_acc.edge_index["edge_id"])
    assert psut.table["edge_id"].is_unique                     # exactly once
    assert set(psut.table["block"]) == set(BLOCK_SIZES)
    sizes = psut.table.groupby(["period", "valuation", "block"]).size()
    for t in PERIODS:
        for v in VALUATIONS:
            for b, n in BLOCK_SIZES.items():
                assert sizes[(t, v, b)] == n
    expected = {b: (t, v, b) for t in PERIODS for v in VALUATIONS for b in BLOCK_SIZES}
    assert len(sizes) == len(PERIODS) * len(VALUATIONS) * len(BLOCK_SIZES) == 24 and expected


def test_block_matches_the_edge_endpoints(psut):
    kinds = psut.table.groupby("block")[["from_entity", "to_entity"]].agg(lambda s: set(s))
    assert kinds.loc["S"].tolist() == [{"Process"}, {"Flow"}]
    assert kinds.loc["U"].tolist() == [{"Flow"}, {"Process"}]
    assert kinds.loc["Y"].tolist() == [{"Flow"}, {"SB"}]
    assert kinds.loc["E"].tolist() == [{"SB"}, {"Process"}]
    assert kinds.loc["O"].tolist() == [{"SB"}, {"Process"}]
    assert kinds.loc["P"].tolist() == [{"Process"}, {"SB"}]
    assert set(psut.table.loc[psut.table["block"] == "E", "from_sector"]) == {"Nature"}
    assert set(psut.table.loc[psut.table["block"] == "O", "from_sector"]) == {"Scrap"}


def test_edge_outside_the_taxonomy_raises(toy_acc):
    """A Flow → SB·SEM edge matches no block rule: an error, not a silent drop."""
    from pysuto.psut import PSUT
    extra = pd.DataFrame([{"edge_id": toy_acc.n_edges, "from_id": 4, "to_id": 5,
                           "time": "2020", "valuation": "CC"}])
    toy_acc.edge_index = pd.concat([toy_acc.edge_index, extra], ignore_index=True)
    toy_acc.n_edges += 1
    F = np.append(_full_vector(toy_acc)[:-1], 1.0)
    with pytest.raises(ValueError, match=r"outside the PSUT block taxonomy.*'Flow'.*'SB', 'SEM'"):
        PSUT.from_results(_legacy_results(toy_acc, F), toy_acc)


def test_unknown_primary_raises(toy_acc, F):
    from pysuto.psut import PSUT
    with pytest.raises(ValueError, match="primary 'Gross'"):
        PSUT.from_results(_legacy_results(toy_acc, F), toy_acc, primary="Gross")


# ---------------------------------------------------------------------------
# 2) Block views: labels from the node universe, values from a*
# ---------------------------------------------------------------------------

PROCS = pd.MultiIndex.from_tuples(
    [("A", "Mining"), ("A", "Smelting"), ("B", "Mining"), ("B", "Smelting"),
     ("C", "Mining"), ("C", "Smelting")], names=["region", "process"])
FLOWS = pd.MultiIndex.from_tuples(
    [("A", "Cathode"), ("A", "Ore"), ("B", "Cathode"), ("B", "Ore"),
     ("C", "Cathode"), ("C", "Ore")], names=["region", "flow"])


def test_views_are_labelled_from_the_node_universe(psut):
    S, U = psut.S("2020"), psut.U("2020")
    assert S.index.equals(PROCS) and S.columns.equals(FLOWS)
    assert U.index.equals(FLOWS) and U.columns.equals(PROCS)
    assert S.index.names == ["region", "process"] and S.columns.names == ["region", "flow"]
    # region C has nodes but no edges: present, all zero
    assert (S.loc["C"] == 0).all().all() and (S["C"] == 0).all().all()
    # identical labels in every period
    assert psut.S("2021").index.equals(S.index) and psut.U("2021").columns.equals(U.columns)


def test_boundary_views_use_role_named_levels(psut):
    Y, E, O, P, W = (psut.Y("2020"), psut.E("2020"), psut.O("2020"),
                     psut.P("2020"), psut.W("2020"))
    fd  = pd.MultiIndex.from_tuples([("A", "Use"), ("B", "Use")], names=["region", "final_demand"])
    nat = pd.MultiIndex.from_tuples([("A", "Nature"), ("B", "Nature")], names=["region", "nature"])
    sec = pd.MultiIndex.from_tuples([("A", "Scrap"), ("A", "Slag"), ("B", "Slag")],
                                    names=["region", "secondary"])
    assert Y.index.equals(FLOWS) and Y.columns.equals(fd)
    assert E.index.equals(nat) and E.columns.equals(PROCS)
    assert O.index.equals(sec) and O.columns.equals(PROCS)
    assert P.index.equals(PROCS) and P.columns.equals(sec)
    assert W.index.equals(PROCS) and W.columns.equals(nat)
    assert W.to_numpy().sum() == 0.0                          # empty block, still shaped
    with pytest.raises(KeyError, match="unknown block"):
        psut.block("Z")


def test_views_hold_the_solved_values(psut):
    S, U, Y, E, O, P = (psut.S("2020"), psut.U("2020"), psut.Y("2020"),
                        psut.E("2020"), psut.O("2020"), psut.P("2020"))
    assert S.loc[("A", "Mining"), ("A", "Ore")] == 100.0
    assert S.loc[("A", "Smelting"), ("A", "Cathode")] == 115.0
    assert U.loc[("B", "Ore"), ("A", "Smelting")] == 20.0        # cross-region use
    assert U.loc[("A", "Ore"), ("B", "Smelting")] == 0.0          # no such edge
    assert Y.loc[("A", "Cathode"), ("A", "Use")] == 115.0
    assert E.loc[("A", "Nature"), ("A", "Mining")] == 100.0
    assert O.loc[("A", "Scrap"), ("A", "Smelting")] == 5.0
    assert P.loc[("B", "Smelting"), ("B", "Slag")] == 4.0
    for b, total in CC_TOTALS_2020.items():
        assert psut.block(b, "2020").to_numpy().sum() == pytest.approx(total)
    assert psut.S("2021").loc[("A", "Mining"), ("A", "Ore")] == pytest.approx(110.0)


def test_per_node_mass_balance_from_the_table_alone(psut):
    """Flow nodes: supply in = use out + final use out.
    Process nodes: use in + nature in + secondary in = supply out + residual out + discard."""
    for period in PERIODS:
        for valuation in VALUATIONS:
            b = {k: psut.block(k, period, valuation) for k in "SUYEOPW"}
            flow_in  = b["S"].sum(axis=0)
            flow_out = b["U"].sum(axis=1) + b["Y"].sum(axis=1)
            assert flow_in.index.equals(flow_out.index)
            np.testing.assert_allclose(flow_in.to_numpy(), flow_out.to_numpy(), atol=1e-9)
            proc_in  = b["U"].sum(axis=0) + b["E"].sum(axis=0) + b["O"].sum(axis=0)
            proc_out = b["S"].sum(axis=1) + b["P"].sum(axis=1) + b["W"].sum(axis=1)
            assert proc_in.index.equals(proc_out.index)
            np.testing.assert_allclose(proc_in.to_numpy(), proc_out.to_numpy(), atol=1e-9)
            assert proc_in.to_numpy().sum() > 0                 # not vacuous


# ---------------------------------------------------------------------------
# 3) Valuations: every one in the table, the primary as the default
# ---------------------------------------------------------------------------

def test_every_valuation_in_one_table_with_a_primary(psut):
    assert psut.valuations == ["Mass", "CC"] and psut.primary == "CC"
    assert set(psut.table["valuation"]) == {"Mass", "CC"}
    pd.testing.assert_frame_equal(psut.S("2020"), psut.S("2020", valuation="CC"))
    pd.testing.assert_frame_equal(psut.S("2020", valuation="Mass"), 3.0 * psut.S("2020"))
    with pytest.raises(KeyError, match="valuation 'Gross'"):
        psut.U("2020", valuation="Gross")


def test_realized_conc_is_the_ratio_of_two_valuations(psut):
    rc = psut.realized_conc()
    assert len(rc) == N_PER_SLICE * len(PERIODS)
    assert {"val_CC", "val_Mass", "conc"} <= set(rc.columns)
    assert "edge_id" not in rc.columns                        # joined on the edge template
    np.testing.assert_allclose(rc["conc"].to_numpy(), 1.0 / 3.0)
    one = psut.realized_conc("2021", numerator="Mass")
    assert len(one) == N_PER_SLICE and set(one["period"]) == {"2021"}
    np.testing.assert_allclose(one["conc"].to_numpy(), 3.0)
    with pytest.raises(ValueError, match="same valuation"):
        psut.realized_conc(numerator="CC", denominator="CC")


def test_realized_conc_is_nan_where_the_denominator_is_zero(toy_acc, F):
    from pysuto.psut import PSUT
    F = F.copy()
    F[0] = 0.0                                                # 2020 / Mass / edge 0 (E)
    rc = PSUT.from_results(_legacy_results(toy_acc, F), toy_acc).realized_conc("2020")
    row = rc[(rc["block"] == "E") & (rc["to_region"] == "A")]
    assert len(row) == 1 and np.isnan(row["conc"].iloc[0])
    assert rc["conc"].notna().sum() == N_PER_SLICE - 1


# ---------------------------------------------------------------------------
# 4) Year-local cols_t → global edge ids
# ---------------------------------------------------------------------------

def test_year_local_cols_t_maps_to_global_edges(toy_acc, F, psut):
    from pysuto.psut import PSUT
    block = PSUT.from_results(_block_results(toy_acc, F), toy_acc)
    pd.testing.assert_frame_equal(block.table, psut.table)
    pd.testing.assert_frame_equal(block.Y("2021", "Mass"), psut.Y("2021", "Mass"))


def test_cols_t_not_covering_the_period_raises(toy_acc, F):
    from pysuto.psut import PSUT
    res = _block_results(toy_acc, F)
    res["2021"] = SimpleNamespace(timestep="2021", F_opt=res["2020"].F_opt, cols_t=res["2020"].cols_t)
    with pytest.raises(ValueError, match="not covered by cols_t"):
        PSUT.from_results(res, toy_acc)


def test_legacy_vector_of_wrong_length_raises(toy_acc, F):
    from pysuto.psut import PSUT
    res = {"2020": SimpleNamespace(timestep="2020", F_opt=F[:10], cols_t=None)}
    with pytest.raises(ValueError, match="cols_t is None"):
        PSUT.from_results(res, toy_acc)


def test_results_key_must_match_timestep(toy_acc, F):
    from pysuto.psut import PSUT
    res = {"2021": SimpleNamespace(timestep="2020", F_opt=F.copy(), cols_t=None)}
    with pytest.raises(ValueError, match="holds a result for timestep"):
        PSUT.from_results(res, toy_acc)


# ---------------------------------------------------------------------------
# 5) Time is carried, never aggregated
# ---------------------------------------------------------------------------

def test_periods_latest_at(psut):
    assert psut.periods() == ["2020", "2021"]
    assert psut.latest() == "2021"
    one = psut.at(2020)                      # int coerced to the label
    assert one.periods() == ["2020"]
    assert (one.primary, one.valuations, one.acc_id) == ("CC", ["Mass", "CC"], "toy")
    pd.testing.assert_frame_equal(one.S(), psut.S("2020"))     # default = latest of the slice
    pd.testing.assert_frame_equal(psut.S(), psut.S("2021"))
    with pytest.raises(KeyError):
        psut.at("1999")
    with pytest.raises(KeyError):
        psut.U("1999")


# ---------------------------------------------------------------------------
# 6) Persistence: registry round-trip, one parquet per period
# ---------------------------------------------------------------------------

def test_registry_round_trip(tmp_path, psut):
    from pysuto.registry import PsutRunRegistry
    reg = PsutRunRegistry(str(tmp_path / "psut"))
    run_id = reg.register(
        psut,
        parents={"inputs": {"acc": {"id": "toy"}},
                 "upstream": {"opt_run_id": "opt_x", "const_run_id": "const_x"}},
        label="toy",
    )
    assert run_id.startswith("psut_")
    run_dir = tmp_path / "psut" / run_id
    assert (run_dir / "timesteps" / "t_2020.parquet").exists()
    assert (run_dir / "timesteps" / "t_2021.parquet").exists()
    assert (run_dir / "nodes.parquet").exists()

    sidecar = reg.load_sidecar(run_id)
    assert sidecar["parents"]["upstream"]["opt_run_id"] == "opt_x"
    assert sidecar["stage_config"] == {"valuations": ["Mass", "CC"], "primary": "CC",
                                       "blocks": ["E", "O", "P", "S", "U", "Y"]}
    assert sidecar["stage_summary"]["timesteps"] == ["2020", "2021"]
    per_t = sidecar["stage_summary"]["per_timestep"]["2020"]
    assert per_t["n_rows"] == 2 * N_PER_SLICE
    assert per_t["val_by_block"]["CC"] == pytest.approx(CC_TOTALS_2020)
    assert per_t["val_by_block"]["Mass"]["S"] == pytest.approx(3 * CC_TOTALS_2020["S"])

    loaded = PsutRunRegistry(str(tmp_path / "psut")).load(run_id)
    pd.testing.assert_frame_equal(loaded.table, psut.table)
    pd.testing.assert_frame_equal(loaded.nodes, psut.nodes)
    assert (loaded.valuations, loaded.primary) == (["Mass", "CC"], "CC")
    assert (loaded.acc_id, loaded.opt_run_id, loaded.const_run_id) == ("toy", "opt_x", "const_x")
    pd.testing.assert_frame_equal(loaded.O("2021", "Mass"), psut.O("2021", "Mass"))
    pd.testing.assert_frame_equal(loaded.realized_conc(), psut.realized_conc())

    with pytest.raises(KeyError):
        reg.load("psut_nonexistent")


def test_registry_loads_a_slice1_sidecar(tmp_path, psut):
    """Runs persisted by Slice 1 recorded one ``valuation``; they still load."""
    from pysuto.registry import PsutRunRegistry
    reg = PsutRunRegistry(str(tmp_path / "psut"))
    run_id = reg.register(psut, parents={"inputs": {}, "upstream": {}})
    path = tmp_path / "psut" / run_id / "metadata.json"
    sidecar = json.loads(path.read_text())
    sidecar["stage_config"] = {"valuation": "CC", "blocks": ["S", "U"]}
    path.write_text(json.dumps(sidecar))
    loaded = reg.load(run_id)
    assert (loaded.valuations, loaded.primary) == (["CC"], "CC")


# ---------------------------------------------------------------------------
# 7) from_opt_run: provenance chain opt → const → acc
# ---------------------------------------------------------------------------

def _persist_opt_chain(tmp_path, toy_acc, F):
    """A registered OptimRun whose const sidecar names acc 'toy'."""
    from pysuto.registry import ConstraintRunRegistry, OptimRunRegistry
    from pysuto.registry._sidecar import build_sidecar

    const_dir = str(tmp_path / "proc" / "const")
    opt_dir   = str(tmp_path / "proc" / "opt")

    creg = ConstraintRunRegistry(const_dir)
    run_dir = creg._make_run_dir("const_toy")
    creg._write_sidecar(run_dir, build_sidecar(
        run_id="const_toy", stage="const", label="", created_at="2026-01-01T00:00:00Z",
        parents={"inputs": {"acc": {"id": "toy", "version": 1}}, "upstream": {}},
    ))
    creg._add_entry(run_id="const_toy", label="", created_at="2026-01-01T00:00:00Z")
    creg.save()

    opt_id = OptimRunRegistry(opt_dir).register(
        _optim_results(toy_acc, F),
        parents={"inputs": {}, "upstream": {"const_run_id": "const_toy"}},
    )
    return opt_id, opt_dir, const_dir


@pytest.fixture
def patched_acc_load(monkeypatch, toy_acc):
    """`AccountingStructure.load` answers 'toy' with the in-memory toy structure."""
    from pysuto.acc import AccountingStructure
    calls = []

    def fake_load(cls, id, **kwargs):
        calls.append((id, kwargs))
        assert id == "toy"
        return toy_acc

    monkeypatch.setattr(AccountingStructure, "load", classmethod(fake_load))
    return calls


def test_from_opt_run_resolves_acc_through_provenance(tmp_path, toy_acc, F, psut, patched_acc_load):
    from pysuto.psut import PSUT
    opt_id, opt_dir, const_dir = _persist_opt_chain(tmp_path, toy_acc, F)
    acc_dir = str(tmp_path / "proc" / "acc")

    got = PSUT.from_opt_run(opt_id, opt_dir=opt_dir, const_dir=const_dir, acc_dir=acc_dir)

    assert patched_acc_load == [("toy", {"storage_dir": acc_dir})]
    pd.testing.assert_frame_equal(got.table, psut.table)
    assert (got.acc_id, got.opt_run_id, got.const_run_id) == ("toy", opt_id, "const_toy")
    assert (got.valuations, got.primary) == (["Mass", "CC"], "CC")

    subset = PSUT.from_opt_run(opt_id, timesteps=[2021], primary="Mass", opt_dir=opt_dir,
                               const_dir=const_dir, acc_dir=acc_dir)
    assert subset.periods() == ["2021"] and subset.primary == "Mass"
    with pytest.raises(KeyError, match="1999"):
        PSUT.from_opt_run(opt_id, timesteps=["1999"], opt_dir=opt_dir,
                          const_dir=const_dir, acc_dir=acc_dir)


def test_from_opt_run_without_const_parent_raises(tmp_path, toy_acc, F):
    from pysuto.psut import PSUT
    from pysuto.registry import OptimRunRegistry
    opt_dir = str(tmp_path / "opt")
    opt_id = OptimRunRegistry(opt_dir).register(
        _optim_results(toy_acc, F), parents={"inputs": {}, "upstream": {}})
    with pytest.raises(KeyError, match="const_run_id"):
        PSUT.from_opt_run(opt_id, opt_dir=opt_dir, const_dir=str(tmp_path / "const"),
                          acc_dir=str(tmp_path / "acc"))


# ---------------------------------------------------------------------------
# 8) CLI: scripts/psut.py <opt_run_id> → one parquet per period + provenance
# ---------------------------------------------------------------------------

def test_cli_writes_one_parquet_per_period(tmp_path, monkeypatch, toy_acc, F, patched_acc_load):
    from pysuto.registry import PsutRunRegistry
    opt_id, _, _ = _persist_opt_chain(tmp_path, toy_acc, F)
    monkeypatch.chdir(tmp_path)                       # logs/ lands in the tmp dir

    mod_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "psut.py")
    spec = importlib.util.spec_from_file_location("psut_cli", mod_path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    run_id = cli.main(opt_id, label="cli", primary="CC", data_dir=str(tmp_path))

    run_dir = tmp_path / "proc" / "psut" / run_id
    for t in PERIODS:
        assert (run_dir / "timesteps" / f"t_{t}.parquet").exists()
    sidecar = PsutRunRegistry(str(tmp_path / "proc" / "psut")).load_sidecar(run_id)
    assert sidecar["parents"]["upstream"] == {"opt_run_id": opt_id, "const_run_id": "const_toy"}
    assert sidecar["parents"]["inputs"]["acc"]["id"] == "toy"
    assert sidecar["label"] == "cli"
    assert sidecar["stage_config"]["primary"] == "CC"
