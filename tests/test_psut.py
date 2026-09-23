"""
test_psut.py
============
Stage 6 · Slice 1 — the tidy-canonical PSUT (issue #18, ADR 0037).

Fixture topology (psut_toy_acc)
-------------------------------
Regions A and B carry a two-process chain; region C has nodes but no edges
(labels that must still appear in every block view).

  Nature/SB/r  →  Mining/Process/r      (E — boundary, outside Slice 1)
  Mining/Process/r  →  Ore/Flow/r       (S)
  Ore/Flow/r  →  Smelting/Process/r     (U, domestic)
  Ore/Flow/B  →  Smelting/Process/A     (U, cross-region)
  Smelting/Process/r  →  Cathode/Flow/r (S)
  Smelting/Process/r  →  Slag/SB/r      (P — boundary, outside Slice 1)
  Cathode/Flow/r  →  Use/SB/r           (Y — boundary, outside Slice 1)

13 edges per (period, valuation) × periods 2020, 2021 × valuations Mass, CC
= 52 edges. CC values (2020; 2021 = ×1.1; Mass = ×3) are chosen so every
Process and Flow node balances exactly.
"""

from __future__ import annotations

import importlib.util
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
]

# (from_id, to_id, block, CC value in 2020)
EDGES = [
    (0, 1,  "E", 100.0),
    (1, 2,  "S", 100.0),
    (2, 3,  "U", 100.0),
    (9, 3,  "U",  20.0),      # cross-region use B → A
    (3, 4,  "S", 110.0),
    (3, 5,  "P",  10.0),
    (4, 6,  "Y", 110.0),
    (7, 8,  "E",  60.0),
    (8, 9,  "S",  60.0),
    (9, 10, "U",  40.0),
    (10, 11, "S", 36.0),
    (10, 12, "P",  4.0),
    (11, 13, "Y", 36.0),
]

# Boundary mass per node, CC 2020 — what Slice 1 leaves out of the table but
# the node balances need.
E_IN  = {("A", "Mining"): 100.0, ("B", "Mining"): 60.0}       # SB(Nature) → Process
P_OUT = {("A", "Smelting"): 10.0, ("B", "Smelting"): 4.0}    # Process → SB(SEM)
Y_OUT = {("A", "Cathode"): 110.0, ("B", "Cathode"): 36.0}    # Flow → SB(FD)


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
        base = EDGES[eid % len(EDGES)][3]
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
    return PSUT.from_results(_legacy_results(toy_acc, F), toy_acc, valuation="CC")


# ---------------------------------------------------------------------------
# 1) Block decomposition
# ---------------------------------------------------------------------------

def test_every_supply_use_edge_lands_in_exactly_one_block(toy_acc, psut):
    meta = toy_acc.edge_meta
    is_S = (meta["from_entity"] == "Process") & (meta["to_entity"] == "Flow")
    is_U = (meta["from_entity"] == "Flow") & (meta["to_entity"] == "Process")
    expected = set(meta.loc[(is_S | is_U) & (meta["valuation"] == "CC"), "edge_id"])

    assert set(psut.table["edge_id"]) == expected
    assert psut.table["edge_id"].is_unique                     # exactly once
    assert set(psut.table["block"]) == {"S", "U"}
    assert set(psut.table["valuation"]) == {"CC"}
    # 4 S + 3 U edges per period, over 2 periods
    assert psut.table.groupby(["period", "block"]).size().to_dict() == {
        ("2020", "S"): 4, ("2020", "U"): 3, ("2021", "S"): 4, ("2021", "U"): 3,
    }


def test_boundary_edges_never_enter_the_table(psut):
    assert not (psut.table["from_entity"] == "SB").any()
    assert not (psut.table["to_entity"] == "SB").any()


def test_unknown_valuation_raises(toy_acc, F):
    from pysuto.psut import PSUT
    with pytest.raises(ValueError, match="valuation 'Gross'"):
        PSUT.from_results(_legacy_results(toy_acc, F), toy_acc, valuation="Gross")


# ---------------------------------------------------------------------------
# 2) Block views: labels from the node universe, values from a*
# ---------------------------------------------------------------------------

def test_views_are_labelled_from_the_node_universe(psut):
    S = psut.S("2020")
    U = psut.U("2020")
    procs = pd.MultiIndex.from_tuples(
        [("A", "Mining"), ("A", "Smelting"), ("B", "Mining"), ("B", "Smelting"),
         ("C", "Mining"), ("C", "Smelting")], names=["region", "process"])
    flows = pd.MultiIndex.from_tuples(
        [("A", "Cathode"), ("A", "Ore"), ("B", "Cathode"), ("B", "Ore"),
         ("C", "Cathode"), ("C", "Ore")], names=["region", "flow"])
    assert S.index.equals(procs) and S.columns.equals(flows)
    assert U.index.equals(flows) and U.columns.equals(procs)
    assert S.index.names == ["region", "process"] and S.columns.names == ["region", "flow"]
    # region C has nodes but no edges: present, all zero
    assert (S.loc["C"] == 0).all().all() and (S["C"] == 0).all().all()
    # identical labels in every period
    assert psut.S("2021").index.equals(S.index) and psut.U("2021").columns.equals(U.columns)


def test_views_hold_the_solved_values(psut):
    S, U = psut.S("2020"), psut.U("2020")
    assert S.loc[("A", "Mining"), ("A", "Ore")] == 100.0
    assert S.loc[("A", "Smelting"), ("A", "Cathode")] == 110.0
    assert S.loc[("B", "Smelting"), ("B", "Cathode")] == 36.0
    assert U.loc[("B", "Ore"), ("A", "Smelting")] == 20.0        # cross-region use
    assert U.loc[("B", "Ore"), ("B", "Smelting")] == 40.0
    assert U.loc[("A", "Ore"), ("B", "Smelting")] == 0.0          # no such edge
    assert psut.S("2021").loc[("A", "Mining"), ("A", "Ore")] == pytest.approx(110.0)
    assert S.to_numpy().sum() == pytest.approx(306.0)             # 100+110+60+36
    assert U.to_numpy().sum() == pytest.approx(160.0)             # 100+20+40


def test_per_node_mass_balance_holds(psut):
    """Flow nodes: supply in = use out + Y. Process nodes: use in + E = supply
    out + P. The boundary terms are the toy's known values."""
    for period, k in [("2020", 1.0), ("2021", 1.1)]:
        S, U = psut.S(period), psut.U(period)
        supply_into_flow = S.sum(axis=0)          # indexed (region, flow)
        use_out_of_flow  = U.sum(axis=1)
        for flow in supply_into_flow.index:
            y = Y_OUT.get(flow, 0.0) * k
            assert supply_into_flow[flow] == pytest.approx(use_out_of_flow[flow] + y, abs=1e-9)
        use_into_process   = U.sum(axis=0)        # indexed (region, process)
        supply_out_process = S.sum(axis=1)
        for proc in use_into_process.index:
            e = E_IN.get(proc, 0.0) * k
            p = P_OUT.get(proc, 0.0) * k
            assert use_into_process[proc] + e == pytest.approx(supply_out_process[proc] + p, abs=1e-9)


# ---------------------------------------------------------------------------
# 3) Year-local cols_t → global edge ids
# ---------------------------------------------------------------------------

def test_year_local_cols_t_maps_to_global_edges(toy_acc, F, psut):
    from pysuto.psut import PSUT
    block = PSUT.from_results(_block_results(toy_acc, F), toy_acc, valuation="CC")
    pd.testing.assert_frame_equal(block.table, psut.table)
    pd.testing.assert_frame_equal(block.S("2021"), psut.S("2021"))


def test_cols_t_not_covering_the_period_raises(toy_acc, F):
    from pysuto.psut import PSUT
    res = _block_results(toy_acc, F)
    res["2021"] = SimpleNamespace(timestep="2021", F_opt=res["2020"].F_opt, cols_t=res["2020"].cols_t)
    with pytest.raises(ValueError, match="not covered by cols_t"):
        PSUT.from_results(res, toy_acc, valuation="CC")


def test_legacy_vector_of_wrong_length_raises(toy_acc, F):
    from pysuto.psut import PSUT
    res = {"2020": SimpleNamespace(timestep="2020", F_opt=F[:10], cols_t=None)}
    with pytest.raises(ValueError, match="cols_t is None"):
        PSUT.from_results(res, toy_acc, valuation="CC")


def test_results_key_must_match_timestep(toy_acc, F):
    from pysuto.psut import PSUT
    res = {"2021": SimpleNamespace(timestep="2020", F_opt=F.copy(), cols_t=None)}
    with pytest.raises(ValueError, match="holds a result for timestep"):
        PSUT.from_results(res, toy_acc, valuation="CC")


# ---------------------------------------------------------------------------
# 4) Time is carried, never aggregated
# ---------------------------------------------------------------------------

def test_periods_latest_at(psut):
    assert psut.periods() == ["2020", "2021"]
    assert psut.latest() == "2021"
    one = psut.at(2020)                      # int coerced to the label
    assert one.periods() == ["2020"]
    assert one.valuation == "CC" and one.acc_id == "toy"
    pd.testing.assert_frame_equal(one.S(), psut.S("2020"))     # default = latest of the slice
    pd.testing.assert_frame_equal(psut.S(), psut.S("2021"))
    with pytest.raises(KeyError):
        psut.at("1999")
    with pytest.raises(KeyError):
        psut.U("1999")


# ---------------------------------------------------------------------------
# 5) Persistence: registry round-trip, one parquet per period
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
    assert sidecar["stage_config"] == {"valuation": "CC", "blocks": ["S", "U"]}
    assert sidecar["stage_summary"]["timesteps"] == ["2020", "2021"]
    assert sidecar["stage_summary"]["per_timestep"]["2020"]["val_by_block"] == {"S": 306.0, "U": 160.0}

    loaded = PsutRunRegistry(str(tmp_path / "psut")).load(run_id)
    pd.testing.assert_frame_equal(loaded.table, psut.table)
    pd.testing.assert_frame_equal(loaded.nodes, psut.nodes)
    assert loaded.valuation == "CC"
    assert (loaded.acc_id, loaded.opt_run_id, loaded.const_run_id) == ("toy", "opt_x", "const_x")
    pd.testing.assert_frame_equal(loaded.U("2021"), psut.U("2021"))

    with pytest.raises(KeyError):
        reg.load("psut_nonexistent")


# ---------------------------------------------------------------------------
# 6) from_opt_run: provenance chain opt → const → acc
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

    subset = PSUT.from_opt_run(opt_id, timesteps=[2021], opt_dir=opt_dir,
                               const_dir=const_dir, acc_dir=acc_dir)
    assert subset.periods() == ["2021"]
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
# 7) CLI: scripts/psut.py <opt_run_id> → one parquet per period + provenance
# ---------------------------------------------------------------------------

def test_cli_writes_one_parquet_per_period(tmp_path, monkeypatch, toy_acc, F, patched_acc_load):
    from pysuto.registry import PsutRunRegistry
    opt_id, _, _ = _persist_opt_chain(tmp_path, toy_acc, F)
    monkeypatch.chdir(tmp_path)                       # logs/ lands in the tmp dir

    mod_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "psut.py")
    spec = importlib.util.spec_from_file_location("psut_cli", mod_path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    run_id = cli.main(opt_id, label="cli", data_dir=str(tmp_path))

    run_dir = tmp_path / "proc" / "psut" / run_id
    for t in PERIODS:
        assert (run_dir / "timesteps" / f"t_{t}.parquet").exists()
    sidecar = PsutRunRegistry(str(tmp_path / "proc" / "psut")).load_sidecar(run_id)
    assert sidecar["parents"]["upstream"] == {"opt_run_id": opt_id, "const_run_id": "const_toy"}
    assert sidecar["parents"]["inputs"]["acc"]["id"] == "toy"
    assert sidecar["label"] == "cli"
