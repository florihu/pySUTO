"""
test_toy_coproduct.py
=====================
The toy case with a co-product (#130): ``python -m pysuto run --case toy_coproduct --from
feed --to psut --data-dir <tmp>`` runs the six stages of the toy world whose smelter has
two Flow products, ``Metal`` and ``Dust``. One module-scoped chain, run as a subprocess
from an empty working directory, is read back through the registries: every hard row
closes to 1e-10, Stage 3's agg eff row pools both products, and the imputation log shows
the co-product rule (one unknown product filled as the sum's remainder, two unknown left
to the later fillers).
"""
from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysuto import acc_store
from pysuto.case import resolve_case
from pysuto.tutorial import toy, toy_coproduct
from pysuto.registry import ConstraintRunRegistry, OptimRunRegistry

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    root, cwd = tmp_path_factory.mktemp("co_data"), tmp_path_factory.mktemp("co_cwd")
    proc = subprocess.run(
        [sys.executable, "-m", "pysuto", "run", "--case", "toy_coproduct", "--from",
         "feed", "--to", "psut", "--data-dir", str(root)],
        cwd=str(cwd), capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert proc.returncode == 0, f"{proc.stdout[-1500:]}\n{proc.stderr[-3000:]}"
    printed = {line.split()[0]: line.split()[1:] for line in proc.stdout.splitlines()}
    ids = {s: printed[s][0] for s in ("imp", "const", "opt", "uncprop", "psut")}
    acc = acc_store.load(toy.ACC_ID_COPRODUCT,
                                   storage_dir=str(root / "proc" / "acc"))
    return {"root": root, "ids": ids, "acc": acc}


def test_the_case_is_the_toy_under_its_own_id():
    case = resolve_case("toy_coproduct")
    assert case == dataclasses.replace(toy_coproduct.CASE, case_source="alias:toy_coproduct",
                                       case_sha256=case.case_sha256)
    assert (case.id, case.acc_id) == ("toy_coproduct", toy.ACC_ID_COPRODUCT)
    assert case.imp is toy.CASE.imp and case.const is toy.CASE.const


def test_the_truth_balances_every_node_with_dust_and_leaves_the_toy_alone():
    frame = toy.truth(coproduct=True)
    assert len(frame) == 480 and len(toy.truth()) == 420
    entity = {s["name"]: s["entity"] for s in toy.acc_spec(True)["sectors"]}
    for valuation, sub in frame.groupby("valuation"):
        inflow  = sub.groupby(["time", "to_region", "to_sector"])["value"].sum()
        outflow = sub.groupby(["time", "from_region", "from_sector"])["value"].sum()
        both = pd.concat([inflow.rename_axis(["time", "region", "sector"]),
                          outflow.rename_axis(["time", "region", "sector"])], axis=1
                         ).fillna(0.0).reset_index()
        both = both[both["sector"].map(entity).isin(["Process", "Flow"]).to_numpy()
                    & ~both["sector"].eq("Mining").to_numpy()]
        assert np.allclose(both.iloc[:, 3], both.iloc[:, 4], rtol=1e-12), valuation
    pooled = frame[(frame["from_sector"] == "Smelting")
                   & frame["to_sector"].isin(["Metal", "Dust"])
                   & (frame["valuation"] == toy.CONTENT)]
    inputs = frame[(frame["to_sector"] == "Smelting")
                   & (frame["valuation"] == toy.CONTENT)]
    ratio = (pooled.groupby(["time", "from_region"])["value"].sum()
             / inputs.groupby(["time", "to_region"])["value"].sum().rename_axis(
                 ["time", "from_region"]))
    assert np.allclose(ratio, 0.97, rtol=1e-12)


def test_every_hard_row_closes_to_1e_10_over_six_stages(chain):
    root, ids = chain["root"], chain["ids"]
    const = ConstraintRunRegistry(str(root / "proc" / "const")).load(ids["const"])
    hard  = const.row_meta["is_hard"].to_numpy().astype(bool)
    assert hard.sum() > 0
    for t, res in OptimRunRegistry(str(root / "proc" / "opt")).load(ids["opt"]).items():
        assert res.solver_status == "solved"
        assert np.abs(res.residuals[hard[res.rows_t]]).max() <= 1e-10, t


def test_stage_3_agg_eff_row_pools_both_products(chain):
    root, ids, acc = chain["root"], chain["ids"], chain["acc"]
    const = ConstraintRunRegistry(str(root / "proc" / "const")).load(ids["const"])
    rows = const.row_meta.index[const.row_meta["constraint_id"].astype(str)
                                .str.startswith("ratio_effsmelting_Portua_Content_2003")]
    assert len(rows) == 1
    G = const.G.tocsr()[int(rows[0])]
    em = acc.edge_meta.set_index("edge_id")
    coeff = {(em.loc[j, "from_sector"], em.loc[j, "to_sector"]): v
             for j, v in zip(G.indices, G.data) if v != 0.0}
    assert coeff[("Smelting", "Metal")] == 1.0 and coeff[("Smelting", "Dust")] == 1.0
    assert coeff[("Scrap", "Smelting")] == pytest.approx(-0.97)
    assert ("Smelting", "Slag") not in coeff


def test_one_unknown_product_is_filled_by_the_sum_and_two_unknown_stay_open(chain):
    root, ids, acc = chain["root"], chain["ids"], chain["acc"]
    log = pd.read_parquet(root / "proc" / "imp" / ids["imp"] / "imputation_log.parquet")
    em = acc.edge_meta.set_index("edge_id")
    dust = em.index[(em["from_sector"] == "Smelting") & (em["to_sector"] == "Dust")
                    & (em["valuation"] == toy.CONTENT)]
    rows = log[log["edge_id"].isin(dust)]
    by_year = {(em.loc[e, "from_region"], em.loc[e, "time"]): m
               for e, m in zip(rows["edge_id"], rows["method"])}
    # published region-years: Dust is the remainder of the sum, filled by the network
    assert by_year[("Isla", "2002")] == "network_forward"
    # the Bureau's two gaps: Metal and Dust both unknown, the network writes neither
    for gap in (("Isla", "2003"), ("Montara", "2004")):
        assert by_year[gap] == "time_linear", gap
