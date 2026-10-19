"""
issue78_observed_zero_diag.py
=============================
#78 measurement of #76: a weighted ICSG fan's domestic leg at trade-share weight 0 is an
**observed zero** — covered at 0 with σ = max(1 % of its fans' usage, 0.3 kt) — so no
imputer refills it. Reads persisted run artefacts, both chains' feed runs (their
``observed_zero`` mask), the imputation logs, the constraint runs' σ_a; builds the sum-row
table in memory (no per-year CSVs, #46). Prints #78's pass-rule items and the numbers its
predictions are scored on, for <new> against <base>. Shared sections come from the #69,
#75, #79 and #92 diagnostics:

    O   observed zeros: count by sector pair; imputer writes on them in each chain
        (item 1); their posterior and z, largest |z| by region (item 3)
    F   ROW's domestic semis leg and its semis prior against the ICSG fan (item 2)
    G   Germany (#32): its mine-chain sum rows' z² by year; its posterior mine output; where
        its smelter balance goes (concentrate imports, matte, anode, scrap)
    K   the same class elsewhere: mine-chain z² of Korea, Spain, Slovakia, Botswana
    E   the Round-E items (ADR 0025); ROW's CP→Cathode reported, not judged (item 4)
    S   stock legs newly above |z| 5, their size, and the net stock posterior (item 3)
    C   common-set sum-row z² by region, sector and year; rows only one run has
    A   attribution of the largest moves

    python scripts/explo/issue78_observed_zero_diag.py <opt_base> <opt_new>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/explo")
import build_trade_shares as BTS                          # noqa: E402
import issue69_aggregate_gap_diag as P69                  # noqa: E402
import issue75_scandinavia_split_diag as P75              # noqa: E402
import issue79_baci490_taiwan_diag as P79                 # noqa: E402
import issue86_congo_sacu_diag as P86                     # noqa: E402
import issue92_refined_scope_diag as P92                  # noqa: E402
import roundL_reexport_diag as L                          # noqa: E402
from pysuto.acc import AccountingStructure                # noqa: E402
from pysuto.config import load_config                     # noqa: E402

ROOT     = Path("data/proc")
GERMANY  = "Germany"
SAME     = ("Korea, Rep.", "Spain", "Slovakia", "Botswana")
MINE     = ("Ore", "MF", "Concentrate", "Mining")


def _observed_zero(opt: str) -> np.ndarray:
    """The ICSG feed run's observed-zero mask (all False on a feed built before #76)."""
    feed = L._chain(opt)[2]["icsg"]
    r = np.load(ROOT / "feed" / "registry" / feed / "result.npz")
    return r["observed_zero"] if "observed_zero" in r.files else np.zeros(
        len(r["F_root"]), dtype=bool)


# ---------------------------------------------------------------------------
# O — observed zeros (items 1, 3)
# ---------------------------------------------------------------------------

def observed(E: dict, acc, base: str, new: str) -> None:
    """O: the observed-zero legs, the imputer's writes on them, their posterior and z."""
    oz = _observed_zero(new)
    ei = BTS._decoded_edges(acc)
    legs = ei[oz[ei.edge_id.to_numpy()]]
    print(f"\n== O observed zeros on {L._chain(new)[2]['icsg']}: {len(legs)} legs; by sector "
          f"pair {legs.groupby(['root_sector_from', 'root_sector_to']).size().to_dict()}")
    ids = legs.edge_id.to_numpy()
    for o in (base, new):
        imp = L._chain(o)[1]
        log = pd.read_parquet(ROOT / "imp" / imp / "imputation_log.parquet")
        w = log[log.edge_id.isin(ids)]
        e = E[o].set_index("edge_id")
        kt = e.reindex(w.edge_id.unique()).prior.abs().sum()
        print(f"   {o} ({imp}): imputer writes on these legs {len(w)} "
              f"({w.method.value_counts().to_dict()}), their prior Σ {kt:,.0f} kt CC (item 1)")
    z = P69._with_sigma(E[new], new).set_index("edge_id").reindex(ids)
    z = z.join(legs.set_index("edge_id")[["root_region_to", "root_sector_to"]])
    print(f"   posterior on the observed zeros: Σ {z.post.sum():,.1f} kt CC, Σ|post| "
          f"{z.post.abs().sum():,.1f}; |z| > 3: {int((z.z.abs() > 3).sum())}, > 5: "
          f"{int((z.z.abs() > 5).sum())}")
    top = z.reindex(z.z.abs().sort_values(ascending=False).index).head(8)
    print("   largest |z|: " + "  ".join(
        f"{r.root_region_to} {r.root_sector_to} {r.time} post {r.post:.1f} z {r.z:+.1f}"
        for _, r in top.iterrows()))
    by = z.groupby("root_region_to").post.sum().sort_values(ascending=False).head(8)
    print("   posterior by region (largest): " + "  ".join(f"{k} {v:,.1f}" for k, v in by.items()))


# ---------------------------------------------------------------------------
# G — Germany (#32), K — the same class elsewhere
# ---------------------------------------------------------------------------

def _mine_rows(j: pd.DataFrame, region: str) -> pd.DataFrame:
    s = P92._sector(j.label)
    return j[(j.region == region) & s.isin(MINE)]


def germany(E: dict, j: pd.DataFrame, base: str, new: str) -> None:
    """G: Germany's mine-chain z², its posterior mine output, and its smelter balance."""
    m = _mine_rows(j, GERMANY)
    by = m.groupby("year")[["z2_base", "z2_new"]].sum()
    by = by[(by.z2_base + by.z2_new) > 100]
    print(f"\n== G {GERMANY} (#32): mine-chain sum rows Σz² {m.z2_base.sum():,.0f} → "
          f"{m.z2_new.sum():,.0f}; by year (base/new) "
          + " ".join(f"{t[-2:]}:{a:,.0f}/{b:,.0f}" for t, (a, b) in by.round(0).iterrows()))
    g = j[j.region == GERMANY]
    print(f"   {GERMANY} common-set Δz² {g.dz2.sum():+,.0f}; by sector "
          + "  ".join(f"{k} {v:+,.0f}" for k, v in
                      g.groupby(P92._sector(g.label)).dz2.sum().sort_values().items()))
    for o in (base, new):
        e = E[o][E[o].valuation == "CC"]
        mine = e[(e.sector_from == "CO") & (e.sector_to == "Mining") & (e.region_to == GERMANY)]
        sx = e[(e.sector_from == "Ore") & (e.sector_to == "SXEW") & (e.region_to == GERMANY)]
        c_in = e[(e.sector_from == "Concentrate") & (e.sector_to == "Smelting")
                 & (e.region_to == GERMANY)]
        dom = c_in.region_from == GERMANY
        sc = e[(e.sector_from == "Scrap") & (e.sector_to == "Smelting") & (e.region_to == GERMANY)]
        an = e[(e.sector_from == "Anode") & (e.region_from == GERMANY) & (e.region_to != GERMANY)]
        print(f"   {o}: CO→Mining post Σ {mine.post.sum():,.1f} (prior {mine.prior.sum():,.1f}); "
              f"Ore→SXEW post {sx.post.sum():,.1f}; concentrate into smelting domestic "
              f"{c_in[dom].prior.sum():,.0f}/{c_in[dom].post.sum():,.0f}, imported "
              f"{c_in[~dom].prior.sum():,.0f}/{c_in[~dom].post.sum():,.0f}; smelter scrap "
              f"{sc.prior.sum():,.0f}/{sc.post.sum():,.0f}; anode exports "
              f"{an.prior.sum():,.0f}/{an.post.sum():,.0f} (Σ prior/post 1995–2024, kt CC)")
    print("\n== K the same class: mine-chain sum rows Σz², base → new")
    for reg in SAME:
        m = _mine_rows(j, reg)
        print(f"   {reg}: {m.z2_base.sum():,.0f} → {m.z2_new.sum():,.0f}")


def main(base: str, new: str) -> None:
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id,
                                   storage_dir=str(ROOT / "acc"))
    E, S, j = P69.load(base, new)
    observed(E, acc, base, new)
    P79.feeds(E, acc, base, new)
    germany(E, j, base, new)
    P79.rest_of_world(E, j, base, new)
    P69.round_e(E, S, j, base, new)
    P79.stock_z(E, base, new)
    P92.stock_sizes(E, base, new)
    P69.common_set(j, base)
    P86.by_year(j)
    by = j.groupby(P92._sector(j.label)).dz2.sum().sort_values()
    print("   common-set Δz² by sector: " + "  ".join(f"{k} {v:+,.0f}" for k, v in by.items()))
    P75.one_run_only(S, base, new)
    P69.attribution(j)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
