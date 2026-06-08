"""
build_trade_shares.py
=====================
Build BACI-trade-share weights for ICSG's single-endpoint producer fan-out
(ADR 0002 option D) and emit them as a ``params``-kind input artifact
``trade_shares`` that the ICSG feed build consumes as ``user_weights_df``.

Why
---
ICSG ``Use`` rows classify only the consumer (producer region open), so each
fans across producer regions. The pre-existing fix (ADR 0002) split that fan
*uniformly*; on a bilateral trade edge X→Y the uniform ICSG-CC value then
disagrees with the real BACI-Mass observation, breaking the ``conc_*_supply``
ratio. Option D reweights the fan by BACI trade shares so ICSG-CC inherits
BACI's bilateral geography.

The weighting (per sector_from, sector_to, consumer region Y, year t), in
copper-content (CC) units so the domestic/import split is unit-consistent:

    import leg  X≠Y : raw weight = conc_sector · BACI_Mass_Froot(X→Y, t)
    domestic    X==Y: raw weight = max(0, ICSG_usage_CC(Y, t)
                                          − Σ_X conc_sector · BACI_Mass(X→Y, t))

``ConcordanceTable._impute_weights`` then normalises each src fan's raw weights
to sum 1, so ``F_root_CC(import) = conc·BACI_Mass`` and
``F_root_CC(domestic) = usage − imports`` (usage conserved). Groups BACI does
not cover at all get no rows → uniform fallback (unchanged behaviour).

``ICSG_usage_CC`` is read from the *existing* ICSG feed ``F_root``: the fan total
Σ_X F_root(X→Y) equals the raw ICSG usage regardless of the weights used to
build it (mass-preserving), so it is the same number the raw carries.

SCOPE — traded refined/intermediate products only:
    Cathode, Concentrate, Anode, Matte.
Scrap is an ICSG Use type but is NOT traded → deliberately excluded.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from pysuto.acc import AccountingStructure
from pysuto.config import load_config

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------

# Traded supplier sectors whose ICSG producer fan is BACI-weighted. Scrap is
# NOT traded and must never appear here.
TRADED_SECTORS = ["Cathode", "Concentrate", "Anode", "Matte"]
# Sectors actually emitted this run (commit 2 ships Cathode; commit 4 widens).
SECTORS = ["Cathode"]

_INPUT_ROOT = os.path.join("data", "input")
ACC_DIR     = os.path.join("data", "proc", "acc")
FEED_REG    = os.path.join("data", "proc", "feed", "registry")
OUT_DIR     = os.path.join(_INPUT_ROOT, "params", "trade_shares", "current")

assert "Scrap" not in TRADED_SECTORS and "Scrap" not in SECTORS, "Scrap is not traded"


# ---------------------------------------------------------------------------
# Pure share computation (unit-tested)
# ---------------------------------------------------------------------------

def compute_trade_share_weights(
    edges: pd.DataFrame,
    conc_by_sector: dict[str, float],
) -> pd.DataFrame:
    """Compute raw fan weights from a decoded CC-edge frame.

    ``edges`` columns (one row per CC fan-target edge):
        root_sector_from, root_sector_to, root_region_from, root_region_to,
        root_time, icsg (ICSG F_root CC), baci_mass (BACI F_root Mass on the
        paired Mass edge; NaN/0 where no trade).

    Returns rows with columns
        root_sector_from, root_sector_to, root_region_from, root_region_to,
        root_time, weight
    only for consumer groups BACI covers (≥1 import). The raw weights are NOT
    normalised here — ConcordanceTable._impute_weights does that per src fan.
    """
    df = edges.copy()
    df["baci_mass"] = df["baci_mass"].fillna(0.0)
    df["is_domestic"] = df["root_region_from"] == df["root_region_to"]
    conc = df["root_sector_from"].map(conc_by_sector)
    df["imp_cc"] = np.where(df["is_domestic"], 0.0, conc * df["baci_mass"])

    grp_cols = ["root_sector_from", "root_sector_to", "root_region_to", "root_time"]
    g = df.groupby(grp_cols, sort=False)
    usage   = g["icsg"].transform("sum")
    imp_sum = g["imp_cc"].transform("sum")

    df["weight"] = np.where(
        df["is_domestic"],
        np.maximum(0.0, usage - imp_sum),   # domestic residual
        df["imp_cc"],                       # import leg (0 where untraded)
    )

    # Emit only groups BACI covers (some import) AND ICSG fans into (usage>0).
    covered = (imp_sum > 0) & (usage > 0)
    out = df.loc[covered, grp_cols[:2] + ["root_region_from", "root_region_to",
                                          "root_time", "weight"]]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _latest_feed_froot(label: str) -> np.ndarray:
    dirs = sorted(glob.glob(os.path.join(FEED_REG, f"feed_{label}_*")))
    if not dirs:
        raise FileNotFoundError(f"no feed_{label}_* run under {FEED_REG}")
    run = dirs[-1]
    print(f"   {label:10s} ← {os.path.basename(run)}")
    return np.load(os.path.join(run, "result.npz"))["F_root"]


def _conc_by_sector() -> dict[str, float]:
    inputs = _input_registries()
    csv = inputs.params.resolve("conc") / "params.csv"
    p = pd.read_csv(csv, sep=";", comment="#", dtype=str)
    p = p[(p["param_type"] == "concentration")
          & (p["param_id"].str.endswith("_supply"))]
    out = {row["a_sector_from"]: float(row["value_nominal"]) for _, row in p.iterrows()}
    return {s: out[s] for s in TRADED_SECTORS if s in out}


def _input_registries():
    from pysuto.input_registry import InputRegistries
    return InputRegistries.initialize(root=_INPUT_ROOT)


def _decoded_edges(acc: AccountingStructure) -> pd.DataFrame:
    ei = acc.edge_index.copy()
    nidx = acc.node_index.reset_index(drop=True)
    ei["root_sector_from"] = nidx["sector"].values[ei["from_id"].values]
    ei["root_sector_to"]   = nidx["sector"].values[ei["to_id"].values]
    ei["root_region_from"] = nidx["region"].values[ei["from_id"].values]
    ei["root_region_to"]   = nidx["region"].values[ei["to_id"].values]
    ei["root_time"]        = ei["time"]
    return ei


def build_edges_frame(acc, icsg_froot, baci_froot, sectors) -> pd.DataFrame:
    """One row per CC fan-target edge for `sectors`, with icsg + paired baci_mass."""
    ei = _decoded_edges(acc)
    ei = ei[ei["root_sector_from"].isin(sectors)]
    cc   = ei[ei["valuation"] == "CC"].copy()
    mass = ei[ei["valuation"] == "Mass"][["from_id", "to_id", "time", "edge_id"]]
    mass = mass.rename(columns={"edge_id": "mass_edge_id"})
    cc = cc.merge(mass, on=["from_id", "to_id", "time"], how="left")
    cc["icsg"]      = icsg_froot[cc["edge_id"].values]
    cc["baci_mass"] = np.where(cc["mass_edge_id"].notna(),
                               baci_froot[cc["mass_edge_id"].fillna(0).astype(int).values],
                               np.nan)
    return cc[["root_sector_from", "root_sector_to", "root_region_from",
               "root_region_to", "root_time", "icsg", "baci_mass"]]


def main() -> None:
    print("=" * 60)
    print("Building BACI trade-share weights (ADR 0002 option D)")
    print(f"   sectors: {SECTORS}  (Scrap excluded — not traded)")

    acc_id = load_config().pipeline.acc_id
    acc = AccountingStructure.load(id=acc_id, storage_dir=ACC_DIR)
    conc_by_sector = _conc_by_sector()
    print(f"   conc: {conc_by_sector}")
    icsg = _latest_feed_froot("icsg")
    baci = _latest_feed_froot("baci")

    edges = build_edges_frame(acc, icsg, baci, SECTORS)
    shares = compute_trade_share_weights(edges, conc_by_sector)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "shares.csv")
    shares.to_csv(out_csv, index=False)
    print(f"   wrote {len(shares):,} weight rows → {out_csv}")
    print(f"   consumer groups: "
          f"{shares.groupby(['root_sector_from','root_sector_to','root_region_to','root_time']).ngroups:,}")

    # Snapshot/version the artifact via the params registry (provenance).
    inputs = _input_registries()
    entry = inputs.params.entry("trade_shares")
    print(f"   registered trade_shares v{entry['version']} ({entry['code']})")


if __name__ == "__main__":
    main()
