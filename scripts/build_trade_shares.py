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
``F_root_CC(domestic) = usage − imports`` (usage conserved).

Groups with **no** recorded imports (ADR 0018): if BACI observes the consumer
region at all, ``Σ imports = 0`` is a real observation of domestic sourcing and
the residual ``usage − 0 = usage`` yields an all-domestic fan. If BACI is blind
to the region (Taiwan was, until #77 mapped BACI 490 "Other Asia, nes" to it),
no rows are emitted and the fan stays undistributed per ADR 0008. Before ADR
0018 every import-free group was unemitted, which silently discarded 469 ICSG
fans (~45 Mt CC) — see REP roundD-fan-recovery.

Domestic leg bounded by production (ADR 0024). The residual "needs no ICSG
production data" — and that is the defect: where BACI imports in CC fall short
of what the consumer's process used, the shortfall is booked as *domestic*
supply the region never produced (India: 100–456 kt/yr of concentrate against
23–48 kt mined). For sectors in ``BOUND_DOMESTIC_SECTORS`` the domestic raw
weight is therefore capped at the consumer region's own ICSG production of
the product, and the remainder of the usage is spread over the import legs in
BACI proportion (usage still conserved):

    dom_raw   = min(max(0, usage − Σ imp_cc), production_CC(Y, t))
    imp_raw_X = imp_cc_X · (usage − dom_raw) / Σ imp_cc

Bounded only where the recorded imports can carry the remainder,
``Σ imp_cc ≥ residual − production`` (the import legs are at most doubled).
An import-free fan has no leg to carry it and keeps the ADR 0018
all-domestic form; a near-self-sufficient smelter (United States 1997–2003:
usage 996, mine 886, imports 26 kt CC) has legs that would need to grow
fourfold — that gap is ICSG-internal, not BACI's, and spreading it would
fabricate imports instead of domestic supply (measured: it cut US mine
output 10–30 pp in those years through the interpolated neighbours).

``ICSG_usage_CC`` is read from the registered ICSG input's raw ``Use`` rows
(``icsg_raw_usage``), never from an ICSG feed run: that run is built from these
very weights, so reading its ``F_root`` handed the builder the *previous* ICSG
data and the documented order (trade shares → ICSG feed) was exact only on a
second pass (#66). The weights therefore depend on registered inputs alone —
the BACI feed run, built first, and the ICSG input.

SCOPE — traded refined/intermediate products only:
    Cathode, Concentrate, Anode, Matte.
Scrap is an ICSG Use type but is NOT traded → deliberately excluded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysuto import acc_store
from copper_feed import (
    BACI_ID, BOUND_DOMESTIC_SECTORS, CATHODE_CARRIER_SECTOR_TO, SIGNED_SECTORS_TO,
    TRADED_SECTORS,
    WITHHOLD_CHAIN_INCONSISTENT,
    chain_inconsistent_refining, icsg_raw_production, icsg_raw_usage, latest_feed_run,
    trade_share_parents)
from pysuto.config import load_config
from pysuto.feedprep.trade_shares import (
    baci_observed_regions, build_edges_frame, compute_trade_share_weights,
    imports_exhausted_groups, traded_conc_lookup)
from pysuto.input_registry import PARAMS_PARENTS, InputRegistries
from pysuto.registry import stage_lock

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------
# The labels (traded sectors, signed stock legs, the bound's sectors, ADR 0020's gate) are
# the copper case's, ``copper_feed``; the algorithm is ``pysuto.feedprep.
# trade_shares``. Sectors actually emitted this run:
SECTORS = list(TRADED_SECTORS)

_INPUT_ROOT = os.path.join("data", "input")
ACC_DIR     = os.path.join("data", "proc", "acc")
FEED_REG    = os.path.join("data", "proc", "feed", "registry")
OUT_DIR     = os.path.join(_INPUT_ROOT, "params", "trade_shares", "current")


def set_data_dir(data_dir: str) -> None:
    """Point the script at ``data_dir`` instead of ``data`` (the Stage 1 hook, #109)."""
    global _INPUT_ROOT, ACC_DIR, FEED_REG, OUT_DIR
    _INPUT_ROOT = os.path.join(data_dir, "input")
    ACC_DIR     = os.path.join(data_dir, "proc", "acc")
    FEED_REG    = os.path.join(data_dir, "proc", "feed", "registry")
    OUT_DIR     = os.path.join(_INPUT_ROOT, "params", "trade_shares", "current")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _latest_feed_froot(label: str):
    run = latest_feed_run(FEED_REG, label)
    print(f"   {label:10s} ← {os.path.basename(run)}")
    return np.load(os.path.join(run, "result.npz"))["F_root"]


def write_parents(out_dir: str, inputs: InputRegistries) -> dict:
    """Record the weights' parents (#53) beside their params.csv, where the params registry
    hashes it, so a rebuild from another run is a new version; return them."""
    built_from = trade_share_parents(inputs, FEED_REG)
    with open(os.path.join(out_dir, PARAMS_PARENTS), "w", encoding="utf-8") as f:
        json.dump(built_from, f, indent=2, sort_keys=True)
    return built_from


def check_baci_feed_current(inputs: InputRegistries, registry_dir: str) -> None:
    """Raise unless the newest BACI feed run was built from the BACI input in force.

    ``trade_share_parents`` names that run, but a BACI input registered after it ran would
    leave the weights consistent with a run that read the older matrix (#53's class).
    """
    run = latest_feed_run(registry_dir, BACI_ID)
    with open(os.path.join(run, "metadata.json"), encoding="utf-8") as f:
        built = json.load(f)["parents"]["inputs"]["feed"]["version"]
    now = inputs.feed.entry(BACI_ID)["version"]
    if built != now:
        raise ValueError(f"{os.path.basename(run)} was built from {BACI_ID} v{built}, but "
                         f"v{now} is in force — rerun baci_feed first")


def _input_registries():
    """Read-only: a lookup never versions a ``current/`` (#93)."""
    return InputRegistries.load(root=_INPUT_ROOT)


def _register_inputs():
    """Version every edited ``current/`` — ``main``'s call, as a pipeline step's."""
    return InputRegistries.initialize(root=_INPUT_ROOT)


def main() -> None:
    print("=" * 60)
    print("Building BACI trade-share weights (ADR 0002 option D)")
    print(f"   sectors: {SECTORS}  (Scrap excluded — not traded)")
    _register_inputs()
    inputs = _input_registries()
    check_baci_feed_current(inputs, FEED_REG)

    acc_id = load_config().pipeline.acc_id
    acc = acc_store.load(id=acc_id, storage_dir=ACC_DIR)
    conc_lookup = traded_conc_lookup(inputs.params.resolve("conc") / "params.csv",
                                     acc.valuation_roles.gross, TRADED_SECTORS)
    _wild = {s_: v for (s_, r_), v in conc_lookup.items() if r_ == "*"}
    _reg = len(conc_lookup) - len(_wild)
    print(f"   conc: {_wild}" + (f"  (+{_reg} region-resolved rows)" if _reg else ""))
    baci = _latest_feed_froot("baci")

    edges = build_edges_frame(acc, baci, SECTORS, SIGNED_SECTORS_TO,
                              CATHODE_CARRIER_SECTOR_TO)
    observed = baci_observed_regions(acc, baci)
    usage_raw = icsg_raw_usage(inputs, SECTORS)
    # ADR 0020's withhold, disabled by ADR 0021 — see WITHHOLD_CHAIN_INCONSISTENT.
    inconsistent = chain_inconsistent_refining(inputs)
    withhold = ({("Anode", "ER", r, t) for r, t in inconsistent}
                if WITHHOLD_CHAIN_INCONSISTENT else set())
    print(f"   chain-inconsistent refining: {len(inconsistent)} region-years"
          f"   withheld: {len(withhold)} (ADR 0021 disables the ADR-0020 gate)")
    # ADR 0027: a hub whose imports the netting exhausted has a *constructed*
    # zero import sum; it must not be read as observed domestic sourcing.
    exhausted = imports_exhausted_groups(
        edges, inputs.feed.resolve("baci") / "reexport_ledger.csv")
    withhold = withhold | exhausted
    print(f"   imports exhausted by the ADR 0027 netting: {len(exhausted)} fan "
          f"groups withheld (left undistributed, ADR 0008)")
    all_regions = set(acc.node_index["region"])
    blind = sorted(all_regions - observed)
    print(f"   BACI-observed regions: {len(observed)}/{len(all_regions)}"
          f"   trade-blind (fans stay undistributed): {blind or 'none'}")
    production = icsg_raw_production(inputs, BOUND_DOMESTIC_SECTORS, edges)
    print(f"   production rows (raw ICSG, ADR 0027 definitions): {len(production):,} "
          f"for {BOUND_DOMESTIC_SECTORS}")
    kw = dict(signed_sectors_to=SIGNED_SECTORS_TO)
    shares = compute_trade_share_weights(edges, usage_raw, conc_lookup, observed,
                                         withhold, production, **kw)

    # ADR 0024 diagnostic: how many groups the production bound touched and
    # how much usage moved from the domestic leg onto the import legs.
    unbounded = compute_trade_share_weights(edges, usage_raw, conc_lookup, observed,
                                            withhold, None, **kw)
    dcols = ["root_sector_from", "root_sector_to", "root_region_from", "root_region_to", "root_time"]
    both = (unbounded.merge(shares, on=dcols, suffixes=("_free", "_bound")))
    dom_both = both[both["root_region_from"] == both["root_region_to"]]
    moved = (dom_both["weight_free"] - dom_both["weight_bound"])
    print(f"   ADR 0024 production bound ({BOUND_DOMESTIC_SECTORS}): "
          f"{int((moved > 1e-9).sum()):,} domestic legs bounded, "
          f"{moved.sum():,.0f} kt CC moved from domestic residual onto import legs")

    # ADR 0018 diagnostic: which groups are emitted only because of the
    # import-free domestic fallback, and how much ICSG tonnage they carry.
    pre = compute_trade_share_weights(edges, usage_raw, conc_lookup, None, **kw)
    gcols = ["root_sector_from", "root_sector_to", "root_region_to", "root_time"]
    new_grp = (shares[gcols].drop_duplicates()
               .merge(pre[gcols].drop_duplicates(), how="left", indicator=True))
    n_new = int((new_grp["_merge"] == "left_only").sum())
    dom = shares[shares["root_region_from"] == shares["root_region_to"]]
    recovered = dom.merge(new_grp[new_grp["_merge"] == "left_only"][gcols],
                          on=gcols)["weight"].sum()
    print(f"   ADR 0018 domestic fallback: {n_new:,} groups recovered, "
          f"{recovered:,.0f} kt CC of ICSG usage that previously went "
          f"undistributed")

    os.makedirs(OUT_DIR, exist_ok=True)
    # Named params.csv because the params input-registry only hashes that file
    # for content-versioning (input_registry _LOAD_BEARING["params"]).
    out_csv = os.path.join(OUT_DIR, "params.csv")
    shares.to_csv(out_csv, index=False)
    print(f"   wrote {len(shares):,} weight rows → {out_csv}")
    built_from = write_parents(OUT_DIR, _input_registries())
    print(f"   built from : {built_from['upstream']['feed_run_ids']}, {built_from['inputs']}")
    print(f"   consumer groups: "
          f"{shares.groupby(['root_sector_from','root_sector_to','root_region_to','root_time']).ngroups:,}")

    # Snapshot/version the artifact via the params registry (provenance).
    entry = _register_inputs().params.entry("trade_shares")
    print(f"   registered trade_shares v{entry['version']} ({entry['code']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data-dir", default="data",
                    help="data root holding input/ and proc/ (default: %(default)s)")
    args = ap.parse_args()
    set_data_dir(args.data_dir)
    with stage_lock("build_trade_shares", root=os.path.join(args.data_dir, "proc")):
        main()
