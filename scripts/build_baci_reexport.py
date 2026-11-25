"""
build_baci_reexport.py
======================
Re-attribute **re-exported** bilateral trade to its origin and destination at
feed time (ADR 0027) and emit the result as a new registered version of the
``baci`` input feed, leaving the source version untouched.

Why
---
A BACI row is modelled as ``Flow(A) → Process(Z)`` — "A's product was consumed
by Z's process". BACI, however, records **gross border crossings**: a cathode
bundle that sails Chile → Rotterdam → Hamburg appears twice, ``Chile→NL`` and
``NL→Germany``. The second row asks the model to find Dutch cathode
*production*; there is none, so ``NetworkImputer._do_flow_balance`` writes a
phantom refinery (or the QP deletes the observed export), and the first row
asks Dutch semis (5 kt) to absorb 387 kt. The fan meanwhile discards the
excess import at feed time, so the material vanishes from the priors *and*
must be re-invented behind the export.

The rule — four observed numbers per (product s, region h, year t), all in
contained copper (CC = BACI mass × ``conc`` nominal; ICSG already reports CC):

    I = Σ_A BACI(A→h)                imports of s into h
    X = Σ_Z BACI(h→Z)                exports of s from h
    U = ICSG intake of s in h        (see `icsg_intake_production`)
    P = ICSG production of s in h

    R = max(0, I − U)                surplus imports:   imported, not used
    E = max(0, X − P)                uncovered exports: exported, not producible
    T = min(X, I, max(R, E))         the re-export volume

``T ≤ X`` and ``T ≤ I`` are what make the rule safe by construction: a real
producer with a thin ICSG row (Kazakhstan cathode: I = 0, X = 412, P = 299)
has ``T = 0`` however large E is — **producers are never netted**. ``R`` is the
use-side reading of the same physical fact and ``E`` the supply-side one; the
rule takes the larger and lets the caps keep it physical. ``E`` is the
production-first convention (domestic output supplies exports before imports
do) and catches the ``U > I`` hubs that ``R`` alone is blind to — Korea,
Germany, France: the phantom-mine cases.

Re-allocation, with origin shares ``a_A = BACI(A→h)/I`` and destination shares
``b_Z = BACI(h→Z)/X``:

    BACI'(A→h) = BACI(A→h) − T·a_A
    BACI'(h→Z) = BACI(h→Z) − T·b_Z
    BACI'(A→Z) = BACI(A→Z) + T·a_A·b_Z          (A ≠ Z)
    A = Z is a RE-IMPORT (or, for A = ROW, third-country trade folded onto one
    aggregate): dropped, and reported in the ledger.

Three exclusions, each measured before it was adopted (see the Round L REP):

* **ROW is never a hub.** It is 186 economies folded into one region: it has no
  warehouse, its I − U mixes 186 balances, and its largest inbound legs are a
  BACI units defect (Papua New Guinea concentrate booked to North Korea,
  7.8–20.8 Mt gross/yr in 2019–22 against PNG's real ~1.5 Mt). ROW remains an
  ordinary origin and destination for other hubs.
* **Screened origins are never re-attributed.** An origin whose sector-year
  exports exceed ``OUTLIER_K × (P + I)`` with ``P > 0`` — the Stage-2 outlier
  screen's own rule restated at Stage 1 — keeps its inbound legs gross, so the
  Philippines hub (PNG is 97–99 % of its imports in 2019/20/22) cannot launder
  that defect onto China, Japan and Korea.
* **ICSG gap-years are skipped.** Where a region reports the product in ICSG in
  other years but has neither U nor P in year t, ``T = 0`` and the ledger flags
  ``icsg_gap``. So does a region-year whose flows ICSG reports inside an
  aggregate row, whatever stock rows it carries itself (#67): U includes the
  MESC/USC rows, and Sweden's lone LME changes had it netted as a pass-through
  hub in every year 1996–2003 although its smelting, refining and semis sit
  under ICSG's "Scandinavia" (→ ROW). The membership is declared in the
  registered ``icsg_aggregates`` input, never inferred from region names, and
  the ledger's ``icsg_aggregate`` column names the aggregate. A region that
  *never* reports the product keeps U = P = 0 — that is the honest value, not a
  gap.

**Crumbs.** Proportional allocation over every origin × destination pair of a
hub produces tens of thousands of sub-kilotonne legs that would enter coverage
and drag ADR 0017's sector medians toward zero. A pair with no prior direct
trade is therefore created only at ``MIN_NEW_LEG_CC`` or above; an origin's
sub-threshold pairs are redistributed over its kept destinations, and an origin
left with none does not participate.

**Uncertainty.** A changed leg is partly an inference, so the feed carries
``σ = cv_pedigree · √(v_obs² + (k·Δ)²)`` with ``k = √2`` (the adjustment derives
from two observed legs), delivered through the existing per-edge
``cv_root_scale`` seam that ADR 0003's ROW penalty already uses. The scale is
floored at 1: netting must never make a leg *more* certain than its pedigree.

Outputs (all into ``data/input/feed/baci/current/``; the input registry bumps
the version and snapshots the previous one):

    values.csv              the netted matrix, same six columns
    reexport_ledger.csv     per (sector, region, year): I X U P R E T … label
    reexport_legs.csv       per changed root leg: v_obs Δ v_new
    reexport_cv_scale.csv   per changed root leg: cv_root_scale (≥ 1)

SCOPE — traded products only: Concentrate, Matte, Anode, Cathode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from copper_feed import (
    AGGREGATES_ID, BACI_ID, GROSS_VALUATION, HS_OF, HUB_EXCLUDE, ICSG_ID, NETTED_SECTORS,
    TRADED_SECTORS, icsg_intake_production)
from feedprep.reexport import (
    MAX_CV_SCALE, OUTLIER_K, check_icsg_baci_version, check_region_map, cv_scale,
    icsg_aggregate_members, label_row, load_root_trade, net_reexports, rebuild_values,
    reexport_volumes, screened_origins)
from feedprep.trade_shares import traded_conc_lookup
from pysuto.input_registry import InputRegistries
from pysuto.registry import stage_lock

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------

# The registered BACI input version the netting reads is ``--source-version``, required:
# the gross version the ICSG input was parsed against. Never the live `current/` copy:
# netting a netted file would be a no-op (the rule is idempotent) but would hide which
# matrix a run was built from.

# The sectors, the HS codes of a new row, the hub exclusion and the knobs of the netting
# are the copper case's (``copper_feed``) and ``feedprep.reexport``'s.
SECTORS = NETTED_SECTORS

_INPUT_ROOT   = os.path.join("data", "input")
FEED_ID       = BACI_ID
OUT_DIR       = os.path.join(_INPUT_ROOT, "feed", FEED_ID, "current")


def set_data_dir(data_dir: str) -> None:
    """Point the script at ``data_dir`` instead of ``data`` (the Stage 1 hook, #109)."""
    global _INPUT_ROOT, OUT_DIR
    _INPUT_ROOT = os.path.join(data_dir, "input")
    OUT_DIR     = os.path.join(_INPUT_ROOT, "feed", FEED_ID, "current")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _input_registries():
    """Read-only: a lookup never versions a ``current/`` (#93)."""
    return InputRegistries.load(root=_INPUT_ROOT)


def _register_inputs():
    """Version every edited ``current/`` — ``main``'s call, as a pipeline step's."""
    return InputRegistries.initialize(root=_INPUT_ROOT)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_dir(version: int) -> str:
    """The immutable snapshot the netting reads, hash-checked against the registry."""
    src = os.path.join(_INPUT_ROOT, "feed", FEED_ID, f"v{version}")
    with open(os.path.join(_INPUT_ROOT, "feed", "registry.json"), encoding="utf-8") as f:
        reg = json.load(f)
    entry = next((v for v in reg[FEED_ID]["versions"] if v["version"] == version), None)
    if entry is None:
        raise ValueError(f"baci v{version} is not in the input registry")
    got = _sha256(os.path.join(src, "values.csv"))
    if got != entry["hashes"]["values.csv"]:
        raise ValueError(
            f"baci v{version}/values.csv sha256 {got[:16]} != registered "
            f"{entry['hashes']['values.csv'][:16]} — the snapshot was modified")
    print(f"   source     : {src}  (v{version}, code={entry['code']}, "
          f"sha {got[:16]})")
    return src




def carry_quantity_sidecars(src_dir: str, out_dir: str) -> None:
    """Carry the parse's quantity sidecars from the source version to the netted output.

    ``quantity_cv_scale.csv`` (ADR 0033's σ multiplier, read through the ``cv_root_scale``
    seam) and ``quantity_recon.csv`` (the audit trail) travel with the netted feed. A source
    without one removes the output's copy, so an older one never keeps scaling σ (#281).
    """
    for name in ("quantity_cv_scale.csv", "quantity_recon.csv"):
        qsrc, qdst = os.path.join(src_dir, name), os.path.join(out_dir, name)
        if os.path.isfile(qsrc):
            if os.path.abspath(qsrc) != os.path.abspath(qdst):
                shutil.copyfile(qsrc, qdst)
                print(f"   carried forward {name} from {src_dir}")
        elif os.path.isfile(qdst):
            os.remove(qdst)
            print(f"   removed {name} from {out_dir}: {src_dir} has none")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(source_version: int) -> None:
    print("=" * 70)
    print("Re-attributing re-exported trade at feed time (ADR 0027)")
    print(f"   sectors    : {SECTORS}")
    src_dir = _source_dir(source_version)
    check_region_map(src_dir, OUT_DIR)
    check_icsg_baci_version(_register_inputs().feed.resolve(ICSG_ID), source_version,
                            FEED_ID)

    conc_lookup = traded_conc_lookup(
        _input_registries().params.resolve("conc") / "params.csv", GROSS_VALUATION,
        TRADED_SECTORS)
    _wild = {s_: v for (s_, r_), v in conc_lookup.items() if r_ == "*"}
    _reg = len(conc_lookup) - len(_wild)
    print(f"   conc       : {_wild}" + (f"  (+{_reg} region-resolved)" if _reg else ""))
    raw, legs_old, _, _ = load_root_trade(src_dir, conc_lookup)
    print(f"   raw rows   : {len(raw):,}   cross-region root legs: {len(legs_old):,}"
          f"   gross {legs_old['cc'].sum():,.0f} kt CC")

    inputs = _input_registries()
    intake, production, reported = icsg_intake_production(inputs)
    params = inputs.params
    aggregate_of = icsg_aggregate_members(params.resolve(AGGREGATES_ID) / "params.csv",
                                          AGGREGATES_ID)
    print(f"   ICSG aggregates: {len(aggregate_of)} member region-years "
          f"({AGGREGATES_ID} v{params.entry(AGGREGATES_ID)['version']})")
    screened = screened_origins(legs_old, production)
    print(f"   screened origins (exports > {OUTLIER_K}×(P+I), P>0): "
          f"{len(screened)} sector-region-years")

    volumes = reexport_volumes(legs_old, intake, production, reported, aggregate_of,
                               HUB_EXCLUDE)
    volumes["label"] = [label_row(r) for r in volumes.itertuples(index=False)]

    legs_new, events = net_reexports(legs_old, volumes, screened)

    # ── report ───────────────────────────────────────────────────────────────
    t_applied = events["T_applied"].sum() if len(events) else 0.0
    print(f"\n   T applied  : {t_applied:,.0f} kt CC "
          f"({t_applied / legs_old['cc'].sum() * 100:.2f} % of gross trade) in "
          f"{len(events):,} hub-years, {events['pass_no'].max() + 1 if len(events) else 0} passes")
    print(f"   re-imports and folded self-pairs dropped: "
          f"{events['self_dropped'].sum() if len(events) else 0:,.0f} kt CC")
    print(f"   new legs   : {int(events['n_new_legs'].sum()) if len(events) else 0:,}")
    if len(events):
        by_s = events.groupby("sector")["T_applied"].sum().round(0)
        print("   by sector  : " + "  ".join(f"{k} {v:,.0f}" for k, v in by_s.items()))
        top = (events.groupby(["sector", "region"])["T_applied"].sum()
               .sort_values(ascending=False).head(12))
        print("   top hubs   : " + ",  ".join(
            f"{s} {r} {v:,.0f}" for (s, r), v in top.items()))
    lab = volumes.groupby("label").agg(n=("T", "size"), T=("T", "sum"),
                                       resid=("resid", "sum")).round(0)
    print("\n   ledger by label:\n" + lab.to_string())
    print(f"   residual still clipped at the fan: "
          f"{volumes['resid'].clip(lower=0).sum():,.0f} kt CC")
    print(f"   imports_exhausted (withheld from the ADR 0018 fan): "
          f"{int(volumes['imports_exhausted'].sum())} groups")

    scales = cv_scale(legs_old, legs_new)
    print(f"\n   σ scale    : {len(scales):,} changed legs; "
          f"median {scales['cv_scale'].median():.3f}  "
          f"p90 {scales['cv_scale'].quantile(0.9):.3f}  "
          f"max {scales['cv_scale'].max():.3f}  "
          f"share>2 {(scales['cv_scale'] > 2).mean() * 100:.1f} %  "
          f"at the cap {(scales['cv_scale'] >= MAX_CV_SCALE).sum():,}")

    values_new = rebuild_values(raw, legs_old, legs_new, conc_lookup, HS_OF)
    print(f"   values.csv : {len(raw):,} rows → {len(values_new):,}   "
          f"mass {raw['Value'].sum():,.0f} → {values_new['Value'].sum():,.0f} kt")

    # ── write ────────────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    values_new.to_csv(os.path.join(OUT_DIR, "values.csv"), index=False)
    led = volumes.merge(
        events.groupby(["sector", "region", "time"]).agg(
            T_applied=("T_applied", "sum"), self_dropped=("self_dropped", "sum"),
            n_new_legs=("n_new_legs", "sum")).reset_index(),
        on=["sector", "region", "time"], how="left")
    led["origins_screened"] = [
        len({a for (s, a, t) in screened if s == r.sector and t == r.time})
        for r in led.itertuples(index=False)]
    led.to_csv(os.path.join(OUT_DIR, "reexport_ledger.csv"), index=False)
    key = ["sector", "region_from", "region_to", "time"]
    legs_old.merge(legs_new, on=key, how="outer", suffixes=("_obs", "_new")) \
        .fillna({"cc_obs": 0.0, "cc_new": 0.0}) \
        .assign(delta=lambda d: d["cc_new"] - d["cc_obs"]) \
        .query("abs(delta) > 1e-9") \
        .to_csv(os.path.join(OUT_DIR, "reexport_legs.csv"), index=False)
    scales.to_csv(os.path.join(OUT_DIR, "reexport_cv_scale.csv"), index=False)
    # ADR 0033's σ multiplier is produced by the parse, one stage further up,
    # and rides the same cv_root_scale seam — so it has to travel with the
    # netted feed rather than being left behind in the source version.
    carry_quantity_sidecars(src_dir, OUT_DIR)
    print(f"\n   wrote values.csv + 3 ledgers → {OUT_DIR}")

    entry = _register_inputs().feed.entry(FEED_ID)
    print(f"   registered {FEED_ID} v{entry['version']} ({entry['code']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--source-version", type=int, required=True,
                    help="registered gross baci input version to net — the one the ICSG "
                         "input was parsed against")
    ap.add_argument("--data-dir", default="data",
                    help="data root holding input/ (default: %(default)s)")
    args = ap.parse_args()
    set_data_dir(args.data_dir)
    with stage_lock("build_baci_reexport", root=os.path.join(args.data_dir, "proc")):
        main(args.source_version)
