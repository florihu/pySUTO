"""
baci_parse_feed.py
==================
Pre-process raw BACI HS92 files into the feedpipe-ready values.csv for the
baci feed (data/input/feed/baci/v0/).

What this script does
---------------------
1. Iterates over all BACI_HS92_Y<YYYY>_V202601.csv files in BACI_RAW_DIR.
2. Filters early to only the 8 HS codes covered by the conc.xlsx sector_map:
       260300  Copper ores and concentrates   → Concentrate
       740110  Copper mattes                  → Matte
       740120  Cement copper                  → Matte
       740200  Unrefined copper / anodes      → Anode
       740311  Copper cathodes                → Cathode
       740312  Copper wire-bars               → Cathode   (#91: refined copper;
       740313  Copper billets                 → Cathode    cross-region trade 4.5–15.1 %
       740319  Other refined copper           → Cathode    of 740311's a year, 1995–2024)
3. Drops rows with missing or zero quantity.
4. Aggregates by (Year, Sector, Region_from, Region_to) — sum quantity *and value*.
4b. Drops refined copper declared into a country it only passes through (#95,
    ``TRANSIT_DECLARATIONS``); then, in the years one code repeats another's
    declarations, keeps the larger refined-copper export per partner and drops
    the code's other rows (#88, #95, ``MIRROR_DUPLICATES``).
5. Reconstructs the quantity of legs whose value contradicts it, on either side
   of the band (ADR 0033, below).
6. Converts quantity from tonnes (BACI native) to kt  (÷ 1000).
7. Adds a constant Unit column "kt".
8. Writes data/input/feed/baci/v0/values.csv + two ADR 0033 sidecars.

No copper-content weighting is applied — the value is simply the traded
mass in kt as reported by BACI.

The quantity reconstruction (ADR 0033)
--------------------------------------
HS 740311 is refined cathode (~99.99 % Cu), so its unit value *is* the copper
price in the same release, same year, same valuation convention.  The
**value-implied Cu fraction** ``unit_value(product) / unit_value(cathode)``
must therefore lie inside ADR 0028's physical band [0.20, 0.35].  Chile leaves
it in 18 of 30 years, reaching 0.65-0.68 — blister territory, not concentrate —
and BACI's *value* agrees with ICSG production in every year (0.92-1.09) while
its *quantity* is 60 % out in exactly those years.  The quantity is the broken
field, so it is rebuilt from the value at the region's own measured fraction.

The reconstruction is two-sided (Round Z).  Both sides need a concentrate leg
of at least ``MIN_LEG_T`` and rebuild it by the same ``q = v / (P_cu · g_ref)``:

above  leg and exporter-year fraction > ``BAND_HI`` (the ceiling): too little
       mass for the value; the leg is only ever raised, never below ``q_obs``.
below  leg fraction < ``LO_LEG_GATE`` (floor / 10) in an exporter-year below
       ``LO_YEAR_GATE`` (the floor): too much mass for the value; the leg is
       only ever lowered, never above ``q_obs``.

Papua New Guinea's 2019-2022 legs are the low-side case: PNG → Philippines
2020 is 21,277,530 t at 3.6 USD/t, a fraction of 0.0006, kilograms booked as
tonnes.  The leg gate is not the floor: a value is net of payables and
treatment charges, so real concentrate sits at 0.15-0.20.

_Avoid_: reading the fraction as a grade.  GLOSSARY.md's **Concentrate grade**
entry already warns that values of 0.6-0.9 "are not grades at all" but a
statement that the recorded export mass is too small.

Output columns
--------------
Year         int    trade year
Sector       int    HS92 code (matches sector_map src_sector)
Region_from  int    BACI exporter country code (matches region_map src_region)
Region_to    int    BACI importer country code (matches region_map src_region)
Value        float  traded quantity in kt
Unit         str    "kt"
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from pysuto.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

_INPUT_ROOT  = os.path.join("data", "input")
BACI_RAW_DIR = os.path.join("data", "input", "raw", "BACI_HS92_V202601")
OUT_PATH     = os.path.join("data", "input", "feed", "baci", "v0", "values.csv")

HS_CODES = {260300, 740110, 740120, 740200, 740311, 740312, 740313, 740319}

CATHODE_HS = 740311          # ~99.99 % Cu — its unit value is the copper price
REFINED_HS = {740311, 740312, 740313, 740319}   # HS 7403.1x, one Cathode sector (#91)
RECON_HS   = {260300}        # ADR 0033 scope: only concentrate mass enters
                             # ADR 0029's cap = 0.35·X + U, so reconstructing
                             # matte/anode/cathode cannot move the forced term.

BAND_LO, BAND_HI = 0.20, 0.35       # ADR 0028's hard concentrate cone
LO_LEG_GATE      = BAND_LO / 10.0   # low side (Round Z): an order of magnitude
                                    # below the floor, clear of 0.15-0.20, which
                                    # is real concentrate net of payables
LO_YEAR_GATE     = BAND_LO          # ...in an exporter-year itself below the cone
G_NOM            = 0.275            # the conc row's nominal
CV_GRADE_DECL    = 0.0375 / 0.275   # ...and its declared σ, as a cv: 0.1364
CV_PED           = 0.209            # BACI Mass pedigree cv (ADR 0017)
MAX_CV_SCALE     = 5.0              # ADR 0027's cap; past ~4.8 ADR 0017's
                                    # damping ceiling makes it inert anyway
MIN_LEG_T        = 1_000.0          # tonnes. Below this a leg's unit value is
                                    # rounding dust (Chile 2021 carries partners
                                    # at an implied fraction of 8.6).
MIN_IN_BAND      = 2                # fewer than this and the spread is not
                                    # measurable — which is ignorance, not
                                    # precision, so σ goes to the cap.

# A code whose rows repeat copper another code already declares: code → (the code it
# mirrors, first, last such year) (#88, #95).  BACI 178 "Congo" carries DR Congo's
# refined copper as its partners declare it (#84).  DR Congo's own declarations (180,
# all of HS 7403.1x, #91) fall 5–323 kt short of ICSG production in every year through
# 2014 and 476 kt in 2024, which 178 fills.  In 2015–2023 they lie within −74 … +422 kt
# of it, and 178 (226–465 kt a year) on top would count the copper twice — but not to
# every partner: 178 → UAE is 80–252 kt a year against 180 → UAE 53–142 kt, and dropping
# 178 outright (#88) left the UAE re-exporting cathode it no longer imported (common-set
# Δz² +25,625 vs opt_20261015_093646_ddd9d3d9, on opt_20261019_142026_83bfe843).  So per
# partner the larger refined-copper export is kept (#95); every other row of 178 in the
# window is still dropped (#88), since no evidence covers it.
MIRROR_DUPLICATES = {178: (180, 2015, 2023)}

# Exporter code → countries its refined copper only passes through, dropped in every
# year (#95).  DR Congo's refined copper leaves through Dar es Salaam and Beira, and the
# final buyer declares DR Congo as origin too, so the leg into the port country counts it
# twice.  180 → Tanzania is 302.3, 130.5, 35.6, 98.0, 264.9, 247.4, 196.3, 134.7, 143.2 kt
# in 2015–2023 (0.1 kt in 2024); 180 → Mozambique 15.8, 61.1, 28.3, 47.2, 103.0, 126.0 kt in
# 2018–2023 (0.1 kt in 2015); none before 2015, and 178 sends neither anything.  Tanzania
# exported 213 kt of refined copper in 2015–2023, Mozambique 16 kt; neither refines copper
# (ICSG production 0).  Only HS 7403.1x: concentrate into them was not analysed.
TRANSIT_DECLARATIONS = {180: (834,    # Tanzania (Dar es Salaam)
                              508)}   # Mozambique (Beira)

OUT_DIR       = os.path.dirname(OUT_PATH)
RECON_PATH    = os.path.join(OUT_DIR, "quantity_recon.csv")
CV_SCALE_PATH = os.path.join(OUT_DIR, "quantity_cv_scale.csv")


# ---------------------------------------------------------------------------
# #88, #95 — declarations that repeat copper declared elsewhere
# ---------------------------------------------------------------------------

def drop_transit_declarations(agg, transit) -> pd.DataFrame:
    """``agg`` without the refined-copper rows (HS 7403.1x) that a declared exporter sends
    into a country the copper only passes through, in every year.  A declared (exporter,
    hub) pair that matches no rows raises: it is stale."""
    for exporter, hubs in transit.items():
        for hub in hubs:
            rows = ((agg["Region_from"] == exporter) & (agg["Region_to"] == hub)
                    & agg["Sector"].isin(REFINED_HS))
            if not rows.any():
                raise ValueError(f"TRANSIT_DECLARATIONS: no refined-copper rows of code "
                                 f"{exporter} into {hub}")
            by = agg.loc[rows].groupby("Year")["q"].sum() / 1e3
            logger.info("#95 refined copper of code %d into %d is a transit declaration: "
                        "%d rows dropped, kt by year %s", exporter, hub, int(rows.sum()),
                        by.round(1).to_dict())
            agg = agg.loc[~rows]
    return agg.reset_index(drop=True)


def reconcile_mirror_duplicates(agg, duplicates) -> pd.DataFrame:
    """``agg`` reconciled in the years ``(first, last)`` a code repeats the copper its
    mirrored code declares; every other year is untouched.

    Refined-copper exports (HS 7403.1x, the two codes as exporter) keep, per year and
    partner, the code whose declaration is larger, its rows under their own HS and region
    codes; the mirrored code wins a tie (#95).  Every other row of the code, either side,
    is dropped (#88): the evidence covers only refined exports.  The comparison pools the
    refined codes per partner because the Cathode sector reads them as one (#91) and the
    two sides of one shipment need not declare the same code: in 2015–2017 DR Congo (180)
    declares 52–71 kt a year to the UAE as 740319 that the UAE declares 740311 under 178,
    so a per-code max would keep that copper twice.  A declaration raises as stale when
    either code has no rows in its years, or when it drops nothing.
    """
    for code, (mirrored, first, last) in duplicates.items():
        in_window = agg["Year"].between(first, last)
        for c in (code, mirrored):
            of_c = (agg["Region_from"] == c) | (agg["Region_to"] == c)
            if not (in_window & of_c).any():
                raise ValueError(f"MIRROR_DUPLICATES: no rows of code {c} "
                                 f"in {first}–{last}")
        codes = [code, mirrored]
        refined_export = (in_window & agg["Sector"].isin(REFINED_HS)
                          & agg["Region_from"].isin(codes) & ~agg["Region_to"].isin(codes))
        declared = (agg[refined_export].groupby(["Year", "Region_to", "Region_from"])["q"]
                    .sum().unstack("Region_from", fill_value=0.0)
                    .reindex(columns=codes, fill_value=0.0))
        smaller = pd.Series(np.where(declared[code] > declared[mirrored], mirrored, code),
                            index=declared.index)
        partner_year = pd.MultiIndex.from_arrays([agg["Year"], agg["Region_to"]])
        smaller_code = smaller.reindex(partner_year).to_numpy()
        outdeclared = refined_export & (agg["Region_from"].to_numpy() == smaller_code)
        of_code = (agg["Region_from"] == code) | (agg["Region_to"] == code)
        other = in_window & of_code & ~refined_export
        if not (outdeclared | other).any():
            raise ValueError(f"MIRROR_DUPLICATES: code {code} drops nothing against "
                             f"{mirrored} in {first}–{last}")
        refined_by = agg.loc[outdeclared].groupby("Region_from")["q"].sum() / 1e3
        other_by = agg.loc[other].groupby("Sector")["q"].sum() / 1e3
        logger.info("#95 codes %d and %d in %d–%d: the smaller refined-copper export per "
                    "partner dropped, kt by code %s; #88 code %d's other rows dropped, kt by "
                    "sector %s", code, mirrored, first, last, refined_by.round(1).to_dict(),
                    code, other_by.round(1).to_dict())
        agg = agg.loc[~(outdeclared | other)]
    return agg.reset_index(drop=True)


# ---------------------------------------------------------------------------
# ADR 0033 — the value-implied Cu fraction, and the reconstruction it licenses
# ---------------------------------------------------------------------------

def copper_price(agg) -> pd.Series:
    """Per year: the world cathode unit value in USD/t — the copper price.

    Taken from HS 740311 pooled over every reporter, so it does not depend on
    any one exporter having a cathode line of its own.
    """
    c = agg[agg["Sector"] == CATHODE_HS]
    g = c.groupby("Year")[["v", "q"]].sum()
    return (g["v"] * 1000.0 / g["q"]).rename("P_cu")


def value_implied_cu_fraction(legs, p_cu) -> pd.Series:
    """``unit_value(leg) / copper price`` — NOT a grade (see the module docstring)."""
    unit = legs["v"] * 1000.0 / legs["q"]
    return unit / legs["Year"].map(p_cu)


def reference_fractions(legs, p_cu) -> pd.DataFrame:
    """Per exporter: ``g_ref`` and the σ multiplier its reconstruction earns.

    ``g_ref`` is the median of that region's **in-band** legs — measured, never
    assumed.  Its uncertainty is the region's own spread about it, floored at
    the ``conc`` row's declared cv: the model may not claim to know a grade
    better than the row it already publishes.  A region with fewer than
    ``MIN_IN_BAND`` in-band legs has no measurable spread, which is ignorance,
    so it goes to the cap rather than the floor.
    """
    # Calibrated on exporter-YEARS, not on individual legs: that is where the
    # ICSG cross-check licenses the value-to-copper relationship, it is the
    # statistic ADR 0029 pooled (Chile 0.2783 on its in-band years), and a
    # per-leg median is dragged low by small legs — which would inflate the
    # reconstructed mass, since q = v/(P_cu·g_ref).
    yr = legs.groupby(["Region_from", "Year"])[["v", "q"]].sum()
    f = (yr["v"] * 1000.0 / yr["q"]
         / yr.index.get_level_values("Year").map(p_cu))
    in_band = (yr["q"] >= MIN_LEG_T) & f.between(BAND_LO, BAND_HI)
    d = pd.DataFrame({"region": yr.index.get_level_values("Region_from"),
                      "f": f.to_numpy()})[in_band.to_numpy()]
    g = d.groupby("region")["f"]
    out = pd.DataFrame({"g_ref": g.median(), "sd": g.std(), "n_in_band": g.size()})
    out["g_ref"] = out["g_ref"].fillna(G_NOM).clip(BAND_LO, BAND_HI)
    cv = (out["sd"] / out["g_ref"]).clip(lower=CV_GRADE_DECL)
    cv[out["n_in_band"] < MIN_IN_BAND] = np.inf          # ignorance, not precision
    out["cv_gref"] = cv
    out["cv_scale"] = np.sqrt(1.0 + (cv / CV_PED) ** 2).clip(1.0, MAX_CV_SCALE)
    return out


def reconstruct_quantity(legs, refs, p_cu) -> pd.DataFrame:
    """Rebuild the quantity of every leg whose value contradicts it.

    ``q_new = v / (P_cu · g_ref)`` — the value and the copper price are well
    measured; ``g_ref`` is the uncertain input, and that is the uncertainty the
    leg then carries, on either side.  ``side`` says which contradiction a leg
    carries, and ``flagged`` is ``side != "none"``:

    ``above``  leg and exporter-year fraction > ``BAND_HI``: too little mass for
               the value; ``q_new`` is never below ``q_obs``.
    ``below``  leg fraction < ``LO_LEG_GATE`` in an exporter-year below
               ``LO_YEAR_GATE``: too much mass for the value (PNG 2019-2022
               books kilograms as tonnes); ``q_new`` is never above ``q_obs``.

    Legs under ``MIN_LEG_T`` are never flagged on either side.  Unflagged legs
    keep ``q_new = q_obs`` and ``cv_scale = 1``.
    """
    out = legs.copy()
    out["cu_frac"] = value_implied_cu_fraction(out, p_cu)
    out["g_ref"] = out["Region_from"].map(refs["g_ref"]).fillna(G_NOM)
    out["cv_scale"] = out["Region_from"].map(refs["cv_scale"]).fillna(MAX_CV_SCALE)
    # Two gates, and both are needed. The ICSG cross-check validates the
    # *aggregate* value-to-copper relationship per exporter-year, so that is
    # where the decision to intervene is licensed: in a year whose aggregate
    # fraction is in band the recorded mass already matches the value, and
    # inflating individual legs there would fabricate mass. Within a flagged
    # year the per-leg fraction decides *which* legs moved — the defect is a
    # gradient across partners (Chile 2021: Brazil 0.19 … China 0.86), not a
    # uniform per-country offset. The low side keeps the year gate at the floor
    # (Congo D.R. 2020: one leg at 0.005 drags the year to 0.051) but puts the
    # leg gate a decade lower, because 0.15-0.20 is real concentrate net of
    # payables while the unit errors sit below 0.01.
    conc = out[out["Sector"].isin(RECON_HS)]
    yr = conc.groupby(["Region_from", "Year"])[["v", "q"]].sum()
    yr_frac = (yr["v"] * 1000.0
               / yr["q"]
               / yr.index.get_level_values("Year").map(p_cu))
    keys = pd.MultiIndex.from_arrays([out["Region_from"], out["Year"]])
    out["year_frac"] = yr_frac.reindex(keys).to_numpy()
    scope = out["Sector"].isin(RECON_HS) & (out["q"] >= MIN_LEG_T)
    above = scope & (out["cu_frac"] > BAND_HI) & (out["year_frac"] > BAND_HI)
    below = scope & (out["cu_frac"] < LO_LEG_GATE) & (out["year_frac"] < LO_YEAR_GATE)
    out["side"] = np.select([above, below], ["above", "below"], "none")
    out["flagged"] = above | below
    out["q_obs"] = out["q"]
    q_val = out["v"] * 1000.0 / (out["Year"].map(p_cu) * out["g_ref"])
    out["q_new"] = np.select([above, below],
                             [np.maximum(q_val, out["q_obs"]),
                              np.minimum(q_val, out["q_obs"])],
                             out["q_obs"])
    out.loc[~out["flagged"], "cv_scale"] = 1.0
    return out


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def _write_sidecars(rec, refs) -> None:
    """The per-leg audit trail, and the σ multiplier in the root vocabulary.

    ``quantity_cv_scale.csv`` must match ``reexport_cv_scale.csv``'s schema —
    root sector and region *names*, not BACI codes — because both are read
    through the same ``cv_root_scale`` seam in ``baci_feed.py``.
    """
    conc_path = os.path.join(_INPUT_ROOT, "feed", "baci", "current", "conc.xlsx")
    smap = pd.read_excel(conc_path, sheet_name="sector_map")
    rmap = pd.read_excel(conc_path, sheet_name="region_map")
    sec = dict(zip(smap["src_sector"], smap["root_sector"]))
    reg = dict(zip(rmap["src_region"], rmap["root_region"]))

    rec.assign(recovered=rec["q_new"] - rec["q_obs"]) \
       .loc[rec["flagged"], ["Year", "Sector", "Region_from", "Region_to", "side",
                             "v", "q_obs", "q_new", "recovered", "cu_frac",
                             "year_frac", "g_ref", "cv_scale"]] \
       .sort_values(["Year", "Region_from", "Region_to"]) \
       .to_csv(RECON_PATH, index=False)

    f = rec[rec["flagged"]].copy()
    f["sector"] = f["Sector"].map(sec)
    f["region_from"] = f["Region_from"].map(reg)
    f["region_to"] = f["Region_to"].map(reg)
    f["time"] = f["Year"]
    # several BACI codes fold onto one root region (most of them onto ROW), so a
    # root leg can gather more than one source leg; weight by the mass each
    # carries rather than letting an arbitrary one win.
    key = ["sector", "region_from", "region_to", "time"]
    f["_w"] = f["cv_scale"] * f["q_new"]
    out = f.groupby(key).agg(q_obs=("q_obs", "sum"), q_new=("q_new", "sum"),
                             _w=("_w", "sum")).reset_index()
    out["cv_scale"] = out["_w"] / out["q_new"]
    out = out.drop(columns="_w")
    out.to_csv(CV_SCALE_PATH, index=False)
    logger.info("  sidecars: %d flagged legs → %s", len(f), RECON_PATH)
    logger.info("            %d root legs    → %s (cv_scale %.2f–%.2f)",
                len(out), CV_SCALE_PATH, out["cv_scale"].min(), out["cv_scale"].max())


def main() -> None:
    files = sorted(
        f for f in os.listdir(BACI_RAW_DIR)
        if f.startswith("BACI") and f.endswith(".csv")
    )
    if not files:
        logger.error("No BACI CSV files found in %r", BACI_RAW_DIR)
        sys.exit(1)

    logger.info("Found %d BACI files — filtering for HS codes: %s",
                len(files), sorted(HS_CODES))

    chunks, dropped_v, kept_v = [], 0.0, 0.0
    for fname in files:
        path = os.path.join(BACI_RAW_DIR, fname)
        df = pd.read_csv(path, usecols=["t", "i", "j", "k", "v", "q"], dtype={"k": int})
        df = df[df["k"].isin(HS_CODES)]
        ok = df["q"].notna() & (df["q"] > 0)
        # rows carrying a value but no quantity are dropped here and their value
        # is lost with them — reported so the loss is never silent (ADR 0033)
        dropped_v += float(df.loc[~ok, "v"].sum())
        kept_v += float(df.loc[ok, "v"].sum())
        df = df[ok]
        if not df.empty:
            chunks.append(df)
        logger.info("  %s: %d rows kept", fname, len(df))

    if not chunks:
        logger.error("No rows matched the HS code filter.")
        sys.exit(1)

    raw = pd.concat(chunks, ignore_index=True)
    logger.info("rows with value but no quantity: %.0f kUSD dropped (%.3f %% of value)",
                dropped_v, 100.0 * dropped_v / max(kept_v + dropped_v, 1e-9))

    # Aggregate: sum quantity AND value per (year, HS code, exporter, importer)
    agg = (
        raw.groupby(["t", "k", "i", "j"], sort=False)[["q", "v"]]
        .sum()
        .reset_index()
        .rename(columns={"t": "Year", "k": "Sector",
                         "i": "Region_from", "j": "Region_to"})
    )
    for col in ["Year", "Sector", "Region_from", "Region_to"]:
        agg[col] = agg[col].astype(int)
    agg = drop_transit_declarations(agg, TRANSIT_DECLARATIONS)
    agg = reconcile_mirror_duplicates(agg, MIRROR_DUPLICATES)

    # --- ADR 0033: rebuild the quantity of legs whose value contradicts it ---
    p_cu = copper_price(agg)
    conc = agg[agg["Sector"].isin(RECON_HS)]
    refs = reference_fractions(conc, p_cu)
    rec = reconstruct_quantity(agg, refs, p_cu)
    for side in ("above", "below"):
        s = rec[rec["side"] == side]
        logger.info("ADR 0033 %s the band: %d of %d concentrate legs in %d "
                    "exporter-years flagged; mass %.0f → %.0f kt (%+.0f kt)",
                    side, len(s), len(conc),
                    s.groupby(["Region_from", "Year"]).ngroups,
                    s["q_obs"].sum() / 1e3, s["q_new"].sum() / 1e3,
                    (s["q_new"] - s["q_obs"]).sum() / 1e3)

    # Rename and convert tonnes → kt
    agg = rec[["Year", "Sector", "Region_from", "Region_to"]].copy()
    agg["Value"] = rec["q_new"] / 1000.0
    agg["Unit"]  = "kt"
    agg = agg.sort_values(["Year", "Sector", "Region_from", "Region_to"]).reset_index(drop=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    agg.to_csv(OUT_PATH, index=False)
    _write_sidecars(rec, refs)

    logger.info("Wrote %d rows → %s", len(agg), OUT_PATH)
    logger.info("  years:    %d – %d", agg["Year"].min(), agg["Year"].max())
    logger.info("  sectors:  %s", sorted(agg["Sector"].unique()))
    logger.info("  total kt: %.1f", agg["Value"].sum())


if __name__ == "__main__":
    main()
