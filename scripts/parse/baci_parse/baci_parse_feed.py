"""
baci_parse_feed.py
==================
Pre-process raw BACI HS92 files into the feedpipe-ready values.csv for the
baci feed (data/input/feed/baci/v0/).

What this script does
---------------------
1. Iterates over all BACI_HS92_Y<YYYY>_V202601.csv files in BACI_RAW_DIR.
2. Filters early to only the 5 HS codes covered by the conc.xlsx sector_map:
       260300  Copper ores and concentrates   → Concentrate
       740110  Copper mattes                  → Matte
       740120  Cement copper                  → Matte
       740200  Unrefined copper / anodes      → Anode
       740311  Copper cathodes                → Cathode
3. Drops rows with missing or zero quantity.
4. Aggregates by (Year, Sector, Region_from, Region_to) — sum quantity *and value*.
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

HS_CODES = {260300, 740110, 740120, 740200, 740311}

CATHODE_HS = 740311          # ~99.99 % Cu — its unit value is the copper price
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

OUT_DIR       = os.path.dirname(OUT_PATH)
RECON_PATH    = os.path.join(OUT_DIR, "quantity_recon.csv")
CV_SCALE_PATH = os.path.join(OUT_DIR, "quantity_cv_scale.csv")


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
    # several BACI codes fold onto one root region (186 of them onto ROW), so a
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
