"""
ore_grade_parse.py
==================
Transform the raw integrated-estimation-factor CSV into a fully populated
parameter CSV (params template format) for the CO/SB ore-grade concentration
parameter.

Background
----------
The raw file  data/input/params/all_integrated_est_fac_2024-07-04.csv
stores, for each country × year, the ratio

    factor = t_crude_ore / t_Cu_metal

The inverse is the Cu ore grade (concentration):

    ore_grade = 1 / factor   [t_Cu per t_crude_ore]

This is a *concentration* parameter linking the Mass and CC valuations of
every CO/SB edge in the accounting structure.  Two parameter rows are written
per (root_region, root_time):
  - use-direction: CO/SB appears as the to-node of the edge
  - supply-direction: CO/SB appears as the from-node of the edge

ROW aggregation
---------------
Countries that map to ROW are averaged (arithmetic mean of ore grades) per
year.  This is unweighted because production quantities are not available in
this file.

Pedigree
--------
User-specified scores (reliability=2, completeness=2, temporal_correlation=1,
geographical_correlation=1, other_correlation=2).

Output
------
data/input/params/ore_grade_params.csv   (semicolon-delimited, params template)

Usage
-----
cd /path/to/pySUTO
python scripts/parse/ore_grade_parse.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from pysuto.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

RAW_FILE   = os.path.join("data", "input", "params",
                           "all_integrated_est_fac_2024-07-04.csv")
MAP_FILE   = os.path.join("data", "input", "params", "ore_grade_map.xlsx")
OUT_FILE   = os.path.join("data", "input", "params", "ore_grade_params.csv")

MATERIAL_ID = "Me.Cu"

# Pedigree scores (Laner et al. 2016), 1 = best / 4 = worst
PEDIGREE = dict(
    reliability=2,
    completeness=2,
    temporal_correlation=1,
    geographical_correlation=1,
    other_correlation=2,
)

# CO/SB sector and entity in the root accounting structure
ROOT_SECTOR = "CO"
ROOT_ENTITY = "SB"

# Source string appended to each output row
SOURCE_LABEL = "S&P Global / FINEPRINT mining data; price-based allocation of ore (all_integrated_est_fac_2024-07-04.csv)"

# Params template column order (must match params_template.csv header)
TEMPLATE_COLS = [
    "param_id", "param_type", "description",
    "a_sector_from", "a_entity_from", "a_region_from",
    "a_sector_to",   "a_entity_to",   "a_region_to",
    "a_time",        "a_valuation",
    "b_sector_from", "b_entity_from", "b_region_from",
    "b_sector_to",   "b_entity_to",   "b_region_to",
    "b_time",        "b_valuation",
    "value_nominal", "unc_type",
    "value_sd", "value_min", "value_max",
    "pedigree_reliability", "pedigree_completeness",
    "pedigree_temporal_correlation", "pedigree_geographical_correlation",
    "pedigree_other_correlation",
    "epsilon", "bound_lower", "bound_upper",
    "source", "comment",
]


# ---------------------------------------------------------------------------
# LOAD MAPPINGS
# ---------------------------------------------------------------------------

def load_mappings(map_file: str) -> tuple[dict, dict]:
    """Return (alphanumiso_to_region, year_to_root) lookup dicts."""
    xls = pd.ExcelFile(map_file)

    alpha_df = xls.parse("alphanumiso_map")
    alpha_map = dict(zip(alpha_df["alphanumiso"], alpha_df["root_region"]))

    time_df = xls.parse("time_map")
    time_map = dict(zip(time_df["src_time"].astype(str),
                        time_df["root_time"].astype(str)))

    return alpha_map, time_map


# ---------------------------------------------------------------------------
# LOAD AND FILTER RAW DATA
# ---------------------------------------------------------------------------

def load_raw(raw_file: str, material_id: str,
             alpha_map: dict, time_map: dict) -> pd.DataFrame:
    """Read raw CSV, filter to material_id, apply region and time mappings."""
    df = pd.read_csv(raw_file, sep=";", dtype=str)
    df = df[df["material_id"] == material_id].copy()
    logger.info("  %d rows for material_id=%s", len(df), material_id)

    df["factor"] = pd.to_numeric(df["factor"], errors="coerce")
    df = df[df["factor"] > 0].copy()

    df["root_region"] = df["alphanumiso"].map(alpha_map)
    df["root_time"]   = df["year"].map(time_map)

    unmapped_regions = df["root_region"].isna().sum()
    if unmapped_regions:
        logger.warning("  %d rows with unmapped alphanumiso (dropped)", unmapped_regions)
    unmapped_times = df["root_time"].isna().sum()
    if unmapped_times:
        logger.info("  %d rows outside root time range (dropped)", unmapped_times)

    df = df.dropna(subset=["root_region", "root_time"])
    df["ore_grade"] = 1.0 / df["factor"]
    return df


# ---------------------------------------------------------------------------
# AGGREGATE ROW
# ---------------------------------------------------------------------------

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    For direct-mapped regions: one row per (root_region, root_time).
    For ROW: arithmetic mean of ore_grade across all aggregated countries
    per root_time.
    """
    # Load mapping type to distinguish direct vs aggregated
    xls       = pd.ExcelFile(MAP_FILE)
    alpha_df  = xls.parse("alphanumiso_map")
    type_map  = dict(zip(alpha_df["alphanumiso"], alpha_df["mapping_type"]))
    df        = df.copy()
    df["mapping_type"] = df["alphanumiso"].map(type_map)

    direct = (
        df[df["mapping_type"] == "direct"]
        .groupby(["root_region", "root_time"], as_index=False)
        .agg(ore_grade=("ore_grade", "mean"))
    )

    row_agg = (
        df[df["mapping_type"] == "aggregated"]
        .groupby("root_time", as_index=False)
        .agg(ore_grade=("ore_grade", "mean"),
             n_countries=("alphanumiso", "nunique"))
    )
    row_agg["root_region"] = "ROW"
    logger.info("  ROW: aggregated mean over %s countries (unweighted)",
                row_agg["n_countries"].iloc[0] if len(row_agg) else 0)

    combined = pd.concat([direct, row_agg[["root_region", "root_time", "ore_grade"]]],
                         ignore_index=True)
    return combined.sort_values(["root_region", "root_time"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# BUILD PARAM ROWS
# ---------------------------------------------------------------------------

def make_rows(agg: pd.DataFrame) -> list[dict]:
    """
    For each (root_region, root_time), emit one supply-direction param row:
      CO/SB as from-node (a_sector_from=CO, a_entity_from=SB)

    CO only appears as a from-node in the accounting structure (CO→Mining).
    No process outputs CO — it is extracted from the natural resource stock —
    so a use-direction row (CO as to-node) would never match any admissible edge.

    Bounds are data-driven per region: bound_lower = min ore_grade across all
    years in the root time range; bound_upper = max ore_grade across all years.
    """
    # per-region bounds from the full time series
    bounds = (
        agg.groupby("root_region")["ore_grade"]
        .agg(bound_lower="min", bound_upper="max")
    )

    rows = []
    for _, r in agg.iterrows():
        region = r["root_region"]
        time   = r["root_time"]
        grade  = round(r["ore_grade"], 8)
        tag    = f"{region.replace(', ', '_').replace(' ', '_')}_{time}"

        bl = round(bounds.loc[region, "bound_lower"], 8)
        bu = round(bounds.loc[region, "bound_upper"], 8)

        row = {
            "param_id":    f"ore_grade_{tag}",
            "param_type":  "concentration",
            "description": f"Cu ore grade {region} {time} (CO/SB from-node)",
            "a_sector_from":  ROOT_SECTOR,
            "a_entity_from":  ROOT_ENTITY,
            "a_region_from":  region,
            "a_sector_to":    "*",
            "a_entity_to":    "*",
            "a_region_to":    "*",
            "a_time":         time,
            "a_valuation":    "Mass",
            "b_sector_from":  "same", "b_entity_from": "same", "b_region_from": "same",
            "b_sector_to":    "same", "b_entity_to":   "same", "b_region_to":   "same",
            "b_time":         "same",
            "b_valuation":    "CC",
            "value_nominal":  grade,
            "unc_type":       "pedigree",
            "value_sd":       "", "value_min": "", "value_max": "",
            "pedigree_reliability":              PEDIGREE["reliability"],
            "pedigree_completeness":             PEDIGREE["completeness"],
            "pedigree_temporal_correlation":     PEDIGREE["temporal_correlation"],
            "pedigree_geographical_correlation": PEDIGREE["geographical_correlation"],
            "pedigree_other_correlation":        PEDIGREE["other_correlation"],
            "epsilon":        "",
            "bound_lower":    bl,
            "bound_upper":    bu,
            "source":         SOURCE_LABEL,
            "comment":        "",
        }
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Loading mappings from %s", MAP_FILE)
    alpha_map, time_map = load_mappings(MAP_FILE)
    logger.info("  %d alphanumiso codes, %d root time steps", len(alpha_map), len(time_map))

    logger.info("Loading raw data from %s", RAW_FILE)
    df = load_raw(RAW_FILE, MATERIAL_ID, alpha_map, time_map)
    logger.info("  %d rows after mapping + filtering", len(df))

    logger.info("Aggregating to (root_region, root_time)")
    agg = aggregate(df)
    logger.info("  %d (region, time) combinations", len(agg))

    logger.info("Building param rows")
    rows = make_rows(agg)
    logger.info("  %d param rows (supply-direction only × %d entries)", len(rows), len(agg))

    out = pd.DataFrame(rows, columns=TEMPLATE_COLS)
    out.to_csv(OUT_FILE, sep=";", index=False)
    logger.info("Written: %s", OUT_FILE)

    # Quick summary
    n_regions   = out["a_region_from"].replace("*", pd.NA).dropna().nunique()
    n_years     = out["a_time"].nunique()
    logger.info("  coverage: ~%d root regions × %d years", n_regions, n_years)


if __name__ == "__main__":
    main()
