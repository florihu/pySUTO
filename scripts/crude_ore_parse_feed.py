"""
crude_ore_parse_feed.py
=======================
Pre-process the Kroiss Global crude metal ore extraction dataset into the
feedpipe-ready values.csv for the crude_ore feed (data/input/feed/crude_ore/v0/).

What this script does
---------------------
1. Reads Global_crude_metal_ore_extraction_1970-2022_07.2024.xlsx
   (sheet: Metal_extraction_1970-2022).
2. Filters to copper rows only: material_name == "Copper - associated ore".
3. Drops rows with zero or missing value.
4. Filters years to 1995–2022 (root accounting time range).
5. Converts quantity from tonnes (native) to kt (÷ 1000).
6. Adds a constant Unit column "kt".
7. Writes data/input/feed/crude_ore/v0/values.csv.

Output columns
--------------
Year      int    extraction year
Material  str    "Copper - associated ore" (matches sector_map src_sector)
Region    str    country name (matches region_map src_region)
Value     float  extraction quantity in kt
Unit      str    "kt"
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

RAW_FILE = os.path.join(
    "data", "input", "raw", "Crude_ore_Kroiss",
    "Global_crude_metal_ore_extraction_1970-2022_07.2024.xlsx",
)
RAW_SHEET  = "Metal_extraction_1970-2022"
OUT_PATH   = os.path.join("data", "input", "feed", "crude_ore", "v0", "values.csv")

MATERIAL_FILTER = "Copper - associated ore"
YEAR_MIN        = 1995

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Reading %s", RAW_FILE)
    df = pd.read_excel(RAW_FILE, sheet_name=RAW_SHEET, header=0)
    logger.info("  total rows: %d", len(df))

    # Filter copper
    copper = df[df["material_name"] == MATERIAL_FILTER].copy()
    logger.info("  copper rows (all years): %d", len(copper))

    # Drop zero / missing
    copper = copper[copper["value"].notna() & (copper["value"] > 0)]

    # Filter to root time range
    copper = copper[copper["year"] >= YEAR_MIN]
    logger.info("  copper rows (year >= %d, value > 0): %d", YEAR_MIN, len(copper))

    # Rename, convert, add Unit
    copper = copper.rename(columns={
        "country_name":  "Region",
        "material_name": "Material",
        "year":          "Year",
        "value":         "Value",
    })
    copper["Value"] = copper["Value"] / 1000.0
    copper["Unit"]  = "kt"
    copper["Year"]  = copper["Year"].astype(int)

    out = (
        copper[["Year", "Material", "Region", "Value", "Unit"]]
        .sort_values(["Year", "Region"])
        .reset_index(drop=True)
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    logger.info("Wrote %d rows → %s", len(out), OUT_PATH)
    logger.info("  years:    %d – %d", out["Year"].min(), out["Year"].max())
    logger.info("  regions:  %d unique", out["Region"].nunique())
    logger.info("  total kt: %.1f", out["Value"].sum())


if __name__ == "__main__":
    main()
