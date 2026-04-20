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
4. Aggregates by (Year, Sector, Region_from, Region_to) — sum quantity.
5. Converts quantity from tonnes (BACI native) to kt  (÷ 1000).
6. Adds a constant Unit column "kt".
7. Writes data/input/feed/baci/v0/values.csv.

No copper-content weighting is applied — the value is simply the traded
mass in kt as reported by BACI.

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

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

BACI_RAW_DIR = os.path.join("data", "input", "raw", "BACI_HS92_V202601")
OUT_PATH     = os.path.join("data", "input", "feed", "baci", "v0", "values.csv")

HS_CODES = {260300, 740110, 740120, 740200, 740311}

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    files = sorted(
        f for f in os.listdir(BACI_RAW_DIR)
        if f.startswith("BACI") and f.endswith(".csv")
    )
    if not files:
        sys.exit(f"No BACI CSV files found in {BACI_RAW_DIR!r}")

    print(f"Found {len(files)} BACI files — filtering for HS codes: {sorted(HS_CODES)}")

    chunks = []
    for fname in files:
        path = os.path.join(BACI_RAW_DIR, fname)
        df = pd.read_csv(path, usecols=["t", "i", "j", "k", "q"], dtype={"k": int})
        df = df[df["k"].isin(HS_CODES)]
        df = df[df["q"].notna() & (df["q"] > 0)]
        if not df.empty:
            chunks.append(df)
        print(f"  {fname}: {len(df):,} rows kept")

    if not chunks:
        sys.exit("No rows matched the HS code filter.")

    raw = pd.concat(chunks, ignore_index=True)

    # Aggregate: sum quantity per (year, HS code, exporter, importer)
    agg = (
        raw.groupby(["t", "k", "i", "j"], sort=False)["q"]
        .sum()
        .reset_index()
    )

    # Rename and convert tonnes → kt
    agg = agg.rename(columns={"t": "Year", "k": "Sector",
                               "i": "Region_from", "j": "Region_to",
                               "q": "Value"})
    agg["Value"] = agg["Value"] / 1000.0
    agg["Unit"]  = "kt"

    # Cast to tidy types
    for col in ["Year", "Sector", "Region_from", "Region_to"]:
        agg[col] = agg[col].astype(int)
    agg = agg.sort_values(["Year", "Sector", "Region_from", "Region_to"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    agg.to_csv(OUT_PATH, index=False)

    print(f"\nWrote {len(agg):,} rows → {OUT_PATH}")
    print(f"  Years:    {agg['Year'].min()} – {agg['Year'].max()}")
    print(f"  Sectors:  {sorted(agg['Sector'].unique())}")
    print(f"  Total kt: {agg['Value'].sum():,.1f}")


if __name__ == "__main__":
    main()
