"""process_icsg.py
===============
ICSG stock & flow processing.

Steps:
1. Net addition to stock (NAS) from the ICSG stock levels of every vintage
   (`net_addition_to_stock`, GLOSSARY.md "Net addition to stock"):
   NAS(t) = level(t) - level(t-1) of one series in one vintage; a blank year
   yields no NAS for itself or for the year after it.
2. Map the NAS rows onto the flows schema (Region, Type, Year, Value, Process,
   Supply_Use, Unit, Year_loaded_from_file):
       PSC  -> Type = "PSC",     Process = "Cathode_Pool", Supply_Use = "Supply"
       MESC -> Type = "Cathode", Process = "MESC",         Supply_Use = "Use"
       USC  -> Type = "Cathode", Process = "USC",          Supply_Use = "Use"
3. Concatenate NAS rows with the flows of icsg_comb.py
4. Enforce consistent dtypes: Region/Type/Process/Supply_Use/Unit -> str, Year -> int, Value -> float
5. Save the NAS rows (icsg_nas.csv) and the combined table (icsg_flows_comb.csv);
   the stock levels this reads (icsg_stocks.csv) are never overwritten
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from pysuto.logging import get_logger


logger = get_logger(__name__)

CLEAN_DIR = "data/proc/feedprep/icsg_clean"

# A printed total and the sum of its types differ by rounding (at most 0.15 kt
# in the 2005-2025 yearbooks) or by a defect (1.5 kt and more: Japan's 0.000
# placeholders from 2021, Korea's merged 1995-2004 cell).
TOTAL_TOLERANCE_KT = 0.5

# combined type key (Stock_type + "_" + Type) -> stock-change leg
RENAME_DICT = {
    "PMC_Producers":          "PSC",
    "PMC_Producers/Merchants":"PSC",
    "Exchanges_LME":          "MESC",
    "Exchanges_SHFE":         "MESC",
    "Exchanges_COMEX":        "MESC",
    "PMC_Consumers":          "USC",
    "PMC_Merchants":          "USC",
    "PMC_Government":         "USC",
    "PMC_Total":              "PSC",    # a lone total counts as producers
}

SERIES_KEY = ["Region", "Stock_type", "Type", "Unit", "Year_loaded_from_file"]
FLOW_COLS  = ["Region", "Type", "Year", "Value", "Process", "Supply_Use", "Unit",
              "Year_loaded_from_file"]


def _warn_reporting_gaps(lv, is_total):
    """Log every type reported in one year and blank in the next of its vintage."""
    typed = lv[~is_total]
    last = lv.groupby(["Stock_type", "Year_loaded_from_file"])["Year"].transform("max")
    nxt = typed[SERIES_KEY].assign(Year=typed["Year"] + 1)
    reported = pd.MultiIndex.from_frame(typed[SERIES_KEY + ["Year"]])
    stops = (~pd.MultiIndex.from_frame(nxt[SERIES_KEY + ["Year"]]).isin(reported)
             & (typed["Year"] < last[~is_total]).values)
    for _, r in nxt[stops].iterrows():
        logger.warning(f"{r['Region']} {r['Stock_type']} {r['Type']}: reported in "
                       f"{r['Year'] - 1}, blank in {r['Year']} "
                       f"(vintage {r['Year_loaded_from_file']}); no change is booked")


def net_addition_to_stock(levels):
    """NAS rows in the flows schema from the stock levels of every vintage.

    Parameters
    ----------
    levels : DataFrame
        Region, Type, Year, Value, Unit, Stock_type, Year_loaded_from_file;
        one row per series and year of each vintage, as icsg_comb.py writes it.

    Returns
    -------
    DataFrame with FLOW_COLS.
    """
    lv = levels.dropna(subset=["Value"]).copy()
    lv["Year"] = lv["Year"].astype(int)
    lv = lv.sort_values(SERIES_KEY + ["Year"], kind="mergesort")
    dup = lv.duplicated(SERIES_KEY + ["Year"], keep=False)
    if dup.any():
        raise ValueError("two stock levels for one series and year:\n"
                         f"{lv[dup].to_string()}")

    # A year's types are its breakdown unless a printed total contradicts them;
    # then the total stands alone, as a lone total does.  A total is always a
    # series of its own, so it differences across a year its types do not.
    year_key = ["Region", "Stock_type", "Unit", "Year_loaded_from_file", "Year"]
    is_total = lv["Type"].eq("Total")
    _warn_reporting_gaps(lv, is_total)
    sum_types = lv[~is_total].groupby(year_key)["Value"].sum()
    total = lv[is_total].set_index(year_key)["Value"]
    both = pd.concat([sum_types.rename("types"), total.rename("total")],
                     axis=1, join="inner")
    contradicted = both[(both["types"] - both["total"]).abs() > TOTAL_TOLERANCE_KT]
    for (region, stock, _, vintage, year), r in contradicted.iterrows():
        logger.warning(f"{region} {stock} {year} (vintage {vintage}): types sum to "
                       f"{r['types']:.1f} kt, the printed total is {r['total']:.1f} kt; "
                       "the total stands alone")
    at_year = pd.MultiIndex.from_frame(lv[year_key])
    lv["breakdown"] = at_year.isin(sum_types.index) & ~at_year.isin(contradicted.index)
    lv = lv[is_total | lv["breakdown"]]

    # A total's change is booked only between two years in which it stands
    # alone: the year it takes over from its types mixes a stock change with a
    # change of reporting, and such a change is not recovered (triage Q9).
    g = lv.groupby(SERIES_KEY, sort=False)
    consecutive = (lv["Year"] - g["Year"].shift()) == 1
    lv["NAS"] = lv["Value"] - g["Value"].shift()
    alone_both_years = ~lv["breakdown"] & g["breakdown"].shift().eq(False)
    nas = lv[consecutive & (lv["Type"].ne("Total") | alone_both_years)].copy()

    # the series of one leg add within a vintage: USC = consumers + merchants +
    # government, MESC = every exchange at one location
    type_key = nas["Stock_type"] + "_" + nas["Type"]
    nas["Leg"] = type_key.map(RENAME_DICT)
    if nas["Leg"].isna().any():
        raise KeyError(f"stock types with no stock-change leg: "
                       f"{sorted(set(type_key[nas['Leg'].isna()]))}")
    leg_key = ["Region", "Leg", "Unit", "Year"]
    nas = nas.groupby(leg_key + ["Year_loaded_from_file"], as_index=False)["NAS"].sum()

    # then the newest vintage that has the year
    nas = (nas.sort_values("Year_loaded_from_file", kind="mergesort")
              .drop_duplicates(subset=leg_key, keep="last")
              .sort_values(leg_key, kind="mergesort"))

    psc = nas["Leg"].values == "PSC"
    return pd.DataFrame({
        "Region":     nas["Region"].values,
        "Type":       np.where(psc, "PSC", "Cathode"),
        "Year":       nas["Year"].values,
        "Value":      nas["NAS"].values,
        "Process":    np.where(psc, "Cathode_Pool", nas["Leg"].values),
        "Supply_Use": np.where(psc, "Supply", "Use"),
        "Unit":       nas["Unit"].values,
        "Year_loaded_from_file": nas["Year_loaded_from_file"].values,
    })[FLOW_COLS]


def main():
    # -- 1. Load the combined levels and flows of icsg_comb.py ------------------
    flows_raw = pd.read_csv(f"{CLEAN_DIR}/icsg_flows.csv")
    levels    = pd.read_csv(f"{CLEAN_DIR}/icsg_stocks.csv")

    print(f"Loaded stock levels: {levels.shape}  |  flows: {flows_raw.shape}")

    # -- 2. NAS onto the flows schema ------------------------------------------
    nas_mapped = net_addition_to_stock(levels)

    print(f"\nType / Process distribution in NAS:")
    print(nas_mapped.groupby(["Type", "Process"]).size().reset_index(name="count").to_string(index=False))

    # -- 3. Concatenate with flows ---------------------------------------------
    combined = pd.concat([flows_raw, nas_mapped], ignore_index=True)

    print(f"\nCombined shape: {combined.shape}")

    # in the case where there are multiple loaded instances (different year loaded from file)
    # take always the most recent version
    combined = combined.sort_values("Year_loaded_from_file",
                                    kind="mergesort").drop_duplicates(
        subset=["Region", "Type", "Year", "Process", "Supply_Use", "Unit"],
        keep="last"
    ).reset_index(drop=True)

    assert not combined.duplicated(subset=["Region", "Type", "Year", "Process", "Supply_Use", "Unit"]).any(), \
        "Duplicate rows found after concatenation and deduplication!"

    # -- 4. Enforce consistent dtypes ------------------------------------------
    str_cols = ["Region", "Type", "Process", "Supply_Use", "Unit"]
    for col in str_cols:
        combined[col] = combined[col].astype(str).str.strip()

    combined["Year"]                  = combined["Year"].astype(int)
    combined["Value"]                 = combined["Value"].astype(float)
    combined["Year_loaded_from_file"] = combined["Year_loaded_from_file"].astype(int)

    print("\nFinal dtypes:")
    print(combined.dtypes)
    print(f"\nProcess values:    {combined['Process'].unique()}")
    print(f"Supply_Use values: {combined['Supply_Use'].unique()}")
    print(f"Year range:        {combined['Year'].min()} - {combined['Year'].max()}")

    # -- 5. Save outputs -------------------------------------------------------
    nas_mapped.to_csv(f"{CLEAN_DIR}/icsg_nas.csv", index=False)
    print(f"\nSaved: {CLEAN_DIR}/icsg_nas.csv  ({len(nas_mapped)} rows)")

    combined.to_csv(f"{CLEAN_DIR}/icsg_flows_comb.csv", index=False)
    print(f"Saved: {CLEAN_DIR}/icsg_flows_comb.csv  ({len(combined)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
