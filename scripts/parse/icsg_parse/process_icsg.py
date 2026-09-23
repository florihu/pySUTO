"""
ICSG Stock & Flow Processing Script
------------------------------------
Steps:
1. Calculate net addition to stock (NAS) from ICSG stocks data
   - Year 1995 is "year 0": its value is NOT differenced — stock addition for 1996
     is simply stocks[1996] - stocks[1995]  (i.e. 1995 acts as the baseline)
   - For all subsequent years: NAS(t) = stocks(t) - stocks(t-1)
2. Map stock columns onto flow schema (Region, Type, Year, Value, Process, Supply_Use, Unit, Year_loaded_from_file)
   - Combined type key (Stock_type + "_" + Type) is remapped via RENAME_DICT to PSC / MESC / USC
   - Column mapping depends on remapped type:
       PSC  -> Type = "PSC",     Process = "Cathode_Pool", Supply_Use = "Supply"
       MESC -> Type = "Cathode", Process = "MESC",         Supply_Use = "Supply"
       USC  -> Type = "Cathode", Process = "USC",          Supply_Use = "Supply"
3. Concatenate NAS rows with the original flows dataframe
4. Enforce consistent dtypes: Region/Type/Process/Supply_Use/Unit -> str, Year -> int, Value -> float
5. Save results
"""
 
import pandas as pd
import numpy as np
 
# -- 1. Load raw data ----------------------------------------------------------
flows_raw  = pd.read_csv("data/proc/feedprep/icsg_clean/icsg_flows.csv")
stocks_raw = pd.read_csv("data/proc/feedprep/icsg_clean/icsg_stocks.csv")
 
print(f"Loaded stocks: {stocks_raw.shape}  |  flows: {flows_raw.shape}")
 
# -- 2. Calculate Net Addition to Stock (NAS) ----------------------------------
stocks_sorted = stocks_raw.sort_values(
    ["Region", "Stock_type", "Type", "Unit", "Year"]
).reset_index(drop=True)
 
group_keys = ["Region", "Stock_type", "Type", "Unit", "Year_loaded_from_file"]
 
stocks_sorted["NAS"] = stocks_sorted.groupby(group_keys)["Value"].diff()
 
# Drop 1995 baseline rows (NaN diff -- no prior year)
nas_df = stocks_sorted.dropna(subset=["NAS"]).copy()
nas_df = nas_df.rename(columns={"NAS": "Value_NAS"})
 
print(f"NAS rows after dropping 1995 baseline NaNs: {len(nas_df)}")
print(f"Years covered in NAS: {sorted(nas_df['Year'].unique())}")
 
# Build combined type key, then remap to PSC / MESC / USC
nas_df["Type_key"] = nas_df["Stock_type"] + "_" + nas_df["Type"]
 
RENAME_DICT = {
    "PMC_Producers":          "PSC",
    "PMC_Producers/Merchants":"PSC",
    "Exchanges_LME":          "MESC",
    "Exchanges_SHFE":         "MESC",
    "Exchanges_COMEX":        "MESC",
    "PMC_Consumers":          "USC",
    "PMC_Merchants":          "USC",
    "PMC_Government":         "USC",
}
nas_df["Type_key"] = nas_df["Type_key"].replace(RENAME_DICT)
 
# -- 3. Map onto the flows schema (type-dependent) ----------------------------
#
# PSC:        Type = "PSC",     Process = "Cathode_Pool", Supply_Use = "Supply"
# MESC / USC: Type = "Cathode", Process = <type_key>,     Supply_Use = "Use"
 
nas_mapped = pd.DataFrame({
    "Region": nas_df["Region"].values,
    "Type":   np.where(
                  nas_df["Type_key"].values == "PSC",
                  "PSC",
                  "Cathode"
              ),
    "Year":   nas_df["Year"].values,
    "Value":  nas_df["Value_NAS"].values,
    "Process": np.where(
                   nas_df["Type_key"].values == "PSC",
                   "Cathode_Pool",
                   nas_df["Type_key"].values   # "MESC" or "USC"
               ),
    "Supply_Use":  np.where(
                        nas_df["Type_key"].values == "PSC",
                        "Supply",
                        "Use"
                    ),
    "Unit":                  nas_df["Unit"].values,
    "Year_loaded_from_file": nas_df["Year_loaded_from_file"].values,
})
 
print("\nSample NAS mapped rows:")
print(nas_mapped.head(8).to_string(index=False))
print(f"\nType / Process distribution in NAS:")
print(nas_mapped.groupby(["Type", "Process"]).size().reset_index(name="count").to_string(index=False))
 
# -- 4. Concatenate with flows -------------------------------------------------
combined = pd.concat([flows_raw, nas_mapped], ignore_index=True)
 
print(f"\nCombined shape: {combined.shape}")


# in the case where there are multiple loaded instances (different year loaded from file)
# take always the most recent version
combined = combined.sort_values("Year_loaded_from_file").drop_duplicates(
    subset=["Region", "Type", "Year", "Process", "Supply_Use", "Unit"],
    keep="last"
).reset_index(drop=True)

assert not combined.duplicated(subset=["Region", "Type", "Year", "Process", "Supply_Use", "Unit"]).any(), \
    "Duplicate rows found after concatenation and deduplication!"
 
# -- 5. Enforce consistent dtypes ---------------------------------------------
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
 
# -- 6. Save outputs -----------------------------------------------------------
out_dir = "data/proc/feedprep/icsg_clean"
 
nas_mapped.to_csv(f"{out_dir}/icsg_stocks.csv", index=False)
print(f"\nSaved: {out_dir}/icsg_stocks.csv  ({len(nas_mapped)} rows)")
 
combined.to_csv(f"{out_dir}/icsg_flows_comb.csv", index=False)
print(f"Saved: {out_dir}/icsg_flows_comb.csv  ({len(combined)} rows)")
 
print("\nDone.")
 
