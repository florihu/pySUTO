"""

This script is used to make functions that convert input data into a standardized format.

"""

import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import re


from pysuto.logging import get_logger


l = get_logger(__name__)


# -------------------------
# Helper: parse copper content
# -------------------------
def parse_content(val):
    """
    Convert Soulier Est. Content to fraction (float).
    Supports ranges or multiple numbers by taking the average.
    """
    if pd.isna(val):
        return np.nan
    nums = [float(n) for n in re.findall(r"\d+\.?\d*", str(val))]
    return np.mean(nums)/100 if nums else np.nan

# -------------------------
# Load Soulier copper concentration
# -------------------------
def load_copper_concentration(path):
    l.info("Loading Soulier copper concentration from %s", path)
    df = pd.read_csv(path)
    df.columns = df.columns.str.replace(' ', '_')
    df['HS92_Code'] = df['HS92_Code'].astype(int)
    df['Conc'] = df['Est._Content'].apply(parse_content)
    # keep only HS92_Code, concentration, and Root_sector
    df = df[['HS92_Code','Conc','Root_sector']]
    l.info("Loaded %d HS codes with copper concentration", len(df))
    return df.set_index('HS92_Code')

# -------------------------
# Load BACI files efficiently
# -------------------------
def load_baci_data(baci_path, hs_conc, chunksize=1_000_000):
    """
    Reads BACI CSV files in chunks, filters HS codes with copper,
    calculates Copper_flow in kt, and returns concatenated DataFrame.
    """
    cols_to_use = ['t','k','i','j','q']
    hs_codes = hs_conc.index.tolist()
    dfs = []

    l.info("Processing BACI files in %s", baci_path)
    for f in os.listdir(baci_path):
        if not f.startswith('BACI'):
            continue
        file_path = os.path.join(baci_path, f)
        l.info("Reading file: %s", f)

        for chunk_idx, chunk in enumerate(pd.read_csv(file_path, usecols=cols_to_use, chunksize=chunksize), 1):
            rows_before = len(chunk)
            # keep only HS codes with copper
            chunk = chunk[chunk['k'].isin(hs_codes)]
            if chunk.empty:
                continue
            # merge with concentration and Root_sector
            chunk = chunk.merge(hs_conc, left_on='k', right_index=True, how='left')
            # compute Copper flow in kt
            chunk['Copper_flow'] = chunk['q'] * chunk['Conc'] / 1000
            # drop zero or missing copper
            chunk = chunk[chunk['Copper_flow'] > 0]
            rows_after = len(chunk)
            l.info(
                "File %s, chunk %d: %d rows loaded, %d rows kept after copper filter",
                f, chunk_idx, rows_before, rows_after
            )
            if not chunk.empty:
                dfs.append(chunk)

    df = pd.concat(dfs, ignore_index=True)
    # rename columns for clarity
    df = df.rename(columns={
        't':'Year',
        'k':'HS92_Code',
        'i':'Region_origin',
        'j':'Region_destination',
        'Root_sector':'Sector'
    })
    # convert repetitive string columns to categorical
    for col in ['Region_origin','Region_destination','Sector']:
        df[col] = df[col].astype('category')

    l.info("Finished loading all BACI files. Total rows: %d", len(df))
    return df

# -------------------------
# Aggregate to tidy base table
# -------------------------
def baci_to_base(df):
    """
    Aggregate BACI copper flows to tidy table with columns:
    Year, Sector, Region_from, Region_to, Copper Flow Value
    """
    tidy = df.groupby(['Year','Sector','Region_origin','Region_destination'], dropna=False)['Copper_flow'].sum().reset_index()
    tidy = tidy.rename(columns={
        'Region_origin':'Region_from',
        'Region_destination':'Region_to'
    })
    # keep only rows with valid sector and positive copper flow
    tidy = tidy[tidy['Sector'].notna() & (tidy['Copper_flow'] > 0)]
    return tidy

# -------------------------
# Convert region codes to names
# -------------------------
def reg_conv(df, rconv):
    """
    Merge DataFrame with region conversion table to add region names.
    """
    df = df.merge(rconv, left_on='Region_from', right_on='country_code', how='left')
    df = df.rename(columns={'country_name':'Region_from_name'}).drop(columns=['country_code'])
    df = df.merge(rconv, left_on='Region_to', right_on='country_code', how='left')
    df = df.rename(columns={'country_name':'Region_to_name'}).drop(columns=['country_code'])

    return df

# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":
    of = r'data/proc/datafeed/baci'

    baci_path = r'data/input/raw/BACI_HS92_V202601'
    hs_conc_path = r'data/input/raw/Soulier_2018/Copper_HS92_Table_Merged.csv'
    reg_conv_path = r'data/input/raw/BACI_HS92_V202601/country_codes_V202601.csv'

    l.info("Starting BACI -> copper base mapping script")

    # Load copper concentrations
    hs_conc = load_copper_concentration(hs_conc_path)

    # Load region conversion table
    reg_df = pd.read_csv(reg_conv_path)[['country_code', 'country_name']]

    # Load BACI files and compute copper flows
    df = load_baci_data(baci_path, hs_conc)

    # Aggregate to tidy table
    tidy_df = baci_to_base(df)

    # Convert region codes to names
    tidy_df = reg_conv(tidy_df, reg_df)

    # Save output
    os.makedirs(of, exist_ok=True)
    out_path = os.path.join(of, 'baci_copper_root.csv')
    tidy_df.to_csv(out_path, index=False)

    l.info("Done! Output saved to: %s", out_path)