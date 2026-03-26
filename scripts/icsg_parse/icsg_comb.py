import pandas as pd
import numpy as np
import re
import os


from pysuto.utils import get_path
from pysuto.logging import get_logger



REG_REN = {
    # United States
    "U.S.": "United States",
    "United": "United States",

    # Korea variants
    "Korean": "South Korea",
    "Korean Rep.": "South Korea",
    "Korean rep.": "South Korea",
    "North Korea": "North Korea",

    # Congo variants
    "Congo": "Congo Rep.",
    "Congo Rep.": "Congo Rep.",
    "Congo DR": "Congo, D.R.",
    "Congo, D.R.": "Congo, D.R.",

    # Czech
    "Czech Rep.": "Czech Rep.",
    "Czech Republic": "Czech Rep.",

    # Dominican Republic
    "Dominican Rep.": "Dominican Rep.",
    "Dominican Republic": "Dominican Rep.",

    # Russia
    "Russian Fed.": "Russia",

    # Serbia / Montenegro
    "Serbia & Montenegro": "Serbia and Montenegro",

    # Macedonia naming
    "Macedonia": "North Macedonia",

    # Belgium-Luxembourg variants
    "Belgium- Luxembourg": "Belgium-Luxembourg",
    "Belgium-Luxembourg": "Belgium-Luxembourg",

    # Case issues
    "kyrgyzstan": "Kyrgyzstan",

    # Ambiguous / incomplete
    "South": None,       # probably incomplete entry
    "Others": None,      # not a country

    # optional normalization
    "Scandinavia": "Scandinavia"
}


PRICE_RENAME = {
# COMEX prices
"COMEX HG, 1st Pos.": "COMEX_1st",
"COMEX 1M Settlement": "COMEX_1M",

# LME prices
"LME Grade A, Cash": "LME_Cash",
"LME Settlement": "LME_Settlement",
"LME 3M Offer": "LME_3M",

# U.S. producer prices
"U.S. Producer Price": "US_Prod",
"U Producer Price": "US_Prod",

# Shanghai Futures Exchange
"SHFE Cash": "SHFE_Cash"
}

UNIT_MAP = {
    "LME Grade A, Cash": "$/metric tonne",
    "LME Settlement": "$/metric tonne",
    "LME 3M Offer": "$/metric tonne",
    "COMEX HG, 1st Pos.": "cents/pound",
    "COMEX 1M Settlement": "cents/pound"
}

UNIT_RENAME = {
    # LME / USD per metric tonne
    "$/metric tonne": "USD/t",
    "US$/ metric tonne": "USD/t",
    
    # COMEX copper / cents per pound
    "Cents/pound": "cents/lb",
    "cents/pound": "cents/lb",

    # Shanghai / Yuan per metric tonne
    "Yuan/ metric tonne": "CNY/t"
}



#stored as (Flow, Process)
# already stored in SXEW internal fuck up by ICSG
SKIP_PAIRS = [('SX-EW', 'Mining'),
              ('Mine SX-EW', 'Mining'),
              ('SX-EW', 'Refining'),
              ('Refinery SX-EW', 'Refining'),
              ('Electrowon', 'Refining')
              ]
#indexed flow old, process old: flow new
REN_TYPE ={('Low-grade', 'Smelting'):'Primary',
           ('Smelter Primary', 'Smelting'): 'Primary',
           ('Smelter Second.', 'Smelting'): 'Secondary',

           ('Refinery Second.','Refining'): 'Secondary',
           ('Refinery Primary', 'Refining'): 'Primary',
           ('Cu Alloy', 'Semi'): 'Copper Alloy'
        
             }


def type_ren(s, f):
    """Clean types and rename flow types."""

    # remove missing types in stocks
    s = s[s['Type'].notna()].copy()

    skip_set = set(SKIP_PAIRS)

    # drop unwanted (Type, Process) pairs
    mask = ~pd.MultiIndex.from_frame(f[['Type','Process']]).isin(skip_set)
    f = f.loc[mask].copy()

    # rename flow types
    key = list(zip(f['Type'], f['Process']))
    f['Type'] = [REN_TYPE.get(k, t) for k, t in zip(key, f['Type'])]

    return s.reset_index(drop=True), f.reset_index(drop=True)






def creg(s, f):
    """Standardize region names."""

    s['Region'] = s['Region'].replace(REG_REN)
    f['Region'] = f['Region'].replace(REG_REN)

    s = s.dropna(subset=['Region'])
    f = f.dropna(subset=['Region'])

    return s, f


def stot(sraw):
    """
    Standardize 'Total' Type entries in stocks.
    """

    group_keys = ['Region', 'Year', 'Stock_type']

    type_counts = sraw.groupby(group_keys)['Type'].transform('nunique')
    is_total = sraw['Type'].eq('Total')

    # rename if only entry
    sraw.loc[is_total & (type_counts == 1), 'Type'] = 'Producers'

    # drop redundant totals
    s = sraw.loc[~(is_total & (type_counts > 1))]

    return s.reset_index(drop=True)

def pcl(p):
    """Clean price dataframe."""

    p['Type'] = p['Exchange_type'].replace(PRICE_RENAME)

    mask = p['Unit'].isna()
    p.loc[mask, 'Unit'] = p.loc[mask, 'Exchange_type'].map(UNIT_MAP)

    p = p.drop(columns='Exchange_type')

    cols = ['Type', 'Year', 'Value', 'Value_type', 'Unit', 'Year_loaded_from_file']
    p = p[cols]

    p['Unit'] = p['Unit'].replace(UNIT_RENAME)

    return p


def remove_total_if_specific_exists(df):
    """Drop Total when specific types exist."""

    group_cols = ['Region', 'Process', 'Year', 'Supply_Use']

    has_specific = (
        df['Type'].ne('Total')
        .groupby(df[group_cols].apply(tuple, axis=1))
        .transform('any')
    )

    df = df.loc[~((df['Type'] == 'Total') & has_specific)]

    return df.reset_index(drop=True)


def main(ys=[2005, 2014, 2023, 2025],
         of=r'data/proc/datafeed/icsg_clean'):

    ps, ss, fs = [], [], []
    base = r'data/proc/datafeed/icsg_clean'

    for y in ys:

        p = pd.read_csv(os.path.join(base, f'icsg_{y}_prices.csv'))
        p['Year_loaded_from_file'] = y
        ps.append(p)

        f = pd.read_csv(os.path.join(base, f'icsg_{y}_flows.csv'))
        f['Year_loaded_from_file'] = y
        fs.append(f)

        s = pd.read_csv(os.path.join(base, f'icsg_{y}_stocks.csv'))
        s['Year_loaded_from_file'] = y
        ss.append(s)

    s = pd.concat(ss, ignore_index=True)
    f = pd.concat(fs, ignore_index=True)
    p = pd.concat(ps, ignore_index=True)

    # ---------------- clean flows + stocks ----------------

    s, f = creg(s, f)
    s, f = type_ren(s, f)

    s = stot(s)
    f = remove_total_if_specific_exists(f)

    # types
    s['Year'] = s['Year'].astype(int)
    f['Year'] = f['Year'].astype(int)

    s['Year_loaded_from_file'] = s['Year_loaded_from_file'].astype(int)
    f['Year_loaded_from_file'] = f['Year_loaded_from_file'].astype(int)

    s['Value'] = s['Value'].astype(float)
    f['Value'] = f['Value'].astype(float)

    # dedup keys
    s_key = [c for c in s.columns if c not in ['Year_loaded_from_file', 'Value']]
    f_key = [c for c in f.columns if c not in ['Year_loaded_from_file', 'Value']]

    s = (
        s.sort_values('Year_loaded_from_file')
        .drop_duplicates(subset=s_key, keep='last')
        .reset_index(drop=True)
    )

    f = (
        f.sort_values('Year_loaded_from_file')
        .drop_duplicates(subset=f_key, keep='last')
        .reset_index(drop=True)
    )

    assert not s.duplicated(subset=s_key).any()
    assert not f.duplicated(subset=f_key).any()

    # ---------------- prices ----------------

    p = pcl(p)

    p_key = [c for c in p.columns if c != 'Year_loaded_from_file']

    p = (
        p.sort_values('Year_loaded_from_file')
        .drop_duplicates(subset=p_key, keep='last')
        .reset_index(drop=True)
    )

    assert not p.duplicated(subset=p_key).any()

    # ---------------- save ----------------

    s.to_csv(os.path.join(of, 'icsg_stocks.csv'), index=False)
    f.to_csv(os.path.join(of, 'icsg_flows.csv'), index=False)
    p.to_csv(os.path.join(of, 'icsg_prices.csv'), index=False)

if __name__ == "__main__":
    main()