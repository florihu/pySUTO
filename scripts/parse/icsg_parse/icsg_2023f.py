"""icsg_2023f.py
=============
ICSG 2023 and 2025 yearbooks (one layout): sheet-to-flow parser.

Values are kt Cu: the parse stamps ``Unit = 'kt'`` and the icsg feed's ``valuation_map``
reads it as CC (contained copper), not gross mass.
"""
import pandas as pd
import numpy as np
import re
import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
from icsg_paths import get_path
from pysuto.logging import get_logger



logger = get_logger(__name__)
out_folder = r'data/proc/feedprep/icsg_clean'


REGION_RENAME = {
        # typos
        'Philippppines':    'Philippines',
        'OthOtheersrs':     'Others',
        'kyrgyzstan':       'Kyrgyzstan',
        # footnote suffixes
        'Germany 1/':       'Germany',
        'Zambia 2/':        'Zambia',
        'Zambia 3/':        'Zambia',
        'Congo, D.R.  1/':  'Congo, D.R.',
        'Chile 1/':         'Chile',
        'China 3/ 9/':      'China',
        'United States 1/': 'United States',
        'Others 6/':        'Others',
        'Benelux 3/':        'Benelu',
        # name consolidation
        'U.S.':             'United States',
        'Russian Fed.':     'Russia',
        'Czech Rep.':       'Czech Republic',
        'Dominican Rep.':   'Dominican Republic',
        'Korean Rep.':      'South Korea',
        'Korean rep.':      'South Korea',
        'Korean':           'South Korea',
        # Belgium variants
        'Belgium-':             'Belgium-Luxembourg',
        'Belgium- Luxembourg':  'Belgium-Luxembourg',
        # Taiwan
        'Taiwan (China)':   'Taiwan',
        'Taipei, China':    'Taiwan',
        # Congo
        'Congo DR':         'Congo, D.R.',
        'Congo Rep.':       'Congo',
        # Serbia
        'Serbia & Montenegro': 'Serbia',
        # Macedonia
        'Macedonia':        'North Macedonia',
        # ambiguous truncations -> drop
        'South':            None,
        'United':           None,
    }



def icsg_loader(file_name = r'ICSG 2023 Statistical Yearbook.xlsx', id=2023):

    '''
    This loader is valid for released Excel files starting from 2023
    '''
    path = get_path(file_name)

    # Define the list as [Process, Supply/Use]
    sheet_dict = {
        '3.Mine': ['Mining', 'Supply'],
        '4.Sml': ['Smelting', 'Supply'],
        '5.Ref': ['Refining', 'Supply'],
        '6.SXEW': ['SXEW', 'Supply'],
        '7.Usage': ['Semi', 'Use'],
        '10.Semis': ['Semi', 'Supply'],
       # '9.Price': ('Price', None, None), 
       # '8.Stock': ('Stock', None, None),
    }

    rename_dict = {
        'COUNTRY': 'Region',
        'Source': 'Type',
        'Feed': 'Type',
        'Semis': 'Type',
    }

    

    stck = pd.read_excel(path, sheet_name='8.Stock', header=2)
    stckcl = clean_stock(stck)
    

    p = pd.read_excel(path, sheet_name='9.Price')
    pc = clean_prices_struc(p)


    fcol = []

    for sheet, name in sheet_dict.items():

        process = name[0]
        flow_type = name[1]

        df_init = pd.read_excel(path, sheet_name=sheet, header=2)

        df = load_flows_struc(df_init, sheet, process, flow_type, rename_dict, name=sheet)
        
        fcol.append(df)
    

    flows = pd.concat(fcol, ignore_index=True)


    #save
    os.makedirs(out_folder, exist_ok=True)
    flow_path = os.path.join(out_folder, f'icsg_{id}_flows.csv')
    stock_path = os.path.join(out_folder, f'icsg_{id}_stocks.csv')
    price_path = os.path.join(out_folder, f'icsg_{id}_prices.csv')
    flows.to_csv(flow_path, index=False)
    stckcl.to_csv(stock_path, index=False)
    pc.to_csv(price_path, index=False)


    
def clean_prices_struc(df_raw):

    df = df_raw.copy()

    # real exchange column names
    exchange_cols = df.iloc[1, 2:5].tolist()

    df = df.iloc[:, :5]
    df.columns = ["section", "year"] + exchange_cols

    current_section = None
    rows = []

    value_map = {
        "ANNUAL AVERAGES": "Average",
        "ANNUAL HIGHS": "High",
        "ANNUAL LOWS": "Low"
    }

    for _, r in df.iterrows():

        if r["section"] in value_map:
            current_section = value_map[r["section"]]
            continue

        if pd.notna(r["year"]) and current_section:
            year = int(r["year"])

            for exch in exchange_cols:
                price = r[exch]

                if pd.notna(price):
                    rows.append({
                        "Year": year,
                        "Exchange_type": exch,
                        "Value_type": current_section,
                        "Value": float(price)
                    })
    df  = pd.DataFrame(rows)

    return df



def clean_stock(df_raw):

    df = df_raw.copy()

    # keep the label column and the year columns: the 2023 sheet ends with an
    # averages column, the 2025 sheet with its last year (2024) (#35)
    year_cols = [c for c in df.columns[1:] if re.match(r"^\d{4}$", str(c))]
    df = df[[df.columns[0]] + year_cols]

    #Lets start with stock type = Exchange
    # includ in th thr row SHFE thats the first part
    
    shfeidx = df[df.iloc[:, 0].str.contains('SHFE', na=False)].index[0]
    lmeidx = df[df.iloc[:, 0].str.contains('LME', na=False)].index[0]


    ex = df.iloc[lmeidx+1:shfeidx+1, :].copy()
    #rename first column in type_help
    ex = ex.rename(columns={ex.columns[0]: 'Type'})
    ex['Region'] = ex['Type'].str.strip()
    ex['Region'] = ex['Region'].replace({'COMEX': 'United States', 'SHFE': 'China'})
    ex['Stock_type'] = 'Exchanges'
    #in type col replace all country names with lme
    ex['Type'] = ex['Type'].where(ex['Type'].str.contains(r'COMEX|SHFE', na=False), other='LME')
    
    # long ex with it cols type region
    ex_long = ex.melt(id_vars=['Type', 'Region', 'Stock_type'], var_name='Year', value_name='Value')
    ex_long['Unit'] = 'kt'
    ex_long['Year'] = ex_long['Year'].astype(int)
    ex_long['Value'] = ex_long['Value'].astype(float)
    #------------------------------------------------------------

    #PMC

    #-------------------------------------------------------------

    # the second is from shfe (not included) to REFINED STOCK SUMMARY
    refidx = df[df.iloc[:, 0].str.contains('REFINED STOCK SUMMARY', na=False)].index[0]
    pmc = df.iloc[shfeidx+2:refidx, :].copy()
    #rename first col to type
    pmc = pmc.rename(columns={pmc.columns[0]: 'Type'})
    # the region is the whole label before its type, its total or its footnote:
    # the first word of 'South Africa (producers) 1/' is not a region (#35)
    pmc['Region'] = (pmc['Type'].str.replace(r'\s*\(.*$', '', regex=True)
                                .str.replace(r'\s+Total\b.*$', '', regex=True)
                                .str.replace(r'\s+\d+/.*$', '', regex=True)
                                .str.strip())
    # type is everything that is in brackets ()
    pmc['Type'] = pmc['Type'].str.extract(r'\((.*?)\)', expand=False).str.strip()

    # replace nan, 'only total data since 2021'  with total
    pmc['Type'] = pmc['Type'].replace({np.nan: 'Total', 'only total data since 2021': 'Total'})
    # tall letters for all type instaces
    pmc['Type'] = pmc['Type'].str.title()
    pmc['Stock_type'] = 'PMC'    

    #melt id cols type region stock type
    pmc_long = pmc.melt(id_vars=['Type', 'Region', 'Stock_type'], var_name='Year', value_name='Value')
    pmc_long['Unit'] = 'kt'
    pmc_long['Year'] = pmc_long['Year'].astype(int)
    pmc_long['Value'] = pmc_long['Value'].astype(float)

     #concat ex and pmc
    df = pd.concat([ex_long, pmc_long], ignore_index=True)


    return df

def load_flows_struc(df_init, sheet, process, flow_type, rename_dict, name):

    df = df_init.copy()

    # ── 1. process-level type overrides ──────────────────────────────────────
    if process == 'SXEW':
        df['Type'] = 'Cathode'
    elif (process == 'Semi') and (flow_type == 'Use'):
        df['Type'] = 'Cathode'

    # ── 2. find source row ───────────────────────────────────────────────────
    source_rows = df[df.iloc[:, 0].astype(str).str.contains('Source:', na=False)]
    if not source_rows.empty:
        source_row = source_rows.index[0]
    else:
        source_row = len(df)

    # ── 3. drop unnamed columns ──────────────────────────────────────────────
    df = df.loc[:, ~df.columns.astype(str).str.contains('Unnamed', na=False)]

    # ── 4. special sheet cleaning ────────────────────────────────────────────
    if name == 'Semi_use':
        df = df[~df.iloc[:, 0].astype(str).str.contains('EUROPEAN UNION \+ UK 6/', na=False)]

    # ── 5. remove header/footer rows ─────────────────────────────────────────
    df = df.iloc[1:source_row, :].copy()

    # ── 6. rename columns ────────────────────────────────────────────────────
    df = df.rename(columns=rename_dict)

    # ── 7. drop aggregated regions ───────────────────────────────────────────
    AGG_PATTERN = r'\b(TOTALS?|WORLD|Other|OthOtheersrs|EUROPEAN|EU|By)\b'
    df = df[~df['Region'].astype(str).str.contains(AGG_PATTERN, case=False, na=False)]

    # ── 8. remove missing regions ────────────────────────────────────────────
    df = df[df['Region'].notna()]

    # ── 9. remove 'Not' rows but keep Total (handled later) ───────────────────
    df = df[~df['Type'].astype(str).str.contains('Not', na=False)]

    # ── 10. forward-fill merged regions ──────────────────────────────────────
    df['Region'] = df['Region'].replace('', np.nan).bfill()

    # ── 11. reshape to long format ───────────────────────────────────────────
    df = df.melt(id_vars=['Region', 'Type'], var_name='Year', value_name='Value')

    # ── 12. attach metadata ──────────────────────────────────────────────────
    df['Process'] = process
    df['Unit'] = 'kt'
    df['Supply_Use'] = flow_type

    # ── 13. clean year/value types ───────────────────────────────────────────
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    # ── 14. clean whitespace ─────────────────────────────────────────────────
    df['Region'] = df['Region'].str.strip()
    df['Type'] = df['Type'].astype(str).str.strip()

    # ── 15. remove missing values ─────────────────────────────────────────────────
    # Zeros are kept: a later yearbook's revision to zero must supersede an
    # older vintage's value in icsg_comb, which drops them after its dedup (#111).
    df = df[df['Value'].notna()]

    # ── 16. conditional removal of Total rows ─────────────────────────────────
    total_mask = df['Type'].str.contains(r'\bTotal\b', case=False, na=False)

    has_specific_value = (
        (~total_mask) & df['Value'].notna() & (df['Value'] != 0)
    ).groupby([df['Region'], df['Year']]).transform('any')

    df = df[~(total_mask & has_specific_value)]

    # –– 17. region renaming with validation ──────────────────────────────────────
    df['Region'] = df['Region'].replace(REGION_RENAME)

    # ── 17. final reset ──────────────────────────────────────────────────────
    df = df.reset_index(drop=True)

    return df

if __name__ == "__main__":

  file_name = r'ICSG 2025 Statistical Yearbook.xlsx'
  icsg_loader(file_name, id = 2025)
  file_name = r'ICSG 2023 Statistical Yearbook.xlsx'
  icsg_loader(file_name, id = 2023)