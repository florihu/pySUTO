import pandas as pd
import numpy as np
import re
import os


from pysuto.utils import get_path
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
    ex['Region'].replace({'COMEX': 'United States', 'SHFE': 'China'}, inplace=True)
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















# import os
# import pandas as pd
# import logging

# logger = logging.getLogger(__name__)

# def read_icsg_all_years(out_path=r'data/proc/datafeed/icsg', 
#                         conc_path=r'data/input/conc/icsg.xlsx'):

#     # --- Collect data from different years ---
#     df_2023 = icsg_2023()
#     df_2014 = icsg_2014()
#     df_2005 = icsg_2005()

#     # --- Load base concordances ---
#     conc = pd.read_excel(conc_path, sheet_name=None)
#     conc_sec = conc['Sector_int']
#     conc_reg = conc['Region']

#     # --- Assert column consistency ---
#     assert set(df_2023.columns) == set(df_2014.columns) == set(df_2005.columns), \
#         "DataFrames have different columns"

#     # --- Combine all data ---
#     df = pd.concat([df_2023, df_2014, df_2005], ignore_index=True)
#     df['Region'] = df['Region'].str.strip()

#     # --- Region renaming ---
#     reg_rename = {
#         'Azerbeijan': 'Azerbaijan',
#         'North Macedonia': 'Macedonia',
#         'Belgium-': 'Belgium-Luxembourg',
#         'Belgium- Luxembourg': 'Belgium-Luxembourg',
#         'kyrgyzstan': 'Kyrgyzstan',
#         'Congo Rep.': 'Congo',
#         'Congo DR': 'Congo, D.R.', 
#         'Philippppines': 'Philippines',
#         'United': 'United States',
#         'Zambia  ': 'Zambia',
#     }

#     df.replace({'Region': reg_rename}, inplace=True)

#     # --- Check for unknown regions ---
#     missing_regions = set(df['Region'].unique()) - set(conc_reg['Source_name'].unique())
#     if missing_regions:
#         logger.warning(f"Regions in ICSG data not found in base classification: {missing_regions}")

#     # --- Build flow-process mapping ---
#     cdict = {
#         (flow_root, process_root): (flow_base, process_base)
#         for flow_root, process_root, flow_base, process_base in zip(
#             conc_sec['Flow_root'],
#             conc_sec['Process_root'],
#             conc_sec['Flow_base'],
#             conc_sec['Process_base']
#         )
#     }

#     # --- Detect missing mappings before applying ---
#     df_pairs = set(df[['Flow', 'Process']].itertuples(index=False, name=None))
#     mapping_keys = set(cdict.keys())
#     missing_pairs = df_pairs - mapping_keys

#     if missing_pairs:
#         missing_preview = list(missing_pairs)[:10]
#         raise ValueError(
#             f"The following (Flow, Process) combinations are not in the renaming dictionary: "
#             f"{missing_preview}{' ...' if len(missing_pairs) > 10 else ''}"
#         )

#     # --- Replace (Flow, Process) pairs efficiently ---
#     mapped = df[['Flow', 'Process']].apply(
#         lambda x: cdict.get((x['Flow'], x['Process'])),
#         axis=1,
#         result_type='expand'
#     )

#     df[['Flow', 'Process']] = mapped

#     # --- Save output ---
#     stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
#     os.makedirs(out_path, exist_ok=True)
#     out_file = os.path.join(out_path, f'ICSG_raw_clean_{stamp}.csv')
#     df.to_csv(out_file, index=False)

#     # --- Final consistency check ---
#     check_consistency(df)
#     logger.info(f"ICSG data saved to {out_file}")

#     return df


# def check_consistency(df):
#     '''
#     A pair Region, Flow, Process should be unique for each year
#     '''
#     duplicates = df[df.duplicated(subset=['Region', 'Flow', 'Process', 'Year'], keep=False)]
#     if not duplicates.empty:
#         logger.warning(f"Duplicate entries found for the same Region, Flow, Process, Year combination:\n{duplicates}")


#     # there are no nan allowed in Region, Flow, Process and value
#     if df[['Region', 'Flow', 'Process', 'Value']].isnull().any().any():
#         logger.warning("NaN values found in Region, Flow, Process, or Value columns")
#     if (df['Value'] < 0).any():
#         logger.warning("Negative values found in Value column")

# def calc_missing_flows(tc_path = r'data\input\raw\TCs.xlsx',
#                             df_path = r'data\proc\datafeed\icsg\ICSG_raw_clean_20251007_115132.csv',
#                             conc_path = r'data\input\raw\Conc.xlsx'):

#     tc = pd.read_excel(tc_path)
#     df = pd.read_csv(df_path)
#     conc = pd.read_excel(conc_path)


#     not_needed_flows = ['Wire_rod', 'Copper', 'Copper_alloy']
#     year = 1995

#     df = df[df['Year'] == year]
#     df = df[~df['Flow'].isin(not_needed_flows)]

#     # drop year col
#     df = df.drop(columns=['Year'])


#     df = df.groupby(['Region', 'Flow', 'Process', 'Supply_Use', 'Unit'], as_index=False)['Value'].sum()

#     # PYRO
#     # calculate waste rock - ore
#     df_ore = df[df['Flow'] == 'Concentrates'].copy()
#     df_ore['Flow'] = 'Ore'
#     df_ore['Value'] = df_ore['Value'] / tc.loc[tc['Process'] == 'Milling_Crushing_Floatation', 'Value'].iloc[0]
#     df_ore['Value'] = df_ore['Value'] / conc.loc[conc['Flow'] == 'Ore', 'Value'].iloc[0]
#     df_ore['Supply_Use'] = 'Use'
#     df_ore['Process'] = 'Milling_Crushing_Floatation'

#     # Concentrates
#     df_conc = df[df['Flow'] == 'Concentrates'].copy()
#     df_conc['Flow'] = 'Concentrates'
#     df_conc['Value'] = df_conc['Value'] / conc.loc[conc['Flow'] == 'Concentrate', 'Value'].iloc[0]
#     df_conc['Supply_Use'] = 'Supply'
#     df_conc['Process'] = 'Milling_Crushing_Floatation'
  
    
#     # Tailings
#     df_tailings = df[df['Flow'] == 'Concentrates'].copy()
#     df_tailings['Flow'] = 'Tailings'
#     df_tailings['Value'] = df_tailings['Value'] * (1/tc.loc[tc['Process'] == 'Milling_Crushing_Floatation', 'Value'].iloc[0]-1)
#     df_tailings['Value'] = df_tailings['Value'] / conc.loc[conc['Flow'] == 'Tailings', 'Value'].iloc[0]
#     df_tailings['Supply_Use'] = 'Supply'
#     df_tailings['Process'] = 'Milling_Crushing_Floatation'


#     #Waste rock
#     df_waste_rock = df_ore.copy()
#     df_waste_rock['Flow'] = 'Waste Rock'
#     df_waste_rock['Value'] = df_waste_rock['Value'] * tc.loc[tc['Process'] == 'Mining', 'Value'].iloc[0]
#     df_waste_rock['Supply_Use'] = 'Supply'
#     df_waste_rock['Process'] = 'Mining'

#     # Crude ore
#     df_crude_ore = df_ore.copy()
#     df_crude_ore['Flow'] = 'Crude Ore'
#     df_crude_ore['Value'] = df_crude_ore['Value'] + df_waste_rock['Value']
#     df_crude_ore['Supply_Use'] = 'Use'
#     df_crude_ore['Process'] = 'Mining'


#     # HYDRO
#     #Refined_copper_hydro
#     df_cathode_hydro = df[(df['Flow'] == 'Refined_primary') & (df['Process'] == 'SXEW')].copy()
#     df_cathode_hydro['Flow'] = 'Refined_copper_hydro'
#     df_cathode_hydro['Process'] = 'SXEW'
#     df_cathode_hydro['Supply_Use'] = 'Supply'
#     df_cathode_hydro['Value'] = df_cathode_hydro['Value'] / conc.loc[conc['Flow'] == 'Refined_copper_hydro', 'Value'].iloc[0]

#     # Tailings
#     df_tailings_2 = df[(df['Flow'] == 'Refined_primary') & (df['Process'] == 'SXEW')].copy()
#     df_tailings_2['Flow'] = 'Tailings'
#     df_tailings_2['Value'] = df_tailings_2['Value'] * (1/tc.loc[tc['Process'] == 'SXEW', 'Value'].iloc[0]-1)
#     df_tailings_2['Value'] = df_tailings_2['Value'] / conc.loc[conc['Flow'] == 'Tailings', 'Value'].iloc[0]
#     df_tailings_2['Supply_Use'] = 'Supply'
#     df_tailings_2['Process'] = 'SXEW'

    
#     # Ore
#     df_ore_2 = df[(df['Flow'] == 'Refined_primary') & (df['Process'] == 'SXEW')].copy()
#     df_ore_2['Flow'] = 'Ore'
#     df_ore_2['Value'] = df_ore_2['Value'] / tc.loc[tc['Process'] == 'SXEW', 'Value'].iloc[0]
#     df_ore_2['Value'] = df_ore_2['Value'] / conc.loc[conc['Flow'] == 'Ore', 'Value'].iloc[0]
#     df_ore_2['Supply_Use'] = 'Use'
#     df_ore_2['Process'] = 'SXEW'

#     #Waste rock
#     df_waste_rock_2 = df_ore_2.copy()
#     df_waste_rock_2['Flow'] = 'Waste Rock'
#     df_waste_rock_2['Value'] = df_waste_rock_2['Value'] * tc.loc[tc['Process'] == 'Mining', 'Value'].iloc[0]
#     df_waste_rock_2['Supply_Use'] = 'Supply'
#     df_waste_rock_2['Process'] = 'Mining'

#     # Crude ore
#     df_crude_ore_2 = df_ore_2.copy()
#     df_crude_ore_2['Flow'] = 'Crude Ore'
#     df_crude_ore_2['Supply_Use'] = 'Use'
#     df_crude_ore_2['Value'] = df_crude_ore_2['Value'] + df_waste_rock_2['Value']
#     df_crude_ore_2['Process'] = 'Mining'


#     df_smelting_primary = df[df['Flow'].isin(['Smelting_primary']) & (df['Process'] == 'Smelting')].copy()
#     df_smelting_primary['Flow'] = 'Concentrates'
#     # gropby concentrates and sum up -> we do not differentiate between low grade and concentrates
#     df_smelting_primary = df_smelting_primary.groupby(['Region', 'Flow', 'Process', 'Supply_Use', 'Unit']).sum().reset_index()
#     df_smelting_primary['Value'] = df_smelting_primary['Value'] / tc.loc[tc['Process'] == 'Smelting', 'Value'].iloc[0]
#     df_smelting_primary['Value'] = df_smelting_primary['Value'] / conc.loc[conc['Flow'] == 'Concentrate', 'Value'].iloc[0]
#     df_smelting_primary['Process'] = 'Smelting'
#     df_smelting_primary['Supply_Use'] = 'Use'


#     df_scrap_class_3 = df[df['Flow'].isin(['Scrap']) & (df['Process'] == 'Smelting')].copy()
#     df_scrap_class_3['Flow'] = 'Scrap_class_3'
#     df_scrap_class_3['Value'] = df_scrap_class_3['Value'] / tc.loc[tc['Process'] == 'Smelting', 'Value'].iloc[0]
#     df_scrap_class_3['Value'] = df_scrap_class_3['Value'] / conc.loc[conc['Flow'] == 'Scrap_class_3', 'Value'].iloc[0]
#     df_scrap_class_3['Process'] = 'Smelting'
#     df_scrap_class_3['Supply_Use'] = 'Use'


#     df_matte = df[(df['Flow'].isin(['Scrap', 'Smelting_primary'])) & (df['Process'] == 'Smelting')].copy()
#     df_matte['Flow'] = 'Matte'
#     df_matte = df_matte.groupby(['Region', 'Flow', 'Process', 'Supply_Use', 'Unit']).sum().reset_index()
#     df_matte['Value'] = df_matte['Value'] / conc.loc[conc['Flow'] == 'Matte', 'Value'].iloc[0]
#     df_matte['Supply_Use'] = 'Supply'

#     # calculate slag via difference slag = concentrates + scrap - matte
#     df_slag = df[(df['Flow'].isin(['Scrap', 'Smelting_primary'])) & (df['Process'] == 'Smelting')].copy()
#     df_slag['Flow'] = 'Slag'
#     df_slag = df_slag.groupby(['Region', 'Flow', 'Process', 'Supply_Use', 'Unit']).sum().reset_index()
#     df_slag['Value'] = df_slag['Value'] * (1/tc.loc[tc['Process'] == 'Smelting', 'Value'].iloc[0]-1)
#     df_slag['Value'] = df_slag['Value'] / conc.loc[conc['Flow'] == 'Slag', 'Value'].iloc[0]
#     df_slag['Supply_Use'] = 'Supply'

    

#     df_anode = df[(df['Flow'].isin(['Scrap', 'Smelting_primary'])) & (df['Process'] == 'Smelting')].copy()
#     df_anode['Flow'] = 'Anode'
#     df_anode['Value'] = df_anode['Value'] * tc.loc[tc['Process'] == 'Converting_fire_refining', 'Value'].iloc[0]
#     df_anode['Value'] = df_anode['Value'] / conc.loc[conc['Flow'] == 'Anode', 'Value'].iloc[0]
#     df_anode['Process'] = 'Converting_fire_refining'
#     df_anode['Supply_Use'] = 'Supply'

#     # calculate slag via difference slag = anode - matte
#     df_slag_2 = df[(df['Flow'].isin(['Scrap', 'Smelting_primary'])) & (df['Process'] == 'Smelting')].copy()
#     df_slag_2['Flow'] = 'Slag'
#     df_slag_2['Value'] = df_slag_2['Value'] * (1-tc.loc[tc['Process'] == 'Converting_fire_refining', 'Value'].iloc[0])
#     df_slag_2['Value'] = df_slag_2['Value'] / conc.loc[conc['Flow'] == 'Slag', 'Value'].iloc[0]
#     df_slag_2['Process'] = 'Converting_fire_refining'
#     df_slag_2['Supply_Use'] = 'Supply'



#     df_cathode_pyro = df[(df['Flow'].isin(['Scrap', 'Refined_primary'])) & (df['Process'] == 'Refining')].copy()
#     df_cathode_pyro['Flow'] = 'Refined_copper_pyro'
#     df_cathode_pyro = df_cathode_pyro.groupby(['Region',  'Flow', 'Process', 'Supply_Use', 'Unit']).sum().reset_index()
#     df_cathode_pyro['Value'] = df_cathode_pyro['Value'] * conc.loc[conc['Flow'] == 'Refined_copper_pyro', 'Value'].iloc[0]
#     df_cathode_pyro['Process'] = 'Electrolytic_refining'
#     df_cathode_pyro['Supply_Use'] = 'Supply'


#     df_anode_2 = df[(df['Flow'].isin(['Refined_primary'])) & (df['Process'] == 'Refining')].copy()
#     df_anode_2['Flow'] = 'Anode'
#     df_anode_2['Value'] = df_anode_2['Value'] / conc.loc[conc['Flow'] == 'Anode', 'Value'].iloc[0]
#     df_anode_2['Process'] = 'Electrolytic_refining'
#     df_anode_2['Supply_Use'] = 'Use'

    
#     df_scrap_class_2 = df[(df['Flow'].isin(['Scrap'])) & (df['Process'] == 'Refining')].copy()
#     df_scrap_class_2['Flow'] = 'Scrap_class_2'
#     df_scrap_class_2['Value'] = df_scrap_class_2['Value'] / conc.loc[conc['Flow'] == 'Scrap_class_2', 'Value'].iloc[0]
#     df_scrap_class_2['Process'] = 'Electrolytic_refining'
#     df_scrap_class_2['Supply_Use'] = 'Use'


#     df_cathode = df[df['Flow'].isin(['Cathode'])].copy()
#     df_cathode['Flow'] = 'Refined_copper'
#     df_cathode['Process'] = 'Refined_copper_ag'
#     df_cathode['Supply_Use'] = 'Supply'

#     # concat crude ore and waste rock
#     df_mine = pd.concat([df_crude_ore, df_waste_rock, df_crude_ore_2, df_waste_rock_2], ignore_index=True)

#     df_mine['Process'] = 'Mining'

#     # # total scrap = scrap class 2 + scrap class 3
#     df_mine = df_mine.groupby(['Region', 'Flow', 'Process', 'Supply_Use', 'Unit']).sum().reset_index()
    
#     # df_tot_scrap = pd.concat([df_scrap_class_2, df_scrap_class_3], ignore_index=True)
#     # df_tot_scrap['Flow'] = 'Total_scrap'
#     # df_tot_scrap['Process'] = 'Scrap'
#     # df_tot_scrap['Supply_Use'] = 'Use'
#     # df_tot_scrap = df_tot_scrap.groupby(['Region', 'Flow', 'Process', 'Supply_Use', 'Unit']).sum().reset_index()
    

#     df_list = [df_mine, df_ore, df_conc, df_tailings, df_cathode_hydro, df_tailings_2,
#                                 df_ore_2, df_smelting_primary, df_scrap_class_3, df_matte, df_slag, df_anode,
#                                 df_slag_2, df_cathode_pyro, df_anode_2, df_scrap_class_2, df_cathode]
    
#     # give an error if one of the dfs is empty
#     for i, d in enumerate(df_list):
#         if d.empty:
#             raise ValueError(f"DataFrame at index {i} is empty.")

#     # Concatenate all dataframes
#     df_missing_flows = pd.concat(df_list, ignore_index=True)
    
#     # filter out zeros
#     df_missing_flows = df_missing_flows[df_missing_flows['Value'] != 0]

#     # sum up the values for the same Region, Flow, Process, Supply_Use, Unit combinations
#     # sum for the anode2 and anode1 regarding sec and prim input
#     df_missing_flows = df_missing_flows.groupby(['Region', 'Flow', 'Process', 'Supply_Use', 'Unit'], as_index=False)['Value'].sum()

#     # rename things: {'Crude Ore', 'Concentrates', 'Waste Rock'}
#     rename_dict = {
#         'Crude Ore': 'Crude_ore',
#         'Concentrates': 'Concentrate',
#         'Waste Rock': 'Waste_rock',
#         }
#     df_missing_flows['Flow'] = df_missing_flows['Flow'].replace(rename_dict)


#     # consistency check
#     struc = lookup()['Structure']
#     consistency_check(df_missing_flows, struc)

#     return df_missing_flows



# def consistency_check(df, structure):

#     '''
#     check function if all flows and process instances of df are in structure
#     '''
    
#     # are there flows in df that aare not in structure?
#     df_flows = df['Flow'].unique()
#     struct_flows = structure['Flow'].unique()
#     missing_flows = set(df_flows) - set(struct_flows)
#     assert len(missing_flows) == 0, f"Flows in df not in structure: {missing_flows}"

#     # are there process in df that aare not in structure?
#     df_processes = df['Process'].unique()
#     struct_processes = structure['Process'].unique()
#     missing_processes = set(df_processes) - set(struct_processes)
#     assert len(missing_processes) == 0, f"Processes in df not in structure: {missing_processes}"

#     # every flow process combination exists only once per region
#     duplicates = df.duplicated(subset=['Region',  'Flow', 'Process', 'Supply_Use'], keep=False)
#     assert not duplicates.any(), f"Duplicate flow-process combinations found in df"

#     # no negative values
#     assert (df['Value'] >= 0).all(), "Negative values found in missing flows"


# def calculate_domestic_flows():
#     'In this function we calculate the domestic flows that ultimately result from mass balance'
#     df = calc_missing_flows()
#     look = lookup()
    

#     structure = look['Structure']
#     trade = look['Trade']

#     trade_look = trade[trade['Value'] == 0]
#     structure = structure[structure['Value'] == 1]


#     # for every flow in trade_look Flow check what process it belongs to in the structure. If the process is not included in df then copy the existing slice and
#     # change the process set all the values to the same value for supply == use .. change supply_use to 'Use' and vice versa
#     # this stuff is only possible for domestics flows

#     collect = []

#     for _, row in trade_look.iterrows():
#         flow = row['Flow']
        
#         df_flow = df[df['Flow'] == flow]
#         processes = df_flow['Process'].unique()

#         if len(processes) == 0:
#             logger.warning(f"No processes found for flow {flow} in df - skipping bec there is no data")
#             continue

#         assert 2>= len(processes), f"More than two processes found for flow {flow}"

#         # get the processes from the structure
#         struct_processes = structure[structure['Flow'] == flow]['Process'].unique()
#         missing_processes = set(struct_processes) - set(processes)

#         if len(missing_processes) == 0:
#             logger.info(f"No missing processes for flow {flow}")
#             continue

#         for process in missing_processes:

#             df_flow_rev = df_flow.copy()
#             df_flow_rev['Process'] = process
#             df_flow_rev['Supply_Use'] = df_flow_rev['Supply_Use'].apply(lambda x: 'Use' if x == 'Supply' else 'Supply')
#             collect.append(df_flow_rev)
    
#     df_missing = pd.concat(collect, ignore_index=True)

#     df_final = pd.concat([df, df_missing], ignore_index=True)

#     # group and sum up
#     df_final = df_final.groupby(['Region', 'Flow', 'Process', 'Supply_Use', 'Unit'], as_index=False)['Value'].sum()

#     return df_final


# def region_consistency_check(df, conc_map):
#     '''
#     Check if all regions in df are in conc_map
#     '''
#     df_regions = df['Region'].unique()
#     conc_regions = conc_map['Source_name'].unique()
#     missing_regions = set(df_regions) - set(conc_regions)
#     assert len(missing_regions) == 0, f"Regions in df not in conc_map: {missing_regions}"


# def transform_to_region_base(base_path=r'data\input\conc\icsg.xlsx'):
#     """
#     Transform df regions to conc_map regions.
    
#     Rules:
#     - 1:1 mapping: region in df maps directly to region in conc_map → keep value as is.
#     - 1:n mapping: one region in df maps to multiple in conc_map → split value equally.
#     - n:1 mapping: multiple regions in df map to one in conc_map → sum the values.
#     """
    
#     # Step 1: Load your flows df and the concordance table
#     df = calculate_domestic_flows()
#     conc_map = pd.read_excel(base_path, sheet_name='Region')

#     # region consistency check
#     region_consistency_check(df, conc_map)

#     # Step 2: Merge df with mapping
#     merged = df.merge(conc_map, left_on='Region', right_on='Source_name', how='left')

#     # Step 3: Compute mapping counts per Source_name
#     counts = conc_map.groupby('Source_name')['Base_name'].count().reset_index()
#     counts.rename(columns={'Base_name': 'map_count'}, inplace=True)

#     # Merge counts into merged df
#     merged = merged.merge(counts, on='Source_name', how='left')

#     # Step 4: Handle 1:n mapping → distribute equally
#     merged['Value_adj'] = merged['Value'] / merged['map_count']

#     # Step 5: Aggregate to final conc_region
#     result = (
#         merged
#         .groupby(['Base_name', 'Flow', 'Process', 'Supply_Use', 'Unit'], as_index=False)
#         .agg({'Value_adj': 'sum'})
#     )

#     result.rename(columns={'Value_adj': 'Value',
#     'Base_name': 'Region'}, inplace=True)

#     result['Value'] = result.Value.round(3)

#     # split into accounting entities
#     time_stamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

#     use = result[result['Supply_Use'] == 'Use'].copy()
#     supply = result[result['Supply_Use'] == 'Supply'].copy()

#     o_flows = ['Total_scrap']
#     e_flows = ['Crude_ore']
#     p_flows = ['Waste_rock', 'Tailings', 'Slag']
#     y_flows = ['Refined_copper']

#     o = use[use['Flow'].isin(o_flows)]
#     e = use[use['Flow'].isin(e_flows)]
#     y = use[use['Flow'].isin(y_flows)]
#     p = supply [supply['Flow'].isin(p_flows)]
    

#     use = use[~use['Flow'].isin(o_flows + e_flows + y_flows)]
#     supply = supply[~supply['Flow'].isin(p_flows)]

#     folder_name = r"data/proc/datafeed/icsg"

#     os.makedirs(folder_name, exist_ok=True)

#     use.to_csv(os.path.join(folder_name, f'U_prod_icsg_{time_stamp}.csv'), index=False)
#     supply.to_csv(os.path.join(folder_name, f'S_prod_icsg_{time_stamp}.csv'), index=False)
#     o.to_csv(os.path.join(folder_name, f'O_prod_icsg_{time_stamp}.csv'), index=False)
#     e.to_csv(os.path.join(folder_name, f'E_prod_icsg_{time_stamp}.csv'), index=False)
#     p.to_csv(os.path.join(folder_name, f'P_prod_icsg_{time_stamp}.csv'), index=False)
#     y.to_csv(os.path.join(folder_name, f'Y_prod_icsg_{time_stamp}.csv'), index=False)


#     logger.info(f"Transformed data saved to {folder_name}")






# ####################################################################################################

# # functions to create region totals for supply and use entities

# ####################################################################################################



# def use_region_totals(
#         origin = r'data\proc\datafeed\icsg',
#         dest_folder_tot = r'data\proc\datafeed\icsg\region_totals',
#         name_tot = 'U'
#     ):

#     upaths = ['Y_prod_icsg_20251015_103115.csv', 'U_prod_icsg_20251015_103115.csv',
#               'O_prod_icsg_20251015_103115.csv', 'E_prod_icsg_20251015_103115.csv']
#     l = lookup()
#     ent = l['Sector2Entity']
#     dims = [
#              'Sector_origin', 'Entity_origin',
#             'Region_destination', 'Sector_destination', 'Entity_destination', 'Value'
#         ]
    
    
#     u = pd.read_csv(os.path.join(origin, upaths[0]))
#     y = pd.read_csv(os.path.join(origin, upaths[1]))
#     o = pd.read_csv(os.path.join(origin, upaths[2]))
#     e = pd.read_csv(os.path.join(origin, upaths[3]))
#     use_tot = pd.concat([u, y, o, e], ignore_index=True)

#     use_tot.rename(columns={'Region': 'Region_destination', 
#                         'Flow': 'Sector_origin',
#                         'Process': 'Sector_destination',
#                        }, inplace=True)
#     use_tot.drop(columns=['Unit',  'Supply_Use'], inplace=True)

#     use_tot = use_tot.merge(ent, left_on='Sector_origin', right_on='Sector_name', how='left')
#     use_tot = use_tot.rename(columns={'Entity': 'Entity_origin'})
#     use_tot = use_tot.merge(ent, left_on='Sector_destination', right_on='Sector_name', how='left')
#     use_tot = use_tot.rename(columns={'Entity': 'Entity_destination'})
#     use_tot = use_tot[dims]


#     time_stamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
#     dest = os.path.join(dest_folder_tot, f'{name_tot}_{time_stamp}.csv')

#     os.makedirs(dest_folder_tot, exist_ok=True)
#     use_tot.to_csv(dest, index=False)
#     logger.info(f"Use region totals saved to {dest}")

#     return None


# def supply_region_totals(origin = r'data\proc\datafeed\icsg',
#         dest_folder = r'data\proc\datafeed\icsg\region_totals',
#         name = 'S'):
     
#     spaths = ['S_prod_icsg_20251020_101950.csv', 'P_prod_icsg_20251020_101950.csv',]
#     l = lookup()
#     ent = l['Sector2Entity']
#     dims = [
#              'Region_origin', 'Sector_origin', 'Entity_origin',
#             'Sector_destination', 'Entity_destination', 'Value'
#         ]
    
#     s = pd.read_csv(os.path.join(origin, spaths[0]))
#     p = pd.read_csv(os.path.join(origin, spaths[1]))
#     supply_tot = pd.concat([s, p], ignore_index=True)
#     supply_tot.rename(columns={'Region': 'Region_origin', 
#                         'Flow': 'Sector_destination',
#                         'Process': 'Sector_origin',}, inplace=True)
    
#     supply_tot.drop(columns=['Unit', 'Supply_Use'], inplace=True)

#     supply_tot = supply_tot.merge(ent, left_on='Sector_origin', right_on='Sector_name', how='left')
#     supply_tot = supply_tot.rename(columns={'Entity': 'Entity_origin'})
#     supply_tot = supply_tot.merge(ent, left_on='Sector_destination', right_on='Sector_name', how='left')
#     supply_tot = supply_tot.rename(columns={'Entity': 'Entity_destination'})
#     supply_tot = supply_tot[dims]


#     time_stamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
#     dest = os.path.join(dest_folder, f'{name}_{time_stamp}.csv')
#     os.makedirs(dest_folder, exist_ok=True)
#     supply_tot.to_csv(dest, index=False)

#     logger.info(f"Supply region totals saved to {dest}")

#     return None


# ####################################################################################################

# # functions to prepare the boundary input and output in inital estimate format

# ####################################################################################################

# def use_to_ie(
#         origin = r'data\proc\datafeed\icsg',
#         dest_folder = r'data\proc\datafeed\dom_calc',
#     ):

#     upaths = ['O_prod_icsg_20251020_101950.csv', 'E_prod_icsg_20251020_101950.csv']

#     l = lookup()
#     ent = l['Sector2Entity']
#     dims = ['Region_origin',
#              'Sector_origin', 'Entity_origin',
#             'Region_destination', 'Sector_destination', 'Entity_destination', 'Value'
#         ]
    
    
    
#     o = pd.read_csv(os.path.join(origin, upaths[0]))
#     e = pd.read_csv(os.path.join(origin, upaths[1]))
    

#     o.rename(columns={'Region': 'Region_destination', 
#                         'Flow': 'Sector_origin',
#                         'Process': 'Sector_destination',
#                        }, inplace=True)
#     e.rename(columns={'Region': 'Region_destination', 
#                         'Flow': 'Sector_origin',
#                         'Process': 'Sector_destination',
#                        }, inplace=True)

    
#     o = merge_entity_to_df(o, ent)
#     e = merge_entity_to_df(e, ent)
#     o['Region_origin'] = o['Region_destination']
#     e['Region_origin'] = e['Region_destination']
#     o = o[dims]
#     e = e[dims]
    

#     time_stamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

#     os.makedirs(dest_folder, exist_ok=True)

#     dest_o = os.path.join(dest_folder, f'O_{time_stamp}.csv')
#     dest_e = os.path.join(dest_folder, f'E_{time_stamp}.csv')

#     o.to_csv(dest_o, index=False)
#     e.to_csv(dest_e, index=False)

#     logger.info(f"Use region totals saved to {dest_o} and {dest_e}")
#     return None


# def merge_entity_to_df(df, ent):
#         df = df.merge(ent, left_on='Sector_origin', right_on='Sector_name', how='left')
#         df = df.rename(columns={'Entity': 'Entity_origin'})
#         df = df.merge(ent, left_on='Sector_destination', right_on='Sector_name', how='left')
#         df = df.rename(columns={'Entity': 'Entity_destination'})
#         return df


# def supply_to_ie(
#         origin = r'data\proc\datafeed\icsg',
#         dest_folder = r'data\proc\datafeed\dom_calc',
#     ):

#     upaths = ['P_prod_icsg_20251020_101950.csv']

#     l = lookup()
#     ent = l['Sector2Entity']
#     dims = ['Region_origin',
#              'Sector_origin', 'Entity_origin',
#             'Region_destination', 'Sector_destination', 'Entity_destination', 'Value'
#         ]
    
#     p = pd.read_csv(os.path.join(origin, upaths[0]))
    

#     p.rename(columns={'Region': 'Region_origin', 
#                         'Flow': 'Sector_destination',
#                         'Process': 'Sector_origin',
#                        }, inplace=True)
    
#     p = merge_entity_to_df(p, ent)
#     p['Region_destination'] = p['Region_origin']
    
    
#     p = p[dims]

#     time_stamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

#     os.makedirs(dest_folder, exist_ok=True)

#     dest_p = os.path.join(dest_folder, f'P_{time_stamp}.csv')
    

#     p.to_csv(dest_p, index=False)

#     logger.info(f"Supply region totals saved to {dest_p}")


#     return None


# Tranform the data feed into








if __name__ == "__main__":

  file_name = r'ICSG 2025 Statistical Yearbook.xlsx'
  icsg_loader(file_name, id = 2025)
  file_name = r'ICSG 2023 Statistical Yearbook.xlsx'
  icsg_loader(file_name, id = 2023)