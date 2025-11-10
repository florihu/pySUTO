import sys
import os
import pandas as pd
import numpy as np
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from model.core.datafeed.util import lookup


from model.util import get_path, folder_name_check

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def icsg_2023():

    file_name = r'ICSG_Yearbooks\ICSG 2023 Statistical Yearbook.xlsx'
    path = get_path(file_name)
    

    # Define the list as [Process, Supply/Use]
    sheet_dict = {
        '3.Mine': ['Mining', 'Supply'],
        '4.Sml': ['Smelting', 'Use'],
        '5.Ref': ['Refining', 'Use'],
        '6.SXEW': ['Refining_SXEW', 'Use'],
        '7.Usage': ['Semi', 'Use'],
        '10.Semis': ['Semi', 'Supply'],
    }
    rename_dict = {
        'COUNTRY': 'Region',
        'Source': 'Flow',
        'Feed': 'Flow',
        'Semis': 'Flow',
    }

    collect = []

    for sheet, name in sheet_dict.items():

        process = name[0]
        flow_type = name[1]

        df_init = pd.read_excel(path, sheet_name=sheet, header=2)


        if process == 'Refining_SXEW':
            df_init['Flow'] = 'Cathode'
        elif (process == 'Semi') and (flow_type == 'Use'):
            df_init['Flow'] = 'Cathode'

        # find row that contains 'Source:' in first column
        source_row = df_init[df_init.iloc[:, 0].str.contains('Source:', na=False)].index[0]

        if name =='Semi_use':
            # delete row that contains EUROPEAN UNION + UK 6/ in the first column
            df_init = df_init[~df_init.iloc[:, 0].str.contains('EUROPEAN UNION + UK 6/', na=False)]

        # remove all rows after this row
        df = df_init.iloc[1:source_row, :]
        df.rename(columns=rename_dict, inplace=True)

        # Exclude Regions that contain : T, TOTALS, 'WORLD', 'Other', 'OthOtheersrs', 'EUROPEAN', 'EU', 'By'
        df = df[~df['Region'].str.contains('T|TOTALS|WORLD|Other|OthOtheersrs|EUROPEAN|EU|By', na=False)]

        df =df[~df.Region.isna()]

        # find all relevant data points in the Flow column that include 'Total' and 'Not'

        df = df[~df['Flow'].str.contains('Total|Not', na=False)]

        # exclude flows that contain , or .
        df = df[~df['Flow'].str.contains(r',|\.', na=False)]

        # forward fill region nan
        df['Region'] = df['Region'].fillna(method='bfill')

        df = df.melt(id_vars=['Region', 'Flow'], var_name='Year', value_name='Value')
        df['Process'] = process
        df['Unit'] = 'kt'
        df['Year'] = df['Year'].astype(int)
        df['Value'] = df['Value'].astype(float)
        df['Supply_Use'] = flow_type
        collect.append(df)
    

    df = pd.concat(collect, ignore_index=True)

    # find all relevant data points in the Flow column that include 'Total' and 'Not'
    relevant_flows = df[~df['Flow'].str.contains('Total|Not', na=False)]

    return relevant_flows


def icsg_2014():

    file_name = r'ICSG_Yearbooks\ICSG 2014 Statistical Yearbook.xlsx'
    path = get_path(file_name)
    

    # Define the list as [Process, Supply/Use]
    sheet_dict = {
        'Mining_1': ['Mining', 'Supply'],
        'Mining_2': ['Mining', 'Supply'],
        'Smelting_1': ['Smelting', 'Use'],
        'Smelting_2': ['Smelting', 'Use'],
        'Refining_1': ['Refining', 'Use'],
        'Refining_2': ['Refining', 'Use'],
        'Refining_3': ['Refining', 'Use'],
        'SXEW': ['Refining_SXEW', 'Use'],
        'Semi_use_1': ['Semi', 'Use'],
        'Semi_use_2': ['Semi', 'Use'],
        'Semi_supply_1': ['Semi', 'Supply'],
        'Semi_supply_2': ['Semi', 'Supply'],
    }


    rename_dict = {
        'COUNTRY': 'Region',
        'Source': 'Flow',
        'Feed': 'Flow',
        'Semis': 'Flow',
        'Semis-': 'Flow',
    }

    collect = []

    for sheet, name in sheet_dict.items():

        process = name[0]
        flow_type = name[1]

        if sheet == 'SXEW':
            header = 3  # skip the first row for SXEW
        else:
            header = 2

        df_init = pd.read_excel(path, sheet_name=sheet, header=header)


        if sheet in ['Semi_use_1', 'Semi_use_2']:
            df_init['Flow'] = 'Cathode'
        elif sheet == 'SXEW':
            df_init['Flow'] = 'Cathode_SXEW'
        elif sheet in ['Smelting_2', 'Semi_supply_1', 'Semi_supply_2']:
            #drop second column
            df_init = df_init.drop(df_init.columns[1], axis=1)

        # find row that contains 'Source:' or 'ICSG' in first column
        source_row = df_init[df_init.iloc[:, 0].str.contains('Source|ICSG', na=False)].index[0]      

        df = df_init.iloc[1:source_row, :].copy()  # remove all rows after this row
        df.rename(columns=rename_dict, inplace=True)

        # Exclude Regions that contain : T, TOTALS, 'WORLD', 'Other', 'OthOtheersrs', 'EUROPEAN', 'EU', 'By'
        df = df[~df['Region'].str.contains('T|TOTALS|WORLD|Other|OthOtheersrs|EUROPEAN|EU|By', na=False)]

        df =df[~df.Region.isna()]

        # find all relevant data points in the Flow column that include 'Total' and 'Not'

        df = df[~df['Flow'].str.contains('Total|Not', na=False)]

        # exclude flows that contain , or .
        df = df[~df['Flow'].str.contains(r',|\.', na=False)]

        # forward fill region nan
        df['Region'] = df['Region'].fillna(method='bfill')

        # remove "n/" from region with n any integer number"
        df['Region'] = df['Region'].str.replace(r'(\d+/)+\s*', '', regex=True)


        rename_flow_dict = {
            'SSecondary': 'Secondary',
            'Primaryy': 'Primary'}

        df['Flow'] = df['Flow'].replace(rename_flow_dict)

        df = df.melt(id_vars=['Region', 'Flow'], var_name='Year', value_name='Value')

        # Values that contain multiple dots >1 replace with 0
        df['Value'] = df['Value'].astype(str).apply(lambda x: '0' if x.count('.') > 1 else x)

        # Step 1: Split Flow and Value by newline
        df['Flow'] = df['Flow'].astype(str).str.split('\n')
        df['Value'] = df['Value'].astype(str).str.split('\n')

        # Step 2: Pad 'Value' list with NaNs to match length of 'Flow'
        def pad_value_list(val_list, target_len):
            if isinstance(val_list, list):
                return val_list + [np.nan] * (target_len - len(val_list))
            return [np.nan] * target_len

        df['Value'] = df.apply(lambda row: pad_value_list(row['Value'], len(row['Flow'])), axis=1)

        # Step 3: Explode both Flow and Value
        df = df.explode(['Flow', 'Value'], ignore_index=True)

        # Step 4: Clean and convert Value
        df['Value'] = df['Value'].replace('', np.nan)
        df['Value'] = df['Value'].str.replace(',', '', regex=False)
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
        df['Year'] = df['Year'].astype(int)

        df['Process'] = process

        df['Supply_Use'] = flow_type
        df['Unit'] = 'kt'

        collect.append(df)
    

    df = pd.concat(collect, ignore_index=True)

    # drop year 2013
    df = df[df['Year'] != 2013]

    # fill value NaN with 0
    df['Value'] = df['Value'].fillna(0)


    return df


def icsg_2005():

    file_name = "ICSG 2005 Statisctial Yearbook.xlsx"
    path = get_path(file_name)
    

    # Define the list as [Process, Supply/Use]
    sheet_dict = {
        'Mining_1': ['Mining', 'Supply'],
        'Mining_2': ['Mining', 'Supply'],
        'Smelting_1': ['Smelting', 'Use'],
        'Smelting_2': ['Smelting', 'Use'],
        'Refining_1': ['Refining', 'Use'],
        'Refining_2': ['Refining', 'Use'],
        'Refining_3': ['Refining', 'Use'],
        'SXEW': ['Refining_SXEW', 'Use'],
        'Semi_use_1': ['Semi', 'Use'],
        'Semi_use_2': ['Semi', 'Use'],
        'Semi_supply_1': ['Semi', 'Supply'],
        'Semi_supply_2': ['Semi', 'Supply'],
    }


    rename_dict = {
        'COUNTRY': 'Region',
        'Source': 'Flow',
        'Feed': 'Flow',
        'Semis': 'Flow',
        'Semis-': 'Flow',
        'Source\n1/': 'Flow',
        'Feed Type': 'Flow',
        'Feed Source  1/': 'Flow',
        'Semis- Type': 'Flow',
    }

    collect = []

    for sheet, name in sheet_dict.items():

        process = name[0]
        flow_type = name[1]

        if sheet == 'SXEW':
            header = 1  # skip the first row for SXEW
        else:
            header = 0
        

        df_init = pd.read_excel(path, sheet_name=sheet, header=header)

        # conv cols to str and identify cols id with unnamed delete col

        df_init.columns = df_init.columns.astype(str)
        df_init = df_init.loc[:, ~df_init.columns.str.contains('^Unnamed')]


        if sheet in ['Semi_use_1', 'Semi_use_2']:
             df_init['Flow'] = 'Cathode'
        elif sheet == 'SXEW':
             df_init['Flow'] = 'Cathode_SXEW'
        # elif sheet in ['Smelting_2', 'Semi_supply_1', 'Semi_supply_2']:
        #     #drop second column
        #     df_init = df_init.drop(df_init.columns[1], axis=1)

        # find row that contains 'Source:' or 'ICSG' in first column
        source_row = df_init[df_init.iloc[:, 0].str.contains('Source|ICSG', na=False)].index[0]      

        df = df_init.iloc[:source_row, :].copy()  # remove all rows after this row
        df.rename(columns=rename_dict, inplace=True)


        # Exclude Regions that contain : T, TOTALS, 'WORLD', 'Other', 'OthOtheersrs', 'EUROPEAN', 'EU', 'By'
        df = df[~df['Region'].str.contains('T|TOTALS|WORLD|Other|OthOtheersrs|EUROPEAN|EU|By', na=False)]

        df =df[~df.Region.isna()]

        # find all relevant data points in the Flow column that include 'Total' and 'Not'

        df = df[~df['Flow'].str.contains('Total|Not', na=False)]

        # exclude flows that contain , or .
        df = df[~df['Flow'].str.contains(r',|\.', na=False)]

        # forward fill region nan
        df['Region'] = df['Region'].fillna(method='bfill')

        # remove "n/" from region with n any integer number"
        df['Region'] = df['Region'].str.replace(r'(\d+/)+\s*', '', regex=True)


        rename_flow_dict = {
            'SSecondary': 'Secondary',
            'Primaryy': 'Primary',
            'Primary Secondary': 'Primary\nSecondary',
            'Electrowon Primary': 'Electrowon\nPrimary',
            'Electrowon Primary Secondary': 'Electrowon\nPrimary\nSecondary',
            'Electrowon\nPrimary Secondary': 'Electrowon\nPrimary\nSecondary',
            'Electrowon Primary\nSecondary': 'Electrowon\nPrimary\nSecondary'}

        df['Flow'] = df['Flow'].replace(rename_flow_dict)

        df = df.melt(id_vars=['Region', 'Flow'], var_name='Year', value_name='Value')

        # Values that contain multiple dots >1 replace with 0
        #df['Value'] = df['Value'].astype(str).apply(lambda x: '0' if x.count('.') > 1 else x)

        # Step 1: Split Flow and Value by newline
        df['Flow'] = df['Flow'].astype(str).str.split('\n')
        df['Value'] = df['Value'].astype(str).str.split('\n')

        # Step 2: Pad 'Value' list with NaNs to match length of 'Flow'
        def pad_value_list(val_list, target_len):
            if isinstance(val_list, list):
                return val_list + [np.nan] * (target_len - len(val_list))
            return [np.nan] * target_len

        df['Value'] = df.apply(lambda row: pad_value_list(row['Value'], len(row['Flow'])), axis=1)

        # Step 3: Explode both Flow and Value
        df = df.explode(['Flow', 'Value'], ignore_index=True)

        # Step 4: Clean and convert Value
        df['Value'] = df['Value'].replace('', np.nan)
        df['Value'] = df['Value'].str.replace(',', '', regex=False)
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')


        # ValueError: invalid literal for int() with base 10: '2004\np/ - replace with 2004
        df['Year'] = df['Year'].astype(str).str.split('\n').str[0]

        df['Year'] = df['Year'].astype(int)

        df['Process'] = process

        df['Supply_Use'] = flow_type
        df['Unit'] = 'kt'

        collect.append(df)
    

    df = pd.concat(collect, ignore_index=True)

    # fill value NaN with 0
    df['Value'] = df['Value'].fillna(0)

    


    return df


import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def read_icsg_all_years(out_path=r'data/proc/datafeed/icsg', 
                        conc_path=r'data/input/conc/icsg.xlsx'):

    # --- Collect data from different years ---
    df_2023 = icsg_2023()
    df_2014 = icsg_2014()
    df_2005 = icsg_2005()

    # --- Load base concordances ---
    conc = pd.read_excel(conc_path, sheet_name=None)
    conc_sec = conc['Sector_int']
    conc_reg = conc['Region']

    # --- Assert column consistency ---
    assert set(df_2023.columns) == set(df_2014.columns) == set(df_2005.columns), \
        "DataFrames have different columns"

    # --- Combine all data ---
    df = pd.concat([df_2023, df_2014, df_2005], ignore_index=True)
    df['Region'] = df['Region'].str.strip()

    # --- Region renaming ---
    reg_rename = {
        'Azerbeijan': 'Azerbaijan',
        'North Macedonia': 'Macedonia',
        'Belgium-': 'Belgium-Luxembourg',
        'Belgium- Luxembourg': 'Belgium-Luxembourg',
        'kyrgyzstan': 'Kyrgyzstan',
        'Congo Rep.': 'Congo',
        'Congo DR': 'Congo, D.R.', 
        'Philippppines': 'Philippines',
        'United': 'United States',
        'Zambia  ': 'Zambia',
    }

    df.replace({'Region': reg_rename}, inplace=True)

    # --- Check for unknown regions ---
    missing_regions = set(df['Region'].unique()) - set(conc_reg['Source_name'].unique())
    if missing_regions:
        logger.warning(f"Regions in ICSG data not found in base classification: {missing_regions}")

    # --- Build flow-process mapping ---
    cdict = {
        (flow_root, process_root): (flow_base, process_base)
        for flow_root, process_root, flow_base, process_base in zip(
            conc_sec['Flow_root'],
            conc_sec['Process_root'],
            conc_sec['Flow_base'],
            conc_sec['Process_base']
        )
    }

    # --- Detect missing mappings before applying ---
    df_pairs = set(df[['Flow', 'Process']].itertuples(index=False, name=None))
    mapping_keys = set(cdict.keys())
    missing_pairs = df_pairs - mapping_keys

    if missing_pairs:
        missing_preview = list(missing_pairs)[:10]
        raise ValueError(
            f"The following (Flow, Process) combinations are not in the renaming dictionary: "
            f"{missing_preview}{' ...' if len(missing_pairs) > 10 else ''}"
        )

    # --- Replace (Flow, Process) pairs efficiently ---
    mapped = df[['Flow', 'Process']].apply(
        lambda x: cdict.get((x['Flow'], x['Process'])),
        axis=1,
        result_type='expand'
    )

    df[['Flow', 'Process']] = mapped

    # --- Save output ---
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(out_path, exist_ok=True)
    out_file = os.path.join(out_path, f'ICSG_raw_clean_{stamp}.csv')
    df.to_csv(out_file, index=False)

    # --- Final consistency check ---
    check_consistency(df)
    logger.info(f"ICSG data saved to {out_file}")

    return df


def check_consistency(df):
    '''
    A pair Region, Flow, Process should be unique for each year
    '''
    duplicates = df[df.duplicated(subset=['Region', 'Flow', 'Process', 'Year'], keep=False)]
    if not duplicates.empty:
        logger.warning(f"Duplicate entries found for the same Region, Flow, Process, Year combination:\n{duplicates}")


    # there are no nan allowed in Region, Flow, Process and value
    if df[['Region', 'Flow', 'Process', 'Value']].isnull().any().any():
        logger.warning("NaN values found in Region, Flow, Process, or Value columns")
    if (df['Value'] < 0).any():
        logger.warning("Negative values found in Value column")

def calc_missing_flows(tc_path = r'data\input\raw\TCs.xlsx',
                            df_path = r'data\proc\datafeed\icsg\ICSG_raw_clean_20251007_115132.csv',
                            conc_path = r'data\input\raw\Conc.xlsx'):

    tc = pd.read_excel(tc_path)
    df = pd.read_csv(df_path)
    conc = pd.read_excel(conc_path)


    not_needed_flows = ['Wire_rod', 'Copper', 'Copper_alloy']
    year = 1995

    df = df[df['Year'] == year]
    df = df[~df['Flow'].isin(not_needed_flows)]

    # drop year col
    df = df.drop(columns=['Year'])


    df = df.groupby(['Region', 'Flow', 'Process', 'Supply_Use', 'Unit'], as_index=False)['Value'].sum()

    # PYRO
    # calculate waste rock - ore
    df_ore = df[df['Flow'] == 'Concentrates'].copy()
    df_ore['Flow'] = 'Ore'
    df_ore['Value'] = df_ore['Value'] / tc.loc[tc['Process'] == 'Milling_Crushing_Floatation', 'Value'].iloc[0]
    df_ore['Value'] = df_ore['Value'] / conc.loc[conc['Flow'] == 'Ore', 'Value'].iloc[0]
    df_ore['Supply_Use'] = 'Use'
    df_ore['Process'] = 'Milling_Crushing_Floatation'

    # Concentrates
    df_conc = df[df['Flow'] == 'Concentrates'].copy()
    df_conc['Flow'] = 'Concentrates'
    df_conc['Value'] = df_conc['Value'] / conc.loc[conc['Flow'] == 'Concentrate', 'Value'].iloc[0]
    df_conc['Supply_Use'] = 'Supply'
    df_conc['Process'] = 'Milling_Crushing_Floatation'
  
    
    # Tailings
    df_tailings = df[df['Flow'] == 'Concentrates'].copy()
    df_tailings['Flow'] = 'Tailings'
    df_tailings['Value'] = df_tailings['Value'] * (1/tc.loc[tc['Process'] == 'Milling_Crushing_Floatation', 'Value'].iloc[0]-1)
    df_tailings['Value'] = df_tailings['Value'] / conc.loc[conc['Flow'] == 'Tailings', 'Value'].iloc[0]
    df_tailings['Supply_Use'] = 'Supply'
    df_tailings['Process'] = 'Milling_Crushing_Floatation'


    #Waste rock
    df_waste_rock = df_ore.copy()
    df_waste_rock['Flow'] = 'Waste Rock'
    df_waste_rock['Value'] = df_waste_rock['Value'] * tc.loc[tc['Process'] == 'Mining', 'Value'].iloc[0]
    df_waste_rock['Supply_Use'] = 'Supply'
    df_waste_rock['Process'] = 'Mining'

    # Crude ore
    df_crude_ore = df_ore.copy()
    df_crude_ore['Flow'] = 'Crude Ore'
    df_crude_ore['Value'] = df_crude_ore['Value'] + df_waste_rock['Value']
    df_crude_ore['Supply_Use'] = 'Use'
    df_crude_ore['Process'] = 'Mining'


    # HYDRO
    #Refined_copper_hydro
    df_cathode_hydro = df[(df['Flow'] == 'Refined_primary') & (df['Process'] == 'SXEW')].copy()
    df_cathode_hydro['Flow'] = 'Refined_copper_hydro'
    df_cathode_hydro['Process'] = 'SXEW'
    df_cathode_hydro['Supply_Use'] = 'Supply'
    df_cathode_hydro['Value'] = df_cathode_hydro['Value'] / conc.loc[conc['Flow'] == 'Refined_copper_hydro', 'Value'].iloc[0]

    # Tailings
    df_tailings_2 = df[(df['Flow'] == 'Refined_primary') & (df['Process'] == 'SXEW')].copy()
    df_tailings_2['Flow'] = 'Tailings'
    df_tailings_2['Value'] = df_tailings_2['Value'] * (1/tc.loc[tc['Process'] == 'SXEW', 'Value'].iloc[0]-1)
    df_tailings_2['Value'] = df_tailings_2['Value'] / conc.loc[conc['Flow'] == 'Tailings', 'Value'].iloc[0]
    df_tailings_2['Supply_Use'] = 'Supply'
    df_tailings_2['Process'] = 'SXEW'

    
    # Ore
    df_ore_2 = df[(df['Flow'] == 'Refined_primary') & (df['Process'] == 'SXEW')].copy()
    df_ore_2['Flow'] = 'Ore'
    df_ore_2['Value'] = df_ore_2['Value'] / tc.loc[tc['Process'] == 'SXEW', 'Value'].iloc[0]
    df_ore_2['Value'] = df_ore_2['Value'] / conc.loc[conc['Flow'] == 'Ore', 'Value'].iloc[0]
    df_ore_2['Supply_Use'] = 'Use'
    df_ore_2['Process'] = 'SXEW'

    #Waste rock
    df_waste_rock_2 = df_ore_2.copy()
    df_waste_rock_2['Flow'] = 'Waste Rock'
    df_waste_rock_2['Value'] = df_waste_rock_2['Value'] * tc.loc[tc['Process'] == 'Mining', 'Value'].iloc[0]
    df_waste_rock_2['Supply_Use'] = 'Supply'
    df_waste_rock_2['Process'] = 'Mining'

    # Crude ore
    df_crude_ore_2 = df_ore_2.copy()
    df_crude_ore_2['Flow'] = 'Crude Ore'
    df_crude_ore_2['Supply_Use'] = 'Use'
    df_crude_ore_2['Value'] = df_crude_ore_2['Value'] + df_waste_rock_2['Value']
    df_crude_ore_2['Process'] = 'Mining'


    df_smelting_primary = df[df['Flow'].isin(['Smelting_primary']) & (df['Process'] == 'Smelting')].copy()
    df_smelting_primary['Flow'] = 'Concentrates'
    # gropby concentrates and sum up -> we do not differentiate between low grade and concentrates
    df_smelting_primary = df_smelting_primary.groupby(['Region', 'Flow', 'Process', 'Supply_Use', 'Unit']).sum().reset_index()
    df_smelting_primary['Value'] = df_smelting_primary['Value'] / tc.loc[tc['Process'] == 'Smelting', 'Value'].iloc[0]
    df_smelting_primary['Value'] = df_smelting_primary['Value'] / conc.loc[conc['Flow'] == 'Concentrate', 'Value'].iloc[0]
    df_smelting_primary['Process'] = 'Smelting'
    df_smelting_primary['Supply_Use'] = 'Use'


    df_scrap_class_3 = df[df['Flow'].isin(['Scrap']) & (df['Process'] == 'Smelting')].copy()
    df_scrap_class_3['Flow'] = 'Scrap_class_3'
    df_scrap_class_3['Value'] = df_scrap_class_3['Value'] / tc.loc[tc['Process'] == 'Smelting', 'Value'].iloc[0]
    df_scrap_class_3['Value'] = df_scrap_class_3['Value'] / conc.loc[conc['Flow'] == 'Scrap_class_3', 'Value'].iloc[0]
    df_scrap_class_3['Process'] = 'Smelting'
    df_scrap_class_3['Supply_Use'] = 'Use'


    df_matte = df[(df['Flow'].isin(['Scrap', 'Smelting_primary'])) & (df['Process'] == 'Smelting')].copy()
    df_matte['Flow'] = 'Matte'
    df_matte = df_matte.groupby(['Region', 'Flow', 'Process', 'Supply_Use', 'Unit']).sum().reset_index()
    df_matte['Value'] = df_matte['Value'] / conc.loc[conc['Flow'] == 'Matte', 'Value'].iloc[0]
    df_matte['Supply_Use'] = 'Supply'

    # calculate slag via difference slag = concentrates + scrap - matte
    df_slag = df[(df['Flow'].isin(['Scrap', 'Smelting_primary'])) & (df['Process'] == 'Smelting')].copy()
    df_slag['Flow'] = 'Slag'
    df_slag = df_slag.groupby(['Region', 'Flow', 'Process', 'Supply_Use', 'Unit']).sum().reset_index()
    df_slag['Value'] = df_slag['Value'] * (1/tc.loc[tc['Process'] == 'Smelting', 'Value'].iloc[0]-1)
    df_slag['Value'] = df_slag['Value'] / conc.loc[conc['Flow'] == 'Slag', 'Value'].iloc[0]
    df_slag['Supply_Use'] = 'Supply'

    

    df_anode = df[(df['Flow'].isin(['Scrap', 'Smelting_primary'])) & (df['Process'] == 'Smelting')].copy()
    df_anode['Flow'] = 'Anode'
    df_anode['Value'] = df_anode['Value'] * tc.loc[tc['Process'] == 'Converting_fire_refining', 'Value'].iloc[0]
    df_anode['Value'] = df_anode['Value'] / conc.loc[conc['Flow'] == 'Anode', 'Value'].iloc[0]
    df_anode['Process'] = 'Converting_fire_refining'
    df_anode['Supply_Use'] = 'Supply'

    # calculate slag via difference slag = anode - matte
    df_slag_2 = df[(df['Flow'].isin(['Scrap', 'Smelting_primary'])) & (df['Process'] == 'Smelting')].copy()
    df_slag_2['Flow'] = 'Slag'
    df_slag_2['Value'] = df_slag_2['Value'] * (1-tc.loc[tc['Process'] == 'Converting_fire_refining', 'Value'].iloc[0])
    df_slag_2['Value'] = df_slag_2['Value'] / conc.loc[conc['Flow'] == 'Slag', 'Value'].iloc[0]
    df_slag_2['Process'] = 'Converting_fire_refining'
    df_slag_2['Supply_Use'] = 'Supply'



    df_cathode_pyro = df[(df['Flow'].isin(['Scrap', 'Refined_primary'])) & (df['Process'] == 'Refining')].copy()
    df_cathode_pyro['Flow'] = 'Refined_copper_pyro'
    df_cathode_pyro = df_cathode_pyro.groupby(['Region',  'Flow', 'Process', 'Supply_Use', 'Unit']).sum().reset_index()
    df_cathode_pyro['Value'] = df_cathode_pyro['Value'] * conc.loc[conc['Flow'] == 'Refined_copper_pyro', 'Value'].iloc[0]
    df_cathode_pyro['Process'] = 'Electrolytic_refining'
    df_cathode_pyro['Supply_Use'] = 'Supply'


    df_anode_2 = df[(df['Flow'].isin(['Refined_primary'])) & (df['Process'] == 'Refining')].copy()
    df_anode_2['Flow'] = 'Anode'
    df_anode_2['Value'] = df_anode_2['Value'] / conc.loc[conc['Flow'] == 'Anode', 'Value'].iloc[0]
    df_anode_2['Process'] = 'Electrolytic_refining'
    df_anode_2['Supply_Use'] = 'Use'

    
    df_scrap_class_2 = df[(df['Flow'].isin(['Scrap'])) & (df['Process'] == 'Refining')].copy()
    df_scrap_class_2['Flow'] = 'Scrap_class_2'
    df_scrap_class_2['Value'] = df_scrap_class_2['Value'] / conc.loc[conc['Flow'] == 'Scrap_class_2', 'Value'].iloc[0]
    df_scrap_class_2['Process'] = 'Electrolytic_refining'
    df_scrap_class_2['Supply_Use'] = 'Use'


    df_cathode = df[df['Flow'].isin(['Cathode'])].copy()
    df_cathode['Flow'] = 'Refined_copper'
    df_cathode['Process'] = 'Refined_copper_ag'
    df_cathode['Supply_Use'] = 'Supply'

    # concat crude ore and waste rock
    df_mine = pd.concat([df_crude_ore, df_waste_rock, df_crude_ore_2, df_waste_rock_2], ignore_index=True)

    df_mine['Process'] = 'Mining'

    # # total scrap = scrap class 2 + scrap class 3
    df_mine = df_mine.groupby(['Region', 'Flow', 'Process', 'Supply_Use', 'Unit']).sum().reset_index()
    
    # df_tot_scrap = pd.concat([df_scrap_class_2, df_scrap_class_3], ignore_index=True)
    # df_tot_scrap['Flow'] = 'Total_scrap'
    # df_tot_scrap['Process'] = 'Scrap'
    # df_tot_scrap['Supply_Use'] = 'Use'
    # df_tot_scrap = df_tot_scrap.groupby(['Region', 'Flow', 'Process', 'Supply_Use', 'Unit']).sum().reset_index()
    

    df_list = [df_mine, df_ore, df_conc, df_tailings, df_cathode_hydro, df_tailings_2,
                                df_ore_2, df_smelting_primary, df_scrap_class_3, df_matte, df_slag, df_anode,
                                df_slag_2, df_cathode_pyro, df_anode_2, df_scrap_class_2, df_cathode]
    
    # give an error if one of the dfs is empty
    for i, d in enumerate(df_list):
        if d.empty:
            raise ValueError(f"DataFrame at index {i} is empty.")

    # Concatenate all dataframes
    df_missing_flows = pd.concat(df_list, ignore_index=True)
    
    # filter out zeros
    df_missing_flows = df_missing_flows[df_missing_flows['Value'] != 0]

    # sum up the values for the same Region, Flow, Process, Supply_Use, Unit combinations
    # sum for the anode2 and anode1 regarding sec and prim input
    df_missing_flows = df_missing_flows.groupby(['Region', 'Flow', 'Process', 'Supply_Use', 'Unit'], as_index=False)['Value'].sum()

    # rename things: {'Crude Ore', 'Concentrates', 'Waste Rock'}
    rename_dict = {
        'Crude Ore': 'Crude_ore',
        'Concentrates': 'Concentrate',
        'Waste Rock': 'Waste_rock',
        }
    df_missing_flows['Flow'] = df_missing_flows['Flow'].replace(rename_dict)


    # consistency check
    struc = lookup()['Structure']
    consistency_check(df_missing_flows, struc)

    return df_missing_flows



def consistency_check(df, structure):

    '''
    check function if all flows and process instances of df are in structure
    '''
    
    # are there flows in df that aare not in structure?
    df_flows = df['Flow'].unique()
    struct_flows = structure['Flow'].unique()
    missing_flows = set(df_flows) - set(struct_flows)
    assert len(missing_flows) == 0, f"Flows in df not in structure: {missing_flows}"

    # are there process in df that aare not in structure?
    df_processes = df['Process'].unique()
    struct_processes = structure['Process'].unique()
    missing_processes = set(df_processes) - set(struct_processes)
    assert len(missing_processes) == 0, f"Processes in df not in structure: {missing_processes}"

    # every flow process combination exists only once per region
    duplicates = df.duplicated(subset=['Region',  'Flow', 'Process', 'Supply_Use'], keep=False)
    assert not duplicates.any(), f"Duplicate flow-process combinations found in df"

    # no negative values
    assert (df['Value'] >= 0).all(), "Negative values found in missing flows"


def calculate_domestic_flows():
    'In this function we calculate the domestic flows that ultimately result from mass balance'
    df = calc_missing_flows()
    look = lookup()
    

    structure = look['Structure']
    trade = look['Trade']

    trade_look = trade[trade['Value'] == 0]
    structure = structure[structure['Value'] == 1]


    # for every flow in trade_look Flow check what process it belongs to in the structure. If the process is not included in df then copy the existing slice and
    # change the process set all the values to the same value for supply == use .. change supply_use to 'Use' and vice versa
    # this stuff is only possible for domestics flows

    collect = []

    for _, row in trade_look.iterrows():
        flow = row['Flow']
        
        df_flow = df[df['Flow'] == flow]
        processes = df_flow['Process'].unique()

        if len(processes) == 0:
            logger.warning(f"No processes found for flow {flow} in df - skipping bec there is no data")
            continue

        assert 2>= len(processes), f"More than two processes found for flow {flow}"

        # get the processes from the structure
        struct_processes = structure[structure['Flow'] == flow]['Process'].unique()
        missing_processes = set(struct_processes) - set(processes)

        if len(missing_processes) == 0:
            logger.info(f"No missing processes for flow {flow}")
            continue

        for process in missing_processes:

            df_flow_rev = df_flow.copy()
            df_flow_rev['Process'] = process
            df_flow_rev['Supply_Use'] = df_flow_rev['Supply_Use'].apply(lambda x: 'Use' if x == 'Supply' else 'Supply')
            collect.append(df_flow_rev)
    
    df_missing = pd.concat(collect, ignore_index=True)

    df_final = pd.concat([df, df_missing], ignore_index=True)

    # group and sum up
    df_final = df_final.groupby(['Region', 'Flow', 'Process', 'Supply_Use', 'Unit'], as_index=False)['Value'].sum()

    return df_final


def region_consistency_check(df, conc_map):
    '''
    Check if all regions in df are in conc_map
    '''
    df_regions = df['Region'].unique()
    conc_regions = conc_map['Source_name'].unique()
    missing_regions = set(df_regions) - set(conc_regions)
    assert len(missing_regions) == 0, f"Regions in df not in conc_map: {missing_regions}"


def transform_to_region_base(base_path=r'data\input\conc\icsg.xlsx'):
    """
    Transform df regions to conc_map regions.
    
    Rules:
    - 1:1 mapping: region in df maps directly to region in conc_map → keep value as is.
    - 1:n mapping: one region in df maps to multiple in conc_map → split value equally.
    - n:1 mapping: multiple regions in df map to one in conc_map → sum the values.
    """
    
    # Step 1: Load your flows df and the concordance table
    df = calculate_domestic_flows()
    conc_map = pd.read_excel(base_path, sheet_name='Region')

    # region consistency check
    region_consistency_check(df, conc_map)

    # Step 2: Merge df with mapping
    merged = df.merge(conc_map, left_on='Region', right_on='Source_name', how='left')

    # Step 3: Compute mapping counts per Source_name
    counts = conc_map.groupby('Source_name')['Base_name'].count().reset_index()
    counts.rename(columns={'Base_name': 'map_count'}, inplace=True)

    # Merge counts into merged df
    merged = merged.merge(counts, on='Source_name', how='left')

    # Step 4: Handle 1:n mapping → distribute equally
    merged['Value_adj'] = merged['Value'] / merged['map_count']

    # Step 5: Aggregate to final conc_region
    result = (
        merged
        .groupby(['Base_name', 'Flow', 'Process', 'Supply_Use', 'Unit'], as_index=False)
        .agg({'Value_adj': 'sum'})
    )

    result.rename(columns={'Value_adj': 'Value',
    'Base_name': 'Region'}, inplace=True)

    result['Value'] = result.Value.round(3)

    # split into accounting entities
    time_stamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

    use = result[result['Supply_Use'] == 'Use'].copy()
    supply = result[result['Supply_Use'] == 'Supply'].copy()

    o_flows = ['Total_scrap']
    e_flows = ['Crude_ore']
    
    p_flows = ['Waste_rock', 'Tailings', 'Slag']
    y_flows = ['Refined_copper']

    o = use[use['Flow'].isin(o_flows)]
    e = use[use['Flow'].isin(e_flows)]
    y = use[use['Flow'].isin(y_flows)]
    p = supply [supply['Flow'].isin(p_flows)]
    

    use = use[~use['Flow'].isin(o_flows + e_flows + y_flows)]
    supply = supply[~supply['Flow'].isin(p_flows)]

    folder_name = r"data/proc/datafeed/icsg"

    os.makedirs(folder_name, exist_ok=True)

    use.to_csv(os.path.join(folder_name, f'U_prod_icsg_{time_stamp}.csv'), index=False)
    supply.to_csv(os.path.join(folder_name, f'S_prod_icsg_{time_stamp}.csv'), index=False)
    o.to_csv(os.path.join(folder_name, f'O_prod_icsg_{time_stamp}.csv'), index=False)
    e.to_csv(os.path.join(folder_name, f'E_prod_icsg_{time_stamp}.csv'), index=False)
    p.to_csv(os.path.join(folder_name, f'P_prod_icsg_{time_stamp}.csv'), index=False)
    y.to_csv(os.path.join(folder_name, f'Y_prod_icsg_{time_stamp}.csv'), index=False)


    logger.info(f"Transformed data saved to {folder_name}")






####################################################################################################

# functions to create region totals for supply and use entities

####################################################################################################



def use_region_totals(
        origin = r'data\proc\datafeed\icsg',
        dest_folder_tot = r'data\proc\datafeed\icsg\region_totals',
        name_tot = 'U'
    ):

    upaths = ['Y_prod_icsg_20251015_103115.csv', 'U_prod_icsg_20251015_103115.csv',
              'O_prod_icsg_20251015_103115.csv', 'E_prod_icsg_20251015_103115.csv']
    l = lookup()
    ent = l['Sector2Entity']
    dims = [
             'Sector_origin', 'Entity_origin',
            'Region_destination', 'Sector_destination', 'Entity_destination', 'Value'
        ]
    
    
    u = pd.read_csv(os.path.join(origin, upaths[0]))
    y = pd.read_csv(os.path.join(origin, upaths[1]))
    o = pd.read_csv(os.path.join(origin, upaths[2]))
    e = pd.read_csv(os.path.join(origin, upaths[3]))
    use_tot = pd.concat([u, y, o, e], ignore_index=True)

    use_tot.rename(columns={'Region': 'Region_destination', 
                        'Flow': 'Sector_origin',
                        'Process': 'Sector_destination',
                       }, inplace=True)
    use_tot.drop(columns=['Unit',  'Supply_Use'], inplace=True)

    use_tot = use_tot.merge(ent, left_on='Sector_origin', right_on='Sector_name', how='left')
    use_tot = use_tot.rename(columns={'Entity': 'Entity_origin'})
    use_tot = use_tot.merge(ent, left_on='Sector_destination', right_on='Sector_name', how='left')
    use_tot = use_tot.rename(columns={'Entity': 'Entity_destination'})
    use_tot = use_tot[dims]


    time_stamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    dest = os.path.join(dest_folder_tot, f'{name_tot}_{time_stamp}.csv')

    os.makedirs(dest_folder_tot, exist_ok=True)
    use_tot.to_csv(dest, index=False)
    logger.info(f"Use region totals saved to {dest}")

    return None


def supply_region_totals(origin = r'data\proc\datafeed\icsg',
        dest_folder = r'data\proc\datafeed\icsg\region_totals',
        name = 'S'):
     
    spaths = ['S_prod_icsg_20251020_101950.csv', 'P_prod_icsg_20251020_101950.csv',]
    l = lookup()
    ent = l['Sector2Entity']
    dims = [
             'Region_origin', 'Sector_origin', 'Entity_origin',
            'Sector_destination', 'Entity_destination', 'Value'
        ]
    
    s = pd.read_csv(os.path.join(origin, spaths[0]))
    p = pd.read_csv(os.path.join(origin, spaths[1]))
    supply_tot = pd.concat([s, p], ignore_index=True)
    supply_tot.rename(columns={'Region': 'Region_origin', 
                        'Flow': 'Sector_destination',
                        'Process': 'Sector_origin',}, inplace=True)
    
    supply_tot.drop(columns=['Unit', 'Supply_Use'], inplace=True)

    supply_tot = supply_tot.merge(ent, left_on='Sector_origin', right_on='Sector_name', how='left')
    supply_tot = supply_tot.rename(columns={'Entity': 'Entity_origin'})
    supply_tot = supply_tot.merge(ent, left_on='Sector_destination', right_on='Sector_name', how='left')
    supply_tot = supply_tot.rename(columns={'Entity': 'Entity_destination'})
    supply_tot = supply_tot[dims]


    time_stamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    dest = os.path.join(dest_folder, f'{name}_{time_stamp}.csv')
    os.makedirs(dest_folder, exist_ok=True)
    supply_tot.to_csv(dest, index=False)

    logger.info(f"Supply region totals saved to {dest}")

    return None


####################################################################################################

# functions to prepare the boundary input and output in inital estimate format

####################################################################################################

def use_to_ie(
        origin = r'data\proc\datafeed\icsg',
        dest_folder = r'data\proc\datafeed\dom_calc',
    ):

    upaths = ['O_prod_icsg_20251020_101950.csv', 'E_prod_icsg_20251020_101950.csv']

    l = lookup()
    ent = l['Sector2Entity']
    dims = ['Region_origin',
             'Sector_origin', 'Entity_origin',
            'Region_destination', 'Sector_destination', 'Entity_destination', 'Value'
        ]
    
    
    
    o = pd.read_csv(os.path.join(origin, upaths[0]))
    e = pd.read_csv(os.path.join(origin, upaths[1]))
    

    o.rename(columns={'Region': 'Region_destination', 
                        'Flow': 'Sector_origin',
                        'Process': 'Sector_destination',
                       }, inplace=True)
    e.rename(columns={'Region': 'Region_destination', 
                        'Flow': 'Sector_origin',
                        'Process': 'Sector_destination',
                       }, inplace=True)

    
    o = merge_entity_to_df(o, ent)
    e = merge_entity_to_df(e, ent)
    o['Region_origin'] = o['Region_destination']
    e['Region_origin'] = e['Region_destination']
    o = o[dims]
    e = e[dims]
    

    time_stamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

    os.makedirs(dest_folder, exist_ok=True)

    dest_o = os.path.join(dest_folder, f'O_{time_stamp}.csv')
    dest_e = os.path.join(dest_folder, f'E_{time_stamp}.csv')

    o.to_csv(dest_o, index=False)
    e.to_csv(dest_e, index=False)

    logger.info(f"Use region totals saved to {dest_o} and {dest_e}")
    return None



def merge_entity_to_df(df, ent):
        df = df.merge(ent, left_on='Sector_origin', right_on='Sector_name', how='left')
        df = df.rename(columns={'Entity': 'Entity_origin'})
        df = df.merge(ent, left_on='Sector_destination', right_on='Sector_name', how='left')
        df = df.rename(columns={'Entity': 'Entity_destination'})
        return df


def supply_to_ie(
        origin = r'data\proc\datafeed\icsg',
        dest_folder = r'data\proc\datafeed\dom_calc',
    ):

    upaths = ['P_prod_icsg_20251020_101950.csv']

    l = lookup()
    ent = l['Sector2Entity']
    dims = ['Region_origin',
             'Sector_origin', 'Entity_origin',
            'Region_destination', 'Sector_destination', 'Entity_destination', 'Value'
        ]
    
    p = pd.read_csv(os.path.join(origin, upaths[0]))
    

    p.rename(columns={'Region': 'Region_origin', 
                        'Flow': 'Sector_destination',
                        'Process': 'Sector_origin',
                       }, inplace=True)
    
    p = merge_entity_to_df(p, ent)
    p['Region_destination'] = p['Region_origin']
    
    
    p = p[dims]

    time_stamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

    os.makedirs(dest_folder, exist_ok=True)

    dest_p = os.path.join(dest_folder, f'P_{time_stamp}.csv')
    

    p.to_csv(dest_p, index=False)

    logger.info(f"Supply region totals saved to {dest_p}")


    return None


# Tranform the data feed into 
if __name__ == "__main__":
    supply_to_ie()
    