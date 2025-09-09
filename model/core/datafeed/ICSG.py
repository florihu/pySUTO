import sys
import os
import pandas as pd
import numpy as np

from DataFeed import sectorial_lookup_table



# Add the core directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from util import get_path, read_concordance_table




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
        df = df[~df['Flow'].str.contains(',|\.', na=False)]

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
        df = df[~df['Flow'].str.contains(',|\.', na=False)]

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
        df = df[~df['Flow'].str.contains(',|\.', na=False)]

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


def read_icsg_all_years(out_path = r'data\processed\data_feed' ):
     # Collect data from different years
    df_2023 = icsg_2023()
    df_2014 = icsg_2014()
    df_2005 = icsg_2005()

    # assert the columns are identical
    assert set(df_2023.columns) == set(df_2014.columns) == set(df_2005.columns), "DataFrames have different columns"

    # Concatenate all dataframes
    df = pd.concat([df_2023, df_2014, df_2005], ignore_index=True)


    # remove all spaces at the end of the region column
    df['Region'] = df['Region'].str.strip()


    # from : to
    reg_rename =  {
    'Belgium-': 'Belgium-Luxembourg',
    'Belgium- Luxembourg': 'Belgium-Luxembourg',

    'Congo Rep.': 'Congo',
    'Congo DR': 'Congo, D.R.', 

    'Philippppines': 'Philippines',

    'United': 'United States',

    }

    flow_rename = {'Copper ': 'Copper', 
    'SX-EW': 'Refinery SXEW',
    'Concentrates': 'Mine Concentrates',
    'Smelter Primary': 'Smelting Primary'}

    df.replace({'Region': reg_rename}, inplace=True)

    df.replace({'Flow': flow_rename}, inplace=True)

    process_rename = {'Smelter': 'Smelting'}
    df.replace({'Process': process_rename}, inplace=True)


    # For Flows ['Primary', 'Secondary'] combine with the Process name of the row
    df['Flow'] = df.apply(
        lambda x: f"{x['Process']} {x['Flow']}" if x['Flow'] in ['Primary', 'Secondary', 'Low-grade', 'Electrowon'] else x['Flow'],
        axis=1
    )

    df = df[~df['Flow'].isin(['Smelting Electrowon'])]

    flow_rename_2 = {'Refining Electrowon': 'Cathode_SXEW'}

    df.replace({'Flow': flow_rename_2}, inplace=True)

    # save df to csv

    df.to_csv(os.path.join(out_path, 'ICSG_clean_v1.csv'), index=False)
    
    return df


def calc_missing_flows(tc_path = r'data\input\raw\TCs.xlsx',
                            df_path = r'data\processed\data_feed\ie\ICSG_clean_v1.csv',
                            conc_path = r'data\input\raw\Conc.xlsx'):

    tc = pd.read_excel(tc_path)
    df = pd.read_csv(df_path)
    conc = pd.read_excel(conc_path)

    reg_rename =  {'Korean Rep. ': 'Korea Rep.'}
    df.replace({'Region': reg_rename}, inplace=True)

    not_needed_flows = ['Wire Rod', 'Copper', 'Cu Alloy']
    year = 1995

    df = df[df['Year'] == year]
    df = df[~df['Flow'].isin(not_needed_flows)]


    # PYRO
    # calculate waste rock - ore
    df_ore = df[df['Flow'] == 'Mine Concentrates'].copy()
    df_ore['Flow'] = 'Ore'
    df_ore['Value'] = df_ore['Value'] / tc.loc[tc['Process'] == 'Milling_Crushing_Floatation', 'Value'].iloc[0]
    df_ore['Value'] = df_ore['Value'] / conc.loc[conc['Flow'] == 'Ore', 'Value'].iloc[0]
    df_ore['Supply_Use'] = 'Use'
    df_ore['Process'] = 'Milling_Crushing_Floatation'

    # Concentrates
    df_conc = df[df['Flow'] == 'Mine Concentrates'].copy()
    df_conc['Flow'] = 'Concentrates'
    df_conc['Value'] = df_conc['Value'] / conc.loc[conc['Flow'] == 'Concentrate', 'Value'].iloc[0]
    df_conc['Supply_Use'] = 'Supply'
    df_conc['Process'] = 'Mining'
  
    
    # Tailings
    df_tailings = df[df['Flow'] == 'Mine Concentrates'].copy()
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
    df_cathode_hydro = df[df['Flow'] == 'Cathode_SXEW'].copy()
    df_cathode_hydro['Flow'] = 'Refined_copper_hydro'
    df_cathode_hydro['Process'] = 'SXEW'
    df_cathode_hydro['Supply_Use'] = 'Supply'
    df_cathode_hydro['Value'] = df_cathode_hydro['Value'] / conc.loc[conc['Flow'] == 'Refined_copper_hydro', 'Value'].iloc[0]

    # Tailings
    df_tailings_2 = df[df['Flow'] == 'Cathode_SXEW'].copy()
    df_tailings_2['Flow'] = 'Tailings'
    df_tailings_2['Value'] = df_tailings_2['Value'] * (1/tc.loc[tc['Process'] == 'SXEW', 'Value'].iloc[0]-1)
    df_tailings_2['Value'] = df_tailings_2['Value'] / conc.loc[conc['Flow'] == 'Tailings', 'Value'].iloc[0]
    df_tailings_2['Supply_Use'] = 'Supply'
    df_tailings_2['Process'] = 'SXEW'

    
    # Ore
    df_ore_2 = df[df['Flow'] == 'Cathode_SXEW'].copy()
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


    df_smelting_primary = df[df['Flow'].isin(['Smelting Primary', 'Smelting Low-grade'])].copy()
    df_smelting_primary['Flow'] = 'Concentrates'
    # gropby concentrates and sum up -> we do not differentiate between low grade and concentrates
    df_smelting_primary = df_smelting_primary.groupby(['Region', 'Year', 'Flow', 'Process', 'Supply_Use', 'Unit']).sum().reset_index()
    df_smelting_primary['Value'] = df_smelting_primary['Value'] / tc.loc[tc['Process'] == 'Smelting', 'Value'].iloc[0]
    df_smelting_primary['Value'] = df_smelting_primary['Value'] / conc.loc[conc['Flow'] == 'Concentrate', 'Value'].iloc[0]
    df_smelting_primary['Process'] = 'Smelting'
    df_smelting_primary['Supply_Use'] = 'Use'


    df_scrap_class_3 = df[df['Flow'] == 'Smelting Secondary'].copy()
    df_scrap_class_3['Flow'] = 'Scrap_class_3'
    df_scrap_class_3['Value'] = df_scrap_class_3['Value'] / tc.loc[tc['Process'] == 'Smelting', 'Value'].iloc[0]
    df_scrap_class_3['Value'] = df_scrap_class_3['Value'] / conc.loc[conc['Flow'] == 'Scrap_class_3', 'Value'].iloc[0]
    df_scrap_class_3['Process'] = 'Smelting'
    df_scrap_class_3['Supply_Use'] = 'Use'


    df_matte = df[df['Flow'].isin(['Smelting Primary', 'Smelting Low-grade', 'Smelting Secondary'])].copy()
    df_matte['Flow'] = 'Matte'
    df_matte = df_matte.groupby(['Region', 'Year', 'Flow', 'Process', 'Supply_Use', 'Unit']).sum().reset_index()
    df_matte['Value'] = df_matte['Value'] / conc.loc[conc['Flow'] == 'Matte', 'Value'].iloc[0]
    df_matte['Supply_Use'] = 'Supply'

    # calculate slag via difference slag = concentrates + scrap - matte
    df_slag = df[df['Flow'].isin(['Smelting Primary', 'Smelting Low-grade', 'Smelting Secondary'])].copy()
    df_slag['Flow'] = 'Slag'
    df_slag = df_slag.groupby(['Region', 'Year', 'Flow', 'Process', 'Supply_Use', 'Unit']).sum().reset_index()
    df_slag['Value'] = df_slag['Value'] * (1/tc.loc[tc['Process'] == 'Smelting', 'Value'].iloc[0]-1)
    df_slag['Value'] = df_slag['Value'] / conc.loc[conc['Flow'] == 'Slag', 'Value'].iloc[0]
    df_slag['Supply_Use'] = 'Supply'

    

    df_anode = df[df['Flow'].isin(['Smelting Primary', 'Smelting Low-grade', 'Smelting Secondary'])].copy()
    df_anode['Flow'] = 'Anode'
    df_anode['Value'] = df_anode['Value'] * tc.loc[tc['Process'] == 'Converting_fire_refining', 'Value'].iloc[0]
    df_anode['Value'] = df_anode['Value'] / conc.loc[conc['Flow'] == 'Anode', 'Value'].iloc[0]
    df_anode['Process'] = 'Converting_fire_refining'
    df_anode['Supply_Use'] = 'Supply'

    # calculate slag via difference slag = anode - matte
    df_slag_2 = df[df['Flow'].isin(['Smelting Primary', 'Smelting Low-grade', 'Smelting Secondary'])].copy()
    df_slag_2['Flow'] = 'Slag'
    df_slag_2['Value'] = df_slag_2['Value'] * (1-tc.loc[tc['Process'] == 'Converting_fire_refining', 'Value'].iloc[0])
    df_slag_2['Value'] = df_slag_2['Value'] / conc.loc[conc['Flow'] == 'Slag', 'Value'].iloc[0]
    df_slag_2['Process'] = 'Converting_fire_refining'
    df_slag_2['Supply_Use'] = 'Supply'



    df_cathode_pyro = df[df['Flow'].isin(['Refining Primary', 'Refining Secondary'])].copy()
    df_cathode_pyro['Flow'] = 'Refined_copper_pyro'
    df_cathode_pyro = df_cathode_pyro.groupby(['Region', 'Year', 'Flow', 'Process', 'Supply_Use', 'Unit']).sum().reset_index()
    df_cathode_pyro['Value'] = df_cathode_pyro['Value'] * conc.loc[conc['Flow'] == 'Refined_copper_hydro', 'Value'].iloc[0]
    df_cathode_pyro['Process'] = 'Electrolytic_refining'
    df_cathode_pyro['Supply_Use'] = 'Supply'


    df_anode_2 = df[df['Flow'] == 'Refining Primary'].copy()
    df_anode_2['Flow'] = 'Anode'
    df_anode_2['Value'] = df_anode_2['Value'] / conc.loc[conc['Flow'] == 'Anode', 'Value'].iloc[0]
    df_anode_2['Process'] = 'Electrolytic_refining'
    df_anode_2['Supply_Use'] = 'Use'

    
    df_scrap_class_2 = df[df['Flow'] == 'Refining Secondary'].copy()
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

    # groupby flow and sum
    df_mine = df_mine.groupby(['Region', 'Year', 'Flow', 'Process', 'Supply_Use', 'Unit']).sum().reset_index()

    df_tot_scrap = pd.concat([df_scrap_class_2, df_scrap_class_3], ignore_index=True)
    df_tot_scrap['Flow'] = 'Total_scrap'
    df_tot_scrap = df_tot_scrap.groupby(['Region', 'Year', 'Flow', 'Process', 'Supply_Use', 'Unit']).sum().reset_index()
    df_tot_scrap['Process'] = 'Scrap'
    df_tot_scrap['Supply_Use'] = 'Use'

    # Concatenate all dataframes
    df_missing_flows = pd.concat([df_mine, df_ore, df_conc, df_tailings, df_cathode_hydro, df_tailings_2,
                                df_ore_2, df_smelting_primary, df_scrap_class_3, df_matte, df_slag, df_anode,
                                df_slag_2, df_cathode_pyro, df_anode_2, df_scrap_class_2, df_cathode, df_tot_scrap], ignore_index=True)
    
    # filter out zeros
    df_missing_flows = df_missing_flows[df_missing_flows['Value'] != 0]

    # assert no negative values
    assert (df_missing_flows['Value'] >= 0).all(), "Negative values found in missing flows"

    ren = {'Waste Rock': 'Waste_rock',
        'Concentrates': 'Concentrate',
        'Crude Ore': 'Crude_ore'}

    df_missing_flows.replace({'Flow': ren}, inplace=True)



    return df_missing_flows

def sectorial_lookup_table():
    structure_path = r'data\input\structure.xlsx'
    structure = pd.read_excel(structure_path, sheet_name=None)

    # Example: work with the "Supply" sheet
    supply = structure['Supply']

    # First column = Flow
    supply = supply.rename(columns={supply.columns[0]: "Process"})

    # Melt matrix into long format: Flow, Process, Value
    supply_long = supply.melt(
        id_vars=["Process"],          # keep flows fixed
        var_name="Flow",        # column headers become "Process"
        value_name="Value"         # cell values
    )
    supply_long['Supply_Use'] = 'Supply'
    
    use = structure['Use']
    use = use.rename(columns={use.columns[0]: "Flow"})
    use_long = use.melt(
        id_vars=["Flow"],          # keep flows fixed
        var_name="Process",        # column headers become "Process"
        value_name="Value"         # cell values
    )

    use_long['Supply_Use'] = 'Use'

    # sort columns
    supply_long = supply_long[['Flow', 'Process', 'Supply_Use', 'Value']]
    use_long = use_long[['Flow', 'Process', 'Supply_Use', 'Value']]

    # concat
    structure_long = pd.concat([supply_long, use_long], ignore_index=True)

    return structure_long

def flow_process_check(df, structure):
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


def calculate_domestic_flows():
    df = calc_missing_flows()
    structure = sectorial_lookup_table()
    flow_process_check(df, structure)

    structure_path = r'data\input\structure.xlsx'

    trade_look = pd.read_excel(structure_path, sheet_name='Trade')

    trade_look = trade_look[trade_look.Value ==0]

    structure = structure[structure['Value'] == 1]


    # for every flow in trade_look Flow check what process it belongs to in the structure. If the process is not included in df then copy the existing slice and
    # change the process set all the values to the same value for supply == use .. change supply_use to 'Use' and vice versa

    collect = []

    for _, row in trade_look.iterrows():
        flow = row['Flow']
        
        df_flow = df[df['Flow'] == flow]
        processes = df_flow['Process'].unique()

        assert 2>= len(processes) > 0, f"More than two or none processes found for flow {flow}"
        
        # only if there is one process we need to add the missing one
        if len(processes) == 2:
            continue

        # get the processes from the structure
        struct_processes = structure[structure['Flow'] == flow]['Process'].unique()
        missing_processes = set(struct_processes) - set(processes)

        if len(missing_processes) == 0:
            continue


        for process in missing_processes:

            df_flow_rev = df_flow.copy()
            df_flow_rev['Process'] = process
            df_flow_rev['Supply_Use'] = df_flow_rev['Supply_Use'].apply(lambda x: 'Use' if x == 'Supply' else 'Supply')
            collect.append(df_flow_rev)
    
    df_missing = pd.concat(collect, ignore_index=True)

    df_final = pd.concat([df, df_missing], ignore_index=True)

    return df_final

        

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
        .groupby(['Base_name', 'Year', 'Flow', 'Process', 'Supply_Use', 'Unit'], as_index=False)
        .agg({'Value_adj': 'sum'})
    )

    result.rename(columns={'Value_adj': 'Value',
    'Base_name': 'Region'}, inplace=True)

    # /todo rename: cathode hydro pyro etc. there seems to be also an error especially for SXEW are duplicated values etc.

    #save to csv
    out_path = r'data\processed\data_feed\ie\icsg_ie_v1.csv'
    result.to_csv(out_path, index=False)
    
    return result




# Tranform the data feed into 
if __name__ == "__main__":
    transform_to_region_base()
    