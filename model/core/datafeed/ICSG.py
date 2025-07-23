import sys
import os
import pandas as pd
import numpy as np

from DataFeed import DataFeed



# Add the core directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from util import get_path


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


def main():
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

    df.replace({'Region': reg_rename}, inplace=True)

    
    
    return df

def data_feed_friendly():

    df = main()

    df['Scenario'] = 'Ex_post'
    
    

    supply = df[df['Supply_Use'] == 'Supply'].copy()
    use = df[df['Supply_Use'] == 'Use'].copy()

    rename_supply = {'Flow': 'Sector_destination',
                        'Region': 'Region_origin',
                        'Process': 'Sector_origin',}


    supply['Entity_destination'] = 'Flow'
    supply['Entity_origin'] = 'Process'
    supply['Layer'] = 'Copper_mass'
    supply['Region_destination'] = 'NULL'
 

    # rename supply columns
    supply.rename(columns=rename_supply, inplace=True)



    rename_use = {'Flow': 'Sector_destination',
                        'Region': 'Region_destination',
                        'Process': 'Sector_origin'}

    use['Entity_destination'] = 'Process'
    use['Entity_origin'] = 'Flow'
    use['Layer'] = 'Copper_mass'
    supply['Region_origin'] = 'NULL'

    # rename use columns
    use.rename(columns=rename_use, inplace=True)



    path = r'data\processed\ie'
    

    # save supply and use dataframes to csv files
    supply.to_csv(os.path.join(path, 'ICSG_supply.csv'), index=False)
    use.to_csv(os.path.join(path, 'ICSG_use.csv'), index=False)


if __name__ == "__main__":
    data_feed_friendly()
    