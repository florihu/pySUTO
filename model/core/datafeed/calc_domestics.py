import os
import sparse
import logging
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import sys
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Now use an absolute import
from model.util import clean_cols, read_concordance_table
from model.core.datafeed.DataFeed import lookup

def calc_domestic_trade_flows(noise_zero =10**-6):

    """
    Calculate domestic trade flows by adjusting supply and use values from ICSG data
    with import and export data from BACI dataset.
    
    """
    # --- Load data ---
    folder_name = r'data/proc/datafeed'

    u_baci_path = r'baci\U_withoutdomestic_baci_20250923_083406.csv'
    y_baci_path = r'baci\Y_baci_20250923_083406.csv'
    u_icsg_path = r'icsg\U_prod_icsg_20250923_093733.csv'
    y_icsg_path = r'icsg\Y_prod_icsg_20250923_093733.csv'

    u_baci_path = os.path.join(folder_name, u_baci_path)
    u_icsg_path = os.path.join(folder_name, u_icsg_path)
    y_baci_path = os.path.join(folder_name, y_baci_path)
    y_icsg_path = os.path.join(folder_name, y_icsg_path)

    u_baci = pd.read_csv(u_baci_path)
    u_icsg = pd.read_csv(u_icsg_path)
    y_baci = pd.read_csv(y_baci_path)
    y_icsg = pd.read_csv(y_icsg_path)

    baci = pd.concat([u_baci, y_baci], ignore_index=True)
    icsg = pd.concat([u_icsg, y_icsg], ignore_index=True)

    # Calculate imports from baci
    imports = (
        baci.groupby(["Region_destination", "Sector_origin", 'Entity_origin'])["Value"]
        .sum()
        .reset_index()
        .rename(columns={
            "Region_destination": "Region",
            "Sector_origin": "Flow",
            "Entity_destination": "Entity",
            "Value": "Imports"
        })
    )

    # exports baci
    exports = (
        baci.groupby(["Region_origin", "Sector_origin"])["Value"]
        .sum()
        .reset_index()
        .rename(columns={
            "Region_origin": "Region",
            "Sector_origin": "Flow",
            "Value": "Exports"
        })
    )

    # total use per region and flows

    icsg_grouped = (
        icsg.groupby(["Region", "Flow"])["Value"]
        .sum()
        .reset_index()
    )

    icsg_grouped.rename(columns={"Value": "Total_use"}, inplace=True)

    # merged = icsg_grouped.merge(imports[['Region', 'Flow', 'Imports']], on=["Region", "Flow"], how="left")
    # merge imports and exports
    merged = icsg_grouped.merge(imports[['Region', 'Flow', 'Imports']], on=["Region", "Flow"], how="left")
    merged = merged.merge(exports[['Region', 'Flow', 'Exports']], on=["Region", "Flow"], how="left")

    lookups = lookup()

    trade_lookup = lookups['Trade']

    trade_flows = trade_lookup[trade_lookup['Value'] == 1]['Flow'].unique()
    merged = merged[merged['Flow'].isin(trade_flows)]
    merged['Imports'] = merged.Imports.fillna(0)
    merged['Exports'] = merged.Exports.fillna(0)

    # Calculate domestic use
    merged['Domestic_Use'] = merged['Total_use'] - merged['Imports'] + merged['Exports']

    merged.loc[merged['Domestic_Use'] < noise_zero, 'Domestic_Use'] = 0
    

    # get structure lookup
    structure = lookups['Structure']
    structure = structure[structure['Value'] == 1]
    
    # --- Get things into supply use logic ---
    
    # merge domestic use with structure filtere use
    dom = merged.merge(structure[structure['Supply_Use'] == 'Use'][['Flow', 'Process']], on='Flow', how='left')

   
    dom_col_rename = { 'Region': 'Region_origin',
                     'Flow': 'Sector_origin',
                     'Process': 'Sector_destination',
                     'Domestic_Use': 'Value'}

    dom = dom.rename(columns=dom_col_rename)

    dom['Region_destination'] = dom['Region_origin']

    sec_to_ent = lookups['Sector2Entity']

    dom = dom.merge(sec_to_ent, left_on='Sector_origin', right_on='Sector_name', how='left')
    dom = dom.rename(columns={'Entity': 'Entity_origin'})
    dom = dom.merge(sec_to_ent, left_on='Sector_destination', right_on='Sector_name', how='left')
    dom = dom.rename(columns={'Entity': 'Entity_destination'})
    dom = dom.drop(columns=['Sector_name_x', 'Sector_name_y'])

    col_order = ['Region_origin',  'Sector_origin', 'Entity_origin', 'Region_destination',  'Sector_destination', 'Entity_destination', 'Value']
    dom = dom[col_order]

    # filter out zero values
    dom = dom[dom['Value'] > 0]

    y = dom[dom.Sector_origin == 'Refined_copper'].copy()
    u = dom[dom.Sector_origin != 'Refined_copper'].copy()



    folder = 'data/proc/datafeed/dom_calc'
    os.makedirs(folder, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

    u_path = os.path.join(folder, f'U_domestic_{timestamp}.csv')
    y_path = os.path.join(folder, f'Y_domestic_{timestamp}.csv')
    u.to_csv(u_path, index=False)
    y.to_csv(y_path, index=False)
    

    return None

# todo : We need to do the domestic final demand calc as well for final demand... can also be traded.



#def dom_to_sup_use():
    columns = ['Region_origin', 'Sector_origin', 'Region_destination', 'Sector_destination', 'Value']

    icsg_path = r'data\processed\data_feed\ie\icsg_ie_v1.csv'
    icsg = pd.read_csv(icsg_path)

    lookups = lookup()
    trade = lookups['Trade']
    structure = lookups['Structure']
    entity = lookups['Entity']


    non_trade_flows = trade[trade['Value'] == 0]['Flow'].unique()

    icsg = icsg[icsg['Flow'].isin(non_trade_flows)]

    #devide supply and use
    supply = icsg[icsg['Supply_Use'] == 'Supply'].copy()

    supply_rename = {'Region': 'Region_destination',
                        'Flow': 'Sector_destination',
                        'Process': 'Sector_origin',
                        'Value': 'Value'}
    supply = supply.rename(columns=supply_rename)
    supply['Region_origin'] = supply['Region_destination']

    supply = supply[columns]

    use = icsg[icsg['Supply_Use'] == 'Use'].copy()
    use_rename = {'Region': 'Region_origin',
                     'Flow': 'Sector_origin',
                     'Process': 'Sector_destination',
                     'Value': 'Value'}
    use = use.rename(columns=use_rename)
    use['Region_destination'] = use['Region_origin']
    use = use[columns]

    domestic = pd.concat([supply, use], ignore_index=True)

    # add entity
    secent = entity[['Sector_name', 'Entity']]
    domestic = domestic.merge(secent, left_on='Sector_origin', right_on='Sector_name', how='left')
    domestic = domestic.rename(columns={'Entity': 'Entity_origin'})
    domestic = domestic.merge(secent, left_on='Sector_destination', right_on='Sector_name', how='left')
    domestic = domestic.rename(columns={'Entity': 'Entity_destination'})
    domestic = domestic.drop(columns=['Sector_name_x', 'Sector_name_y'])

    final_col_order = ['Region_origin',  'Sector_origin', 'Entity_origin', 'Region_destination',  'Sector_destination', 'Entity_destination', 'Value']
    domestic = domestic[final_col_order]

    out_path = r'data\processed\data_feed\ie\domestic_ie_v1.csv'
    domestic.to_csv(out_path, index=False)
    return None



if __name__ == "__main__":
    calc_domestic_trade_flows()
    
