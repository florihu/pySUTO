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
from model.core.datafeed.util import lookup

def calc_domestic_trade_flows(noise_zero =10**-6):

    """
    Calculate domestic trade flows by adjusting supply and use values from ICSG data
    with import and export data from BACI dataset.
    
    """
    # --- Load data ---
    folder_name = r'data/proc/datafeed'

    u_baci_path = r'baci\U_withoutdomestic_baci_20250923_083406.csv'
    y_baci_path = r'baci\Y_baci_20250923_083406.csv'
    u_icsg_path = r'icsg\region_totals\U_20251015_110046.csv'
    s_icsg_path = r'icsg\region_totals\S_20251020_102243.csv'

    u_baci_path = os.path.join(folder_name, u_baci_path)
    u_icsg_path = os.path.join(folder_name, u_icsg_path)
    y_baci_path = os.path.join(folder_name, y_baci_path)
    s_icsg_path = os.path.join(folder_name, s_icsg_path)

    u_baci = pd.read_csv(u_baci_path)
    u_icsg = pd.read_csv(u_icsg_path)
    y_baci = pd.read_csv(y_baci_path)
    s_icsg = pd.read_csv(s_icsg_path)

    # approx Total Supply = Total Use
    y_icsg = s_icsg[s_icsg['Sector_destination'] == 'Refined_copper'].copy()

    y_icsg['Sector_origin'] = 'Refined_copper'
    y_icsg['Entity_origin'] = 'Flow'
    y_icsg['Sector_destination'] = 'SEM'
    y_icsg['Entity_destination'] = 'SB'
    y_icsg.rename(columns={'Region_origin': 'Region_destination'}, inplace=True)

    baci = pd.concat([u_baci, y_baci], ignore_index=True)
    icsg = pd.concat([u_icsg, y_icsg], ignore_index=True)

    # Calculate imports from baci
    imports = (
        baci.groupby(["Region_destination", "Sector_origin"])["Value"]
        .sum()
        .reset_index()
        .rename(columns={'Value': 'Imports'})
    )

    # exports baci
    exports = (
        baci.groupby(["Region_origin", "Sector_origin"])["Value"]
        .sum()
        .reset_index()
        .rename(columns={'Value': 'Exports'})
    )


    # total use per region and flows
    icsg_grouped = (
        icsg.groupby(["Region_destination", "Sector_origin"])["Value"]
        .sum()
        .reset_index()
    )

    icsg_grouped.rename(columns={"Value": "Total_use"}, inplace=True)

    
    # merge imports and exports
    merged = icsg_grouped.merge(imports[['Region_destination', 'Sector_origin', 'Imports']], left_on=['Region_destination', 'Sector_origin'],
                               right_on=['Region_destination', 'Sector_origin'],
                               how='left', validate='1:1')
    merged = merged.merge(exports[['Region_origin', 'Sector_origin', 'Exports']], left_on=['Region_destination', 'Sector_origin'],
                            right_on=['Region_origin', 'Sector_origin'],
                            how='left', validate='1:1')
    merged.drop(columns=['Region_origin'], inplace=True)

    lookups = lookup()

    trade_lookup = lookups['Trade']

    trade_flows = trade_lookup[trade_lookup['Value'] == 1]['Flow'].unique()
    merged = merged[merged['Sector_origin'].isin(trade_flows)]
    merged['Imports'] = merged.Imports.fillna(0)
    merged['Exports'] = merged.Exports.fillna(0)

    # Calculate domestic use
    merged['Domestic_Use'] = merged['Total_use'] - merged['Imports'] + merged['Exports']

    merged.loc[merged['Domestic_Use'] < noise_zero, 'Domestic_Use'] = 0
    

    # get structure lookup
    structure = lookups['Structure']
    structure = structure[structure['Value'] == 1]
    structure = structure[(structure['Flow'].isin(trade_flows)) & (structure['Supply_Use'] == 'Use')]  
    
    # --- Get things into supply use logic ---
    
    # merge domestic use with structure filtere use
    dom = merged.merge(structure[['Flow', 'Process']], left_on=['Sector_origin'], right_on=['Flow'], how='left', validate='m:1')
    dom.drop(columns=['Flow'], inplace=True)

   
    dom_col_rename = { 
                     'Process': 'Sector_destination',
                     'Domestic_Use': 'Value'}

    dom = dom.rename(columns=dom_col_rename)

    dom['Region_origin'] = dom['Region_destination']

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




if __name__ == "__main__":
    calc_domestic_trade_flows()
    
