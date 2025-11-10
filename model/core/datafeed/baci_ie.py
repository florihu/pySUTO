import os
import logging
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import sys
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Now use an absolute import
from model.util import clean_cols, read_concordance_table, folder_name_check
from model.core.datafeed.DataFeed import lookup

baci_rename = {
            't': 'Year',
            'k': 'HS92_Code',
            'i': 'Region_origin',
            'j': 'Region_destination',
            'v': 'Monetary_value',
            'q': 'Quantity',
        }
    


def baci_read():
    """
    This function reads the BACI data and converts it into a standardized format.
    """
    path = r'data\input\raw\BACI_HS92_V202501\BACI_HS92_Y1995_V202501.csv'
    conc_path = r'data\input\conc\baci.xlsx'

    # read all conc sheets
    conc_sheets = pd.read_excel(conc_path, sheet_name=None)

    relevant_sector_codes = conc_sheets['Sector']['Source_id'].astype(str).tolist()

    baci = pd.read_csv(path)

    baci = baci.rename(columns=baci_rename)

    # filter relevant sector codes
    baci = baci[baci['HS92_Code'].astype(str).isin(relevant_sector_codes)]

    return baci

def imputation_baci():

    # calculate the mean rate Quantity/Monetary_value per HS92_Code
    df = baci_read()
    df['Price'] = df['Monetary_value'] / df['Quantity'] 
    #calculate mean and std
    price = (
        df.groupby(['HS92_Code', 'Region_destination'])['Price']
        .agg(['mean', 'std'])
        .reset_index()
    )
    #rename
    price = price.rename(columns={'mean': 'Mean_price', 'std': 'Std_price'})

    # merge on df
    df = df.merge(price, on=['HS92_Code', 'Region_destination'], how='left')
    # calculate missing Quantity if quantity is missing
    df['Quantity_imputed'] = np.where(df['Quantity'].isna(), df['Monetary_value'] / df['Mean_price'], df['Quantity'])
    
    # filter out flows that are still na
    df = df[~df['Quantity_imputed'].isna()]

    # drop uncecessary columns
    df = df.drop(columns=['Price', 'Mean_price', 'Std_price', 'Quantity'])
    # rename
    df = df.rename(columns={'Quantity_imputed': 'Quantity'})


    return df

def investigate_for_outlier():
    fig_base_folder = r'figs\explo'
    df = imputation_baci()
    # do per hs code a histogram of quantity in a sns facet plot
    g = sns.FacetGrid(df, col='HS92_Code', col_wrap=4, sharex=False, sharey=False)
    g.map(sns.histplot, 'Quantity', bins=30)
    
    plt.tight_layout()
    plt.show()


    plt_path = os.path.join(fig_base_folder, 'baci_quantity_hist.png')
    # save
    plt.savefig(plt_path)


    # do the same for log(Quantity)

    df['Log_Quantity'] = np.log1p(df['Quantity'])

    g = sns.FacetGrid(df, col='HS92_Code', col_wrap=4, sharex=False, sharey=False)
    g.map(sns.histplot, 'Log_Quantity', bins=30)
    plt.tight_layout()
    plt.show()
    plt_path = os.path.join(fig_base_folder, 'baci_log_quantity_hist.png')
    plt.savefig(plt_path)
    return None


# def sector_region_to_base():
#     conc_path = r'data\input\conc\baci.xlsx'
#     sector_conc = pd.read_excel(conc_path, sheet_name='Sector')
#     region_conc = pd.read_excel(conc_path, sheet_name='Region')

#     df = imputation_baci()

#     # ---- SECTOR PART ----
#     df = df.merge(sector_conc, left_on='HS92_Code', right_on='Source_id', how='left')

#     weights = (
#         sector_conc.groupby('Source_id')
#         .size().reset_index(name='n_source')
#     )
#     weights['weight'] = 1 / weights['n_source']

#     df = df.merge(weights[['Source_id', 'weight']], on='Source_id', how='left')

#     df['Weighted_Quantity'] = df['Quantity'] * df['weight']
#     df['Weighted_Monetary_value'] = df['Monetary_value'] * df['weight']

#     df = df.groupby(
#         ['Year', 'Region_origin', 'Region_destination', 'Base_name'],
#         as_index=False
#     )[['Weighted_Quantity', 'Weighted_Monetary_value']].sum()

#     # ---- REGION PART ----
#     # Merge origin concordance
#     df = df.merge(region_conc.rename(
#         columns={'Source_id': 'Region_origin', 'Base_name': 'Region_origin_base'}),
#         on='Region_origin', how='left'
#     )

#     # Merge destination concordance
#     df = df.merge(region_conc.rename(
#         columns={'Source_id': 'Region_destination', 'Base_name': 'Region_destination_base'}),
#         on='Region_destination', how='left'
#     )

#     # ---- REGION WEIGHTS ----
#     region_weights = (
#         region_conc.groupby('Source_id')
#         .size().reset_index(name='n_source')
#     )
#     region_weights['weight_region'] = 1 / region_weights['n_source']

#     # Merge weights for origin
#     df = df.merge(
#         region_weights.rename(columns={
#             'Source_id': 'Region_origin',
#             'weight_region': 'weight_origin'
#         }),
#         on='Region_origin', how='left'
#     )

#     # Merge weights for destination
#     df = df.merge(
#         region_weights.rename(columns={
#             'Source_id': 'Region_destination',
#             'weight_region': 'weight_destination'
#         }),
#         on='Region_destination', how='left'
#     )
#     # Calculate combined weight
#     df['Combined_weight'] = df['weight_origin'] * df['weight_destination']
#     # Apply combined weight to the weighted quantities and monetary values
#     df['Final_Weighted_Quantity'] = df['Weighted_Quantity'] * df['Combined_weight']
#     df['Final_Weighted_Monetary_value'] = df['Weighted_Monetary_value'] * df['Combined_weight']
#     # Final aggregation
#     df = df.groupby(
#         ['Year', 'Region_origin_base', 'Region_destination_base', 'Base_name'],
#         as_index=False
#     )[['Final_Weighted_Quantity', 'Final_Weighted_Monetary_value']].sum()
#     # Rename columns to final names
#     df = df.rename(columns={
#         'Region_origin_base': 'Region_origin',
#         'Region_destination_base': 'Region_destination',
#         'Base_name': 'Sector',
#         'Final_Weighted_Quantity': 'Quantity',
#         'Final_Weighted_Monetary_value': 'Monetary_value'
#     })

#     df['Quantity'] = (df['Quantity'].astype(float)  / 10**3).round(3) # in kt

#     return df

def sector_region_to_base():
    """
    This function aggregates the BACI data to base sectors and regions using a weighted approach.
    It first merges the BACI data with sector and region concordance tables, then applies weights to account for
    many-to-one mappings, and finally aggregates the data to the base level.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the aggregated trade data with columns for Year, Region_origin, Region_destination,
        Sector, Quantity (in kt), and Monetary_value.
    
    """

    conc_path = r'data\input\conc\baci.xlsx'
    sector_conc = pd.read_excel(conc_path, sheet_name='Sector')
    region_conc = pd.read_excel(conc_path, sheet_name='Region')

    df = imputation_baci()

    # ---- SECTOR PART ----
    df = df.merge(sector_conc, left_on='HS92_Code', right_on='Source_id', how='left')

    # sector weights
    weights = (
        sector_conc.groupby('Source_id')
        .size().reset_index(name='n_source')
    )
    weights['weight'] = 1 / weights['n_source']

    df = df.merge(weights[['Source_id', 'weight']], on='Source_id', how='left')

    df['Weighted_Quantity'] = df['Quantity'] * df['weight']
    df['Weighted_Monetary_value'] = df['Monetary_value'] * df['weight']

    df = df.groupby(
        ['Year', 'Region_origin', 'Region_destination', 'Base_name'],
        as_index=False
    )[['Weighted_Quantity', 'Weighted_Monetary_value']].sum()

    # ---- REGION PART ----
    # Merge origin concordance
    df = df.merge(region_conc.rename(
        columns={'Source_id': 'Region_origin', 'Base_name': 'Region_origin_base'}),
        on='Region_origin', how='left'
    )

    # Merge destination concordance
    df = df.merge(region_conc.rename(
        columns={'Source_id': 'Region_destination', 'Base_name': 'Region_destination_base'}),
        on='Region_destination', how='left'
    )

    # --- REMOVE DOMESTIC FLOWS CREATED BY COLLAPSE ---
    df = df[df['Region_origin_base'] != df['Region_destination_base']]

    # ---- REGION WEIGHTS ----
    region_weights = (
        region_conc.groupby('Source_id')
        .size().reset_index(name='n_source')
    )
    region_weights['weight_region'] = 1 / region_weights['n_source']

    # Merge weights for origin
    df = df.merge(
        region_weights.rename(columns={
            'Source_id': 'Region_origin',
            'weight_region': 'weight_origin'
        }),
        on='Region_origin', how='left'
    )

    # Merge weights for destination
    df = df.merge(
        region_weights.rename(columns={
            'Source_id': 'Region_destination',
            'weight_region': 'weight_destination'
        }),
        on='Region_destination', how='left'
    )

    # Calculate combined weight
    df['Combined_weight'] = df['weight_origin'] * df['weight_destination']

    # Apply combined weight to weighted quantities and monetary values
    df['Final_Weighted_Quantity'] = df['Weighted_Quantity'] * df['Combined_weight']
    df['Final_Weighted_Monetary_value'] = df['Weighted_Monetary_value'] * df['Combined_weight']

    # Final aggregation
    df = df.groupby(
        ['Year', 'Region_origin_base', 'Region_destination_base', 'Base_name'],
        as_index=False
    )[['Final_Weighted_Quantity', 'Final_Weighted_Monetary_value']].sum()

    # Rename columns to final names
    df = df.rename(columns={
        'Region_origin_base': 'Region_origin',
        'Region_destination_base': 'Region_destination',
        'Base_name': 'Sector',
        'Final_Weighted_Quantity': 'Quantity',
        'Final_Weighted_Monetary_value': 'Monetary_value'
    })

    # Convert quantity to kt
    df['Quantity'] = (df['Quantity'].astype(float)  / 10**3).round(3)

    return df


def baci_to_supply_use():
    baci = sector_region_to_base()

    total_supply = baci.groupby(['Region_origin', 'Sector'])['Quantity'].sum().reset_index()

    total_supply.rename(columns={'Quantity': 'Value'}, inplace=True)

    lookup_struc = lookup()['Structure']

    lookup_struc = lookup_struc[lookup_struc['Value'] == 1]

    supply_look = lookup_struc[lookup_struc['Supply_Use'] == 'Supply'][['Process', 'Flow']]

    # merge with supply_look
    sup_merge = total_supply.merge(supply_look, left_on='Sector', right_on='Flow', how='left')
    sup_merge = sup_merge[~sup_merge['Process'].isna()]
    sup_merge = sup_merge.drop(columns=['Flow'])
    sup_merge.rename(columns={'Sector': 'Sector_destination', 'Process': 'Sector_origin'}, inplace=True)
    sup_merge['Region_destination'] = sup_merge['Region_origin']
    # todo: Final demand removed -> belongs to use

    use_look = lookup_struc[lookup_struc['Supply_Use'] == 'Use'][['Process', 'Flow']]
    baci_use = baci.merge(use_look, left_on='Sector', right_on='Flow', how='left')
    baci_use = baci_use[~baci_use['Process'].isna()]
    baci_use = baci_use.drop(columns=['Flow', 'Monetary_value', 'Year'])
    baci_use = baci_use.rename(columns={'Sector':'Sector_origin', 'Process': 'Sector_destination', 'Quantity': 'Value'})

    final_demand = baci_use[baci_use['Sector_destination'] == 'Final_demand']

    baci_use = baci_use[baci_use['Sector_destination'] != 'Final_demand']

    # establish col order
    col_order = ['Region_origin', 'Sector_origin', 'Region_destination', 'Sector_destination', 'Value']

    supply = sup_merge[col_order]
    use = baci_use[col_order]
    final_demand = final_demand[col_order]

    supply = add_entity(supply)
    use = add_entity(use)
    final_demand = add_entity(final_demand)

    # save to datafeed folder_name 
    folder_name = r'data\processed\datafeed\baci'

    # get timestamp
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

    supply.to_csv( folder_name_check(folder_name, f'S_withoutdomestic_baci_{timestamp}.csv'), index=False)
    use.to_csv(folder_name_check(folder_name, f'U_withoutdomestic_baci_{timestamp}.csv'), index=False)
    final_demand.to_csv(folder_name_check(folder_name, f'Y_baci_{timestamp}.csv'), index=False)



    return None


def add_entity(df):
    
    secent = lookup()['Sector2Entity']

    df = df.merge(secent, left_on='Sector_origin', right_on='Sector_name', how='left', validate='m:1')
    #rename entity to Entity_origin
    df = df.rename(columns={'Entity': 'Entity_origin'})
    df = df.merge(secent, left_on='Sector_destination', right_on='Sector_name', how='left')
    df = df.rename(columns={'Entity': 'Entity_destination'})

    df = df.drop(columns=['Sector_name_x', 'Sector_name_y'])
    c_order = ['Region_origin',  'Sector_origin', 'Entity_origin', 'Region_destination', 'Sector_destination',  'Entity_destination', 'Value']

    return df[c_order]

if __name__ == "__main__":
    baci_to_supply_use()