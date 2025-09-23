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
    u_icsg_path = r'icsg\U_prod_icsg_20250923_093733.csv'

    baci_path = os.path.join(folder_name, u_baci_path)
    icsg_path = os.path.join(folder_name, u_icsg_path)


    baci = pd.read_csv(baci_path)
    icsg = pd.read_csv(icsg_path)

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
    use = merged.merge(structure[structure['Supply_Use'] == 'Use'][['Flow', 'Process']], on='Flow', how='left')

   
    use_col_rename = { 'Region': 'Region_origin',
                     'Flow': 'Sector_origin',
                     'Process': 'Sector_destination',
                     'Domestic_Use': 'Value'}

    use = use.rename(columns=use_col_rename)

    use['Region_destination'] = use['Region_origin']

    sec_to_ent = lookups['Sector2Entity']

    use = use.merge(sec_to_ent, left_on='Sector_origin', right_on='Sector_name', how='left')
    use = use.rename(columns={'Entity': 'Entity_origin'})
    use = use.merge(sec_to_ent, left_on='Sector_destination', right_on='Sector_name', how='left')
    use = use.rename(columns={'Entity': 'Entity_destination'})
    use = use.drop(columns=['Sector_name_x', 'Sector_name_y'])

    col_order = ['Region_origin',  'Sector_origin', 'Entity_origin', 'Region_destination',  'Sector_destination', 'Entity_destination', 'Value']
    use = use[col_order]

    # filter out zero values
    use = use[use['Value'] > 0]


    folder = 'data/proc/datafeed/dom_calc'
    os.makedirs(folder, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

    out_path = os.path.join(folder, f'U_domestic_{timestamp}.csv')
    use.to_csv(out_path, index=False)

    return None


# todo : We need to do the domestic final demand calc as well for final demand... can also be traded.

def icsg_to_supply_use():

    use_paths = ['E_prod_icsg_20250923_093733.csv', 
             'O_prod_icsg_20250923_093733.csv'
             
             ]
    supply_paths = ['P_prod_icsg_20250923_093733.csv',
             'S_prod_icsg_20250923_093733.csv']
    
    look = lookup()

    structure = look['Structure']
    structure = structure[structure['Value'] == 1]
    entity = look['sector2Entity']
    

    for u in use_paths:
        use_path = os.path.join(r'data\processed\data_feed\ie', u)
        use = pd.read_csv(use_path)

        use = use.merge(structure[structure['Supply_Use'] == 'Use'][['Flow', 'Process']], on='Flow', how='left')

        use_rename = {'Region': 'Region_origin',
                     'Flow': 'Sector_origin',
                     'Process': 'Sector_destination',
                     'Value': 'Value'}

        use = use.rename(columns=use_rename)
        use['Region_destination'] = use['Region_origin']

        sec_to_ent = lookups['Sector2Entity']

        use = use.merge(sec_to_ent, left_on='Sector_origin', right_on='Sector_name', how='left')
        use = use.rename(columns={'Entity': 'Entity_origin'})
        use = use.merge(sec_to_ent, left_on='Sector_destination', right_on='Sector_name', how='left')
        use = use.rename(columns={'Entity': 'Entity_destination'})
        use = use.drop(columns=['Sector_name_x', 'Sector_name_y'])

        col_order = ['Region_origin',  'Sector_origin', 'Entity_origin', 'Region_destination',  'Sector_destination', 'Entity_destination', 'Value']
        use = use[col_order]

        out_path = os.path.join(r'data\processed\data_feed\ie', f'use_{u}')
        use.to_csv(out_path, index=False)


    






    return None

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


def build_index():
    l = lookup()  # assumes this gives your dict of lookup DataFrames
    base = pd.read_excel(r'data\input\conc\base.xlsx', sheet_name=None)
    dim = [
        'Region_origin', 'Sector_origin', 'Entity_origin',
        'Region_destination', 'Sector_destination', 'Entity_destination'
    ]

    # ----- sector pairs (only valid ones) -----
    structure = l['Structure']
    structure = structure[structure['Value'] == 1]

    # valid supply combinations
    supply = (
        structure[structure['Supply_Use'] == 'Supply']
        .rename(columns={'Process': 'Sector_origin', 'Flow': 'Sector_destination'})
        [['Sector_origin', 'Sector_destination']]
        .drop_duplicates()
    )

    # valid use combinations
    use = (
        structure[structure['Supply_Use'] == 'Use']
        .rename(columns={'Flow': 'Sector_origin', 'Process': 'Sector_destination'})
        [['Sector_origin', 'Sector_destination']]
        .drop_duplicates()
    )

    valid_sector_pairs = pd.concat([supply, use], ignore_index=True).drop_duplicates()

    # ----- entity pairs via sector-entity map -----
    sector_entity = l['Entity'][['Sector_name', 'Entity']].drop_duplicates()

    valid_pairs = (
        valid_sector_pairs
        .merge(sector_entity.rename(columns={'Sector_name': 'Sector_origin'}), on='Sector_origin')
        .rename(columns={'Entity': 'Entity_origin'})
        .merge(sector_entity.rename(columns={'Sector_name': 'Sector_destination'}), on='Sector_destination')
        .rename(columns={'Entity': 'Entity_destination'})
    )

    # ----- region pairs (cartesian) -----
    regions = base['Region']['Name'].unique()
    region_pairs = pd.DataFrame(
        [(ro, rd) for ro in regions for rd in regions],
        columns=['Region_origin', 'Region_destination']
    )

    # ----- domestic-trade map -----
    domestic_map = l['Trade']  # this should point to your "Flow–Value" table
    domestic_map = domestic_map.rename(columns={'Flow': 'Sector_destination'})  # align naming

    # add domestic restriction flag
    valid_pairs = valid_pairs.merge(domestic_map, on='Sector_destination', how='left').fillna({'Value': 1})

    # ----- build index -----
    index = (
        region_pairs
        .merge(valid_pairs, how="cross")
        # apply domestic-only restriction
        .query("(Value == 1) or (Region_origin == Region_destination)")
        .drop(columns='Value')  # optional: keep clean
        .loc[:, dim]
    )

    index.to_parquet(r'data\processed\data_feed\ie\index_v1.parquet', compression='snappy')

    return index




def merge_data_to_index_pd():
    baci_path = r'data\processed\data_feed\ie\baci_ie_v1.csv'
    dom_path = r'data\processed\data_feed\ie\domestic_ie_v1.csv'
    dom_trade_path = r'data\processed\data_feed\ie\domestic_trade_v1.csv'

    col_order = [
        'Region_origin', 'Sector_origin', 'Entity_origin',
        'Region_destination', 'Sector_destination', 'Entity_destination', 'Value'
    ]

    # --- Load flows (only needed columns)
    baci = pd.read_csv(baci_path, usecols=col_order)
    dom = pd.read_csv(dom_path, usecols=col_order)
    dom_trade = pd.read_csv(dom_trade_path, usecols=col_order)

    # Combine & deduplicate
    combined = pd.concat([baci, dom, dom_trade], ignore_index=True)
    combined = combined.drop_duplicates(subset=col_order[:-1])

    # --- Build the index space
    index = build_index()   # returns a DataFrame with all possible combinations
    index_tuples = list(map(tuple, index[col_order[:-1]].to_numpy()))

    # --- Map flows to integer positions
    # assign an integer ID to each unique tuple
    index_map = {key: i for i, key in enumerate(index_tuples)}

    # --- Initialize IE matrix as NumPy array
    IE = np.zeros(len(index_tuples), dtype=np.float64)

    # --- Fill IE matrix using dictionary lookups
    for row in combined.itertuples(index=False):
        key = tuple(getattr(row, c) for c in col_order[:-1])
        if key in index_map:   # should always be true if mapping is consistent
            IE[index_map[key]] = row.Value

    # IE is now a dense NumPy vector aligned with index_tuples
    density = np.count_nonzero(IE) / IE.size
    print(f"IE matrix density: {density:.6f} ({np.count_nonzero(IE)} non-zeros out of {IE.size})")
    
    # acompare with pandas
    df = pd.DataFrame({'Value': IE}, index=pd.MultiIndex.from_tuples(index_tuples, names=col_order[:-1]))
    print(f"Pandas DataFrame RAM size: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

    # confirm that index order matches IE order
    assert all(t1 == t2 for t1, t2 in zip(df.index.to_list(), index_tuples)), "Index mismatch!"

    # save results but only store values not index
    
    # store df as parquet
    df.to_parquet(r'data\processed\data_feed\ie\ie_matrix.parquet', compression='snappy')

   
    return IE, index_tuples

def plot_ie_density(IE, index_tuples):
    """
    Calculate density of a sparse IE array and plot origin-destination flows.

    Parameters
    ----------
    IE : np.ndarray
        1D array of flows aligned with index_tuples.
    index_tuples : list of tuples
        Each tuple = (Region_origin, Sector_origin, Entity_origin,
                      Region_destination, Sector_destination, Entity_destination)
    """

    # --- Density ---
    nonzero = np.count_nonzero(IE)
    total = IE.size
    density = nonzero / total
    print(f"Non-zero entries: {nonzero:,} / {total:,} ({density:.4%} density)")

    # --- Build mapping to origin/destination blocks ---
    origins = [(r, s, e) for (r, s, e, _, _, _) in index_tuples]
    dests   = [(r, s, e) for (_, _, _, r, s, e) in index_tuples]

    unique_origins = {o: i for i, o in enumerate(sorted(set(origins)))}
    unique_dests   = {d: j for j, d in enumerate(sorted(set(dests)))}

    # --- Aggregate into OD matrix ---
    OD = np.zeros((len(unique_origins), len(unique_dests)))
    for val, o, d in zip(IE, origins, dests):
        if val != 0:
            OD[unique_origins[o], unique_dests[d]] += val

    # --- Plot ---
    plt.figure(figsize=(12, 12))
    plt.imshow(OD, cmap="viridis_r", aspect="auto", interpolation="nearest")
    plt.colorbar(label="Flow value")
    plt.title(f"Origin-Destination Flow Matrix\nDensity = {density:.2%}")

    plt.xlabel("Destinations")
    plt.ylabel("Origins")

    # Turn off ticks completely
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.savefig(r'figs\explo\ie_od_matrix.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    calc_domestic_trade_flows()
    
