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

    baci_path = r'data\processed\data_feed\ie\baci_ie_v1.csv'
    icsg_path = r'data\processed\data_feed\ie\icsg_ie_v1.csv'

    baci = pd.read_csv(baci_path)
    icsg = pd.read_csv(icsg_path)

    # Calculate imports from baci
    imports = (
        baci.groupby(["Region_destination", "Sector_destination", 'Entity_destination'])["Value"]
        .sum()
        .reset_index()
        .rename(columns={
            "Region_destination": "Region",
            "Sector_destination": "Flow",
            "Entity_destination": "Entity",
            "Value": "Imports"
        })
    )

    # exports baci
    exports = (
        baci.groupby(["Region_origin", "Sector_origin", 'Entity_origin'])["Value"]
        .sum()
        .reset_index()
        .rename(columns={
            "Region_origin": "Region",
            "Sector_origin": "Flow",
            "Entity_origin": "Entity",
            "Value": "Exports"
        })
    )

    icsg_grouped = (
        icsg.groupby(["Region", "Flow", "Supply_Use"])["Value"]
        .sum()
        .reset_index()
        .pivot(index=["Region",  "Flow"], columns="Supply_Use", values="Value")
        .reset_index()
    )

    merged = (
        icsg_grouped
        .merge(imports[['Region', 'Flow', 'Imports']], on=["Region", "Flow"], how="left")
        .merge(exports[['Region', 'Flow', 'Exports']], on=["Region", "Flow"], how="left")
    )

    merged = merged.fillna(0)

    lookups = lookup()

    trade_lookup = lookups['Trade']

    trade_flows = trade_lookup[trade_lookup['Value'] == 1]['Flow'].unique()
    merged = merged[merged['Flow'].isin(trade_flows)]

    # Calculate domestic supply flows
    supply = merged[merged['Supply'] > 0].copy()
    supply['Domestic_Supply'] = supply['Supply'] - supply['Exports']

    # substite negative domestic supply with .1
    supply['Domestic_Supply'] = np.where(supply['Domestic_Supply'] < 0, noise_zero, supply['Domestic_Supply'])

    # Calculate domestic use flows
    use = merged[merged['Use'] > 0].copy()
    use['Domestic_Use'] = use['Use'] - use['Imports']
    use = use[['Region', 'Flow', 'Domestic_Use']]

    use['Domestic_Use'] = np.where(use['Domestic_Use'] < 0, noise_zero, use['Domestic_Use'])

    # get structure lookup
    structure = lookups['Structure']
    structure = structure[structure['Value'] == 1]
    

    # --- Get things into supply use logic ---

    select_cols = ['Region_origin', 'Sector_origin', 'Region_destination', 'Sector_destination', 'Value']

    # merge domestic use with structure filtere use
    use = use.merge(structure[structure['Supply_Use'] == 'Use'][['Flow', 'Process']], on='Flow', how='left')

    # merge domestic supply with structure filtere supply
    supply = supply.merge(structure[structure['Supply_Use'] == 'Supply'][['Flow', 'Process']], on='Flow', how='left')

    use_col_rename = { 'Region': 'Region_origin',
                     'Flow': 'Sector_origin',
                     'Process': 'Sector_destination',
                     'Domestic_Use': 'Value'}

    use = use.rename(columns=use_col_rename)

    use['Region_destination'] = use['Region_origin']
    use = use[select_cols]


    use_supply = use.merge(structure[structure['Supply_Use'] == 'Supply'][['Flow', 'Process']], left_on='Sector_origin', right_on='Flow', how='left')
    
    use_supply.drop(columns=['Flow', 'Sector_destination'], inplace=True)

    use_supply = use_supply.rename(columns={'Process': 'Sector_origin', 
                                          'Sector_origin': 'Sector_destination'})

    use_supply = use_supply[select_cols]



    supply_col_rename = { 'Region': 'Region_destination',
                        'Flow': 'Sector_destination',
                        'Process': 'Sector_origin',
                        'Domestic_Supply': 'Value'} 
    supply = supply.rename(columns=supply_col_rename)
    supply['Region_origin'] = supply['Region_destination']
    supply = supply[select_cols]

    supply_use = supply.merge(structure[structure['Supply_Use'] == 'Use'][['Flow', 'Process']], left_on='Sector_destination', right_on='Flow', how='left')

    supply_use = supply_use[supply_use['Process'].notna()]

    supply_use.drop(columns=['Flow', 'Sector_origin'], inplace=True)
    supply_use = supply_use.rename(columns={'Process': 'Sector_destination', 
                                          'Sector_destination': 'Sector_origin'})
    supply_use = supply_use[select_cols]

    domestic_flows = pd.concat([use, supply, use_supply, supply_use], ignore_index=True)


    # add entity
    secent = lookups['Entity'][['Sector_name', 'Entity']]
    domestic_flows = domestic_flows.merge(secent, left_on='Sector_origin', right_on='Sector_name', how='left')
    domestic_flows = domestic_flows.rename(columns={'Entity': 'Entity_origin'})
    domestic_flows = domestic_flows.merge(secent, left_on='Sector_destination', right_on='Sector_name', how='left')
    domestic_flows = domestic_flows.rename(columns={'Entity': 'Entity_destination'})
    domestic_flows = domestic_flows.drop(columns=['Sector_name_x', 'Sector_name_y'])
    final_col_order = ['Region_origin',  'Sector_origin', 'Entity_origin', 'Region_destination',  'Sector_destination', 'Entity_destination', 'Value']
    domestic_flows = domestic_flows[final_col_order]


    #save to csv
    out_path = r'data\processed\data_feed\ie\domestic_trade_v1.csv'
    domestic_flows.to_csv(out_path, index=False)

    return None
    

def dom_to_sup_use():
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
    ie, index_tuples = merge_data_to_index_pd()
    
