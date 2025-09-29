import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from model.core.datafeed import lookup, build_index



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
    None