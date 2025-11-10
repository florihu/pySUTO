import os
import sys



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from model.core.datafeed.util import lookup, get_latest_index



def merge_data_to_index_pd(
        d_paths = ['S_withoutdomestic_baci_20250923_083406.csv', 'U_withoutdomestic_baci_20250923_083406.csv', 'Y_baci_20250923_083406.csv',
               'U_domestic_20251020_134454.csv', 'Y_domestic_20251020_134454.csv', 'E_20251021_091741.csv', 'O_20251021_091741.csv', 'P_20251021_091828.csv'],
    
        d_folders = ['baci', 'baci', 'baci',
                    'dom_calc', 'dom_calc', 'dom_calc', 'dom_calc', 'dom_calc'],

        index_folder = r'data\proc\index',
        out_name = 'ie',
        out_folder = r'data\proc\ie',
        tol = 1e-3

        ):
    

    col_order = [
        'Region_origin', 'Sector_origin', 'Entity_origin',
        'Region_destination', 'Sector_destination', 'Entity_destination', 'Value'
    ]

    collect = [pd.read_csv(os.path.join(r'data\proc\datafeed', folder, path))[col_order] for folder, path in zip(d_folders, d_paths)]

    combined = pd.concat(collect, ignore_index=True)

    # --- Load index ---
    index_path = get_latest_index(index_folder)
    index_df = pd.read_parquet(index_path)

    # --- Merge with index ---
    merged = pd.merge(index_df, combined,
                      on=['Region_origin', 'Sector_origin', 'Entity_origin',
                          'Region_destination', 'Sector_destination', 'Entity_destination'],
                      how='left', validate='one_to_one')
    
    merged['Value'] = merged['Value'].fillna(tol)

    ie = merged['Value'].to_numpy(dtype=np.float64)

    # --- print some analytics ---
    print(f"IE density: {np.count_nonzero(ie)} non-zero entries out of {ie.size} total ({np.count_nonzero(ie)/ie.size:.4%})")

    # --- sanity check ---
    assert ie.shape[0] == index_df.shape[0], "Mismatch in IE array length and index length"
    assert np.all(ie >= 0), "Negative values found in IE array"

    

    # --- Save IE array ---
    time_stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, f"{out_name}_{time_stamp}")
    np.save(out_path, ie)
    

    return None



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
    merge_data_to_index_pd()
