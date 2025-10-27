'''
Build the G matrix for region-sector tc constraints

'''

import logging
import os
import re
from datetime import datetime
from itertools import product
from typing import Iterable, Tuple, List, Optional
import sys



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sparse as sp
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from model.core.datafeed.util import get_latest_index, lookup



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def transform_value_mat_eff():

    """
    Transform value for tc constraints
    """
    struc = lookup()['Structure']
    tc_ = pd.read_excel(os.path.join(r'data/input/raw', 'TCs.xlsx'))
    c= pd.read_excel(os.path.join(r'data/input/raw', 'Conc.xlsx'))

    struc_o = struc[(struc.Supply_Use == 'Supply') & (struc.Value == 1)].copy()
    struc_d = struc[(struc.Supply_Use == 'Use') & (struc.Value == 1)].copy()
    
    tc = tc_[tc_.Type == 'Material_efficiency'].copy()
    tc = tc [['Process', 'Value']]


    merged_o = pd.merge(tc, struc_o[['Flow', 'Process']],
                      left_on='Process',
                      right_on='Process',
                      how='left')
    
    waste_flows = ['Waste_rock', 'Tailings', 'Slag']

    merged_o = merged_o[~merged_o['Flow'].isin(waste_flows)]

    merged_o.rename(columns={'Flow': 'Flow_output'}, inplace=True)

    merged_d = pd.merge(merged_o, struc_d[['Flow', 'Process']],on='Process', how='left')
    merged_d.rename(columns={'Flow': 'Flow_input'}, inplace=True)
    merged_d.rename(columns={'Value': 'TC_value'}, inplace=True)


    c_o = pd.merge(merged_d, c[['Flow', 'Value']], left_on='Flow_output', right_on='Flow', how='left')
    c_o.rename(columns={'Value': 'Conc_output'}, inplace=True)
    c_o.drop(columns=['Flow'], inplace=True)
    c_od = pd.merge(c_o, c[['Flow', 'Value']], left_on='Flow_input', right_on='Flow', how='left')
    c_od.rename(columns={'Value': 'Conc_input'}, inplace=True)
    c_od.drop(columns=['Flow'], inplace=True)


    c_od['Transformed_value'] = 1 / c_od['TC_value'] * (c_od['Conc_output'] / c_od['Conc_input'])
    c_od = c_od[['Process', 'Flow_input', 'Flow_output', 'Transformed_value']]

    # append waste rock specific assignment
    w = dict()  
    
    w['Process']  = 'Mining'
    w['Flow_input']  = 'Waste_rock'
    w['Flow_output'] = 'Ore'
    w['Transformed_value'] = tc_[tc_['Process'] == 'Mining']['Value'].values[0]

    w_df = pd.DataFrame([w])
    c_od = pd.concat([c_od, w_df], ignore_index=True)
    

    return c_od


    



def init_G_mat_efficiency_fast(sanity: bool = True, type_: str = 'material_efficiency'):
    """
    Optimized initialization of G matrix for material efficiency TC constraints.
    """
    index_path = get_latest_index(r'data/proc/index')
    index = pd.read_parquet(index_path).reset_index(drop=True)
    tcs = transform_value_mat_eff()

    unique_regions = index['Region_origin'].unique()

    # Precompute lookup dictionaries for faster masking
    # Group by (Sector_origin, Sector_destination, Region_destination)
    in_lookup = (
        index
        .reset_index()
        .groupby(['Sector_origin', 'Sector_destination', 'Region_destination'])['index']
        .apply(list)
        .to_dict()
    )
    # Group by (Sector_destination, Sector_origin, Region_origin)
    out_lookup = (
        index
        .reset_index()
        .groupby(['Sector_destination', 'Sector_origin', 'Region_origin'])['index']
        .apply(list)
        .to_dict()
    )

    rows, cols, vals, idx = [], [], [], []
    i = 0

    # Iterate over TC rows (not over entire index anymore)
    for _, row in tcs.iterrows():
        f_in, f_out, proc, val = row['Flow_input'], row['Flow_output'], row['Process'], row['Transformed_value']

        for region in unique_regions:

            if proc == 'Mining':
                coord_in = out_lookup.get((f_in, proc, region), [])
            else:
                coord_in = in_lookup.get((f_in, proc, region), [])
            
            
            coord_out = out_lookup.get((f_out, proc, region), [])

            if not coord_in and not coord_out:
                ValueError(f'TC constraint has no matching variables in index. Process: {proc}, Flow_input: {f_in}, Flow_output: {f_out}, Region: {region}')

            # vectorized appends
            n_in, n_out = len(coord_in), len(coord_out)
            if n_in:
                rows.extend([i] * n_in)
                cols.extend(coord_in)
                vals.extend([-1.0] * n_in)
            if n_out:
                rows.extend([i] * n_out)
                cols.extend(coord_out)
                vals.extend([val] * n_out)

            idx.append((type_, f_out, f_in, region))
            i += 1

    # Build sparse matrix
    G = sp.COO((vals, (rows, cols)), shape=(i, len(index)))

    c = np.zeros(i, dtype=np.float64)
    c_sigma = c + 1e-1
    idx_str = np.array([f"{t}_{fo}_{fi}_{r}" for (t, fo, fi,  r) in idx], dtype=object)

    assert len(idx_str) == G.shape[0], "Index length does not match number of rows in G"

    density = 100 * G.nnz / (G.shape[0] * G.shape[1])
    logger.info(f"Sparse matrix G built with shape {G.shape} and density {density:.6e}%")
    G_mem_MB = G.data.nbytes + G.coords.nbytes / (1024**2)
    logger.info(f"Sparse matrix G memory size: {G_mem_MB:.2f} MB")

    # Save
    out_folder = os.path.join("data", "proc", "const")
    os.makedirs(out_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sp.save_npz(os.path.join(out_folder, f"G_{type_}_{timestamp}.npz"), G)
    np.save(os.path.join(out_folder, f"c_{type_}_{timestamp}.npy"), c)
    np.save(os.path.join(out_folder, f"c_sigma_{type_}_{timestamp}.npy"), c_sigma)
    np.save(os.path.join(out_folder, f"c_idx_{type_}_{timestamp}.npy"), idx_str)
    logger.info("All outputs saved successfully.")

    return G, c, c_sigma, idx_str



if __name__ == "__main__":
    init_G_mat_efficiency_fast()
    