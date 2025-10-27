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


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)





def get_latest_file(directory: str, prefix: str, suffix: str) -> Optional[str]:
    """
    Get the most recent file matching pattern like 'prefix_YYYYMMDD_HHMMSS.suffix'.
    """
    pattern = re.compile(rf"{re.escape(prefix)}_(\d{{8}}_\d{{6}}){re.escape(suffix)}$")
    candidates = []
    for f in os.listdir(directory):
        match = pattern.match(f)
        if match:
            ts = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
            candidates.append((ts, f))
    if not candidates:
        # warning
        ValueError(f'No files found for prefix: {prefix} and suffix: {suffix} in {directory}')
        
    latest = max(candidates, key=lambda x: x[0])[1]
    return os.path.join(directory, latest)


def collect_latest_constraints() -> dict:
    """
    Collects the latest constraint matrices and vectors for each type/element combination.
    """
    root_dir = r"data/proc/const"
    types = ["use_region_totals", "supply_region_totals", "material_efficiency", "mass_balance"]
    elements = ["G", "c", "c_sigma", "c_idx"]

    loaders = {
        "G": sp.load_npz,
        "c": np.load,
        "c_sigma": np.load,
        "c_idx": lambda path: np.load(path, allow_pickle=True),
    }

    results = {}

    for t in types:
        for e in elements:
            suffix = ".npz" if e == "G" else ".npy"
            prefix = f"{e}_{t}"  # matches "G_supply_region_totals_20251021_085108.npz"
            latest_path = get_latest_file(root_dir, prefix, suffix)

            if latest_path:
                data = loaders[e](latest_path)
                

                # double index for result
                results[(t, e)] = data

                logger.info(f"Loaded {latest_path}")
            else:
                logger.warning(f"No matching files for {prefix}*{suffix} in {root_dir}")

    return results


def main():
    res = collect_latest_constraints()

    # concat G
    G = sp.concatenate([res[('use_region_totals', 'G')], res[('supply_region_totals', 'G')],
                   res[('material_efficiency', 'G')], res[('mass_balance', 'G')]], axis=0)
    c = np.concatenate([res[('use_region_totals', 'c')], res[('supply_region_totals', 'c')],
                        res[('material_efficiency', 'c')], res[('mass_balance', 'c')]])
    c_sigma = np.concatenate([res[('use_region_totals', 'c_sigma')], res[('supply_region_totals', 'c_sigma')],
                              res[('material_efficiency', 'c_sigma')], res[('mass_balance', 'c_sigma')]])
    c_idx = np.concatenate([res[('use_region_totals', 'c_idx')], res[('supply_region_totals', 'c_idx')],
                            res[('material_efficiency', 'c_idx')], res[('mass_balance', 'c_idx')]])
    
    logger.info(f"Concatenated G shape: {G.shape}, c length: {len(c)}, c_sigma length: {len(c_sigma)}, c_idx length: {len(c_idx)}")

    out_folder = os.path.join("data", "proc", "const")
    os.makedirs(out_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path_G = os.path.join(out_folder, f"G_{timestamp}.npz")
    out_path_c = os.path.join(out_folder, f"c_{timestamp}.npy")
    out_path_csigma = os.path.join(out_folder, f"c_sigma_{timestamp}.npy")
    out_c_idx = os.path.join(out_folder, f"c_idx_{timestamp}.npy")
    sp.save_npz(out_path_G, G)
    np.save(out_path_c, c)
    np.save(out_path_csigma, c_sigma)
    np.save(out_c_idx, c_idx)
    logger.info(f"Saved concatenated G to {out_path_G}, c to {out_path_c}, c_sigma to {out_path_csigma}, c_idx to {out_c_idx}")





    
if __name__ == "__main__":
    main()