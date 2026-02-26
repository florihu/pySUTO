


import pandas as pd

import logging
import os

from pathlib import Path
from datetime import datetime
from typing import Optional
import re
import sys


_TS_RE = re.compile(r'(\d{8}_\d{6})')


def lookup():
    structure_path = r'data\input\structure.xlsx'
    structure = pd.read_excel(structure_path, sheet_name=None)

    # Example: work with the "Supply" sheet
    supply = structure['Supply']

    # First column = Flow
    supply = supply.rename(columns={supply.columns[0]: "Process"})

    # Melt matrix into long format: Flow, Process, Value
    supply_long = supply.melt(
        id_vars=["Process"],          # keep flows fixed
        var_name="Flow",        # column headers become "Process"
        value_name="Value"         # cell values
    )
    supply_long['Supply_Use'] = 'Supply'
    
    use = structure['Use']
    use = use.rename(columns={use.columns[0]: "Flow"})
    use_long = use.melt(
        id_vars=["Flow"],          # keep flows fixed
        var_name="Process",        # column headers become "Process"
        value_name="Value"         # cell values
    )

    use_long['Supply_Use'] = 'Use'

    # sort columns
    supply_long = supply_long[['Flow', 'Process', 'Supply_Use', 'Value']]
    use_long = use_long[['Flow', 'Process', 'Supply_Use', 'Value']]

    # concat
    structure_long = pd.concat([supply_long, use_long], ignore_index=True)

    trade = structure['Trade']

    entity = structure['Sector2Entity']
    ent_ent = structure['Accounting']

    collect = {'Structure': structure_long,
               'Trade': trade,
               'Sector2Entity': entity,
               'Accounting': ent_ent
               }

    return collect


def get_latest_index(folder: str) -> str:
    """
    Return the path to the latest parquet file in `folder`.
    Priority for deciding "latest":
      1) timestamp parsed from filename using pattern YYYYMMDD_HHMMSS
      2) file modification time (fallback)
    Raises FileNotFoundError when no .parquet files exist.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = [f for f in os.listdir(folder) if f.lower().endswith('.parquet')]
    if not files:
        raise FileNotFoundError(f"No parquet files found in {folder}")

    candidates = []
    for f in files:
        full = os.path.join(folder, f)
        ts = _parse_timestamp_from_name(f)
        if ts is None:
            # fallback to file modification time
            ts = datetime.fromtimestamp(os.path.getmtime(full))
        candidates.append((ts, full))

    # pick the newest timestamp
    latest_path = max(candidates, key=lambda x: x[0])[1]
    return latest_path


def _parse_timestamp_from_name(fname: str):
    m = _TS_RE.search(fname)
    if not m:
        return None
    ts_text = m.group(1)
    try:
        return datetime.strptime(ts_text, "%Y%m%d_%H%M%S")
    except ValueError:
        return None



if __name__ == '__main__':
    lookup()