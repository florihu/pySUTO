import os
import sys
import pandas as pd
import logging
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from model.core.datafeed.util import lookup


def build_index_vOLD():
    """
    Build a region–sector–entity index with domestic/trade restrictions.

    Steps:
    1. Load lookups and base data.
    2. Identify valid sector pairs (supply/use structure).
    3. Map sectors to entities → valid entity pairs.
    4. Build all possible region pairs (cartesian).
    5. Merge in domestic restriction rules (entity → entity).
    6. Apply rule:
        - Domestic == 1 → only same-region flows.
        - Domestic == 0 → allow cross-region trade.
    7. Export index.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Building region-sector-entity index...")

    time_start = pd.Timestamp.now()

    # --- Load lookups ---
    l = lookup()  # dict of DataFrames from your lookup
    base = pd.read_excel(r"data\input\conc\base.xlsx", sheet_name=None)

    # final index dimensions
    dim = [
        "Region_origin", "Sector_origin", "Entity_origin",
        "Region_destination", "Sector_destination", "Entity_destination"
    ]

    # --- Step 1: valid sector pairs (from structure table) ---
    structure = l["Structure"].query("Value == 1")

    supply = (
        structure.query("Supply_Use == 'Supply'")
        .rename(columns={"Process": "Sector_origin", "Flow": "Sector_destination"})
        [["Sector_origin", "Sector_destination"]]
    )

    use = (
        structure.query("Supply_Use == 'Use'")
        .rename(columns={"Flow": "Sector_origin", "Process": "Sector_destination"})
        [["Sector_origin", "Sector_destination"]]
    )

    valid_sector_pairs = pd.concat([supply, use], ignore_index=True).drop_duplicates()

    # --- Step 2: sector → entity mapping ---
    sector_entity = l["Sector2Entity"][["Sector_name", "Entity"]].drop_duplicates()

    valid_pairs = (
        valid_sector_pairs
        .merge(sector_entity.rename(columns={"Sector_name": "Sector_origin"}), on="Sector_origin")
        .rename(columns={"Entity": "Entity_origin"})
        .merge(sector_entity.rename(columns={"Sector_name": "Sector_destination"}), on="Sector_destination")
        .rename(columns={"Entity": "Entity_destination"})
    )

    # --- Step 3: region pairs (cartesian product) ---
    regions = base["Region"]["Name"].unique()
    region_pairs = pd.DataFrame([(ro, rd) for ro in regions for rd in regions],
                                columns=["Region_origin", "Region_destination"])

    # --- Step 4: domestic restriction rules ---
    domestic_map = l["Accounting"]  # must contain: Entity_origin, Entity_destination, Domestic
    valid_pairs = valid_pairs.merge(domestic_map, on=["Entity_origin", "Entity_destination"], how="left")

    # Default: assume domestic restriction if no rule found
    valid_pairs["Domestic"] = valid_pairs["Domestic"].fillna(1).astype(int)

    # --- Step 5: build full index (regions × valid pairs) ---
    index = (
        region_pairs
        .merge(valid_pairs, how="cross")
        .query("(Domestic == 0) or (Region_origin == Region_destination)")
        .drop(columns="Domestic")
        .loc[:, dim]
    )

    time_end = pd.Timestamp.now()
    logger.info(f"Index built in {(time_end - time_start).total_seconds():.2f} seconds.")

    # print dimensionallity of problem
    print(f"Index size: {len(index)} rows")
    print(f"Regions: {len(regions)}")
    print(f"Sectors: {sector_entity['Sector_name'].nunique()}")
    print(f"Entities: {sector_entity['Entity'].nunique()}")
    # estimate RAM usage
    print(f"Estimated RAM size: {index.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    

    folder_name = r"data\proc\index"
    os.makedirs(folder_name, exist_ok=True)
    time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # --- Step 6: export ---
    index_path = os.path.join(folder_name, f"index_{time}.parquet")
    index.to_parquet(index_path, compression="snappy")
    logger.info(f"Index saved to '{index_path}'.")

    return None

def build_index():
    """
    Efficient index builder.

    Produces:
      - all same-region rows for every valid sector->sector / entity->entity pair
      - cross-region rows only for pairs where Domestic==0 AND Trade==1

    This avoids building the full regions x pairs cross-product and then filtering,
    which is much more memory/time efficient when many pairs are domestic-only.
    """
    

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Building region-sector-entity index (efficient)...")

    ts0 = pd.Timestamp.now()

    # --- Load lookups and base data ---
    l = lookup()
    base = pd.read_excel(r"data\input\conc\base.xlsx", sheet_name=None)

    # final columns
    dim = [
        "Region_origin", "Sector_origin", "Entity_origin",
        "Region_destination", "Sector_destination", "Entity_destination"
    ]

    # --- Step 1: structure -> valid sector pairs, mark Use vs Supply ---
    structure = l["Structure"].query("Value == 1").copy()
    structure["is_use"] = structure["Supply_Use"].astype(str).str.lower() == "use"

    supply = (
        structure[structure["Supply_Use"] == "Supply"]
        .rename(columns={"Process": "Sector_origin", "Flow": "Sector_destination"})
        [["Sector_origin", "Sector_destination", "is_use"]]
    )

    use = (
        structure[structure["Supply_Use"] == "Use"]
        .rename(columns={"Flow": "Sector_origin", "Process": "Sector_destination"})
        [["Sector_origin", "Sector_destination", "is_use"]]
    )

    valid_sector_pairs = pd.concat([supply, use], ignore_index=True).drop_duplicates().reset_index(drop=True)

    # --- Step 2: sector -> entity mapping ---
    sector_entity = l["Sector2Entity"][["Sector_name", "Entity"]].drop_duplicates()
    valid_pairs = (
        valid_sector_pairs
        .merge(sector_entity.rename(columns={"Sector_name": "Sector_origin"}), on="Sector_origin", how="left")
        .rename(columns={"Entity": "Entity_origin"})
        .merge(sector_entity.rename(columns={"Sector_name": "Sector_destination"}), on="Sector_destination", how="left")
        .rename(columns={"Entity": "Entity_destination"})
    )

    # --- Step 3: add Domestic rules (entity->entity) ---
    domestic_map = l.get("Accounting", pd.DataFrame())
    if not domestic_map.empty:
        valid_pairs = valid_pairs.merge(domestic_map, on=["Entity_origin", "Entity_destination"], how="left")
    # default domestic == 1 (must be domestic) if not specified
    valid_pairs["Domestic"] = valid_pairs["Domestic"].fillna(1).astype(np.int8)

    # --- Step 4: add flow trade rules (apply only to Use rows) ---
    trade_map = l.get("Trade") 
    if trade_map is None:
        logger.info("No trade lookup found; assuming all flows tradeable.")
        valid_pairs["Trade"] = 1
    else:
        if not {"Flow", "Value"}.issubset(trade_map.columns):
            logger.warning("Trade lookup missing expected columns 'Flow'/'Value'. Assuming tradeable.")
            valid_pairs["Trade"] = 1
        else:
            trade_df = trade_map[["Flow", "Value"]].rename(columns={"Flow": "Sector_origin", "Value": "Trade"})
            valid_pairs = valid_pairs.merge(trade_df, on="Sector_origin", how="left")
            # default tradeable if not provided (changeable)
            valid_pairs["Trade"] = valid_pairs["Trade"].fillna(1).astype(np.int8)
            # enforce trade==1 for supply rows (trade rule only meaningful for use)
            valid_pairs.loc[~valid_pairs["is_use"].astype(bool), "Trade"] = 1

    # --- Convert some columns to arrays for fast indexing ---
    regions = np.asarray(base["Region"]["Name"].unique(), dtype=object)
    n_regions = regions.size

    # ensure no missing entity mappings (fail early if any)
    if valid_pairs[["Entity_origin", "Entity_destination"]].isnull().any().any():
        raise ValueError("Missing Entity mapping for some Sectors. Check Sector2Entity lookup.")

    # keep only needed columns on valid_pairs (make small ndarray views)
    vp = valid_pairs.reset_index(drop=True)
    n_pairs = len(vp)

    # Part A: same-region rows for every valid pair
    # For each pair p and each region r create (r,r,p)
    # length = n_pairs * n_regions
    pair_idx_A = np.repeat(np.arange(n_pairs), n_regions)
    region_arr_A = np.tile(regions, n_pairs)  # repeated sequence of regions for each pair

    dfA = pd.DataFrame({
        "Region_origin": region_arr_A,
        "Region_destination": region_arr_A,
        "Sector_origin": vp["Sector_origin"].values[pair_idx_A],
        "Entity_origin": vp["Entity_origin"].values[pair_idx_A],
        "Sector_destination": vp["Sector_destination"].values[pair_idx_A],
        "Entity_destination": vp["Entity_destination"].values[pair_idx_A],
    })

    # Part B: cross-region rows only for pairs that allow trade (Domestic==0 & Trade==1)
    cond_tradeable = (vp["Domestic"].values == 0) & (vp["Trade"].values == 1)
    tradeable_idx = np.nonzero(cond_tradeable)[0]
    if tradeable_idx.size > 0 and n_regions > 1:
        # Build full region x region grid once
        origin_grid = np.repeat(regions, n_regions)   # length n_regions^2
        dest_grid = np.tile(regions, n_regions)      # length n_regions^2
        # mask out equal-region (we only want cross-region here)
        neq_mask = origin_grid != dest_grid
        origin_cross = origin_grid[neq_mask]
        dest_cross = dest_grid[neq_mask]
        # repeat the origin/dest grids for each tradeable pair
        m = tradeable_idx.size
        origin_rep = np.tile(origin_cross, m)
        dest_rep = np.tile(dest_cross, m)
        # indices mapping to pairs (repeat each tradeable pair for each origin/dest combination)
        pair_idx_B = np.repeat(tradeable_idx, origin_cross.size)
        dfB = pd.DataFrame({
            "Region_origin": origin_rep,
            "Region_destination": dest_rep,
            "Sector_origin": vp["Sector_origin"].values[pair_idx_B],
            "Entity_origin": vp["Entity_origin"].values[pair_idx_B],
            "Sector_destination": vp["Sector_destination"].values[pair_idx_B],
            "Entity_destination": vp["Entity_destination"].values[pair_idx_B],
        })
        index = pd.concat([dfA, dfB], ignore_index=True)
    else:
        index = dfA  # no tradeable cross-region pairs

    # Reorder columns to the requested dim order
    index = index.loc[:, dim]

    # Optionally convert some columns to categorical to reduce memory (helpful later)
    for c in ["Region_origin", "Region_destination", "Sector_origin", "Sector_destination",
              "Entity_origin", "Entity_destination"]:
        index[c] = index[c].astype("category")

    ts1 = pd.Timestamp.now()
    logger.info(f"Index built in {(ts1 - ts0).total_seconds():.2f} s; rows={len(index)}")

    # diagnostics
    print(f"Index size: {len(index)} rows")
    print(f"Regions: {n_regions}")
    print(f"Sectors: {sector_entity['Sector_name'].nunique()}")
    print(f"Entities: {sector_entity['Entity'].nunique()}")
    print(f"Estimated RAM size: {index.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

    # save
    folder_name = r"data\proc\index"
    os.makedirs(folder_name, exist_ok=True)
    time = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    index_path = os.path.join(folder_name, f"index_{time}.parquet")
    index.to_parquet(index_path, compression="snappy")
    logger.info(f"Index saved to '{index_path}'.")

    return None


if __name__ == "__main__":
    build_index()