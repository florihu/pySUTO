"""
init_icsg_feed.py
=================
Initialise the ICSG copper-flow DataFeed end-to-end.

This script is a thin driver.  All pipeline logic lives in feedpipe.py.

Steps (mirroring feedpipe.py)
------------------------------
A  Read metadata sidecar → DataFeedMetadata
B  Load raw data + concordance tables
C  Build ConcordanceTable → WeightMatrix          (via feedpipe step_B/C)
D  Build F_src vector + propagate uncertainty     (via feedpipe step_D)
E  Register + persist everything                  (via feedpipe.register_feed)
F  Diagnostics

═══════════════════════════════════════════════════════════
USER CONFIGURATION  ←  only section you ever edit
═══════════════════════════════════════════════════════════

INPUT (you provide these files):

  data/input/feed/<FEED_ID>/v<VERSION>/
      values.csv          ← raw flow data
      conc.xlsx           ← concordance mapping workbook
      meta.xlsx           ← pedigree / provenance sidecar

  data/proc/acc/<ACC_ID>/  ← accounting structure built by init_acc.py

OUTPUT (written automatically via DataFeedRegistry):

  data/proc/
    registry/
      registry.json          ← master index across all feeds × versions
    feed/
      icsg/
        metadata.json        ← DataFeedMetadata (human-readable)
        icsg.pkl             ← WeightMatrix pickle
        result.npz           ← F_root, sigma_root, cv_root, cv_src, observed_zero (#76)
    diagnostics/
      icsg_v<VERSION>_diagnostics.png
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------

FEED_ID = "icsg"

# #76: a weighted fan's domestic leg at trade-share weight 0 is an observed zero,
# covered at 0 with the net-stock seed's pricing (ADR 0040) on the usage U its fans
# distribute: σ = max(OBSERVED_ZERO_CV·|U|, OBSERVED_ZERO_FLOOR_KT). Mirrors
# collect_aggregate_impute's STOCK_SEED_CV / STOCK_SEED_ABS_FLOOR_KT: change both or
# neither.
OBSERVED_ZERO_CV       = 0.01
OBSERVED_ZERO_FLOOR_KT = 0.3
# The signed stock legs: their zero-weight domestic leg is a supply-mix share
# (ADR 0043) whose zero the net-stock seed prices (ADR 0040), not an observed zero.
STOCK_SECTORS_TO       = ("MESC", "USC")
# ACC_ID is loaded from config/config.yaml → pipeline.acc_id below, after
# the pysuto package is added to sys.path.

# ---------------------------------------------------------------------------
# FIXED PATHS  (resolved at runtime via the input registry; do not edit)
# ---------------------------------------------------------------------------

_INPUT_ROOT  = os.path.join("data", "input")
ACC_DIR      = os.path.join("data", "proc", "acc")
REGISTRY_DIR = os.path.join("data", "proc", "feed", "registry")

# Filled in by main() once the input registry has been initialised.
RAW_DATA: str  = ""
CONC_FILE: str = ""
META_FILE: str = ""

# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(__file__))

from pysuto.acc      import AccountingStructure
from pysuto.config   import load_config
from pysuto.meta     import DataFeedMetadata
from pysuto.conc     import ConcordanceTable, WeightMatrix
from pysuto.registry import DataFeedRegistry
from pysuto.input_registry import InputRegistries
from pysuto.feedpipe import (
    step_A_read_metadata,
    step_B_build_concordance,
    step_C_build_weight_matrix,
    propagate_with_row_penalty,
    build_F_src as _build_F_src,
    register_feed,
)
from pysuto.diag import FeedDiagnostics

ACC_ID = load_config().pipeline.acc_id


# ===========================================================================
# A  LOAD ACCOUNTING STRUCTURE
# ===========================================================================

def load_acc_structure() -> AccountingStructure:
    print(f"\n{'='*60}")
    print("A  Loading AccountingStructure")
    print(f"   id  : {ACC_ID}")
    print(f"   dir : {ACC_DIR}")

    acc_path = os.path.join(ACC_DIR, ACC_ID)
    if not os.path.isdir(acc_path):
        available = (
            "\n".join(
                f"  • {d}" for d in sorted(os.listdir(ACC_DIR))
                if os.path.isdir(os.path.join(ACC_DIR, d))
            )
            if os.path.isdir(ACC_DIR) else
            f"  ('{ACC_DIR}' does not exist)"
        )
        raise FileNotFoundError(
            f"\nAccounting structure '{ACC_ID}' not found at '{acc_path}'.\n"
            f"Set pipeline.acc_id in config/config.yaml.\n"
            f"Available structures:\n{available}"
        )

    acc = AccountingStructure.load(id=ACC_ID, storage_dir=ACC_DIR)
    print(f"   nodes           : {acc.n_nodes:,}")
    print(f"   admissible edges: {len(acc.edge_index):,}")
    print(f"   time range      : {acc.time_range[0]} – {acc.time_range[-1]}")
    print(f"   valuations      : {acc.valuation}")
    return acc


# ===========================================================================
# B  LOAD RAW DATA + CONCORDANCE
# ===========================================================================

def load_raw_data() -> pd.DataFrame:
    _check_file(RAW_DATA, "raw flow data")
    df = pd.read_csv(RAW_DATA)
    df["_sector_key"] = df.apply(_make_sector_key, axis=1)
    df["Year"] = df["Year"].astype(str)
    return df


def _make_sector_key(row: pd.Series) -> str:
    """Composite key for sector_map: disambiguates process+material combos."""
    proc, typ, su = row["Process"], row["Type"], row["Supply_Use"]
    if su == "Supply":
        return proc
    if proc == "Smelting" and typ in ("Scrap", "Srap"):
        return "Smelting_Scrap"
    if proc == "Smelting":
        return "Smelting_Concentrate"
    if proc == "Refining" and typ == "Scrap":
        return "Refining_Scrap"
    if proc == "Refining":
        # ADR 0021: the refining Use row is total refined production placed on
        # the anode entering ER, not the matte entering CFR.
        return "Refining_Anode"
    return proc


def load_concordance_tables() -> dict[str, pd.DataFrame]:
    _check_file(CONC_FILE, "concordance workbook")
    sheets = pd.read_excel(CONC_FILE, sheet_name=None)

    # icsg_type / supply_use feed the fan-policy leg filter (ADR 0008)
    sector_map    = sheets["sector_map"][["src_sector", "root_sector",
                                          "icsg_type",  "supply_use"]].drop_duplicates()
    region_map    = sheets["region_map"][["src_region",     "root_region"]].drop_duplicates()
    time_map      = sheets["time_map"][  ["src_time",       "root_time"]  ].drop_duplicates()
    valuation_map = sheets["valuation_map"][["src_valuation","root_valuation"]].drop_duplicates()
    var_map       = sheets["variable_map"][["raw_var", "src_var", "Supply_Use"]].copy()

    time_map["src_time"]  = time_map["src_time"].astype(str)
    time_map["root_time"] = time_map["root_time"].astype(str)

    return dict(
        sector_map=sector_map,
        region_map=region_map,
        time_map=time_map,
        valuation_map=valuation_map,
        var_map=var_map,
    )


def load_meta_sidecar() -> DataFeedMetadata:
    """
    Load pedigree from the Excel sidecar if it exists,
    otherwise fall back to a default DataFeedMetadata.
    """
    if os.path.isfile(META_FILE):
        print(f"   meta sidecar : {META_FILE}")
        return step_A_read_metadata(META_FILE)

    print(f"   meta sidecar : not found, using default pedigree scores")
    return DataFeedMetadata(feed_name=FEED_ID)


# ===========================================================================
# C  BUILD CONCORDANCE TABLE + WEIGHT MATRIX  (delegates to feedpipe)
# ===========================================================================

def build_concordance(
    raw_df: pd.DataFrame,
    tables: dict,
    acc: AccountingStructure,
    user_weights_df: pd.DataFrame | None = None,
) -> ConcordanceTable:
    print(f"\n{'='*60}")
    print("C  Building ConcordanceTable")

    var_map_df = tables["var_map"].copy()
    var_map_df["Supply_Use"] = var_map_df["Supply_Use"].replace("", pd.NA)

    # Fan policy (ADR 0008): each ICSG Use variable names its input product
    # (sector_map icsg_type) — its fan may only land on legs of that supplier
    # sector. Without this, the '<any>' open side fans across every incoming
    # edge of the consumer node (e.g. Smelting_Scrap value on Concentrate
    # legs) and route-keyed trade-share weights bleed across products.
    sm = tables["sector_map"]
    leg_filter = dict(
        sm.loc[sm["supply_use"].astype(str).str.lower() == "use",
               ["src_sector", "icsg_type"]].itertuples(index=False, name=None)
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ct = step_B_build_concordance(
            raw_df                = raw_df,
            acc_struct            = acc,
            sector_map_df         = tables["sector_map"],
            region_map_df         = tables["region_map"],
            time_map_df           = tables["time_map"],
            valuation_map_df      = tables["valuation_map"],
            var_map_df            = var_map_df,
            feed_id               = FEED_ID,
            aux_col               = "Supply_Use",
            # Fan-out across a node's edges (resolved by weights) is allowed.
            # A classified relation with no edge in the acc is rejected
            # unconditionally by the structural-validity guard in
            # ConcordanceTable.from_raw (independent of this flag).
            allow_unmatched_edges = True,
            # BACI-trade-share producer weights (ADR 0002 option D); None ⇒
            # uniform fan, which the fan policy below then refuses to
            # distribute (ADR 0008).
            user_weights_df       = user_weights_df,
            leg_filter            = leg_filter,
            drop_unweighted_fans  = True,
        )

    unmatched     = getattr(ct, "unmatched_raw_rows", [])
    undistributed = getattr(ct, "undistributed_raw_rows", [])
    print(f"   src edges (unique): {ct.n_src_edges:,}")
    print(f"   root edges        : {ct.n_root_edges:,}")
    print(f"   concordance rows  : {len(ct.tab):,}")
    print(f"   unmatched rows    : {len(unmatched):,}")
    print(f"   undistributed     : {len(undistributed):,} raw rows (unweighted fans)")
    for w in caught:
        print(f"   ⚠  {w.message}")
    return ct


def build_weight_matrix(ct: ConcordanceTable) -> WeightMatrix:
    print(f"\n{'='*60}")
    print("C  Building WeightMatrix")
    wm = step_C_build_weight_matrix(ct)
    print(f"   shape : {wm.weight_matrix.shape}")
    print(f"   nnz   : {wm.weight_matrix.nnz}")
    return wm


# ===========================================================================
# D  BUILD F_SRC + PROPAGATE UNCERTAINTY
# ===========================================================================

def build_F_src(raw_df: pd.DataFrame, ct: ConcordanceTable) -> np.ndarray:
    """Aggregate raw values to the src-edge level (delegates to feedpipe)."""
    F_src = _build_F_src(raw_df, ct, label="icsg")
    print(f"\n{'='*60}")
    print("D  F_src vector")
    print(f"   length      : {len(F_src):,}")
    print(f"   non-zero    : {np.count_nonzero(F_src):,}")
    print(f"   total (kt)  : {F_src.sum():,.1f}")
    return F_src


def observed_zero_sigma(legs: pd.DataFrame, F_src: np.ndarray,
                        sigma_root: np.ndarray) -> np.ndarray:
    """σ of every observed zero, NaN on every other root edge (#76).

    ``legs`` lists the zero-weight domestic legs (``src_id``, ``edge_id``) of
    ``ConcordanceTable.zero_weight_domestic_legs``. Each edge is priced on the
    usage its fans distribute, ``U = Σ |F_src[src_id]|``: an aggregate region
    (ROW) fans each member country's row onto the same legs. A leg another fan
    feeds (``sigma_root > 0``) is a flow, not an observed zero.
    """
    edge = legs["edge_id"].to_numpy()
    usage = np.zeros(len(sigma_root))
    np.add.at(usage, edge, np.abs(F_src[legs["src_id"].to_numpy()]))
    out = np.full(len(sigma_root), np.nan)
    out[edge] = np.maximum(OBSERVED_ZERO_CV * usage[edge], OBSERVED_ZERO_FLOOR_KT)
    out[sigma_root > 0] = np.nan
    return out


def stamp_observed_zeros(ct: ConcordanceTable, F_src: np.ndarray, result) -> int:
    """Mark the feed's observed zeros on ``result`` and give them their σ (#76).

    Sets ``result.observed_zero`` and overwrites ``result.sigma_root`` there, after
    propagation, so no per-edge CV multiplier reaches them. Returns their count.
    """
    sigma0 = observed_zero_sigma(ct.zero_weight_domestic_legs(STOCK_SECTORS_TO),
                                 F_src, result.sigma_root)
    result.observed_zero = ~np.isnan(sigma0)
    result.sigma_root[result.observed_zero] = sigma0[result.observed_zero]
    return int(result.observed_zero.sum())


def propagate(wm: WeightMatrix, F_src: np.ndarray, meta: DataFeedMetadata, acc):
    print(f"\n{'='*60}")
    print("D  Propagating uncertainty (scipy.sparse)")
    result, row_summary = propagate_with_row_penalty(wm, F_src, meta, acc)
    print(f"   CV_src       : {result.cv_src * 100:.2f} %")
    print(f"   F_root sum   : {result.F_root.sum():,.1f} kt")
    print(f"   mean CV_root : {np.nanmean(result.cv_root) * 100:.2f} %")
    return result, row_summary


# ===========================================================================
# E  REGISTER + PERSIST  (delegates to feedpipe.register_feed)
# ===========================================================================

def persist_and_register(
    ct:     ConcordanceTable,
    wm:     WeightMatrix,
    F_src:  np.ndarray,
    result,
    meta:   DataFeedMetadata,
    parents: dict,
    row_summary: dict = None,
):
    print(f"\n{'='*60}")
    print("E  Registering feed")

    registry = DataFeedRegistry(odir=REGISTRY_DIR)
    registry.load_index()
    wm.odir = REGISTRY_DIR

    entry = register_feed(
        wm, meta, result, registry,
        parents = parents,
        F_src   = F_src,
        stage_summary = row_summary,
    )

    print(f"   run_id           {entry.run_id}")
    print(f"   run dir          {entry.run_dir}")
    return entry


# ===========================================================================
# F  DIAGNOSTICS
# ===========================================================================

def diagnostics(
    raw_df: pd.DataFrame,
    ct:     ConcordanceTable,
    F_src:  np.ndarray,
    result,
    entry,
) -> None:
    print(f"\n{'='*60}")
    print("F  Generating diagnostics")

    diag = FeedDiagnostics(
        feed_run_id = entry.run_id,
        feed_id     = FEED_ID,
        acc_id      = ACC_ID,
        raw_df      = raw_df,
        ct          = ct,
        F_src       = F_src,
        result      = result,
        conc_file   = CONC_FILE,
    )
    diag.plot(odir=os.path.join(entry.run_dir, "diag"))
    diag.summary()

    print(f"\n  Written to: {entry.run_dir}/")


# ===========================================================================
# HELPERS
# ===========================================================================

def _check_file(path: str, label: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"\nMissing {label}:\n  {path}\n"
            f"Place the file there or update the path constant at the top of this script."
        )


# ===========================================================================
# MAIN
# ===========================================================================

def _resolve_input_paths(inputs: InputRegistries) -> dict:
    global RAW_DATA, CONC_FILE, META_FILE
    feed_dir  = inputs.feed.resolve(FEED_ID)
    RAW_DATA  = str(feed_dir / "values.csv")
    CONC_FILE = str(feed_dir / "conc.xlsx")
    META_FILE = str(feed_dir / "meta.xlsx")
    return {
        "acc":  inputs.acc.entry(ACC_ID),
        "feed": inputs.feed.entry(FEED_ID),
    }


# Params id holding the BACI-trade-share producer weights (ADR 0002 option D).
TRADE_SHARES_ID = "trade_shares"


def load_trade_shares(inputs: InputRegistries):
    """Load the optional BACI-trade-share weights as user_weights_df.

    Returns (df, entry) or (None, None) when the artifact is not registered, so
    the ICSG build falls back to the uniform producer fan (unchanged behaviour).
    Run scripts/build_trade_shares.py to (re)generate it.
    """
    if TRADE_SHARES_ID not in inputs.params.ids():
        print(f"   trade_shares : not registered — uniform fan")
        return None, None
    path = inputs.params.resolve(TRADE_SHARES_ID) / "params.csv"
    df = pd.read_csv(path)
    entry = inputs.params.entry(TRADE_SHARES_ID)
    print(f"   trade_shares : {TRADE_SHARES_ID} v{entry['version']} "
          f"(code={entry['code']}, {len(df):,} weight rows)")
    return df, entry


def main() -> None:
    # 0 — input registries
    inputs    = InputRegistries.initialize(root=_INPUT_ROOT)
    input_ref = _resolve_input_paths(inputs)
    print(f"\n{'='*60}")
    print(f"   input feed : {input_ref['feed']['id']} v{input_ref['feed']['version']}"
          f"  (code={input_ref['feed']['code']})")
    print(f"   input acc  : {input_ref['acc']['id']} v{input_ref['acc']['version']}")

    # A — accounting structure
    acc = load_acc_structure()

    # B — raw data + concordance tables + metadata sidecar
    print(f"\n{'='*60}")
    print("B  Loading inputs")
    print(f"   raw data     : {RAW_DATA}")
    print(f"   concordance  : {CONC_FILE}")
    print(f"   meta sidecar : {META_FILE}")
    raw_df = load_raw_data()
    tables = load_concordance_tables()
    meta   = load_meta_sidecar()
    trade_shares_df, trade_shares_entry = load_trade_shares(inputs)
    print(f"   rows loaded  : {len(raw_df):,}")
    print(f"   sector map   : {len(tables['sector_map'])} entries")
    print(f"   region map   : {len(tables['region_map'])} entries")
    print(f"   time map     : {len(tables['time_map'])} entries")
    if trade_shares_entry is not None:
        input_ref["trade_shares"] = trade_shares_entry

    # C — concordance + weight matrix
    ct = build_concordance(raw_df, tables, acc, user_weights_df=trade_shares_df)
    wm = build_weight_matrix(ct)

    # D — source flow vector + uncertainty propagation
    F_src  = build_F_src(raw_df, ct)
    result, row_summary = propagate(wm, F_src, meta, acc)
    n_zero = stamp_observed_zeros(ct, F_src, result)
    print(f"   observed zeros: {n_zero:,} domestic legs at trade-share weight 0 (#76)")

    # E — register + persist (via DataFeedRegistry / feedpipe)
    parents = {"inputs": input_ref, "upstream": {}}
    entry = persist_and_register(ct, wm, F_src, result, meta, parents=parents,
                                 row_summary=row_summary)

    # F — diagnostics
    diagnostics(raw_df, ct, F_src, result, entry)


if __name__ == "__main__":
    main()
