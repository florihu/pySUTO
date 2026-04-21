"""
init_baci_feed.py
=================
Initialise the BACI copper-trade DataFeed end-to-end.

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
      values.csv          ← bilateral trade flows (from baci_parse_feed.py)
      conc.xlsx           ← concordance mapping workbook
      meta.xlsx           ← pedigree / provenance sidecar

  data/proc/acc/<ACC_ID>/  ← accounting structure built by init_copper_acc.py

OUTPUT (written automatically via DataFeedRegistry):

  data/proc/
    registry/
      registry.json          ← master index across all feeds × versions
    feed/
      baci/
        metadata.json        ← DataFeedMetadata (human-readable)
        baci.pkl             ← WeightMatrix pickle
        result.npz           ← F_root, sigma_root, cv_root, cv_src
    diagnostics/
      baci_v<VERSION>_diagnostics.png

  logs/
      baci_v<VERSION>_<timestamp>.log  ← full run log
"""

from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------

FEED_ID = "baci"
VERSION = 0
ACC_ID  = "copper"

# ---------------------------------------------------------------------------
# FIXED PATHS  (match project folder structure; do not edit)
# ---------------------------------------------------------------------------

_INPUT_ROOT  = os.path.join("data", "input", "feed")
_FEED_VDIR   = os.path.join(_INPUT_ROOT, FEED_ID, f"v{VERSION}")

RAW_DATA     = os.path.join(_FEED_VDIR, "values.csv")
CONC_FILE    = os.path.join(_FEED_VDIR, "conc.xlsx")
META_FILE    = os.path.join(_FEED_VDIR, "meta.xlsx")
WEIGHTS_FILE = os.path.join(_FEED_VDIR, "weights.csv")   # optional user weight table

ACC_DIR      = os.path.join("data", "proc", "acc")
REGISTRY_DIR = os.path.join("data", "proc", "feed", "registry")
DIAG_DIR     = os.path.join("data", "proc", "feed", "diagnostics")

# ---------------------------------------------------------------------------
# LOGGING  — one timestamped file per run, plus console
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysuto.logging import get_logger, add_run_handler

_RUN_LOG = os.path.join(
    "logs", f"{FEED_ID}_v{VERSION}_{datetime.now():%Y%m%d_%H%M%S}.log"
)
add_run_handler(_RUN_LOG)
logger = get_logger(__name__)

# ---------------------------------------------------------------------------

from pysuto.acc      import AccountingStructure
from pysuto.meta     import DataFeedMetadata
from pysuto.conc     import ConcordanceTable, WeightMatrix
from pysuto.registry import DataFeedRegistry
from pysuto.feedpipe import (
    step_A_read_metadata,
    step_B_build_concordance,
    step_C_build_weight_matrix,
    step_D_apply_and_propagate,
    register_feed,
)
from pysuto.diag import FeedDiagnostics


# ===========================================================================
# A  LOAD ACCOUNTING STRUCTURE
# ===========================================================================

def load_acc_structure() -> AccountingStructure:
    logger.info("=" * 60)
    logger.info("A  Loading AccountingStructure")
    logger.info("   id  : %s", ACC_ID)
    logger.info("   dir : %s", ACC_DIR)

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
            f"Set ACC_ID at the top of this file.\n"
            f"Available structures:\n{available}"
        )

    acc = AccountingStructure.load(id=ACC_ID, storage_dir=ACC_DIR)
    logger.info("   nodes           : %d", acc.n_nodes)
    logger.info("   admissible edges: %d", len(acc.edge_index))
    logger.info("   time range      : %s – %s", acc.time_range[0], acc.time_range[-1])
    logger.info("   valuations      : %s", acc.valuation)
    return acc


# ===========================================================================
# B  LOAD RAW DATA + CONCORDANCE
# ===========================================================================

def load_raw_data() -> pd.DataFrame:
    _check_file(RAW_DATA, "raw flow data")
    df = pd.read_csv(RAW_DATA)
    # Year must be str for time_map lookup; Sector stays int for sector_map lookup
    df["Year"] = df["Year"].astype(str)
    return df


def load_concordance_tables() -> dict[str, pd.DataFrame]:
    _check_file(CONC_FILE, "concordance workbook")
    sheets = pd.read_excel(CONC_FILE, sheet_name=None)

    sector_map    = sheets["sector_map"][["src_sector",      "root_sector"]].drop_duplicates()
    region_map    = sheets["region_map"][["src_region",       "root_region"]].drop_duplicates()
    time_map      = sheets["time_map"][  ["src_time",         "root_time"]  ].drop_duplicates()
    valuation_map = sheets["valuation_map"][["src_valuation", "root_valuation"]].drop_duplicates()
    var_map       = sheets["variable_map"][["raw_var", "src_var"]].copy()

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
    if os.path.isfile(META_FILE):
        logger.info("   meta sidecar : %s", META_FILE)
        return step_A_read_metadata(META_FILE)
    logger.warning("   meta sidecar not found — using default pedigree scores")
    return DataFeedMetadata(feed_name=FEED_ID)


# ===========================================================================
# C  BUILD CONCORDANCE TABLE + WEIGHT MATRIX  (delegates to feedpipe)
# ===========================================================================

def load_user_weights() -> pd.DataFrame | None:
    """Load optional user weight table; returns None if file absent."""
    if not os.path.isfile(WEIGHTS_FILE):
        logger.info("   weights file not found — using equal allocation")
        return None
    wdf = pd.read_csv(WEIGHTS_FILE)
    logger.info("   weights file : %s  (%d entries)", WEIGHTS_FILE, len(wdf))
    return wdf


def build_concordance(
    raw_df: pd.DataFrame,
    tables: dict,
    acc: AccountingStructure,
    user_weights: pd.DataFrame | None = None,
) -> ConcordanceTable:
    logger.info("=" * 60)
    logger.info("C  Building ConcordanceTable")

    var_map_df = tables["var_map"].copy()

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
            aux_col               = None,  # BACI has explicit from/to cols — no discriminator
            allow_unmatched_edges = True,
            user_weights_df       = user_weights,
        )

    for w in caught:
        logger.warning("   ⚠  %s", w.message)
    return ct


def build_weight_matrix(ct: ConcordanceTable) -> WeightMatrix:
    logger.info("=" * 60)
    logger.info("C  Building WeightMatrix")
    return step_C_build_weight_matrix(ct)


# ===========================================================================
# D  BUILD F_SRC + PROPAGATE UNCERTAINTY
# ===========================================================================

def build_F_src(raw_df: pd.DataFrame, ct: ConcordanceTable) -> np.ndarray:
    """Aggregate raw values to the src-edge level aligned to ct.src_edges."""
    src_raw = (
        ct.tab[["raw_id"] + list(ConcordanceTable.SRC_COLS)]
        .drop_duplicates(subset=list(ConcordanceTable.SRC_COLS))
    )
    merged = src_raw.merge(
        raw_df.reset_index().rename(columns={"index": "_raw_idx"}),
        left_on="raw_id", right_on="_raw_idx", how="left",
    )
    agg = merged.groupby(list(ConcordanceTable.SRC_COLS))["Value"].sum().reset_index()

    F_src = np.zeros(ct.n_src_edges)
    for _, row in agg.iterrows():
        key = tuple(row[c] for c in ConcordanceTable.SRC_COLS)
        if key in ct.src_edge_to_id:
            F_src[ct.src_edge_to_id[key]] = row["Value"]

    logger.info("=" * 60)
    logger.info("D  F_src vector")
    logger.info("   length    : %d", len(F_src))
    logger.info("   non-zero  : %d", int(np.count_nonzero(F_src)))
    logger.info("   total kt  : %.1f", float(F_src.sum()))
    return F_src


def propagate(wm: WeightMatrix, F_src: np.ndarray, meta: DataFeedMetadata):
    logger.info("=" * 60)
    logger.info("D  Propagating uncertainty")
    return step_D_apply_and_propagate(wm, F_src, meta)


# ===========================================================================
# E  REGISTER + PERSIST  (delegates to feedpipe.register_feed)
# ===========================================================================

def persist_and_register(
    ct:     ConcordanceTable,
    wm:     WeightMatrix,
    F_src:  np.ndarray,
    result,
    meta:   DataFeedMetadata,
) -> None:
    logger.info("=" * 60)
    logger.info("E  Registering feed")

    registry = DataFeedRegistry(odir=REGISTRY_DIR)
    registry.load_index()
    wm.odir = REGISTRY_DIR

    register_feed(wm, meta, result, registry, overwrite=True)

    feed_dir = os.path.join(REGISTRY_DIR, FEED_ID)
    np.save(os.path.join(feed_dir, "F_src.npy"), F_src)
    ct.save(path=os.path.join(feed_dir, "concordance.pkl"))

    logger.info("   registry.json    ✓")
    logger.info("   metadata.json    ✓")
    logger.info("   %s.pkl           ✓", FEED_ID)
    logger.info("   result.npz       ✓")
    logger.info("   F_src.npy        ✓")
    logger.info("   concordance.pkl  ✓")


# ===========================================================================
# F  DIAGNOSTICS
# ===========================================================================

def diagnostics(
    raw_df: pd.DataFrame,
    ct:     ConcordanceTable,
    F_src:  np.ndarray,
    result,
) -> None:
    logger.info("=" * 60)
    logger.info("F  Generating diagnostics")

    diag = FeedDiagnostics(
        feed_id   = FEED_ID,
        version   = VERSION,
        acc_id    = ACC_ID,
        raw_df    = raw_df,
        ct        = ct,
        F_src     = F_src,
        result    = result,
        conc_file = CONC_FILE,
    )
    diag.plot(odir=DIAG_DIR)
    diag.summary()

    logger.info("   written to: %s/", REGISTRY_DIR)
    logger.info("               %s/", DIAG_DIR)
    logger.info("   run log   : %s", _RUN_LOG)


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

def main() -> None:
    logger.info("=" * 60)
    logger.info("init_baci_feed.py  —  feed=%s  version=%d  acc=%s",
                FEED_ID, VERSION, ACC_ID)
    logger.info("run log → %s", _RUN_LOG)

    # A — accounting structure
    acc = load_acc_structure()

    # B — raw data + concordance tables + metadata sidecar
    logger.info("=" * 60)
    logger.info("B  Loading inputs")
    logger.info("   raw data     : %s", RAW_DATA)
    logger.info("   concordance  : %s", CONC_FILE)
    logger.info("   meta sidecar : %s", META_FILE)
    raw_df       = load_raw_data()
    tables       = load_concordance_tables()
    meta         = load_meta_sidecar()
    user_weights = load_user_weights()
    logger.info("   rows loaded  : %d", len(raw_df))
    logger.info("   sector map   : %d entries", len(tables["sector_map"]))
    logger.info("   region map   : %d entries", len(tables["region_map"]))
    logger.info("   time map     : %d entries", len(tables["time_map"]))

    # C — concordance + weight matrix
    ct = build_concordance(raw_df, tables, acc, user_weights)
    wm = build_weight_matrix(ct)

    # D — source flow vector + uncertainty propagation
    F_src  = build_F_src(raw_df, ct)
    result = propagate(wm, F_src, meta)

    # E — register + persist
    persist_and_register(ct, wm, F_src, result, meta)

    # F — diagnostics
    diagnostics(raw_df, ct, F_src, result)

    logger.info("=" * 60)
    logger.info("Done.")


if __name__ == "__main__":
    main()
