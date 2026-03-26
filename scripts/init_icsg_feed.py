"""
init_icsg_feed.py
=================
Initialise the ICSG copper-flow DataFeed end-to-end.

Steps
-----
A  Load shared AccountingStructure (pre-built, user specifies ACC_ID)
B  Read raw ICSG data + concordance tables
C  Build ConcordanceTable  →  WeightMatrix
D  Build F_src vector + propagate uncertainty
E  Save artefacts + update registry
F  Diagnostics

═══════════════════════════════════════════════════════════
USER CONFIGURATION  ←  only section you ever edit
═══════════════════════════════════════════════════════════

INPUT (you provide these files):

  data/input/
    acc/
      <ACC_ID>/              ← accounting structure built by init_acc.py
    feed/
      icsg/                  ← raw flow data
        icsg_flows_clean.csv
    conc/
      icsg.xlsx              ← concordance mapping workbook
    meta/
      ICSG.xlsx              ← pedigree / provenance sidecar

OUTPUT (written automatically):

  data/proc/
    feed/
      icsg/
        v<VERSION>/
          input/
            concordance.pkl      ← ConcordanceTable
            weight_matrix.pkl    ← WeightMatrix
          output/
            F_src.npy
            F_root.npy
            sigma_root.npy
            cv_root.npy
          meta/
            metadata.json
            diagnostics.png
    registry/
      registry.json              ← index across all feeds × versions
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ===========================================================================
# USER CONFIGURATION  ←  edit these, nothing else
# ===========================================================================

# ── Accounting structure ────────────────────────────────────────────────────
# Must already exist at  data/input/acc/<ACC_ID>/

# ── Feed versioning ─────────────────────────────────────────────────────────
# Bump VERSION when re-running with revised data or concordance.
# Each version is stored independently; registry points to the active one.



# ===========================================================================
# FIXED PATHS  ←  match project folder structure; do not edit
# ===========================================================================

FEED_ID = "icsg"
VERSION = 0

# ── Input paths (user-managed) ──────────────────────────────────────────────
_INPUT_ROOT  = "data/input/feed"
RAW_DATA     = os.path.join(_INPUT_ROOT,  FEED_ID, f'v{VERSION}' "values.csv")
CONC_FILE    = os.path.join(_INPUT_ROOT,  FEED_ID, f'v{VERSION}' "conc.xlsx")
META_FILE    = os.path.join(_INPUT_ROOT, FEED_ID, f'v{VERSION}' "meta.xlsx")


# acc
ACC_DIR = "data/proc/acc"
ACC_ID = "copper"





# ── Output paths (written automatically) ───────────────────────────────────
_PROC_ROOT   = "data/proc"
REGISTRY_DIR = os.path.join(_PROC_ROOT, "registry")

def _feed_root()   -> str: return os.path.join(_PROC_ROOT, "feed", FEED_ID)
def _version_dir() -> str: return os.path.join(_feed_root(), f"v{VERSION}")
def _input_dir()   -> str: return os.path.join(_version_dir(), "input")
def _output_dir()  -> str: return os.path.join(_version_dir(), "output")
def _meta_dir()    -> str: return os.path.join(_version_dir(), "meta")

# ===========================================================================
sys.path.insert(0, os.path.dirname(__file__))

from pysuto.acc      import AccountingStructure
from pysuto.meta     import DataFeedMetadata
from pysuto.conc     import ConcordanceTable, WeightMatrix
from pysuto.registry import DataFeedRegistry


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
            f"Set ACC_ID at the top of this file.\n"
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
        return "Refining_Anode"
    return proc


def load_concordance_tables() -> dict[str, pd.DataFrame]:
    _check_file(CONC_FILE, "concordance workbook")
    sheets = pd.read_excel(CONC_FILE, sheet_name=None)

    sector_map    = sheets["sector_map"][["src_sector",    "root_sector"]].drop_duplicates()
    region_map    = sheets["region_map"][["src_region",    "root_region"]].drop_duplicates()
    time_map      = sheets["time_map"][  ["src_time",      "root_time"]  ].drop_duplicates()
    valuation_map = sheets["valuation_map"][["src_valuation","root_valuation"]].drop_duplicates()
    var_map       = sheets["variable_map"][["raw_var", "src_var", "Supply_Use"]].copy()

    time_map["src_time"]  = time_map["src_time"].astype(str)
    time_map["root_time"] = time_map["root_time"].astype(str)

    return dict(sector_map=sector_map, region_map=region_map,
                time_map=time_map, valuation_map=valuation_map,
                var_map=var_map)


def load_meta_sidecar() -> DataFeedMetadata:
    """
    Load pedigree from data/input/meta/ICSG.xlsx if it exists,
    otherwise fall back to the hardcoded constants above.
    """
    if os.path.isfile(META_FILE):
        print(f"   meta sidecar : {META_FILE}")
        return DataFeedMetadata.from_excel(META_FILE)

    print(f"   meta sidecar : not found, using hardcoded pedigree constants")
    return DataFeedMetadata.from_excel(path = META_FILE, sheet_name=None)  # empty DataFrame → defaults 


# ===========================================================================
# C  BUILD CONCORDANCE TABLE + WEIGHT MATRIX
# ===========================================================================

def build_concordance(raw_df: pd.DataFrame,
                      tables: dict,
                      acc: AccountingStructure) -> ConcordanceTable:
    print(f"\n{'='*60}")
    print("C  Building ConcordanceTable")

    var_map_df = tables["var_map"].copy()
    var_map_df["Supply_Use"] = var_map_df["Supply_Use"].replace("", pd.NA)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ct = ConcordanceTable.from_raw(
            raw_df           = raw_df,
            acc_struct       = acc,
            sector_map_df    = tables["sector_map"],
            region_map_df    = tables["region_map"],
            time_map_df      = tables["time_map"],
            valuation_map_df = tables["valuation_map"],
            var_map_df       = var_map_df,
            aux_col          = "Supply_Use",
            allow_unmatched_edges = True,
            id               = FEED_ID,
        )

    unmatched = getattr(ct, "unmatched_raw_rows", [])
    print(f"   src edges (unique): {ct.n_src_edges:,}")
    print(f"   root edges        : {ct.n_root_edges:,}")
    print(f"   concordance rows  : {len(ct.tab):,}")
    print(f"   unmatched rows    : {len(unmatched):,}")
    for w in caught:
        print(f"   ⚠  {w.message}")
    return ct


def build_weight_matrix(ct: ConcordanceTable) -> WeightMatrix:
    print(f"\n{'='*60}")
    print("C  Building WeightMatrix")
    wm = WeightMatrix(concordance=ct, id=FEED_ID, odir=_input_dir())
    print(f"   shape : {wm.weight_matrix.shape}")
    return wm


# ===========================================================================
# D  BUILD F_SRC + PROPAGATE UNCERTAINTY
# ===========================================================================

def build_F_src(raw_df: pd.DataFrame, ct: ConcordanceTable) -> np.ndarray:
    """Aggregate raw values to the src-edge level aligned to ct.src_edges."""
    merged = ct.tab.merge(
        raw_df.reset_index().rename(columns={"index": "_raw_idx"}),
        left_on="raw_id", right_on="_raw_idx", how="left",
    )
    agg = merged.groupby(list(ConcordanceTable.SRC_COLS))["Value"].sum().reset_index()

    F_src = np.zeros(ct.n_src_edges)
    for _, row in agg.iterrows():
        key = tuple(row[c] for c in ConcordanceTable.SRC_COLS)
        if key in ct.src_edge_to_id:
            F_src[ct.src_edge_to_id[key]] = row["Value"]

    print(f"\n{'='*60}")
    print("D  F_src vector")
    print(f"   length      : {len(F_src):,}")
    print(f"   non-zero    : {np.count_nonzero(F_src):,}")
    print(f"   total (kt)  : {F_src.sum():,.1f}")
    return F_src


def propagate(wm: WeightMatrix, F_src: np.ndarray, meta: DataFeedMetadata):
    print(f"\n{'='*60}")
    print("D  Propagating uncertainty")
    pg = meta.to_pedigree_scores()
    wm.set_pedigree(pg)
    wm.set_F_src(F_src)
    result = wm.apply_with_uncertainty()
    print(f"   CV_src       : {result.cv_src * 100:.2f} %")
    print(f"   F_root sum   : {result.F_root.sum():,.1f} kt")
    print(f"   mean CV_root : {np.nanmean(result.cv_root) * 100:.2f} %")
    return result


# ===========================================================================
# E  SAVE ARTEFACTS  +  UPDATE REGISTRY
# ===========================================================================

def save_feed(ct: ConcordanceTable,
              wm: WeightMatrix,
              F_src: np.ndarray,
              result,
              meta: DataFeedMetadata) -> None:
    """
    Write all artefacts into the versioned feed folder:

        data/proc/feed/icsg/v<VERSION>/
            input/   concordance.pkl   weight_matrix.pkl
            output/  F_src.npy  F_root.npy  sigma_root.npy  cv_root.npy
            meta/    metadata.json
    """
    print(f"\n{'='*60}")
    print("E  Saving artefacts")
    print(f"   → {_version_dir()}")

    for d in (_input_dir(), _output_dir(), _meta_dir()):
        os.makedirs(d, exist_ok=True)

    # input/ ── structural mapping objects
    ct.odir = _input_dir()
    ct.save(path=os.path.join(_input_dir(), "concordance.pkl"))
    print(f"   input/concordance.pkl    ✓")

    wm.odir = _input_dir()
    wm.save(path=os.path.join(_input_dir(), "weight_matrix.pkl"))
    print(f"   input/weight_matrix.pkl  ✓")

    # output/ ── numerical results
    np.save(os.path.join(_output_dir(), "F_src.npy"),      F_src)
    np.save(os.path.join(_output_dir(), "F_root.npy"),     result.F_root)
    np.save(os.path.join(_output_dir(), "sigma_root.npy"), result.sigma_root)
    np.save(os.path.join(_output_dir(), "cv_root.npy"),    result.cv_root)
    print(f"   output/F_src.npy         ✓")
    print(f"   output/F_root.npy        ✓")
    print(f"   output/sigma_root.npy    ✓")
    print(f"   output/cv_root.npy       ✓")

    # meta/ ── provenance record (diagnostics.png added later by step F)
    meta_dict = {
        "feed_id":                  FEED_ID,
        "version":                  VERSION,
        "acc_id":                   ACC_ID,
        "iedc_type":                meta.iedc_type,
        "reliability":              meta.reliability,
        "completeness":             meta.completeness,
        "temporal_correlation":     meta.temporal_correlation,
        "geographical_correlation": meta.geographical_correlation,
        "other_correlation":        meta.other_correlation,
        "date_of_assessment":       meta.date_of_assessment,
        "person":                   meta.person,
        "dataset_link":             meta.dataset_link,
        "cv_total":                 float(result.cv_src),
        "n_src_edges":              ct.n_src_edges,
        "n_root_edges":             ct.n_root_edges,
        "F_root_sum_kt":            float(result.F_root.sum()),
        "input_files": {
            "raw_data":    RAW_DATA,
            "concordance": CONC_FILE,
            "meta_sidecar": META_FILE,
        },
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    with open(os.path.join(_meta_dir(), "metadata.json"), "w") as f:
        json.dump(meta_dict, f, indent=2)
    print(f"   meta/metadata.json       ✓")


def register_feed(result) -> None:
    """
    Upsert this feed version into  data/proc/registry/registry.json.
    Preserves all other feeds already in the file.
    """
    print(f"\n{'='*60}")
    print("E  Updating registry")

    os.makedirs(REGISTRY_DIR, exist_ok=True)
    registry_path = os.path.join(REGISTRY_DIR, "registry.json")

    registry = {}
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            registry = json.load(f)

    # Each feed keeps a history list + an active pointer
    entry = {
        "version":     VERSION,
        "acc_id":      ACC_ID,
        "version_dir": _version_dir(),
        "input_dir":   _input_dir(),
        "output_dir":  _output_dir(),
        "meta_dir":    _meta_dir(),
        "cv_total":    float(result.cv_src),
        "F_root_sum_kt": float(result.F_root.sum()),
        "created_at":  datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    if FEED_ID not in registry:
        registry[FEED_ID] = {"active_version": VERSION, "versions": {}}

    registry[FEED_ID]["active_version"]           = VERSION
    registry[FEED_ID]["versions"][str(VERSION)]   = entry

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"   {registry_path}  ✓  (feed={FEED_ID}, v={VERSION})")


# ===========================================================================
# F  DIAGNOSTICS
# ===========================================================================

def diagnostics(raw_df: pd.DataFrame,
                ct: ConcordanceTable,
                F_src: np.ndarray,
                result) -> None:
    print(f"\n{'='*60}")
    print("F  Generating diagnostics")

    unmatched_idx = set(getattr(ct, "unmatched_raw_rows", []))
    matched_mask  = ~raw_df.index.isin(unmatched_idx)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"ICSG Feed  v{VERSION}  (acc: {ACC_ID})",
                 fontsize=14, fontweight="bold")

    # 1. Row coverage
    ax = axes[0, 0]
    counts = [matched_mask.sum(), len(unmatched_idx)]
    bars = ax.bar(["Matched", "Unmatched"], counts,
                  color=["#2ECC71", "#E74C3C"], edgecolor="white")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{cnt:,}\n({cnt / len(raw_df) * 100:.1f} %)",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("Row coverage", fontweight="bold")
    ax.set_ylabel("# raw rows")
    ax.set_ylim(0, max(counts) * 1.2)
    ax.grid(axis="y", alpha=0.3)

    # 2. Unmatched breakdown
    ax = axes[0, 1]
    if unmatched_idx:
        unm = raw_df.loc[list(unmatched_idx)].copy()
        mapped_regs = pd.read_excel(
            CONC_FILE, sheet_name="region_map")["src_region"].tolist()
        unm["reason"] = "other"
        unm.loc[~unm["Region"].isin(mapped_regs), "reason"] = "unmapped region"
        rc = unm["reason"].value_counts()
        ax.barh(rc.index, rc.values, color="#E67E22", edgecolor="white")
        ax.set_title("Unmatched – reason", fontweight="bold")
        ax.set_xlabel("# rows")
        ax.grid(axis="x", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "All rows matched ✓",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="#2ECC71")
        ax.set_title("Unmatched – reason", fontweight="bold")

    # 3. Matched value by process
    ax = axes[0, 2]
    pv = raw_df.loc[matched_mask].groupby("Process")["Value"].sum().sort_values()
    ax.barh(pv.index, pv.values, color="#3498DB", edgecolor="white")
    ax.set_title("Matched value by process (kt)", fontweight="bold")
    ax.set_xlabel("kt")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(axis="x", alpha=0.3)

    # 4. F_src distribution
    ax = axes[1, 0]
    nz = F_src[F_src != 0]
    if len(nz):
        ax.hist(nz, bins=40, color="#9B59B6", edgecolor="white", linewidth=0.5)
    ax.set_title("F_src distribution (kt, non-zero)", fontweight="bold")
    ax.set_xlabel("kt")
    ax.set_ylabel("# src edges")
    ax.grid(axis="y", alpha=0.3)

    # 5. CV_root distribution
    ax = axes[1, 1]
    cv_pct = result.cv_root * 100
    valid  = cv_pct[np.isfinite(cv_pct) & (cv_pct > 0)]
    if len(valid):
        ax.hist(valid, bins=30, color="#E74C3C", edgecolor="white", linewidth=0.5)
    ax.set_title("CV_root distribution (%)", fontweight="bold")
    ax.set_xlabel("CV (%)")
    ax.set_ylabel("# root edges")
    ax.grid(axis="y", alpha=0.3)

    # 6. Value coverage by year
    ax = axes[1, 2]
    yr_all   = raw_df.groupby("Year")["Value"].sum().sort_index()
    yr_match = raw_df.loc[matched_mask].groupby("Year")["Value"].sum()
    yr_match = yr_match.reindex(yr_all.index, fill_value=0)
    yrs = yr_all.index.astype(int)
    ax.fill_between(yrs, yr_all.values,   alpha=0.3, color="#E74C3C",
                    label="Total (incl. unmatched)")
    ax.fill_between(yrs, yr_match.values, alpha=0.7, color="#2ECC71",
                    label="Matched")
    ax.set_title("Value coverage by year (kt)", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("kt")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    plt.tight_layout()
    plt.savefig(os.path.join(_meta_dir(), "diagnostics.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   meta/diagnostics.png  ✓")

    # ── text summary ───────────────────────────────────────────────────────
    total     = len(raw_df)
    matched_n = matched_mask.sum()
    unmatch_n = len(unmatched_idx)
    val_match = raw_df.loc[matched_mask]["Value"].sum()
    val_total = raw_df["Value"].sum()

    print(f"\n{'─'*60}")
    print(f"  SUMMARY  –  {FEED_ID}  v{VERSION}  (acc: {ACC_ID})")
    print(f"{'─'*60}")
    print(f"  Raw rows total  : {total:>8,}")
    print(f"  Rows matched    : {matched_n:>8,}  ({matched_n / total * 100:.1f} %)")
    print(f"  Rows unmatched  : {unmatch_n:>8,}  ({unmatch_n / total * 100:.1f} %)")
    print(f"  Value matched   : {val_match:>10,.1f} kt")
    print(f"  Value total     : {val_total:>10,.1f} kt")
    print(f"  Value coverage  :   {val_match / val_total * 100:.1f} %")
    print(f"  Src edges       : {ct.n_src_edges:>8,}")
    print(f"  Root edges      : {ct.n_root_edges:>8,}")
    print(f"  CV_src          :   {result.cv_src * 100:.2f} %")

    if unmatched_idx:
        print(f"\n  Top unmatched regions:")
        unm = raw_df.loc[list(unmatched_idx)]
        for reg, cnt in unm["Region"].value_counts().head(10).items():
            print(f"    {reg:<32} {cnt:>5} rows")

    print(f"{'─'*60}")
    print(f"\n  Written to:")
    print(f"    {_input_dir()}/")
    print(f"    {_output_dir()}/")
    print(f"    {_meta_dir()}/")
    print(f"    {os.path.join(REGISTRY_DIR, 'registry.json')}")


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

def main():
    # A
    acc = load_acc_structure()

    # B
    print(f"\n{'='*60}")
    print("B  Loading inputs")
    print(f"   raw data     : {RAW_DATA}")
    print(f"   concordance  : {CONC_FILE}")
    raw_df = load_raw_data()
    tables = load_concordance_tables()
    meta   = load_meta_sidecar()
    print(f"   rows loaded  : {len(raw_df):,}")
    print(f"   sector map   : {len(tables['sector_map'])} entries")
    print(f"   region map   : {len(tables['region_map'])} entries")
    print(f"   time map     : {len(tables['time_map'])} entries")

    # C
    ct = build_concordance(raw_df, tables, acc)
    wm = build_weight_matrix(ct)

    # D
    F_src  = build_F_src(raw_df, ct)
    result = propagate(wm, F_src, meta)

    # E
    save_feed(ct, wm, F_src, result, meta)
    register_feed(result)

    # F
    diagnostics(raw_df, ct, F_src, result)


if __name__ == "__main__":
    main()
