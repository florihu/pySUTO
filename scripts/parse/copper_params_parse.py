"""
copper_params_parse.py
======================
Convert copper_conc.csv and copper_efficiency.csv — already expressed in
root-system coordinates — into params-template-compliant CSV files.

Inputs
------
data/input/params/copper_conc.csv
    Global Cu-content concentrations (t_Cu / t_material) per flow/SB sector.
    Uncertainty: min/max range.

data/input/params/copper_efficiency.csv
    Global Cu recovery efficiencies per process (t_Cu_out / t_Cu_in).
    Uncertainty: min/max range.

Outputs
-------
data/input/params/copper_conc_params.csv
    Concentration parameters — two rows per sector (use-direction + supply-
    direction) so the ParametricImputer can link Mass↔CC edges regardless
    of which endpoint holds the flow node.

data/input/params/copper_efficiency_params.csv
    Efficiency parameters — one row per process linking the primary input
    USE edge to the primary output SUPPLY edge on the CC valuation.

Sector → entity mapping (from root.ods Sector_Entity sheet)
------------------------------------------------------------
Flow entity : Ore, Concentrate, Matte, Anode, CER, CSXEW, Cathode
SB entity   : CO, WR, Tailings, Slag, Scrap, FU, USC, PSC, MESC

Efficiency process chain (primary Cu pathway, CC valuation)
------------------------------------------------------------
MF        : (Ore/Flow) → MF/Process → (Concentrate/Flow)
Smelting  : (Concentrate/Flow) → Smelting/Process → (Matte/Flow)
CFR       : (Matte/Flow) → CFR/Process → (Anode/Flow)
SXEW      : (Ore/Flow) → SXEW/Process → (CSXEW/Flow)
ER        : (Anode/Flow) → ER/Process → (CER/Flow)

Note on Scrap classes
---------------------
copper_conc.csv distinguishes Scrap_class_1 (high-grade, 90-98% Cu) and
Scrap_class_2 (low-grade, 40-60% Cu).  The root accounting structure has a
single 'Scrap/SB' sector.  Both classes are mapped to Scrap/SB and receive
separate param_ids so they remain traceable; the user must extend the root
if class-level resolution is needed.

Usage
-----
cd /path/to/pySUTO
python scripts/parse/copper_params_parse.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from pysuto.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

CONC_IN  = os.path.join("data", "input", "params", "copper_conc.csv")
EFF_IN   = os.path.join("data", "input", "params", "copper_efficiency.csv")
CONC_OUT = os.path.join("data", "input", "params", "copper_conc_params.csv")
EFF_OUT  = os.path.join("data", "input", "params", "copper_efficiency_params.csv")

# Valid edge directions per sector, derived from rules.ods Supply / Use sheets.
# to_node  : sector appears as output of some process (Supply rules) → use-direction row valid
# from_node: sector appears as input to some process (Use rules)     → supply-direction row valid
#
# Supply:  Mining→{WR,Ore}; MF→{Concentrate,Tailings}; Smelting→{Matte,Slag};
#          CFR→{Slag,Anode}; SXEW→{Tailings,CSXEW}; ER→{Slag,CER}; CP→{Cathode,PSC}
# Use:     CO→Mining; Ore→{MF,SXEW}; Concentrate→Smelting; Matte→CFR;
#          Scrap→{Smelting,CFR}; Anode→ER; CSXEW→CP; CER→CP; Cathode→{FU,USC,MESC}
VALID_DIRECTIONS: dict[str, dict[str, bool]] = {
    "CO":          {"to_node": False, "from_node": True},   # no process outputs CO
    "WR":          {"to_node": True,  "from_node": False},  # WR has no downstream use
    "Ore":         {"to_node": True,  "from_node": True},
    "Concentrate": {"to_node": True,  "from_node": True},
    "Matte":       {"to_node": True,  "from_node": True},
    "Tailings":    {"to_node": True,  "from_node": False},  # Tailings has no downstream use
    "Slag":        {"to_node": True,  "from_node": False},  # Slag has no downstream use
    "Scrap":       {"to_node": False, "from_node": True},   # no process outputs Scrap
    "Anode":       {"to_node": True,  "from_node": True},
    "CER":         {"to_node": True,  "from_node": True},
    "CSXEW":       {"to_node": True,  "from_node": True},
    "Cathode":     {"to_node": True,  "from_node": True},
}

# Root sector → entity (from root.ods Sector_Entity sheet)
SECTOR_ENTITY: dict[str, str] = {
    "CO":          "SB",
    "WR":          "SB",
    "Ore":         "Flow",
    "Concentrate": "Flow",
    "Matte":       "Flow",
    "Tailings":    "SB",
    "Slag":        "SB",
    "Scrap":       "SB",
    "Anode":       "Flow",
    "CER":         "Flow",
    "CSXEW":       "Flow",
    "Cathode":     "Flow",
}

# copper_conc.csv "Flow" column → root sector (handles non-standard names)
CONC_SECTOR_MAP: dict[str, str] = {
    "CO":           "CO",
    "WR":           "WR",
    "Ore":          "Ore",
    "Concentrate":  "Concentrate",
    "Matte":        "Matte",
    "Tailings":     "Tailings",
    "Slag":         "Slag",
    "Scrap_class_1": "Scrap",   # class distinction not in root; mapped to Scrap/SB
    "Scrap_class_2": "Scrap",
    "Anode":        "Anode",
    "CSXEW":        "CSXEW",
    "CER":          "CER",
    "Cathode":      "Cathode",
}

# Process chain for efficiency parameters.
# Each entry: input-flow sector/entity → process → output-flow sector/entity
# All operate on CC valuation (Cu content to Cu content).
EFFICIENCY_CHAIN: dict[str, dict] = {
    "MF (Flotation)": {
        "process_sector": "MF",
        "a_sector_from":  "Ore",        "a_entity_from": "Flow",
        "b_sector_to":    "Concentrate","b_entity_to":   "Flow",
    },
    "Smelting": {
        "process_sector": "Smelting",
        "a_sector_from":  "Concentrate","a_entity_from": "Flow",
        "b_sector_to":    "Matte",      "b_entity_to":   "Flow",
    },
    "CFR": {
        "process_sector": "CFR",
        "a_sector_from":  "Matte",      "a_entity_from": "Flow",
        "b_sector_to":    "Anode",      "b_entity_to":   "Flow",
    },
    "SXEW": {
        "process_sector": "SXEW",
        "a_sector_from":  "Ore",        "a_entity_from": "Flow",
        "b_sector_to":    "CSXEW",      "b_entity_to":   "Flow",
    },
    "ER": {
        "process_sector": "ER",
        "a_sector_from":  "Anode",      "a_entity_from": "Flow",
        "b_sector_to":    "CER",        "b_entity_to":   "Flow",
    },
}

TEMPLATE_COLS = [
    "param_id", "param_type", "description",
    "a_sector_from", "a_entity_from", "a_region_from",
    "a_sector_to",   "a_entity_to",   "a_region_to",
    "a_time",        "a_valuation",
    "b_sector_from", "b_entity_from", "b_region_from",
    "b_sector_to",   "b_entity_to",   "b_region_to",
    "b_time",        "b_valuation",
    "value_nominal", "unc_type",
    "value_sd", "value_min", "value_max",
    "pedigree_reliability", "pedigree_completeness",
    "pedigree_temporal_correlation", "pedigree_geographical_correlation",
    "pedigree_other_correlation",
    "epsilon", "bound_lower", "bound_upper",
    "source", "comment",
]

EMPTY_PEDIGREE = {k: "" for k in [
    "pedigree_reliability", "pedigree_completeness",
    "pedigree_temporal_correlation", "pedigree_geographical_correlation",
    "pedigree_other_correlation",
]}


# ---------------------------------------------------------------------------
# CONCENTRATION CONVERSION
# ---------------------------------------------------------------------------

def convert_conc(path_in: str) -> list[dict]:
    df = pd.read_csv(path_in)
    rows: list[dict] = []

    for _, r in df.iterrows():
        raw_sector = r["Flow"]
        root_sector = CONC_SECTOR_MAP.get(raw_sector)
        if root_sector is None:
            logger.warning("  Unknown flow sector '%s' — skipped", raw_sector)
            continue

        entity   = SECTOR_ENTITY[root_sector]
        vmin     = float(r["Value_min_tCu_per_t"])
        vmax     = float(r["Value_max_tCu_per_t"])
        nominal  = round((vmin + vmax) / 2, 6)
        source   = str(r["Source"])
        comment  = str(r.get("Comment", ""))

        # Suffix for param_id: use original raw_sector to preserve class info
        pid_base = raw_sector.lower().replace(" ", "_").replace(",", "")

        base = {
            "param_type":  "concentration",
            "unc_type":    "min_max",
            "value_nominal": nominal,
            "value_sd":    "", "value_min": vmin, "value_max": vmax,
            **EMPTY_PEDIGREE,
            "epsilon":     "",
            "bound_lower": 0, "bound_upper": 1,
            "b_valuation": "CC",
            "b_time":      "same",
            "b_sector_from": "same", "b_entity_from": "same", "b_region_from": "same",
            "b_sector_to":   "same", "b_entity_to":   "same", "b_region_to":   "same",
            "a_valuation": "Mass",
            "a_time":      "*",
            "a_region_from": "*", "a_region_to": "*",
            "source":      source,
        }

        mapped_comment = (f"Mapped from '{raw_sector}' → root '{root_sector}/{entity}'. "
                          + comment) if raw_sector != root_sector else comment
        valid = VALID_DIRECTIONS.get(root_sector, {"to_node": True, "from_node": True})

        # use-direction: flow node as to-node (only if some process outputs this sector)
        if valid["to_node"]:
            use = base.copy()
            # underscore-free: param_id is the constraint-id variant slot (naming._safe)
            use["param_id"]      = f"conc_{pid_base}_use".replace("_", "")
            use["description"]   = (f"Cu concentration in {raw_sector} ({entity}) "
                                     f"— use edges (to-node={root_sector}/{entity})")
            use["a_sector_from"] = "*"
            use["a_entity_from"] = "*"
            use["a_sector_to"]   = root_sector
            use["a_entity_to"]   = entity
            use["comment"]       = mapped_comment
            rows.append(use)

        # supply-direction: flow node as from-node (only if this sector feeds some process)
        if valid["from_node"]:
            sup = base.copy()
            sup["param_id"]      = f"conc_{pid_base}_supply".replace("_", "")
            sup["description"]   = (f"Cu concentration in {raw_sector} ({entity}) "
                                     f"— supply edges (from-node={root_sector}/{entity})")
            sup["a_sector_from"] = root_sector
            sup["a_entity_from"] = entity
            sup["a_sector_to"]   = "*"
            sup["a_entity_to"]   = "*"
            sup["comment"]       = mapped_comment
            rows.append(sup)

    return rows


# ---------------------------------------------------------------------------
# EFFICIENCY CONVERSION
# ---------------------------------------------------------------------------

def convert_eff(path_in: str) -> list[dict]:
    df = pd.read_csv(path_in)
    rows: list[dict] = []

    for _, r in df.iterrows():
        process_name = r["Process"]
        chain = EFFICIENCY_CHAIN.get(process_name)
        if chain is None:
            logger.warning("  Unknown process '%s' — skipped", process_name)
            continue

        vmin    = float(r["Value_min_tCu_per_tCu"])
        vmax    = float(r["Value_max_tCu_per_tCu"])
        nominal = round((vmin + vmax) / 2, 6)
        source  = str(r["Source"])
        pid     = ("eff_" + process_name.lower().replace(" ", "_")
                   .replace("(", "").replace(")", "")).replace("_", "")

        proc_sector = chain["process_sector"]

        row = {
            "param_id":    pid,
            "param_type":  "efficiency",
            "description": (f"Cu recovery in {process_name}: "
                            f"{chain['a_sector_from']}/{chain['a_entity_from']} → "
                            f"{proc_sector}/Process → "
                            f"{chain['b_sector_to']}/{chain['b_entity_to']}"),
            # Edge A: input USE edge → process
            "a_sector_from":  chain["a_sector_from"],
            "a_entity_from":  chain["a_entity_from"],
            "a_region_from":  "*",
            "a_sector_to":    proc_sector,
            "a_entity_to":    "Process",
            "a_region_to":    "same",   # same-region process node
            "a_time":         "*",
            "a_valuation":    "CC",
            # Edge B: process SUPPLY edge → output
            "b_sector_from":  proc_sector,
            "b_entity_from":  "Process",
            "b_region_from":  "same",
            "b_sector_to":    chain["b_sector_to"],
            "b_entity_to":    chain["b_entity_to"],
            "b_region_to":    "same",
            "b_time":         "same",
            "b_valuation":    "same",   # CC (same as edge A)
            "value_nominal":  nominal,
            "unc_type":       "min_max",
            "value_sd":       "", "value_min": vmin, "value_max": vmax,
            **EMPTY_PEDIGREE,
            "epsilon":        "",
            "bound_lower":    0,
            "bound_upper":    1,
            "source":         source,
            "comment":        "",
        }
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Converting %s", CONC_IN)
    conc_rows = convert_conc(CONC_IN)
    conc_df   = pd.DataFrame(conc_rows, columns=TEMPLATE_COLS)
    conc_df.to_csv(CONC_OUT, sep=";", index=False)
    logger.info("  %d rows → %s", len(conc_df), CONC_OUT)

    logger.info("Converting %s", EFF_IN)
    eff_rows = convert_eff(EFF_IN)
    eff_df   = pd.DataFrame(eff_rows, columns=TEMPLATE_COLS)
    eff_df.to_csv(EFF_OUT, sep=";", index=False)
    logger.info("  %d rows → %s", len(eff_df), EFF_OUT)


if __name__ == "__main__":
    main()
