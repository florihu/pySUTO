"""
acc.py
======
AccountingStructure — defines the admissible node/edge space for a
physical accounting system, plus per-edge lower/upper bounds used by
the linear optimiser.

Root loading
------------
The root classification is read from an ODS/Excel file with four sheets:

    Sector_Entity : columns  Sector_name, Entity_name
    Region        : column   Region_name
    Valuation     : column   Valuation_name
    Time          : column   Time_name

Rule loading
------------
Admissibility rules are read from an ODS/Excel file with four sheets:

    Supply     : rows = from-Process/SB sectors, cols = to-Flow/SB sectors
                 A cell value of 1 means the (from→to) sector pair is a
                 valid *supply* edge (Process→Flow/SB).

    Use        : rows = from-Flow/SB sectors, cols = to-Process sectors
                 A cell value of 1 means the (from→to) sector pair is a
                 valid *use* edge (Flow/SB→Process).

    Trade      : columns Flow, Traded
                 A row with Traded==1 flags that flow as internationally
                 tradable.  Non-listed or NaN flows are non-tradable.

    Accounting : columns Entity_from, Entity_to, Same_reg
                 Same_reg==1  → the edge is only admissible between nodes
                                in the SAME region.
                 Same_reg==NaN→ cross-region edges are also admissible.

Combined admissibility logic
----------------------------
An edge (ni → nj, time t, valuation v) is admissible iff:

1. The (sector_from, sector_to) pair appears in Supply OR Use with value 1.
2. The entity-pair (entity_from, entity_to) is listed in Accounting.
3. If the Accounting row for that entity-pair has Same_reg == 1:
       region_from == region_to   (same-region only)
   Else if the edge is in the Use direction AND the from-sector is a
   tradable flow:
       any region pair is admissible
   Else (Supply direction or non-tradable Use):
       region_from == region_to

In plain English:
  • Supply edges (Process→Flow/SB) are ALWAYS same-region.
  • Use edges where the flow is tradable CAN be cross-region.
  • Use edges where the flow is non-tradable are same-region.
  • SB→Process edges follow the Accounting Same_reg flag.

Bounds design
-------------
Default bounds:  l_i = 0,  u_i = 1e9  (all flows are non-negative)

Exceptions are stored in an ODS/Excel file (sheet "Sheet1" or "bounds").
A rule matches by (sector_from, sector_to, valuation); region and time
are optional filters.  Unspecified filter columns (*) match everything.

Excel/ODS format  columns:
    Sector_from / sector_from : str
    Sector_to   / sector_to   : str
    Valuation   / valuation   : str
    Region_from / region_from : str   (optional)
    Region_to   / region_to   : str   (optional)
    Time        / time        : str   (optional)
    Lower / lower             : float
    Upper / upper             : float
"""

from __future__ import annotations

import inspect
import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import sparse

try:
    from .logging import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)


# =============================================================================
# Zero-dependency ODS reader  (stdlib only: zipfile + xml.etree)
# =============================================================================

def _read_ods(path: str, sheet_name=None) -> pd.DataFrame:
    """
    Read one sheet from an ODS file using only Python stdlib.
    No odfpy / openpyxl required.

    Parameters
    ----------
    path       : path to the .ods file
    sheet_name : sheet name string, or None for the first sheet

    Returns
    -------
    pd.DataFrame with string/None values (header row becomes column names).
    Numeric strings are left as str; callers cast with pd.to_numeric where needed.
    """
    import zipfile
    import xml.etree.ElementTree as ET

    _NS = {
        'table' : 'urn:oasis:names:tc:opendocument:xmlns:table:1.0',
        'text'  : 'urn:oasis:names:tc:opendocument:xmlns:text:1.0',
        'office': 'urn:oasis:names:tc:opendocument:xmlns:office:1.0',
    }
    _T = 'urn:oasis:names:tc:opendocument:xmlns:table:1.0'

    with zipfile.ZipFile(path) as zf:
        root = ET.fromstring(zf.read('content.xml'))

    spreadsheet = root.find('.//office:spreadsheet', _NS)
    tables      = spreadsheet.findall('table:table', _NS)
    sheet_map   = {t.attrib.get(f'{{{_T}}}name'): t for t in tables}

    if sheet_name is None:
        table = tables[0]
    elif sheet_name in sheet_map:
        table = sheet_map[sheet_name]
    else:
        raise KeyError(
            f"Sheet '{sheet_name}' not found in '{path}'. "
            f"Available: {list(sheet_map.keys())}"
        )

    rows_out = []
    for row_el in table.findall('table:table-row', _NS):
        cells = []
        for cell in row_el.findall('table:table-cell', _NS):
            repeat  = int(cell.attrib.get(f'{{{_T}}}number-columns-repeated', 1))
            val_el  = cell.find('text:p', _NS)
            val     = val_el.text if val_el is not None else None
            cells.extend([val] * repeat)
        # strip trailing empty cells
        while cells and cells[-1] is None:
            cells.pop()
        if any(c is not None for c in cells):
            rows_out.append(cells)

    if not rows_out:
        return pd.DataFrame()

    maxlen   = max(len(r) for r in rows_out)
    rows_out = [r + [None] * (maxlen - len(r)) for r in rows_out]
    df       = pd.DataFrame(rows_out[1:], columns=rows_out[0])
    return df


def _ods_sheet_names(path: str) -> list:
    """Return the list of sheet names in an ODS file (stdlib only)."""
    import zipfile
    import xml.etree.ElementTree as ET
    _T = 'urn:oasis:names:tc:opendocument:xmlns:table:1.0'
    with zipfile.ZipFile(path) as zf:
        root = ET.fromstring(zf.read('content.xml'))
    ss     = root.find('.//{urn:oasis:names:tc:opendocument:xmlns:office:1.0}spreadsheet')
    tables = ss.findall(f'{{{_T}}}table')
    return [t.attrib.get(f'{{{_T}}}name') for t in tables]


def _read_bounds_file(path: str) -> pd.DataFrame:
    """Read a bounds spreadsheet from an ODS or XLSX file."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx', '.xls'):
        xl = pd.ExcelFile(path)
        sheet = "bounds" if "bounds" in xl.sheet_names else xl.sheet_names[0]
        df = xl.parse(sheet)
        for col in ("lower", "upper"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    else:
        sheets = _ods_sheet_names(path)
        sheet = "bounds" if "bounds" in sheets else sheets[0]
        return _read_ods(path, sheet)


_INF = np.inf
DEFAULT_UPPER_BOUND: float = 1e9   # practical large value (not +inf) as default upper


# =============================================================================
# RuleSet  — parsed admissibility rules from the rules ODS file
# =============================================================================

class RuleSet:
    """
    Parsed representation of the four rules sheets.

    Attributes
    ----------
    supply_pairs : set of (sector_from, sector_to)
        Valid Process/SB → Flow/SB sector pairs.
    use_pairs : set of (sector_from, sector_to)
        Valid Flow/SB → Process sector pairs.
    tradable_flows : set of str
        Sector names that are internationally tradable.
    accounting : dict mapping (entity_from, entity_to) → same_region (bool)
        Whether the entity-pair requires same-region placement.
    all_pairs : set of (sector_from, sector_to)
        Union of supply_pairs and use_pairs.
    """

    def __init__(self, rules_file_path: str):
        if not os.path.exists(rules_file_path):
            raise FileNotFoundError(f"Rules file not found: '{rules_file_path}'")

        self.supply_pairs   = self._parse_supply(rules_file_path)
        self.use_pairs      = self._parse_use(rules_file_path)
        self.tradable_flows = self._parse_trade(rules_file_path)
        self.accounting     = self._parse_accounting(rules_file_path)
        self.all_pairs      = self.supply_pairs | self.use_pairs

    # ------------------------------------------------------------------

    def _parse_supply(self, path: str) -> set:
        df = _read_ods(path, "Supply")
        row_col = df.columns[0]          # first column = from-sector
        pairs = set()
        for _, row in df.iterrows():
            sector_from = str(row[row_col]).strip()
            for col in df.columns[1:]:
                if pd.notna(row[col]) and str(row[col]).strip() == "1":
                    pairs.add((sector_from, str(col).strip()))
        return pairs

    def _parse_use(self, path: str) -> set:
        df = _read_ods(path, "Use")
        row_col = df.columns[0]          # first column = from-sector (Flow/SB)
        pairs = set()
        for _, row in df.iterrows():
            sector_from = str(row[row_col]).strip()
            for col in df.columns[1:]:
                if pd.notna(row[col]) and str(row[col]).strip() == "1":
                    pairs.add((sector_from, str(col).strip()))
        return pairs

    def _parse_trade(self, path: str) -> set:
        df = _read_ods(path, "Trade")
        df.columns = [c.strip().lower() for c in df.columns]
        # normalise column names to lowercase
        tradable = set()
        for _, row in df.iterrows():
            if pd.notna(row.get("traded")) and str(row.get("traded","")).strip() == "1":
                tradable.add(str(row["flow"]).strip())
        return tradable

    def _parse_accounting(self, path: str) -> dict:
        df = _read_ods(path, "Accounting")
        df.columns = [c.strip() for c in df.columns]
        mapping = {}
        for _, row in df.iterrows():
            ef = str(row["Entity_from"]).strip()
            et = str(row["Entity_to"]).strip()
            same = pd.notna(row.get("Same_reg")) and str(row["Same_reg"]).strip() == "1"
            mapping[(ef, et)] = same
        return mapping

    # ------------------------------------------------------------------

    def is_sector_pair_admissible(self, sector_from: str, sector_to: str) -> bool:
        return (sector_from, sector_to) in self.all_pairs

    def requires_same_region(
        self,
        sector_from:  str,
        sector_to:    str,
        entity_from:  str,
        entity_to:    str,
    ) -> bool:
        """
        Returns True if this edge must be same-region.

        Logic:
          1. If the entity-pair Accounting row explicitly says Same_reg=1 → True.
          2. If this is a Use edge (Flow/SB → Process) AND the from-sector
             is a tradable flow → False (cross-region OK).
          3. Otherwise → True (same-region by default).
        """
        acc_same = self.accounting.get((entity_from, entity_to), True)
        if acc_same:
            return True

        # Accounting allows cross-region: only true for tradable-flow Use edges
        in_use = (sector_from, sector_to) in self.use_pairs
        if in_use and sector_from in self.tradable_flows:
            return False

        return True  # supply edges and non-tradable use edges stay same-region

    def summary(self) -> str:
        return (
            f"RuleSet: {len(self.supply_pairs)} supply pairs, "
            f"{len(self.use_pairs)} use pairs, "
            f"{len(self.tradable_flows)} tradable flows, "
            f"{len(self.accounting)} accounting entity-pair rules"
        )


# =============================================================================
# EdgeBounds
# =============================================================================

class EdgeBounds:
    """
    Per-edge lower and upper bound vectors, aligned to an edge_index.

    Attributes
    ----------
    l : np.ndarray, shape (n_edges,)
        Lower bounds.  Default 0 unless overridden.
    u : np.ndarray, shape (n_edges,)
        Upper bounds.  Default DEFAULT_UPPER_BOUND unless overridden.
    n_exceptions : int
        Number of edges whose bounds differ from the default.
    """

    DEFAULT_LOWER: float = 0.0
    DEFAULT_UPPER: float = DEFAULT_UPPER_BOUND

    def __init__(self,
                 edge_index:  pd.DataFrame,
                 node_index:  pd.DataFrame,
                 bounds_df:   Optional[pd.DataFrame] = None):
        self.n_edges = len(edge_index)
        self.l = np.full(self.n_edges, self.DEFAULT_LOWER, dtype=float)
        self.u = np.full(self.n_edges, self.DEFAULT_UPPER, dtype=float)

        if bounds_df is not None and len(bounds_df) > 0:
            self._apply_rules(edge_index, node_index, bounds_df)

        non_default = (self.l != self.DEFAULT_LOWER) | (self.u != self.DEFAULT_UPPER)
        self.n_exceptions = int(non_default.sum())

    # ------------------------------------------------------------------

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise column names to lowercase snake_case for robustness."""
        df = df.copy()
        df.columns = [c.strip().lower() for c in df.columns]
        return df

    def _apply_rules(self,
                     edge_index: pd.DataFrame,
                     node_index: pd.DataFrame,
                     bounds_df:  pd.DataFrame) -> None:
        bounds_df = self._normalise_columns(bounds_df)

        ni = node_index.loc[edge_index["from_id"].values]
        nj = node_index.loc[edge_index["to_id"].values]

        sf = ni["sector"].values
        rf = ni["region"].values
        st = nj["sector"].values
        rt = nj["region"].values
        tm = edge_index["time"].values.astype(str)
        vl = edge_index["valuation"].values.astype(str)

        required_cols = {"sector_from", "sector_to", "valuation", "lower", "upper"}
        missing = required_cols - set(bounds_df.columns)
        if missing:
            raise ValueError(
                f"Bounds file is missing required columns: {missing}\n"
                f"Required: sector_from, sector_to, valuation, lower, upper\n"
                f"Optional: region_from, region_to, time"
            )

        for _, rule in bounds_df.iterrows():
            mask = np.ones(self.n_edges, dtype=bool)

            def _match(arr, val):
                v = str(val).strip()
                return (np.ones(self.n_edges, dtype=bool)
                        if v in ("*", "", "nan") else arr == v)

            mask &= _match(sf, rule["sector_from"])
            mask &= _match(st, rule["sector_to"])
            mask &= _match(vl, rule["valuation"])

            if "region_from" in bounds_df.columns:
                mask &= _match(rf, rule["region_from"])
            if "region_to" in bounds_df.columns:
                mask &= _match(rt, rule["region_to"])
            if "time" in bounds_df.columns:
                mask &= _match(tm, rule["time"])

            if mask.sum() == 0:
                continue

            if pd.notna(rule["lower"]) and str(rule["lower"]).strip() not in ("", "nan"):
                self.l[mask] = float(rule["lower"])
            if pd.notna(rule["upper"]) and str(rule["upper"]).strip() not in ("", "nan"):
                self.u[mask] = float(rule["upper"])

    # ------------------------------------------------------------------

    def validate(self,
                 F: np.ndarray,
                 edge_index: pd.DataFrame,
                 raise_on_violation: bool = True,
                 ) -> pd.DataFrame:
        lo_viol  = F < self.l
        hi_viol  = F > self.u
        any_viol = lo_viol | hi_viol
        if not any_viol.any():
            return pd.DataFrame()

        viol_idx = np.where(any_viol)[0]
        report   = edge_index.iloc[viol_idx].copy()
        report["lower"]     = self.l[viol_idx]
        report["value"]     = F[viol_idx]
        report["upper"]     = self.u[viol_idx]
        report["violation"] = np.where(
            lo_viol[viol_idx], "below_lower",
            np.where(hi_viol[viol_idx], "above_upper", "")
        )
        if raise_on_violation:
            _cols = ["edge_id", "from_id", "to_id", "time", "valuation",
                     "lower", "value", "upper", "violation"]
            raise ValueError(
                f"{len(viol_idx)} edge(s) violate bounds.\n"
                f"{report[_cols].to_string(index=False)}"
            )
        return report

    def as_dict(self) -> dict:
        return {"l": self.l, "u": self.u}

    def summary(self) -> str:
        neg_lower    = (self.l < 0).sum()
        finite_upper = np.isfinite(self.u).sum()
        return (
            f"EdgeBounds: {self.n_edges} edges | "
            f"{self.n_exceptions} exceptions | "
            f"{neg_lower} negative-lower | "
            f"{finite_upper} finite-upper"
        )


# =============================================================================
# AccountingStructure
# =============================================================================

class AccountingStructure:
    """
    Defines the admissible node/edge space for a physical accounting system
    and holds per-edge lower/upper bounds for the linear optimiser.

    Initialisation modes
    --------------------
    A) From ODS/Excel files (recommended)::

        acc = AccountingStructure(
            id             = "copper",
            root_file_path = "data/root.ods",
            rules_file_path= "data/rules.ods",
            bounds_file_path= "data/bounds.ods",  # optional
            storage_dir    = "data/proc/acc",      # optional
        )

    B) From Python objects (original API, kept for backwards-compat)::

        acc = AccountingStructure(
            id    = "copper",
            root  = {"nodes": <DataFrame>, "time_range": [...], "valuation": [...]},
            rules = [<callable>, ...],
        )

    C) From a previously saved directory::

        acc = AccountingStructure.load("copper", storage_dir="data/proc/acc")

    Parameters
    ----------
    id                : str
        Unique identifier for this structure.
    root_file_path    : str, optional
        Path to the root ODS file (sheets: Sector_Entity, Region, Valuation, Time).
    rules_file_path   : str, optional
        Path to the rules ODS file (sheets: Supply, Use, Trade, Accounting).
    bounds_file_path  : str, optional
        Path to the bounds ODS file.
    root              : dict, optional
        Python dict with keys "nodes" (DataFrame), "time_range", "valuation".
        Used only when root_file_path is not provided.
    rules             : list of callable, optional
        Legacy callable rules.  Used only when rules_file_path is not provided.
    storage_dir       : str
        Directory for saving/loading the structure (default: "data/proc/acc").
    """

    def __init__(self, id, root=None, rules=None, **kwargs):
        self.id          = id or "accounting_structure"
        self.storage_dir = kwargs.get("storage_dir", r"data/proc/acc")
        self.logger      = get_logger(self.__class__.__name__)

        # ── 1. Root (node universe, time, valuation) ─────────────────
        root_file = kwargs.get("root_file_path", None)
        if root_file is not None:
            self._load_root_from_file(root_file)
        elif kwargs.get("init_from_file", False):
            assert "root_file_path" in kwargs
            self.syst_desc_from_file(kwargs["root_file_path"])
        else:
            assert root is not None, \
                "Provide root_file_path= OR root= dict."
            self.root_nodes = root["nodes"].copy()
            self.time_range = list(dict.fromkeys(root["time_range"]))
            self.valuation  = list(dict.fromkeys(root["valuation"]))

        # ── 2. Rules ──────────────────────────────────────────────────
        rules_file = kwargs.get("rules_file_path", None)
        if rules_file is not None:
            self.ruleset = RuleSet(rules_file)
            self.logger.info(self.ruleset.summary())
            self.rules       = []          # legacy list stays empty
            self._node_rules = []
            self._edge_rules = []
        else:
            self.ruleset = None
            self.rules   = rules or []
            self._classify_rules()

        # ── 3. Build indices and tensors ──────────────────────────────
        self._build_node_index()
        self._build_dim_indices()

        # Count theoretical edges BEFORE applying rules
        n_theoretical = (
            self.n_nodes * self.n_nodes
            * len(self.time_range)
            * len(self.valuation)
        )

        self.A_nodes        = self._build_node_mask()
        self.A              = self._build_tensor()
        self.edge_index     = self._build_edge_index()
        self.edge_index_all = self._build_edge_index(admissible_only=False)
        self.n_edges_all    = len(self.edge_index_all)

        # Report edge reduction
        n_after_node_rules = int(self.A_nodes.nnz) * len(self.time_range) * len(self.valuation)
        self._report_reduction(n_theoretical, n_after_node_rules, self.n_edges)

        # ── 4. Bounds ─────────────────────────────────────────────────
        bounds_path  = kwargs.get("bounds_file_path", None)
        self.bounds  = self._load_bounds(bounds_path)
        self.logger.info(self.bounds.summary())

        self.logger.info(
            f"AccountingStructure '{self.id}' initialised: "
            f"{self.n_nodes} nodes, {self.n_edges} admissible edges, "
            f"A shape={self.A.shape}, nnz={self.A.nnz}"
        )

    # ------------------------------------------------------------------
    # Root from ODS  (NEW)
    # ------------------------------------------------------------------

    def _load_root_from_file(self, path: str) -> None:
        """
        Read root definition from the four-sheet ODS file.

        Sheets
        ------
        Sector_Entity : Sector_name, Entity_name  (one row per sector)
        Region        : Region_name
        Valuation     : Valuation_name
        Time          : Time_name
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Root file not found: '{path}'")
        self.logger.info(f"Loading root from: {path}")

        se  = _read_ods(path, "Sector_Entity")
        reg = _read_ods(path, "Region")
        val = _read_ods(path, "Valuation")
        tim = _read_ods(path, "Time")

        # Build all (sector, entity, region) triples
        records = []
        for _, row in se.iterrows():
            sector = str(row["Sector_name"]).strip()
            entity = str(row["Entity_name"]).strip()
            for _, rrow in reg.iterrows():
                region = str(rrow["Region_name"]).strip()
                records.append((sector, entity, region))

        self.root_nodes = pd.DataFrame(records, columns=["sector", "entity", "region"])
        self.time_range = [str(t) for t in tim["Time_name"].tolist()]
        self.valuation  = [str(v) for v in val["Valuation_name"].tolist()]

        self.logger.info(
            f"  Root: {len(se)} sectors × {len(reg)} regions = "
            f"{len(self.root_nodes)} node definitions | "
            f"{len(self.time_range)} time points | "
            f"{len(self.valuation)} valuations"
        )

    # ------------------------------------------------------------------
    # Edge-reduction report  (NEW)
    # ------------------------------------------------------------------

    def _report_reduction(
        self,
        n_theoretical:     int,
        n_after_node_rules: int,
        n_admissible:       int,
    ) -> None:
        def pct(a, b):
            return f"{100*a/b:.1f}%" if b else "N/A"

        lines = [
            "",
            "┌─────────────────────────────────────────────────────────┐",
            "│              Edge-space reduction report                │",
            "├─────────────────────────────────────────────────────────┤",
            f"│  Theoretical (all node-pairs × T × V) : {n_theoretical:>12,d}  │",
            f"│  After node-pair rules                : {n_after_node_rules:>12,d}  │"
            f"  ({pct(n_after_node_rules, n_theoretical)} of theoretical)",
            f"│  After all rules (admissible)         : {n_admissible:>12,d}  │"
            f"  ({pct(n_admissible, n_theoretical)} of theoretical)",
            f"│  Reduction from node-pair to final    : "
            f"{n_after_node_rules - n_admissible:>12,d}  │"
            f"  ({pct(n_after_node_rules - n_admissible, n_after_node_rules)} additional removal)",
            "└─────────────────────────────────────────────────────────┘",
        ]
        report = "\n".join(lines)
        self.logger.info(report)
        print(report)   # also print to stdout for immediate visibility

    # ------------------------------------------------------------------
    # Bounds loading  (extended to handle ODS + column-name variants)
    # ------------------------------------------------------------------

    def _load_bounds(self, path: Optional[str]) -> EdgeBounds:
        bounds_df = None
        if path is not None:
            if not os.path.exists(path):
                self.logger.warning(
                    f"Bounds file not found: '{path}'. "
                    f"Using default bounds (l=0, u={DEFAULT_UPPER_BOUND:.0e})."
                )
            else:
                self.logger.info(f"Loading edge bounds from: {path}")
                # Try sheet name "bounds" first, fall back to first sheet
                try:
                    bounds_df = _read_bounds_file(path)
                    self.logger.info(
                        f"  Loaded {len(bounds_df)} bound rule(s)."
                    )
                except Exception as e:
                    self.logger.warning(f"Could not read bounds file: {e}")
                    bounds_df = None

        return EdgeBounds(
            edge_index=self.edge_index,
            node_index=self.node_index,
            bounds_df=bounds_df,
        )

    def reload_bounds(self, path: str) -> None:
        """Hot-reload bounds from a new file without rebuilding the structure."""
        self.logger.info(f"Reloading edge bounds from: {path}")
        bounds_df = _read_bounds_file(path)
        self.bounds = EdgeBounds(
            edge_index=self.edge_index,
            node_index=self.node_index,
            bounds_df=bounds_df,
        )
        self.logger.info(self.bounds.summary())

    # ------------------------------------------------------------------
    # Convenience properties for ConstraintBuilder
    # ------------------------------------------------------------------

    @property
    def l(self) -> np.ndarray:
        return self.bounds.l

    @property
    def u(self) -> np.ndarray:
        return self.bounds.u

    # ------------------------------------------------------------------
    # Admissibility  —  RuleSet path
    # ------------------------------------------------------------------

    def _node_pair_admissible_ruleset(self, ni_dict: dict, nj_dict: dict) -> bool:
        """Check admissibility using the parsed RuleSet (ODS rules)."""
        sf = ni_dict["sector"]
        st = nj_dict["sector"]

        # Must appear in supply or use sector-pair table
        if not self.ruleset.is_sector_pair_admissible(sf, st):
            return False

        # Must appear in accounting entity-pair table
        ef = ni_dict["entity"]
        et = nj_dict["entity"]
        if (ef, et) not in self.ruleset.accounting:
            return False

        return True

    def _edge_admissible_ruleset(
        self,
        ni_dict: dict,
        nj_dict: dict,
        time,
        valuation,
    ) -> bool:
        """Full edge admissibility using RuleSet (includes region logic)."""
        if not self._node_pair_admissible_ruleset(ni_dict, nj_dict):
            return False

        sf = ni_dict["sector"]
        st = nj_dict["sector"]
        ef = ni_dict["entity"]
        et = nj_dict["entity"]
        rf = ni_dict["region"]
        rt = nj_dict["region"]

        same_req = self.ruleset.requires_same_region(sf, st, ef, et)
        if same_req and rf != rt:
            return False

        return True

    # ------------------------------------------------------------------
    # Admissibility  —  legacy callable path  (unchanged)
    # ------------------------------------------------------------------

    def _classify_rules(self):
        self._node_rules = []
        self._edge_rules = []
        for rule in self.rules:
            sig    = inspect.signature(rule)
            n_args = sum(
                1 for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            )
            (self._edge_rules if n_args > 2 else self._node_rules).append(rule)

    def _node_pair_admissible(self, ni_dict: dict, nj_dict: dict) -> bool:
        if self.ruleset is not None:
            return self._node_pair_admissible_ruleset(ni_dict, nj_dict)
        for rule in self._node_rules:
            if not rule(ni_dict, nj_dict):
                return False
        return True

    def _edge_admissible(self, ni_dict, nj_dict, time, valuation) -> bool:
        if self.ruleset is not None:
            return self._edge_admissible_ruleset(ni_dict, nj_dict, time, valuation)
        if not self._node_pair_admissible(ni_dict, nj_dict):
            return False
        for rule in self._edge_rules:
            if not rule(ni_dict, nj_dict, time, valuation):
                return False
        return True

    # ------------------------------------------------------------------
    # Legacy Excel initialisation  (unchanged)
    # ------------------------------------------------------------------

    def syst_desc_from_file(self, path):
        self.logger.info(f"Initialising structure from file: {path}")
        file = pd.read_excel(path)
        assert {"sector_entity", "region", "time_range", "valuation"}.issubset(
            file.columns
        ), "Excel file must contain: sector_entity, region, time_range, valuation"

        sector_entity = file["sector_entity"].str.split("_", expand=True)
        sector_entity.columns = ["sector", "entity"]
        region = file["region"].unique()
        self.root_nodes = pd.DataFrame(
            [(s, e, r)
             for s, e in zip(sector_entity["sector"], sector_entity["entity"])
             for r in region],
            columns=["sector", "entity", "region"],
        )
        self.time_range = list(dict.fromkeys(file["time_range"].tolist()))
        self.valuation  = list(dict.fromkeys(file["valuation"].tolist()))

    # ------------------------------------------------------------------
    # Node index  (unchanged)
    # ------------------------------------------------------------------

    def _build_node_index(self):
        self.node_to_id = {}
        self.id_to_node = {}
        node_id = 0
        for _, node in self.root_nodes.iterrows():
            key = (node["sector"], node["entity"], node["region"])
            if key not in self.node_to_id:
                self.node_to_id[key] = node_id
                self.id_to_node[node_id] = key
                node_id += 1

        self.n_nodes  = node_id
        self.n_timeps = len(self.time_range)
        self.n_valtns = len(self.valuation)
        self.node_index = pd.DataFrame.from_dict(
            self.id_to_node, orient="index", columns=["sector", "entity", "region"]
        )
        self.logger.info(f"Total nodes: {self.n_nodes}")

    def _build_dim_indices(self):
        self.time_to_idx = {t: i for i, t in enumerate(self.time_range)}
        self.val_to_idx  = {v: i for i, v in enumerate(self.valuation)}

    # ------------------------------------------------------------------
    # Node-only admissibility mask  (unchanged)
    # ------------------------------------------------------------------

    def _build_node_mask(self) -> sparse.COO:
        self.logger.info("Building 2-D node admissibility mask...")
        c0, c1 = [], []
        keys = list(self.id_to_node.values())

        for i, ni in enumerate(keys):
            ni_dict = dict(zip(["sector", "entity", "region"], ni))
            for j, nj in enumerate(keys):
                nj_dict = dict(zip(["sector", "entity", "region"], nj))
                if self._node_pair_admissible(ni_dict, nj_dict):
                    c0.append(i); c1.append(j)

        data = np.ones(len(c0), dtype=np.int8)
        return sparse.COO(
            coords=np.array([c0, c1], dtype=np.int64),
            data=data,
            shape=(self.n_nodes, self.n_nodes),
        )

    # ------------------------------------------------------------------
    # Full 4-D admissibility tensor  (unchanged)
    # ------------------------------------------------------------------

    def _build_tensor(self) -> sparse.COO:
        self.logger.info("Building 4-D admissibility tensor...")
        c0, c1, c2, c3 = [], [], [], []
        keys = list(self.id_to_node.values())

        for i, ni in enumerate(keys):
            ni_dict = dict(zip(["sector", "entity", "region"], ni))
            for j, nj in enumerate(keys):
                nj_dict = dict(zip(["sector", "entity", "region"], nj))

                if not self._node_pair_admissible(ni_dict, nj_dict):
                    continue

                for t_idx, t in enumerate(self.time_range):
                    for v_idx, v in enumerate(self.valuation):
                        if not self._edge_admissible(ni_dict, nj_dict, t, v):
                            continue
                        c0.append(i);     c1.append(j)
                        c2.append(t_idx); c3.append(v_idx)

        data = np.ones(len(c0), dtype=np.int8)
        return sparse.COO(
            coords=np.array([c0, c1, c2, c3], dtype=np.int64),
            data=data,
            shape=(self.n_nodes, self.n_nodes, self.n_timeps, self.n_valtns),
        )

    # ------------------------------------------------------------------
    # Edge index DataFrame  (unchanged)
    # ------------------------------------------------------------------

    def _build_edge_index(self, admissible_only=True) -> pd.DataFrame:
        rows, cols, times, vals, edge_ids = [], [], [], [], []
        edge_id = 0
        keys    = list(self.id_to_node.values())

        for i, ni in enumerate(keys):
            ni_dict = dict(zip(["sector", "entity", "region"], ni))
            for j, nj in enumerate(keys):
                nj_dict = dict(zip(["sector", "entity", "region"], nj))

                if admissible_only and not self._node_pair_admissible(ni_dict, nj_dict):
                    continue

                for t in self.time_range:
                    for v in self.valuation:
                        if admissible_only and not self._edge_admissible(
                            ni_dict, nj_dict, t, v
                        ):
                            continue
                        rows.append(i);     cols.append(j)
                        times.append(t);    vals.append(v)
                        edge_ids.append(edge_id)
                        edge_id += 1

        if admissible_only:
            self.n_edges = len(edge_ids)
        return pd.DataFrame({
            "edge_id":   edge_ids,
            "from_id":   rows,
            "to_id":     cols,
            "time":      times,
            "valuation": vals,
        })

    # ------------------------------------------------------------------
    # compress  (unchanged)
    # ------------------------------------------------------------------

    def compress(self, F) -> np.ndarray:
        if isinstance(F, sparse.SparseArray):
            F = F.todense()
        ei     = self.edge_index
        t_idxs = ei["time"].map(self.time_to_idx).values.astype(int)
        v_idxs = ei["valuation"].map(self.val_to_idx).values.astype(int)
        return F[
            ei["from_id"].values.astype(int),
            ei["to_id"].values.astype(int),
            t_idxs,
            v_idxs,
        ]

    # ------------------------------------------------------------------
    # Lookup helpers  (unchanged)
    # ------------------------------------------------------------------

    def decode_node(self, node_id: int):
        return self.id_to_node[node_id]

    def encode_node(self, sector, entity, region) -> int:
        return self.node_to_id[(sector, entity, region)]

    def decode_edge(self, edge_id: int):
        row = self.edge_index.iloc[edge_id]
        return (
            self.decode_node(int(row["from_id"])),
            self.decode_node(int(row["to_id"])),
            row["time"],
            row["valuation"],
        )

    # ------------------------------------------------------------------
    # Save / Load  (extended to persist bounds)
    # ------------------------------------------------------------------

    def save(self, **kwargs):
        path = os.path.join(self.storage_dir, self.id)
        os.makedirs(path, exist_ok=True)

        sparse.save_npz(os.path.join(path, "A.npz"),       self.A)
        sparse.save_npz(os.path.join(path, "A_nodes.npz"), self.A_nodes)

        self.edge_index.to_csv(    os.path.join(path, "edge_index.csv"),     index=False)
        self.edge_index_all.to_csv(os.path.join(path, "edge_index_all.csv"), index=False)
        self.node_index.to_csv(    os.path.join(path, "node_index.csv"),     index=False)
        self.root_nodes.to_csv(    os.path.join(path, "root_nodes.csv"),     index=False)

        np.save(os.path.join(path, "bounds_l.npy"), self.bounds.l)
        np.save(os.path.join(path, "bounds_u.npy"), self.bounds.u)

        meta = {
            "time_range": self.time_range,
            "valuation":  self.valuation,
            "node_to_id": self.node_to_id,
            "id_to_node": self.id_to_node,
        }
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(meta, f)
        self.logger.info(f"Structure saved to {path}")

    @classmethod
    def load(cls, id, **kwargs):
        obj             = cls.__new__(cls)
        obj.id          = id
        obj.storage_dir = kwargs.get("storage_dir", r"data/proc/acc")
        path            = os.path.join(obj.storage_dir, id)
        assert os.path.isdir(path), \
            f"Directory '{id}' not found in {obj.storage_dir}"

        obj.A        = sparse.load_npz(os.path.join(path, "A.npz"))
        obj.A_nodes  = sparse.load_npz(os.path.join(path, "A_nodes.npz"))

        obj.edge_index     = pd.read_csv(os.path.join(path, "edge_index.csv"))
        obj.edge_index_all = pd.read_csv(os.path.join(path, "edge_index_all.csv"))
        obj.root_nodes     = pd.read_csv(os.path.join(path, "root_nodes.csv"))
        obj.node_index     = pd.read_csv(
            os.path.join(path, "node_index.csv")
        )

        with open(os.path.join(path, "metadata.pkl"), "rb") as f:
            meta = pickle.load(f)

        obj.time_range  = meta["time_range"]
        obj.valuation   = meta["valuation"]
        obj.node_to_id  = meta["node_to_id"]
        obj.id_to_node  = meta["id_to_node"]
        obj.n_nodes     = len(obj.id_to_node)
        obj.n_timeps    = len(obj.time_range)
        obj.n_valtns    = len(obj.valuation)
        obj.n_edges     = len(obj.edge_index)
        obj.n_edges_all = len(obj.edge_index_all)
        obj.time_to_idx = {t: i for i, t in enumerate(obj.time_range)}
        obj.val_to_idx  = {v: i for i, v in enumerate(obj.valuation)}
        obj.rules       = []
        obj._node_rules = []
        obj._edge_rules = []
        obj.ruleset     = None
        obj.logger      = get_logger(cls.__name__)

        l_path = os.path.join(path, "bounds_l.npy")
        u_path = os.path.join(path, "bounds_u.npy")
        bounds = EdgeBounds.__new__(EdgeBounds)
        bounds.n_edges = obj.n_edges
        if os.path.exists(l_path) and os.path.exists(u_path):
            bounds.l = np.load(l_path)
            bounds.u = np.load(u_path)
        else:
            bounds.l = np.zeros(obj.n_edges,  dtype=float)
            bounds.u = np.full(obj.n_edges, DEFAULT_UPPER_BOUND, dtype=float)
        non_default = (
            (bounds.l != EdgeBounds.DEFAULT_LOWER) |
            (bounds.u != EdgeBounds.DEFAULT_UPPER)
        )
        bounds.n_exceptions = int(non_default.sum())
        obj.bounds = bounds
        return obj
