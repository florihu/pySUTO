"""
acc.py
======
AccountingStructure — defines the admissible node/edge space for a
physical accounting system, plus per-edge lower/upper bounds used by
the linear optimiser.

Acc spec
--------
An accounting structure is declared by one dict, the **acc spec** (ADR 0046)::

    {"sectors":    [{"name": "Mining", "entity": "Process"},
                    {"name": "Scrap",  "entity": "SB", "sub": "SEM",       # sub optional
                     "role": "secondary"}, ...],                          # SB only
     "regions":    ["Chile", ...],                  # nodes = sectors × regions
     "valuations": [{"name": "Mass", "role": "gross"},
                    {"name": "CC",   "role": "content"}],
     "times":      ["1995", ...],                   # every label a str
     "supply":     {"Mining": ["Ore", "WR"], ...},  # adjacency lists
     "use":        {"Ore": ["MF", "SXEW"], ...},
     "tradable":   ["Concentrate", ...],
     "accounting": [{"from": "Flow", "to": "Process", "same_region": False}, ...],
     "bounds":     [...],    # optional: records with the bounds file's column names
     "masks":      [...]}    # optional: records with the masks.csv column names

``AccountingStructure.from_spec`` builds from it. Two readers produce it:
``read_acc_spec_json`` from one ``acc.json``, ``read_acc_spec_ods`` from the four files
described below, which ``AccountingStructure(root_file_path=..., rules_file_path=...)``
reads the same way; ``read_acc_spec_dir`` picks the reader from what a directory holds.

Roles
-----
The labels are the case's; the **roles** are the framework's (ADR 0034, GLOSSARY "Role").
Every ``SB`` sector binds its label to a :class:`BoundaryRole` (``nature``, ``secondary``,
``final_demand``) and every valuation to a :class:`ValuationRole` (``content``,
``gross``). The library reads roles through :attr:`AccountingStructure.boundary_roles` (SB sector
→ boundary role) and :attr:`AccountingStructure.valuation_roles` (``.content``, ``.gross``),
never the labels. The entity kinds ``Process | Flow | SB`` are reserved framework words,
not roles; a ``secondary`` node is an input on a use edge and a residual output on a
supply edge, so edge direction tells Scrap from Slag. Build and load raise unless
sector → entity is 1:1, every SB sector and no other carries a known boundary role, and
there is exactly one ``content`` and one ``gross`` valuation.

Root loading
------------
The root classification is read from an ODS/Excel file with four sheets:

    Sector_Entity : columns  Sector_name, Entity_name, role (the SB rows)
    Region        : column   Region_name
    Valuation     : columns  Valuation_name, role
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

Named masks
-----------
After ``__init__`` returns, ``self.masks`` is a ``dict[str, np.ndarray]`` of
boolean masks of shape ``(n_edges,)``. Built-in entries: ``supply``, ``use``,
``tradable``, ``sb``. User-defined masks can be added via an optional
``masks.csv`` co-located with ``root.ods`` (see :mod:`pysuto.masks`), via
``apply_mask_overlay(path)``, or programmatically via
``register_mask(name, array_or_callable)``.
"""

from __future__ import annotations

import inspect
import json
import os
import pickle
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .input_registry import ACC_ODS_FILES, ACC_SPEC, InputRegistry

try:
    from .logging import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

from pysuto.masks import read_masks_csv


# =============================================================================
# Roles  — the framework's meta-model; a case binds its labels to them (ADR 0034)
# =============================================================================

class BoundaryRole(str, Enum):
    """The framework role of an ``SB`` sector (GLOSSARY "Role")."""
    NATURE       = "nature"
    SECONDARY    = "secondary"
    FINAL_DEMAND = "final_demand"


class ValuationRole(str, Enum):
    """The framework role of a valuation: contained metal or gross product mass."""
    CONTENT = "content"
    GROSS   = "gross"


#: The entity kinds whose nodes carry a balance; ``SB`` nodes are open endpoints.
BALANCE_ENTITIES = frozenset({"Flow", "Process"})


@dataclass(frozen=True)
class ValuationRoles:
    """The valuation labels a case binds to the two valuation roles."""
    content: str
    gross:   str


def boundary_role_values(roles: dict, sectors) -> np.ndarray:
    """The boundary role of each sector label as its value string (``"nature"``,
    ``"secondary"``, ``"final_demand"``), ``None`` where the sector carries none.

    Compare the result with ``BoundaryRole.X.value``: numpy does not compare an object
    array elementwise with a ``str``-Enum scalar.
    """
    by_value = {s: BoundaryRole(r).value for s, r in roles.items()}
    return np.array([by_value.get(s) for s in np.asarray(sectors, dtype=object)],
                    dtype=object)


def _bind_roles(sectors: list, valuations: list) -> tuple:
    """Validate the role bindings and return ``(roles, valuations)``.

    Parameters
    ----------
    sectors : list of dict
        Records with ``name``, ``entity`` and, on an ``SB`` sector, ``role``.
    valuations : list of dict
        Records with ``name`` and ``role``.

    Returns
    -------
    tuple
        ``{SB sector label: BoundaryRole}`` and a :class:`ValuationRoles`.

    Raises
    ------
    ValueError
        A sector declared twice (sector → entity is 1:1), an ``SB`` sector without a
        boundary role, a role on a ``Process`` or ``Flow`` sector, an unknown role name,
        a valuation without a role, or not exactly one ``content`` and one ``gross``
        valuation.
    """
    entities: dict = {}
    for s in sectors:
        entities.setdefault(s["name"], []).append(s["entity"])
    twice = {name: ents for name, ents in entities.items() if len(ents) > 1}
    if twice:
        raise ValueError(f"acc spec: sector → entity must be 1:1; declared more than "
                         f"once: {twice}")

    roles: dict = {}
    for s in sectors:
        role = s.get("role")
        if s["entity"] != "SB":
            if role is not None:
                raise ValueError(f"acc spec: sector {s['name']!r} is a {s['entity']} "
                                 f"sector; only an SB sector carries a role, got {role!r}")
            continue
        if role is None:
            raise ValueError(f"acc spec: SB sector {s['name']!r} has no boundary role; "
                             f"expected one of {[r.value for r in BoundaryRole]}")
        try:
            roles[s["name"]] = BoundaryRole(role)
        except ValueError:
            raise ValueError(f"acc spec: SB sector {s['name']!r} has unknown boundary "
                             f"role {role!r}; expected one of "
                             f"{[r.value for r in BoundaryRole]}") from None

    by_role: dict = {r: [] for r in ValuationRole}
    for v in valuations:
        role = v.get("role")
        try:
            by_role[ValuationRole(role)].append(v["name"])
        except ValueError:
            what = "no role" if role is None else f"unknown role {role!r}"
            raise ValueError(f"acc spec: valuation {v['name']!r} has {what}; expected "
                             f"one of {[r.value for r in ValuationRole]}") from None
    wrong = {r.value: names for r, names in by_role.items() if len(names) != 1}
    if wrong:
        raise ValueError(f"acc spec: exactly one content and one gross valuation are "
                         f"required; got {wrong}")
    return roles, ValuationRoles(content=by_role[ValuationRole.CONTENT][0],
                             gross=by_role[ValuationRole.GROSS][0])


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


def _read_bounds_file(path: str) -> list:
    """Read a bounds spreadsheet (ODS or XLSX) as records, one dict per rule."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx', '.xls'):
        xl = pd.ExcelFile(path)
        sheet = "bounds" if "bounds" in xl.sheet_names else xl.sheet_names[0]
        df = xl.parse(sheet)
        for col in ("lower", "upper"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        sheets = _ods_sheet_names(path)
        sheet = "bounds" if "bounds" in sheets else sheets[0]
        df = _read_ods(path, sheet)
    return df.to_dict("records")


_INF = np.inf
DEFAULT_UPPER_BOUND: float = 1e9   # practical large value (not +inf) as default upper
DEFAULT_STORAGE_DIR: str   = r"data/proc/acc"


# =============================================================================
# Acc spec  — one declaration; ODS/CSV and JSON are its two formats (ADR 0046)
# =============================================================================

_SPEC_REQUIRED: tuple = ("sectors", "regions", "valuations", "times",
                         "supply", "use", "tradable", "accounting")
_SPEC_OPTIONAL: tuple = ("bounds", "masks")

# The keys of one entry. A sector's "role" is optional here; _bind_roles requires it on
# an SB sector and refuses it elsewhere (#119).
_SECTOR_KEYS:     tuple = ("name", "entity", "sub", "role")
_SECTOR_OPTIONAL: tuple = ("sub", "role")
_VALUATION_KEYS:  tuple = ("name", "role")
_ACCOUNTING_KEYS: tuple = ("from", "to", "same_region")


def _read_root_ods(path: str) -> dict:
    """The root ODS file as the spec's sectors, regions, valuations and times.

    Sheets: Sector_Entity (Sector_name, Entity_name, optional Entity_name_1 → ``sub``,
    ``role`` on the SB rows), Region (Region_name), Valuation (Valuation_name, ``role``),
    Time (Time_name). An empty ``role`` cell declares no role.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Root file not found: '{path}'")
    se  = _read_ods(path, "Sector_Entity")
    reg = _read_ods(path, "Region")
    val = _read_ods(path, "Valuation")
    tim = _read_ods(path, "Time")

    sectors = []
    for _, row in se.iterrows():
        sector = {"name":   str(row["Sector_name"]).strip(),
                  "entity": str(row["Entity_name"]).strip()}
        if "Entity_name_1" in se.columns and pd.notna(row["Entity_name_1"]):
            sector["sub"] = str(row["Entity_name_1"]).strip()
        if "role" in se.columns and pd.notna(row["role"]):
            sector["role"] = str(row["role"]).strip()
        sectors.append(sector)

    valuations = []
    for _, row in val.iterrows():
        valuation = {"name": str(row["Valuation_name"])}
        if "role" in val.columns and pd.notna(row["role"]):
            valuation["role"] = str(row["role"]).strip()
        valuations.append(valuation)

    return {
        "sectors":    sectors,
        "regions":    [str(r).strip() for r in reg["Region_name"]],
        "valuations": valuations,
        "times":      [str(t) for t in tim["Time_name"]],
    }


def _read_matrix(path: str, sheet: str) -> dict:
    """A 0/1 rules sheet as adjacency lists: row sector → column sectors marked "1"."""
    df = _read_ods(path, sheet)
    row_col = df.columns[0]              # first column = from-sector
    adjacency: dict = {}
    for _, row in df.iterrows():
        sector_from = str(row[row_col]).strip()
        for col in df.columns[1:]:
            if pd.notna(row[col]) and str(row[col]).strip() == "1":
                adjacency.setdefault(sector_from, []).append(str(col).strip())
    return adjacency


def _read_rules_ods(path: str) -> dict:
    """The rules ODS file as the spec's supply, use, tradable and accounting entries.

    Sheets: Supply and Use (0/1 matrices, first column = from-sector), Trade (Flow,
    Traded), Accounting (Entity_from, Entity_to, Same_reg).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Rules file not found: '{path}'")

    def _is_one(value) -> bool:
        return bool(pd.notna(value) and str(value).strip() == "1")

    trade = _read_ods(path, "Trade")
    trade.columns = [c.strip().lower() for c in trade.columns]
    tradable = [str(row["flow"]).strip() for _, row in trade.iterrows()
                if _is_one(row.get("traded"))]

    acc = _read_ods(path, "Accounting")
    acc.columns = [c.strip() for c in acc.columns]
    accounting = [
        {"from":        str(row["Entity_from"]).strip(),
         "to":          str(row["Entity_to"]).strip(),
         "same_region": _is_one(row.get("Same_reg"))}
        for _, row in acc.iterrows()
    ]
    return {
        "supply":     _read_matrix(path, "Supply"),
        "use":        _read_matrix(path, "Use"),
        "tradable":   tradable,
        "accounting": accounting,
    }


def read_acc_spec_ods(root_path:   str,
                      rules_path:  str,
                      bounds_path: Optional[str] = None,
                      masks_path:  Optional[str] = None) -> dict:
    """Read an acc spec from the ODS/CSV files of the copper case.

    Parameters
    ----------
    root_path, rules_path : str
        ``root.ods`` and ``rules.ods`` (sheets in the module docstring).
    bounds_path : str, optional
        ``bounds.ods`` (or ``.xlsx``); its rows become the spec's ``bounds`` records.
    masks_path : str, optional
        ``masks.csv``; its rows become the spec's ``masks`` records.

    Returns
    -------
    dict
        The acc spec; ``bounds`` / ``masks`` only when their file is given.
    """
    spec = {**_read_root_ods(root_path), **_read_rules_ods(rules_path)}
    if bounds_path is not None:
        spec["bounds"] = _read_bounds_file(bounds_path)
    if masks_path is not None:
        spec["masks"] = read_masks_csv(masks_path)
    return spec


def read_acc_spec_json(path: str) -> dict:
    """Read an acc spec from one ``acc.json``: the document is the spec dict itself."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_acc_spec_dir(path) -> dict:
    """Read the acc spec one directory declares (an input registry's ``current/`` or
    ``vN/``): from ``acc.json`` when it holds one, else from ``root.ods`` + ``rules.ods``
    with ``bounds.ods`` and ``masks.csv`` where present.

    Raises
    ------
    ValueError
        *path* holds ``acc.json`` and any file of the ODS/CSV format: one of the two
        would be ignored (ADR 0046).
    """
    path = Path(path)
    if (path / ACC_SPEC).is_file():
        ods_beside_json = [n for n in ACC_ODS_FILES if (path / n).is_file()]
        if ods_beside_json:
            raise ValueError(
                f"{path} holds {ACC_SPEC} and {ods_beside_json}: an acc spec is "
                f"read from {ACC_SPEC} or from root.ods + rules.ods, never both "
                f"(ADR 0046)"
            )
        return read_acc_spec_json(str(path / ACC_SPEC))
    optional = {name: str(path / name) if (path / name).is_file() else None
                for name in ("bounds.ods", "masks.csv")}
    return read_acc_spec_ods(str(path / "root.ods"), str(path / "rules.ods"),
                             optional["bounds.ods"], optional["masks.csv"])


def _check_keys(entry, keys: tuple, optional: tuple, where: str) -> None:
    """``KeyError`` naming a missing key of *entry*; ``ValueError`` on an unknown key."""
    if not isinstance(entry, dict):
        raise TypeError(f"acc spec: {where} must be a dict; got {entry!r}")
    for key in keys:
        if key not in optional and key not in entry:
            raise KeyError(f"acc spec: {where} is missing required key {key!r}")
    unknown = sorted(set(entry) - set(keys), key=str)
    if unknown:
        raise ValueError(f"acc spec: {where} has unknown key(s) {unknown}; "
                         f"expected {list(keys)}")


def _validate_spec(spec: dict) -> None:
    """Raise on a missing or unknown key, a value of the wrong type, or a rule naming a
    sector or entity the spec does not declare. Nothing is coerced; the roles are
    validated when the root is set (:func:`_bind_roles`)."""
    _check_keys(spec, _SPEC_REQUIRED + _SPEC_OPTIONAL, _SPEC_OPTIONAL, "the top level")
    for i, sector in enumerate(spec["sectors"]):
        _check_keys(sector, _SECTOR_KEYS, _SECTOR_OPTIONAL, f"sectors[{i}]")
    for i, valuation in enumerate(spec["valuations"]):
        _check_keys(valuation, _VALUATION_KEYS, (), f"valuations[{i}]")
    for i, rule in enumerate(spec["accounting"]):
        _check_keys(rule, _ACCOUNTING_KEYS, (), f"accounting[{i}]")
        if not isinstance(rule["same_region"], bool):
            raise TypeError(f"acc spec: accounting[{i}] same_region must be a bool; "
                            f"got {rule['same_region']!r}")

    labels = ([s[k] for s in spec["sectors"] for k in _SECTOR_KEYS if k in s]
              + list(spec["regions"]) + [v["name"] for v in spec["valuations"]]
              + list(spec["times"]))
    not_str = [x for x in labels if not isinstance(x, str)]
    if not_str:
        raise TypeError(f"acc spec labels must be strings; got {not_str[:5]!r}")

    sectors = {s["name"] for s in spec["sectors"]}
    named = {key: set(spec[key]) | {t for tos in spec[key].values() for t in tos}
             for key in ("supply", "use")}
    named["tradable"] = set(spec["tradable"])
    for key, names in named.items():
        undeclared = sorted(names - sectors, key=str)
        if undeclared:
            raise ValueError(f"acc spec: {key} names sector(s) {undeclared}, which no "
                             f"sectors entry declares")

    entities = {s["entity"] for s in spec["sectors"]}
    for rule in spec["accounting"]:
        for end in ("from", "to"):
            if rule[end] not in entities:
                raise ValueError(
                    f"acc spec: accounting pair {rule['from']!r} → {rule['to']!r} names "
                    f"entity {rule[end]!r}, which no sector declares; "
                    f"declared: {sorted(entities)}"
                )


def _rule_tables(spec: dict) -> tuple:
    """The spec's rule entries as the four tables :meth:`RuleSet.from_tables` takes."""
    supply_pairs = {(f, t) for f, tos in spec["supply"].items() for t in tos}
    use_pairs    = {(f, t) for f, tos in spec["use"].items()    for t in tos}
    accounting   = {(r["from"], r["to"]): r["same_region"] for r in spec["accounting"]}
    return supply_pairs, use_pairs, set(spec["tradable"]), accounting


# =============================================================================
# Acc compatibility  — two versions compared on the keys a consumer reads (#119)
# =============================================================================

#: The keys that define the edge universe — what Stage 2 checks against its feed runs
#: (a feed reads no role).
EDGE_UNIVERSE_KEYS: tuple = _SPEC_REQUIRED
#: The edge universe and the roles — what Stages 4 (warm start) and 6 check against the
#: built acc (the PSUT blocks are read by role).
EDGE_UNIVERSE_ROLE_KEYS: tuple = _SPEC_REQUIRED + ("role",)
#: Every key — what Stages 2 and 3 check against the built acc (Stage 2 clips to the
#: bounds and reads the recipes by role; bounds and masks enter G).
ALL_SPEC_KEYS: tuple = _SPEC_REQUIRED + _SPEC_OPTIONAL + ("role",)

# The copper case's labels, bound for one legacy path only: a version that declares no
# roles (acc/copper v4 and older) is compared through them (#119).
_LEGACY_BOUNDARY_ROLES  = {"Nature": "nature", "SEM": "secondary", "FD": "final_demand"}
_LEGACY_VALUATION_ROLES = {"Mass": "gross", "CC": "content"}


def _declared_roles(spec: dict) -> list:
    """``(kind, label, role)`` of every SB sector and valuation of a spec, sorted. A
    label with no role takes the copper binding; any other raises, a role is never
    guessed."""
    out = []
    for s in spec["sectors"]:
        if s["entity"] == "SB":
            role = s.get("role") or _LEGACY_BOUNDARY_ROLES.get(s.get("sub"))
            if role is None:
                raise ValueError(f"SB sector {s['name']!r} declares no role and "
                                 f"{s.get('sub')!r} is not a copper boundary label")
            out.append(("sector", s["name"], role))
    for v in spec["valuations"]:
        role = v.get("role") or _LEGACY_VALUATION_ROLES.get(v["name"])
        if role is None:
            raise ValueError(f"valuation {v['name']!r} declares no role and is not a "
                             f"copper valuation")
        out.append(("valuation", v["name"], role))
    return sorted(out)


def _canonical(spec: dict, key: str):
    """One key of a spec in the form two specs are compared on. Order is kept where it
    numbers nodes or edges (sectors, regions, valuations, times) or decides precedence
    (bound and mask records); adjacency is a set. ``role`` is the bound roles of the SB
    sectors and valuations (the other keys carry labels only). An absent optional key is
    an empty list."""
    if key == "role":
        return _declared_roles(spec)
    value = spec.get(key)
    if key == "sectors":
        return [(s["name"], s["entity"], s.get("sub")) for s in value]
    if key == "valuations":
        return [v["name"] for v in value]
    if key in ("supply", "use"):
        return sorted((f, t) for f, tos in value.items() for t in tos)
    if key == "tradable":
        return sorted(value)
    if key == "accounting":
        return sorted((r["from"], r["to"], bool(r["same_region"])) for r in value)
    if key in ("bounds", "masks"):
        return [{k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else v)
                 for k, v in record.items()} for record in (value or [])]
    return list(value)


def acc_spec_differences(a: dict, b: dict, keys) -> list:
    """The keys among *keys* on which acc specs *a* and *b* differ (see
    :func:`_canonical`).

    Raises
    ------
    ValueError
        ``"role"`` is in *keys* and a spec declares no role for a label that is not one
        of the copper case's.
    """
    return [key for key in keys if _canonical(a, key) != _canonical(b, key)]


def require_compatible_acc(acc_registry, acc_id: str, version_a: int, version_b: int,
                           keys, *, context: str = "", remedy: str = "") -> None:
    """Raise unless versions *version_a* and *version_b* of acc *acc_id* agree on *keys*;
    log at info level every other key they differ in.

    Both specs are read from the input registry's snapshots (``acc_registry.resolve``,
    then :func:`read_acc_spec_dir`), so the format of a version (ODS files or
    ``acc.json``) does not count. ``"role"`` is a key like the others: where a consumer
    reads roles it is compared, and a version that declares none (v4 and older) is
    compared through the copper binding.

    Parameters
    ----------
    acc_registry
        The ``acc`` :class:`~pysuto.input_registry.InputRegistry`.
    keys
        The keys the consumer reads: :data:`EDGE_UNIVERSE_KEYS` (Stage 2 against its
        feed runs), :data:`EDGE_UNIVERSE_ROLE_KEYS` (Stages 4 and 6 against the built
        acc) or :data:`ALL_SPEC_KEYS` (Stages 2 and 3 against the built acc).
    context, remedy
        The start and the end of the message: what was compared, what to do.

    Raises
    ------
    ValueError
        The two specs differ in a key of *keys*; the message names those keys.
    """
    if version_a == version_b:
        return
    a = read_acc_spec_dir(acc_registry.resolve(acc_id, version_a))
    b = read_acc_spec_dir(acc_registry.resolve(acc_id, version_b))
    context = context or f"acc {acc_id} v{version_a} and v{version_b}"
    breaking = acc_spec_differences(a, b, [k for k in ALL_SPEC_KEYS if k in keys])
    if breaking:
        raise ValueError(f"{context}; they differ in {breaking}"
                         + (f": {remedy}" if remedy else ""))
    ignored = []
    for key in (k for k in ALL_SPEC_KEYS if k not in keys):
        try:
            if _canonical(a, key) != _canonical(b, key):
                ignored.append(key)
        except ValueError:                  # undeclared roles: not compared, so no raise
            ignored.append(key)
    if ignored:
        get_logger("AccountingStructure").info(
            "%s; they differ in %s, which this check does not compare", context, ignored)


def check_feed_accs(feed_sidecars: dict, acc_registry: InputRegistry,
                    acc_entry: dict) -> None:
    """Check that every feed run was built on an acc compatible with Stage 2's acc
    input entry (#53, #119).

    Parameters
    ----------
    feed_sidecars : ``{feed_run_id: sidecar}`` of the selected feed runs.
    acc_registry : the acc registry, whose ``vN/`` snapshots hold the specs compared.
    acc_entry : Stage 2's own acc input entry ``{id, version, ...}``.

    A feed built on another version of the acc passes when the two versions' specs
    agree on the keys that define the edge universe (:data:`EDGE_UNIVERSE_KEYS`:
    sectors, regions, valuations, times, supply, use, tradable, accounting). A feed
    reads no bound, mask or role, so a difference there is logged at info level, and
    the format of a version (ODS files or acc.json) does not count.

    Raises
    ------
    KeyError
        A feed sidecar records no ``parents.inputs.acc``.
    ValueError
        A feed was built on another acc id, or on a version whose spec differs in an
        edge-universe key; the message names the feed run, both versions and the keys.
    """
    own = (acc_entry["id"], acc_entry["version"])
    for run_id, sidecar in feed_sidecars.items():
        ref = sidecar.get("parents", {}).get("inputs", {}).get("acc")
        if not ref:
            raise KeyError(f"feed run {run_id} records no parents.inputs.acc, so its acc "
                           f"cannot be checked against {own[0]} v{own[1]}")
        theirs = (ref["id"], ref["version"])
        if theirs == own:
            continue
        if theirs[0] != own[0]:
            raise ValueError(f"feed run {run_id} was built on acc {theirs[0]} "
                             f"v{theirs[1]}, this run uses acc {own[0]} v{own[1]}")
        require_compatible_acc(
            acc_registry, own[0], theirs[1], own[1], EDGE_UNIVERSE_KEYS,
            context=(f"feed run {run_id} was built on acc {own[0]} v{theirs[1]}, "
                     f"this run uses v{own[1]}"),
            remedy="rebuild the feed")


# =============================================================================
# RuleSet  — parsed admissibility rules from the rules ODS file
# =============================================================================

class RuleSet:
    """
    The four admissibility tables. ``RuleSet(path)`` reads them from the rules ODS file;
    :meth:`from_tables` takes them in memory (the acc-spec path).

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
        self._set_tables(*_rule_tables(_read_rules_ods(rules_file_path)))

    @classmethod
    def from_tables(cls,
                    supply_pairs: set,
                    use_pairs:    set,
                    tradable:     set,
                    accounting:   dict) -> "RuleSet":
        """Build from the four tables: ``(sector_from, sector_to)`` supply and use pairs,
        the tradable flow sectors, and ``(entity_from, entity_to) → same_region``."""
        obj = cls.__new__(cls)
        obj._set_tables(supply_pairs, use_pairs, tradable, accounting)
        return obj

    def _set_tables(self, supply_pairs, use_pairs, tradable, accounting) -> None:
        self.supply_pairs   = set(supply_pairs)
        self.use_pairs      = set(use_pairs)
        self.tradable_flows = set(tradable)
        self.accounting     = dict(accounting)
        self.all_pairs      = self.supply_pairs | self.use_pairs

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


class _SparseCOO4D:
    """Minimal 4-D sparse COO tensor (scipy.sparse is 2-D only)."""

    def __init__(self, data: np.ndarray, coords: np.ndarray, shape: tuple):
        self.data   = data
        self.coords = coords  # shape (4, nnz)
        self.shape  = shape

    @property
    def nnz(self) -> int:
        return len(self.data)

    def todense(self) -> np.ndarray:
        arr = np.zeros(self.shape, dtype=self.data.dtype)
        arr[tuple(self.coords)] = self.data
        return arr

    def save_npz(self, path: str) -> None:
        np.savez(path, data=self.data, coords=self.coords,
                 shape=np.array(self.shape, dtype=np.int64))

    @classmethod
    def load_npz(cls, path: str) -> "_SparseCOO4D":
        npz = np.load(path)
        return cls(npz["data"], npz["coords"], tuple(npz["shape"].tolist()))


# =============================================================================

class AccountingStructure:
    """
    Defines the admissible node/edge space for a physical accounting system
    and holds per-edge lower/upper bounds for the linear optimiser.

    Initialisation modes
    --------------------
    A) From an acc spec (see the module docstring)::

        acc = AccountingStructure.from_spec(spec, "toy", storage_dir="data/proc/acc")

    A') From ODS/Excel files — read into an acc spec, then built as A::

        acc = AccountingStructure(
            id             = "copper",
            root_file_path = "data/root.ods",
            rules_file_path= "data/rules.ods",
            bounds_file_path= "data/bounds.ods",  # optional
            storage_dir    = "data/proc/acc",      # optional
        )

    B) From Python objects (original API, kept for the index-machinery tests)::

        acc = AccountingStructure(
            id    = "copper",
            root  = {"nodes": <DataFrame>, "time_range": [...], "valuation": [...]},
            rules = [<callable>, ...],
        )

       It takes no acc spec, so it binds **no roles**: :attr:`roles` and
       :attr:`valuations` are ``None``, and a root with an ``SB`` node raises.

    C) From a previously saved directory::

        acc = AccountingStructure.load("copper", storage_dir="data/proc/acc")

    Attributes
    ----------
    roles : dict or None
        ``{SB sector label: BoundaryRole}``; ``None`` on mode B.
    valuations : ValuationRoles or None
        The labels bound to the ``content`` and ``gross`` roles; ``None`` on mode B.
    valuation : list of str
        The valuation axis, in edge order.

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
        self._set_identity(id, kwargs.get("storage_dir", DEFAULT_STORAGE_DIR))

        root_file   = kwargs.get("root_file_path",   None)
        rules_file  = kwargs.get("rules_file_path",  None)
        bounds_path = kwargs.get("bounds_file_path", None)
        if bounds_path is not None and not os.path.exists(bounds_path):
            self.logger.warning(
                f"Bounds file not found: '{bounds_path}'. "
                f"Using default bounds (l=0, u={DEFAULT_UPPER_BOUND:.0e})."
            )
            bounds_path = None
        # Optional acc-local user masks (data/input/acc/<id>/current/masks.csv)
        masks_path = kwargs.get("masks_file_path", None)
        if masks_path is None and root_file is not None:
            masks_path = os.path.join(os.path.dirname(root_file), "masks.csv")
        if masks_path is not None and not os.path.exists(masks_path):
            masks_path = None

        # ── A'. ODS files → acc spec → the from_spec build ────────────
        if root_file is not None and rules_file is not None:
            self.logger.info(f"Reading acc spec from: {root_file}, {rules_file}")
            self._init_from_spec(
                read_acc_spec_ods(root_file, rules_file, bounds_path, masks_path))
            return

        # ── 1. Root (node universe, time, valuation) ─────────────────
        if root_file is not None:
            self._load_root_from_file(root_file)
        else:
            if kwargs.get("init_from_file", False):
                assert "root_file_path" in kwargs
                self.syst_desc_from_file(kwargs["root_file_path"])
            else:
                assert root is not None, \
                    "Provide root_file_path= OR root= dict."
                self.root_nodes = root["nodes"].copy()
                self.time_range = list(dict.fromkeys(root["time_range"]))
                self.valuation  = list(dict.fromkeys(root["valuation"]))
            self._set_no_roles()

        # ── 2. Rules ──────────────────────────────────────────────────
        if rules_file is not None:
            self._set_ruleset(RuleSet(rules_file))
        else:
            self.ruleset = None
            self.rules   = rules or []
            self._classify_rules()

        self._build(
            bounds = None if bounds_path is None else _read_bounds_file(bounds_path),
            masks  = None if masks_path  is None else read_masks_csv(masks_path),
        )

    # ------------------------------------------------------------------
    # Acc spec  (ADR 0046)
    # ------------------------------------------------------------------

    @classmethod
    def from_spec(cls,
                  spec:        dict,
                  id:          str,
                  *,
                  storage_dir: str = DEFAULT_STORAGE_DIR) -> "AccountingStructure":
        """Build the structure an acc spec declares (keys in the module docstring).

        Parameters
        ----------
        spec : dict
            The acc spec, e.g. from :func:`read_acc_spec_json` or
            :func:`read_acc_spec_ods`.
        id : str
            Unique identifier for this structure.
        storage_dir : str
            Directory for saving/loading the structure (default ``"data/proc/acc"``).

        Raises
        ------
        KeyError
            A required key is missing, at the top level or in an entry (the message
            names it).
        ValueError
            An unknown key, a rule naming a sector or entity no sector entry declares,
            or a role binding that breaks ADR 0034 (see :func:`_bind_roles`).
        TypeError
            A label that is not a string, an entry that is not a dict, or a
            ``same_region`` that is not a bool.
        """
        obj = cls.__new__(cls)
        obj._set_identity(id, storage_dir)
        obj._init_from_spec(spec)
        return obj

    def _set_identity(self, id, storage_dir: str) -> None:
        self.id          = id or "accounting_structure"
        self.storage_dir = storage_dir
        self.logger      = get_logger(self.__class__.__name__)

    def _init_from_spec(self, spec: dict) -> None:
        _validate_spec(spec)
        self._set_root(spec)
        self._set_ruleset(RuleSet.from_tables(*_rule_tables(spec)))
        self._build(bounds=spec.get("bounds"), masks=spec.get("masks"))

    def _set_ruleset(self, ruleset: "RuleSet") -> None:
        self.ruleset = ruleset
        self.logger.info(self.ruleset.summary())
        self.rules       = []          # legacy list stays empty
        self._node_rules = []
        self._edge_rules = []

    def _build(self, bounds: Optional[list], masks: Optional[list]) -> None:
        """Steps 3–6, shared by every initialisation mode: indices and tensors, the
        bounds and user-mask records of the spec (``None`` = none declared), the masks."""
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
        self._set_bounds(bounds)

        # ── 5. Structural edge masks (for ConstraintEngine) ───────────
        self._build_edge_masks()

        # ── 6. Optional acc-local user masks ──────────────────────────
        self._set_user_masks(masks)

        self.logger.info(
            f"AccountingStructure '{self.id}' initialised: "
            f"{self.n_nodes} nodes, {self.n_edges} admissible edges, "
            f"A shape={self.A.shape}, nnz={len(self.A.data)}"
        )

    # ------------------------------------------------------------------
    # Root from ODS  (NEW)
    # ------------------------------------------------------------------

    def _load_root_from_file(self, path: str) -> None:
        """Read the root from the four-sheet ODS file (see :func:`read_acc_spec_ods`)."""
        self.logger.info(f"Loading root from: {path}")
        self._set_root(_read_root_ods(path))

    def _set_root(self, spec: dict) -> None:
        """Node universe = the spec's ``sectors`` × ``regions``; time and valuation axes;
        the role bindings (validated, see :func:`_bind_roles`)."""
        sectors, regions = spec["sectors"], spec["regions"]
        self.boundary_roles, self.valuation_roles = _bind_roles(sectors, spec["valuations"])
        self.root_nodes = pd.DataFrame(
            [(s["name"], s["entity"], region, s.get("sub"))
             for s in sectors for region in regions],
            columns=["sector", "entity", "region", "entity_sub"],
        )
        self.time_range = list(spec["times"])
        self.valuation  = [v["name"] for v in spec["valuations"]]

        self.logger.info(
            f"  Root: {len(sectors)} sectors × {len(regions)} regions = "
            f"{len(self.root_nodes)} node definitions | "
            f"{len(self.time_range)} time points | "
            f"{len(self.valuation)} valuations"
        )

    def _set_no_roles(self) -> None:
        """Mode B binds no roles, so it cannot declare an ``SB`` node (every SB sector
        carries a boundary role, ADR 0034)."""
        sb = sorted(set(self.root_nodes.loc[self.root_nodes["entity"] == "SB", "sector"]))
        if sb:
            raise ValueError(
                f"AccountingStructure {self.id!r}: the procedural constructor binds no "
                f"roles, and SB sector(s) {sb} need a boundary role — declare the "
                f"structure as an acc spec (AccountingStructure.from_spec)"
            )
        self.boundary_roles      = None
        self.valuation_roles = None

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
    # Bounds  (records with the bounds file's column names)
    # ------------------------------------------------------------------

    def _set_bounds(self, records: Optional[list]) -> None:
        if records:
            self.logger.info(f"  Loaded {len(records)} bound rule(s).")
        self.bounds = EdgeBounds(
            edge_index=self.edge_index,
            node_index=self.node_index,
            bounds_df=pd.DataFrame(records) if records else None,
        )
        self.logger.info(self.bounds.summary())

    def reload_bounds(self, path: str) -> None:
        """Hot-reload bounds from a new file without rebuilding the structure."""
        self.logger.info(f"Reloading edge bounds from: {path}")
        self._set_bounds(_read_bounds_file(path))

    # ------------------------------------------------------------------
    # Structural edge masks  (used by ConstraintEngine)
    # ------------------------------------------------------------------

    def _build_edge_masks(self) -> None:
        """
        Populate ``self.masks`` with the built-in structural masks.

        All masks have shape ``(n_edges,)`` and are aligned to ``self.edge_index``.
        Built-in names:

          supply       — (sector_from, sector_to) in ruleset.supply_pairs
          use          — (sector_from, sector_to) in ruleset.use_pairs
          tradable     — sector_from in ruleset.tradable_flows
          sb           — entity_to == "SB" (system-boundary endpoint)
          cross_region — region_from != region_to (trade edges; the edge scope
                         for bilateral feeds, which must never inform domestic
                         edges — mapped self-trade exclusion, ADR 0014)

        When ruleset is None (legacy callable-rules mode), supply/use/tradable
        masks fall back to checking entity_from/entity_to directly:
          supply: entity_from in {Process} → entity_to in {Flow, SB}
          use:    entity_from in {Flow, SB} → entity_to in {Process}

        User-defined masks are layered on top by ``_set_user_masks`` (acc-local)
        and ``apply_mask_overlay`` (separate mask input kind).
        """
        ei = self.edge_index
        ni = self.node_index.loc[ei["from_id"].values]
        nj = self.node_index.loc[ei["to_id"].values]
        sf = ni["sector"].values
        ef = ni["entity"].values
        st = nj["sector"].values
        et = nj["entity"].values

        if self.ruleset is not None:
            sp_set  = self.ruleset.supply_pairs
            up_set  = self.ruleset.use_pairs
            tr_set  = self.ruleset.tradable_flows
            supply   = np.array([(s, d) in sp_set for s, d in zip(sf, st)], dtype=bool)
            use      = np.array([(s, d) in up_set for s, d in zip(sf, st)], dtype=bool)
            tradable = np.array([s in tr_set for s in sf], dtype=bool)
        else:
            supply   = np.isin(ef, ["Process"]) & np.isin(et, ["Flow", "SB"])
            use      = np.isin(ef, ["Flow", "SB"]) & np.isin(et, ["Process"])
            tradable = np.zeros(len(ei), dtype=bool)

        sb = (et == "SB")
        cross_region = (ni["region"].values != nj["region"].values)

        self.masks: dict[str, np.ndarray] = {
            "supply":       supply,
            "use":          use,
            "tradable":     tradable,
            "sb":           sb,
            "cross_region": cross_region,
        }

    # ------------------------------------------------------------------
    # User-mask API
    # ------------------------------------------------------------------

    def register_mask(self, name: str, mask) -> None:
        """Add or overwrite a named mask.

        ``mask`` may be a boolean ``np.ndarray`` of shape ``(n_edges,)`` or a
        callable receiving the expanded edge index (with ``sector_from``,
        ``region_from``, etc. columns) and returning a boolean array.
        """
        from pysuto.masks import coerce_mask

        arr = coerce_mask(name, mask, self.edge_index, self.node_index, self.n_edges)
        if name in self.masks:
            self.logger.warning("register_mask: overwriting existing mask %r", name)
        self.masks[name] = arr
        self.logger.info("register_mask: %r → %d / %d edges",
                         name, int(arr.sum()), self.n_edges)

    def apply_mask_overlay(self, csv_path: str) -> None:
        """Load a masks CSV and merge it into ``self.masks``.

        Existing entries with the same name are overwritten (separate-kind
        overlay beats acc-local definitions). Used to apply a mask from the
        separate ``mask`` input kind on top of acc-local masks.
        """
        from pysuto.masks import MaskRuleSet

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Mask overlay file not found: {csv_path!r}")
        resolved = MaskRuleSet.from_csv(csv_path).resolve(
            self.edge_index, self.node_index
        )
        for name, arr in resolved.items():
            if name in self.masks:
                self.logger.info("apply_mask_overlay: overriding mask %r from %s",
                                 name, csv_path)
            self.masks[name] = arr

    def _set_user_masks(self, records: Optional[list]) -> None:
        """Resolve the acc-local user-mask records (``masks.csv`` rows, or the spec's
        ``masks``) and merge them into ``self.masks``.

        Called once at the end of the build; ``None`` or no records is a no-op. The
        names are recorded in ``user_mask_names`` so :meth:`save` can persist them —
        :meth:`load` never sees the records again.
        """
        self.user_mask_names: list[str] = []
        if not records:
            return
        from pysuto.masks import MaskRuleSet

        self.logger.info("Loading %d user-mask rule(s)", len(records))
        resolved = MaskRuleSet.from_records(records).resolve(
            self.edge_index, self.node_index
        )
        for name, arr in resolved.items():
            if name in self.masks:
                self.logger.warning("masks.csv: overriding built-in mask %r", name)
            self.masks[name] = arr
            self.user_mask_names.append(name)

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
        entity_sub_by_id = {}
        has_sub = "entity_sub" in self.root_nodes.columns
        node_id = 0
        for _, node in self.root_nodes.iterrows():
            key = (node["sector"], node["entity"], node["region"])
            if key not in self.node_to_id:
                self.node_to_id[key] = node_id
                self.id_to_node[node_id] = key
                if has_sub:
                    entity_sub_by_id[node_id] = node["entity_sub"]
                node_id += 1

        self.n_nodes  = node_id
        self.n_timeps = len(self.time_range)
        self.n_valtns = len(self.valuation)
        self.node_index = pd.DataFrame.from_dict(
            self.id_to_node, orient="index", columns=["sector", "entity", "region"]
        )
        if has_sub:
            self.node_index["entity_sub"] = pd.Series(entity_sub_by_id)
        else:
            self.node_index["entity_sub"] = None
        self.logger.info(f"Total nodes: {self.n_nodes}")

    def _build_dim_indices(self):
        self.time_to_idx = {t: i for i, t in enumerate(self.time_range)}
        self.val_to_idx  = {v: i for i, v in enumerate(self.valuation)}

    # ------------------------------------------------------------------
    # Node-only admissibility mask  (unchanged)
    # ------------------------------------------------------------------

    def _build_node_mask(self) -> sp.csr_matrix:
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
        return sp.csr_matrix(
            (data, (np.array(c0, dtype=np.int64), np.array(c1, dtype=np.int64))),
            shape=(self.n_nodes, self.n_nodes),
        )

    # ------------------------------------------------------------------
    # Full 4-D admissibility tensor  (unchanged)
    # ------------------------------------------------------------------

    def _build_tensor(self) -> "_SparseCOO4D":
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
        shape = (self.n_nodes, self.n_nodes, self.n_timeps, self.n_valtns)
        return _SparseCOO4D(data, np.array([c0, c1, c2, c3], dtype=np.int64), shape)

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
        if sp.issparse(F):
            F = F.toarray()
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

    @cached_property
    def edge_meta(self) -> pd.DataFrame:
        """``edge_index`` joined with ``node_index`` attributes on both endpoints.

        Columns: ``edge_id, from_id, to_id, time, valuation`` plus
        ``from_<attr>`` and ``to_<attr>`` for every column in ``node_index``
        (``sector, entity, region`` and ``entity_sub`` if present).
        Computed once per instance.
        """
        ei = self.edge_index.reset_index(drop=True)
        ni_from = self.node_index.loc[ei["from_id"].values].reset_index(drop=True)
        ni_to   = self.node_index.loc[ei["to_id"].values].reset_index(drop=True)
        ni_from = ni_from.add_prefix("from_")
        ni_to   = ni_to.add_prefix("to_")
        return pd.concat([ei, ni_from, ni_to], axis=1)

    @cached_property
    def edge_template_code(self) -> np.ndarray:
        """Time-invariant template id per global edge — length ``E``, aligned to
        ``edge_index`` order.

        Two edges that differ **only** in ``time`` share one code: the template
        key is ``(from_id, to_id, valuation)`` (the node ids are time-independent
        — see :meth:`_build_edge_index`). This is the single home for the
        cross-year edge identity used by the solver's warm-start carry-forward
        (``a_prior(t) = a_post(t-1)``) and by the solver diagnostics.
        """
        ei = self.edge_index
        codes, _ = pd.factorize(
            pd.Series(zip(ei["from_id"].to_numpy(),
                          ei["to_id"].to_numpy(),
                          ei["valuation"].to_numpy()))
        )
        return np.asarray(codes, dtype=np.int64)

    def warm_start_gather(self, cols_prev: np.ndarray,
                          cols_cur: np.ndarray) -> np.ndarray:
        """Gather index aligning a previous period's values onto the current
        period's edges **by template**.

        Returns ``g`` (length ``len(cols_cur)``) such that
        ``value_cur[i] := value_prev[g[i]]`` carries the same
        ``(from_id, to_id, valuation)`` template forward. ``g[i] == -1`` marks a
        current-period template with no counterpart in ``cols_prev`` (a new edge),
        for which the caller must fall back to the cold prior. When the two
        periods share an identical template order (the copper case), ``g`` is the
        identity ``arange`` — the carry-forward is positional.
        """
        code = self.edge_template_code
        prev_pos = {int(c): i for i, c in enumerate(code[np.asarray(cols_prev)])}
        return np.fromiter(
            (prev_pos.get(int(c), -1) for c in code[np.asarray(cols_cur)]),
            dtype=np.int64, count=len(cols_cur),
        )

    @cached_property
    def process_recipes(self):
        """Structural process-recipe skeletons grouped per Process node.

        One recipe per ``(process_sector, region, time)`` with at least one
        process-incident content supply edge: its ``inputs`` (use edges into the
        process, content valuation, any region), ``product`` (the single supply
        edge to a ``Flow``), and ``wastes`` (supply edges to ``SB`` nodes of
        boundary role ``secondary``). Derived from ``edge_meta`` + the
        ``supply``/``use`` masks + :attr:`roles`, all of which :meth:`load`
        restores. The recipe carries no parameters;
        :class:`~pysuto.impute.network.NetworkImputer` attaches
        efficiency/concentration from the params CSVs.
        """
        from pysuto.impute.network import derive_process_recipes
        return derive_process_recipes(self)

    # ------------------------------------------------------------------
    # Save / Load  (extended to persist bounds)
    # ------------------------------------------------------------------

    def save(self, **kwargs):
        path = os.path.join(self.storage_dir, self.id)
        os.makedirs(path, exist_ok=True)

        self.A.save_npz(os.path.join(path, "A.npz"))
        sp.save_npz(os.path.join(path, "A_nodes.npz"), self.A_nodes)

        self.edge_index.to_csv(    os.path.join(path, "edge_index.csv"),     index=False)
        self.edge_index_all.to_csv(os.path.join(path, "edge_index_all.csv"), index=False)
        self.node_index.to_csv(    os.path.join(path, "node_index.csv"),     index=False)
        self.root_nodes.to_csv(    os.path.join(path, "root_nodes.csv"),     index=False)

        np.save(os.path.join(path, "bounds_l.npy"), self.bounds.l)
        np.save(os.path.join(path, "bounds_u.npy"), self.bounds.u)

        # acc-local user masks (masks.csv) — load() rebuilds only the structural
        # masks, so without this file a user mask silently vanishes after load.
        masks_path = os.path.join(path, "masks_user.npz")
        if self.user_mask_names:
            np.savez(masks_path, **{n: self.masks[n] for n in self.user_mask_names})
        elif os.path.exists(masks_path):
            os.remove(masks_path)  # a stale file would resurrect removed masks

        # Roles and rule tables, so load() returns what __init__ built: the rule-derived
        # masks (supply, use, tradable) are rebuilt from the same tables (#119). Mode B
        # records None for each (no roles; callable rules are not persistable).
        rs   = self.ruleset
        meta = {
            "time_range": self.time_range,
            "valuation":  self.valuation,
            "node_to_id": self.node_to_id,
            "id_to_node": self.id_to_node,
            "boundary_roles": (None if self.boundary_roles is None
                           else {s: r.value for s, r in self.boundary_roles.items()}),
            "valuation_roles": (None if self.valuation_roles is None else
                                {self.valuation_roles.content: ValuationRole.CONTENT.value,
                                 self.valuation_roles.gross:   ValuationRole.GROSS.value}),
            "rule_tables": (None if rs is None else
                            {"supply_pairs": rs.supply_pairs, "use_pairs": rs.use_pairs,
                             "tradable": rs.tradable_flows, "accounting": rs.accounting}),
        }
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(meta, f)
        self.logger.info(f"Structure saved to {path}")

    @classmethod
    def load(cls, id, **kwargs):
        obj             = cls.__new__(cls)
        obj.id          = id
        obj.storage_dir = kwargs.get("storage_dir", DEFAULT_STORAGE_DIR)
        path            = os.path.join(obj.storage_dir, id)
        assert os.path.isdir(path), \
            f"Directory '{id}' not found in {obj.storage_dir}"

        obj.A        = _SparseCOO4D.load_npz(os.path.join(path, "A.npz"))
        obj.A_nodes  = sp.load_npz(os.path.join(path, "A_nodes.npz"))

        obj.edge_index     = pd.read_csv(os.path.join(path, "edge_index.csv"))
        obj.edge_index_all = pd.read_csv(os.path.join(path, "edge_index_all.csv"))
        # Ensure time column is string — pandas infers year integers from CSV
        obj.edge_index["time"]     = obj.edge_index["time"].astype(str)
        obj.edge_index_all["time"] = obj.edge_index_all["time"].astype(str)
        obj.root_nodes     = pd.read_csv(os.path.join(path, "root_nodes.csv"))
        obj.node_index     = pd.read_csv(os.path.join(path, "node_index.csv"))
        if "entity_sub" not in obj.node_index.columns:
            obj.node_index["entity_sub"] = None

        with open(os.path.join(path, "metadata.pkl"), "rb") as f:
            meta = pickle.load(f)
        missing = [k for k in ("boundary_roles", "valuation_roles", "rule_tables") if k not in meta]
        if missing:
            raise ValueError(
                f"acc {id!r} at {path} records no {missing}: it was saved before roles "
                f"and rule tables were persisted (#119) — rebuild it with "
                f"scripts/acc.py --id {id}"
            )

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
        rt = meta["rule_tables"]
        obj.ruleset     = (None if rt is None else RuleSet.from_tables(
            rt["supply_pairs"], rt["use_pairs"], rt["tradable"], rt["accounting"]))
        obj.logger      = get_logger(cls.__name__)
        obj.logger.info("Loading AccountingStructure '%s' from %s", id, path)

        # The same validation as the build: the persisted roles bound to the nodes.
        if meta["boundary_roles"] is None:
            obj._set_no_roles()
        else:
            sectors = obj.node_index.drop_duplicates(["sector", "entity"])
            obj.boundary_roles, obj.valuation_roles = _bind_roles(
                [{"name": s, "entity": e, "role": meta["boundary_roles"].get(s)}
                 for s, e in zip(sectors["sector"], sectors["entity"])],
                [{"name": v, "role": (meta["valuation_roles"] or {}).get(v)}
                 for v in obj.valuation],
            )

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

        obj.logger.info(
            "   nodes=%d  edges=%d  time=%s–%s  valuations=%s  bound_exceptions=%d",
            obj.n_nodes, obj.n_edges,
            obj.time_range[0], obj.time_range[-1],
            obj.valuation, bounds.n_exceptions,
        )
        obj._build_edge_masks()

        obj.user_mask_names = []
        masks_path = os.path.join(path, "masks_user.npz")
        if os.path.exists(masks_path):
            with np.load(masks_path) as z:
                for name in z.files:
                    arr = z[name]
                    if arr.shape != (obj.n_edges,):
                        raise ValueError(
                            f"masks_user.npz: mask {name!r} has shape {arr.shape}, "
                            f"expected ({obj.n_edges},) — rebuild the structure "
                            f"with scripts/acc.py"
                        )
                    obj.masks[name] = arr.astype(bool)
                    obj.user_mask_names.append(name)
            obj.logger.info("   user masks restored: %s", ", ".join(obj.user_mask_names))
        return obj


# =============================================================================
# Build record — which acc input version a built structure was made from (#53)
# =============================================================================

# Written by scripts/acc.py beside the saved structure, in <storage_dir>/<id>/.
BUILD_RECORD = "build_record.json"


def write_build_record(storage_dir: str, entry: dict) -> str:
    """Record ``{id, version, code}`` of the acc input entry the structure saved
    under ``<storage_dir>/<id>/`` was built from; return the record's path."""
    path = os.path.join(storage_dir, entry["id"], BUILD_RECORD)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: entry[k] for k in ("id", "version", "code")}, f, indent=2)
    return path


def check_built_acc(storage_dir: str, acc_ref: dict, acc_registry, keys,
                    require_record: bool = False) -> None:
    """Check the build record of the structure under ``<storage_dir>/<id>/`` against
    ``acc_ref``, the acc input version (``{id, version, ...}``) a run's sidecar names.

    A record naming another version of the same acc passes when the two versions' specs
    agree on ``keys`` (:func:`require_compatible_acc`, read from ``acc_registry``):
    :data:`ALL_SPEC_KEYS` for Stages 2 and 3, :data:`EDGE_UNIVERSE_ROLE_KEYS` for
    Stages 4 (warm) and 6 (#119). A structure built before
    the record existed has none: unless ``require_record``, that is logged as a warning
    and the caller goes on, so the runs made from it still load. Reads the record and
    the two specs only, so a caller checks before the (multi-GB) load.

    Raises
    ------
    ValueError
        ``acc_ref`` lacks its id or version; the record names another acc id, or a
        version whose spec differs in a key of ``keys``; or ``require_record`` and the
        structure has no record.
    """
    ref = (acc_ref.get("id"), acc_ref.get("version"))
    if not ref[0] or ref[1] is None:
        raise ValueError(f"the acc ref {acc_ref!r} names no acc id and version, so the "
                         f"structure in {storage_dir} cannot be checked against it")
    path = os.path.join(storage_dir, ref[0], BUILD_RECORD)
    if not os.path.isfile(path):
        msg = (f"{os.path.dirname(path)}: the built structure has no build record "
               f"(built before #53), so it cannot be checked against acc "
               f"{ref[0]} v{ref[1]}")
        if require_record:
            raise ValueError(f"{msg}: rebuild it with scripts/acc.py")
        get_logger("AccountingStructure").warning(msg)
        return
    with open(path, encoding="utf-8") as f:
        rec = json.load(f)
    context = (f"{os.path.dirname(path)} was built from acc {rec['id']} "
               f"v{rec['version']}, but the run names acc {ref[0]} v{ref[1]}")
    remedy  = ("rebuild the structure with scripts/acc.py, or run against the acc it "
               "was built from")
    if rec["id"] != ref[0]:
        raise ValueError(f"{context}: {remedy}")
    require_compatible_acc(acc_registry, ref[0], rec["version"], ref[1], keys,
                           context=context, remedy=remedy)
