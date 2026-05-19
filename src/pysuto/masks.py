"""
masks.py
========
Named boolean masks over an ``AccountingStructure``'s edge index.

A *mask* is a `(n_edges,)` boolean array aligned to ``acc.edge_index``. Masks
are used by:

  * :class:`~pysuto.impute.flat_prior.FlatPriorImputer` — per-mask CV ranges
    and an optional multiplicative "ROW" modulator.
  * Constraint builders — restrict a constraint's edge scope to a named mask.
  * Datafeeds — restrict a feed to in-scope edges via ``edge_scope_mask`` in
    its meta file.

User entry points
-----------------
1. **CSV rules** — a long-format file with one row per (mask, include-or-exclude,
   wildcard selector) tuple. Schema::

       mask_name | logic   | sector_from | sector_to | region_from | region_to | valuation | time | entity_from | entity_to
       row       | include | *           | *         | ROW         | *         | *         | *    | *           | *
       row       | include | *           | *         | *           | ROW       | *         | *    | *           | *
       domestic  | include | *           | *         | *           | *         | *         | *    | *           | *
       domestic  | exclude | *           | *         | ROW         | *         | *         | *    | *           | *
       domestic  | exclude | *           | *         | *           | ROW       | *         | *    | *           | *

   - ``mask_name`` and ``logic`` are required. Every selector column is optional;
     missing or wildcard (``*``, empty, ``nan``) cells match everything.
   - For a given ``mask_name``, ``include`` rows are OR-unioned, then
     ``exclude`` rows are AND-NOT'd. Row order within the file is preserved.

2. **Python API** — :meth:`AccountingStructure.register_mask` accepts a raw
   boolean array, or a callable receiving the expanded edge index and returning
   a boolean array.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Union

import numpy as np
import pandas as pd

from pysuto.logging import get_logger

_log = get_logger(__name__)


# Selector columns supported in masks.csv. Anything else in the file is ignored
# (so users can add a free-text "description" column without errors).
_SELECTOR_COLS: tuple[str, ...] = (
    "sector_from",
    "sector_to",
    "region_from",
    "region_to",
    "valuation",
    "time",
    "entity_from",
    "entity_to",
)

# Cell values that mean "match anything" for a selector.
_WILDCARD_VALUES: frozenset[str] = frozenset({"*", "", "nan", "none", "any"})

MaskPredicate = Callable[[pd.DataFrame], np.ndarray]
MaskInput = Union[np.ndarray, MaskPredicate]


def _match(arr: np.ndarray, val: object) -> np.ndarray:
    """Wildcard-aware equality. Mirrors EdgeBounds._match in acc.py."""
    s = str(val).strip().lower()
    if s in _WILDCARD_VALUES:
        return np.ones(len(arr), dtype=bool)
    return arr == str(val).strip()


@dataclass(frozen=True)
class _MaskRule:
    name: str
    logic: str  # "include" or "exclude"
    selectors: dict[str, object]  # selector_col -> string value


class MaskRuleSet:
    """A collection of CSV-loaded mask rules, resolvable against an edge index."""

    def __init__(self, rules: list[_MaskRule]) -> None:
        self._rules = list(rules)

    @property
    def names(self) -> list[str]:
        seen: list[str] = []
        for r in self._rules:
            if r.name not in seen:
                seen.append(r.name)
        return seen

    @classmethod
    def from_csv(cls, path: Union[str, Path]) -> "MaskRuleSet":
        df = pd.read_csv(path, dtype=str, comment="#").fillna("*")
        required = {"mask_name", "logic"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"masks CSV {path!r} is missing required columns: {sorted(missing)}. "
                f"Required: mask_name, logic; optional: {list(_SELECTOR_COLS)}"
            )

        rules: list[_MaskRule] = []
        for i, row in df.iterrows():
            name = str(row["mask_name"]).strip()
            if not name or name.lower() in _WILDCARD_VALUES:
                continue
            logic = str(row["logic"]).strip().lower()
            if logic not in ("include", "exclude"):
                raise ValueError(
                    f"masks CSV {path!r} row {i}: logic must be 'include' or 'exclude', got {logic!r}"
                )
            selectors = {
                col: str(row.get(col, "*")).strip()
                for col in _SELECTOR_COLS
                if col in df.columns
            }
            rules.append(_MaskRule(name=name, logic=logic, selectors=selectors))
        return cls(rules)

    def resolve(
        self,
        edge_index: pd.DataFrame,
        node_index: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        """Return ``{mask_name: bool_array}`` for every name in this set."""
        if not self._rules:
            return {}

        n_edges = len(edge_index)
        ni = node_index.loc[edge_index["from_id"].values]
        nj = node_index.loc[edge_index["to_id"].values]
        col_arrays: dict[str, np.ndarray] = {
            "sector_from": ni["sector"].values.astype(str),
            "sector_to":   nj["sector"].values.astype(str),
            "region_from": ni["region"].values.astype(str),
            "region_to":   nj["region"].values.astype(str),
            "entity_from": ni["entity"].values.astype(str),
            "entity_to":   nj["entity"].values.astype(str),
            "valuation":   edge_index["valuation"].values.astype(str),
            "time":        edge_index["time"].values.astype(str),
        }

        masks: dict[str, np.ndarray] = {
            name: np.zeros(n_edges, dtype=bool) for name in self.names
        }

        for rule in self._rules:
            row_mask = np.ones(n_edges, dtype=bool)
            for col, val in rule.selectors.items():
                row_mask &= _match(col_arrays[col], val)
            if rule.logic == "include":
                masks[rule.name] |= row_mask
            else:  # exclude
                masks[rule.name] &= ~row_mask

        for name, m in masks.items():
            _log.info(
                "MaskRuleSet.resolve: %r → %d / %d edges",
                name, int(m.sum()), n_edges,
            )
        return masks


def coerce_mask(
    name: str,
    mask: MaskInput,
    edge_index: pd.DataFrame,
    node_index: pd.DataFrame,
    n_edges: int,
) -> np.ndarray:
    """Normalise a user-supplied mask (array or callable) to a bool ndarray.

    Used by :meth:`AccountingStructure.register_mask`.
    """
    if callable(mask):
        from pysuto.param_relation import expand_edge_index

        ei_exp = expand_edge_index(edge_index, node_index)
        out = np.asarray(mask(ei_exp), dtype=bool)
    else:
        out = np.asarray(mask, dtype=bool)

    if out.shape != (n_edges,):
        raise ValueError(
            f"register_mask({name!r}): expected shape ({n_edges},), got {out.shape}"
        )
    return out


def load_masks_csv(
    path: Union[str, Path],
    edge_index: pd.DataFrame,
    node_index: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Convenience: load a masks CSV and resolve in one call."""
    return MaskRuleSet.from_csv(path).resolve(edge_index, node_index)
