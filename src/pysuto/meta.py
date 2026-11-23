"""meta.py
=======
``DataFeedMetadata``: a feed's identification, pedigree scores and provenance, read
from the ``meta.xlsx`` sidecar of its registered input.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Any, Union

import numpy as np
import pandas as pd

from pysuto.unc import PedigreeScores, SENSITIVITY_PARAMS


@dataclass
class DataFeedMetadata:
    """
    All metadata for one DataFeed, mirroring the layout of its Excel sidecar:

    Grouping               Field                     Excel column
    ──────────────────────────────────────────────────────────────
    Identification         feed_name                 Feed_name
                           iedc_type                 IEDC_type
    Unc (pedigree)         reliability               Reliability
                           completeness              Completeness
                           temporal_correlation      Temporal_correlation
                           geographical_correlation  Geographical_correlation
                           other_correlation         Other_correlation
                           <indicator>_sensitivity   <Indicator>_sensitivity (optional)
    Submission / review    version                   Version
                           date_of_assessment        Date_of_assessment
                           person                    Person
                           dataset_link              Dataset_link
    Computed outputs       cv_total                  (derived)
                           F_root_sum                (derived)
                           sigma_root_mean           (derived)
                           computation_timestamp     (derived)
    """

    # --- Identification -----------------------------------------------
    feed_name:  str = ""
    iedc_type:  str = ""

    # --- Pedigree (Laner et al. 2016) ---------------------------------
    reliability:              int = 1
    completeness:             int = 1
    temporal_correlation:     int = 1
    geographical_correlation: int = 1
    other_correlation:        int = 1

    # Sensitivity per non-reliability indicator
    completeness_sensitivity:     str = "medium_sensitive"
    temporal_sensitivity:         str = "medium_sensitive"
    geographical_sensitivity:     str = "medium_sensitive"
    other_sensitivity:            str = "medium_sensitive"

    # --- Submission / review ------------------------------------------
    version:            Any = 0
    date_of_assessment: str = ""
    person:             str = ""
    dataset_link:       str = ""

    # --- Scope ---------------------------------------------------------
    # Optional name of a mask in acc_struct.masks. When set, the feed
    # build drops concordance rows whose resolved root edge is outside
    # this mask. Empty string ("") disables the filter.
    edge_scope_mask:    str = ""

    # When True, ROW-touching edges (region_from/region_to == "ROW") are
    # propagated with geographical_correlation pushed to its worst score (4),
    # inflating their σ. Default False = no penalty / no behaviour change.
    row_penalty:        bool = False

    # --- Computed (filled in by registry.register()) ------------------
    cv_total:             Optional[float] = None
    F_root_sum:           Optional[float] = None   # sum of root flows
    sigma_root_mean:      Optional[float] = None   # mean sigma across edges
    computation_timestamp: str            = ""

    # ------------------------------------------------------------------
    # Factory: read from the Excel sidecar
    # ------------------------------------------------------------------

    @classmethod
    def from_excel(
        cls,
        path: str,
        sheet_name: Union[int, str] = 0,
        name_col:      str = "Meta_name",
        value_col:     str = "Value",
    ) -> "DataFeedMetadata":
        """
        Parse the Excel metadata table of a feed's sidecar:

            Grouping | Meta_name                  | Value   | Required
            ─────────────────────────────────────────────────────────
            Ident.   | Feed_name                  | icsg    | True
            Unc      | Reliability                | 1       | True
            …

        ``name_col`` and ``value_col`` name the columns read; the defaults match the
        registered sidecars. A
        ``<Indicator>_sensitivity`` row sets that indicator's sensitivity level
        (default ``medium_sensitive``).

        Raises
        ------
        ValueError
            If one of the five pedigree scores is absent or not an integer, or a
            sensitivity is not one of ``unc.SENSITIVITY_PARAMS``.
        """
        df = pd.read_excel(path, sheet_name=sheet_name)
        df.columns = [c.strip() for c in df.columns]
        lookup: Dict[str, Any] = {
            str(row[name_col]).strip(): row[value_col]
            for _, row in df.iterrows()
            if pd.notna(row[name_col])
        }

        # Helper: case-insensitive get with fallback
        def get(key: str, default=None):
            for k, v in lookup.items():
                if k.lower() == key.lower():
                    return v
            return default

        def get_score(key: str) -> int:
            v = get(key)
            try:
                score = int(v)
                if score != float(v):
                    raise ValueError(v)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{path}: pedigree score {key!r} must be an integer, got {v!r}"
                ) from None
            return score

        def get_sensitivity(key: str) -> str:
            v = str(get(key, "medium_sensitive")).strip()
            if v not in SENSITIVITY_PARAMS:
                raise ValueError(
                    f"{path}: {key!r} must be one of {sorted(SENSITIVITY_PARAMS)}, "
                    f"got {v!r}"
                )
            return v

        def get_bool(key: str, default: bool = False) -> bool:
            v = get(key, default)
            if isinstance(v, bool):
                return v
            if v is None:
                return default
            return str(v).strip().lower() in {"true", "1", "yes", "y", "t"}

        return cls(
            feed_name  = str(get("Feed_name", "")),
            iedc_type  = str(get("IEDC_type",  "")),
            reliability              = get_score("Reliability"),
            completeness             = get_score("Completeness"),
            temporal_correlation     = get_score("Temporal_correlation"),
            geographical_correlation = get_score("Geographical_correlation"),
            other_correlation        = get_score("Other_correlation"),
            completeness_sensitivity = get_sensitivity("Completeness_sensitivity"),
            temporal_sensitivity     = get_sensitivity("Temporal_sensitivity"),
            geographical_sensitivity = get_sensitivity("Geographical_sensitivity"),
            other_sensitivity        = get_sensitivity("Other_sensitivity"),
            version            = get("Version",            0),
            date_of_assessment = str(get("Date_of_assessment", "")),
            person             = str(get("Person",             "")),
            dataset_link       = str(get("Dataset_link",       "")),
            edge_scope_mask    = str(get("Edge_scope_mask",    "") or "").strip(),
            row_penalty        = get_bool("Row_penalty",       False),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "DataFeedMetadata":
        """Build from a flat dict (a sidecar's ``to_dict``); unknown keys are ignored."""
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_pedigree_scores(self):
        """Return a PedigreeScores object built from the pedigree fields."""
        return PedigreeScores(
            reliability              = self.reliability,
            completeness             = self.completeness,
            temporal_correlation     = self.temporal_correlation,
            geographical_correlation = self.geographical_correlation,
            other_correlation        = self.other_correlation,
            completeness_sensitivity     = self.completeness_sensitivity,
            temporal_sensitivity         = self.temporal_sensitivity,
            geographical_sensitivity     = self.geographical_sensitivity,
            other_sensitivity            = self.other_sensitivity,
        )

    def fill_computed(self, result) -> None:
        """Populate derived fields from a PropagationResult."""
        self.cv_total            = float(result.cv_src)
        self.F_root_sum          = float(np.nansum(result.F_root))
        self.sigma_root_mean     = float(np.nanmean(result.sigma_root))
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        self.computation_timestamp = now.isoformat(timespec="seconds") + "Z"
