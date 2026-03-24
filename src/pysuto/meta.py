import dataclasses

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
 
import numpy as np
import pandas as pd

from pysuto.utils import _EnPEncoder, _NpEncoder


@dataclass
class DataFeedMetadata:
    """
    All metadata for one DataFeed, mirroring your Excel layout:
 
    Grouping               Field                     Excel column
    ──────────────────────────────────────────────────────────────
    Identification         feed_name                 Feed_name
                           iedc_type                 IEDC_type
    Unc (pedigree)         reliability               Reliability
                           completeness              Completeness
                           temporal_correlation      Temporal_correlation
                           geographical_correlation  Geographical_correlation
                           other_correlation         Other_correlation
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
 
    # --- Computed (filled in by registry.register()) ------------------
    cv_total:             Optional[float] = None
    F_root_sum:           Optional[float] = None   # sum of root flows
    sigma_root_mean:      Optional[float] = None   # mean sigma across edges
    computation_timestamp: str            = ""
 
    # ------------------------------------------------------------------
    # Factory: read from your Excel layout
    # ------------------------------------------------------------------
 
    @classmethod
    def from_excel(
        cls,
        path: str,
        sheet_name: str = 0,
        grouping_col:  str = "Grouping",
        name_col:      str = "Meta_name",
        value_col:     str = "Value",
    ) -> "DataFeedMetadata":
        """
        Parse the Excel metadata table you described:
 
            Grouping | Meta_name                  | Value   | Required
            ─────────────────────────────────────────────────────────
            Ident.   | Feed_name                  | icsg    | True
            Unc      | Reliability                | 1       | True
            …
 
        Column names are configurable; defaults match your example.
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
 
        def get_int(key: str, default: int = 1) -> int:
            v = get(key, default)
            try:
                return int(v)
            except (TypeError, ValueError):
                return default
 
        return cls(
            feed_name  = str(get("Feed_name", "")),
            iedc_type  = str(get("IEDC_type",  "")),
            reliability              = get_int("Reliability",              1),
            completeness             = get_int("Completeness",             1),
            temporal_correlation     = get_int("Temporal_correlation",     1),
            geographical_correlation = get_int("Geographical_correlation", 1),
            other_correlation        = get_int("Other_correlation",        1),
            version            = get("Version",            0),
            date_of_assessment = str(get("Date_of_assessment", "")),
            person             = str(get("Person",             "")),
            dataset_link       = str(get("Dataset_link",       "")),
        )
 
    @classmethod
    def from_dict(cls, d: dict) -> "DataFeedMetadata":
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})
 
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
 
    def to_pedigree_scores(self):
        """Return a PedigreeScores object built from the pedigree fields."""
        from pysuto.unc import PedigreeScores
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
        self.computation_timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"