"""
diag — diagnostic visualisation classes for every pipeline stage.

Public surface (re-exported here so call sites can keep using
``from pysuto.diag import X``):

* :class:`FeedDiagnostics`   — Stage 1: feed reading, concordance coverage,
                                F_src distribution, CV propagation.
* :class:`AggDiagnostic`     — Stage 2: aggregation coverage, Omega weights,
                                f_agg/σ_agg distributions.
* :class:`ImpDiagnostic`     — Stage 2: per-step imputation distributions,
                                coverage evolution, TimeImputer timeseries.
* :class:`CoverageDiagnostic`— Stage 2: post-imputation system-coverage CSVs
                                (supply/use by commodity, boundary by entity_sub).
* :class:`StatsDiagnostic`   — Stage 2: post-imputation val-distribution CSVs
                                (per (sector, region, valuation) by from/to endpoint).
* :class:`ConstDiagnostic`   — Stage 3: G-matrix sparsity, prior distribution,
                                constraint residuals ε₀ = c − G·a₀.
* :class:`ConstStatsDiagnostic`— Stage 3: prior row-level constraint CSV
                                (lhs_prior = G·a₀ per constraint_id, all rows).
* :class:`SolverDiag`        — Stage 4: the opt-run figure portfolio — solver
                                health, convergence traces, objective
                                decomposition, objective share by sector /
                                identity, sector totals + retention, rocket.
* :class:`SolverStatsDiagnostic`— Stage 4: post-solve row-level CSVs (per-edge
                                defect a*−a₀; per-constraint lhs_prior/lhs_post).

Design conventions
------------------
- Each class is constructed with its data; nothing is computed at import time.
- ``.plot(odir=..., show=False)`` writes PNG(s) and returns a list of paths.
- ``.summary()`` prints a text block to stdout and returns it as a string.
- Matplotlib is imported lazily inside methods so this module can be
  imported in headless environments without triggering a backend error.
- No pysuto-internal imports in the class bodies; types are duck-typed
  so this subpackage stays at the leaf of the import graph.
"""

from .feed     import FeedDiagnostics
from .imp      import ImpDiagnostic, ownership_violations
from .const    import ConstDiagnostic
from .agg      import AggDiagnostic
from .solver   import SolverDiag
from .stats    import (
    CoverageDiagnostic,
    StatsDiagnostic,
    ConstStatsDiagnostic,
    SolverStatsDiagnostic,
)

__all__ = [
    "FeedDiagnostics",
    "ImpDiagnostic",
    "ownership_violations",
    "CoverageDiagnostic",
    "StatsDiagnostic",
    "ConstStatsDiagnostic",
    "ConstDiagnostic",
    "AggDiagnostic",
    "SolverDiag",
    "SolverStatsDiagnostic",
]
