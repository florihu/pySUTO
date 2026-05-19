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
* :class:`ConstDiagnostic`   — Stage 3: G-matrix sparsity, prior distribution,
                                constraint residuals ε₀ = c − G·a₀.
* :class:`OptimDiagnostics`  — Stage 4: optimisation residuals, σ_opt vs σ_agg,
                                convergence across timesteps.
* :class:`SolverDiag`        — Stage 4: solver-side residual/objective plots.

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

from .feed   import FeedDiagnostics
from .imp    import ImpDiagnostic
from .optim  import OptimDiagnostics
from .const  import ConstDiagnostic
from .agg    import AggDiagnostic
from .solver import SolverDiag

__all__ = [
    "FeedDiagnostics",
    "ImpDiagnostic",
    "OptimDiagnostics",
    "ConstDiagnostic",
    "AggDiagnostic",
    "SolverDiag",
]
