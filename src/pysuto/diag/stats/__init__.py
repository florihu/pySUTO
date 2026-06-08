"""
diag.stats — per-bucket statistics diagnostics (one stage-neutral home).

Groups the stats-table diagnostics and their shared builders:

* :class:`CoverageDiagnostic`    (``coverage.py``)— Stage 2: post-imputation system
                                                    coverage CSVs + the ``has_val`` rule.
* :class:`StatsDiagnostic`       (``val.py``)    — Stage 2: imputed-val distribution
                                                    per (sector, region, valuation).
* :class:`ConstStatsDiagnostic`  (``const.py``)  — Stage 3: prior soft-constraint
                                                    residual/violation table.
* :class:`SolverStatsDiagnostic` (``solver.py``) — Stage 4: solver defect distribution
                                                    + solved soft-constraint violations.

The shared, stateless builders (:func:`grouped_stats`, :func:`soft_constraint_table`)
live in ``_core.py`` and are the DRY seam across all three. ``StatsDiagnostic`` also
composes :class:`CoverageDiagnostic` (reuses its per-edge frame + ``has_val``).
"""

from .coverage import CoverageDiagnostic
from .val      import StatsDiagnostic
from .const    import ConstStatsDiagnostic
from .solver   import SolverStatsDiagnostic

__all__ = [
    "CoverageDiagnostic",
    "StatsDiagnostic",
    "ConstStatsDiagnostic",
    "SolverStatsDiagnostic",
]
