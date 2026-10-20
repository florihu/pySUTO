"""
diag.stats — per-bucket statistics diagnostics (one stage-neutral home).

Groups the stats-table diagnostics and their shared builders:

* :class:`CoverageDiagnostic`    (``coverage.py``)— Stage 2: post-imputation system
                                                    coverage CSVs + the ``has_val`` rule.
* :class:`StatsDiagnostic`       (``val.py``)    — Stage 2: imputed-val distribution
                                                    per (sector, region, valuation).
* :class:`ConstStatsDiagnostic`  (``const.py``)  — Stage 3: prior row-level
                                                    constraint table (lhs_prior = G·a0).
* :class:`SolverStatsDiagnostic` (``solver.py``) — Stage 4: row-level edge table
                                                    (defect a*−a_prior) + constraint
                                                    table (lhs_prior + lhs_post =
                                                    c_star), and the cached all-years
                                                    frames behind ``SolverDiag``.
* ``objective.py``                               — Stage 4: stateless objective-share
                                                    builders (family frames, Σz² by
                                                    sector key, identity ranking,
                                                    z profiles, objective decomposition).
* :class:`UncPropStatsDiagnostic`(``uncprop.py``)— Stage 5: all-years edge frame
                                                    (σ_a*, contraction κ, defect z,
                                                    prior class, κ–z quadrant,
                                                    standardised shift z/√κ) behind
                                                    ``UncPropDiag``, its calibration
                                                    table and the Birge ratio.
* :class:`ProblemStatsDiagnostic`(``problem.py``)— Stages 2–4: the problem one
                                                    period is handed (edge provenance,
                                                    chain profile, prior breach and
                                                    active set per row, a region's
                                                    standard-form slice) behind the
                                                    Results' problem figure.

The shared, stateless builders (:func:`grouped_stats`, :func:`constraint_table`,
:func:`sector_totals`, :func:`boundary_edges`, :func:`filter_regions`) live in
``_core.py`` and are the DRY seam across all three. ``StatsDiagnostic`` also
composes :class:`CoverageDiagnostic` (reuses its per-edge frame + ``has_val``).
"""

from .coverage import CoverageDiagnostic
from .val      import StatsDiagnostic
from .const    import ConstStatsDiagnostic
from .solver   import SolverStatsDiagnostic
from .uncprop  import UncPropStatsDiagnostic
from .problem  import ProblemStatsDiagnostic

__all__ = [
    "CoverageDiagnostic",
    "StatsDiagnostic",
    "ConstStatsDiagnostic",
    "SolverStatsDiagnostic",
    "UncPropStatsDiagnostic",
    "ProblemStatsDiagnostic",
]
