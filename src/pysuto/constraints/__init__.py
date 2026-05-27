"""
constraints — Stage 3 plugin constraint pipeline.

Public surface (re-exported here so call sites can keep using
``from pysuto.constraints import X``):

* :class:`BaseConstraintBuilder`  — abstract base
* :class:`ConstraintBlock`        — sparse block of rows from one builder
* :class:`ConstraintEngine`       — orchestrator
* :class:`ConstraintRun`          — persisted output of one engine run

Builders (registered in :class:`ConstraintEngine` in any order):
* :class:`MassBalanceBuilder`     — Type 1: hard mass-balance equalities
* :class:`SumConstraintBuilder`   — Type 2: soft sum equalities
* :class:`RatioConstraintBuilder` — Type 3: soft ratio equalities
* :class:`RatioBoundBuilder`      — Type 4: ratio-bound cones
* :class:`FlowBoundsBuilder`      — Type 5: flow-bound cones
"""

from .base         import BaseConstraintBuilder, ConstraintBlock
from .run          import ConstraintRun
from .engine       import ConstraintEngine, ConstraintUnitMixingError, MIXED_VALUATION_OK
from .mass_balance import MassBalanceBuilder
from .sum          import SumConstraintBuilder, SIGMA_C_INACTIVE, BOUND_C_INACTIVE
from .ratio        import RatioConstraintBuilder, RatioBoundBuilder
from .flow_bound   import FlowBoundsBuilder

__all__ = [
    "BaseConstraintBuilder",
    "ConstraintBlock",
    "ConstraintRun",
    "ConstraintEngine",
    "ConstraintUnitMixingError",
    "MIXED_VALUATION_OK",
    "MassBalanceBuilder",
    "SumConstraintBuilder",
    "SIGMA_C_INACTIVE",
    "BOUND_C_INACTIVE",
    "RatioConstraintBuilder",
    "RatioBoundBuilder",
    "FlowBoundsBuilder",
]
