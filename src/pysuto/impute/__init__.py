"""
impute — Stage 2c imputation pipeline.

Public surface (re-exported here so call sites can keep using
``from pysuto.impute import X``):

* :class:`BaseImputer`         — abstract base
* :class:`ImputationEngine`    — orchestrator
* :class:`ImputationLog`       — per-edge fill provenance table
* :class:`ImputationMetadata`  — run-level per-step summary

Imputers (run order is up to the engine's registration sequence):
* :class:`TradeImputer`        — seed cross-region Flow→Process CC from BACI Mass twin
* :class:`NetworkImputer`      — process-recipe traversal (mass closure on CC)
* :class:`ParametricImputer`   — CSV bidirectional pairs + λ directed relations
* :class:`FormulaImputer`      — multi-source balance equations
* :class:`TimeImputer`         — temporal interpolation / extrapolation
* :class:`FlatPriorImputer`    — final fallback (f=0, large σ)

Supporting dataclasses (used in :class:`ParametricImputer` /
:class:`FormulaImputer` constructors):
* :class:`ParametricRelation`, :class:`FormulaRelation`, :class:`ParamRef`,
  :class:`SourceTerm`
"""

from .base       import BaseImputer
from .engine     import ImputationEngine, ImputationLog, ImputationMetadata
from .parametric import ParametricImputer, ParametricRelation, _reg_sensitivity
from .formula    import FormulaImputer, FormulaRelation, ParamRef, SourceTerm
from .network    import NetworkImputer, ProcessRecipe, derive_process_recipes
from .trade      import TradeImputer
from .time       import TimeImputer
from .share      import ShareImputer
from .flat_prior import FlatPriorImputer

__all__ = [
    "BaseImputer",
    "ImputationEngine",
    "ImputationLog",
    "ImputationMetadata",
    "TradeImputer",
    "NetworkImputer",
    "ProcessRecipe",
    "derive_process_recipes",
    "ParametricImputer",
    "ParametricRelation",
    "FormulaImputer",
    "FormulaRelation",
    "ParamRef",
    "SourceTerm",
    "TimeImputer",
    "ShareImputer",
    "FlatPriorImputer",
    "_reg_sensitivity",
]
