"""
flow_bound.py
=============
``FlowBoundsBuilder`` — Type-5 cone slots that the engine fills with the
:class:`AccountingStructure` bounds (one ``a_i ≤ u_i, a_i ≥ l_i`` cone pair
per edge).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pysuto.logging import get_logger

from .base import BaseConstraintBuilder, ConstraintBlock, _empty_block

_log = get_logger(__name__)


class FlowBoundsBuilder(BaseConstraintBuilder):
    """
    Flow bounds from acc_struct — stored as ConstraintRun.u / .l, NOT as G rows.

    ConstraintEngine.run() copies acc_struct.bounds.u and .l directly into the
    ConstraintRun fields.  Registering this builder is a no-op for G; it exists
    as a documentation marker that bounds are part of the problem formulation.
    """

    def build_static(self, acc_struct) -> ConstraintBlock:
        non_default = (
            (acc_struct.bounds.l != acc_struct.bounds.DEFAULT_LOWER) |
            (acc_struct.bounds.u != acc_struct.bounds.DEFAULT_UPPER)
        )
        _log.info(
            "FlowBoundsBuilder: %d edges with non-default bounds → stored in u/l, not G",
            int(non_default.sum()),
        )
        return _empty_block(acc_struct.n_edges)


# =============================================================================
# ConstraintRun  — output of ConstraintEngine.run()
# =============================================================================

