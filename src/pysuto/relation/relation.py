"""
relation.py
===========
:class:`Relation` — one linear relation over edges, the object every stage reads.

``Σ coefficients[k] · a[edges[k]]  (sense)  rhs ± σ``. A **constraint row** is a
relation emitted into ``G`` (:func:`pysuto.constraints.emit.emit`); the network imputer
solves the same relations for a lone unknown. ``sense`` is ``eq`` or ``le`` (ADR 0011,
#11), so equalities and the emit-only inequalities are one type. Mass balances and ratio
equalities are built as relations today; sum and bound builders still emit rows directly.

This module imports nothing from ``pysuto.constraints`` (import direction ``acc →
relation → impute → constraints``): the ``constraint_id`` is composed at emit time from
``type``, ``variant``, ``specs`` and ``time``.

Public API
----------
Relation        One relation: edges, coefficients, rhs, sense, hardness, σ, identity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# The relation types the network imputer also builds, and the two senses (the
# ``constraint_id`` / ``row_meta`` vocabulary of :mod:`pysuto.constraints.naming`).
TYPE_MASSBAL = "massbal"
TYPE_RATIO   = "ratio"
SENSE_EQ     = "eq"
SENSE_LE     = "le"


@dataclass
class Relation:
    """One linear relation ``Σ coefficients · a[edges] (sense) rhs ± σ``.

    Parameters
    ----------
    edges, coefficients : ndarray
        ``int64`` edge ids and ``float64`` coefficients, same length; a numerator edge
        list comes first on a ratio relation. Order is the row's column order.
    type, variant, specs, time
        The ``constraint_id`` slots: ``{type}_{variant}_{specs...}_{time}`` (grammar in
        :mod:`pysuto.constraints.naming`; each token underscore-free).
    valuation : str
        The ``row_meta`` valuation of the row.
    rhs : float
        Right-hand side ``c``.
    sense : {"eq", "le"}
        ``G·a = c`` or ``G·a ≤ c``.
    hard : bool
        Hard row ⇔ ``σ_c = 0``; a hard relation emits σ = 0 whatever ``sigma`` says.
    sigma : float
        σ of a soft row. For a ratio relation (``n_num`` set) it is the *relative* σ_r,
        rescaled to flow space per year by :func:`~pysuto.constraints.emit.ratio_sigma_c`.
    scale_mode : {"symmetric", "denominator"}, optional
        How a ratio relation's σ_r is scaled (ratio relations only).
    n_num : int, optional
        Ratio relations: the first ``n_num`` edges are the numerator, the rest the
        denominator, whose coefficients are ``−r``.
    meta : dict
        The ``row_meta`` columns beyond type/variant/sense/time/valuation/is_hard.
        A params-built ratio relation also carries ``sigma_param`` here: the params row's
        σ in *absolute ratio units*, whereas ``sigma`` is the relative σ_r
        (``sigma_param / |r|``); a fill reads ``meta["sigma_param"]``, Stage 3 ``sigma``.
    agg : bool
        The relation pools several edges per side (``agg=true`` params row).
    source : str
        What the relation was built from (params id, ``"massbal"``).
    """

    edges:        np.ndarray
    coefficients: np.ndarray
    type:         str
    variant:      str
    specs:        tuple
    time:         str
    valuation:    str
    rhs:          float = 0.0
    sense:        str = "eq"
    hard:         bool = True
    sigma:        float = 0.0
    scale_mode:   Optional[str] = None
    n_num:        Optional[int] = None
    meta:         dict = field(default_factory=dict)
    agg:          bool = False
    source:       str = ""
