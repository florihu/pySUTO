"""
emit.py
=======
Relation → constraint rows: the one adapter from :class:`~pysuto.relation.relation.Relation`
to a :class:`~pysuto.constraints.base.ConstraintBlock`, plus the ratio relations' dynamic σ
rule (σ_r → flow space against the Stage-2 prior).

Public API
----------
emit(relations, n_edges)                         Relations → ConstraintBlock (static)
ratio_scale_index(relations)                     Flattened numerator / denominator index
ratio_sigma_c(index, static, f_agg, t, ...)      Per-year σ_c of ratio relations
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pysuto.relation.relation import Relation

from .base import ConstraintBlock, _empty_block
from .naming import SENSE_EQ, SENSE_LE, build_constraint_id, empty_row_meta_dict
from .sum import SIGMA_C_INACTIVE


def emit(relations: list[Relation], n_edges: int) -> ConstraintBlock:
    """Emit ``relations`` as rows of one block, in the given order.

    ``G`` carries each relation's coefficients at its edges; ``c`` its ``rhs``;
    ``σ_c`` is 0 on a hard relation and ``sigma`` on a soft one; ``row_meta`` holds the
    ``constraint_id`` composed from ``type``/``variant``/``specs``/``time`` and the
    relation's own ``meta`` columns. Raises ``ValueError`` on a ``sense`` other than
    ``eq``/``le``. No relations → a zero-row block.
    """
    if not relations:
        return _empty_block(n_edges)
    M = len(relations)
    lens = np.fromiter((len(r.edges) for r in relations), dtype=np.int64, count=M)
    rows = np.repeat(np.arange(M, dtype=np.int64), lens)
    cols = np.concatenate([np.asarray(r.edges, dtype=np.int64) for r in relations])
    vals = np.concatenate([np.asarray(r.coefficients, dtype=np.float64)
                           for r in relations])
    meta_list = []
    for r in relations:
        if r.sense not in (SENSE_EQ, SENSE_LE):
            raise ValueError(f"Relation {r.type}/{r.variant}: unknown sense {r.sense!r}")
        meta = empty_row_meta_dict()
        meta.update({
            "constraint_id": build_constraint_id(
                                r.type, r.variant, *r.specs, year=r.time),
            "type":          r.type,
            "variant":       r.variant,
            "is_hard":       bool(r.hard),
            "sense":         r.sense,
            "time":          r.time,
            "valuation":     r.valuation,
            **r.meta,
        })
        meta_list.append(meta)
    return ConstraintBlock(
        G_rows   = sp.csr_matrix((vals, (rows, cols)), shape=(M, n_edges)),
        c        = np.array([r.rhs for r in relations], dtype=np.float64),
        sigma_c  = np.array([0.0 if r.hard else r.sigma for r in relations],
                            dtype=np.float64),
        row_meta = pd.DataFrame(meta_list),
    )


def ratio_scale_index(relations: list[Relation]):
    """Flattened (row, col) index arrays of the numerator and denominator sides of every
    ratio relation, plus ``|r|`` and the denominator-mode flag per row — the input of
    :func:`ratio_sigma_c`, evaluated in one vectorised pass per timestep."""
    rn, cn, rd, cd, rabs, den_mode = [], [], [], [], [], []
    for row_id, rel in enumerate(relations):
        num_cols = rel.edges[:rel.n_num]
        den_cols = rel.edges[rel.n_num:]
        rn += [row_id] * len(num_cols); cn += list(num_cols)
        rd += [row_id] * len(den_cols); cd += list(den_cols)
        rabs.append(abs(rel.coefficients[rel.n_num]))
        den_mode.append(rel.scale_mode == "denominator")
    return (np.asarray(rn, dtype=np.int64), np.asarray(cn, dtype=np.int64),
            np.asarray(rd, dtype=np.int64), np.asarray(cd, dtype=np.int64),
            np.asarray(rabs), np.asarray(den_mode, dtype=bool))


def ratio_sigma_c(index, static: ConstraintBlock, f_agg: np.ndarray, t,
                  sigma_rel_floor: float, scale_abs_floor: float) -> np.ndarray:
    """σ_c of ratio rows for year ``t``: the relative σ_r of ``static.sigma_c`` (floored)
    times the flow-space scale against the prior ``f_agg``.

    Symmetric ratios (r ≠ 0) scale by ``max(Σ|a₀,num|, |r|·Σ|a₀,denom|)``; net-stock
    bands (``scale_mode="denominator"``, r = 0) by the throughput ``Σ|a₀,denom|`` alone,
    else ``|r|·den = 0`` would collapse σ_c to the absolute floor (quasi-hard). The
    absolute floor keeps σ_c finite when the throughput collapses. Rows whose ``time``
    differs from ``t`` get ``SIGMA_C_INACTIVE``.
    """
    rn, cn, rd, cd, rabs, den_mode = index
    M = static.G_rows.shape[0]
    num_scale = np.zeros(M)
    den_scale = np.zeros(M)
    np.add.at(num_scale, rn, np.abs(f_agg[cn]))
    np.add.at(den_scale, rd, np.abs(f_agg[cd]))
    scale   = np.where(den_mode, den_scale, np.maximum(num_scale, rabs * den_scale))
    scale   = np.maximum(scale, scale_abs_floor)
    sigma_r = np.maximum(static.sigma_c, sigma_rel_floor)
    sigma_c = sigma_r * scale

    times    = static.row_meta["time"].astype(str).to_numpy()
    sigma_c[times != str(t)] = SIGMA_C_INACTIVE
    return sigma_c
