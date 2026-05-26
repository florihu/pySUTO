"""
engine.py
=========
``ConstraintEngine`` — orchestrator that runs a registered list of
:class:`BaseConstraintBuilder` instances, assembles the static G matrix
once, and produces per-timestep (c, σ_c) overlays.

Includes the two engine-local helpers:

* :func:`_lp_regularize` — σ conditioning (cv_min and abs_floor combined via
                           Lp recipe) applied to σ_a / σ_c before Clarabel.
* :func:`_concat_blocks` — vstack a list of :class:`ConstraintBlock` into one.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pysuto.logging import get_logger
from pysuto.registry import ConstraintRunRegistry, make_run_id

from .base import BaseConstraintBuilder, ConstraintBlock, _empty_block
from .run  import ConstraintRun

_log = get_logger(__name__)


# G rows of these types are allowed to span multiple valuations (e.g.
# ratio_eq rows that bridge Mass<->CC via a Cu-grade coefficient r).
# Every other type must contain edges of exactly one valuation — the
# structural unit-coherence guarantee on which the σ-weighted QP relies.
MIXED_VALUATION_OK = frozenset({"ratio_eq", "ratio_bound"})


class ConstraintUnitMixingError(ValueError):
    """Raised when a G row of a non-whitelisted type mixes valuations."""


def _lp_regularize(
    sigma:     np.ndarray,
    centre:    np.ndarray,
    cv_min:    float,
    abs_floor: float,
    p:         float,
) -> tuple[np.ndarray, int, int]:
    """
    Lp-combine σ with a CV floor (cv_min·|centre|) and an absolute floor,
    matching the lp_combine recipe from pysuto.lp:

        σ' = ( |σ|^p + |cv_min·centre|^p + |abs_floor|^p )^(1/p)

    Returns (σ', n_cv_lifted, n_abs_lifted) — counts of entries whose σ
    was lifted ≥10% by each component (diagnostic only).
    """
    cv_term  = cv_min * np.abs(centre)
    abs_term = abs_floor

    sigma_new = (
        np.abs(sigma)    ** p
        + np.abs(cv_term)  ** p
        + np.abs(abs_term) ** p
    ) ** (1.0 / p)

    # "lifted ≥10%" — floor term is at least 10% the size of the
    # original σ (for σ=0, any positive floor counts).
    n_cv  = int((10.0 * cv_term  >= np.abs(sigma)).sum() if cv_min    > 0 else 0)
    n_abs = int((10.0 * abs_term >= np.abs(sigma)).sum() if abs_floor > 0 else 0)
    return sigma_new, n_cv, n_abs

try:
    from ..logging import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

_log = get_logger(__name__)


# =============================================================================
# ConstraintBlock  — output of a single builder for one phase (static/dynamic)
# =============================================================================


def _concat_blocks(blocks: list[ConstraintBlock], n_edges: int) -> ConstraintBlock:
    non_empty = [b for b in blocks if b.G_rows.shape[0] > 0]
    if not non_empty:
        return _empty_block(n_edges)
    return ConstraintBlock(
        G_rows   = sp.vstack([b.G_rows for b in non_empty], format="csr"),
        c        = np.concatenate([b.c for b in non_empty]),
        sigma_c  = np.concatenate([b.sigma_c for b in non_empty]),
        row_meta = pd.concat([b.row_meta for b in non_empty], ignore_index=True),
    )


# =============================================================================
# BaseConstraintBuilder
# =============================================================================


class ConstraintEngine:
    """
    Orchestrates an ordered list of BaseConstraintBuilder instances.

    Usage::

        engine = ConstraintEngine(acc_struct)
        engine.register(MassBalanceBuilder())
        engine.register(SumConstraintBuilder.from_acc_struct(acc_struct))
        engine.register(RatioConstraintBuilder.from_csv("params.csv", acc_struct))
        engine.register(RatioBoundBuilder.from_csv("params.csv", acc_struct))
        engine.register(FlowBoundsBuilder())

        run = engine.run(agg_result, timesteps=["2010", "2011", ...])

    run.G is built once from all static blocks.
    run.timestep_data[t] = {"c": ..., "sigma_c": ...} is assembled per timestep
    by overlaying dynamic blocks onto the static c / sigma_c.
    """

    def __init__(
        self,
        acc_struct,
        storage_dir: str = "data/proc/const",
        sigma_a_cv_min:    float = 0.05,
        sigma_a_abs_floor: float = 1e-6,
        sigma_c_cv_min:    float = 0.05,
        sigma_c_abs_floor: float = 1e-3,
        p: float = 1.3,
    ):
        self.acc_struct  = acc_struct
        self._builders:  list[BaseConstraintBuilder] = []
        self._registry   = ConstraintRunRegistry(storage_dir)
        self._sigma_a_cv_min    = float(sigma_a_cv_min)
        self._sigma_a_abs_floor = float(sigma_a_abs_floor)
        self._sigma_c_cv_min    = float(sigma_c_cv_min)
        self._sigma_c_abs_floor = float(sigma_c_abs_floor)
        self._p                 = float(p)

    def register(self, builder: BaseConstraintBuilder) -> None:
        self._builders.append(builder)

    def _check(self, cr: ConstraintRun) -> None:
        """Consistency checks on a completed ConstraintRun. Called automatically in run()."""
        acc = self.acc_struct
        rm  = cr.row_meta
        ok  = True

        # 1. G column count matches n_edges
        if cr.G.shape[1] != acc.n_edges:
            _log.error("CHECK FAIL: G.shape[1]=%d != n_edges=%d", cr.G.shape[1], acc.n_edges)
            ok = False
        else:
            _log.info("CHECK OK: G.shape = (%d rows, %d edges)", cr.G.shape[0], cr.G.shape[1])

        # 2. G column references all in bounds
        if cr.G.nnz > 0:
            max_col = int(cr.G.indices.max())
            if max_col >= acc.n_edges:
                _log.error("CHECK FAIL: G references edge %d >= n_edges %d", max_col, acc.n_edges)
                ok = False

        # 3. Constraint type breakdown
        type_counts = rm["type"].value_counts() if len(rm) > 0 else pd.Series(dtype=int)
        _log.info("Constraint type breakdown:\n%s",
                  type_counts.to_string() if len(type_counts) else "  (empty)")

        # 4. Expected mass_balance row count: one per (balanced_node × time)
        ei, ni = acc.edge_index, acc.node_index
        t_arr  = ei["time"].values
        expected_mb = 0
        for time in acc.time_range:
            t_mask = (t_arr == str(time))
            nodes_in  = set(ei.loc[t_mask, "to_id"].values)
            nodes_out = set(ei.loc[t_mask, "from_id"].values)
            for nid in nodes_in & nodes_out:
                if str(ni.loc[nid, "entity"]) in {"Flow", "Process"}:
                    expected_mb += 1
        actual_mb = int(type_counts.get("mass_balance", 0))
        if actual_mb == expected_mb:
            _log.info("CHECK OK: mass_balance rows = %d (matches expected)", actual_mb)
        else:
            _log.warning(
                "CHECK WARN: mass_balance rows = %d, expected %d from topology",
                actual_mb, expected_mb,
            )
            ok = False

        # 5. Hard constraints must have sigma_c == 0 (check first timestep)
        hard_mask = rm["is_hard"].values if len(rm) > 0 else np.array([], dtype=bool)
        if np.any(hard_mask) and cr.timestep_data:
            first_t = next(iter(cr.timestep_data))
            if np.any(cr.timestep_data[first_t]["sigma_c"][hard_mask] != 0.0):
                _log.warning("CHECK WARN: hard constraint rows have non-zero sigma_c at t=%s", first_t)
                ok = False
            else:
                _log.info("CHECK OK: hard constraint sigma_c = 0 (t=%s)", first_t)

        # 6. u >= l for all edges
        bad_bounds = int(np.sum(cr.u < cr.l))
        if bad_bounds:
            _log.warning("CHECK WARN: %d edges have u < l", bad_bounds)
            ok = False
        else:
            _log.info("CHECK OK: u >= l for all %d edges", acc.n_edges)

        # 7. Unit coherence: no G row outside the whitelist may mix valuations.
        if cr.G.nnz > 0 and "type" in rm.columns:
            edge_val = acc.edge_index["valuation"].values
            types_arr = rm["type"].to_numpy()
            G_csr = cr.G.tocsr()
            checked = 0
            for row_id in range(G_csr.shape[0]):
                if types_arr[row_id] in MIXED_VALUATION_OK:
                    continue
                cols = G_csr.indices[G_csr.indptr[row_id]: G_csr.indptr[row_id + 1]]
                if cols.size == 0:
                    continue
                row_vals = set(edge_val[cols].tolist())
                if len(row_vals) > 1:
                    cid = rm.iloc[row_id]["constraint_id"]
                    raise ConstraintUnitMixingError(
                        f"G row {row_id} (type={types_arr[row_id]!r}, "
                        f"constraint_id={cid!r}) mixes valuations {sorted(row_vals)}. "
                        f"Every sum_eq / mass_balance row must contain edges of "
                        f"exactly one valuation."
                    )
                checked += 1
            _log.info("CHECK OK: unit-coherence verified on %d rows "
                      "(whitelist: %s)", checked, sorted(MIXED_VALUATION_OK))

        _log.info("Constraint check %s", "PASSED" if ok else "FAILED (see warnings above)")

    def run(
        self,
        agg_result,
        timesteps:    list,
        agg_id:       str = "",
        run_id:       Optional[str] = None,
        label:        str = "",
        persist:      bool = True,
        diag_plot:    bool = False,
        parents:      Optional[dict] = None,
        stage_config: Optional[dict] = None,
    ) -> ConstraintRun:
        """
        Build G once, then assemble c(t) / sigma_c(t) for each timestep.

        Parameters
        ----------
        agg_result  : AggregationResult with f_agg and sigma_agg
        timesteps   : ordered list of time labels (strings or ints)
        agg_id      : provenance identifier for the agg_result. If empty and
                      ``agg_result._imp_run_id`` is set (i.e. the result came
                      out of ImputationRunRegistry.register/load), that run_id
                      is used automatically.
        run_id      : explicit run ID; auto-generated if None
        persist     : if True, save to registry
        diag_plot   : if True, generate and save ConstDiagnostic figures to
                      {storage_dir}/{run_id}/diag/ after the run
        """
        if not agg_id:
            agg_id = getattr(agg_result, "_imp_run_id", None) or ""
        acc        = self.acc_struct
        n_edges    = acc.n_edges
        f_agg      = agg_result.f_agg
        sigma_agg  = agg_result.sigma_agg

        # ── 0. NaN precondition ───────────────────────────────────────
        # The downstream solver (Stage 3b) requires a fully-defined prior.
        # Register pysuto.impute.FlatPriorImputer last in the imputation
        # pipeline (or halt upstream) to satisfy this contract.
        if np.isnan(f_agg).any() or np.isnan(sigma_agg).any():
            n_nan_f = int(np.isnan(f_agg).sum())
            n_nan_s = int(np.isnan(sigma_agg).sum())
            raise ValueError(
                f"agg_result contains NaN (f_agg: {n_nan_f}, sigma_agg: {n_nan_s}). "
                f"Register FlatPriorImputer last in the imputation pipeline "
                f"(or halt upstream) to ensure a fully-defined prior before "
                f"constraint building."
            )

        # ── 0b. σ_a prior Lp regularization ───────────────────────────
        # Lifts unphysically tight σ on edges via the same Lp recipe used
        # in impute.py. Operates on sigma_agg BEFORE build_dynamic, so the
        # regularized σ also flows into p_mean-mode σ_c propagation.
        #
        # This ordering is deliberate (see REP 2026-05-13 §7 and the
        # FlatPriorImputer docstring): the relative CV term
        # `sigma_a_cv_min · |a₀|` is a *modeling* claim about how tightly
        # we believe σ_a, not a numerical safeguard, so it should flow
        # into σ_c. The absolute floor `sigma_a_abs_floor` is the safety
        # net — once the FlatPrior 3-level scale cascade is in place it
        # should essentially never fire on imputer-filled edges.
        if self._sigma_a_cv_min > 0 or self._sigma_a_abs_floor > 0:
            sigma_agg, n_cv, n_abs = _lp_regularize(
                sigma_agg, f_agg,
                self._sigma_a_cv_min, self._sigma_a_abs_floor, self._p,
            )
            _log.info(
                "σ_a Lp regularization (p=%.2f): %d edges CV-lifted "
                "(CV_min=%.3f), %d edges abs-lifted (abs_floor=%.0e)",
                self._p, n_cv, n_abs,
                self._sigma_a_cv_min, self._sigma_a_abs_floor,
            )

        # ── 1. Build static G (time-invariant) ────────────────────────
        static_blocks = [b.build_static(acc) for b in self._builders]
        static_full   = _concat_blocks(static_blocks, n_edges)
        G_static      = static_full.G_rows
        c_static      = static_full.c.copy()
        sc_static     = static_full.sigma_c.copy()
        row_meta      = static_full.row_meta.copy()
        M             = G_static.shape[0]

        _log.info("ConstraintEngine: G shape (%d, %d)", M, n_edges)

        # ── 2. Per-timestep c and sigma_c ─────────────────────────────
        timestep_data: dict[str, dict] = {}
        for t in timesteps:
            t_label = str(t)
            # Identify rows belonging to this timestep from dynamic blocks
            c_t      = c_static.copy()
            sigma_c_t = sc_static.copy()

            # Track which static rows each dynamic block covers
            row_offset = 0
            for b, sb in zip(self._builders, static_blocks):
                dyn = b.build_dynamic(acc, f_agg, sigma_agg, t_label)
                bM  = sb.G_rows.shape[0]
                if dyn.G_rows.shape[0] == bM and bM > 0:
                    c_t[row_offset: row_offset + bM]       = dyn.c
                    sigma_c_t[row_offset: row_offset + bM] = dyn.sigma_c
                row_offset += bM

            timestep_data[t_label] = {"c": c_t, "sigma_c": sigma_c_t}

        # ── 2b. σ_c prior Lp regularization (free-cone rows only) ─────
        # Lifts unphysically tight σ on soft constraint rows via the same
        # Lp recipe used for σ_a. Hard rows (zero, nonneg) keep σ_c = 0 —
        # the downstream Clarabel transform never reads them.
        free_mask = (row_meta["cone_type"].to_numpy() == "free")
        if free_mask.any() and (
            self._sigma_c_cv_min > 0 or self._sigma_c_abs_floor > 0
        ):
            n_cv_total, n_abs_total = 0, 0
            for t_label, td in timestep_data.items():
                sc, c_arr = td["sigma_c"], td["c"]
                sc_free, n_cv, n_abs = _lp_regularize(
                    sc[free_mask], c_arr[free_mask],
                    self._sigma_c_cv_min, self._sigma_c_abs_floor, self._p,
                )
                sc[free_mask] = sc_free
                td["sigma_c"] = sc
                n_cv_total  += n_cv
                n_abs_total += n_abs
            _log.info(
                "σ_c Lp regularization (p=%.2f, %d free rows × %d timesteps): "
                "%d (row, t) entries CV-lifted (CV_min=%.3f), %d abs-lifted "
                "(abs_floor=%.0e)",
                self._p, int(free_mask.sum()), len(timestep_data),
                n_cv_total, n_abs_total,
                self._sigma_c_cv_min, self._sigma_c_abs_floor,
            )

        # ── 3. Assemble ConstraintRun ──────────────────────────────────
        if run_id is None:
            run_id = make_run_id("const")

        cr = ConstraintRun(
            G             = G_static,
            row_meta      = row_meta,
            a0            = f_agg.copy(),
            sigma_a       = sigma_agg.copy(),
            u             = acc.bounds.u.copy(),
            l             = acc.bounds.l.copy(),
            timestep_data = timestep_data,
            run_id        = run_id,
            acc_id        = acc.id,
            agg_id        = agg_id,
            created_at    = datetime.utcnow().isoformat(timespec="seconds") + "Z",
            label         = label,
            _p            = self._p,
        )

        self._check(cr)

        if persist:
            self._registry.register(cr, parents=parents, stage_config=stage_config)

        if diag_plot:
            from ..diag import ConstDiagnostic
            odir = os.path.join(self._registry.storage_dir, cr.run_id, "diag")
            ConstDiagnostic(cr, label=cr.run_id).plot(odir=odir)

        return cr
