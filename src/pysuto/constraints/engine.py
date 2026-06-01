"""
engine.py
=========
``ConstraintEngine`` — orchestrator that runs a registered list of
:class:`BaseConstraintBuilder` instances, assembles the structural
:class:`StructG` once, and produces per-period (data, c, σ_c, active)
overlays into the operator's data tape.

Includes the engine-local helpers:

* :func:`_lp_regularize` — σ conditioning (cv_min and abs_floor combined via
                           Lp recipe) applied to σ_a / σ_c before Clarabel.
* :func:`_concat_blocks` — vstack a list of :class:`ConstraintBlock` into one,
                           preserving each builder's row order and ``active``
                           mask.
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
from .struct_g import StructG

_log = get_logger(__name__)


# G rows of these types are allowed to span multiple valuations (e.g.
# ratio_eq rows that bridge Mass<->CC via a Cu-grade coefficient r).
# Every other type must contain edges of exactly one valuation.
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
    Lp-combine σ with a CV floor (cv_min·|centre|) and an absolute floor.

        σ' = ( |σ|^p + |cv_min·centre|^p + |abs_floor|^p )^(1/p)

    Returns (σ', n_cv_lifted, n_abs_lifted).
    """
    cv_term  = cv_min * np.abs(centre)
    abs_term = abs_floor

    sigma_new = (
        np.abs(sigma)    ** p
        + np.abs(cv_term)  ** p
        + np.abs(abs_term) ** p
    ) ** (1.0 / p)

    n_cv  = int((10.0 * cv_term  >= np.abs(sigma)).sum() if cv_min    > 0 else 0)
    n_abs = int((10.0 * abs_term >= np.abs(sigma)).sum() if abs_floor > 0 else 0)
    return sigma_new, n_cv, n_abs


def _concat_blocks(blocks: list[ConstraintBlock], n_edges: int) -> ConstraintBlock:
    """Vertically concatenate non-empty blocks, preserving ``active`` mask."""
    non_empty = [b for b in blocks if b.G_rows.shape[0] > 0]
    if not non_empty:
        return _empty_block(n_edges)
    return ConstraintBlock(
        G_rows   = sp.vstack([b.G_rows for b in non_empty], format="csr"),
        c        = np.concatenate([b.c for b in non_empty]),
        sigma_c  = np.concatenate([b.sigma_c for b in non_empty]),
        row_meta = pd.concat([b.row_meta for b in non_empty], ignore_index=True),
        active   = np.concatenate([b.active for b in non_empty]),
    )


class ConstraintEngine:
    """
    Orchestrates an ordered list of BaseConstraintBuilder instances.

    Each builder's ``build_static`` emits its row sparsity once; the engine
    initialises a :class:`StructG` whose data tape replicates the static
    coefficients across every period. The per-period loop then calls each
    builder's ``build_dynamic(t)`` to obtain the per-period (data, c,
    σ_c, active) overlay; the engine writes those into the operator
    tape at the builder's row offset.

    Usage::

        engine = ConstraintEngine(acc_struct)
        engine.register(MassBalanceBuilder())
        engine.register(SumConstraintBuilder.from_acc_struct(acc_struct))
        engine.register(RatioConstraintBuilder.from_csv("params.csv", acc_struct))
        engine.register(RatioBoundBuilder.from_csv("params.csv", acc_struct))
        engine.register(FlowBoundsBuilder())

        run = engine.run(agg_result, timesteps=acc_struct.time_range)
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
        """Consistency checks on a completed ConstraintRun."""
        acc = self.acc_struct
        rm  = cr.row_meta
        G   = cr.G
        ok  = True

        # 1. G column count matches n_edges
        if G.n_edges != acc.n_edges:
            _log.error("CHECK FAIL: G.n_edges=%d != acc.n_edges=%d", G.n_edges, acc.n_edges)
            ok = False
        else:
            _log.info("CHECK OK: StructG shape (%d constraints, %d edges, %d timesteps)",
                      G.n_constraints, G.n_edges, G.n_timesteps)

        # 2. Column references in bounds
        if G.nnz > 0:
            max_col = int(G.indices.max())
            if max_col >= acc.n_edges:
                _log.error("CHECK FAIL: G references edge %d >= n_edges %d", max_col, acc.n_edges)
                ok = False

        # 3. Constraint type breakdown
        type_counts = rm["type"].value_counts() if len(rm) > 0 else pd.Series(dtype=int)
        _log.info("Constraint type breakdown:\n%s",
                  type_counts.to_string() if len(type_counts) else "  (empty)")

        # 4. Mass balance row count: one per (balanced_node × valuation)
        ei, ni = acc.edge_index, acc.node_index
        v_arr  = ei["valuation"].values
        expected_mb = 0
        for valuation in acc.valuation:
            v_mask = (v_arr == valuation)
            nodes_in  = set(ei.loc[v_mask, "to_id"].values)
            nodes_out = set(ei.loc[v_mask, "from_id"].values)
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

        # 5. Hard constraints must have sigma_c == 0 (check first period)
        hard_mask = rm["is_hard"].values if len(rm) > 0 else np.array([], dtype=bool)
        if np.any(hard_mask):
            if np.any(cr.sigma_c[hard_mask, 0] != 0.0):
                _log.warning("CHECK WARN: hard constraint rows have non-zero sigma_c at t=0")
                ok = False
            else:
                _log.info("CHECK OK: hard constraint sigma_c = 0 (t=0)")

        # 6. u >= l for all edges
        bad_bounds = int(np.sum(cr.u < cr.l))
        if bad_bounds:
            _log.warning("CHECK WARN: %d edges have u < l", bad_bounds)
            ok = False
        else:
            _log.info("CHECK OK: u >= l for all %d edges", acc.n_edges)

        # 7. Unit coherence
        if G.nnz > 0 and "type" in rm.columns:
            edge_val = acc.edge_index["valuation"].values
            types_arr = rm["type"].to_numpy()
            checked = 0
            for row_id in range(G.n_constraints):
                if types_arr[row_id] in MIXED_VALUATION_OK:
                    continue
                cols = G.indices[G.indptr[row_id]: G.indptr[row_id + 1]]
                if cols.size == 0:
                    continue
                row_vals = set(edge_val[cols].tolist())
                if len(row_vals) > 1:
                    cid = rm.iloc[row_id]["constraint_id"]
                    raise ConstraintUnitMixingError(
                        f"G row {row_id} (type={types_arr[row_id]!r}, "
                        f"constraint_id={cid!r}) mixes valuations {sorted(row_vals)}."
                    )
                checked += 1
            _log.info("CHECK OK: unit-coherence verified on %d rows "
                      "(whitelist: %s)", checked, sorted(MIXED_VALUATION_OK))

        _log.info("Constraint check %s", "PASSED" if ok else "FAILED (see warnings above)")

    def run(
        self,
        agg_result,
        timesteps:    Optional[list] = None,
        agg_id:       str = "",
        run_id:       Optional[str] = None,
        label:        str = "",
        persist:      bool = True,
        diag_plot:    bool = False,
        parents:      Optional[dict] = None,
        stage_config: Optional[dict] = None,
    ) -> ConstraintRun:
        """
        Build the structural StructG once, then assemble per-period
        coefficient/RHS/activation overlays for each timestep.

        Parameters
        ----------
        agg_result  : AggregationResult with 2-D ``a_agg`` and ``sigma_agg``
        timesteps   : ordered list of time labels. Defaults to
                      ``acc.time_range``.
        """
        if not agg_id:
            agg_id = getattr(agg_result, "_imp_run_id", None) or ""
        acc        = self.acc_struct
        n_edges    = acc.n_edges
        a_agg      = agg_result.a_agg
        sigma_agg  = agg_result.sigma_agg

        if timesteps is None:
            timesteps = list(acc.time_range)
        time_range_str = [str(t) for t in acc.time_range]
        n_timesteps = len(timesteps)

        # ── 0. NaN precondition ───────────────────────────────────────
        if np.isnan(a_agg).any() or np.isnan(sigma_agg).any():
            n_nan_f = int(np.isnan(a_agg).sum())
            n_nan_s = int(np.isnan(sigma_agg).sum())
            raise ValueError(
                f"agg_result contains NaN (a_agg: {n_nan_f}, sigma_agg: {n_nan_s}). "
                f"Register FlatPriorImputer last in the imputation pipeline "
                f"(or halt upstream) to ensure a fully-defined prior before "
                f"constraint building."
            )

        # ── 0b. σ_a prior Lp regularization ───────────────────────────
        if self._sigma_a_cv_min > 0 or self._sigma_a_abs_floor > 0:
            sigma_agg, n_cv, n_abs = _lp_regularize(
                sigma_agg, a_agg,
                self._sigma_a_cv_min, self._sigma_a_abs_floor, self._p,
            )
            _log.info(
                "σ_a Lp regularization (p=%.2f): %d cells CV-lifted "
                "(CV_min=%.3f), %d cells abs-lifted (abs_floor=%.0e)",
                self._p, n_cv, n_abs,
                self._sigma_a_cv_min, self._sigma_a_abs_floor,
            )

        # ── 1. Build static blocks (structural) ───────────────────────
        static_blocks = [b.build_static(acc) for b in self._builders]
        # Track each builder's static row offset + nnz offset.
        offsets: list[tuple[int, int]] = []   # (row_offset, nnz_offset)
        row_off = 0
        nnz_off = 0
        for sb in static_blocks:
            offsets.append((row_off, nnz_off))
            row_off += sb.G_rows.shape[0]
            nnz_off += sb.G_rows.nnz

        static_full = _concat_blocks(static_blocks, n_edges)
        if static_full.G_rows.shape[0] == 0:
            G_struct = StructG(
                indptr   = np.zeros(1, dtype=np.int32),
                indices  = np.empty(0, dtype=np.int32),
                data     = np.empty((0, n_timesteps), dtype=float),
                shape    = (0, n_edges),
                row_meta = static_full.row_meta,
            )
            c_2d       = np.empty((0, n_timesteps))
            sigma_c_2d = np.empty((0, n_timesteps))
            active_2d  = np.empty((0, n_timesteps), dtype=bool)
            _log.info("ConstraintEngine: empty run (no constraints registered)")
        else:
            G_struct = StructG.from_static(
                static_full.G_rows, static_full.row_meta, n_timesteps,
            )
            M = G_struct.n_constraints
            c_2d       = np.tile(static_full.c[:, None], (1, n_timesteps))
            sigma_c_2d = np.tile(static_full.sigma_c[:, None], (1, n_timesteps))
            active_2d  = np.tile(static_full.active[:, None], (1, n_timesteps))
            _log.info("ConstraintEngine: StructG shape (%d, %d), %d nnz, %d timesteps",
                      M, n_edges, G_struct.nnz, n_timesteps)

        # ── 2. Per-period overlay loop ────────────────────────────────
        for t in timesteps:
            t_label = str(t)
            if t_label not in time_range_str:
                raise ValueError(
                    f"ConstraintEngine.run: timestep {t_label!r} not in "
                    f"acc.time_range={time_range_str}"
                )
            t_idx = time_range_str.index(t_label)

            a_slice  = a_agg[:, t_idx]
            sa_slice = sigma_agg[:, t_idx]

            for b, sb, (r_off, n_off) in zip(self._builders, static_blocks, offsets):
                if sb.G_rows.shape[0] == 0:
                    continue
                dyn = b.build_dynamic(acc, a_slice, sa_slice, t_label)
                bM  = sb.G_rows.shape[0]
                if dyn.G_rows.shape[0] == 0:
                    # Builder declined to refresh — keep static data + active.
                    continue
                if dyn.G_rows.shape[0] != bM:
                    raise ValueError(
                        f"build_dynamic for {type(b).__name__} returned "
                        f"{dyn.G_rows.shape[0]} rows but build_static had "
                        f"{bM} rows — sparsity invariant violated."
                    )
                dyn_csr = dyn.G_rows.tocsr()
                if dyn_csr.nnz != sb.G_rows.nnz:
                    raise ValueError(
                        f"build_dynamic for {type(b).__name__} returned "
                        f"{dyn_csr.nnz} nnz but build_static had {sb.G_rows.nnz} "
                        f"— sparsity invariant violated."
                    )
                # Write per-period coefficients into the operator's data tape.
                slot = slice(n_off, n_off + sb.G_rows.nnz)
                G_struct.data[slot, t_idx] = dyn_csr.data
                row_slot = slice(r_off, r_off + bM)
                c_2d[row_slot, t_idx]       = dyn.c
                sigma_c_2d[row_slot, t_idx] = dyn.sigma_c
                active_2d[row_slot, t_idx]  = dyn.active

        # ── 2b. σ_c prior Lp regularization (free-cone rows only) ─────
        if G_struct.n_constraints > 0:
            free_mask = (G_struct.row_meta["cone_type"].to_numpy() == "free")
            if free_mask.any() and (
                self._sigma_c_cv_min > 0 or self._sigma_c_abs_floor > 0
            ):
                n_cv_total, n_abs_total = 0, 0
                for ti in range(n_timesteps):
                    sc_free, n_cv, n_abs = _lp_regularize(
                        sigma_c_2d[free_mask, ti], c_2d[free_mask, ti],
                        self._sigma_c_cv_min, self._sigma_c_abs_floor, self._p,
                    )
                    sigma_c_2d[free_mask, ti] = sc_free
                    n_cv_total  += n_cv
                    n_abs_total += n_abs
                _log.info(
                    "σ_c Lp regularization (p=%.2f, %d free rows × %d timesteps): "
                    "%d (row, t) entries CV-lifted (CV_min=%.3f), %d abs-lifted "
                    "(abs_floor=%.0e)",
                    self._p, int(free_mask.sum()), n_timesteps,
                    n_cv_total, n_abs_total,
                    self._sigma_c_cv_min, self._sigma_c_abs_floor,
                )

        # ── 3. Assemble ConstraintRun ──────────────────────────────────
        if run_id is None:
            run_id = make_run_id("const")

        cr = ConstraintRun(
            G          = G_struct,
            row_meta   = G_struct.row_meta,
            a0         = a_agg[:, 0].copy(),                  # 1995 seed
            sigma_a    = sigma_agg.copy(),                    # (n_edges, n_timesteps)
            u          = acc.bounds.u.copy(),
            l          = acc.bounds.l.copy(),
            c          = c_2d,
            sigma_c    = sigma_c_2d,
            active     = active_2d,
            timesteps  = [str(t) for t in timesteps],
            run_id     = run_id,
            acc_id     = acc.id,
            agg_id     = agg_id,
            created_at = datetime.utcnow().isoformat(timespec="seconds") + "Z",
            label      = label,
            _p         = self._p,
        )

        self._check(cr)

        if persist:
            self._registry.register(cr, parents=parents, stage_config=stage_config)

        if diag_plot:
            from ..diag import ConstDiagnostic
            odir = os.path.join(self._registry.storage_dir, cr.run_id, "diag")
            ConstDiagnostic(cr, label=cr.run_id).plot(odir=odir)

        return cr
