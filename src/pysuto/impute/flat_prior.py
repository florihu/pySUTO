"""
flat_prior.py
=============
``FlatPriorImputer`` — final-fallback flat prior (f=0, large σ) for residual NaN.

Module-local :func:`_build_group_ids` (single class user — :class:`FlatPriorImputer`)
lives here too.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from pysuto.logging import get_logger
from pysuto.param_relation import expand_edge_index

from .base import BaseImputer, _log

logger = get_logger(__name__)


def _build_group_ids(keys: list[tuple]) -> tuple[np.ndarray, int]:
    """
    Hash a list of per-edge tuple keys to dense integer group ids.

    Returns ``(group_id, n_groups)`` where ``group_id[i]`` is the
    zero-based id of edge i's group. Order of first occurrence determines
    id assignment, which keeps results deterministic.
    """
    group_id = np.empty(len(keys), dtype=np.int64)
    seen: dict = {}
    for i, k in enumerate(keys):
        g = seen.get(k)
        if g is None:
            g = len(seen)
            seen[k] = g
        group_id[i] = g
    return group_id, len(seen)


class FlatPriorImputer(BaseImputer):
    """
    Final-fallback imputer: fills any residual-NaN edges with a flat prior
    sized by an accounting-mask CV × group scale, with two refinements
    introduced after `REP/analysis_optim_const_copper_…_20260513_112419.md`.

        f_j     = 0
        σ_j     = CV(j) · scale(j)

    **CV(j) — coverage-aware, per-mask interpolation.**
    Each mask (e.g. ``supply``, ``use`` — any name present in
    ``acc_struct.masks``) carries a pair ``(cv_lo, cv_hi)``. For a given edge
    j the CV is interpolated by the coverage density of its
    ``(sector_from, region_from)`` neighbourhood::

        ρ(j)   = fraction of edges in (sector_from, region_from) with
                 ``agg_result.coverage == True``
        CV(j)  = cv_lo[mask(j)] + (cv_hi[mask(j)] - cv_lo[mask(j)]) · (1 - ρ(j))

    Data-rich neighbourhoods (ρ ≈ 1) get the tight CV; data-poor
    neighbourhoods (ρ ≈ 0) get the loose CV. Masks are resolved in
    insertion order, first-match-wins; unmatched edges receive
    ``default_cv`` with no density modulation. If no coverage array has
    been attached at impute time (``attach_agg_result`` was not called),
    every edge is treated as fully covered (ρ = 1, CV = cv_lo) and a
    warning is logged.

    **scale(j) — 3-level cascade.**
    Cascading fallback so an edge whose narrow group has no observed flow
    still receives a meaningful scale (the prior single-level rule
    collapsed σ to 0 on ~40 k copper edges, which then pinned them at
    σ_a = 1e-6 via the constraints-side floor)::

        level 1: mean(|a_agg|) over (sector_from, entity_from,
                                    region_from, valuation)
        level 2: mean(|a_agg|) over (sector_from, valuation)
                 — used when level 1 is empty
        level 3: median(|a_agg|) over finite, non-zero entries globally
                 — used when both narrower groups are empty;
                   falls back to 1.0 only if the whole array is empty.

    Only finite, non-zero |a_agg| values are counted at every level (so
    upstream-imputed zeros don't pull the scale down to zero).

    Register LAST in the pipeline; ``ConstraintEngine.run()`` will refuse a
    NaN-bearing agg_result without it.
    """

    def __init__(
        self,
        cv_by_mask: dict[str, tuple[float, float]],
        default_cv: float,
        acc_struct,
        p: float = 1.3,
        masks: Optional[dict[str, np.ndarray]] = None,
    ) -> None:
        """
        Parameters
        ----------
        cv_by_mask : dict[str, tuple[float, float]]
            Mapping ``mask_name → (cv_lo, cv_hi)``. Insertion order encodes
            priority (first-matching mask wins). For backward compatibility
            a scalar ``cv`` is accepted and treated as ``(cv, cv)`` (no
            density modulation). Each name must be resolvable either in
            ``masks`` (caller-supplied overlay) or ``acc_struct.masks``.
            Every ``cv_hi`` must be ``<= 1.0``.
        default_cv : float
            CV applied to edges that match no listed mask. Used as a single
            value, no density modulation. Must be ``<= 1.0``.
        acc_struct : AccountingStructure
            Provides ``acc_struct.masks`` (named bool arrays) and the edge /
            node index DataFrames for group-key construction.
        p : float
            Lp exponent for downstream uncertainty propagation (recorded
            in ``config()``; not used inside this imputer directly).
        masks : dict[str, np.ndarray] | None
            Optional overlay of caller-supplied boolean masks (shape
            ``(n_edges,)``). Looked up before ``acc_struct.masks`` when
            resolving names in ``cv_by_mask``. Use to inject masks not
            registered on the accounting structure (e.g. row-specific
            sub-masks computed in the init script).
        """
        # Normalise scalar CV → (lo, hi) so back-compat callers still work.
        self.cv_by_mask: dict[str, tuple[float, float]] = {}
        for name, val in cv_by_mask.items():
            if isinstance(val, (tuple, list)):
                lo, hi = float(val[0]), float(val[1])
            else:
                lo = hi = float(val)
            self.cv_by_mask[name] = (lo, hi)

        for name, (_lo, hi) in self.cv_by_mask.items():
            if hi > 1.0:
                raise ValueError(
                    f"FlatPriorImputer: cv_hi for mask {name!r} is {hi} > 1.0"
                )

        self.default_cv = float(default_cv)
        if self.default_cv > 1.0:
            raise ValueError(
                f"FlatPriorImputer: default_cv is {self.default_cv} > 1.0"
            )

        self.p = float(p)
        self._source_path: Optional[str] = None
        # agg_result.coverage attached by ImputationEngine.run via
        # attach_agg_result() before impute() is called.
        self._coverage: Optional[np.ndarray] = None

        acc_masks: dict = getattr(acc_struct, "masks", None) or {}
        overlay: dict = masks or {}

        # Pre-resolve mask name → bool ndarray (length n_edges) once.
        # Overlay takes precedence over acc_struct.masks.
        self._masks: list[tuple[str, np.ndarray]] = []
        for name in self.cv_by_mask:  # insertion order
            mask = overlay.get(name)
            if mask is None:
                mask = acc_masks.get(name)
            if mask is None:
                raise KeyError(
                    f"No mask {name!r} in overlay or acc_struct.masks. "
                    f"Available (overlay): {sorted(overlay)}; "
                    f"available (acc): {sorted(acc_masks)}"
                )
            self._masks.append((name, np.asarray(mask, dtype=bool)))

        self._n_edges = int(acc_struct.n_edges)

        # Pre-compute three integer group-id arrays per edge:
        #   level 1: (sector_from, entity_from, region_from, valuation) — narrow scale group
        #   level 2: (sector_from, valuation)                           — broader scale fallback
        #   density: (sector_from, region_from)                         — coverage-density neighbourhood
        ei_exp = expand_edge_index(acc_struct.edge_index, acc_struct.node_index)
        if len(ei_exp) != self._n_edges:
            raise ValueError(
                f"acc_struct.n_edges={self._n_edges} mismatches edge_index length={len(ei_exp)}"
            )

        sec_from = ei_exp["sector_from"].astype(str).to_numpy()
        ent_from = ei_exp["entity_from"].astype(str).to_numpy()
        reg_from = ei_exp["region_from"].astype(str).to_numpy()
        val_arr  = ei_exp["valuation"].astype(str).to_numpy()

        self._group_id,         self._n_groups        = _build_group_ids(
            list(zip(sec_from, ent_from, reg_from, val_arr))
        )
        self._group_id_level2,  self._n_groups_level2 = _build_group_ids(
            list(zip(sec_from, val_arr))
        )
        self._density_group_id, self._n_density_groups = _build_group_ids(
            list(zip(sec_from, reg_from))
        )

    def attach_agg_result(self, agg_result) -> None:
        """
        Receive a reference to the current `AggregationResult` so the
        imputer can read per-cell ``coverage`` for density-aware CV.

        Called by ``ImputationEngine.run`` before each step. Other imputers
        without this method are simply skipped by the engine.
        """
        cov = getattr(agg_result, "coverage", None)
        if cov is None:
            self._coverage = None
            return
        cov = np.asarray(cov, dtype=bool)
        if cov.ndim != 2 or cov.shape[0] != self._n_edges:
            raise ValueError(
                f"agg_result.coverage shape {cov.shape} "
                f"!= (n_edges={self._n_edges}, n_timesteps)"
            )
        self._coverage = cov

    @property
    def source_paths(self) -> list[str]:
        return [self._source_path] if self._source_path else []

    def config(self) -> dict:
        return {
            "cv_by_mask":  {k: list(v) for k, v in self.cv_by_mask.items()},
            "default_cv":  self.default_cv,
            "p":           self.p,
            "source_path": self._source_path,
        }

    def _coverage_density(self) -> np.ndarray:
        """
        Per-edge fraction of ``coverage == True`` cells over the edge's
        (sector_from, region_from) neighbourhood. The 2-D coverage
        ``(n_edges, n_timesteps)`` contributes every cell to the
        neighbourhood's running total.

        Returns an all-ones array (and logs a warning) when no coverage
        has been attached, so downstream code degrades gracefully to the
        tight ``cv_lo`` end of the range.
        """
        if self._coverage is None:
            logger.warning(
                "FlatPriorImputer: no coverage attached; "
                "treating every edge as fully covered (ρ=1, CV=cv_lo). "
                "ImputationEngine should call attach_agg_result() before impute()."
            )
            return np.ones(self._n_edges, dtype=float)

        cov_flat = self._coverage.astype(float).ravel()
        gid_flat = np.broadcast_to(
            self._density_group_id[:, None], self._coverage.shape,
        ).ravel()

        sums   = np.bincount(gid_flat, weights=cov_flat,
                             minlength=self._n_density_groups)
        counts = np.bincount(gid_flat,
                             minlength=self._n_density_groups).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            density_per_group = np.where(counts > 0, sums / counts, 1.0)
        return density_per_group[self._density_group_id]

    def _resolve_cv(self) -> np.ndarray:
        """
        Per-edge CV vector with two layers:

        1.  First-match-wins assignment of the edge's mask, using
            ``(cv_lo, cv_hi)`` for that mask.
        2.  Density-aware interpolation within the masked range::

                CV(j) = cv_lo + (cv_hi − cv_lo) · (1 − ρ(j))

            where ρ(j) is computed by ``_coverage_density``.

        Edges that match no listed mask receive ``self.default_cv``
        unchanged (single value, no density modulation).
        """
        cv       = np.full(self._n_edges, self.default_cv, dtype=float)
        assigned = np.zeros(self._n_edges, dtype=bool)
        density  = self._coverage_density()  # ρ(j) in [0, 1]

        for name, mask in self._masks:
            cv_lo, cv_hi = self.cv_by_mask[name]
            new = mask & ~assigned
            if not new.any():
                continue
            cv[new] = cv_lo + (cv_hi - cv_lo) * (1.0 - density[new])
            assigned |= new

        return cv

    def _resolve_scale(self, f: np.ndarray) -> np.ndarray:
        """
        Per-edge scale via 3-level cascade — see class docstring for the
        full rule. Only finite, non-zero ``|f|`` values contribute to any
        level; upstream-imputed zeros never pull the scale down to zero.

        ``f`` is 2-D ``(n_edges, n_timesteps)``; every cell of the row
        contributes to its edge's group at level 1 and level 2.
        """
        n_edges, n_timeps = f.shape
        finite_nonzero = np.isfinite(f) & (f != 0.0)
        abs_f          = np.where(finite_nonzero, np.abs(f), 0.0)
        weight         = finite_nonzero.astype(float)

        # Broadcast per-edge group_id across time → flat (n_edges*n_timesteps,)
        gid1_flat = np.broadcast_to(self._group_id[:, None], f.shape).ravel()
        gid2_flat = np.broadcast_to(self._group_id_level2[:, None], f.shape).ravel()
        abs_f_flat = abs_f.ravel()
        wt_flat    = weight.ravel()

        # Level 1 — narrow group
        sums1   = np.bincount(gid1_flat, weights=abs_f_flat, minlength=self._n_groups)
        counts1 = np.bincount(gid1_flat, weights=wt_flat,    minlength=self._n_groups)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean1 = np.where(counts1 > 0, sums1 / counts1, 0.0)

        # Level 2 — broader (sector_from, valuation) group
        sums2   = np.bincount(gid2_flat, weights=abs_f_flat, minlength=self._n_groups_level2)
        counts2 = np.bincount(gid2_flat, weights=wt_flat,    minlength=self._n_groups_level2)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean2 = np.where(counts2 > 0, sums2 / counts2, 0.0)

        # Level 3 — global median of finite, non-zero |f|
        if finite_nonzero.any():
            global_scale = float(np.median(abs_f[finite_nonzero]))
        else:
            global_scale = 1.0  # whole array empty → trivial unit scale

        scale_l1 = mean1[self._group_id]            # (n_edges,)
        scale_l2 = mean2[self._group_id_level2]     # (n_edges,)

        scale = np.where(scale_l1 > 0, scale_l1,
                         np.where(scale_l2 > 0, scale_l2, global_scale))
        return scale

    def impute(
        self,
        f: np.ndarray,
        sigma: np.ndarray,
        missing: np.ndarray,
        step: int,
        acc_struct,
    ) -> list[dict]:
        # f, sigma, missing all have shape (n_edges, n_timesteps).
        edge_idx, t_idx = np.where(missing)
        if edge_idx.size == 0:
            return []

        cv_per_edge    = self._resolve_cv()        # (n_edges,)
        scale_per_edge = self._resolve_scale(f)    # (n_edges,)
        sigma_target   = cv_per_edge * scale_per_edge  # (n_edges,)

        f[edge_idx, t_idx]     = 0.0
        sigma[edge_idx, t_idx] = sigma_target[edge_idx]

        desc = "no data, no imputer matched: flat prior (cv·scale)"
        return [
            _log(int(edge_idx[i]), int(t_idx[i]), step, "flat_prior", [], [],
                 0.0, float(sigma_target[edge_idx[i]]), desc, None)
            for i in range(edge_idx.size)
        ]


# ---------------------------------------------------------------------------
# Shared log-entry constructor
# ---------------------------------------------------------------------------

