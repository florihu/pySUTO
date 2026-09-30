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
from .naming import TYPE_MASSBAL, TYPE_RATIO
from .run  import ConstraintRun

_log = get_logger(__name__)


# G rows of these types are allowed to span multiple valuations (ratio rows
# bridge Mass<->CC via a Cu-grade coefficient r). Every other type must
# contain edges of exactly one valuation — the structural unit-coherence
# guarantee on which the σ-weighted QP relies.
MIXED_VALUATION_OK = frozenset({TYPE_RATIO})


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


def _partition_G_by_time(
    G:          sp.csr_matrix,
    row_meta:   pd.DataFrame,
    edge_index: pd.DataFrame,
    timesteps:  list,
) -> tuple[dict, dict]:
    """
    Slice the monolithic constraint matrix G into one block per timestep.

    Returns
    -------
    G_blocks       : dict t → sp.csr_matrix of shape (rows_for_t, cols_for_t)
    edge_cols_by_t : dict t → np.ndarray of global edge_ids for t

    Safety: each row's non-zero columns must lie entirely within its own time
    bucket — this is the "no cross-time coupling" invariant. Violations raise
    ValueError with the offending row id.
    """
    if "time" in row_meta.columns and len(row_meta):
        row_time = row_meta["time"].astype(str).to_numpy()
    else:
        row_time = np.array([], dtype=str)
    col_time = edge_index["time"].astype(str).to_numpy()
    G_csr    = G.tocsr() if not sp.isspmatrix_csr(G) else G

    G_blocks:       dict[str, sp.csr_matrix] = {}
    edge_cols_by_t: dict[str, np.ndarray]    = {}

    for t in timesteps:
        t_str  = str(t)
        rows_t = np.where(row_time == t_str)[0]
        cols_t = np.where(col_time == t_str)[0]

        G_sliced = G_csr[rows_t][:, cols_t].tocsr()

        if rows_t.size:
            full_row_slice = G_csr[rows_t]
            cross_time_nnz = full_row_slice.nnz - G_sliced.nnz
            if cross_time_nnz:
                bad_rows_global = rows_t[
                    np.diff(full_row_slice.indptr) != np.diff(G_sliced.indptr)
                ]
                raise ValueError(
                    f"_partition_G_by_time: {cross_time_nnz} non-zero entries "
                    f"lie outside the time-{t_str} column bucket "
                    f"(offending global rows: {bad_rows_global[:5].tolist()}...). "
                    f"This violates the 'no cross-time coupling' invariant."
                )

        G_blocks[t_str]       = G_sliced
        edge_cols_by_t[t_str] = cols_t

    return G_blocks, edge_cols_by_t


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


def _validate_ratio_bounds(G: sp.spmatrix, row_meta: pd.DataFrame) -> None:
    """Reject hard ratio bounds that pin an edge-pair to an empty interval.

    Each per-pair hard ``ratio``/``nonneg`` row encodes one side of a band on
    ``R = num/den`` (``Gx ≤ 0``): a ``+1`` on the numerator with ``-hi`` on the
    denominator means ``R ≤ hi``; a ``-1`` with ``+lo`` means ``R ≥ lo``. When
    several params bound the *same* ``(num, den)`` pair (e.g. two scrap conc
    "classes"), the effective band is ``[max(lo), min(hi)]``; if that is empty
    the only feasible flow is 0 — a silent collapse. Catch it at build time.
    Scoped to per-pair (2-nonzero) rows; set-scoped (``agg``) rows are skipped.
    """
    if not len(row_meta):
        return
    mask = (
        (row_meta["type"].to_numpy() == "ratio")
        & (row_meta["cone_type"].to_numpy() == "nonneg")
        & row_meta["is_hard"].to_numpy()
    )
    rows = np.where(mask)[0]
    if rows.size == 0:
        return
    G_csr = G.tocsr() if not sp.isspmatrix_csr(G) else G
    variant = row_meta["variant"].to_numpy()
    # (num_edge, den_edge) -> [lo, hi, set(variants)]
    bands: dict[tuple[int, int], list] = {}
    for ri in rows:
        cols = G_csr.indices[G_csr.indptr[ri]: G_csr.indptr[ri + 1]]
        vals = G_csr.data[G_csr.indptr[ri]: G_csr.indptr[ri + 1]]
        if cols.size != 2:
            continue  # set-scoped / pooled row — out of scope
        # numerator carries the ±1; denominator carries ∓bound
        if np.isclose(abs(vals[0]), 1.0):
            (num, cnum), (den, cden) = (cols[0], vals[0]), (cols[1], vals[1])
        elif np.isclose(abs(vals[1]), 1.0):
            (num, cnum), (den, cden) = (cols[1], vals[1]), (cols[0], vals[0])
        else:
            continue
        key = (int(num), int(den))
        b = bands.setdefault(key, [-np.inf, np.inf, set()])
        if cnum > 0:                       # R ≤ hi
            b[1] = min(b[1], -cden)
        else:                              # R ≥ lo
            b[0] = max(b[0], cden)
        b[2].add(str(variant[ri]))
    empty = {k: v for k, v in bands.items() if v[0] > v[1] + 1e-12}
    if empty:
        ex = list(empty.items())[:5]
        detail = "; ".join(
            f"edges(num={k[0]},den={k[1]}) band=[{v[0]:.4g},{v[1]:.4g}] "
            f"from {sorted(v[2])}"
            for k, v in ex
        )
        raise ValueError(
            f"_validate_ratio_bounds: {len(empty)} edge-pair(s) carry hard ratio "
            f"bounds with an empty interval (lo>hi) → those flows are pinned to 0. "
            f"First offenders: {detail}"
        )


# =============================================================================
# BaseConstraintBuilder
# =============================================================================


def nodata_exempt_mask(acc, exclude_sectors, data_aware_sectors, region_years) -> np.ndarray:
    """Per-edge exemption from the no-data σ anchor (ADR 0013 amendment 3).

    An edge touching ``exclude_sectors`` (either endpoint) is spared, except that
    one touching ``data_aware_sectors`` is spared only where its
    ``(from_region, time)`` is in ``region_years`` — the region-years in which a
    relation actually sets its magnitude. Elsewhere a zero prior means "no flow",
    and the anchor applies. ``region_years`` holds ``(region, time)`` string pairs.
    """
    em = acc.edge_meta
    fs, ts = em["from_sector"].to_numpy(), em["to_sector"].to_numpy()
    spared = np.isin(fs, list(exclude_sectors)) | np.isin(ts, list(exclude_sectors))
    aware = np.isin(fs, list(data_aware_sectors)) | np.isin(ts, list(data_aware_sectors))
    keys = zip(em["from_region"].astype(str), em["time"].astype(str))
    listed = np.fromiter((k in region_years for k in keys), dtype=bool, count=len(em))
    return spared & (~aware | listed)


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
        prior_box_k: Optional[float] = None,
        nodata_anchor_cv: float = 0.3,
        nodata_anchor_exclude_sectors: tuple = ("CO", "Mining", "Ore", "WR", "Tailings"),
        nodata_anchor_flat_threshold: float = 1e5,
        nodata_anchor_exempt_mask: Optional[np.ndarray] = None,
        pin_flat_prior_zeros: bool = False,
        pin_exclude_sectors: tuple = ("CO", "Mining", "Ore", "WR", "Tailings", "Slag", "PSC"),
    ):
        self.acc_struct  = acc_struct
        self._builders:  list[BaseConstraintBuilder] = []
        self._registry   = ConstraintRunRegistry(storage_dir)
        self._sigma_a_cv_min    = float(sigma_a_cv_min)
        self._sigma_a_abs_floor = float(sigma_a_abs_floor)
        self._sigma_c_cv_min    = float(sigma_c_cv_min)
        self._sigma_c_abs_floor = float(sigma_c_abs_floor)
        self._p                 = float(p)
        self._prior_box_k       = float(prior_box_k) if prior_box_k else None
        self._nodata_anchor_cv  = float(nodata_anchor_cv)
        self._nodata_anchor_exclude = tuple(nodata_anchor_exclude_sectors)
        self._nodata_anchor_flat_threshold = float(nodata_anchor_flat_threshold)
        # Per-edge exemption from the no-data anchor (ADR 0013 amendment 3); None
        # keeps the sector rule of nodata_anchor_exclude_sectors.
        self._nodata_anchor_exempt_mask = (
            None if nodata_anchor_exempt_mask is None
            else np.asarray(nodata_anchor_exempt_mask, dtype=bool))
        self._pin_flat_prior_zeros = bool(pin_flat_prior_zeros)
        self._pin_exclude_sectors  = tuple(pin_exclude_sectors)

    def _apply_prior_box(self, l, u, a0, sigma_a):
        """Tighten flow bounds to ``a₀ ± k·σ_a`` on edges with a real prior.

        Grafts the RAS scale-floor onto the QP box (Lenzen 2009: the balanced
        estimate can never go more than ~kσ below ``a₀``), preventing the
        homogeneous collapse to ``l`` while keeping the σ_a-weighted quadratic
        objective intact. Edges with ``a₀ = 0`` (FlatPrior — no prior
        information) keep their original ``[l, u]`` so the box does not pin
        genuinely-unknown edges to zero. The box never *widens* an existing
        bound; where a tighter box would invert a single edge (``u < l``, e.g.
        a prior that already violates its own sign bound) that edge reverts to
        its original ``[l, u]`` so only the meaningful *cross-constraint*
        infeasibility (e.g. an incoherent ratio pair) surfaces, not spurious
        single-edge empties. Returns new ``(l, u)``.
        """
        k = self._prior_box_k
        if not k or k <= 0:
            return l, u
        has_prior = a0 != 0.0
        lo = a0 - k * sigma_a
        hi = a0 + k * sigma_a
        l_new, u_new = l.copy(), u.copy()
        l_new[has_prior] = np.maximum(l[has_prior], lo[has_prior])
        u_new[has_prior] = np.minimum(u[has_prior], hi[has_prior])
        conflict = u_new < l_new
        l_new[conflict] = l[conflict]
        u_new[conflict] = u[conflict]
        _log.info(
            "Prior box (k=%.2f): tightened bounds on %d/%d edges with a prior "
            "(%d reverted to avoid single-edge u<l)",
            k, int(has_prior.sum() - conflict.sum()), int(has_prior.sum()),
            int(conflict.sum()),
        )
        return l_new, u_new

    def _nodata_target_mask(self, a0, sigma_raw, acc, exclude_sectors, exempt_mask=None):
        """Boolean mask of no-data ("flat-prior") edges outside an exclusion set.

        A no-data edge has ``|a₀| < 1e-9`` and σ still at the FlatPrior
        global-median fallback (``σ ≥ nodata_anchor_flat_threshold``), i.e. it
        had no feed coverage. ``exclude_sectors`` spares edges whose magnitude is
        **structurally determined by a fixed ratio / mass balance rather than by
        feed data** — the extraction chain and the SEM residual outputs — for
        which ``a₀ = 0`` means "value comes from the relation", not "no flow".
        Computed on the *raw* σ (before Lp regularization / the no-data anchor,
        which overwrite σ on exactly these edges). Shared by the soft σ anchor
        and the hard flat-prior-zero pin (different exclusion sets). A per-edge
        ``exempt_mask`` replaces the sector rule (ADR 0013 amendment 3: the
        extraction chain is spared only where a relation sets its magnitude).
        See ADR 0013.
        """
        if exempt_mask is not None:
            if exempt_mask.shape != (len(a0),):
                raise ValueError(
                    f"no-data anchor exempt mask has shape {exempt_mask.shape}, "
                    f"expected ({len(a0)},)")
            is_excluded = exempt_mask
        else:
            em          = acc.edge_meta
            from_sector = em["from_sector"].to_numpy()
            to_sector   = em["to_sector"].to_numpy()
            excl        = np.asarray(exclude_sectors, dtype=object)
            is_excluded = np.isin(from_sector, excl) | np.isin(to_sector, excl)
        nodata      = (np.abs(a0) < 1e-9) & (sigma_raw >= self._nodata_anchor_flat_threshold)
        return nodata & ~is_excluded

    def _anchor_nodata_sigma(self, a0, sigma_a, acc):
        """Soft magnitude-anchor for no-data ("flat-prior") edges.

        Edges with no feed coverage are filled by ``FlatPriorImputer`` with
        ``a₀ = 0`` and a σ whose scale cascade falls back to the *global* flow
        median (≈1e6). That σ is a non-anchor: the prior pull ``(F−0)²/σ² ≈ 0``,
        so the solver uses such edges as free slack and they soak up mass pulled
        through data-free subgraphs (e.g. scrap on the Taiwan/Panama CFR nodes).

        For every no-data edge *outside the extraction front-end* this replaces
        the σ with a sector-appropriate scale
        ``σ = max(abs_floor, cv · median|a₀| of anchored siblings in the same
        (from_sector, valuation))``. The prior mean stays 0, so the now-meaningful
        σ pulls the edge toward its zero seed; a hard mass balance can still force
        genuine flow through at a finite penalty cost (soft anchor, no hard
        infeasibility). The extraction sectors (``Mining/Ore/CO/WR/Tailings``) are
        left untouched because their magnitude is set by a *ratio / mass-balance
        relation* (the stripping ratio ``WR = stripping_ratio·Ore``, the
        CC-conserved ``CO→Ore`` chain), not by the edge's own ``a₀`` (which is 0);
        anchoring it toward zero would fight the relation. Returns a new
        ``sigma_a`` array.
        """
        cv = self._nodata_anchor_cv
        if cv <= 0:
            return sigma_a

        em          = acc.edge_meta
        from_sector = em["from_sector"].to_numpy()
        valuation   = em["valuation"].to_numpy()

        target = self._nodata_target_mask(a0, sigma_a, acc, self._nodata_anchor_exclude,
                                          exempt_mask=self._nodata_anchor_exempt_mask)
        if not target.any():
            _log.info("no-data σ_a anchor (cv=%.2f): 0 edges matched", cv)
            return sigma_a

        # Reference scale per (from_sector, valuation): median |a₀| over the
        # *anchored* siblings (a₀ != 0) in that group. Verified always defined
        # (every no-data sector has anchored siblings); global anchored median
        # is the safety fallback for any empty group.
        anchored = np.abs(a0) > 1e-9
        ref_df = pd.DataFrame({
            "from_sector": from_sector[anchored],
            "valuation":   valuation[anchored],
            "abs_a0":      np.abs(a0)[anchored],
        })
        scale_ref = ref_df.groupby(["from_sector", "valuation"])["abs_a0"].median()
        global_ref = float(np.median(np.abs(a0)[anchored])) if anchored.any() else 1.0

        keys     = list(zip(from_sector[target], valuation[target]))
        ref_vals = scale_ref.reindex(keys).to_numpy()
        ref_vals = np.where(np.isfinite(ref_vals), ref_vals, global_ref)

        new_sigma = np.maximum(self._sigma_a_abs_floor, cv * ref_vals)
        sigma_a = sigma_a.copy()
        sigma_a[target] = new_sigma
        _log.info(
            "no-data σ_a anchor (cv=%.2f): %d edges anchored (excl. %s); "
            "σ_new range [%.3g, %.3g]",
            cv, int(target.sum()), ",".join(self._nodata_anchor_exclude),
            float(new_sigma.min()), float(new_sigma.max()),
        )
        return sigma_a

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

        # 4. Expected mass_balance row count: one per (balanced_node × time × valuation)
        ei, ni = acc.edge_index, acc.node_index
        t_arr  = ei["time"].values
        v_arr  = ei["valuation"].values
        expected_mb = 0
        for time in acc.time_range:
            t_mask_base = (t_arr == str(time))
            for valuation in acc.valuation:
                t_mask = t_mask_base & (v_arr == valuation)
                nodes_in  = set(ei.loc[t_mask, "to_id"].values)
                nodes_out = set(ei.loc[t_mask, "from_id"].values)
                for nid in nodes_in & nodes_out:
                    if str(ni.loc[nid, "entity"]) in {"Flow", "Process"}:
                        expected_mb += 1
        actual_mb = int(type_counts.get(TYPE_MASSBAL, 0))
        if actual_mb == expected_mb:
            _log.info("CHECK OK: mass_balance rows = %d (matches expected)", actual_mb)
        else:
            _log.warning(
                "CHECK WARN: mass_balance rows = %d, expected %d from topology",
                actual_mb, expected_mb,
            )
            ok = False

        # 4b. Ratio rows expected uniform across years. Each ratio template
        # (soft ratio_eq → free cone, hard ratio_bound → nonneg cone) in the
        # params CSV expands to one row per matching (time, valuation), so
        # totals should split evenly across the timesteps in scope. A
        # non-uniform breakdown means some year is missing rows (often: source
        # feed has no coverage that year), which is the same class of bug that
        # motivates per-t deactivation. Both collapse to type=="ratio"; the
        # cone separates eq (free) from bound (nonneg).
        n_years_expected = len(list(acc.time_range))
        for type_name, cone in (("ratio_eq", "free"), ("ratio_bound", "nonneg")):
            type_mask = (
                (rm["type"].to_numpy() == TYPE_RATIO) & (rm["cone_type"].to_numpy() == cone)
                if len(rm) else np.array([], dtype=bool)
            )
            n_type = int(type_mask.sum())
            if n_type == 0 or n_years_expected == 0:
                continue
            per_year = (
                rm.loc[type_mask, "time"].astype(str).value_counts().to_dict()
            )
            expected_years = {str(t) for t in acc.time_range}
            missing_years  = sorted(expected_years - set(per_year))
            extra_years    = sorted(set(per_year) - expected_years)

            if n_type % n_years_expected != 0:
                _log.warning(
                    "CHECK WARN: %s rows = %d not evenly divisible by %d years "
                    "(per-year: %s)",
                    type_name, n_type, n_years_expected,
                    {t: per_year[t] for t in sorted(per_year)},
                )
                ok = False
                continue

            expected_per_year = n_type // n_years_expected
            bad_years = {
                t: c for t, c in per_year.items() if c != expected_per_year
            }
            if not bad_years and not missing_years and not extra_years:
                _log.info(
                    "CHECK OK: %s rows = %d (%d/year uniform across %d years)",
                    type_name, n_type, expected_per_year, n_years_expected,
                )
            else:
                _log.warning(
                    "CHECK WARN: %s rows = %d non-uniform across years. "
                    "Expected %d/year. Missing: %s. Extra: %s. Off-count: %s",
                    type_name, n_type, expected_per_year,
                    missing_years or "none",
                    extra_years or "none",
                    {t: c for t, c in sorted(bad_years.items())} or "none",
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
                        f"Every sum / massbal row must contain edges of "
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

        # ── 0a. Flat-prior-zero pin mask (captured on RAW σ) ──────────
        # Must be computed before Lp regularization and the no-data σ anchor,
        # which both overwrite σ on exactly these edges (the anchor drops it
        # below the 1e5 flat threshold, so a later mask would miss them). When
        # enabled, these genuine no-data leaves are pinned to [0, 0] in the flow
        # bounds below — "no feed data ⇒ no flow". The exclusion set spares
        # structurally-determined edges (extraction chain + SEM residual
        # outputs); see ADR 0013.
        pin_mask = (
            self._nodata_target_mask(f_agg, agg_result.sigma_agg, acc,
                                     self._pin_exclude_sectors)
            if self._pin_flat_prior_zeros else None
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

        # ── 0c. No-data ("flat-prior") σ_a magnitude anchor ───────────
        # Replaces the global-median σ fallback (≈1e6) on no-data edges
        # (a₀=0) outside the extraction front-end with a sector-appropriate
        # scale, so the zero prior actually pulls and these edges stop
        # absorbing mass routed through data-free subgraphs. Runs after Lp
        # regularization so the anchored σ_a flows into a0.npy/sigma_a.npy
        # and p_mean-mode σ_c propagation.
        sigma_agg = self._anchor_nodata_sigma(f_agg, sigma_agg, acc)

        # ── 1. Build static G (time-invariant) ────────────────────────
        static_blocks = [b.build_static(acc) for b in self._builders]
        static_full   = _concat_blocks(static_blocks, n_edges)
        G_static      = static_full.G_rows
        c_static      = static_full.c.copy()
        sc_static     = static_full.sigma_c.copy()
        row_meta      = static_full.row_meta.copy()
        M             = G_static.shape[0]

        _validate_ratio_bounds(G_static, row_meta)

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
                n_cv_total, self._sigma_c_cv_min,
                n_abs_total, self._sigma_c_abs_floor,
            )

        # ── 2c. Per-t partition of G ──────────────────────────────────
        # No constraint builder couples edges across t, so the monolithic G
        # is exactly block-diagonal by time. Slice rows by row_meta["time"]
        # and columns by acc.edge_index["time"]; this lossless partition is
        # what the solver and uncprop consume to skip ~96.7% of dead cols
        # at each per-t solve.
        G_blocks, edge_cols_by_t = _partition_G_by_time(
            G_static, row_meta, acc.edge_index, timesteps,
        )
        if G_blocks:
            first_t = next(iter(G_blocks))
            G0 = G_blocks[first_t]
            _log.info(
                "Per-t partition: %d timesteps × (%d rows × %d cols) blocks "
                "→ per-t solve is %.1f× smaller than the global %d × %d QP",
                len(G_blocks), G0.shape[0], G0.shape[1],
                (M * n_edges) / max(G0.shape[0] * G0.shape[1], 1),
                M, n_edges,
            )

        # ── 2d. Per-t cold prior  a_prior(t) = f_agg sliced to year-t edges ──
        # The optimiser's prior is time-variant: a_prior(t0) = f_agg (cold seed)
        # and a_prior(t>0) = a_post(t-1) under warm-start. Here we store only the
        # cold per-period seed (f_agg restricted to the year's edges, in cols_t
        # order); the warm chain is produced by the solver at solve time. Block
        # path only — the legacy full-vector path keeps using the global prior.
        for t_label, td in timestep_data.items():
            cols_t = edge_cols_by_t.get(t_label)
            if cols_t is not None:
                td["a_prior"] = f_agg[np.asarray(cols_t)].copy()

        # ── 3. Assemble ConstraintRun ──────────────────────────────────
        if run_id is None:
            run_id = make_run_id("const")

        l_box, u_box = self._apply_prior_box(
            acc.bounds.l.copy(), acc.bounds.u.copy(), f_agg, sigma_agg,
        )
        if pin_mask is not None:
            l_box[pin_mask] = 0.0
            u_box[pin_mask] = 0.0
            _log.info("Pin flat-prior zeros: %d edges pinned to [0,0]",
                      int(pin_mask.sum()))

        cr = ConstraintRun(
            G              = G_static,
            row_meta       = row_meta,
            a0             = f_agg.copy(),
            sigma_a        = sigma_agg.copy(),
            u              = u_box,
            l              = l_box,
            timestep_data  = timestep_data,
            run_id         = run_id,
            acc_id         = acc.id,
            agg_id         = agg_id,
            created_at     = datetime.utcnow().isoformat(timespec="seconds") + "Z",
            label          = label,
            _p             = self._p,
            G_blocks       = G_blocks,
            edge_cols_by_t = edge_cols_by_t,
        )

        self._check(cr)

        if persist:
            self._registry.register(cr, parents=parents, stage_config=stage_config)

        if diag_plot:
            from ..diag import ConstDiagnostic, ConstStatsDiagnostic
            odir = os.path.join(self._registry.storage_dir, cr.run_id, "diag")
            ConstDiagnostic(cr, label=cr.run_id).plot(odir=odir)
            ConstStatsDiagnostic(cr, years=("1995",)).write(odir)

        return cr
