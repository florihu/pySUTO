"""
time.py
=======
``TimeImputer`` — temporal interpolation / extrapolation per structural edge.

Under the structural model each row of ``f`` is an independent time series
for one ``(from_id, to_id, valuation)`` edge over ``acc.time_range``;
TimeImputer scans each row's columns for NaN gaps and fills them.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from pysuto.logging import get_logger
from pysuto.lp import lp_combine

from .base import BaseImputer, _log

logger = get_logger(__name__)


class TimeImputer(BaseImputer):
    """
    Temporal interpolation / extrapolation along the time axis of ``f``.

    Each structural edge (row of ``f``) is its own series; the columns
    correspond to ``acc.time_range``. Gaps are filled per ``method``.

    Parameters
    ----------
    window          Number of adjacent time steps to search (both directions
                    for "linear" and "mean"; only backward for "locf";
                    only forward for "nocb").  Override per-direction with
                    window_before / window_after.
    method          "linear"  — two-point linear interpolation.
                    "locf"    — last observation carried forward.
                    "nocb"    — next observation carried backward.
                    "mean"    — unweighted mean within window.
    min_obs         Minimum non-NaN observations required within window.
    gap_inflation   σ multiplier per time-step gap (default 1.0 = none).
    window_before   Override backward window (default = window).
    window_after    Override forward window (default = window).
    """

    def __init__(
        self,
        window: int = 5,
        method: Literal["linear", "locf", "nocb", "mean"] = "linear",
        min_obs: int = 1,
        gap_inflation: float = 1.0,
        window_before: Optional[int] = None,
        window_after:  Optional[int] = None,
        interpolate_only: bool = False,
        p: float = 1.3,
        scope_edges:      Optional[np.ndarray] = None,
        scope_feed_id:    Optional[str]        = None,
        scope_times:      Optional[list]       = None,
        scope_valuations: Optional[list]       = None,
    ) -> None:
        self.window           = window
        self.method           = method
        self.min_obs          = min_obs
        self.gap_inflation    = gap_inflation
        self.wb               = window_before if window_before is not None else window
        self.wa               = window_after  if window_after  is not None else window
        self.interpolate_only = interpolate_only
        self.p                = float(p)

        if scope_edges is not None and scope_feed_id is not None:
            raise ValueError(
                "TimeImputer: pass at most one of scope_edges, scope_feed_id."
            )
        self._scope_feed_id    = scope_feed_id
        self._scope_times      = set(map(str, scope_times)) if scope_times else None
        self._scope_valuations = set(scope_valuations) if scope_valuations else None
        # scope_edges may be a bool mask (length = N_edges) or list of edge ids;
        # resolved to a bool mask lazily in attach_agg_result / impute.
        self._scope_edges_raw  = scope_edges
        self._scope_mask: Optional[np.ndarray] = None
        if scope_edges is not None:
            arr = np.asarray(scope_edges)
            if arr.dtype == bool:
                self._scope_mask = arr

    def attach_agg_result(self, agg_result) -> None:
        """
        Receive ``agg_result`` so ``scope_feed_id`` can be resolved to a
        per-edge mask in :meth:`impute`. The mask covers every edge whose
        series saw ≥1 finite observation from the named feed across any
        timestep — so NaN target slots in those series are still in-scope
        and can receive LOCF/NOCB fills.
        """
        if self._scope_feed_id is None:
            return
        feed_ids = list(getattr(agg_result, "feed_ids", []))
        if self._scope_feed_id not in feed_ids:
            raise ValueError(
                f"TimeImputer: scope_feed_id={self._scope_feed_id!r} not in "
                f"agg_result.feed_ids={feed_ids}"
            )
        F_feed = np.asarray(agg_result.F)[feed_ids.index(self._scope_feed_id)]
        # F_feed shape: (n_edges, n_timesteps); any-finite over time → per-edge mask
        self._feed_obs_mask = np.any(np.isfinite(F_feed), axis=1)

    def impute(
        self,
        f: np.ndarray,
        sigma: np.ndarray,
        missing: np.ndarray,
        step: int,
        acc_struct,
    ) -> list[dict]:
        # f, sigma, missing all have shape (n_edges, n_timesteps).
        ei = acc_struct.edge_index
        n_edges, n_timeps = f.shape

        val_arr = ei["valuation"].to_numpy(dtype=str)
        time_range = list(getattr(acc_struct, "time_range", []))

        # Resolve scope_edges-as-int-list lazily once n_edges is known.
        if (
            self._scope_mask is None
            and self._scope_edges_raw is not None
            and self._scope_feed_id is None
        ):
            mask = np.zeros(n_edges, dtype=bool)
            mask[np.asarray(self._scope_edges_raw, dtype=int)] = True
            self._scope_mask = mask

        # Resolve scope_feed_id from the attached feed-observation mask.
        feed_obs = getattr(self, "_feed_obs_mask", None)
        if self._scope_feed_id is not None and feed_obs is not None:
            self._scope_mask = np.asarray(feed_obs, dtype=bool)

        # scope_times restricts which target columns may be filled.
        scope_tidx: Optional[set] = None
        if self._scope_times is not None:
            scope_tidx = {
                i for i, t_lbl in enumerate(time_range)
                if str(t_lbl) in self._scope_times
            }

        entries: list[dict] = []

        for e_id in range(n_edges):
            if self._scope_mask is not None and not self._scope_mask[e_id]:
                continue
            if self._scope_valuations is not None and val_arr[e_id] not in self._scope_valuations:
                continue

            row_f     = f[e_id]
            row_sigma = sigma[e_id]

            for t_target in range(n_timeps):
                if scope_tidx is not None and t_target not in scope_tidx:
                    continue
                if not np.isnan(row_f[t_target]):
                    continue

                entry = self._fill_one(
                    row_f, row_sigma, t_target, e_id, step,
                )
                if entry is not None:
                    entries.append(entry)

        return entries

    def _fill_one(
        self,
        row_f: np.ndarray,
        row_sigma: np.ndarray,
        t_target: int,
        e_id: int,
        step: int,
    ) -> Optional[dict]:
        method = self.method
        n_t = row_f.shape[0]

        # Collect known (t_idx) observations before/after within window
        before = [
            t for t in range(max(0, t_target - self.wb), t_target)
            if not np.isnan(row_f[t])
        ]
        after = [
            t for t in range(t_target + 1, min(n_t, t_target + self.wa + 1))
            if not np.isnan(row_f[t])
        ]

        f_val: Optional[float] = None
        s_val: Optional[float] = None
        src:   list[int]       = []
        wts:   list[float]     = []
        meth:  str             = method

        if method == "locf":
            if not before:
                return None
            t_obs = before[-1]
            gap   = t_target - t_obs
            f_val = float(row_f[t_obs])
            s_val = float(row_sigma[t_obs]) * self.gap_inflation ** gap
            src, wts, meth = [t_obs], [1.0], "locf"

        elif method == "nocb":
            if not after:
                return None
            t_obs = after[0]
            gap   = t_obs - t_target
            f_val = float(row_f[t_obs])
            s_val = float(row_sigma[t_obs]) * self.gap_inflation ** gap
            src, wts, meth = [t_obs], [1.0], "nocb"

        elif method == "mean":
            candidates = before + after
            if len(candidates) < self.min_obs:
                return None
            n     = len(candidates)
            w     = 1.0 / n
            f_val = float(sum(w * row_f[t] for t in candidates))
            s_val = lp_combine([w * row_sigma[t] for t in candidates], self.p)
            src   = list(candidates)
            wts   = [w] * n
            meth  = "time_mean"

        else:  # linear
            if before and after:
                t_a = before[-1]
                t_b = after[0]
                gap  = t_b - t_a
                lam  = (t_target - t_a) / gap
                f_val = float((1 - lam) * row_f[t_a] + lam * row_f[t_b])
                s_val = lp_combine(
                    [(1 - lam) * row_sigma[t_a], lam * row_sigma[t_b]], self.p,
                ) * (self.gap_inflation ** gap)
                src, wts, meth = [t_a, t_b], [1 - lam, lam], "time_linear"
            elif self.interpolate_only:
                return None
            elif before:
                t_obs = before[-1]
                gap   = t_target - t_obs
                f_val = float(row_f[t_obs])
                s_val = float(row_sigma[t_obs]) * self.gap_inflation ** gap
                src, wts, meth = [t_obs], [1.0], "time_linear"
            elif after:
                t_obs = after[0]
                gap   = t_obs - t_target
                f_val = float(row_f[t_obs])
                s_val = float(row_sigma[t_obs]) * self.gap_inflation ** gap
                src, wts, meth = [t_obs], [1.0], "time_linear"
            else:
                return None

        if f_val is None:
            return None

        row_f[t_target]     = f_val
        row_sigma[t_target] = s_val
        return self._make_entry(e_id, t_target, step, meth, src, wts, f_val, s_val)

    @staticmethod
    def _make_entry(
        e_id: int, t_idx: int, step: int, method: str,
        src: list[int], coeffs: list[float],
        f_val: float, s_val: float,
    ) -> dict:
        # source_edges hold the t_idx of the anchor observation(s) within the
        # same series; coefficients carry the corresponding weights.
        return _log(e_id, t_idx, step, method, src, coeffs, f_val, s_val, "", None)


# ---------------------------------------------------------------------------
# FlatPriorImputer
# ---------------------------------------------------------------------------

