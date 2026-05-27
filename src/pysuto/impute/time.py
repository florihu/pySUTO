"""
time.py
=======
``TimeImputer`` — temporal interpolation / extrapolation per edge series.

Module-local :func:`_time_sort_key` (single caller — :class:`TimeImputer`)
lives here too.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd

from pysuto.logging import get_logger
from pysuto.lp import lp_combine

from .base import BaseImputer, _apply_cv_cap, _log

logger = get_logger(__name__)


def _time_sort_key(t):
    try:
        return (0, int(t))
    except (ValueError, TypeError):
        return (1, str(t))


class TimeImputer(BaseImputer):
    """
    Temporal interpolation / extrapolation for missing edges.

    Edges are grouped by (from_id, to_id, valuation).  Within each group,
    the sorted time series is scanned for gaps and filled according to method.

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
        per-edge mask in :meth:`impute` (where ``acc_struct.edge_index`` is
        available). The mask covers every edge whose (from_id, to_id,
        valuation) series contains ≥1 finite observation from the named
        feed — so NaN target slots in those series are still in-scope and
        can receive LOCF/NOCB fills.
        """
        if self._scope_feed_id is None:
            return
        feed_ids = list(getattr(agg_result, "feed_ids", []))
        if self._scope_feed_id not in feed_ids:
            raise ValueError(
                f"TimeImputer: scope_feed_id={self._scope_feed_id!r} not in "
                f"agg_result.feed_ids={feed_ids}"
            )
        self._feed_obs_mask = ~np.isnan(
            np.asarray(agg_result.F)[feed_ids.index(self._scope_feed_id), :]
        )

    def impute(
        self,
        f: np.ndarray,
        sigma: np.ndarray,
        missing: np.ndarray,
        step: int,
        acc_struct,
    ) -> list[dict]:
        ei = acc_struct.edge_index

        # Build sorted time ordering
        all_times = sorted(set(ei["time"].tolist()), key=_time_sort_key)
        t_to_idx  = {t: i for i, t in enumerate(all_times)}

        # Group edges by series key → {time_idx: edge_id}
        fid_arr = ei["from_id"].to_numpy(dtype=int)
        tid_arr = ei["to_id"].to_numpy(dtype=int)
        val_arr = ei["valuation"].to_numpy(dtype=str)
        tim_arr = ei["time"].values
        eid_arr = ei["edge_id"].to_numpy(dtype=int)

        # Resolve scope_edges-as-int-list lazily (N_edges now known).
        if (
            self._scope_mask is None
            and self._scope_edges_raw is not None
            and self._scope_feed_id is None
        ):
            n_edges = int(eid_arr.max()) + 1 if eid_arr.size else 0
            mask = np.zeros(n_edges, dtype=bool)
            mask[np.asarray(self._scope_edges_raw, dtype=int)] = True
            self._scope_mask = mask

        # Resolve scope_feed_id lazily — expand per-edge feed-observation
        # mask to "every edge whose series saw ≥1 observation from the feed",
        # so NaN target slots in those series are also in-scope.
        feed_obs = getattr(self, "_feed_obs_mask", None)
        if self._scope_feed_id is not None and feed_obs is not None:
            series_seen: dict[tuple, bool] = {}
            for i in range(len(fid_arr)):
                key = (fid_arr[i], tid_arr[i], val_arr[i])
                eid = int(eid_arr[i])
                if feed_obs[eid]:
                    series_seen[key] = True
                else:
                    series_seen.setdefault(key, False)
            n_edges = int(eid_arr.max()) + 1 if eid_arr.size else 0
            mask = np.zeros(n_edges, dtype=bool)
            for i in range(len(fid_arr)):
                key = (fid_arr[i], tid_arr[i], val_arr[i])
                if series_seen.get(key, False):
                    mask[int(eid_arr[i])] = True
            self._scope_mask = mask

        # scope_edges / scope_valuations drop entire edges (and their history)
        # from the series. scope_times is applied later — at the per-target
        # loop — so historical observations remain available as LOCF/NOCB
        # anchors when extrapolating into a scoped target year.
        series: dict[tuple, dict[int, int]] = {}
        scope_tidx: Optional[set] = None
        if self._scope_times is not None:
            scope_tidx = {t_to_idx[t] for t in all_times if str(t) in self._scope_times}

        for i in range(len(fid_arr)):
            eid = int(eid_arr[i])
            if self._scope_mask is not None and not self._scope_mask[eid]:
                continue
            if self._scope_valuations is not None and val_arr[i] not in self._scope_valuations:
                continue
            key = (fid_arr[i], tid_arr[i], val_arr[i])
            series.setdefault(key, {})[t_to_idx[tim_arr[i]]] = eid

        entries: list[dict] = []

        for key, t_map in series.items():
            sorted_tidxs = sorted(t_map.keys())

            for t_target in sorted_tidxs:
                if scope_tidx is not None and t_target not in scope_tidx:
                    continue
                e_id = t_map[t_target]
                if not np.isnan(f[e_id]):
                    continue

                entry = self._fill_one(
                    f, sigma, t_map, t_target, e_id, step,
                )
                if entry is not None:
                    entries.append(entry)

        return entries

    def _fill_one(
        self,
        f: np.ndarray,
        sigma: np.ndarray,
        t_map: dict[int, int],
        t_target: int,
        e_id: int,
        step: int,
    ) -> Optional[dict]:
        method = self.method

        # Collect known observations before and after within window
        before = [
            (t, t_map[t]) for t in sorted(t_map)
            if t < t_target and (t_target - t) <= self.wb and not np.isnan(f[t_map[t]])
        ]
        after = [
            (t, t_map[t]) for t in sorted(t_map)
            if t > t_target and (t - t_target) <= self.wa and not np.isnan(f[t_map[t]])
        ]

        f_val: Optional[float] = None
        s_val: Optional[float] = None
        src:   list[int]       = []
        wts:   list[float]     = []
        meth:  str             = method

        if method == "locf":
            if not before:
                return None
            t_obs, s_id = before[-1]
            gap   = t_target - t_obs
            f_val = float(f[s_id])
            s_val = float(sigma[s_id]) * self.gap_inflation ** gap
            src, wts, meth = [s_id], [1.0], "locf"

        elif method == "nocb":
            if not after:
                return None
            t_obs, s_id = after[0]
            gap   = t_obs - t_target
            f_val = float(f[s_id])
            s_val = float(sigma[s_id]) * self.gap_inflation ** gap
            src, wts, meth = [s_id], [1.0], "nocb"

        elif method == "mean":
            candidates = before + after
            if len(candidates) < self.min_obs:
                return None
            n     = len(candidates)
            w     = 1.0 / n
            f_val = float(sum(w * f[eid] for _, eid in candidates))
            # σ_mean = ( Σ_i |w·σ_i|^p )^(1/p)
            s_val = lp_combine([w * sigma[eid] for _, eid in candidates], self.p)
            src   = [eid for _, eid in candidates]
            wts   = [w] * n
            meth  = "time_mean"

        else:  # linear
            if before and after:
                t_a, a_id = before[-1]
                t_b, b_id = after[0]
                gap  = t_b - t_a
                lam  = (t_target - t_a) / gap
                f_val = float((1 - lam) * f[a_id] + lam * f[b_id])
                # σ_linear = ( |(1-λ)·σ_a|^p + |λ·σ_b|^p )^(1/p) · γ_0^gap
                s_val = lp_combine(
                    [(1 - lam) * sigma[a_id], lam * sigma[b_id]], self.p,
                ) * (self.gap_inflation ** gap)
                src, wts, meth = [a_id, b_id], [1 - lam, lam], "time_linear"
            elif self.interpolate_only:
                return None
            elif before:
                t_obs, s_id = before[-1]
                gap   = t_target - t_obs
                f_val = float(f[s_id])
                s_val = float(sigma[s_id]) * self.gap_inflation ** gap
                src, wts, meth = [s_id], [1.0], "time_linear"
            elif after:
                t_obs, s_id = after[0]
                gap   = t_obs - t_target
                f_val = float(f[s_id])
                s_val = float(sigma[s_id]) * self.gap_inflation ** gap
                src, wts, meth = [s_id], [1.0], "time_linear"
            else:
                return None

        if f_val is None:
            return None

        f[e_id]     = f_val
        sigma[e_id] = s_val
        return self._make_entry(e_id, step, meth, src, wts, f_val, s_val)

    @staticmethod
    def _make_entry(
        e_id: int, step: int, method: str,
        src: list[int], coeffs: list[float],
        f_val: float, s_val: float,
    ) -> dict:
        return _log(e_id, step, method, src, coeffs, f_val, s_val, "", None)


# ---------------------------------------------------------------------------
# FlatPriorImputer
# ---------------------------------------------------------------------------

