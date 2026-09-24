"""Compute per-row z-scores for one timestep of a pySUTO solve.

Used in Step 3 of the data-analysis routine when the user's observation
involves constraint violations. Joins on ``row_meta.parquet`` so the
returned DataFrame includes ``type``, ``is_hard``, ``cone_type``, ``time``
columns — handy for stratifying by constraint family.

z is defined as ``(G @ F_opt - c) / sigma_c`` for ``sigma_c > 0`` and
returned as NaN for hard rows (sigma_c == 0). Only the rows of timestep ``t``
are returned.

Block-path runs (every run since the year-local fix) store ``F_opt`` as the
year's slice over ``cols_t`` (shape ``(E_t,)``), not a full ``(N_edges,)``
vector; it is scattered to global length before the product with ``G``.
Legacy full-vector runs (``cols_t`` absent) are handled as before.

Usage::

    from compute_z import compute_z
    df = compute_z(const_dir, opt_dir, t="2024")
    df.sort_values("abs_z", ascending=False).head(20)

No pysuto imports — works mid-refactor.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def _year_local_index(t_opt):
    """``(cols_t, rows_t)`` of a block-path npz, or ``(None, None)`` for a
    legacy full-vector run (key absent or stored as a pickled ``None``)."""
    if "cols_t" not in t_opt.files:
        return None, None
    try:
        cols, rows = t_opt["cols_t"], t_opt["rows_t"]
    except ValueError:          # object array (None) → needs allow_pickle → legacy
        return None, None
    if np.ndim(cols) == 0:
        return None, None
    return np.asarray(cols, dtype=int), np.asarray(rows, dtype=int)


def compute_z(const_dir: str | Path, opt_dir: str | Path, t: str) -> pd.DataFrame:
    """Return per-row residuals and z-scores for timestep ``t``.

    Parameters
    ----------
    const_dir
        Path to ``data/proc/const/<const_run_id>/``.
    opt_dir
        Path to ``data/proc/opt/<opt_run_id>/``.
    t
        Year string, e.g. ``"2024"``.

    Returns
    -------
    DataFrame with columns:
        ``row_idx, constraint_id, type, is_hard, cone_type, time,
         c, sigma_c, Ga, residual, z, abs_z``.
    """
    const_dir = Path(const_dir)
    opt_dir = Path(opt_dir)

    G = sparse.load_npz(const_dir / "G.npz")
    row_meta = pd.read_parquet(const_dir / "row_meta.parquet")

    t_const = np.load(const_dir / "timesteps" / f"t_{t}.npz")
    c = t_const["c"]
    sigma_c = t_const["sigma_c"]

    t_opt = np.load(opt_dir / "timesteps" / f"t_{t}.npz")
    F_opt = t_opt["F_opt"]

    cols_t, rows_t = _year_local_index(t_opt)
    if cols_t is not None:
        # block path: scatter the year-local slice into a global vector
        F = np.zeros(G.shape[1])
        F[cols_t] = F_opt
        rows = rows_t
    else:
        F = F_opt
        rows = np.flatnonzero(row_meta["time"].astype(str).to_numpy() == str(t))

    Ga = np.asarray(G @ F).ravel()[rows]
    c = np.asarray(c, dtype=float)[rows]
    sigma_c = np.asarray(sigma_c, dtype=float)[rows]
    residual = Ga - c

    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sigma_c > 0, residual / sigma_c, np.nan)

    df = row_meta.iloc[rows].copy().reset_index(drop=True)
    df.insert(0, "row_idx", rows)
    df["c"] = c
    df["sigma_c"] = sigma_c
    df["Ga"] = Ga
    df["residual"] = residual
    df["z"] = z
    df["abs_z"] = np.abs(z)
    return df


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("usage: python compute_z.py <const-dir> <opt-dir> <year>")
        sys.exit(2)
    df = compute_z(sys.argv[1], sys.argv[2], sys.argv[3])
    worst = df.sort_values("abs_z", ascending=False).head(20)
    print(worst.to_string(index=False))
