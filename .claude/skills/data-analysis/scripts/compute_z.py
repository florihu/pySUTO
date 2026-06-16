"""Compute per-row z-scores for one timestep of a pySUTO solve.

Used in Step 3 of the data-analysis routine when the user's observation
involves constraint violations. Joins on ``row_meta.parquet`` so the
returned DataFrame includes ``type``, ``is_hard``, ``cone_type``, ``time``
columns — handy for stratifying by constraint family.

z is defined as ``(G @ F_opt - c) / sigma_c`` for ``sigma_c > 0`` and
returned as NaN for hard rows (sigma_c == 0).

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

    Ga = G @ F_opt
    residual = Ga - c

    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sigma_c > 0, residual / sigma_c, np.nan)

    df = row_meta.copy()
    df.insert(0, "row_idx", np.arange(len(df)))
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
