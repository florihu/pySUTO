"""
issue87_calibration_diag.py
===========================
#87 measurement: how far Stage 5's intervals understate the model's own uncertainty, and
how much of the contraction comes from the scale totals. Reads two uncprop runs on the same
opt run — the default and the `EXCLUDE_SCALE_TOTALS` ablation — through UncPropDiag, and
prints:

    K   calibration of the shifts on the default run: share |z/√κ| > 3 (N(0,1): 0.27 %),
        plain and mass-weighted, and its quantiles, by valuation × prior class
    B   the Birge ratio R = Σz²/dof per year and over the run, and the Birge-scaled 95 %
        half-width z_α·√R
    A   the ablation against the default, by valuation × prior class: mass-weighted κ,
        the confirmed share (κ ≥ 0.5, |z| ≤ 3) by weight, median σ*/σ_a

    python scripts/explo/issue87_calibration_diag.py <uncprop_default> <uncprop_ablation>
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from pysuto.diag.uncprop import UncPropDiag               # noqa: E402

KEYS = ["valuation", "prior_class"]


def _wmean(x: pd.Series, w: pd.Series) -> float:
    return float(np.sum(x * w) / np.sum(w)) if np.sum(w) > 0 else np.nan


def contraction(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame[np.isfinite(frame["kappa"])]
    rows = []
    for k, g in f.groupby(KEYS, observed=True):
        rows.append({**dict(zip(KEYS, k)),
                     "n":            len(g),
                     "kappa_w":      _wmean(g["kappa"], g["weight"]),
                     "confirmed_w":  _wmean((g["quadrant"] == "confirmed").astype(float), g["weight"]),
                     "ratio_median": float(np.median(g["sigma_a_star"] / g["sigma_a"]))})
    return pd.DataFrame(rows).set_index(KEYS)


def main(default: str, ablation: str) -> None:
    d = UncPropDiag.from_uncprop_id(default)
    a = UncPropDiag.from_uncprop_id(ablation)
    R = d.stats.birge_ratio()
    fd = d.stats.edge_frame()
    print(f"== K calibration of the shifts, {default}")
    with pd.option_context("display.float_format", "{:,.4g}".format, "display.width", 200):
        print(d.stats.calibration(fd, R).to_string())
    b = d.stats.birge()
    print(f"\n== B Birge ratio, {default}: run R = {R:.3f}; Birge-scaled 95 % half-width "
          f"1.960·√R = {1.960 * np.sqrt(R):.3f}·σ*")
    with pd.option_context("display.float_format", "{:,.4g}".format, "display.width", 200):
        print(b.to_string())
    ca, cd = contraction(a.stats.edge_frame()), contraction(fd)
    out = cd.join(ca, lsuffix=" default", rsuffix=" ablation")
    for c in ("kappa_w", "confirmed_w", "ratio_median"):
        out[f"Δ{c}"] = out[f"{c} ablation"] - out[f"{c} default"]
    print(f"\n== A scale-total ablation, {ablation} against {default}")
    with pd.option_context("display.float_format", "{:,.4f}".format, "display.width", 220):
        print(out.to_string())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
