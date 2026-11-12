"""
truth.py
========
``TruthDiag`` — every stage's estimate of the edges scored against a **ground truth**.

Only a synthetic case knows the values its feeds were drawn from (the toy case,
:mod:`pysuto.cases.toy`); a real case never does, which is why this diagnostic exists at
all. Three figures, each a question the truth answers:

1. **Error by stage** — how far each stage's estimate (feed → aggregate → imputed prior
   → reconciled) sits from the truth, as a relative error per edge.
2. **Calibration** — whether the posterior σ is honest: ``z = (a* − truth) / σ_a*``
   against N(0, 1), and the share of edges whose ``(1 − α)`` interval holds the truth.
3. **Residual rank** — the solver's prior residuals (:func:`prior_z`), largest first,
   with the planted edges marked.

Data contract: every array is indexed by edge id (length ``n``, ``edge_meta`` row ``i``
is edge ``i``); NaN in an estimate means the stage did not estimate that edge.
``diag/`` is a leaf: nothing here imports the data layer.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from ._plot import _C, _savefig


def prior_z(a_star, a_prior, sigma_a) -> np.ndarray:
    """An edge's prior residual as the solver weighs it, ``z = (a* − a_prior) / σ_a``,
    with ``a_prior`` and ``σ_a`` conditioned as the QP conditions them
    (``pysuto._conditioning.condition_prior``)."""
    return ((np.asarray(a_star, dtype=float) - np.asarray(a_prior, dtype=float))
            / np.asarray(sigma_a, dtype=float))


class TruthDiag:
    """Score stage estimates against a ground truth (see the module docstring).

    Parameters
    ----------
    estimates : dict[str, np.ndarray]
        Stage name → per-edge estimate (NaN = not estimated), in stage order, e.g.
        ``feed``, ``aggregate``, ``prior``, ``reconciled``. ``reconciled`` (``a*``) is
        required for the calibration; ``prior`` and ``reconciled`` for the residual
        rank.
    truth : np.ndarray
        The ground truth per edge.
    sigma_post : np.ndarray or None
        The posterior σ of ``a*`` per edge (Stage 5); ``None`` skips the calibration.
    edge_meta : DataFrame
        ``acc.edge_meta``: ``edge_id``, ``from_/to_sector``, ``from_/to_region``,
        ``time``, ``valuation``, one row per edge in edge-id order.
    sigma_prior : np.ndarray, optional
        The prior σ the QP used per edge; needed for the residual rank.
    planted : sequence of int
        Edge ids of planted defects, marked in the figures and the summary.
    alpha : float
        The interval level, ``(1 − α)``.
    units : dict, optional
        Valuation → unit label (``{"Gross": "kt", "Content": "kt Content"}``); default
        ``"kt <valuation>"`` for every valuation.
    label : str
        Figure title prefix.
    """

    def __init__(self, estimates: dict, truth: np.ndarray,
                 sigma_post: Optional[np.ndarray], edge_meta: pd.DataFrame, *,
                 sigma_prior: Optional[np.ndarray] = None,
                 planted: Sequence[int] = (), alpha: float = 0.05,
                 units: Optional[dict] = None, label: str = "") -> None:
        self.truth = np.asarray(truth, dtype=float)
        n = self.truth.shape[0]
        if self.truth.ndim != 1:
            raise ValueError(f"truth must be 1-D, got shape {self.truth.shape}")
        if len(edge_meta) != n or not np.array_equal(
                edge_meta["edge_id"].to_numpy(), np.arange(n)):
            raise ValueError(f"edge_meta must hold edges 0..{n - 1} in order "
                             f"({len(edge_meta)} rows)")
        self.estimates = {}
        for stage, values in estimates.items():
            arr = np.asarray(values, dtype=float)
            if arr.shape != (n,):
                raise ValueError(f"estimate {stage!r} has shape {arr.shape}, "
                                 f"expected ({n},)")
            self.estimates[stage] = arr
        self.sigma_post  = self._vector(sigma_post,  "sigma_post",  n)
        self.sigma_prior = self._vector(sigma_prior, "sigma_prior", n)
        self.edge_meta   = edge_meta.reset_index(drop=True)
        self.planted     = [int(e) for e in planted]
        if any(not 0 <= e < n for e in self.planted):
            raise ValueError(f"planted edge ids {self.planted} outside 0..{n - 1}")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        valuations = list(dict.fromkeys(self.edge_meta["valuation"]))
        self.units = {v: f"kt {v}" for v in valuations}
        self.units.update(units or {})
        self.label = label

    @property
    def _ranked(self) -> bool:
        """Whether the residual rank can be drawn: a prior, an ``a*`` and the prior σ."""
        return (self.sigma_prior is not None
                and {"prior", "reconciled"} <= set(self.estimates))

    @staticmethod
    def _vector(values, name: str, n: int) -> Optional[np.ndarray]:
        if values is None:
            return None
        arr = np.asarray(values, dtype=float)
        if arr.shape != (n,):
            raise ValueError(f"{name} has shape {arr.shape}, expected ({n},)")
        return arr

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def edge_labels(self, ids) -> list:
        """``Sector[Region] → Sector[Region] year valuation`` for each edge id."""
        em = self.edge_meta.loc[list(ids)]
        return [f"{fs}[{fr}] → {ts}[{tr}] {t} {v}" for fs, fr, ts, tr, t, v in zip(
            em["from_sector"], em["from_region"], em["to_sector"], em["to_region"],
            em["time"], em["valuation"])]

    def relative_error(self, stage: str) -> np.ndarray:
        """``|estimate − truth| / |truth|`` per edge; NaN where the stage has no estimate
        or the truth is zero."""
        est = self.estimates[stage]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(est - self.truth) / np.abs(self.truth)
        return np.where(np.isfinite(est) & (self.truth != 0.0), rel, np.nan)

    def error_by_stage(self) -> pd.DataFrame:
        """Per stage: edges estimated (truth ≠ 0) and the median, p90 and max relative
        error over them."""
        rows = []
        for stage in self.estimates:
            rel = self.relative_error(stage)
            rel = rel[np.isfinite(rel)]
            rows.append({"stage": stage, "n_edges": int(rel.size),
                         "median": float(np.median(rel)) if rel.size else np.nan,
                         "p90":    float(np.quantile(rel, 0.9)) if rel.size else np.nan,
                         "max":    float(rel.max()) if rel.size else np.nan})
        return pd.DataFrame(rows).set_index("stage")

    def _require_posterior(self) -> np.ndarray:
        if self.sigma_post is None or "reconciled" not in self.estimates:
            raise ValueError("calibration needs a 'reconciled' estimate and sigma_post")
        return self.estimates["reconciled"]

    def z_post(self) -> np.ndarray:
        """``(a* − truth) / σ_a*`` per edge; NaN where ``σ_a* = 0``."""
        a_star = self._require_posterior()
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (a_star - self.truth) / self.sigma_post
        return np.where(self.sigma_post > 0.0, z, np.nan)

    def covered(self) -> np.ndarray:
        """Whether each edge's ``(1 − α)`` interval ``a* ± z·σ_a*`` holds the truth; an
        edge with ``σ_a* = 0`` counts as covered only where ``a*`` equals the truth to
        1e-9 (in its valuation's unit)."""
        a_star = self._require_posterior()
        half   = norm.ppf(1.0 - self.alpha / 2.0) * self.sigma_post
        diff   = np.abs(a_star - self.truth)
        return np.where(self.sigma_post > 0.0, diff <= half, diff <= 1e-9)

    def coverage(self) -> float:
        """Share of all edges whose ``(1 − α)`` interval holds the truth."""
        return float(self.covered().mean())

    def residual_rank(self) -> pd.DataFrame:
        """Every edge's prior residual (:func:`prior_z`), largest ``|z|`` first, with the
        planted edges flagged."""
        if self.sigma_prior is None or not {"prior", "reconciled"} <= set(self.estimates):
            raise ValueError("the residual rank needs 'prior', 'reconciled' and "
                             "sigma_prior")
        z     = prior_z(self.estimates["reconciled"], self.estimates["prior"],
                        self.sigma_prior)
        order = np.argsort(-np.abs(z), kind="stable")
        return pd.DataFrame({"edge_id": order, "edge": self.edge_labels(order),
                             "z": z[order], "planted": np.isin(order, self.planted)})

    def retention(self, valuation: str) -> pd.DataFrame:
        """By category (``from_sector → to_sector``) and year, in one valuation: the
        truth, prior and ``a*`` sums in the valuation's unit, the prior and ``a*``
        against the truth, and the retention ``Σa*/Σa_prior`` (GLOSSARY.md
        "Retention"). Categories with a negative truth in any year (a signed leg) are
        left out; a ratio over a zero sum is NaN."""
        em    = self.edge_meta
        unit  = self.units[valuation]
        nan   = np.full(self.truth.size, np.nan)
        frame = pd.DataFrame({
            "category": em["from_sector"] + " → " + em["to_sector"],
            "year":     em["time"].astype(str),
            "truth":    self.truth,
            "prior":    self.estimates.get("prior", nan),
            "a_star":   self.estimates.get("reconciled", nan),
        })[(em["valuation"] == valuation).to_numpy()]
        signed = frame.groupby("category")["truth"].transform("min") < 0
        sums   = frame[~signed].groupby(["category", "year"])[
            ["truth", "prior", "a_star"]].sum()

        def over(num, den):
            return sums[num] / sums[den].where(sums[den] != 0)

        return pd.DataFrame({
            f"truth [{unit}]": sums["truth"],
            f"prior [{unit}]": sums["prior"],
            f"a* [{unit}]":    sums["a_star"],
            "prior/truth":     over("prior", "truth"),
            "a*/truth":        over("a_star", "truth"),
            "retention":       over("a_star", "prior"),
        })

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def plot(self, odir: str, show: bool = False) -> list:
        """Write the three figures (calibration only with ``sigma_post``, the residual
        rank only with ``sigma_prior``) into ``odir``; returns their paths."""
        import matplotlib.pyplot as plt

        paths = [self._plot_error_by_stage(odir, plt)]
        if self.sigma_post is not None and "reconciled" in self.estimates:
            paths.append(self._plot_calibration(odir, plt))
        if self._ranked:
            paths.append(self._plot_residual_rank(odir, plt))
        if show:
            plt.show()
        plt.close("all")
        return paths

    def _title(self, what: str) -> str:
        return f"{self.label} — {what}" if self.label else what

    def _plot_error_by_stage(self, odir: str, plt) -> str:
        stages = list(self.estimates)
        data   = [np.log10(np.maximum(r[np.isfinite(r)], 1e-12))
                  for r in (self.relative_error(s) for s in stages)]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot([d if d.size else [np.nan] for d in data], showfliers=True)
        ax.set_xticks(range(1, len(stages) + 1))
        ax.set_xticklabels([f"{s}\n(n = {d.size})" for s, d in zip(stages, data)])
        for e in self.planted:
            for i, s in enumerate(stages, start=1):
                rel = self.relative_error(s)[e]
                if np.isfinite(rel):
                    ax.plot(i, np.log10(max(rel, 1e-12)), "o", color=_C["red"],
                            label="planted" if i == 1 else None)
        ax.set_ylabel("log₁₀ |estimate − truth| / |truth|")
        ax.set_title(self._title("error against the ground truth by stage"),
                     fontweight="bold")
        if self.planted:
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        return _savefig(fig, os.path.join(odir, "truth_error_by_stage.png"))

    def _plot_calibration(self, odir: str, plt) -> str:
        z = self.z_post()
        z = z[np.isfinite(z)]
        fig, (ax_h, ax_q) = plt.subplots(1, 2, figsize=(12, 5))
        ax_h.hist(np.clip(z, -6, 6), bins=40, density=True, color=_C["blue"],
                  edgecolor="white", label=f"edges (n = {z.size})")
        grid = np.linspace(-6, 6, 241)
        ax_h.plot(grid, norm.pdf(grid), color=_C["navy"], label="N(0, 1)")
        ax_h.set_xlabel("z = (a* − truth) / σ_a*   (clipped to ±6)")
        ax_h.set_ylabel("density")
        ax_h.legend(fontsize=8)
        q_theory = norm.ppf((np.arange(1, z.size + 1) - 0.5) / z.size)
        ax_q.plot(q_theory, np.sort(z), ".", color=_C["blue"])
        lim = max(3.0, float(np.max(np.abs(q_theory))) if z.size else 3.0)
        ax_q.plot([-lim, lim], [-lim, lim], color=_C["navy"], lw=1)
        ax_q.set_xlabel("N(0, 1) quantile")
        ax_q.set_ylabel("z quantile")
        fig.suptitle(self._title(f"calibration — {100 * (1 - self.alpha):.0f} % interval "
                                 f"coverage {100 * self.coverage():.1f} % of "
                                 f"{self.truth.size} edges"), fontweight="bold")
        for ax in (ax_h, ax_q):
            ax.grid(alpha=0.3)
        return _savefig(fig, os.path.join(odir, "truth_calibration.png"))

    def _plot_residual_rank(self, odir: str, plt, top: int = 25) -> str:
        rank = self.residual_rank().head(top)
        fig, ax = plt.subplots(figsize=(10, 0.3 * len(rank) + 1.5))
        colors = [_C["red"] if p else _C["grey"] for p in rank["planted"]]
        ax.barh(np.arange(len(rank))[::-1], np.abs(rank["z"]), color=colors)
        ax.set_yticks(np.arange(len(rank))[::-1])
        ax.set_yticklabels(rank["edge"], fontsize=7)
        ax.set_xlabel("|z| = |a* − a_prior| / σ_a")
        ax.set_title(self._title(f"largest prior residuals (top {len(rank)}; planted in "
                                 f"red)"), fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        return _savefig(fig, os.path.join(odir, "truth_residual_rank.png"))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Print and return the same as the figures, plus retention by category and
        year in each valuation."""
        bar   = "─" * 72
        lines = [bar, f"  TRUTH DIAGNOSTICS — {self.label or 'ground truth'}", bar,
                 "  Relative error against the truth by stage (edges with truth ≠ 0):",
                 self.error_by_stage().to_string(float_format=lambda x: f"{x:.4g}")]
        if self.sigma_post is not None and "reconciled" in self.estimates:
            z = self.z_post()
            z = z[np.isfinite(z)]
            lines += [f"\n  Calibration: {100 * (1 - self.alpha):.0f} % interval "
                      f"coverage {100 * self.coverage():.1f} % of {self.truth.size} edges; "
                      f"z mean {np.mean(z):+.3f}, sd {np.std(z):.3f} (n = {z.size})"]
        if self._ranked:
            rank = self.residual_rank().head(10)
            lines.append("\n  Largest prior residuals |z| (planted marked *):")
            lines += [f"    {'*' if p else ' '} {z:+9.3f}  {edge}"
                      for z, edge, p in zip(rank["z"], rank["edge"], rank["planted"])]
        for valuation in self.units:
            lines += [f"\n  Retention by category and year, {valuation} (signed "
                      f"categories left out):",
                      self.retention(valuation).to_string(
                          float_format=lambda x: f"{x:.4g}")]
        lines.append(bar)
        text = "\n".join(lines)
        print(text)
        return text
