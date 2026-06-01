"""
Throwaway diagnostic: locate where high CV_root comes from in Stage 1 + Stage 2.

Layer 1 (per feed): for each registered feed, decompose
    CV_root[j] = CV_tot * sqrt(sum_i W_ij^2 f_i^2) / |sum_i W_ij f_i|
into a fan-in dilution ratio and check sign content of F_src.

Layer 2 (post aggregation): load the offending imputation run's
agg_result.npz and check whether aggregation moves CV up or down vs the
per-feed CVs.

Outputs are printed; copy the numbers into plans/cv_bloat_analysis.md.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
FEED_DIR = ROOT / "data/proc/feed/registry"
IMP_RUN = ROOT / "data/proc/imp/imp_copper_agg_imp_v0_20260430_061428_05d76679"

FEEDS = ["baci", "crude_ore", "icsg"]


def per_feed_diag(feed_id: str) -> dict:
    fdir = FEED_DIR / feed_id
    with open(fdir / f"{feed_id}.pkl", "rb") as f:
        d = pickle.load(f)
    W: sp.csr_matrix = d["weight_matrix"].tocsc()
    F_src = np.load(fdir / "F_src.npy")
    res = np.load(fdir / "result.npz")
    F_root = res["F_root"]
    sigma_root = res["sigma_root"]
    cv_root = res["cv_root"]
    cv_src = float(res["cv_src"][0])

    n_neg = int((F_src < 0).sum())
    n_zero_f = int((F_src == 0).sum())

    valid_cv = cv_root[np.isfinite(cv_root) & (np.abs(F_root) > 0)]
    pct = lambda q: float(np.percentile(valid_cv, q)) if valid_cv.size else float("nan")

    high_mask = np.isfinite(cv_root) & (cv_root > 1.0) & (np.abs(F_root) > 0)
    high_idx = np.where(high_mask)[0]

    # Per-column (per root edge) fan-in: count of contributing source rows.
    fan_in = np.diff(W.indptr)
    n_finite_root = int((fan_in > 0).sum())

    high_summary = {}
    if high_idx.size:
        sample = high_idx if high_idx.size <= 5000 else np.random.RandomState(0).choice(
            high_idx, 5000, replace=False)
        ratios = np.empty(sample.size)
        max_share = np.empty(sample.size)
        for k, j in enumerate(sample):
            col_start, col_end = W.indptr[j], W.indptr[j + 1]
            rows = W.indices[col_start:col_end]
            wvals = W.data[col_start:col_end]
            fvals = F_src[rows]
            terms = wvals * fvals
            num = abs(terms.sum())
            denom_sq = float(((wvals ** 2) * (fvals ** 2)).sum())
            ratios[k] = (cv_src * np.sqrt(denom_sq) / num) if num > 0 else np.inf
            max_share[k] = float((np.abs(terms).max() / np.abs(terms).sum())
                                 if np.abs(terms).sum() > 0 else 1.0)
        ratio_med = float(np.nanmedian(ratios))
        ratio_p99 = float(np.nanpercentile(ratios, 99))
        share_med = float(np.nanmedian(max_share))
        share_p10 = float(np.nanpercentile(max_share, 10))
        high_summary = dict(
            n_high=int(high_idx.size),
            ratio_median=ratio_med,
            ratio_p99=ratio_p99,
            dominant_share_median=share_med,
            dominant_share_p10=share_p10,
        )

    return dict(
        feed_id=feed_id,
        cv_src=cv_src,
        n_src=int(F_src.size),
        n_src_negative=n_neg,
        n_src_zero=n_zero_f,
        f_src_min=float(F_src.min()),
        f_src_max=float(F_src.max()),
        n_root_total=int(F_root.size),
        n_root_finite_in_W=n_finite_root,
        n_root_with_F=int((np.abs(F_root) > 0).sum()),
        cv_root_p50=pct(50),
        cv_root_p90=pct(90),
        cv_root_p99=pct(99),
        cv_root_max=float(valid_cv.max()) if valid_cv.size else float("nan"),
        cv_root_pct_above_1=float((valid_cv > 1.0).mean()) if valid_cv.size else 0.0,
        cv_root_pct_above_10=float((valid_cv > 10.0).mean()) if valid_cv.size else 0.0,
        cv_root_pct_above_100=float((valid_cv > 100.0).mean()) if valid_cv.size else 0.0,
        fan_in_p50=float(np.median(fan_in[fan_in > 0])) if (fan_in > 0).any() else 0.0,
        fan_in_p99=float(np.percentile(fan_in[fan_in > 0], 99)) if (fan_in > 0).any() else 0.0,
        fan_in_max=int(fan_in.max()),
        **high_summary,
    )


def post_agg_diag(per_feed_results: list[dict]) -> dict:
    agg = np.load(IMP_RUN / "agg_result.npz", allow_pickle=True)
    keys = list(agg.keys())
    print("  agg_result keys:", keys)

    # Recompute aggregation directly from per-feed F and Sigma so we
    # observe pre-imputation CVs only. a_agg/sigma_agg in the npz have
    # been overwritten by the imputation pipeline.
    F = agg["F"]              # (N_f, N_r) per-feed flow, NaN where missing
    S = agg["Sigma"]          # (N_f, N_r) per-feed sigma, NaN where missing
    coverage = agg["coverage"]  # (N_r,) bool
    feed_ids = list(agg["feed_ids"])

    with np.errstate(invalid="ignore", divide="ignore"):
        P = np.where(np.isfinite(S) & (S > 0), 1.0 / (S ** 2), 0.0)
        P_sum = P.sum(axis=0)
        F_filled = np.where(np.isfinite(F), F, 0.0)
        f_agg_pre = (P * F_filled).sum(axis=0) / np.where(P_sum > 0, P_sum, np.nan)
        sigma_agg_pre = 1.0 / np.sqrt(np.where(P_sum > 0, P_sum, np.nan))
        cv_agg_pre = np.where(np.abs(f_agg_pre) > 0,
                              sigma_agg_pre / np.abs(f_agg_pre), np.nan)

    valid = cv_agg_pre[np.isfinite(cv_agg_pre)]
    pct = lambda q: float(np.percentile(valid, q)) if valid.size else float("nan")
    n_covered_pre = int(coverage.sum())

    # Decompose drivers of large CV_agg
    n_feeds_per_edge = (np.isfinite(F)).sum(axis=0)
    multi = (n_feeds_per_edge >= 2)

    high_mask = np.isfinite(cv_agg_pre) & (cv_agg_pre > 1.0)
    high_idx = np.where(high_mask)[0]

    # Of high-CV edges: how many are single-feed vs multi-feed?
    n_high = int(high_idx.size)
    n_high_single = int(((n_feeds_per_edge[high_idx]) == 1).sum()) if n_high else 0
    n_high_multi = int(((n_feeds_per_edge[high_idx]) >= 2).sum()) if n_high else 0

    # For multi-feed high-CV edges, sign-disagreement frequency
    if n_high_multi:
        idx_m = high_idx[n_feeds_per_edge[high_idx] >= 2]
        Fm = F[:, idx_m]
        signs = np.where(np.isfinite(Fm), np.sign(Fm), 0)
        n_pos = (signs > 0).sum(axis=0)
        n_neg = (signs < 0).sum(axis=0)
        n_zero_obs = (signs == 0).sum(axis=0) - (~np.isfinite(Fm)).sum(axis=0)
        sign_disagree = (n_pos > 0) & (n_neg > 0)
        sign_pct = float(sign_disagree.mean())
        zero_pct = float(((n_zero_obs > 0)).mean())
    else:
        sign_pct = 0.0
        zero_pct = 0.0

    # σ-driver: ratio of σ_agg / |a_agg| broken down by which feed dominates
    # the inverse-variance weight on that edge
    return dict(
        feed_ids=str(feed_ids),
        n_root_total=int(F.shape[1]),
        n_covered_pre_impute=n_covered_pre,
        n_multi_feed_edges=int(multi.sum()),
        cv_agg_pre_p50=pct(50),
        cv_agg_pre_p90=pct(90),
        cv_agg_pre_p99=pct(99),
        cv_agg_pre_p999=pct(99.9),
        cv_agg_pre_max=float(valid.max()) if valid.size else float("nan"),
        pct_above_1=float((valid > 1.0).mean()) if valid.size else 0.0,
        pct_above_10=float((valid > 10.0).mean()) if valid.size else 0.0,
        pct_above_100=float((valid > 100.0).mean()) if valid.size else 0.0,
        n_high_total=n_high,
        n_high_single_feed=n_high_single,
        n_high_multi_feed=n_high_multi,
        multi_high_sign_disagreement_pct=sign_pct,
        multi_high_has_zero_obs_pct=zero_pct,
    )


def main() -> None:
    print("=" * 70)
    print("LAYER 1 — per-feed CV propagation")
    print("=" * 70)
    per_feed = []
    for fid in FEEDS:
        print(f"\n— {fid} —")
        r = per_feed_diag(fid)
        per_feed.append(r)
        for k, v in r.items():
            if isinstance(v, float):
                print(f"  {k:<28}{v:.6g}")
            else:
                print(f"  {k:<28}{v}")

    print("\n" + "=" * 70)
    print("LAYER 2 — post-aggregation")
    print("=" * 70)
    r = post_agg_diag(per_feed)
    for k, v in r.items():
        if isinstance(v, float):
            print(f"  {k:<28}{v:.6g}")
        else:
            print(f"  {k:<28}{v}")


if __name__ == "__main__":
    main()
