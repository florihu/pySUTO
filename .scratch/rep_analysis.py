"""
Follow-up REP analysis for opt_20260521_100428_9d33a0b9.

Generates the numerical evidence for:
 - §1 cold-start a_0(1995) → F*(1995) collapse per (sector, valuation)
 - §1 flat time-series claim (std/mean of F* vs a_0 trajectories)
 - §2 FlatPrior a_0=0 audit on Mass edges
 - §3 per-row unit audit in G (Mass/CC mixing)
 - §4 sigma_c sizing relative to |c|
"""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import scipy.sparse as sp

CONST = Path("data/proc/const/const_20260521_091641_8d374993")
OPT = Path("data/proc/opt/opt_20260521_100428_9d33a0b9")
IMP = Path("data/proc/imp/imp_20260521_074525_b1921faf")
ACC_CACHE = Path("data/proc/acc/copper")

# ---- static const artifacts
G = sp.load_npz(CONST/"G.npz").tocsr()
row_meta = pd.read_parquet(CONST/"row_meta.parquet")
a0 = np.load(CONST/"a0.npy")
sigma_a = np.load(CONST/"sigma_a.npy")
lb = np.load(CONST/"l.npy")
ub = np.load(CONST/"u.npy")
E = G.shape[1]; M = G.shape[0]
assert E == 854460

# ---- map edge_id → (sector_from, entity_from, region_from, sector_to, ..., time, valuation)
ei = pd.read_csv(ACC_CACHE/"edge_index.csv")
ni = pd.read_csv(ACC_CACHE/"node_index.csv")  # index = node id
ni["node_id"] = np.arange(len(ni))
ei = ei.merge(ni.add_suffix("_from"), left_on="from_id", right_on="node_id_from", how="left")
ei = ei.merge(ni.add_suffix("_to"),   left_on="to_id",   right_on="node_id_to",   how="left")
assert len(ei) == E
print(f"E={E}  M={M}  rows by type: {row_meta['type'].value_counts().to_dict()}")
print(f"valuations: {ei['valuation'].unique().tolist()}")

# ====================================================================
# § Overall a_0 / sigma_a distributional snapshot
# ====================================================================
print("\n========== §0/1 overall a_0 / sigma_a distribution ==========")
print(f"a_0   == 0  : {int((a0 == 0).sum()):>9,d} / {E:>9,d} = {(a0==0).mean()*100:5.1f}%")
print(f"a_0   != 0  : {int((a0 != 0).sum()):>9,d}")
print(f"|a_0|  median over nonzero  : {np.median(np.abs(a0[a0!=0])):.3e}")
print(f"|a_0|  p95 over nonzero     : {np.quantile(np.abs(a0[a0!=0]),0.95):.3e}")
print(f"|a_0|  max                  : {np.abs(a0).max():.3e}")
print(f"sigma_a min  / median / max : {sigma_a.min():.3e} / {np.median(sigma_a):.3e} / {sigma_a.max():.3e}")

# Split a_0 by valuation
for val in ["Mass", "CC"]:
    mask = (ei["valuation"] == val).to_numpy()
    a = a0[mask]; s = sigma_a[mask]
    nz = (a != 0).sum()
    z = (a == 0).sum()
    print(f"\n[{val}]  edges: {mask.sum():,}  a0==0: {z:,} ({z/mask.sum()*100:.1f}%)  "
          f"a0!=0: {nz:,}")
    if nz > 0:
        print(f"        |a_0|  median(non-zero)={np.median(np.abs(a[a!=0])):.3e}  "
              f"p95={np.quantile(np.abs(a[a!=0]),0.95):.3e}  max={np.abs(a).max():.3e}")
        print(f"        sigma_a  median={np.median(s):.3e}  p95={np.quantile(s,0.95):.3e}  max={s.max():.3e}")

# ====================================================================
# § Load F_opt(t) for all timesteps to study cold-start + temporal variation
# ====================================================================
print("\n========== Loading F_opt(t) for 30 timesteps ==========")
F_opt_all = {}
for ts_file in sorted((OPT/"timesteps").glob("t_*.npz")):
    year = int(ts_file.stem.split("_")[1])
    d = np.load(ts_file)
    F_opt_all[year] = d["F_opt"]
years_sorted = sorted(F_opt_all.keys())
print(f"Loaded F_opt for years {years_sorted[0]}…{years_sorted[-1]}  ({len(years_sorted)} years)")
print(f"F_opt(1995) shape={F_opt_all[1995].shape}  ==E? {F_opt_all[1995].shape[0]==E}")

# ====================================================================
# § c(t) and sigma_c(t) per timestep (load all years)
# ====================================================================
ts_const = {}
for ts_file in sorted((CONST/"timesteps").glob("t_*.npz")):
    year = int(ts_file.stem.split("_")[1])
    d = np.load(ts_file)
    ts_const[year] = {"c": d["c"], "sigma_c": d["sigma_c"]}
print(f"Loaded c, sigma_c for years {sorted(ts_const.keys())[0]}…{sorted(ts_const.keys())[-1]}")

# ====================================================================
# §1 Cold-start collapse — |F*(1995)| vs |a_0| per (sector_from, valuation)
# ====================================================================
print("\n========== §1 Cold-start (t=1995) collapse per (sector_from, valuation) ==========")
F0 = F_opt_all[1995]
df = pd.DataFrame({
    "edge_id": ei["edge_id"].to_numpy(),
    "sector_from": ei["sector_from"].to_numpy(),
    "valuation":   ei["valuation"].to_numpy(),
    "time":        ei["time"].to_numpy(),
    "a0":          a0,
    "sigma_a":     sigma_a,
    "F_opt_1995":  F0,
})
df_t = df[df["time"] == 1995].copy()
df_t["|a0|"]  = np.abs(df_t["a0"])
df_t["|F*|"]  = np.abs(df_t["F_opt_1995"])
df_t["ratio_F_over_a0"] = np.where(df_t["|a0|"] > 0, df_t["|F*|"] / df_t["|a0|"], np.nan)

grp = df_t.groupby(["sector_from", "valuation"]).agg(
    n_edges=("edge_id", "size"),
    n_a0_eq_0=("a0", lambda x: int((x == 0).sum())),
    n_a0_substantial=("|a0|", lambda x: int((x > 1.0).sum())),
    median_a0=("|a0|", "median"),
    sum_a0=("|a0|", "sum"),
    sum_F=("|F*|", "sum"),
).reset_index()
grp["sum_F_over_sum_a0"] = np.where(grp["sum_a0"] > 0, grp["sum_F"] / grp["sum_a0"], np.nan)
grp = grp.sort_values("sum_a0", ascending=False)
print("Top 20 (sector_from, valuation) buckets by Σ|a_0|:")
with pd.option_context("display.max_rows", None, "display.width", 200):
    print(grp.head(20).to_string(index=False))

# Save for REP
grp.to_csv(".scratch/§1_cold_start_per_sector_val.csv", index=False)

# ====================================================================
# §1b temporal flatness — std(F)/mean(F) vs std(a0)/mean(a0)
# ====================================================================
print("\n========== §1b temporal flatness: F*(t) vs a_0(t) ==========")
# Map each edge to its (sector_from, valuation) bucket and year. Then for each bucket
# compute Σ F per year, and compare its std/mean to std/mean of Σ a_0 per year.
df_all = pd.DataFrame({
    "edge_id":     ei["edge_id"].to_numpy(),
    "sector_from": ei["sector_from"].to_numpy(),
    "valuation":   ei["valuation"].to_numpy(),
    "time":        ei["time"].to_numpy(),
    "a0":          a0,
})
# Stack F_opt by year, broadcasting onto edge_id (each edge belongs to exactly one year already)
F_lookup = np.zeros(E)
for y, F in F_opt_all.items():
    F_lookup[ei["time"].to_numpy() == y] = F[ei["time"].to_numpy() == y]
df_all["F"] = F_lookup
# Sum across regions+entities for each (sector_from, valuation, time)
sums = df_all.groupby(["sector_from", "valuation", "time"]).agg(
    sum_a0=("a0", lambda x: np.abs(x).sum()),
    sum_F=("F",  lambda x: np.abs(x).sum()),
).reset_index()
flat = sums.groupby(["sector_from", "valuation"]).agg(
    mean_a0=("sum_a0", "mean"),
    std_a0=("sum_a0", "std"),
    mean_F=("sum_F", "mean"),
    std_F=("sum_F", "std"),
).reset_index()
flat["CV_a0"] = flat["std_a0"] / flat["mean_a0"].replace(0, np.nan)
flat["CV_F"]  = flat["std_F"]  / flat["mean_F"].replace(0, np.nan)
flat["scale_ratio"] = flat["mean_F"] / flat["mean_a0"].replace(0, np.nan)
flat = flat.sort_values("mean_a0", ascending=False)
print("Top 20 (sector_from, valuation) buckets — CV(a_0) vs CV(F*) and scale_ratio:")
with pd.option_context("display.max_rows", None, "display.width", 200):
    print(flat.head(20).to_string(index=False))
flat.to_csv(".scratch/§1b_temporal_flatness.csv", index=False)

# ====================================================================
# §2 FlatPrior a_0 = 0 audit by valuation × sector_from
# ====================================================================
print("\n========== §2 FlatPrior a_0=0 audit ==========")
# A FlatPrior'd edge has a_0 exactly 0 (because the FlatPriorImputer sets f_j = 0 for residual
# NaN edges). sigma_a comes from the cascade. We can identify these edges by a_0 == 0.
fp = df_all.copy()
fp["a0_eq_0"] = (fp["a0"] == 0).astype(int)
sec_val = fp.groupby(["sector_from", "valuation"]).agg(
    n_edges=("a0_eq_0", "size"),
    n_zero=("a0_eq_0", "sum"),
).reset_index()
sec_val["pct_zero"] = sec_val["n_zero"] / sec_val["n_edges"] * 100
sec_val = sec_val.sort_values(["valuation", "pct_zero"], ascending=[True, False])
print("a_0=0 share per (sector_from, valuation):")
with pd.option_context("display.max_rows", None, "display.width", 200):
    print(sec_val.to_string(index=False))
sec_val.to_csv(".scratch/§2_flat_prior_a0_zero.csv", index=False)

# Detail: for FlatPrior'd edges (a_0 == 0), what is sigma_a distribution by valuation?
print("\nsigma_a distribution on a_0==0 edges by valuation:")
for val in ["Mass", "CC"]:
    mask = (ei["valuation"] == val).to_numpy() & (a0 == 0)
    s = sigma_a[mask]
    if s.size:
        print(f"  [{val}]  n={mask.sum():,}  sigma_a  min={s.min():.3e}  "
              f"median={np.median(s):.3e}  p95={np.quantile(s,0.95):.3e}  max={s.max():.3e}")

# ====================================================================
# §3 per-row unit audit in G — sum_eq rows
# ====================================================================
print("\n========== §3 per-row unit audit (sum_eq rows) ==========")
sum_eq_rows = np.where(row_meta["type"].to_numpy() == "sum_eq")[0]
print(f"# sum_eq rows: {len(sum_eq_rows):,}")

# For each sum_eq row, look at the valuations of its nonzero columns in G
val_arr = ei["valuation"].to_numpy()
G_csr = G.tocsr()
mixed_rows = []
val_per_row = []
for k, r in enumerate(sum_eq_rows):
    cols = G_csr.indices[G_csr.indptr[r]:G_csr.indptr[r+1]]
    if cols.size == 0:
        val_per_row.append("empty")
        continue
    vals = set(val_arr[cols].tolist())
    if len(vals) > 1:
        mixed_rows.append(r)
        val_per_row.append("mixed")
    else:
        val_per_row.append(next(iter(vals)))
mixed_n = len(mixed_rows)
print(f"sum_eq rows with mixed valuations: {mixed_n:,} / {len(sum_eq_rows):,} = {mixed_n/len(sum_eq_rows)*100:.1f}%")

vpd = pd.Series(val_per_row).value_counts()
print(f"sum_eq row valuation composition: {vpd.to_dict()}")

# Same for ratio_eq
print("\n----- ratio_eq rows -----")
ratio_eq_rows = np.where(row_meta["type"].to_numpy() == "ratio_eq")[0]
mixed_r = 0
for r in ratio_eq_rows[:50000]:  # subsample for speed; ratio_eq is 435k
    cols = G_csr.indices[G_csr.indptr[r]:G_csr.indptr[r+1]]
    if cols.size == 0: continue
    vals = set(val_arr[cols].tolist())
    if len(vals) > 1:
        mixed_r += 1
print(f"ratio_eq rows with mixed valuations (subsample 50k of {len(ratio_eq_rows):,}): {mixed_r:,}")

# Same for mass_balance
print("\n----- mass_balance rows (hard) -----")
mb_rows = np.where(row_meta["type"].to_numpy() == "mass_balance")[0]
mb_mixed = 0
for r in mb_rows:
    cols = G_csr.indices[G_csr.indptr[r]:G_csr.indptr[r+1]]
    if cols.size == 0: continue
    vals = set(val_arr[cols].tolist())
    if len(vals) > 1:
        mb_mixed += 1
print(f"mass_balance rows with mixed valuations: {mb_mixed:,} / {len(mb_rows):,} = {mb_mixed/len(mb_rows)*100:.1f}%")

# ====================================================================
# §4 sigma_c / |c| for sum_eq rows at t=1995 + pooled across years
# ====================================================================
print("\n========== §4 sigma_c sizing relative to |c| ==========")
# All sum_eq rows are time-stamped; rows whose time != t are deactivated (σ_c = 1e10).
# Filter to the "active" rows at each timestep (σ_c < 1e9).
all_cv = []
for y in years_sorted:
    c = ts_const[y]["c"]
    s = ts_const[y]["sigma_c"]
    # active sum_eq rows for this year (σ_c < 1e9)
    active_mask = (s < 1e9)
    sum_active = active_mask & np.isin(np.arange(M), sum_eq_rows)
    c_active = c[sum_active]
    s_active = s[sum_active]
    cv = s_active / np.maximum(np.abs(c_active), 1e-12)
    finite = np.isfinite(cv) & (np.abs(c_active) > 1e-6)
    all_cv.append(cv[finite])
all_cv = np.concatenate(all_cv)
print(f"Active sum_eq rows total (pooled, c != 0): {len(all_cv):,}")
print(f"CV = sigma_c / |c|  median={np.median(all_cv):.3f}  p50/75/90/95={np.percentile(all_cv,[50,75,90,95])}")
print(f"  > 1.0 (toothless): {(all_cv > 1.0).sum():,} ({(all_cv>1.0).mean()*100:.1f}%)")
print(f"  > 0.5 :            {(all_cv > 0.5).sum():,} ({(all_cv>0.5).mean()*100:.1f}%)")
print(f"  > 0.2 :            {(all_cv > 0.2).sum():,} ({(all_cv>0.2).mean()*100:.1f}%)")
print(f"  < 0.05:            {(all_cv < 0.05).sum():,} ({(all_cv<0.05).mean()*100:.1f}%)")

# CV distribution by valuation composition of the row
print("\nCV by row-valuation composition (t=1995 only):")
c95 = ts_const[1995]["c"]
s95 = ts_const[1995]["sigma_c"]
for k, r in enumerate(sum_eq_rows):
    pass
sum_eq_with_val = pd.DataFrame({"row_id": sum_eq_rows, "valuation_set": val_per_row})
sum_eq_with_val["c"]       = c95[sum_eq_rows]
sum_eq_with_val["sigma_c"] = s95[sum_eq_rows]
sum_eq_with_val["active"]  = (sum_eq_with_val["sigma_c"] < 1e9)
sum_eq_with_val["CV"]      = sum_eq_with_val["sigma_c"] / np.maximum(np.abs(sum_eq_with_val["c"]), 1e-12)
active95 = sum_eq_with_val[sum_eq_with_val["active"] & (np.abs(sum_eq_with_val["c"]) > 1e-6)]
print(f"  active rows at t=1995 with |c|>0: {len(active95):,}")
print("  CV distribution by valuation_set:")
grp_cv = active95.groupby("valuation_set").agg(
    n=("CV", "size"),
    median_CV=("CV", "median"),
    p95_CV=("CV", lambda x: np.quantile(x, 0.95)),
    max_CV=("CV", "max"),
    median_c=("c", lambda x: np.median(np.abs(x))),
    median_sigma=("sigma_c", "median"),
).reset_index()
with pd.option_context("display.max_rows", None, "display.width", 200):
    print(grp_cv.to_string(index=False))
grp_cv.to_csv(".scratch/§4_cv_by_row_valuation.csv", index=False)

# Look at the Ore|ROW 2024 row specifically (the wall in prior REP)
print("\nForensic — Ore|ROW 2024 sum_eq row (the wall):")
for y in [1995, 2024]:
    c_y = ts_const[y]["c"]
    s_y = ts_const[y]["sigma_c"]
    active_y = (s_y < 1e9) & np.isin(np.arange(M), sum_eq_rows)
    candidate = np.where(active_y)[0]
    if candidate.size:
        df_c = pd.DataFrame({
            "row_id": candidate,
            "c": c_y[candidate],
            "sigma_c": s_y[candidate],
            "constraint_id": row_meta.loc[candidate, "constraint_id"].to_numpy(),
        })
        df_c["CV"] = df_c["sigma_c"] / np.maximum(np.abs(df_c["c"]), 1e-12)
        worst = df_c.iloc[df_c["CV"].argsort()[::-1].iloc[:10].to_numpy() if hasattr(df_c["CV"].argsort()[::-1],"iloc") else df_c["CV"].argsort()[::-1][:10]]
        print(f"  Year {y}: top-10 CV rows:")
        print(worst.head(10).to_string(index=False))

print("\nDONE")
