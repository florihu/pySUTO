"""Empirical p95 bounds on |stock_change| / cathode_consumption for MESC/USC/PSC.

Reads the ICSG feed values, computes the ratio of absolute stock change to
cathode-Semi consumption per (region, year), reports per-region and pooled
statistics, plots a 3-panel figure, and writes one params.csv row per
(region, indicator) with agg=true so the RatioBoundBuilder aggregates the
numerator and denominator over all source regions per destination region.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pysuto.acc import AccountingStructure  # noqa: E402

VALUES_CSV = REPO_ROOT / "data/input/feed/icsg/current/values.csv"
EFF_PARAMS_CSV = REPO_ROOT / "data/input/params/eff/current/params.csv"
PARAMS_DIR = REPO_ROOT / "data/input/params/stock_bounds/current"
FIG_PATH = REPO_ROOT / "figs/ex/stock_bounds.png"

ACC_ID = "copper"
ACC_DIR = REPO_ROOT / "data/proc/acc"

# Electrorefining efficiency (Anode/Flow → ER/Process → CER/Flow) applied to
# Anode-Use rows to get refined-cathode production for the PSC denominator.
REFINING_EFF_PARAM_ID = "effer"

# Per-indicator region clustering: indicators listed here are split into
# multiple clusters via largest-gap on log10(per-region p95). Others use a
# single global pool.
CLUSTERED = {
    "MESC": dict(k=2, names=["domestic", "hub"]),  # names ordered low → high
}

INDICATORS = {
    "MESC": dict(Type="Cathode", Process="MESC", Supply_Use="Use"),
    "USC":  dict(Type="Cathode", Process="USC",  Supply_Use="Use"),
    "PSC":  dict(Type="PSC",     Process="Cathode_Pool", Supply_Use="Supply"),
}

# Per-indicator denominator: list of (filter, scale) terms summed across
# (region, year). The scale lets us convert anode input to refined-cathode
# output via the ER (electrorefining) efficiency.
#
# MESC/USC are demand-side stock changes → normalize by Semi-use throughput.
# PSC is the cathode-pool (supply-side) stock change → normalize by total
# refined-cathode production = SXEW + (Anode/Refining/Use × eff_er).
DENOMINATORS = {
    "MESC": [(dict(Type="Cathode", Supply_Use="Use", Process="Semi"), 1.0)],
    "USC":  [(dict(Type="Cathode", Supply_Use="Use", Process="Semi"), 1.0)],
    "PSC":  None,   # built dynamically because the eff scale is loaded at runtime
}

DENOM_LABELS = {
    "MESC": "Cathode/Semi/Use",
    "USC":  "Cathode/Semi/Use",
    "PSC":  "Cathode/SXEW/Supply + eff_er * Anode/Refining/Use",
}

PARAMS_COLUMNS = [
    "param_id", "param_type", "agg", "description",
    "a_sector_from", "a_entity_from", "a_region_from",
    "a_sector_to",   "a_entity_to",   "a_region_to",
    "a_time", "a_valuation",
    "b_sector_from", "b_entity_from", "b_region_from",
    "b_sector_to",   "b_entity_to",   "b_region_to",
    "b_time", "b_valuation",
    "value_nominal", "unc_type", "value_sd", "value_min", "value_max",
    "pedigree_reliability", "pedigree_completeness",
    "pedigree_temporal_correlation", "pedigree_geographical_correlation",
    "pedigree_other_correlation",
    "epsilon", "bound_lower", "bound_upper", "source", "comment",
]

# Edge selectors per indicator (numerator a, denominator b). All filled at
# write time except a_region_from / b_region_from / a_region_to / b_region_to
# which the loop over regions injects. Wildcards on time/valuation mean the
# resolver expands across all years and both Mass/CC valuations.
INDICATOR_EDGES = {
    "MESC": {
        "a": dict(sector_from="Cathode", entity_from="Flow", sector_to="MESC", entity_to="SB"),
        "b": dict(sector_from="Cathode", entity_from="Flow", sector_to="FU",   entity_to="SB"),
        "intra": False,
    },
    "USC": {
        "a": dict(sector_from="Cathode", entity_from="Flow", sector_to="USC", entity_to="SB"),
        "b": dict(sector_from="Cathode", entity_from="Flow", sector_to="FU",  entity_to="SB"),
        "intra": False,
    },
    "PSC": {
        "a": dict(sector_from="CP", entity_from="Process", sector_to="PSC",     entity_to="SB"),
        "b": dict(sector_from="CP", entity_from="Process", sector_to="Cathode", entity_to="Flow"),
        "intra": True,
    },
}


def _slug(region: str) -> str:
    """region → lower-case CSV-safe slug (strips commas, replaces spaces)."""
    return (region.lower()
                  .replace(",", "")
                  .replace(".", "")
                  .replace(" ", "_"))


def load_values() -> pd.DataFrame:
    df = pd.read_csv(VALUES_CSV)
    return df


def filter_series(df: pd.DataFrame, filt: dict, name: str) -> pd.Series:
    mask = np.ones(len(df), dtype=bool)
    for col, val in filt.items():
        mask &= df[col] == val
    return df[mask].groupby(["Region", "Year"])["Value"].sum().rename(name)


def summed_filter_series(df: pd.DataFrame, terms: list, name: str) -> pd.Series:
    parts = [filter_series(df, filt, name="_part") * scale for filt, scale in terms]
    out = parts[0]
    for p in parts[1:]:
        out = out.add(p, fill_value=0)
    return out.rename(name)


def load_eff_value(param_id: str) -> float:
    eff = pd.read_csv(EFF_PARAMS_CSV, sep=";")
    row = eff[eff["param_id"] == param_id]
    if row.empty:
        raise KeyError(f"{param_id} not found in {EFF_PARAMS_CSV}")
    return float(row["value_nominal"].iloc[0])


def ratio_frame(values: pd.Series, denom: pd.Series) -> pd.DataFrame:
    df = pd.concat([values, denom], axis=1).dropna(subset=["value"])
    df = df[df["denom"].notna() & (df["denom"] != 0)]
    df["ratio"] = df["value"].abs() / df["denom"]
    return df


def cluster_regions_largest_gap(per_region: pd.DataFrame, names: list) -> dict:
    """1D 2-means via largest-gap split on log10(per-region p95).

    Zero-p95 regions are forced into the lowest cluster. Returns a mapping
    {region: cluster_name}.
    """
    assert len(names) == 2, "only k=2 supported here"
    nonzero = per_region[per_region["p95"] > 0].copy()
    log_p95 = np.log10(nonzero["p95"]).sort_values()
    if len(log_p95) < 2:
        return {r: names[0] for r in per_region.index}

    diffs = log_p95.diff().dropna()
    split_idx = int(diffs.values.argmax())   # position in sorted-by-log_p95
    sorted_regions = log_p95.index.tolist()
    sorted_vals = log_p95.values
    threshold = (sorted_vals[split_idx] + sorted_vals[split_idx + 1]) / 2

    assignment = {}
    for r in per_region.index:
        if r in nonzero.index and np.log10(per_region.loc[r, "p95"]) > threshold:
            assignment[r] = names[1]
        else:
            assignment[r] = names[0]
    return assignment


def per_region_table(frame: pd.DataFrame) -> pd.DataFrame:
    g = frame.groupby(level="Region")["ratio"]
    table = pd.DataFrame({
        "n":      g.count(),
        "mean":   g.mean(),
        "median": g.median(),
        "p95":    g.quantile(0.95),
        "max":    g.max(),
    }).sort_values("p95", ascending=False)
    return table


def pooled_stats(frame: pd.DataFrame) -> dict:
    r = frame["ratio"].to_numpy()
    v = frame["value"].to_numpy()
    return {
        "n":         int(len(r)),
        "mean":      float(r.mean()),
        "median":    float(np.median(r)),
        "p5":        float(np.quantile(r, 0.05)),
        "p25":       float(np.quantile(r, 0.25)),
        "p75":       float(np.quantile(r, 0.75)),
        "p95":       float(np.quantile(r, 0.95)),
        "p99":       float(np.quantile(r, 0.99)),
        "min_signed": float(v.min()),
        "max_signed": float(v.max()),
        "max_abs":    float(r.max()),
    }


def print_pooled(name: str, stats: dict, denom_label: str) -> None:
    print(f"\n=== {name} — pooled |ratio| = |{name}| / [{denom_label}] ===")
    print(f"  n          : {stats['n']}")
    print(f"  mean       : {stats['mean']:.4f}")
    print(f"  median     : {stats['median']:.4f}")
    print(f"  p5         : {stats['p5']:.4f}")
    print(f"  p25        : {stats['p25']:.4f}")
    print(f"  p75        : {stats['p75']:.4f}")
    print(f"  p95        : {stats['p95']:.4f}   <-- BOUND")
    print(f"  p99        : {stats['p99']:.4f}")
    print(f"  max |ratio|: {stats['max_abs']:.4f}")
    print(f"  signed min : {stats['min_signed']:.2f} kt")
    print(f"  signed max : {stats['max_signed']:.2f} kt")


def print_per_region(name: str, table: pd.DataFrame) -> None:
    print(f"\n--- {name} per-region (sorted by p95 desc) ---")
    fmt = table.copy()
    for c in ("mean", "median", "p95", "max"):
        fmt[c] = fmt[c].map(lambda x: f"{x:.4f}")
    print(fmt.to_string())


def plot_figure(results: dict) -> None:
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), constrained_layout=True)
    for ax, (name, payload) in zip(axes, results.items()):
        ratios = payload["frame"]["ratio"].to_numpy()
        stats = payload["stats"]
        region_medians = payload["per_region"]["median"].to_numpy()
        denom_lbl = payload["denominator_label"]

        ax.hist(ratios, bins=40, color="#3A86A8", alpha=0.75, edgecolor="white")
        ax.axvline(stats["median"], color="#2A9D8F", lw=1.5, ls="--",
                   label=f"median = {stats['median']:.3f}")

        cluster_stats = payload.get("cluster_stats")
        if cluster_stats:
            cluster_colors = {"hub": "#D62828", "domestic": "#F4A261"}
            for cluster, cs in cluster_stats.items():
                color = cluster_colors.get(cluster, "#D62828")
                ax.axvline(cs["p95"], color=color, lw=2,
                           label=f"p95 [{cluster}] = {cs['p95']:.3f}")
            title_extra = " | ".join(
                f"{c}: p95={s['p95']:.3f} (n={s['n']})"
                for c, s in cluster_stats.items()
            )
        else:
            ax.axvline(stats["p95"], color="#D62828", lw=2,
                       label=f"p95 = {stats['p95']:.3f}")
            title_extra = (f"n={stats['n']}  mean={stats['mean']:.3f}  "
                           f"p95={stats['p95']:.3f}  max={stats['max_abs']:.3f}")

        ymax = ax.get_ylim()[1]
        ax.scatter(region_medians, np.full_like(region_medians, ymax * 0.03),
                   marker="|", color="black", s=60, label="per-region medians")

        ax.set_title(
            f"{name}  |  denom = {denom_lbl}  |  {title_extra}",
            fontsize=9,
        )
        ax.set_xlabel(f"|{name}| / [{denom_lbl}]")
        ax.set_ylabel("count")
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("ICSG stock-change bounds — empirical |ratio| distributions", fontsize=12)
    fig.savefig(FIG_PATH, dpi=130)
    plt.close(fig)
    print(f"\nFigure written to {FIG_PATH.relative_to(REPO_ROOT)}")


def _build_param_row(indicator: str, region: str, p95: float,
                     cluster_label: str, denom_lbl: str) -> dict:
    """Build one params.csv row: agg=true ratio bound for (indicator, region)."""
    spec = INDICATOR_EDGES[indicator]
    a = spec["a"]; b = spec["b"]; intra = spec["intra"]

    # RatioBoundBuilder's CSV convention is a-side = DENOMINATOR, b-side = NUMERATOR:
    # the agg row is `Σb − r·Σa`, bounding b/a ≤ r (CONTEXT.md §"Node-scoped vs
    # set-scoped"; ADR 0005). INDICATOR_EDGES labels the stock edge "a" (the natural
    # numerator) and the denominator "b", so we write them into the CSV *swapped* —
    # denominator → a_*, stock → b_*. A straight a→a / b→b mapping inverts the bound
    # to `denom ≤ p95·stock` (forces stock ≥ denom/p95 ~20–42×; the cathode-pool collapse).
    denom, num = b, a   # CSV a-side = denominator, CSV b-side = numerator (stock change)

    a_region_from = region if intra else "*"
    b_region_from = region if intra else "*"
    a_region_to   = region
    b_region_to   = region

    slug = _slug(region)
    ind = indicator.lower()
    # param_id is used as the constraint-id `variant` slot, which must be
    # underscore-free (see constraints/naming._safe); strip separators.
    param_id = f"net_stock_bound_{ind}_{slug}".replace("_", "")
    descr = (f"{indicator} bound — region={region}, cluster={cluster_label}: "
             f"|Σ {ind}| ≤ {p95:.4f} · [Σ {denom_lbl}] over all from_regions")

    return {
        "param_id":   param_id,
        "param_type": "stock_bound",
        "agg":        "true",
        "description": descr,
        "a_sector_from": denom["sector_from"], "a_entity_from": denom["entity_from"],
        "a_region_from": a_region_from,
        "a_sector_to":   denom["sector_to"],   "a_entity_to":   denom["entity_to"],
        "a_region_to":   a_region_to,
        "a_time": "*", "a_valuation": "CC",
        "b_sector_from": num["sector_from"], "b_entity_from": num["entity_from"],
        "b_region_from": b_region_from,
        "b_sector_to":   num["sector_to"],   "b_entity_to":   num["entity_to"],
        "b_region_to":   b_region_to,
        "b_time": "*", "b_valuation": "CC",
        "value_nominal": 0,
        "unc_type": "",
        # RatioBoundBuilder reads the band from value_min/value_max (bound_lower/upper
        # are legacy/provenance) — write the ±p95 share into both.
        "value_sd": "", "value_min": -p95, "value_max": +p95,
        "pedigree_reliability": "", "pedigree_completeness": "",
        "pedigree_temporal_correlation": "",
        "pedigree_geographical_correlation": "",
        "pedigree_other_correlation": "",
        "epsilon": "",
        "bound_lower": -p95,
        "bound_upper": +p95,
        "source": f"ICSG values.csv 1995-2024, p95 of |{indicator}|/[{denom_lbl}]",
        "comment": f"cluster={cluster_label}",
    }


def write_params_csv(per_indicator_p95: dict, regions: list) -> None:
    """Emit one row per (region, indicator). MESC uses the region's cluster p95;
    USC/PSC use a single global p95 each."""
    PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for region in regions:
        for indicator, payload in per_indicator_p95.items():
            if isinstance(payload["p95"], dict):
                # cluster-specific p95: payload["p95"] maps cluster_name → p95
                cluster = payload["region_to_cluster"].get(region, payload["default_cluster"])
                p95 = payload["p95"][cluster]
            else:
                cluster = "all_regions"
                p95 = payload["p95"]
            rows.append(_build_param_row(
                indicator=indicator,
                region=region,
                p95=p95,
                cluster_label=cluster,
                denom_lbl=payload["denominator_label"],
            ))
    df = pd.DataFrame(rows, columns=PARAMS_COLUMNS)
    out = PARAMS_DIR / "params.csv"
    df.to_csv(out, sep=";", index=False)
    print(f"Params written to {out.relative_to(REPO_ROOT)} ({len(rows)} rows)")


def main() -> None:
    df = load_values()

    eff_er = load_eff_value(REFINING_EFF_PARAM_ID)
    print(f"Using {REFINING_EFF_PARAM_ID} = {eff_er} for the anode→cathode "
          f"contribution to the PSC denominator (empirical study only).")
    # WARNING (known bug — do not ship a regenerated PSC bound without fixing this):
    # the `Type="Anode"` term below matches ZERO rows — the ICSG feed has no `Anode`
    # type (types are Cathode/Concentrate/Matte/PSC/Scrap). So the PSC denominator
    # silently collapses to SXEW-only (~16% of refined cathode), inflating PSC's p95
    # ~70× (0.05 → 3.52). The shipped PSC bound is the validated v3 value (0.05); a
    # correct refined-cathode-production denominator + data-provenance check are a
    # separate follow-up. USC/MESC (Cathode→FU denominator) are unaffected.
    DENOMINATORS["PSC"] = [
        (dict(Type="Cathode", Supply_Use="Supply", Process="SXEW"), 1.0),
        (dict(Type="Anode",   Supply_Use="Use",    Process="Refining"), eff_er),
    ]

    results = {}
    for name, filt in INDICATORS.items():
        values = filter_series(df, filt, name="value")
        denom = summed_filter_series(df, DENOMINATORS[name], name="denom")
        frame = ratio_frame(values, denom)
        results[name] = dict(
            frame=frame,
            stats=pooled_stats(frame),
            per_region=per_region_table(frame),
            denominator_label=DENOM_LABELS[name],
        )

    for name, payload in results.items():
        print_pooled(name, payload["stats"], payload["denominator_label"])
        print_per_region(name, payload["per_region"])

    # Per-indicator bound spec used by write_params_csv. For MESC, p95 is a
    # dict keyed by cluster; for USC/PSC, a single global float.
    per_indicator_p95: dict = {}
    for name, payload in results.items():
        denom_lbl = payload["denominator_label"]
        if name in CLUSTERED:
            spec = CLUSTERED[name]
            mapping = cluster_regions_largest_gap(
                payload["per_region"], names=spec["names"]
            )
            payload["clusters"] = mapping
            frame = payload["frame"].copy()
            frame["cluster"] = frame.index.get_level_values("Region").map(mapping)
            payload["cluster_stats"] = {}

            cluster_p95: dict = {}
            for cluster in spec["names"]:
                sub = frame[frame["cluster"] == cluster]
                regions_in_cluster = sorted(
                    {r for r, c in mapping.items() if c == cluster}
                )
                if sub.empty:
                    print(f"\n[!] {name} cluster '{cluster}' is empty — skipping")
                    continue
                stats = pooled_stats(sub)
                payload["cluster_stats"][cluster] = stats
                cluster_p95[cluster] = stats["p95"]
                print(f"\n--- {name} cluster '{cluster}' "
                      f"(n_regions={len(regions_in_cluster)}, n_obs={stats['n']}) ---")
                print(f"  regions: {', '.join(regions_in_cluster)}")
                print(f"  pooled p95 = {stats['p95']:.4f}   "
                      f"median = {stats['median']:.4f}   "
                      f"max = {stats['max_abs']:.4f}")
            per_indicator_p95[name] = {
                "p95": cluster_p95,
                "region_to_cluster": mapping,
                "default_cluster":   spec["names"][0],   # catch-all for regions absent from ICSG
                "denominator_label": denom_lbl,
            }
        else:
            per_indicator_p95[name] = {
                "p95": payload["stats"]["p95"],
                "denominator_label": denom_lbl,
            }

    # Enumerate every region in the accounting structure so absent-from-ICSG
    # regions (e.g. ROW, regions with no MESC observations) still get a bound.
    print(f"\nLoading AccountingStructure '{ACC_ID}' from {ACC_DIR}")
    acc = AccountingStructure.load(ACC_ID, storage_dir=str(ACC_DIR))
    regions = sorted(acc.node_index["region"].unique().tolist())
    print(f"  enumerated {len(regions)} regions from acc.node_index")

    plot_figure(results)
    write_params_csv(per_indicator_p95, regions)


if __name__ == "__main__":
    main()
