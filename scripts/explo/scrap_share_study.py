"""Per-region scrap-share bands for Smelting and CFR inputs (CC).

Reads the ICSG feed values and computes, per (region, year), the share of
secondary (Scrap) input in the total copper-content input of each consuming
process:

    Smelting:  share = Scrap/Smelting/Use / (Scrap/Smelting/Use + Concentrate/Smelting/Use)
    CFR:       share = Scrap/Refining/Use / (Refining/Anode/Use ÷ eff_CFR)

(the feed's ``Refining`` maps to the model's ``CFR``; all ICSG values are
copper content). The CFR denominator is the total CFR input inferred from the
anode leaving it: ADR 0021 moved the refined series onto the anode entering
ER, so there is no ``Matte/Refining/Use`` series any more, and the scrap
series is the direct-to-refinery route only (ADR 0030) — the shipped v1 bands
predate both and their CFR floors would force the double-counted scrap back
in (China floor 0.160 vs a direct-route share of 0.040). Per region the band is ``[max(0, p5−MARGIN),
min(1, p95+MARGIN)]`` of the observed share series (≥ MIN_OBS observations);
regions without a scrap series get the DEFAULT band. One params.csv row per
(region, process) with ``agg=true`` and ``param_type="scrap_share"`` so the
``RatioBoundBuilder`` emits the hard nonneg-cone band

    bl · Σ(all CC inputs to P in region R) ≤ Σ(Scrap→P, CC) ≤ bu · Σ(...)

Why hard-only (deliberate deviation from the ADR 0011/0012 one-relation-
drives-all pattern): a soft centre row pulling scrap toward median·inputs
would fight the NODATA_ANCHOR_CV tight-zero anchor on no-data regions, and
data-backed regions are already centred by their real ICSG priors. The band
exists to stop no-data Scrap→Process edges acting as free copper faucets
(they are in neither scale-total sum family; see REP
analysis_opt_20260901_080337_99aa122f_20260901_all-years-deviation-scan.md).

Regions are the concordance's ROOT names, not the raw feed's source labels
(ADR 0032) — ``South Korea`` is ``Korea, Rep.``, ``Belgium-Luxembourg`` is
``Belgium``, and the 48 ROW members aggregate into one band.

Share convention (GLOSSARY.md: "SEM-secondary (Scrap — missing counts as 0)"):
a share observation exists for every year the region's *scrap* series has a
value; the primary series (Concentrate / Matte) counts as 0 when missing, so
pure-secondary smelters (no concentrate series) correctly show share = 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pysuto.acc import AccountingStructure  # noqa: E402

PARAMS_DIR = REPO_ROOT / "data/input/params/scrap_share/current"
FIG_PATH = REPO_ROOT / "figs/ex/scrap_share.png"

ACC_ID = "copper"
ACC_DIR = REPO_ROOT / "data/proc/acc"

MARGIN = 0.05          # absolute widening of the [p5, p95] band
MIN_OBS = 3            # minimum scrap-series observations for a region-own band
DEFAULT_BAND = (0.0, 0.25)   # regions without a scrap series (covers China-level
                             # shares ~0.24; the NODATA_ANCHOR does the real work
                             # on fully-no-data regions like Panama/Taiwan)

# process → (scrap series filter, primary series filter). Feed "Refining" = model CFR.
PROCESSES = {
    "Smelting": {
        "scrap":   dict(Type="Scrap",       Process="Smelting", Supply_Use="Use"),
        "primary": dict(Type="Concentrate", Process="Smelting", Supply_Use="Use"),
        "primary_label": "Concentrate/Smelting/Use",
    },
    "CFR": {
        "scrap":   dict(Type="Scrap", Process="Refining", Supply_Use="Use"),
        # total CFR input = anode leaving CFR ÷ eff_CFR; the anode entering ER
        # (ADR 0021) is the observable closest to it
        "total":   dict(Type="Anode", Process="Refining", Supply_Use="Use"),
        "total_eff": "effcfr",
        "primary_label": "Refining/Anode/Use ÷ eff_CFR",
    },
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


def _slug(region: str) -> str:
    return (region.lower()
                  .replace(",", "")
                  .replace(".", "")
                  .replace(" ", "_"))


def to_root_regions(df: pd.DataFrame, feed_dir: Path) -> pd.DataFrame:
    """Map the raw feed's SOURCE region labels onto the model's root regions.

    ``values.csv`` carries ICSG's own labels (``South Korea``,
    ``Belgium-Luxembourg``, ``Taiwan, China``); the model's regions are the
    concordance's root names, and the map is applied downstream inside
    ``feedpipe``.  Grouping the raw file by ``Region`` and then looking the
    result up under a root name therefore matches nothing for any renamed or
    aggregated region: Korea and ROW fell back to DEFAULT_BAND, and Belgium's
    and Taiwan's bands were built from a partial series (ADR 0032).
    """
    rm = pd.read_excel(feed_dir / "conc.xlsx", sheet_name="region_map")
    m = dict(zip(rm["src_region"], rm["root_region"]))
    missing = sorted(set(df["Region"]) - set(m))
    assert not missing, f"feed regions absent from region_map: {missing}"
    return df.assign(Region=df["Region"].map(m))


def filter_series(df: pd.DataFrame, filt: dict, name: str) -> pd.Series:
    mask = np.ones(len(df), dtype=bool)
    for col, val in filt.items():
        mask &= df[col] == val
    return df[mask].groupby(["Region", "Year"])["Value"].sum().rename(name)


def share_frame(df: pd.DataFrame, proc: str, eff: dict | None = None) -> pd.DataFrame:
    """(Region, Year) → share, on years where the scrap series is observed.

    With a ``primary`` series: primary counts as 0 when missing (pure-secondary
    smelters → share 1). With a ``total`` series (CFR): the denominator is that
    series ÷ ``eff[total_eff]``, and years without it are dropped. Zero-total
    years are dropped (share undefined).
    """
    spec = PROCESSES[proc]
    scrap = filter_series(df, spec["scrap"], "scrap")
    if "total" in spec:
        total = filter_series(df, spec["total"], "total") / eff[spec["total_eff"]]
        fr = pd.concat([scrap, total], axis=1).dropna(subset=["scrap", "total"])
    else:
        primary = filter_series(df, spec["primary"], "primary")
        fr = pd.concat([scrap, primary], axis=1).dropna(subset=["scrap"])
        fr["primary"] = fr["primary"].fillna(0.0)
        fr["total"] = fr["scrap"] + fr["primary"]
    fr = fr[fr["total"] > 0]
    fr["share"] = (fr["scrap"] / fr["total"]).clip(upper=1.0)
    return fr


def region_bands(fr: pd.DataFrame) -> pd.DataFrame:
    """Per-region band [max(0, p5−MARGIN), min(1, p95+MARGIN)] + stats."""
    g = fr.groupby("Region")["share"]
    out = pd.DataFrame({
        "n":      g.size(),
        "p5":     g.quantile(0.05),
        "median": g.median(),
        "p95":    g.quantile(0.95),
    })
    out = out[out["n"] >= MIN_OBS]
    out["bl"] = np.maximum(0.0, out["p5"] - MARGIN)
    out["bu"] = np.minimum(1.0, out["p95"] + MARGIN)
    return out


def _build_param_row(proc: str, region: str, bl: float, bu: float,
                     nominal: float, n_obs: int, own: bool) -> dict:
    """One agg=true scrap-share band row for (process, region).

    CSV convention (same a/b swap as stock_bounds_study._build_param_row):
    a-side = DENOMINATOR (all CC inputs to the process), b-side = NUMERATOR
    (the Scrap→process CC leg). RatioBoundBuilder emits Σb ≤ bu·Σa and
    Σb ≥ bl·Σa. The denominator wildcards sector_from so it pools every
    input — concentrate/matte legs from any region plus the scrap leg —
    exactly the effsmelting pooled-denominator pattern (ADR 0005).
    """
    slug = _slug(region)
    # param_id fills the constraint-id `variant` slot → must be underscore-free.
    param_id = f"scrap_share_{proc.lower()}_{slug}".replace("_", "")
    prov = ("own series" if own
            else f"no scrap series — default band [{bl:.2f}, {bu:.2f}]")
    descr = (f"Scrap share of {proc} CC inputs — region={region}: "
             f"{bl:.3f} ≤ Σ(Scrap→{proc}) / Σ(all→{proc}) ≤ {bu:.3f} ({prov})")
    return {
        "param_id":   param_id,
        "param_type": "scrap_share",
        "agg":        "true",
        "description": descr,
        # a-side = DENOMINATOR: every CC use-edge into the process in this region
        "a_sector_from": "*",   "a_entity_from": "*", "a_region_from": "*",
        "a_sector_to":   proc,  "a_entity_to":   "Process",
        "a_region_to":   region,
        "a_time": "*", "a_valuation": "CC",
        # b-side = NUMERATOR: the Scrap→process CC leg (intra-region by topology)
        "b_sector_from": "Scrap", "b_entity_from": "SB", "b_region_from": "*",
        "b_sector_to":   proc,    "b_entity_to":   "Process",
        "b_region_to":   region,
        "b_time": "*", "b_valuation": "CC",
        # nominal = observed median share (provenance / soft-mode fallback)
        "value_nominal": round(nominal, 6),
        "unc_type": "",
        "value_sd": "", "value_min": round(bl, 6), "value_max": round(bu, 6),
        "pedigree_reliability": "", "pedigree_completeness": "",
        "pedigree_temporal_correlation": "",
        "pedigree_geographical_correlation": "",
        "pedigree_other_correlation": "",
        "epsilon": "",
        "bound_lower": round(bl, 6),
        "bound_upper": round(bu, 6),
        "source": ("ICSG values.csv 1995-2024, [p5-%.2f, p95+%.2f] of "
                   "Scrap/(Scrap+%s)" % (MARGIN, MARGIN,
                                         PROCESSES[proc]["primary_label"])),
        "comment": f"n_obs={n_obs}" if own else "default_band",
    }


def plot_figure(frames: dict, bands: dict) -> None:
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (proc, fr) in zip(axes, frames.items()):
        b = bands[proc].sort_values("median")
        ax.errorbar(range(len(b)), b["median"],
                    yerr=[b["median"] - b["bl"], b["bu"] - b["median"]],
                    fmt="o", ms=3, capsize=2, lw=0.8)
        ax.set_xticks(range(len(b)))
        ax.set_xticklabels(b.index, rotation=90, fontsize=6)
        ax.axhline(DEFAULT_BAND[1], color="grey", ls="--", lw=0.8,
                   label=f"default bu={DEFAULT_BAND[1]}")
        ax.set_title(f"{proc}: observed scrap share (median, band [bl, bu])")
        ax.set_ylabel("Scrap / total CC inputs")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150)
    print(f"Figure written to {FIG_PATH.relative_to(REPO_ROOT)}")


def _registry():
    from pysuto.input_registry import InputRegistries
    return InputRegistries.initialize(root=str(REPO_ROOT / "data/input"))


def _eff_nominals(inputs) -> dict:
    p = pd.read_csv(inputs.params.resolve("eff") / "params.csv", sep=None,
                    engine="python", comment="#", dtype=str)
    return {r["param_id"]: float(r["value_nominal"]) for _, r in p.iterrows()
            if r.get("param_type") == "efficiency"}


def main() -> None:
    inputs = _registry()
    feed_dir = inputs.feed.resolve("icsg")
    print(f"icsg feed ← {feed_dir}")
    df = to_root_regions(pd.read_csv(feed_dir / "values.csv"), feed_dir)
    eff = _eff_nominals(inputs)
    previous = pd.read_csv(inputs.params.resolve("scrap_share") / "params.csv", sep=";",
                           dtype=str) if (inputs.params.resolve("scrap_share") / "params.csv").exists() else None
    acc = AccountingStructure.load(ACC_ID, storage_dir=str(ACC_DIR))
    regions = sorted(acc.node_index["region"].unique())

    frames, bands, rows = {}, {}, []
    for proc in PROCESSES:
        fr = share_frame(df, proc, eff)
        b = region_bands(fr)
        frames[proc], bands[proc] = fr, b
        print(f"\n=== {proc}: {len(fr)} region-year share obs, "
              f"{len(b)} regions with ≥{MIN_OBS} obs ===")
        print(b.round(3).sort_values("median", ascending=False).head(12).to_string())
        for region in regions:
            if region in b.index:
                r = b.loc[region]
                rows.append(_build_param_row(proc, region, r["bl"], r["bu"],
                                             r["median"], int(r["n"]), own=True))
            else:
                bl, bu = DEFAULT_BAND
                rows.append(_build_param_row(proc, region, bl, bu,
                                             nominal=0.0, n_obs=0, own=False))

    new = pd.DataFrame(rows, columns=PARAMS_COLUMNS)
    if previous is not None:
        # ADR 0032 moves bands on BOTH processes — the regions whose source
        # label differs from their root name were never matched before — so the
        # v1→v2 "Smelting is untouched" invariant no longer holds. Report the
        # diff on both instead; the band that moved is the evidence.
        for proc in ("Smelting", "CFR"):
            cf = new[new.a_sector_to == proc].merge(
                previous[previous.a_sector_to == proc], on="param_id", suffixes=("", "_prev"))
            moved = (~np.isclose(cf.value_min.astype(float), cf.value_min_prev.astype(float))
                     | ~np.isclose(cf.value_max.astype(float), cf.value_max_prev.astype(float)))
            cf = cf[moved]
            print(f"\n{proc} bands changed in {len(cf)} regions:")
            print(cf[["a_region_to", "value_min_prev", "value_max_prev",
                      "value_min", "value_max", "comment"]].to_string(index=False))

    PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    out = PARAMS_DIR / "params.csv"
    new.to_csv(out, sep=";", index=False)
    print(f"\nParams written to {out.relative_to(REPO_ROOT)} ({len(rows)} rows: "
          f"{len(regions)} regions × {len(PROCESSES)} processes)")
    plot_figure(frames, bands)


if __name__ == "__main__":
    main()
