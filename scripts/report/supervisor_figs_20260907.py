"""Figures for the 2026-09-07 supervisor one-pager (docs/rep/supervisor_20260907_masscc-status.md).

Reads only the shipped Round-B run's edge diagnostics
(data/proc/opt/opt_20260903_124655_4b23df2f/diag/edges_*.csv) and writes three PNGs
to figs/report/:

  fig1_sector_totals.png   -- world use-edge totals per sector and year (Mass + CC)
  fig2_sankey_1995.png     -- global 1995 Sankey, all regions aggregated (Mass + CC)
  fig3_adaptation.png      -- prior->posterior adaptation heatmap, sector x region

Run:  python scripts/report/supervisor_figs_20260907.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap

RUN_DIR = Path("data/proc/opt/opt_20260903_124655_4b23df2f/diag")
OUT_DIR = Path("figs/report")
YEARS = range(1995, 2025)

# validated categorical palette (dataviz reference, light mode, fixed order)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK, INK2 = "#0b0b0b", "#52514e"

SECTOR_LABEL = {
    "CO": "Crude ore (in situ)", "Mining": "Mining", "Ore": "Ore",
    "MF": "Milling & flotation", "SXEW": "SX-EW (leaching)",
    "Concentrate": "Concentrate", "Smelting": "Smelting", "Matte": "Matte",
    "CFR": "Converting & fire refining", "Anode": "Anode",
    "ER": "Electrolytic refining", "CER": "Refined Cu (ER)",
    "CSXEW": "Cathode (SX-EW)", "CP": "Cathode production", "Cathode": "Cathode",
    "Scrap": "Scrap", "Tailings": "Tailings", "Slag": "Slag",
    "FU": "Final use", "MESC": "Exchange stocks", "USC": "User stocks",
    "PSC": "PSC", "WR": "WR",
}
CHAIN_ORDER = ["CO", "Mining", "Ore", "MF", "SXEW", "Concentrate", "Smelting",
               "Matte", "CFR", "Anode", "ER", "CER", "CSXEW", "CP", "Cathode", "Scrap"]


def load_edges() -> pd.DataFrame:
    cols = ["from_sector", "from_region", "from_entity", "to_sector", "to_entity",
            "valuation", "a_prior", "a_post", "sigma_a"]
    frames = []
    for y in YEARS:
        df = pd.read_csv(RUN_DIR / f"edges_{y}.csv", usecols=cols)
        df["year"] = y
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- fig 1
def fig1_sector_totals(edges: pd.DataFrame) -> None:
    use = edges[edges.from_entity == "Flow"]
    tot = use.groupby(["valuation", "from_sector", "year"]).a_post.sum()
    order = (tot.loc["Mass"].groupby("from_sector").mean()
             .sort_values(ascending=False).index.tolist())

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharex=True)
    for ax, val, title in zip(axes, ["Mass", "CC"],
                              ["Gross mass", "Contained copper"]):
        for sec, color in zip(order, PALETTE):
            s = tot.loc[(val, sec)]
            ax.plot(s.index, s.values, color=color, lw=2, label=SECTOR_LABEL[sec])
        ax.set_yscale("log")
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("year", color=INK2)
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.grid(axis="y", color="#e6e5e0", lw=0.7)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(range(1995, 2025, 5))
        # direct end labels (contrast relief), nudged apart in log space
        ends = sorted(((tot.loc[(val, sec, 2024)], sec, c)
                       for sec, c in zip(order, PALETTE)), key=lambda t: t[0])
        ys = np.log10([e[0] for e in ends])
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + 0.13)
        for (v, sec, c), y in zip(ends, ys):
            ax.annotate(SECTOR_LABEL[sec], (2024, v), xytext=(2024.6, 10 ** y),
                        fontsize=8, color=c, va="center")
        ax.set_xlim(1995, 2033)
    axes[0].set_ylabel("world use-edge total  [kt / a]", color=INK2)
    axes[0].annotate("crude-ore\nreporting ends 2022", xy=(2022.5, 8e5),
                     xytext=(2010.5, 2.2e5), fontsize=8.5, color=INK2,
                     arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    fig.suptitle("Reconciled world copper flows by commodity, 1995–2024",
                 fontsize=13, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_DIR / "fig1_sector_totals.png", dpi=200,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("fig1: 2015 Mass totals check:",
          {s: f"{tot.loc[('Mass', s, 2015)]:.3g}" for s in order[:3]})


# ---------------------------------------------------------------- fig 2
# Display contraction: commodity (Flow) nodes become the bands between process
# nodes -- the classic MFA depiction; every band value equals the flow-sector
# total, splits proportional where a commodity has several destinations.
CONTRACT = ["Ore", "Concentrate", "Matte", "Anode", "CER", "CSXEW", "Cathode"]
SANKEY_LABEL = {
    "CO": "Crude ore", "Mining": "Mining", "MF": "Milling & flotation",
    "SXEW": "SX-EW", "Smelting": "Smelting", "CFR": "Converting",
    "ER": "El. refining", "CP": "Cathode prod.",
    "Scrap": "Scrap", "Tailings": "Tailings", "Slag": "Slag",
    "FU": "Final use", "MESC": "Exchange stocks", "USC": "User stocks",
}
NODE_X = {"CO": 0.001, "Mining": 0.13, "MF": 0.26, "SXEW": 0.26,
          "Tailings": 0.38, "Scrap": 0.32, "Smelting": 0.42, "CFR": 0.545,
          "ER": 0.69, "Slag": 0.77, "CP": 0.81, "FU": 0.999, "MESC": 0.999,
          "USC": 0.999}
NODE_Y = {"CO": 0.30, "Mining": 0.30, "MF": 0.25, "SXEW": 0.72, "Tailings": 0.88,
          "Scrap": 0.01, "Smelting": 0.20, "CFR": 0.20, "ER": 0.20, "Slag": 0.92,
          "CP": 0.25, "FU": 0.20, "MESC": 0.68, "USC": 0.85}
LOSS = {"Tailings", "Slag"}
# system-boundary sources/sinks (SB entity): drawn recessive + offset from the
# process chain so boundary crossings read as leaving/entering the system
SB_NODES = {"CO", "Scrap", "Tailings", "Slag", "FU", "MESC", "USC"}


def _contract_flows(links: pd.DataFrame) -> pd.DataFrame:
    for f in CONTRACT:
        ins, outs = links[links.dst == f], links[links.src == f]
        keep = links[(links.src != f) & (links.dst != f)]
        new = [{"src": i.src, "dst": o.dst,
                "a_post": i.a_post * o.a_post / outs.a_post.sum()}
               for i in ins.itertuples() for o in outs.itertuples()]
        links = pd.concat([keep, pd.DataFrame(new)], ignore_index=True)
    return links.groupby(["src", "dst"], as_index=False).a_post.sum()


def _sankey_trace(links: pd.DataFrame, domain_y: tuple[float, float],
                  min_label_share: float = 0.0) -> go.Sankey:
    links = _contract_flows(links)
    nodes = [n for n in NODE_X if n in set(links.src) | set(links.dst)]
    idx = {n: i for i, n in enumerate(nodes)}
    lcol = ["rgba(0,0,0,0.13)" if d in LOSS
            else "rgba(235,104,52,0.45)" if s == "Scrap"
            else "rgba(42,120,214,0.30)" for s, d in zip(links.src, links.dst)]
    # hide labels of hairline nodes (Mass panel: the refining chain is <1% of the
    # ore band; the CC panel right below names the same columns)
    size = {n: max(links[links.src == n].a_post.sum(),
                   links[links.dst == n].a_post.sum()) for n in nodes}
    cut = max(size.values()) * min_label_share
    labels = [SANKEY_LABEL[n] if size[n] >= cut and n != "CP" else ""
              for n in nodes]  # CP label collides with the left-drawn FU label
    return go.Sankey(
        domain=dict(y=domain_y),
        arrangement="snap",
        textfont=dict(size=13, color=INK),
        node=dict(label=labels,
                  x=[NODE_X[n] for n in nodes], y=[NODE_Y[n] for n in nodes],
                  color=["#b3b2a9" if n in SB_NODES else "#363636" for n in nodes],
                  pad=10, thickness=8,
                  line=dict(width=0)),
        link=dict(source=[idx[s] for s in links.src],
                  target=[idx[d] for d in links.dst],
                  value=links.a_post, color=lcol),
    )


def fig2_sankey_1995(edges: pd.DataFrame) -> None:
    e95 = edges[edges.year == 1995]
    parts, notes = [], []
    for val in ["Mass", "CC"]:
        li = (e95[e95.valuation == val]
              .groupby(["from_sector", "to_sector"]).a_post.sum()
              .reset_index().rename(columns={"from_sector": "src", "to_sector": "dst"}))
        li = li[li.a_post > 1.0]
        parts.append(li)
        # mass-closure sanity: worst node in-out imbalance (excl. boundary nodes)
        bal = {}
        for n in set(li.src) | set(li.dst):
            if n in {"CO", "Scrap", "Tailings", "Slag", "FU", "MESC", "USC"}:
                continue
            inn, out = li[li.dst == n].a_post.sum(), li[li.src == n].a_post.sum()
            bal[n] = abs(inn - out) / max(inn, out)
        worst = max(bal, key=bal.get)
        print(f"fig2 {val}: worst node imbalance {worst} {bal[worst]:.1%}")
    mass, cc = parts
    fig = go.Figure([_sankey_trace(mass, (0.58, 0.97), min_label_share=0.025),
                     _sankey_trace(cc, (0.0, 0.42))])
    gt = mass[mass.src == "CO"].a_post.sum() / 1e6
    cath = mass[(mass.src == "CP") & (mass.dst == "Cathode")].a_post.sum() / 1e3
    ccm = cc[cc.src == "Mining"].a_post.sum() / 1e3
    fig.update_layout(
        width=1600, height=900, font=dict(color=INK, size=12),
        paper_bgcolor="white",
        margin=dict(l=30, r=140, t=90, b=55),
        title=dict(text="The reconciled copper system, 1995 — world total",
                   font=dict(size=19), x=0.02, y=0.975),
        annotations=[
            dict(text=f"<b>Gross mass</b> — {gt:.2f} Gt crude ore mined → "
                      f"{cath:.1f} Mt cathode (band width in kt)",
                 x=0, y=1.005, xref="paper", yref="paper", showarrow=False,
                 font=dict(size=12, color=INK2), xanchor="left"),
            dict(text=f"<b>Contained copper</b> — same system in Cu content "
                      f"({ccm:.1f} Mt mined); grey losses, orange scrap",
                 x=0, y=0.47, xref="paper", yref="paper", showarrow=False,
                 font=dict(size=12, color=INK2), xanchor="left"),
            dict(text="<i>dark bars = processes · grey bars = system boundary "
                      "(sources/sinks)<br>bands = commodities "
                      "(ore → concentrate → matte → anode → refined Cu → cathode)</i>",
                 x=0, y=-0.03, xref="paper", yref="paper", showarrow=False,
                 font=dict(size=12, color=INK2), xanchor="left", yanchor="top",
                 align="left"),
        ])
    fig.write_image(OUT_DIR / "fig2_sankey_1995.png", scale=2)


# ---------------------------------------------------------------- fig 3
def fig3_adaptation(edges: pd.DataFrame) -> None:
    m = edges[(edges.valuation == "Mass") & (edges.sigma_a < 1e5) & (edges.a_prior > 0)]
    m = m.assign(adiff=(m.a_post - m.a_prior).abs())
    cell = m.groupby(["from_region", "from_sector"])[["adiff", "a_prior"]].sum()
    rank = m.groupby("from_region").adiff.sum().sort_values(ascending=False)
    top = rank.head(20).index.tolist()
    sectors = [s for s in CHAIN_ORDER if s in m.from_sector.unique()]

    rows = top + ["all other regions"]
    grid = np.full((len(rows), len(sectors)), np.nan)
    other = m[~m.from_region.isin(top)].groupby("from_sector")[["adiff", "a_prior"]].sum()
    for j, sec in enumerate(sectors):
        for i, reg in enumerate(top):
            if (reg, sec) in cell.index:
                d, p = cell.loc[(reg, sec)]
                grid[i, j] = d / p
        if sec in other.index and other.loc[sec, "a_prior"] > 0:
            grid[-1, j] = other.loc[sec, "adiff"] / other.loc[sec, "a_prior"]

    cmap = LinearSegmentedColormap.from_list("blue", ["#f6f9fd", "#2a78d6", "#0c2f5c"])
    fig, ax = plt.subplots(figsize=(9.5, 7.6))
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(sectors)),
                  [SECTOR_LABEL[s] for s in sectors], rotation=40, ha="right",
                  fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)), rows, fontsize=8.5, color=INK)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isfinite(v) and v >= 0.005:
                ax.text(j, i, f"{v:.2f}".lstrip("0") if v < 1 else f"{v:.1f}",
                        ha="center", va="center", fontsize=6.4,
                        color="white" if v > 0.55 else INK)
    ax.set_title("Where reconciliation moves the priors most\n"
                 "tonnage-weighted |posterior − prior| / prior, gross mass, "
                 "1995–2024 pooled (data-backed flows only)",
                 fontsize=11.5, color=INK, loc="left")
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.015)
    cb.outline.set_visible(False)
    cb.set_label("relative adaptation (1.0 = moved by 100 % of prior)",
                 fontsize=8.5, color=INK2)
    cb.ax.tick_params(labelsize=8, colors=INK2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_adaptation.png", dpi=200,
                facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("fig3: top regions by |Δ|:", top[:8])


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    edges = load_edges()
    fig1_sector_totals(edges)
    fig2_sankey_1995(edges)
    fig3_adaptation(edges)
    print("done ->", OUT_DIR)
