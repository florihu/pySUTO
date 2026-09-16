"""World copper end use by IWCC category, 2012-2023.

Reads data/input/raw/IWCC/IWCC_2024_End-use-summary-Copper-content.xlsx
(sheet "Copper Content", row 5 = the 16 leaf end-use categories) and writes
two PNGs to figs/ex/:

  iwcc_enduse_absolute.png  -- kt Cu per category (lines) + world total (top strip)
  iwcc_enduse_shares.png    -- share of world total per category (100% stacked bars)

Run:  python scripts/explo/iwcc_enduse.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SRC = Path("data/input/raw/IWCC/IWCC_2024_End-use-summary-Copper-content.xlsx")
OUT_DIR = Path("figs/ex")

INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#ffffff"
GRID = "#e6e5e0"

# row 5 leaves grouped by the row 3/4 hierarchy of the sheet. One hue family per
# group, lightness-stepped within it (OKLCH L 0.74->0.48, max in-gamut chroma);
# validated with the dataviz palette validator: lightness band and chroma floor
# PASS, within-family adjacent pairs sit below the separation floors -- inherent
# to 16 categorical slots, so line dashes and legend-order-equals-stack-order
# carry identity as secondary encoding.
GROUPS = [
    ("Building construction", [
        (1, "Plumbing", "#00bfdd"), (2, "Building plant", "#00a6d5"),
        (3, "Architecture", "#008cce"), (4, "Communications", "#0070ce"),
        (5, "Electrical power", "#2d51c1")]),
    ("Infrastructure", [
        (6, "Power utility", "#fb8900"), (7, "Telecommunications", "#ac1d03")]),
    ("Equipment — industrial", [
        (8, "Electrical", "#00c6ab"), (9, "Non-electrical", "#007137")]),
    ("Equipment — transport", [
        (10, "Automotive electrical", "#bcae00"),
        (11, "Automotive non-electrical", "#a57c00"),
        (12, "Other vehicles", "#854f00")]),
    ("Equipment — other", [
        (13, "Consumer & general products", "#d683f6"),
        (14, "Cooling equipment", "#9b61d9"),
        (15, "Electronic equipment", "#6440b8")]),
    ("Diverse", [(16, "Others", "#dd4f78")]),
]
DASHES = [(None, None), (5, 2), (1.4, 1.6), (7, 2, 1.4, 2), (3, 1.4, 1.4, 1.4)]


def build_series() -> tuple[pd.DataFrame, list[dict]]:
    raw = pd.ExcelFile(SRC).parse("Copper Content", header=None)
    data = raw.iloc[7:, :].astype(float).reset_index(drop=True)
    years = data.iloc[:, 0].astype(int).values
    series, spec = {}, []
    for group, leaves in GROUPS:
        for i, (col, label, color) in enumerate(leaves):
            series[label] = data.iloc[:, col].values
            spec.append({"label": label, "group": group, "color": color,
                         "dashes": DASHES[i % len(DASHES)]})
    df = pd.DataFrame(series, index=years)
    total = data.iloc[:, 17].values
    assert np.allclose(df.sum(axis=1).values, total), "leaves do not sum to Total"
    df["Total"] = total
    return df, spec


def legend_handles(spec, kind: str):
    """Grouped legend: invisible proxy entries carry the group headers."""
    handles, labels, header_rows = [], [], []
    for group, _ in GROUPS:
        header_rows.append(len(labels))
        handles.append(Line2D([], [], linestyle="none"))
        labels.append(group)
        for s in [s for s in spec if s["group"] == group]:
            if kind == "line":
                h = Line2D([], [], color=s["color"], lw=2, dashes=s["dashes"])
            else:
                h = Patch(facecolor=s["color"], edgecolor=SURFACE, lw=0.7)
            handles.append(h)
            labels.append("   " + s["label"])
    return handles, labels, header_rows


def place_legend(ax, spec, kind: str):
    handles, labels, header_rows = legend_handles(spec, kind)
    leg = ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                    frameon=False, fontsize=8.2, labelspacing=0.32,
                    handlelength=1.9, handletextpad=0.6, borderaxespad=0)
    for i, txt in enumerate(leg.get_texts()):
        txt.set_color(INK if i in header_rows else INK2)
        if i in header_rows:
            txt.set_fontweight("bold")
            txt.set_fontsize(8.4)
    return leg


def style(ax):
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)


def fig_absolute(df: pd.DataFrame, spec: list[dict]) -> None:
    years = df.index.values
    fig, (top, main) = plt.subplots(
        2, 1, figsize=(11.2, 6.6), sharex=True, height_ratios=(1, 4),
        gridspec_kw={"hspace": 0.12})

    top.plot(years, df["Total"], color=INK, lw=2)
    top.set_ylim(0, df["Total"].max() * 1.12)
    top.set_ylabel("world total", color=INK2, fontsize=8.5)
    top.annotate(f"{df['Total'].iloc[-1]:,.0f} kt", (years[-1], df["Total"].iloc[-1]),
                 xytext=(4, -1), textcoords="offset points", fontsize=8.5,
                 color=INK, va="center")
    style(top)

    for s in spec:
        main.plot(years, df[s["label"]], color=s["color"], lw=2,
                  dashes=s["dashes"], label=s["label"])
    main.set_ylim(0, df[[s["label"] for s in spec]].values.max() * 1.06)
    main.set_ylabel("copper content  [kt / a]", color=INK2)
    main.set_xlabel("year", color=INK2)
    main.set_xticks(years[::2])
    main.set_xlim(years[0] - 0.3, years[-1] + 0.3)
    style(main)
    place_legend(main, spec, "line")

    fig.suptitle("World copper end use by category, 2012–2023",
                 fontsize=13, color=INK, x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.935, "IWCC 2024 end-use summary, copper-content basis",
             fontsize=9, color=INK2, ha="left")
    fig.savefig(OUT_DIR / "iwcc_enduse_absolute.png", dpi=200,
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_shares(df: pd.DataFrame, spec: list[dict]) -> None:
    years = df.index.values
    labels = [s["label"] for s in spec]
    share = df[labels].div(df["Total"], axis=0) * 100

    fig, ax = plt.subplots(figsize=(11.2, 6.0))
    bottom = np.zeros(len(years))
    for s in reversed(spec):                      # first category ends up on top
        v = share[s["label"]].values
        ax.bar(years, v, bottom=bottom, width=0.74, color=s["color"],
               edgecolor=SURFACE, linewidth=0.7, label=s["label"])
        bottom += v
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 10)])
    ax.set_ylabel("share of world copper end use", color=INK2)
    ax.set_xlabel("year", color=INK2)
    ax.set_xticks(years)
    style(ax)
    place_legend(ax, spec, "bar")

    fig.suptitle("Composition of world copper end use, 2012–2023",
                 fontsize=13, color=INK, x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.925, "IWCC 2024 end-use summary, copper-content basis; "
             "categories sum to 100% each year", fontsize=9, color=INK2, ha="left")
    fig.savefig(OUT_DIR / "iwcc_enduse_shares.png", dpi=200,
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df, spec = build_series()
    fig_absolute(df, spec)
    fig_shares(df, spec)
    first, last = df.index[0], df.index[-1]
    print(f"{len(spec)} categories, {first}-{last}; "
          f"total {df['Total'].iloc[0]:,.0f} -> {df['Total'].iloc[-1]:,.0f} kt")
    print("palette:", ",".join(s["color"] for s in spec))


if __name__ == "__main__":
    main()
