"""World copper end use by IWCC category, 2012-2024 (IWCC 2025 release).

Reads the 2025-release workbook only,
data/input/raw/IWCC/End-Use-Summary-Copper-content(1).xlsx (sheet "Copper
Content": sheet rows 3-5 = group / subgroup / leaf headers, row 6 = descriptions
with "Total" in the last column, data from row 8; column A = year, B-Q = the 16
leaf end-use categories, R = Total). The 2024 release next to it is left alone.

``build_payload(path)`` is the one parse: it checks the layout (column count,
category header row, integer years) and that the 16 categories sum to Total,
raising ValueError otherwise, and returns the ``enduse`` payload of the insight
artifact (#45). Both outputs are drawn from that payload:

  docs/rep/figs/iwcc_enduse/iwcc_enduse.json  -- the payload (tracked; the page
                                                 assembler merges it under ``enduse``)
  figs/ex/iwcc_enduse_absolute.png  -- kt Cu per category (lines) + world total (top strip)
  figs/ex/iwcc_enduse_shares.png    -- share of world total per category (100% stacked bars)

and the headline numbers the page prose quotes are printed (world totals,
x2024/2012 per category, group shares 2012/2023/2024, 2023->2024 share movers).

Run:  python scripts/explo/iwcc_enduse.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SRC = Path("data/input/raw/IWCC/End-Use-Summary-Copper-content(1).xlsx")
OUT_DIR = Path("figs/ex")
JSON_OUT = Path("docs/rep/figs/iwcc_enduse/iwcc_enduse.json")

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

SOURCE = "IWCC end-use summary, 2025 release, copper-content basis"
# measured against the 2024 release: 2023 world total 29,948.6 -> 30,335.3 kt Cu
REVISION = "2021–2023 revised vs the 2024 release (2023 world total +1.3 %)"
SHEET, HEADER_ROW, DATA_ROW, TOTAL_COL = "Copper Content", 4, 7, 17
# 0-based row 4 (sheet row 5), columns 1-16, as IWCC types the leaf headers; compared
# without whitespace and case, since the source breaks words ("Commun ications")
HEADERS = ("Plumbing", "Building Plant", "Architecture", "Communications",
           "Electrical Power", "Power Utility", "Telecommunications", "Electrical",
           "Non Electrical", "Automotive Electrical", "Automotive Non Electrical",
           "Other Vehicles", "Consumer and general products", "Cooling equipment",
           "Electronic equipment", "Diverse")


def _key(cell) -> str:
    return "".join(str(cell).split()).casefold()


def build_payload(path: Path) -> dict:
    """Parse the IWCC end-use workbook into the page's ``enduse`` payload.

    Parameters
    ----------
    path : Path
        The 2025-release workbook (sheet "Copper Content").

    Returns
    -------
    dict
        ``source`` label, ``years`` (int), ``groups`` (the six IWCC headers in
        source order), ``categories`` (per leaf in source order: ``label``,
        ``group``, ``kt`` series) and the world ``total`` series, kt Cu.
    """
    raw = pd.read_excel(path, sheet_name=SHEET, header=None)
    if raw.shape[1] != TOTAL_COL + 1:
        raise ValueError(f"{Path(path).name}: expected {TOTAL_COL + 1} columns (year, "
                         f"16 categories, Total), found {raw.shape[1]}")
    found = [*raw.iloc[HEADER_ROW, 1:TOTAL_COL], raw.iloc[HEADER_ROW + 1, TOTAL_COL]]
    wrong = [(c, f, h) for c, (f, h) in enumerate(zip(found, HEADERS + ("Total",)), 1)
             if _key(f) != _key(h)]
    if wrong:
        raise ValueError(f"{Path(path).name}: the category header row does not match "
                         f"the 2025-release layout; (column, found, expected): {wrong}")
    data = raw.iloc[DATA_ROW:, :]
    year = pd.to_numeric(data.iloc[:, 0], errors="coerce")
    bad = year.isna() | (year != year.round())
    if bad.any():
        raise ValueError(f"{Path(path).name}: non-year cells in column 0 of the data "
                         f"block: {data.iloc[:, 0][bad].tolist()}")
    years = [int(y) for y in year]
    total = data.iloc[:, TOTAL_COL].astype(float).to_numpy()
    leaf_sum = data.iloc[:, 1:TOTAL_COL].astype(float).to_numpy().sum(axis=1)
    off = ~np.isclose(leaf_sum, total)
    if off.any():
        raise ValueError(f"{Path(path).name}: the 16 categories do not sum to Total in "
                         f"{[y for y, o in zip(years, off) if o]}")
    categories = [
        {"label": label, "group": group, "kt": _kt(data.iloc[:, col].astype(float))}
        for group, leaves in GROUPS for col, label, _ in leaves]
    return {"source": SOURCE, "years": years, "groups": [g for g, _ in GROUPS],
            "categories": categories, "total": _kt(total)}


def _kt(values) -> list[float]:
    """Plain floats rounded to 0.1 kt, like the run's insight ``data.json``."""
    return [round(float(v), 1) for v in values]


def build_series(payload: dict) -> tuple[pd.DataFrame, list[dict]]:
    """The payload as a year-indexed frame (16 categories + Total) and plot spec."""
    style_of = {label: (color, DASHES[i % len(DASHES)])
                for _, leaves in GROUPS for i, (_, label, color) in enumerate(leaves)}
    df = pd.DataFrame({c["label"]: c["kt"] for c in payload["categories"]},
                      index=payload["years"])
    df["Total"] = payload["total"]
    spec = [{"label": c["label"], "group": c["group"],
             "color": style_of[c["label"]][0], "dashes": style_of[c["label"]][1]}
            for c in payload["categories"]]
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

    fig.suptitle(f"World copper end use by category, {years[0]}–{years[-1]}",
                 fontsize=13, color=INK, x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.935, SOURCE, fontsize=9, color=INK2, ha="left")
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

    fig.suptitle(f"Composition of world copper end use, {years[0]}–{years[-1]}",
                 fontsize=13, color=INK, x=0.012, ha="left", y=0.985)
    fig.text(0.012, 0.925, f"{SOURCE}; categories sum to 100% each year; {REVISION}",
             fontsize=9, color=INK2, ha="left")
    fig.savefig(OUT_DIR / "iwcc_enduse_shares.png", dpi=200,
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def print_headline(df: pd.DataFrame, spec: list[dict]) -> None:
    """The numbers the page prose quotes, computed from the payload's 0.1 kt values."""
    labels = [s["label"] for s in spec]
    y0, yp, y1 = df.index[0], df.index[-2], df.index[-1]
    tot = df["Total"]
    share = df[labels].div(tot, axis=0) * 100
    print(f"{SOURCE}: {len(spec)} categories, {y0}-{y1}")
    print(f"world total [kt Cu]: {y0} {tot[y0]:,.1f} | {yp} {tot[yp]:,.1f} | "
          f"{y1} {tot[y1]:,.1f}; {y1} {(tot[y1] / tot[y0] - 1) * 100:+.1f} % vs {y0}, "
          f"{(tot[y1] / tot[yp] - 1) * 100:+.1f} % vs {yp}")
    factor = (df.loc[y1, labels] / df.loc[y0, labels]).sort_values(ascending=False)
    print(f"x{y1}/{y0}: " + "; ".join(f"{k} {v:.2f}" for k, v in factor.items()))
    group = share.T.groupby([s["group"] for s in spec], sort=False).sum().T
    print(f"group share of world total [%] {y0}/{yp}/{y1}: " + "; ".join(
        f"{g} {group.loc[y0, g]:.1f}/{group.loc[yp, g]:.1f}/{group.loc[y1, g]:.1f}"
        for g in group.columns))
    delta = share.loc[y1] - share.loc[yp]
    move = delta[delta.abs().sort_values(ascending=False).index]
    print(f"share movers {yp}->{y1} [% of world total]: " + "; ".join(
        f"{k} {share.loc[yp, k]:.1f}->{share.loc[y1, k]:.1f} ({v:+.1f} pp)"
        for k, v in move.items()))


def main() -> None:
    payload = build_payload(SRC)
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, separators=(",", ":")))
    df, spec = build_series(payload)
    fig_absolute(df, spec)
    fig_shares(df, spec)
    print_headline(df, spec)
    print("palette:", ",".join(s["color"] for s in spec))


if __name__ == "__main__":
    main()
