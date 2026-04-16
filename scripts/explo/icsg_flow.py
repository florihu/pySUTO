#!/usr/bin/env python3
"""
icsg_flow.py  —  ICSG copper-flow figures

Usage
-----
Run all figures:
    python icsg_flow.py

Run specific figures:
    python icsg_flow.py --figures mining smelting refining

Available figures
-----------------
  mining               Concentrates share by region (stacked bar)
  smelting             Matte share by region (stacked bar)
  refining             Refining cathode share by region (stacked bar)
  sxew                 SXEW cathode share by region (stacked bar)
  secondary_smelting   Secondary share of smelting over time (line, multi-country)
  secondary_refining   Secondary share of refining over time (line, multi-country)
  semi                 Semi-fabrication share by region (stacked bar)
  flows_global         Absolute supply by region + process (log scale, line)
  flows_process        Absolute supply by process (log scale, line)
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# =============================================================================
# PATHS  (relative to project root, two levels above this script)
# =============================================================================

_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = str(_ROOT / 'data/proc/feedprep/icsg_clean/icsg_flows.csv')
FIG_PATH  = str(_ROOT / 'figs/ex')

# =============================================================================
# COLOR PALETTE
# =============================================================================

COLORS = {
    'United States':  '#457B9D',
    'Chile':          '#2A9D8F',
    'China':          '#E76F51',
    'Peru':           '#E9C46A',
    'Australia':      '#F4A261',
    'Brazil':         '#52B788',
    'Argentina':      '#3D9970',
    'Zambia':         '#A44A3F',
    'Poland':         '#8D99AE',
    'Russia':         '#3D5A80',
    'Kazakhstan':     '#B5838D',
    'Mexico':         '#C9A84C',
    'Germany':        '#6B8F71',
    'Japan':          '#264653',
    'Canada':         '#81B29A',
    'Congo, D.R.':    '#6A4C93',
    'Colombia':       '#2B9EB3',
    'India':          '#F28C38',
    'South Korea':    '#9B5DE5',
    'Rest of World':  '#D9D9D9',
    'France':         '#E63946',
    'Italy':          '#06A77D',
    'Taiwan':         '#118AB2',
    'United Kingdom': '#073B4C',
}

# =============================================================================
# DEFAULT COUNTRY SELECTIONS PER FIGURE
# =============================================================================

C_MINING = [
    "United States", "Chile", "China", "Peru", "Australia",
    "Congo, D.R.", "Argentina", "Zambia", "Poland", "Russia",
    "Kazakhstan", "Canada",
]
C_SMELTING = [
    "United States", "Chile", "China", "Peru", "Australia",
    "Japan", "Argentina", "Zambia", "Poland", "Russia",
    "Kazakhstan", "Mexico", "Germany",
]
C_REFINING = [
    "United States", "Chile", "China", "Peru", "Canada",
    "Japan", "South Korea", "Taiwan", "Zambia", "Poland",
    "Russia", "Colombia", "India", "Germany",
]
C_SXEW = [
    "United States", "Chile", "China", "Peru", "Australia",
    "Congo, D.R.", "Argentina", "Zambia", "Poland", "Russia",
    "Kazakhstan", "Canada",
]
C_SECONDARY_SMELTING = [
    "United States", "Chile", "China", "Peru", "Australia",
    "Japan", "Argentina", "Zambia", "Poland", "Russia",
    "Kazakhstan", "Mexico", "Germany",
]
C_SECONDARY_REFINING = [
    "United States", "Chile", "China", "Peru", "Canada",
    "Japan", "Zambia", "Poland", "Russia", "India", "Germany",
]
C_SEMI = [
    "United States", "Chile", "China", "Peru", "Canada",
    "Japan", "Italy", "South Korea", "Russia", "Germany",
    "France", "Taiwan", "United Kingdom",
]

ALL_FIGURES = [
    'mining', 'smelting', 'refining', 'sxew',
    'secondary_smelting', 'secondary_refining',
    'semi', 'flows_global', 'flows_process',
]

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load and return the ICSG flows CSV."""
    return pd.read_csv(path)

# =============================================================================
# CORE PLOT HELPERS
# =============================================================================

def plot_region_share(f, var, colors, c_select, ax=None):
    """Stacked bar chart: regional share of `var` over time."""
    d = f[f['Type'] == var].copy()
    data = d.groupby(['Year', 'Region'])['Value'].sum().reset_index()
    total = data.groupby('Year')['Value'].sum().rename('Total')
    data = data.join(total, on='Year')
    data['Share'] = data['Value'] / data['Total']

    data['Region_plot'] = data['Region'].where(
        data['Region'].isin(c_select), 'Rest of World'
    )
    data = data.groupby(['Year', 'Region_plot'])['Share'].sum().reset_index()

    df_plot = data.pivot(index='Year', columns='Region_plot', values='Share').fillna(0)
    order = [c for c in c_select if c in df_plot.columns] + \
            (['Rest of World'] if 'Rest of World' in df_plot.columns else [])
    df_plot = df_plot[order]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    df_plot.plot(
        kind='bar', stacked=True, ax=ax,
        color=[colors.get(c, '#D9D9D9') for c in df_plot.columns],
        edgecolor='none', width=0.8,
    )
    ax.set_ylabel('Share')
    ax.set_xlabel('Year')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return ax


def plot_secondary_share(f, c_color, process='Smelting', countries=None):
    """
    Two-subplot figure for a given process across multiple countries.
      Left  : absolute Primary (solid) and Secondary (dashed) timeseries per country.
      Right : secondary share = Secondary / (Primary + Secondary) per country.
    """
    df = f[
        (f['Process'] == process) &
        (f['Supply_Use'] == 'Supply')
    ].copy()
    df['Type'] = df['Type'].replace({'Total': 'Primary'})

    agg = df.groupby(['Region', 'Year', 'Type'], as_index=False)['Value'].sum()
    pivot = agg.pivot_table(
        index=['Region', 'Year'], columns='Type', values='Value', aggfunc='sum'
    ).reset_index()
    pivot.columns.name = None

    if 'Total' in pivot.columns:
        pivot['_total'] = pivot['Total']
    elif 'Primary' in pivot.columns and 'Secondary' in pivot.columns:
        pivot['_total'] = pivot[['Primary', 'Secondary']].sum(axis=1)
    else:
        raise ValueError(f"Cannot compute total — columns: {pivot.columns.tolist()}")

    if 'Secondary' not in pivot.columns:
        raise ValueError(f"No 'Secondary' type found for process='{process}'")

    if 'Primary' not in pivot.columns:
        pivot['Primary'] = 0.0

    pivot['secondary_share'] = pivot['Secondary'] / pivot['_total'] * 100
    pivot = pivot.dropna(subset=['secondary_share'])
    pivot = pivot[pivot['_total'] > 0]

    if countries is not None:
        missing = [c for c in countries if c not in pivot['Region'].unique()]
        if missing:
            print(f"  [warn] countries not found in data: {missing}")
        pivot = pivot[pivot['Region'].isin(countries)]

    if pivot.empty:
        raise ValueError("No data left after country filter.")

    fig, (ax_abs, ax_sh) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'{process} — Primary & Secondary Production',
                 fontsize=14, fontweight='bold', y=1.01)

    for region, d in pivot.groupby('Region'):
        c = c_color.get(region, 'grey')
        d = d.sort_values('Year')

        # left: absolute primary (solid) and secondary (dashed)
        ax_abs.plot(d['Year'], d['Primary'],   color=c, linewidth=1.4,
                    linestyle='-',  alpha=0.85, zorder=2)
        ax_abs.plot(d['Year'], d['Secondary'], color=c, linewidth=1.4,
                    linestyle='--', alpha=0.85, zorder=2)

        # right: secondary share — use scatter for the legend handle
        ax_sh.plot(d['Year'], d['secondary_share'],
                   color=c, linewidth=1.4, alpha=0.85, zorder=2)
        ax_sh.scatter(d['Year'], d['secondary_share'],
                      color=c, s=22, zorder=3,
                      edgecolors='white', linewidths=0.4, label=region)

    # left axis styling
    ax_abs.set_title('Absolute output (kt)', fontsize=12)
    ax_abs.set_xlabel('Year', fontsize=11)
    ax_abs.set_ylabel('kt', fontsize=11)
    ax_abs.grid(axis='y', linestyle='--', alpha=0.4, color='grey')
    ax_abs.spines[['top', 'right']].set_visible(False)
    # manual legend for line styles only (no country labels)
    from matplotlib.lines import Line2D
    ax_abs.legend(
        handles=[
            Line2D([0], [0], color='#555', linewidth=1.4, linestyle='-',  label='Primary'),
            Line2D([0], [0], color='#555', linewidth=1.4, linestyle='--', label='Secondary'),
        ],
        frameon=False, fontsize=10, loc='upper left',
    )

    # right axis styling
    ax_sh.set_title('Secondary share (%)', fontsize=12)
    ax_sh.set_xlabel('Year', fontsize=11)
    ax_sh.set_ylabel('Secondary / Total (%)', fontsize=11)
    ax_sh.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax_sh.set_ylim(0, 105)
    ax_sh.grid(axis='y', linestyle='--', alpha=0.4, color='grey')
    ax_sh.spines[['top', 'right']].set_visible(False)

    full_primary = (
        pivot.groupby('Region')['secondary_share']
        .max()
        .pipe(lambda s: s[s <= 1.0])
        .index.tolist()
    )
    if full_primary:
        label_text = "~100% primary:\n" + ", ".join(sorted(full_primary))
        ax_sh.annotate(label_text, xy=(1.01, 0), xycoords='axes fraction',
                       va='bottom', ha='left', fontsize=9, color='#444444',
                       annotation_clip=False,
                       bbox=dict(boxstyle='round,pad=0.4', fc='#f5f5f5',
                                 ec='#cccccc', lw=0.8))

    n = pivot['Region'].nunique()
    ax_sh.legend(loc='upper left', bbox_to_anchor=(1.01, 1),
                 fontsize=10, ncol=2 if n > 15 else 1,
                 frameon=False, title='Country', title_fontsize=11,
                 markerscale=2.0, handlelength=1.5)

    fig.tight_layout()
    return fig


# =============================================================================
# INDIVIDUAL FIGURE FUNCTIONS
# =============================================================================

def fig_mining(f, fig_path):
    ax = plot_region_share(f, 'Concentrates', COLORS, C_MINING)
    ax.set_title('Mining — Concentrates Share by Region',
                 fontsize=13, fontweight='bold')
    p = os.path.join(fig_path, 'concentrates_share.png')
    plt.savefig(p, dpi=300, bbox_inches='tight')
    print(f'  Saved: {p}')
    plt.show()


def fig_smelting(f, fig_path):
    sm = f[f.Process == 'Smelting']
    sm_grouped = sm.groupby(['Year', 'Region'])['Value'].sum().reset_index()
    sm_grouped['Type'] = 'Matte'
    ax = plot_region_share(sm_grouped, 'Matte', COLORS, C_SMELTING)
    ax.set_title('Smelting — Matte Share by Region',
                 fontsize=13, fontweight='bold')
    p = os.path.join(fig_path, 'matte_share.png')
    plt.savefig(p, dpi=300, bbox_inches='tight')
    print(f'  Saved: {p}')
    plt.show()


def fig_refining(f, fig_path):
    rf = f[f.Process == 'Refining']
    grouped = rf.groupby(['Year', 'Region'])['Value'].sum().reset_index()
    grouped['Type'] = 'Cathode'
    ax = plot_region_share(grouped, 'Cathode', COLORS, C_REFINING)
    ax.set_title('Refining — Cathode Share by Region',
                 fontsize=13, fontweight='bold')
    p = os.path.join(fig_path, 'refining_share.png')
    plt.savefig(p, dpi=300, bbox_inches='tight')
    print(f'  Saved: {p}')
    plt.show()


def fig_sxew(f, fig_path):
    sx = f[f.Process == 'SXEW']
    ax = plot_region_share(sx, 'Cathode', COLORS, C_SXEW)
    ax.set_title('SXEW — Cathode Share by Region',
                 fontsize=13, fontweight='bold')
    p = os.path.join(fig_path, 'sxew_share.png')
    plt.savefig(p, dpi=300, bbox_inches='tight')
    print(f'  Saved: {p}')
    plt.show()


def fig_secondary_smelting(f, fig_path):
    fig = plot_secondary_share(f, COLORS,
                               countries=C_SECONDARY_SMELTING,
                               process='Smelting')
    p = os.path.join(fig_path, 'secondary_share_smelting.png')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    print(f'  Saved: {p}')
    plt.show()


def fig_secondary_refining(f, fig_path):
    fig = plot_secondary_share(f, COLORS,
                               countries=C_SECONDARY_REFINING,
                               process='Refining')
    p = os.path.join(fig_path, 'secondary_share_refining.png')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    print(f'  Saved: {p}')
    plt.show()


def fig_semi(f, fig_path):
    sm = f[(f.Process == 'Semi') & (f.Supply_Use == 'Supply')]
    grouped = sm.groupby(['Year', 'Region'])['Value'].sum().reset_index()
    grouped['Type'] = 'Semi'
    ax = plot_region_share(grouped, 'Semi', COLORS, C_SEMI)
    ax.set_title('Semi-fabrication — Share by Region',
                 fontsize=13, fontweight='bold')
    p = os.path.join(fig_path, 'semi_share.png')
    plt.savefig(p, dpi=300, bbox_inches='tight')
    print(f'  Saved: {p}')
    plt.show()


def fig_flows_global(f, fig_path):
    a = f[f.Supply_Use == 'Supply']
    g = a.groupby(['Year', 'Process', 'Region'])['Value'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=g[g.Value > 1], x='Year', y='Value',
                 hue='Region', style='Process', ax=ax)
    ax.set_yscale('log')
    ax.set_title('Absolute Supply by Region and Process (log scale)', fontsize=13)
    ax.set_ylabel('Supply (kt, log scale)')
    ax.set_xlabel('Year')
    ax.legend([], [], frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    p = os.path.join(fig_path, 'flows_global.png')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    print(f'  Saved: {p}')
    plt.show()


def fig_flows_process(f, fig_path):
    a = f[f.Supply_Use == 'Supply']
    g = a.groupby(['Year', 'Process'])['Value'].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=g[g.Value > 1], x='Year', y='Value', hue='Process', ax=ax)
    ax.set_yscale('log')
    ax.set_title('Absolute Supply by Process (log scale)', fontsize=13)
    ax.set_ylabel('Supply (kt, log scale)')
    ax.set_xlabel('Year')
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    p = os.path.join(fig_path, 'flows_process.png')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    print(f'  Saved: {p}')
    plt.show()


# =============================================================================
# DISPATCH TABLE
# =============================================================================

FIGURE_FUNCS = {
    'mining':             fig_mining,
    'smelting':           fig_smelting,
    'refining':           fig_refining,
    'sxew':               fig_sxew,
    'secondary_smelting': fig_secondary_smelting,
    'secondary_refining': fig_secondary_refining,
    'semi':               fig_semi,
    'flows_global':       fig_flows_global,
    'flows_process':      fig_flows_process,
}

# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--figures', nargs='+', metavar='FIG',
        choices=ALL_FIGURES + ['all'],
        default=['all'],
        help=(
            'Which figures to generate. Pass one or more names, or "all" '
            f'(default). Available: {", ".join(ALL_FIGURES)}'
        ),
    )
    parser.add_argument(
        '--data', default=DATA_PATH,
        help='Path to icsg_flows.csv.',
    )
    parser.add_argument(
        '--out', default=FIG_PATH,
        help='Directory where figures are saved.',
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    figures = ALL_FIGURES if 'all' in args.figures else args.figures

    print(f'Loading data from: {args.data}')
    f = load_data(args.data)
    print(f'  {len(f):,} rows loaded.\n')

    for name in figures:
        print(f'── {name} ──')
        FIGURE_FUNCS[name](f, args.out)


if __name__ == '__main__':
    main()
