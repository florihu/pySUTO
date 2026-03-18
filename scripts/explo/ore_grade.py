import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import os


colors_new = {
    'Chile':                     '#2A9D8F',  # teal (kept)
    'China':                     '#E76F51',  # muted red (kept)
    'United States of America':  '#457B9D',  # steel blue (adapted from US)
    'Peru':                      '#E9C46A',  # warm yellow (kept)
    'Mexico':                    '#C9A84C',  # golden ochre (kept)

    'DR Congo':                  '#6A4C93',  # muted purple (adapted from Congo, D.R.)
    'Indonesia':                 '#3A86A8',  # muted ocean blue (new, fits palette)
    'Serbia':                    '#8D99AE',  # cool gray-blue (similar to Poland)
    'Zambia':                    '#A44A3F',  # deep brick red (kept)
    'Canada':                    '#81B29A',  # muted mint (kept)
    'USSR':                     '#D62828',  # bright red (new, stands out for historical significance)
}

def plot_ore_grade_timeseries(df, figp=None):
    """
    Plot ore grades (1/factor) over time:
    - All countries thin black
    - Mean and 10/90 percentile in red
    - Annotate top 5 countries in 2022 with labels + connector lines
    """
    df = df.copy()

    # Clean and numeric
    df['factor'] = pd.to_numeric(df['factor'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['factor', 'year', 'alphanumiso'])
    df = df[df['factor'] > 0]

    # Compute ore grade
    df['grade'] = 1 / df['factor']
    df = df.groupby(['year', 'alphanumiso'])['grade'].mean().reset_index()

    # Pivot
    pivot_df = df.pivot(index='year', columns='alphanumiso', values='grade').sort_index()

    # Stats
    mean_series = pivot_df.mean(axis=1)
    q10 = pivot_df.quantile(0.1, axis=1)
    q90 = pivot_df.quantile(0.9, axis=1)

    # Top 5 countries in 2022
    if 2022 in pivot_df.index:
        top5 = pivot_df.loc[2022].nlargest(5).index
    else:
        top5 = pivot_df.loc[pivot_df.index.max()].nlargest(5).index

    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))

    # All countries thin black
    for col in pivot_df.columns:
        ax.plot(pivot_df.index, pivot_df[col], color='black', linewidth=0.5, alpha=0.6, zorder=1)

    # Highlight top 5 with thicker black
    for col in top5:
        ax.plot(pivot_df.index, pivot_df[col], linewidth=1.5, zorder=2)

    # Stats in red
    ax.plot(mean_series.index, mean_series, color='red', linewidth=2.5, label='Mean', zorder=4)
    ax.plot(q10.index, q10, color='red', linestyle='--', linewidth=1.5, zorder=3)
    ax.plot(q90.index, q90, color='red', linestyle='--', linewidth=1.5, zorder=3)
    ax.fill_between(mean_series.index, q10, q90, color='red', alpha=0.08, zorder=2)

    # Annotate top 5 with connector lines
    last_year = pivot_df.index.max()
    y_positions = np.linspace(0.8, 0.2, len(top5))  # distribute label vertical positions to avoid overlap (normalized)
    min_y, max_y = ax.get_ylim()
    for i, col in enumerate(top5):
        y_end = pivot_df.loc[last_year, col]
        # map normalized y_positions to actual y-axis
        y_label = min_y + y_positions[i]*(max_y - min_y)

        # Draw connector line
        ax.plot([last_year, last_year + 1.5], [y_end, y_label], color='black', linewidth=0.8, linestyle='-')
        # Draw label
        ax.text(last_year + 1.6, y_label, col, fontsize=10, va='center')

    # Styling
    ax.set_title("Ore Grade (1 / Factor) Over Time", fontsize=14)
    ax.set_xlabel("Year")
    ax.set_ylabel("Ore Grade")
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.15)

    # X ticks
    years = pivot_df.index
    step = max(1, len(years) // 15)
    ax.set_xticks(years[::step])
    ax.set_xticklabels(years[::step], rotation=45)

    # Extend x-axis for labels
    ax.set_xlim(years.min(), last_year + 5)

    plt.tight_layout()

    # Save figure
    if figp is not None:
        os.makedirs(figp, exist_ok=True)
        plt.savefig(os.path.join(figp, "ore_grade_timeseries_top5_annotated.png"), dpi=300, bbox_inches='tight')

    plt.show()

if __name__ == '__main__':

    figp = r'/mnt/nfs_fineprint/tmp/PYSUT/pySUTO/figs/ex'
    p = r'data/input/raw/Crude_ore_Kroiss/all_integrated_est_fac_2024-07-04.csv'
    sn = 'Metal_extraction_1970-2022'

    df = pd.read_csv(p, sep=';')
    df = df[df.material_id == 'Me.Cu']
    plot_ore_grade_timeseries(df, figp)


    
