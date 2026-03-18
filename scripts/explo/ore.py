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

def st_reg_bar(df, figp, c_sel):
    df = df.copy()

    # Ensure numeric
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['value'])

    # -----------------------------
    # 2. Filter data
    # -----------------------------
    df_top = df[df['country_name'].isin(c_sel)]

    # -----------------------------
    # 3. Global totals
    # -----------------------------
    global_total = df.groupby('year')['value'].sum().rename('global_total')

    # -----------------------------
    # 4. Country-year aggregation
    # -----------------------------
    country_year = (
        df_top.groupby(['year', 'country_name'])['value']
        .sum()
        .reset_index()
        .merge(global_total, on='year')
    )

    # Share
    country_year['share'] = country_year['value'] / country_year['global_total']

    # -----------------------------
    # 5. Pivot
    # -----------------------------
    pivot_df = (
        country_year
        .pivot(index='year', columns='country_name', values='share')
        .fillna(0)
    )

    # Sort columns by 2022 importance (nice stacking order)
    pivot_df = pivot_df[pivot_df.loc[2022].sort_values(ascending=False).index]

    # -----------------------------
    # 6. Modern styling
    # -----------------------------
    plt.style.use('default')  # clean base

    fig, ax = plt.subplots(figsize=(14, 7))

    color_list = [colors_new.get(col, '#CCCCCC') for col in pivot_df.columns]

    pivot_df.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        width=0.85,
        color=color_list
    )

    # -----------------------------
    # 7. Styling tweaks
    # -----------------------------
    #ax.set_title("Global Ore Production Shares (Top Producers: 1970 & 2022)", fontsize=14, pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("Share of Global Production", fontsize=11)

    # Format y-axis as %
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")

    # Reduce x ticks (avoid clutter)
    step = max(1, len(pivot_df.index) // 15)
    ax.set_xticks(range(0, len(pivot_df.index), step))
    ax.set_xticklabels(pivot_df.index[::step], rotation=45)

    # Clean look
    ax.spines[['top', 'right']].set_visible(False)

    # Legend outside
    ax.legend(
        title="Country",
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        frameon=False
    )

    plt.tight_layout()

    # -----------------------------
    # 8. Save
    # -----------------------------
    plt.savefig(os.path.join(figp, 'ore_production_shares_modern.png'), dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':

    figp = r'/mnt/nfs_fineprint/tmp/PYSUT/pySUTO/figs/ex'
    p = r'/mnt/nfs_fineprint/tmp/PYSUT/pySUTO/data/input/raw/Crude_ore_Kroiss/Global_crude_metal_ore_extraction_1970-2022_07.2024.xlsx'
    sn = 'Metal_extraction_1970-2022'

    df = pd.read_excel(p, sheet_name=sn)

    df = df[df.material_name == 'Copper - associated ore']

    # Ensure correct types
    df['year'] = df['year'].astype(int)
    df['value'] = df['value'].astype(float, errors='raise') // 1e6  # Convert to million tonnes

    c_select = ['USSR', 'Russian Federation', 'Chile', 'China', 'United States of America', 'Peru', 'Mexico', 'DR Congo', 'Indonesia', 'Serbia', 'Zambia', 'Canada']

    st_reg_bar(df, figp, c_select)