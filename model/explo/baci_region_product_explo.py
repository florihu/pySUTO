import sys
import os
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt
import polars as pl
import numpy as np

sns.set(style="white")

# Dynamically find the parent "code" directory
current_file = Path(__file__).resolve()
code_dir = current_file.parents[3]  # go up from explo -> analysis -> code

# Add 'code/model' to sys.path
sys.path.append(str(code_dir / 'model'))

from Baci import Baci
from util import save_fig
baci = Baci()


def region(target = 'from'):
    """
    This function is used to explore the coverage of the BACI data.

    I want to explore:
    1. Regions from /to with most Copper
    2. Regions from /to with most Copper per capita
    3. Highest flows per model variable
    """

    if target == 'from':
        target = 'Region_from_name'
    elif target == 'to':
        target = 'Region_to_name'

    # Read data from your model (returns a polars.DataFrame)
    df = baci.baci_read()

    # Add Copper_flow column (Quantity * Conc)
    df = df.with_columns(
        (pl.col('Quantity') * pl.col('Conc')).alias('Copper_flow')
    )

    # Group by Region_from_iso2 and Year, sum Copper_flow
    from_group = (
        df.group_by([target, 'Year'])
        .agg(pl.sum('Copper_flow').alias('Copper_flow'))
        .sort([target, 'Year'])
    )

    df = from_group.to_pandas()
    
    # convert into log 10 scale
    df['Copper_flow'] = np.log10(df['Copper_flow'] + 1)

    # biggest regions from in 2021
    biggest_regions = df[df['Year'] == 2021].nlargest(50, 'Copper_flow')[target].unique()
    biggest_regions = biggest_regions.tolist()

    # filter
    df = df[df[target].isin(biggest_regions)]

    # lineplot
    plt.figure(figsize=(20, 10))
    sns.lineplot(data=df, x='Year', y='Copper_flow', hue=target, palette='tab20', legend='full')
    plt.title('Copper flow from biggest 50 regions in 2021')

    plt.ylabel('Copper flow (log)')
    plt.xlabel('Year')

    plt.xticks(rotation=45)
    
    plt.tight_layout()

    # get the legend to the center under the plot
    plt.legend(loc='center', bbox_to_anchor=(0.5, -0.4), ncol=8, fontsize=8)

    plt.savefig(f'Copper_{target}_biggest_50_regions.pdf', dpi=300, bbox_inches='tight')

    plt.show()

    return None


def region_net_trade():
    """
    This function is used to explore the coverage of the BACI data.

    I want to explore:
    1. Regions from /to with most Copper
    2. Regions from /to with most Copper per capita
    3. Highest flows per model variable
    """


    # Read data from your model (returns a polars.DataFrame)
    df = baci.baci_read()

    # Add Copper_flow column (Quantity * Conc)
    df = df.with_columns(
        (pl.col('Quantity') * pl.col('Conc')).alias('Copper_flow')
    )

    # Group by Region_from_iso2 and Year, sum Copper_flow
    exp = (
        df.group_by(['Region_from_name', 'Year'])
        .agg(pl.sum('Copper_flow').alias('Copper_flow_exp'))
        .sort(['Region_from_name', 'Year'])
    )

    imp = (
        df.group_by(['Region_to_name', 'Year'])
        .agg(pl.sum('Copper_flow').alias('Copper_flow_imp'))
        .sort(['Region_to_name', 'Year'])

    )

    df = exp.join(imp, left_on=['Region_from_name', 'Year'], right_on=['Region_to_name', 'Year'], how='inner')
    
    df = df.to_pandas()

    df['Net_flow'] = df['Copper_flow_exp'] - df['Copper_flow_imp']

    # scale net flow to -1 to 1
    df['Net_flow_scale'] = 2 * (df['Net_flow'] - df['Net_flow'].min()) / (df['Net_flow'].max() - df['Net_flow'].min()) -1

    df['Net_flow_abs'] = df['Net_flow'].abs()

    # biggest regions from in 2021
    biggest_regions = df[df['Year'] == 2010].nlargest(50, 'Net_flow_abs')['Region_from_name'].unique()
    biggest_regions = biggest_regions.tolist()

    # filter
    df = df[df['Region_from_name'].isin(biggest_regions)]

    # heatmap
    df_pivot = df.pivot(index='Region_from_name', columns='Year', values='Net_flow_scale')
    df_pivot = df_pivot.fillna(0)
    plt.figure(figsize=(20, 10))
    sns.heatmap(df_pivot, cmap='RdBu', center=0, cbar_kws={'label': 'Net flow scale'}, annot=False)
    plt.title('Net flow from biggest 100 regions in 2010')
    plt.ylabel('Year')
    plt.xlabel('Region from')
    plt.xticks(rotation=45)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    save_fig('Net_flow_biggest_100_regions', dpi=300, format='pdf')
    plt.show()

    return None




if __name__ == "__main__":
    region_net_trade()