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
code_dir = current_file.parents[2]  # go up from explo -> model

# # Add 'code/model' to sys.path
sys.path.append(str(code_dir / 'core'))

from Baci import Baci
from util import save_fig
baci = Baci()



def export_concentrate_plot():
    """
    This function is used to explore the coverage of the BACI data.
    """

    df = baci.baci_read()

    # Zambia filter 
    zambia_filter = (pl.col('Region_from_name') == 'Zambia') & (pl.col('Model_Variable').is_in([ 'Concentrate', 'Intermediates']))
    zambia_df = df.filter(zambia_filter)

    df_export = zambia_df.to_pandas()
    df_grouped = df_export.groupby(['Region_to_name', 'Year']).agg({'Copper_flow': 'sum'}).reset_index()

    top_regions = df_grouped[df_grouped['Year'] == 2020].nlargest(10, 'Copper_flow')['Region_to_name'].tolist()
    df_grouped = df_grouped[df_grouped['Region_to_name'].isin(top_regions)]

    f, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(data=df_grouped, x='Year', y='Copper_flow', hue='Region_to_name', ax=ax, legend='full')

    # Move legend below the plot
    ax.legend(title='Region', bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=3)
    ax.set_xlabel('Year')
    ax.set_ylabel('Copper Exports (Tonnes)')
    ax.set_title('Copper Exports of Intermediates from Zambia by Region')

    # Adjust layout to avoid clipping
    plt.tight_layout()

    # Save with bbox_inches
    save_fig('copper_exports_zambia_by_region', dpi=300, bbox_inches='tight')
    plt.show()


def export_growth_plot():
    """
    This function is used to explore the coverage of the BACI data.
    """

    df = baci.baci_read()

    # Zambia filter 
    zambia_filter = (pl.col('Region_from_name') == 'Zambia') & (pl.col('Model_Variable').is_in([ 'Concentrate', 'Intermediates']))
    zambia_df = df.filter(zambia_filter)

    df_export = zambia_df.to_pandas()
    df_grouped = df_export.groupby(['Region_to_name', 'Year']).agg({'Copper_flow': 'sum'}).reset_index()

    df_grouped = df_grouped.sort_values(['Region_to_name', 'Year'])
    df_grouped['Copper_flow_growth'] = df_grouped.groupby('Region_to_name')['Copper_flow'].pct_change() * 100

    f, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(data=df_grouped, x='Year', y='Copper_flow_growth', hue='Region_to_name', ax=ax)

    # get hue levels an make a red line for switzerland
    hue_levels = df_grouped['Region_to_name'].unique()
    for level in hue_levels:
        if level == 'Switzerland':
            ax.lines[-len(hue_levels) + list(hue_levels).index(level)].set_color('red')

    # no legend
    ax.legend_.remove()
    ax.set_xlabel('Year')
    ax.set_ylabel('Copper Flow Growth Rate (%)')
    ax.set_title('Annual Copper Flow Growth of Intermediates from Zambia by Region')

    
    # Adjust layout to avoid clipping
    plt.tight_layout()

    # Save with bbox_inches
    save_fig('copper_exports_zambia_growth_by_region', dpi=300, bbox_inches='tight')
    plt.show()




def import_to_country(country = 'Switzerland'):
    """
    This function is used to explore the coverage of the BACI data.
    """
    df = baci.baci_read()

    # Zambia filter
    country_filter = (pl.col('Region_to_name') == country) & (pl.col('Model_Variable').is_in(['Concentrate', 'Intermediates']))
    country_df = df.filter(country_filter)

    df_export = country_df.to_pandas()
    df_grouped = df_export.groupby(['Region_from_name', 'Year']).agg({'Copper_flow': 'sum'}).reset_index()

    top_regions = df_grouped[df_grouped['Year'] == 2020].nlargest(10, 'Copper_flow')['Region_from_name'].tolist()
    df_grouped = df_grouped[df_grouped['Region_from_name'].isin(top_regions)]

    f, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(data=df_grouped, x='Year', y='Copper_flow', hue='Region_from_name', ax=ax, legend='full')

    # Move legend below the plot
    ax.legend(title='Region', bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=3)
    ax.set_xlabel('Year')
    ax.set_ylabel('Copper Imports (Tonnes)')
    ax.set_title(f'Copper Imports of Intermediates to {country} by Region')

    # Adjust layout to avoid clipping
    plt.tight_layout()

    # Save with bbox_inches
    save_fig(f'copper_imports_to_{country.lower()}_by_region', dpi=300, bbox_inches='tight')
    plt.show()


def trade_balance_plot(country = 'Switzerland'):
    """
    Plot the absolute and relative copper trade balance for Switzerland based on BACI data.
    Left: Net flow (exports - imports) in tonnes.
    Right: Year-over-year % change in net flow.
    """
    df = baci.baci_read()

    # Filter for Switzerland and selected model variables
    sw_filter = (
        ((pl.col('Region_from_name') == country) | (pl.col('Region_to_name') == country)) &
        (pl.col('Model_Variable').is_in(['Concentrate', 'Intermediates']))
    )
    df = df.filter(sw_filter)

    # Group exports
    exp = (
        df.group_by(['Region_from_name', 'Year'])
        .agg(pl.sum('Copper_flow').alias('Copper_flow_exp'))
        .sort(['Region_from_name', 'Year'])
    )

    # Group imports
    imp = (
        df.group_by(['Region_to_name', 'Year'])
        .agg(pl.sum('Copper_flow').alias('Copper_flow_imp'))
        .sort(['Region_to_name', 'Year'])
    )

    # Merge export and import tables
    df = exp.join(
        imp,
        left_on=['Region_from_name', 'Year'],
        right_on=['Region_to_name', 'Year'],
        how='inner'
    ).to_pandas()

    # Compute net flow and relative change
    df['Net_flow'] = df['Copper_flow_exp'] - df['Copper_flow_imp']
    df.sort_values(['Region_from_name', 'Year'], inplace=True)

    # compute absolute relative change
    df['Net_flow_rel_change'] = df.groupby('Region_from_name')['Net_flow'].diff()
    
    # Filter for Switzerland only
    df_sw = df[df['Region_from_name'] == country].copy()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    # Absolute trade balance
    sns.lineplot(data=df_sw, x='Year', y='Net_flow', ax=axes[0], color='steelblue', linewidth=2)
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=1)
    axes[0].set_title('Copper Trade Balance (Absolute)', fontsize=13)
    axes[0].set_ylabel('Net Flow (Tonnes)')
    axes[0].set_xlabel('Year')

    # Absolute Change
    sns.lineplot(data=df_sw, x='Year', y='Net_flow_rel_change', ax=axes[1], color='darkorange', linewidth=2)
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=1)
    axes[1].set_title('Copper Trade Balance (Absolute Change)', fontsize=13)
    axes[1].set_ylabel('Absolute Change (Tonnes)')
    axes[1].set_xlabel('Year')

    # Improve spacing
    plt.tight_layout()
    # Save the figure
    save_fig(f'copper_trade_balance_{country.lower()}', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    import_to_country('China')