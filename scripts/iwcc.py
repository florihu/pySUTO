import sys
import os
import pandas as pd


from DataFeed import DataFeed



# Add the core directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from util import get_path

def init_iwcc_semi():
    file_name = r'IWCC_2024_Semis-production-and-demand.xlsx'
    path = get_path(file_name)
    sheet_name = 'Prod-Dem Summary External'
    return pd.read_excel(path, sheet_name=sheet_name, header=None)

def by_reg_year_clean(df):
    """
    Cleans the DataFrame by grouping by region and year, summing the values.
    """
    df = df.copy()
    df.iloc[0, 0] = 'Region'
    df.columns = df.iloc[0]
    df = df[1:]  # Remove header row
    df.set_index('Region', inplace=True)

    melt = df.melt(ignore_index=False, var_name='Year', value_name='Value')
    melt['Year'] = melt['Year'].astype(int)
    melt['Value'] = melt['Value'].astype(float)
    melt.reset_index(inplace=True)
    
    return melt

def process_iwcc_semi_data():
    df = init_iwcc_semi()

    all_data = []

    for i in range(0, len(df), 16):
        product_name = df.iloc[i, 1]
        product_block = df.iloc[i:i+14, :]  # 14 rows for product data

        production = product_block.iloc[2:, 0:13]
        demand = product_block.iloc[2:, [0] + list(range(13, product_block.shape[1]))]

        prod_clean = by_reg_year_clean(production)
        prod_clean['Type'] = 'Supply'
        prod_clean['Product'] = product_name

        dem_clean = by_reg_year_clean(demand)
        dem_clean['Type'] = 'Use'
        dem_clean['Product'] = product_name

        all_data.extend([prod_clean, dem_clean])

    final_df = pd.concat(all_data, ignore_index=True)
    return final_df

if __name__ == "__main__":
    df = process_iwcc_semi_data()
    print(df.head())