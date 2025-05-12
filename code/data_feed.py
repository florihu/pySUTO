"""

This script is used to make functions that convert input data into a standardized format.

"""


import pandas as pd


def baci_read():
    return None

def soulier_concentration_read():

    ''' 
    This function reads Soulier 2018 concentration, transform columns into the correct format, and returns a dataframe.

    '''
    p = r'data\input\raw\Soulier_2018\Copper_HS92_Table_Merged.csv'
    df = pd.read_csv(p)

    # substitute all empty spaces in columns with _
    df.columns = df.columns.str.replace(' ', '_')

    df['HS92_Code'] = df['HS92_Code'].astype(int)

    # hs as index
    df.set_index('HS92_Code', inplace=True)

    # source has an import export split here why?
    df.loc[740400, 'Est._Content'] = '64%'

    df['Est._Content'] = df['Est._Content'].str.replace('%', '')

    df.rename(columns={'Est._Content': 'Conc'}, inplace=True)
    
    df['Conc'] = df['Conc'].astype(float) / 100

    # other columns as string
    df[['Model_Variable', 'Source_Code', 'Source_Content', 'Source_Variable']] = df[['Model_Variable', 'Source_Code', 'Source_Content', 'Source_Variable']].astype(str)

    return None


def get_data():
    """
    This function is used to get the data from the data source.
    """
    # Placeholder for data retrieval logic
    # In a real-world scenario, this could be replaced with code to fetch data from a database or an API
    data = {
        'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'value': [1, 2, 3]
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    soulier_concentration_read()