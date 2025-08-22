"""

This script is used to make functions that convert input data into a standardized format.

"""

import os
import logging
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Now use an absolute import
from model.util import clean_cols, read_concordance_table
from model.core.datafeed.DataFeed import DataFeed


class Baci(DataFeed):
    """
    This class is used to read the BACI data and convert it into a standardized format.
    this is a sub class of the DATA_FEED class thereore it inherits all the functions of the DATA_FEED class
    
    """

    def __init__(self, name: str = 'baci', type_: str = 'flow'):
        """
        This function initializes the class and sets the path to the BACI data.
        """
        super().__init__(name=name)


        self.baci_rename = {
            't': 'Year',
            'k': 'HS92_Code',
            'i': 'Region_origin',
            'j': 'Region_destination',
            'v': 'Flow',
            'q': 'Quantity',
        }
        


    def baci_read(self):
        """
        This function reads the BACI data and converts it into a standardized format.
        """
        path = r'data\input\raw\BACI_HS92_V202501'
        country_path = r'data\input\raw\BACI_HS92_V202501\country_codes_V202501.csv'

        files = [f for f in os.listdir(path) if f.startswith('BACI')]
        df = pl.concat([pl.read_csv(os.path.join(path, f)) for f in files], how='vertical')
        df = df.rename(self.baci_rename)

        # Merge with Soulier concentration data
        soul = pl.from_pandas(self.soulier_concentration_read().reset_index())
        df = df.filter(pl.col('HS92_Code').is_in(soul['HS92_Code'].to_list()))
        df = df.join(soul, on='HS92_Code', how='left')

        # Merge with country names
        country_name = pl.read_csv(country_path)
        country_name = clean_cols(country_name)  # assumes this normalizes column names

        # merge to df
        df = df.join(country_name, left_on='Region_origin', right_on='Country_code', how='left')
        

        df = df.join(country_name, left_on='Region_destination', right_on='Country_code', how='left', suffix='_to')
        

        df = df.with_columns(
        (pl.col('Quantity') * pl.col('Conc')).alias('Copper_flow')
        )

        # rename columns consistently
        df = df.rename({'Country_name': 'Region_from_name', 
                        'Country_name_to': 'Region_to_name',

                        'Country_iso2': 'Region_from_iso2',
                        'Country_iso2_to': 'Region_to_iso2',

                        'Country_iso3': 'Region_from_iso3',
                        'Country_iso3_to': 'Region_to_iso3',})

        return df


    def baci_to_base(self):

        df = self.baci_read()

        # filter Year 1995
        df = df.filter(pl.col('Year') == 1995)

        return None

    def soulier_concentration_read(self):

        ''' 
        This function reads Soulier 2018 concentration, transform columns into the correct format, and returns a dataframe.
        // TODO: add uncertaity estimates for the concentrations

        

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

        # todo: 

        return df



def general_data_to_base(conc_path, data_path):

    'This function builds from concordance tables a concordance matrix and multiplies it with the data to be converted to the base classification.'

    # Define dimension names
    dimensions = [
            'Year', 'Layer', 'Scenario',
            'Region_origin', 'Sector_origin', 'Entity_origin',
            'Region_destination', 'Sector_destination', 'Entity_destination'
        ]
    # the concordance is stored in an exchel for every dimension e.g. Region i have a Source_name and Base_name column ... from this the concordance matrix is built
    return None

def read_base_table(self):
        """
        Reads the base table for the initial estimate.
        """
        # Assuming the base table is stored in a CSV file

        path = r'data\input\conc\base.xlsx'
        sheet_name = ['Year', 'Layer', 'Scenario', 'Region', 'Sector', 'Entity']
        relevant_cols = ['ID', 'Name']
        
        # read all the sheets in the excel file and return a dict with sheed name an df with the cols ID and Name
        base_table = pd.read_excel(path, sheet_name=sheet_name, usecols=relevant_cols)

        base = {}
        for sheet in base_table.keys():
            base[sheet] = base_table[sheet].set_index('ID')['Name'].to_dict()

        return 





if __name__ == '__main__':
    # Example usage
    path = r'data\input\conc\baci.xlsx'

    concordance = read_concordance_table(path)