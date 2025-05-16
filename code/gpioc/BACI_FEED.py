"""

This script is used to make functions that convert input data into a standardized format.

"""

import os
import logging
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns



class BACI_FEED:
    """
    This class is used to read the BACI data and convert it into a standardized format.
    """
    def __init__(self):
        """
        This function initializes the class and sets the path to the BACI data.
        """

        self.baci_rename = {
            't': 'Year',
            'k': 'HS92_Code',
            'i': 'Region_from',
            'j': 'Region_to',
            'v': 'Flow',
            'q': 'Quantity',
        }

    def baci_read(self):
        """
        This function reads the BACI data and converts it into a standardized format.
        see https://www.cepii.fr/DATA_DOWNLOAD/baci/doc/DescriptionBACI.html for more information on the BACI data
        """
        path = r'data\input\raw\BACI_HS92_V202501'

        # get the list of files in the directory
        files = os.listdir(path)
        # filter for files that start with 'BACI'
        files = [f for f in files if f.startswith('BACI')]
        # read the files concat along axis = 0 with polars
        df = pl.concat([pl.read_csv(os.path.join(path, f)) for f in files], how='vertical')

        # rename the columns
        df = df.rename(self.baci_rename)

        # get the concentration data from Soulier 2018
        soul = self.soulier_concentration_read()

        soul = soul.reset_index()

        soul = pl.from_dataframe(soul)

        df = df.filter(pl.col('HS92_Code').is_in(soul['HS92_Code'].to_list()))
       
        #merge df with soul we only need ks that are in the soulier dat
        df = df.join(soul, on='HS92_Code', how='left')

        logging.info(f'BACI data read from {path} and merged with Soulier 2018 data')

        return df
    

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


if __name__ == "__main__":
    baci = BACI_FEED()
    baci.baci_read()