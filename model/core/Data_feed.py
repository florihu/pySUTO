


import pandas as pd
import polars as pl
import logging
import os
from System import System



class DataFeed(System):
    """
    This class is used to read the data and convert it into a standardized format.
    """
    def __init__(self, name: str, type_: str = None):  
        """
        The standardized format is stored here
        """
        self.name = name
        self.type = type_

        self.get_unc()

        self.s2b = None
        #assert self.type_ in ['flow', 'yield', 'concentration', 'stock', None], f"Type {self.type_} not recognized"

        self.logger = logging.getLogger(__name__)

        super().__init__()

    def s2b(self, path: str):
        '''
        This function reads the concordance table of of a given source data to the base classification.
        // TODO: check function that assesses if all the base classifications are right.

        Assumption : File is a xlsx file

        '''

        # Read the concordance table
        assert path.endswith('.xlsx'), f"File {path} is not an xlsx file"
        df = pd.read_excel(path, sheet_name=None)
        self.s2b = df
        return None

    def get_unc(self):
        '''
        Read the uncertainty table defined for a given source data
        Assumption : File is a xlsx file
        '''
        unc_folder = r'data/input/unc'

        # the file is named after the name of the data source
        path = f"{unc_folder}/{self.name}.xlsx"
        # assert path exists in folder
        assert os.path.exists(path), f"File {path} does not exist in folder {unc_folder}"
        
        df = pd.read_excel(path, sheet_name='unc')

        df.set_index('Variable', inplace=True)
        
        # to dict
        self.unc = df['Data'].to_dict()
        logging.info(f"Uncertainty data for {self.name} read from {path}")
        
        # // TODO: add checks for the uncertainty relevant data fields and allowed values etc.