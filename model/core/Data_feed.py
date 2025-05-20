


import pandas as pd
import polars as pl

from System import System
class Data_feed(System):
    """
    This class is used to read the data and convert it into a standardized format.
    """
    def __init__(self, name: str = None, type_: str = None):  
        """
        The standardized format is stored here
        """
        self.name = name
        self.type = type_
        #assert self.type_ in ['flow', 'yield', 'concentration', 'stock', None], f"Type {self.type_} not recognized"

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

    def unc(self, path: str):
        '''
        Read the uncertainty table defined for a given source data
        Assumption : File is a xlsx file
        '''
        assert path.endswith('.xlsx'), f"File {path} is not an xlsx file"
        df = pd.read_excel(path, sheet_name=None)
        self.unc = df
        return df


    
