


import pandas as pd
import polars as pl
import logging
import os

from pathlib import Path



class DataFeed:
    """
    Class to read and standardize data from a source, including metadata and classification concordance.
    """
    def __init__(self, name: str):  
        self.name = name
        self.meta = {}
        self.type = None
        self.s2b_table = None
        self.logger = logging.getLogger(__name__)

        #self._load_meta()

    def _load_meta(self):
        """
        Load metadata from a file named after the data source.
        Assumes metadata file is located in 'data/input/meta/{name}.xlsx' and contains a 'meta' sheet.
        """
        meta_path = Path('data/input/meta') / f"{self.name}.xlsx"
        assert meta_path.exists(), f"Metadata file {meta_path} not found."

        df = pd.read_excel(meta_path, sheet_name='meta')
        assert all(df.loc[df['Required'], 'Value'].notna()), f"Missing required values in metadata for {self.name}"
        self.meta = df.set_index('Meta_name')['Value'].to_dict()
        self.type = self.meta.get('IEDC_type', None)

        # assert name == name
        assert self.name == self.meta.get('Feed_name', None), f"Metadata name '{self.meta.get('Name')}' does not match DataFeed name '{self.name}'"

        self.logger.info(f"Metadata for '{self.name}' successfully loaded from '{meta_path}'.")

    def load_s2b_table(self, path: str):
        """
        Load a concordance table mapping source data to the base classification.
        Assumes the input file is an Excel (.xlsx).
        """
        assert path.endswith('.xlsx'), f"Expected an .xlsx file, got: {path}"
        self.s2b_table = pd.read_excel(path, sheet_name=None)
        self.logger.info(f"s2b concordance table loaded from '{path}'.")