
import pandas as pd
import numpy as np
import xarray as xr
import sparse
from itertools import product, permutations

class InitialEstimate:

    def __init__(self):
        '''
        do nothing
        '''
        self.read_base_table()


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

        self.base = {}
        for sheet in base_table.keys():
            self.base[sheet] = base_table[sheet].set_index('ID')['Name'].to_dict()

        return 

    def construct_general_index(self):
        """
        Construct a sparse xarray.DataArray (with COO backend) to store float values,
        indexed by:

            Year
            Layer
            Scenario
            Region_origin
            Sector_origin
            Entity_origin
            Region_destination
            Sector_destination
            Entity_destination

        Only includes combinations for the first Year.
        """
        # Define dimension names
        index_dimensions = [
            'Year', 'Layer', 'Scenario',
            'Region_origin', 'Sector_origin', 'Entity_origin',
            'Region_destination', 'Sector_destination', 'Entity_destination'
        ]

        # Get coordinate labels from self.base
        index_ids = {
            'Year': [min(self.base['Year'].keys())],  # Only first year
            'Layer': list(self.base['Layer'].keys()),
            'Scenario': list(self.base['Scenario'].keys()),
            'Region_origin': list(self.base['Region'].keys()),
            'Sector_origin': list(self.base['Sector'].keys()),
            'Entity_origin': list(self.base['Entity'].keys()),
            'Region_destination': list(self.base['Region'].keys()),
            'Sector_destination': list(self.base['Sector'].keys()),
            'Entity_destination': list(self.base['Entity'].keys())
        }

        # Compute full shape
        shape = [len(index_ids[dim]) for dim in index_dimensions]

        # Create an empty sparse COO array (zero-filled by default)
        empty_data = np.array([], dtype=float)
        empty_coords = np.array([[] for _ in shape], dtype=int)
        sparse_array = sparse.COO(coords=empty_coords, data=empty_data, shape=shape)

        # Wrap in xarray.DataArray
        general_index = xr.DataArray(
            sparse_array,
            coords={dim: index_ids[dim] for dim in index_dimensions},
            dims=index_dimensions
        )

        return general_index


    def initialize_array(self):


        return None

if __name__ == "__main__":
    initial_estimate = InitialEstimate()
    initial_estimate.construct_general_index()
    # You can add more functionality to test the class here