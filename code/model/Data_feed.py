
from System import System

class Data_feed(System):
    """
    This class is used to read the data and convert it into a standardized format.
    """
    def __init__(self):
        """
        The standardized format is stored here
        """
        self.structure = {
            'Year': None,
            'HS92_Code': None,
            'Region_from': None,
            'Region_to': None,
            'Flow': None,
            'Quantity': None,
        }

        super().__init__()