
import polars as pl

import os
import inspect
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import itertools
from plotnine import ggplot, geom_point, facet_wrap, facet_grid, aes, labs, save_as_pdf_pages
import geopandas as gpd

def clean_cols(df):

    '''
    This function takes a polares df converts the cols into upper case and rest small
    spaces are replaced with underscores

    '''

    # assert that df is a polars dataframe
    assert isinstance(df, pl.DataFrame), 'df is not a polars dataframe'
    # convert the columns to first letter to upper case and replace spaces with underscores
    df.columns = [col.title().replace(' ', '_') for col in df.columns]
    # besides the first letter all letters are lower case
    df.columns = [col[0].upper() + col[1:].lower() for col in df.columns]
    # remove any leading or trailing spaces
    df.columns = [col.strip() for col in df.columns]
    # remove any leading or trailing underscores
    df.columns = [col.strip('_') for col in df.columns]
    # remove any leading or trailing dashes
    df.columns = [col.strip('-') for col in df.columns]
    # remove any leading or trailing dots
    df.columns = [col.strip('.') for col in df.columns]
    # remove any leading or trailing commas
    df.columns = [col.strip(',') for col in df.columns]
    # remove any leading or trailing semicolons
    df.columns = [col.strip(';') for col in df.columns]
    # remove any leading or trailing colons
    df.columns = [col.strip(':') for col in df.columns]
    # remove any leading or trailing slashes
    df.columns = [col.strip('/') for col in df.columns]
    # remove any leading or trailing backslashes
    df.columns = [col.strip('\\') for col in df.columns]
    # remove any leading or trailing pipes
    df.columns = [col.strip('|') for col in df.columns]
    # remove any leading or trailing question marks
    df.columns = [col.strip('?') for col in df.columns]

    return df


def get_path(name):
    '''
    Get the path of the specified file or folder within the root directory.
    Searches recursively starting from the directory of the current script.
    
    Parameters:
    - name: str, the name of the file or folder to search for.

    Returns:
    - str or None: the full path to the file or folder if found, otherwise None.
    '''
    # Set the root directory to the directory where this script is located
    root_directory = 'data/input/raw'
    # Convert the root directory to a Path object for easier manipulation
    root_directory = Path(root_directory)
    
    # Use rglob to recursively search for files and directories with the given name
    for path in root_directory.rglob(name):
        return str(path)  # Return the first match as a string
    
    # If not found assertion error
    raise AssertionError(f"File or folder '{name}' not found in '{root_directory}'.")

def save_fig(name, dpi=600, format='pdf', bbox_inches='tight'):

    '''
    save fig to the fig folder.. distinguish between figs from explo scripts and results scripts
    '''

    base_folder = 'figs'
    # Get the calling script’s filename
    calling_script = inspect.stack()[1].filename
    script_name = os.path.basename(calling_script).replace('.py', '')

    # if script in explo folder save in fig/explo else in fig/results
    if 'test' in calling_script:
        base_folder = os.path.join(base_folder, 'explo')
    elif 'results' in calling_script:
        base_folder = os.path.join(base_folder, 'results')


    path = os.path.join(base_folder, script_name)
    if not os.path.exists(path):
        os.makedirs(path)

    file_path = os.path.join(path, name)
    
    # add format to file_path
    file_path = file_path + '.' + format

    plt.savefig(file_path, dpi=dpi, format=format, bbox_inches=bbox_inches)
      # Close the figure to free up memory

    return None

def save_fig_plotnine(plot, name, w=8, h=6, dpi=600, format='png'):
    base_folder = 'fig'
    # Get the calling script’s filename
    calling_script = inspect.stack()[1].filename
    script_name = os.path.basename(calling_script).replace('.py', '')

    path = os.path.join(base_folder, script_name)
    if not os.path.exists(path):
        os.makedirs(path)

    file_path = os.path.join(path, name)

    plot.save(file_path, format = format, width=w, height=h, dpi=dpi)



def df_to_latex(df, filename, multicolumn=False, longtable=False):
    base_folder = 'tab'

    calling_script = inspect.stack()[1].filename
    script_name = os.path.basename(calling_script).replace('.py', '')

    path = os.path.join(base_folder, script_name)
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Convert DataFrame to LaTeX table format
    latex_table = df.to_latex(float_format="%.2f", multicolumn=multicolumn, longtable=longtable)
    # Write LaTeX table to a .tex file
    with open(f'{path}/{filename}.tex', 'w') as f:
        f.write(latex_table)
    return None

def df_to_gpkg(df, filename, crs):
    '''
    Save a DataFrame to a GeoPackage file in the data/gpkg folder. given it is a Geodataframe object
    '''
    base_folder = 'data/int'

    assert 'geometry' in df.columns, 'The DataFrame must have a geometry column to be saved as a GeoPackage file.'
    # transform to gdf
    df = gpd.GeoDataFrame(df, geometry='geometry', crs = crs)

    calling_script = inspect.stack()[1].filename
    script_name = os.path.basename(calling_script).replace('.py', '')

    path = os.path.join(base_folder, script_name)
    if not os.path.exists(path):
        os.makedirs(path)

    file_path = os.path.join(path, filename + '.gpkg')

    df.to_file(file_path, driver='GPKG')

    return None

def df_to_csv_int(data, name):
    ''' 
    
    Save data to a csv file in the data/int folder.
    Parameters:
    - data: pd.DataFrame, the data to save.
    - name: str, the name of the file to save the data to.

    Returns:
    - None

    '''
    base_folder = r'data\int'
    # Get the calling script’s filename
    calling_script = inspect.stack()[1].filename
    script_name = os.path.basename(calling_script).replace('.py', '')

    path = os.path.join(base_folder, script_name)
    if not os.path.exists(path):
        os.makedirs(path)

    file_path = os.path.join(path, name + '.csv')

    data.to_csv(file_path)

    return None


def df_to_excel(filename, df, sheet_name):
    base_folder = r'data\int'
    # Get the calling script’s filename
    calling_script = inspect.stack()[1].filename
    script_name = os.path.basename(calling_script).replace('.py', '')

    path = os.path.join(base_folder, script_name)
    if not os.path.exists(path):
        os.makedirs(path)

    file_path = os.path.join(path, filename + '.xlsx')

    # Use 'a' mode if file exists; otherwise, create a new file with 'w' mode.
    if os.path.exists(file_path):
        mode = 'a'
        sheet_option = 'replace'
    else:
        mode = 'w'
        sheet_option = None  # Not needed when creating a new file

    with pd.ExcelWriter(file_path, engine='openpyxl', mode=mode, if_sheet_exists=sheet_option) as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=True)




