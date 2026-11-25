"""icsg_paths.py
=============
One helper, ``get_path``, used by the ICSG yearbook parsers beside this file.
"""

from __future__ import annotations

from pathlib import Path


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
