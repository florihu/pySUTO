
import polars as pl

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