import sys
import os
import pandas as pd


from DataFeed import DataFeed



# Add the core directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from util import get_path


def init_icsg():

    file_name = r'ICSG_Yearbooks\ICSG 2023 Statistical Yearbook.xlsx'
    path = get_path(file_name)
    
    sheet_dict = {
        '3.Mine': 'Mine_supply',
        '4.Sml': 'Smelter_use',
        '5.Ref': 'Refinery_use',
        '6.SXEW': 'SXEW_use',
        '7.Usage': 'Semi_use',
        '10.Semis': 'Semi_supply',
    }
    rename_dict = {
        'COUNTRY': 'Region',
        'Source': 'Flow',
        'Feed': 'Flow',
    }

    collect = []

    for sheet, name in sheet_dict.items():
        df = pd.read_excel(path, sheet_name=sheet, header=2)

        # find row that contains 'Source:' in first column
        source_row = df[df.iloc[:, 0].str.contains('Source:', na=False)].index[0]
        # remove all rows after this row
        df = df.iloc[1:source_row, :]
        df.rename(columns=rename_dict, inplace=True)
        df = df.melt(id_vars=['Region', 'Flow'], var_name='Year', value_name='Value')
        df['Process'] = name
        df['Unit'] = 'kt'
        df['Year'] = df['Year'].astype(int)
        df['Value'] = df['Value'].astype(float)
        print(df.head())
    
    return df

    

if __name__ == "__main__":
    df = init_icsg()
    print(df.head())
    # df.to_csv('icsg_data.csv', index=False)
    # print("ICS data saved to icsg_data.csv")