
import argparse
import pandas as pd
import os


RENAME_TOT = {'Smelting': 'Primary',
          'Refining': 'Primary',
          'Mining': 'Concentrate'}

RENAME ={('Refining','Primary'): 'Anode',
         ('Smelting','Primary'): 'Concentrate',
         ('Mining','Concentrate'): 'Concentrate',
         ('Mining','Concentrates'): 'Concentrate',
            ('Smelting','Secondary'): 'Srap',
            ('Refining','Secondary'): 'Scrap'
            
            }


def read_eff(path ='data/input/params/copper_efficiency.csv'):

    df = pd.read_csv(path)

    # filter refining and smelting calcualted median Value_min_tCu_per_tCu	Value_max_tCu_per_tCu

    df['Value_median']= df[['Value_min_tCu_per_tCu', 'Value_max_tCu_per_tCu']].median(axis=1)

    df['Process'].replace({'Smelting': 'Smelting', 'ER': 'Refining'}, inplace=True)

    return df



def stats(df):


    # Print unique regions
    print("Unique regions:")
    print(df['Region'].unique().sort())

    # Print unique Region Type  Proces Supply_Use instances
    print("\nUnique Region Type  Proces Supply_Use instances:")
    print(df[['Type',  'Process', 'Supply_Use']].drop_duplicates().sort_values(by='Process'))

    # Print unique Year
    print("\nUnique Year:")
    print(sorted(df['Year'].unique()))



    pass



def cl(raw):
    m = (raw['Supply_Use'] == 'Supply') & (raw['Process'] == 'Semi')

    #1 filter remove semi supply
    df = raw[m != True].copy()

    #2rename totals based on RENAME dict and the process that the total is for
    for process, new_name in RENAME_TOT.items():
        m = (df['Process'] == process) & (df['Type'] == 'Total')
        df.loc[m, 'Type'] = new_name

    #3 for smelting and refining calculate multiple the primary and secondary values with 1/ efficiency channge supply_use to use
    eff = read_eff()
    for process in ['Smelting', 'Refining']:
        df.loc[df['Process'] == process, 'Value'] = df.loc[df['Process'] == process, 'Value'] * (1/eff.loc[eff['Process'] == process, 'Value_median'].values[0])
        df.loc[df['Process'] == process, 'Supply_Use'] = 'Use'



    #3 consistent rename
    for (process, type), new_name in RENAME.items():
        m = (df['Process'] == process) & (df['Type'] == type)
        df.loc[m, 'Type'] = new_name


    # print duplicated pairs
    print("Duplicated Region Type  Process Supply_Use combinations:")
    print(df[df.duplicated(subset=['Region', 'Type', 'Process', 'Supply_Use', 'Year'], keep=False)][['Region', 'Type', 'Process', 'Supply_Use', 'Year']])
    
    #1 no dup in Regoin Type  Process Supply_Use
    assert df[['Region', 'Type', 'Process', 'Supply_Use', 'Year']].drop_duplicates().shape[0] == df.shape[0], "Duplicate Region Type  Process Supply_Use combinations found"

    #2 all unit are kt
    assert (df['Unit'] == 'kt').all(), "Not all units are kt"

    # value to float
    df['Value'] = df['Value'].astype(float, errors='raise')
    
    

    # transform to three digits after
    df['Value'] = df['Value'].round(3)


     # print a sample of vals
    print("\nSample of values:")
    print(df['Value'].sample(10))

    stats(df)

    # years as int
    df['Year'] = df['Year'].astype(int)


    # store
    dr = '/mnt/nfs_fineprint/tmp/PYSUT/pySUTO/data/proc/datafeed/icsg_clean'
    os.makedirs(dr, exist_ok=True)
    df.to_csv(os.path.join(dr, 'icsg_flows_clean.csv'), index=False)
    print(f"Cleaned data saved to {os.path.join(dr, 'icsg_flows_clean.csv')}")

    pass


if __name__ == "__main__":
    
    path =  'data/proc/datafeed/icsg_clean/icsg_flows_comb.csv'

    df = pd.read_csv(path)
    df = cl(df)
