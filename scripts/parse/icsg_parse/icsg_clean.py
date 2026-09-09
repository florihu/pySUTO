
import argparse
import pandas as pd
import os


RENAME_TOT = {'Smelting': 'Primary',
          'Refining': 'Primary',
          'Mining': 'Concentrate'}

# ADR 0021: ICSG refined production measures what the *electrolytic refinery*
# made, so it anchors the anode entering ER — not the matte entering CFR.  The
# old placement divided it back through eff_CFR·eff_ER to the middle of the
# chain, which silently assumed a region refines at home every anode it makes;
# for anode/blister exporters (Zambia, Bulgaria, Congo D.R.) it left the matte
# with no anchored outlet.  Matte→CFR now carries no ICSG anchor: the Matte
# node's hard mass balance already determines it from Smelting→Matte.
RENAME ={('Smelting','Primary'): 'Concentrate',
         ('Mining','Concentrate'): 'Concentrate',
         ('Mining','Concentrates'): 'Concentrate',
            ('Smelting','Secondary'): 'Scrap',
            ('Refining','Secondary'): 'Scrap'

            }


def refining_anode_rows(df):
    """ICSG *total* refined production as one 'Anode' row per region-year.

    Total, not primary: scrap-derived copper reaches ER as anode too, so
    anchoring on the primary series alone understates ER's feed by 19 % world
    wide (Germany 49 %, Austria 100 %, Belgium 41 %) and would relocate the
    contradiction from the Matte node to the Anode node.  The Secondary rows
    stay in place as the Scrap→CFR anchor: that is the same copper measured at
    a different point of its path, not a double count.
    """
    r = df[(df['Process'] == 'Refining') & (df['Supply_Use'] == 'Supply')
           & (df['Type'].isin(('Primary', 'Secondary')))]
    a = r.groupby(['Region', 'Year'], as_index=False).agg(
        Value=('Value', 'sum'),
        Unit=('Unit', 'first'),
        Year_loaded_from_file=('Year_loaded_from_file', 'first'))
    a['Process'], a['Supply_Use'], a['Type'] = 'Refining', 'Supply', 'Anode'
    return a[df.columns]


def apply_input_divisors(df, eff):
    """Convert each production series back to the input it consumed.

    Every series is divided by the efficiency of exactly the stages between
    that input and the reported output (ADR 0021):

      Smelting production  → concentrate/scrap entering Smelting  ÷ eff_Smelting
      Refined (total)      → anode entering ER                    ÷ eff_ER
      Refined (secondary)  → scrap entering CFR, which then passes
                             CFR→Anode→ER                         ÷ eff_CFR·eff_ER

    Before ADR 0021 the refined series was divided by eff_CFR·eff_ER and placed
    on the matte entering CFR — two stages back instead of one, which assumed
    every anode a region makes is refined at home.
    """
    def _eff(process):
        return eff.loc[eff['Process'] == process, 'Value_median'].values[0]

    df = pd.concat([df, refining_anode_rows(df)], ignore_index=True)
    # the Primary series is now carried by the Anode row; it no longer anchors
    # Matte→CFR, so drop it rather than leave a stale mapping behind.
    df = df[~((df['Process'] == 'Refining') & (df['Type'] == 'Primary'))].copy()

    is_sm = df['Process'] == 'Smelting'
    is_an = (df['Process'] == 'Refining') & (df['Type'] == 'Anode')
    is_sc = (df['Process'] == 'Refining') & (df['Type'] == 'Secondary')
    df.loc[is_sm, 'Value'] = df.loc[is_sm, 'Value'] / _eff('Smelting')
    df.loc[is_an, 'Value'] = df.loc[is_an, 'Value'] / _eff('ER')
    df.loc[is_sc, 'Value'] = df.loc[is_sc, 'Value'] / (_eff('CFR') * _eff('ER'))
    df.loc[is_sm | is_an | is_sc, 'Supply_Use'] = 'Use'
    return df


def read_eff(path ='data/input/params/eff/current/copper_efficiency.csv'):

    df = pd.read_csv(path)

    # filter refining and smelting calcualted median Value_min_tCu_per_tCu	Value_max_tCu_per_tCu

    df['Value_median']= df[['Value_min_tCu_per_tCu', 'Value_max_tCu_per_tCu']].median(axis=1)

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

    #3 production → the input it consumed, marked as a Use row
    df = apply_input_divisors(df, read_eff())



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
    dr = 'data/proc/feedprep/icsg_clean'
    os.makedirs(dr, exist_ok=True)
    df.to_csv(os.path.join(dr, 'icsg_flows_clean.csv'), index=False)
    print(f"Cleaned data saved to {os.path.join(dr, 'icsg_flows_clean.csv')}")

    pass


if __name__ == "__main__":
    
    path =  'data/proc/feedprep/icsg_clean/icsg_flows_comb.csv'

    df = pd.read_csv(path)
    df = cl(df)
