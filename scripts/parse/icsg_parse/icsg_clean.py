
import argparse
import os
import shutil

import numpy as np
import pandas as pd

CLEAN_DIR = 'data/proc/feedprep/icsg_clean'
FEED_VALUES = 'data/input/feed/icsg/current/values.csv'
FEED_CONC = 'data/input/feed/icsg/current/conc.xlsx'
ROW_BUCKET = 'ROW'


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


def _eff_of(eff, process):
    return eff.loc[eff['Process'] == process, 'Value_median'].values[0]


def territory_aliases(conc_path=FEED_CONC):
    """Source labels that denote the same reporting territory → the root name.

    ICSG renames territories mid-series (``Taiwan, China`` from 2013) and
    publishes historical joint reporting under a joint label
    (``Belgium-Luxembourg`` to 2013).  The concordance already assigns both
    members of such a pair to one *real-country* root, so that is the test:
    two source labels sharing a non-ROW root are one territory.

    ROW is excluded deliberately.  It is the catch-all bucket for 48 countries,
    and merging its members would nest Serbia's smelter route against Egypt's
    refinery — nesting happens per reporting territory, never per bucket.
    """
    rm = (pd.read_excel(conc_path, sheet_name='region_map')
            [['src_region', 'root_region']].drop_duplicates())
    rm = rm[rm['root_region'] != ROW_BUCKET]
    rm = rm[rm.groupby('root_region')['src_region'].transform('size') > 1]
    assert rm['root_region'].isin(rm['src_region']).all(), (
        "a territory's canonical label is not itself a source label: "
        f"{sorted(set(rm['root_region']) - set(rm['src_region']))}")
    return {s: r for s, r in zip(rm['src_region'], rm['root_region']) if s != r}


def merge_territories(df, aliases):
    """One territory, one label — before anything keys on ``Region`` (ADR 0032).

    The ADR 0030 nesting subtraction looks its smelter route up by
    ``(Region, Year)``, so a territory whose refined-secondary and smelter
    series carry different labels was never nested: ``reindex`` missed,
    ``fillna(0.0)`` zeroed the subtraction, and the smelter route stayed inside
    the refinery series.  In the handover year both labels are published and the
    feed added them.  Renaming first fixes both.

    The handover rows are republications, not new measurements, so the
    duplicate is dropped rather than summed — and a genuine disagreement
    between the two labels raises instead of being silently resolved.
    """
    if not aliases:
        return df
    df = df.copy()
    df['Region'] = df['Region'].replace(aliases)
    key = ['Region', 'Type', 'Process', 'Supply_Use', 'Year']
    dup = df[df.duplicated(subset=key, keep=False)]
    if len(dup):
        spread = dup.groupby(key)['Value'].agg(lambda s: s.max() - s.min())
        print(f"one territory, two labels: {len(spread)} cells published under "
              f"both, max spread {spread.max():.3f} kt")
        assert spread.max() <= 0.5, (
            "the same cell carries different values under two labels:\n"
            f"{spread[spread > 0.5]}")
    return (df.sort_values('Year_loaded_from_file')
              .drop_duplicates(subset=key, keep='last')
              .reset_index(drop=True))


def _smelter_route(df, eff):
    """Per (Region, Year): the secondary smelter production carried through to
    refined copper, ``S_sm · eff_CFR · eff_ER`` — the part of ICSG's secondary
    refined production that entered as smelter scrap (ADR 0030)."""
    k = _eff_of(eff, 'CFR') * _eff_of(eff, 'ER')
    sm = df[(df['Process'] == 'Smelting') & (df['Type'] == 'Secondary')]
    return sm.groupby(['Region', 'Year'])['Value'].sum() * k


def secondary_excess(df, eff):
    """Region-years whose smelter route exceeds their secondary refined production.

    Under the nesting rule these get a zero direct-to-refinery route; they are
    listed, never silently clipped. Japan and Iran are definitional mismatches
    (no anode exports, ratio ≈ 1.1-1.3); Bulgaria exports its secondary anode
    (ADR 0030). Columns: Region, Year, smelter_route_kt, secondary_refined_kt,
    excess_kt — production values, kt Cu.
    """
    rf = (df[(df['Process'] == 'Refining') & (df['Type'] == 'Secondary')]
          .groupby(['Region', 'Year'])['Value'].sum())
    both = pd.concat([_smelter_route(df, eff).rename('smelter_route_kt'),
                      rf.rename('secondary_refined_kt')], axis=1).dropna()
    both['excess_kt'] = both['smelter_route_kt'] - both['secondary_refined_kt']
    return both[both['excess_kt'] > 1e-9].reset_index()


def apply_input_divisors(df, eff):
    """Convert each production series back to the input it consumed.

    Every series is divided by the efficiency of exactly the stages between
    that input and the reported output (ADR 0021):

      Smelting production  → concentrate/scrap entering Smelting  ÷ eff_Smelting
      Refined (total)      → anode entering ER                    ÷ eff_ER
      Refined (secondary)  → direct-to-refinery scrap entering CFR,
                             which then passes CFR→Anode→ER      ÷ eff_CFR·eff_ER

    Direct-to-refinery scrap is secondary refined production MINUS the smelter
    route (ADR 0030): ICSG's secondary refined figure already contains the
    copper that reached the refinery as blister/anode from a secondary smelter
    ("the blister is fire refined to anode copper for electro-refining"), and
    that copper enters the model once, as smelter scrap. So per (Region, Year)

        Scrap→CFR = max(0, S_ref − S_sm · eff_CFR · eff_ER) / (eff_CFR · eff_ER)

    A region-year with a smelter series but no refined-secondary series is
    untouched (nothing is booked twice); one whose smelter route exceeds S_ref
    gets a zero direct route and is reported by ``secondary_excess``.

    Before ADR 0021 the refined series was divided by eff_CFR·eff_ER and placed
    on the matte entering CFR — two stages back instead of one, which assumed
    every anode a region makes is refined at home.
    """
    df = pd.concat([df, refining_anode_rows(df)], ignore_index=True)
    # the Primary series is now carried by the Anode row; it no longer anchors
    # Matte→CFR, so drop it rather than leave a stale mapping behind.
    df = df[~((df['Process'] == 'Refining') & (df['Type'] == 'Primary'))].copy()

    is_sm = df['Process'] == 'Smelting'
    is_an = (df['Process'] == 'Refining') & (df['Type'] == 'Anode')
    is_sc = (df['Process'] == 'Refining') & (df['Type'] == 'Secondary')

    # the direct route, on production values — before any divisor is applied
    k = _eff_of(eff, 'CFR') * _eff_of(eff, 'ER')
    keys = pd.MultiIndex.from_arrays([df.loc[is_sc, 'Region'], df.loc[is_sc, 'Year']])
    nested = _smelter_route(df, eff).reindex(keys).fillna(0.0).to_numpy()
    direct = np.maximum(df.loc[is_sc, 'Value'].to_numpy(dtype=float) - nested, 0.0)

    df.loc[is_sm, 'Value'] = df.loc[is_sm, 'Value'] / _eff_of(eff, 'Smelting')
    df.loc[is_an, 'Value'] = df.loc[is_an, 'Value'] / _eff_of(eff, 'ER')
    df.loc[is_sc, 'Value'] = direct / k
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

    #2b one territory, one label — must precede every step that keys on
    #   Region: the ADR 0030 nesting subtraction and the feed's own summation
    df = merge_territories(df, territory_aliases())

    #3 production → the input it consumed, marked as a Use row
    eff = read_eff()
    excess = secondary_excess(df, eff)
    df = apply_input_divisors(df, eff)



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
    os.makedirs(CLEAN_DIR, exist_ok=True)
    out = os.path.join(CLEAN_DIR, 'icsg_flows_clean.csv')
    df.to_csv(out, index=False)
    print(f"Cleaned data saved to {out}")
    ex_path = os.path.join(CLEAN_DIR, 'icsg_secondary_excess.csv')
    excess.to_csv(ex_path, index=False)
    print(f"Smelter route > secondary refined (direct route clipped to 0): "
          f"{len(excess)} region-years, {excess['excess_kt'].sum():,.0f} kt Cu → {ex_path}")
    if len(excess):
        print(excess.groupby('Region')['excess_kt'].agg(['size', 'sum']).round(0)
              .sort_values('sum', ascending=False).head(8).to_string())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--to-feed", action="store_true",
                    help=f"copy the clean file to {FEED_VALUES} (the registry "
                         "versions it on the next InputRegistries.initialize())")
    args = ap.parse_args()

    path = os.path.join(CLEAN_DIR, 'icsg_flows_comb.csv')
    clean = cl(pd.read_csv(path))
    if args.to_feed:
        shutil.copyfile(clean, FEED_VALUES)
        print(f"Copied to {FEED_VALUES}")
