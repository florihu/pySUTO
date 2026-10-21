
import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

from pysuto.input_registry import InputRegistries
from pysuto.param_relation import conc_for, conc_nominal_lookup

CLEAN_DIR = 'data/proc/feedprep/icsg_clean'
FEED_VALUES = 'data/input/feed/icsg/current/values.csv'
FEED_CONC = 'data/input/feed/icsg/current/conc.xlsx'
# What the parse read beyond its own yearbooks — the BACI and conc versions behind
# the imported-anode subtraction (#63) — in a sidecar's vocabulary.  Beside the
# feed's values.csv the input registry hashes it, so a parse against another
# BACI version is another icsg version.
CLEAN_PARENTS = os.path.join(CLEAN_DIR, 'icsg_flows_clean_parents.json')
FEED_PARENTS = 'data/input/feed/icsg/current/parents.json'
INPUT_ROOT = 'data/input'
BACI_ID = 'baci'
# build_baci_reexport writes this beside every version it nets (ADR 0027)
NETTING_LEDGER = 'reexport_ledger.csv'
ANODE = 'Anode'
ROW_BUCKET = 'ROW'
# Hand-authored: which root region an ICSG aggregate row contains, and in which
# years; the year after is the handover year that publishes both (#67, #73, #74).
AGGREGATES_CSV = 'data/input/params/icsg_aggregates/current/params.csv'
# A label one vintage uses for another territory → that territory's own label
# (#85).  The 2005 yearbook's "Congo" is DR Congo: its mining footnote 3/ says
# the ore "is exported to Zambia for treatment".  The parse maps it to
# "Congo Rep.", which every later vintage uses for Brazzaville.
VINTAGE_LABELS = {('Congo Rep.', 2005): 'Congo, D.R.'}


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


def relabel_by_vintage(df, labels):
    """Give one vintage's rows of a mislabelled territory its own label (#85).

    ``labels`` maps ``(label, vintage)`` to the territory's own label.  The
    relabelled rows are dropped in every year the territory reports under its
    own label in another vintage: the two vintages type the same copper
    differently (DR Congo's 2004 refined output is SX-EW cathode in the 2005
    yearbook, refinery output in the 2014 one), so the cross-vintage dedup,
    which keys on the type, cannot see the overlap and a plain rename would
    add both (ADR 0036).  A label matching no rows, or a vintage carrying both
    labels, raises.  Runs before anything keys on ``Region``, like
    ``merge_territories`` (ADR 0032).
    """
    df = df.copy()
    for (label, vintage), own in labels.items():
        book = df['Year_loaded_from_file'] == vintage
        rows = book & (df['Region'] == label)
        if not rows.any():
            raise ValueError(f"no {label} rows in the {vintage} vintage to relabel")
        if (book & (df['Region'] == own)).any():
            raise ValueError(f"the {vintage} vintage carries both {label} and {own}")
        reported = df.loc[~book & (df['Region'] == own), 'Year']
        dup = rows & df['Year'].isin(reported)
        years = df.loc[rows & ~dup, 'Year']
        print(f"{label} ({vintage} vintage) is {own}: {int((rows & ~dup).sum())} rows "
              f"for {years.min()}–{years.max()} relabelled; {int(dup.sum())} rows in "
              f"{sorted(set(df.loc[dup, 'Year']))} dropped, {own} reports them itself")
        df.loc[rows, 'Region'] = own
        df = df[~dup]
    return df.reset_index(drop=True)


def aggregate_declaration(path=AGGREGATES_CSV):
    """The ``icsg_aggregates`` declaration, with integer years.

    A row says ICSG reports ``root_region`` inside the aggregate row
    ``src_region`` from ``year_from`` to ``year_to``.  The year after is the
    **handover year**: ICSG publishes the aggregate and its members, the first
    year the members report themselves.  A reversed year range raises.
    """
    d = pd.read_csv(path, sep=None, engine='python', dtype=str)
    for c in ('year_from', 'year_to'):
        d[c] = d[c].astype(int)
    bad = d[d['year_from'] > d['year_to']]
    if len(bad):
        raise ValueError(f"icsg_aggregates: year_from after year_to:\n{bad}")
    return d


def split_aggregates(df, declaration):
    """Split each declared aggregate row to its member by the handover year's shares.

    ICSG reports some countries only inside an aggregate row until a handover
    year, then the countries themselves ("Scandinavia" → ROW until 2003, Sweden,
    Finland and Norway from 2004).  In the handover year, the year after
    ``year_to``, it publishes both, so that year states the member's share of
    every series it reports.

    - **Member years (#74).** The aggregate's row is split: the declared member
      gets the handover year's share, the aggregate keeps the rest, and the
      series total is unchanged.  Without it Sweden had a mine and trade but no
      smelter or refinery in 1995–2003, and the model built its cathode through
      a phantom SX-EW chain.
    - **Handover year (#73).** The aggregate republishes its members, so its row
      is dropped wherever the member reports the series; the feed used to add
      both, and the Nordic flows entered twice, once via ROW.

    A series the member does not report in the handover year has no share and
    stays with the aggregate.  These raise: a second member declared for one
    aggregate (its share would be read off rows the first split already
    changed), an aggregate row outside ``year_from`` … handover year, a handover
    year without both forms, a series both report whose share cannot be read
    (aggregate ≤ 0 or a missing value) or lies outside [0, 1], and a member row
    the split would duplicate.  Runs before anything keys on ``Region``, like
    ``merge_territories`` (ADR 0032).
    """
    if not len(declaration):
        return df
    twice = declaration['src_region'][declaration['src_region'].duplicated()]
    if len(twice):
        raise ValueError(f"one declared member per aggregate; declared twice: "
                         f"{sorted(set(twice))}")
    key = ['Process', 'Type', 'Supply_Use']
    df = df.copy()
    for d in declaration.itertuples(index=False):
        agg, mem, h = d.src_region, d.root_region, d.year_to + 1
        years = df.loc[df['Region'] == agg, 'Year']
        outside = sorted(set(years[(years < d.year_from) | (years > h)]))
        if outside:
            raise ValueError(f"{agg} reports outside {d.year_from}–{h}: {outside}")
        a_h = df[(df['Region'] == agg) & (df['Year'] == h)].set_index(key)['Value']
        m_h = df[(df['Region'] == mem) & (df['Year'] == h)].set_index(key)['Value']
        if a_h.empty or m_h.empty:
            raise ValueError(f"{agg} ⊃ {mem}: handover year {h} does not carry both")
        both = a_h.index.intersection(m_h.index)
        a, m = a_h[both], m_h[both]
        unreadable = both[~(a > 0.0).to_numpy() | m.isna().to_numpy()]
        if len(unreadable):
            raise ValueError(f"{agg} ⊃ {mem}: no share can be read in {h} for "
                             f"{list(unreadable)}")
        share = m / a
        beyond = share[(share < 0.0) | (share > 1.0)]
        if len(beyond):
            raise ValueError(f"{agg} ⊃ {mem}: a member share outside [0, 1] in {h}:\n"
                             f"{beyond}")

        inside = (df['Region'] == agg) & df['Year'].between(d.year_from, d.year_to)
        frac = share.reindex(pd.MultiIndex.from_frame(df.loc[inside, key])).to_numpy()
        has = ~np.isnan(frac)
        idx, frac = df.index[inside][has], frac[has]
        new = df.loc[idx].copy()
        new['Region'] = mem
        new['Value'] = df.loc[idx, 'Value'].to_numpy() * frac
        clash = df[df['Region'] == mem].merge(new[key + ['Year']], on=key + ['Year'])
        if len(clash):
            raise ValueError(f"{mem} already reports what the {agg} split would give it:\n"
                             f"{clash[key + ['Year']]}")
        df.loc[idx, 'Value'] = df.loc[idx, 'Value'].to_numpy() * (1.0 - frac)

        handover = ((df['Region'] == agg) & (df['Year'] == h)
                    & pd.MultiIndex.from_frame(df[key]).isin(both))
        print(f"{agg} ⊃ {mem}: {len(new)} member rows for {d.year_from}–{d.year_to}; "
              f"shares read off {h}: "
              + ", ".join(f"{'/'.join(k[:2])} {v:.3f}" for k, v in share.items())
              + f"; {int(handover.sum())} handover-year rows of {agg} dropped")
        df = pd.concat([df[~handover], new], ignore_index=True)
    return df


def _smelter_route(df, eff):
    """Per (Region, Year): the secondary smelter production carried through to
    refined copper, ``S_sm · eff_CFR · eff_ER`` — the part of ICSG's secondary
    refined production that entered as smelter scrap (ADR 0030)."""
    k = _eff_of(eff, 'CFR') * _eff_of(eff, 'ER')
    sm = df[(df['Process'] == 'Smelting') & (df['Type'] == 'Secondary')]
    return sm.groupby(['Region', 'Year'])['Value'].sum() * k


def direct_refinery_route(s_ref, smelter_route, primary, net_imports, eff_er):
    """The direct-to-refinery route per region-year, net of imported anode (#63).

    ``s_ref``, ``smelter_route`` (``S_sm · eff_CFR · eff_ER``) and ``primary``
    are ICSG production values in kt Cu; ``net_imports`` is BACI's net anode
    imports ``N`` in kt CC (see ``net_anode_imports``).  Imported anode passes
    ER only, so it yields ``eff_ER · N`` of refined copper:

        A_imp  = min(max(0, eff_ER · N − P), S_ref)
        direct = max(0, S_ref − S_sm · eff_CFR · eff_ER − A_imp)

    The imports are refined as primary first: only the part beyond the region's
    primary refined output ``P`` can sit inside its secondary refined series,
    and no more than that series holds — refined output less primary output,
    ``(P + S_ref) − P = S_ref``.  Net exports (``N < 0``) contribute 0, and a
    region-year whose ``A_imp`` is 0 keeps ADR 0030's route bit for bit.

    Returns ``(direct, A_imp)``, both refined copper, kt Cu.
    """
    imported = np.minimum(np.maximum(eff_er * net_imports - primary, 0.0), s_ref)
    return np.maximum(s_ref - smelter_route - imported, 0.0), imported


def secondary_routes(df, eff, anode_imports=None):
    """ICSG's secondary refined production split by route, one row per
    ``Refining/Secondary`` row of ``df`` in its order — production values, kt Cu.

    Columns: Region, Year, primary_refined_kt, secondary_refined_kt,
    smelter_route_kt, net_anode_imports_kt (BACI, kt CC), imported_anode_kt
    (``A_imp``) and direct_kt, from ``direct_refinery_route``.  ``anode_imports``
    is keyed ``(Region, Year)``; a region-year it does not carry — every one when
    it is None — has no anode imports, ``N = 0``.  Reads the ``Refining/Primary``
    rows, so it runs before ``apply_input_divisors`` drops them.
    """
    sc = df[(df['Process'] == 'Refining') & (df['Type'] == 'Secondary')]
    keys = pd.MultiIndex.from_arrays([sc['Region'], sc['Year']])
    pr = (df[(df['Process'] == 'Refining') & (df['Type'] == 'Primary')]
          .groupby(['Region', 'Year'])['Value'].sum())
    s_ref = sc['Value'].to_numpy(dtype=float)
    nested = _smelter_route(df, eff).reindex(keys).fillna(0.0).to_numpy()
    primary = pr.reindex(keys).fillna(0.0).to_numpy()
    imports = (np.zeros(len(sc)) if anode_imports is None
               else anode_imports.reindex(keys).fillna(0.0).to_numpy())
    direct, imported = direct_refinery_route(s_ref, nested, primary, imports,
                                             _eff_of(eff, 'ER'))
    return pd.DataFrame({'Region': sc['Region'].to_numpy(), 'Year': sc['Year'].to_numpy(),
                         'primary_refined_kt': primary, 'secondary_refined_kt': s_ref,
                         'smelter_route_kt': nested, 'net_anode_imports_kt': imports,
                         'imported_anode_kt': imported, 'direct_kt': direct})


def net_anode_imports(trade, sector_of, root_of_code, conc_lookup, root_of_label):
    """BACI's net anode imports per ICSG territory and year, kt CC (#63).

        N(r, t) = Σ_A M(A→r) · conc(Anode, A)  −  Σ_Z M(r→Z) · conc(Anode, r)

    over the gross BACI legs ``trade`` (Year, Sector, Region_from, Region_to,
    Value in kt gross mass) whose HS code ``sector_of`` maps to Anode (740200),
    between two different root regions — root self-trade is no trade (ADR 0014).
    ``conc`` is the conc params' nominal keyed by the exporter, as the BACI
    netting and the trade-share fan convert.  An HS code or BACI code the maps
    do not carry, or no Anode nominal, raises.

    Keyed ``(Region, Year)`` by ICSG label: each label whose root is not ROW gets
    its root's series.  A ROW label gets none — the bucket's trade belongs to
    186 economies, not to the territory, and nesting is per territory (ADR 0032).
    """
    miss = sorted(set(trade['Sector']) - set(sector_of))
    if miss:
        raise ValueError(f"BACI HS codes missing from the sector map: {miss}")
    t = trade[trade['Sector'].map(sector_of) == ANODE]
    fr = t['Region_from'].map(root_of_code).rename('root')
    to = t['Region_to'].map(root_of_code).rename('root')
    miss = sorted(set(t.loc[fr.isna(), 'Region_from']) | set(t.loc[to.isna(), 'Region_to']))
    if miss:
        raise ValueError(f"BACI codes missing from the region map: {miss}")
    cross = fr != to
    t, fr, to = t[cross], fr[cross], to[cross]
    conc = conc_for(conc_lookup, np.full(len(t), ANODE, dtype=object), fr.to_numpy())
    if np.isnan(conc).any():
        raise ValueError("the conc params carry no Anode nominal")
    cc = t['Value'] * conc
    net = (cc.groupby([to, t['Year']]).sum()
           .sub(cc.groupby([fr, t['Year']]).sum(), fill_value=0.0))
    labels = pd.DataFrame([(lab, r) for lab, r in root_of_label.items() if r != ROW_BUCKET],
                          columns=['Region', 'root'])
    return (labels.merge(net.rename('kt').reset_index(), on='root')
            .set_index(['Region', 'Year'])['kt'])


def read_net_anode_imports(baci_version, root_of_label, root=INPUT_ROOT):
    """``net_anode_imports`` of the registered BACI version ``baci_version``, and
    the parents it was read from (``{'inputs': {'feed': …, 'params': …}}``).

    Read-only — a lookup never versions a ``current/`` (#93).  The version must
    be a **gross** BACI parse, the one ``build_baci_reexport --source-version``
    nets: the netting re-attributes anode legs, and it reads this parse's ICSG
    input (``icsg_intake_production``), so reading a netted version would make
    the ICSG input depend on itself.  A version carrying the netting's ledger
    raises.
    """
    reg = InputRegistries.load(root=root)
    src = reg.feed.resolve(BACI_ID, baci_version)
    if (src / NETTING_LEDGER).is_file():
        raise ValueError(
            f"baci v{baci_version} is a netted version (it carries {NETTING_LEDGER}); "
            f"pass the gross BACI parse version it was netted from")
    smap = pd.read_excel(src / 'conc.xlsx', sheet_name='sector_map')
    rmap = pd.read_excel(src / 'conc.xlsx', sheet_name='region_map')
    net = net_anode_imports(pd.read_csv(src / 'values.csv'),
                            dict(zip(smap['src_sector'].astype(int), smap['root_sector'])),
                            dict(zip(rmap['src_region'].astype(int), rmap['root_region'])),
                            conc_nominal_lookup(reg.params.resolve('conc') / 'params.csv'),
                            root_of_label)
    parents = {'inputs': {'feed': {BACI_ID: reg.feed.entry(BACI_ID, baci_version)['version']},
                          'params': {'conc': reg.params.entry('conc')['version']}}}
    return net, parents


def icsg_root_of(conc_path=FEED_CONC):
    """ICSG source label → root region, from the feed's concordance."""
    rm = pd.read_excel(conc_path, sheet_name='region_map')
    return dict(zip(rm['src_region'], rm['root_region']))


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


def apply_input_divisors(df, eff, anode_imports=None):
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
    that copper enters the model once, as smelter scrap.  It also contains
    imported anode a region refines beyond its primary output (#63: Austria
    refines ~60 kt Cu a year of imported blister and anode), which enters the
    model as anode trade.  So per (Region, Year)

        Scrap→CFR = max(0, S_ref − S_sm · eff_CFR · eff_ER − A_imp) / (eff_CFR · eff_ER)

    with ``A_imp`` from ``anode_imports`` (kt CC keyed (Region, Year), see
    ``direct_refinery_route``); without it ``A_imp = 0``.

    A region-year with a smelter series but no refined-secondary series is
    untouched (nothing is booked twice); one whose smelter route exceeds S_ref
    gets a zero direct route and is reported by ``secondary_excess``.

    Before ADR 0021 the refined series was divided by eff_CFR·eff_ER and placed
    on the matte entering CFR — two stages back instead of one, which assumed
    every anode a region makes is refined at home.
    """
    # the direct route, on production values — before any divisor is applied and
    # before the Primary rows it reads are dropped
    direct = secondary_routes(df, eff, anode_imports)['direct_kt'].to_numpy()

    df = pd.concat([df, refining_anode_rows(df)], ignore_index=True)
    # the Primary series is now carried by the Anode row; it no longer anchors
    # Matte→CFR, so drop it rather than leave a stale mapping behind.
    df = df[~((df['Process'] == 'Refining') & (df['Type'] == 'Primary'))].copy()

    is_sm = df['Process'] == 'Smelting'
    is_an = (df['Process'] == 'Refining') & (df['Type'] == 'Anode')
    is_sc = (df['Process'] == 'Refining') & (df['Type'] == 'Secondary')

    k = _eff_of(eff, 'CFR') * _eff_of(eff, 'ER')
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



def cl(raw, baci_version):
    m = (raw['Supply_Use'] == 'Supply') & (raw['Process'] == 'Semi')

    #1 filter remove semi supply
    df = raw[m != True].copy()

    #2rename totals based on RENAME dict and the process that the total is for
    for process, new_name in RENAME_TOT.items():
        m = (df['Process'] == process) & (df['Type'] == 'Total')
        df.loc[m, 'Type'] = new_name

    #2a a vintage's mislabelled territory gets its own label (#85)
    df = relabel_by_vintage(df, VINTAGE_LABELS)
    #2b one territory, one label — must precede every step that keys on
    #   Region: the ADR 0030 nesting subtraction and the feed's own summation
    df = merge_territories(df, territory_aliases())
    #2c an aggregate row is split to its declared member, and its handover-year
    #   duplicate is not summed with the members (#73, #74)
    df = split_aggregates(df, aggregate_declaration())

    #3 production → the input it consumed, marked as a Use row; the direct
    #   route nets off imported anode refined beyond primary output (#63)
    eff = read_eff()
    excess = secondary_excess(df, eff)
    imports, parents = read_net_anode_imports(baci_version, icsg_root_of())
    routes = secondary_routes(df, eff, imports)
    df = apply_input_divisors(df, eff, imports)



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
    imp = routes[routes['imported_anode_kt'] > 0.0]
    imp_path = os.path.join(CLEAN_DIR, 'icsg_imported_anode.csv')
    imp.to_csv(imp_path, index=False)
    print(f"Imported anode refined beyond primary output (off the direct route): "
          f"{len(imp)} region-years, {imp['imported_anode_kt'].sum():,.0f} kt Cu → {imp_path}")
    if len(imp):
        print(imp.groupby('Region')['imported_anode_kt'].agg(['size', 'sum']).round(0)
              .sort_values('sum', ascending=False).head(8).to_string())
    with open(CLEAN_PARENTS, 'w', encoding='utf-8') as f:
        json.dump(parents, f, indent=2, sort_keys=True)
    print(f"Read {parents['inputs']} → {CLEAN_PARENTS}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baci-version", type=int, required=True,
                    help="registered gross BACI version whose anode trade the direct "
                         "route nets off (#63) — the version build_baci_reexport.py "
                         "--source-version nets next, never a netted one")
    ap.add_argument("--to-feed", action="store_true",
                    help=f"copy the clean file to {FEED_VALUES} and its parents to "
                         f"{FEED_PARENTS} (the registry versions them on the next "
                         "InputRegistries.initialize())")
    args = ap.parse_args()

    path = os.path.join(CLEAN_DIR, 'icsg_flows_comb.csv')
    clean = cl(pd.read_csv(path), args.baci_version)
    if args.to_feed:
        shutil.copyfile(clean, FEED_VALUES)
        shutil.copyfile(CLEAN_PARENTS, FEED_PARENTS)
        print(f"Copied to {FEED_VALUES} and {FEED_PARENTS}")
