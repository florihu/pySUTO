# Copper Metabolism 1995–2024

> Where the world's copper is mined, processed, traded and used: a reconciled supply–use view
> Florian Huber · 2 November 2026 · pySUTO run opt_20261022_112855_928da10c · all quantities in contained copper

## What pySUTO does and how to read its results

Copper statistics are plentiful but do not add up. The International Copper Study Group (ICSG) reports mine, smelter and refinery output, scrap use and stock changes; national customs offices report trade, compiled in CEPII's BACI database; the materialflows.net series reports the tonnage of ore extracted. Each source has its own gaps, definitions and errors. pySUTO fuses them into one supply–use system for copper: who produces which copper product, who uses it and where it goes, for 46 countries and regions plus a rest of the world, in every year from 1995 to 2024. It follows copper from ore in the ground through concentrate, matte, anode and cathode to the fabricators that first use it, and requires that every tonne entering a process leaves it as a product or a loss. Where sources disagree, it adjusts each reported number as little as possible, weighted by how reliable that number is.

Five points matter when reading the results. First, they are reconciled estimates, not observations, and are described here, not explained: well-reported flows such as national mine output stay close to their sources, thinly reported ones move more, and in Chapter 3 the reconciliation lifts some regions' scrap ratios above their statistics. Second, the system ends at refined cathode: final use means cathode delivered to fabricators, not copper in finished goods, and scrap that fabricators melt directly lies outside it, which is why direct-melt users such as the United States show little scrap. Third, all quantities are contained copper, in Mt Cu or kt Cu, not the mass of ore or product. Fourth, trade comes from customs records with reporting errors; those detected were corrected, but bilateral flows remain the least certain part of the picture. Fifth, each year is reconciled on its own and the figures show central estimates: a one-year jump may come from the data, and the uncertainty ranges the model computes are not drawn.

## 1 · System size: 35 Mt of copper a year cross the system boundary, 2.3 times as much as in 1995

The system takes in copper from two sources, ore in the ground and scrap returned from use, and releases it as cathode to its users, as stock changes and as losses to tailings and slag. Between 1995 and 2024 inputs, final use and losses all grew, but not at the same pace.

### Figure: f01_process_outputs | Process outputs along the chain, 1995–2024 | full
> World output of each step of the copper chain in Mt Cu, from ore mined (lightest) to refined cathode (darkest). The dashed line is cathode from the solvent-extraction electrowinning (SX‑EW) route, which leaches ore and bypasses the smelters. The lines are successive stages of one chain and are not additive.
Alt: Line chart of world copper output for seven chain steps from 1995 to 2024 in Mt Cu. Every line rises; ore mined is the highest, ending near 30 Mt Cu, and the dashed SX-EW cathode line is the lowest, ending near 5 Mt Cu.
- **Mined copper grew faster than concentrate.** Ore output rose ×2.31 (12.8 → 29.6 Mt Cu) but concentrate only ×2.07 (9.0 → 18.8 Mt), because a growing share of ore is leached rather than milled, 22 % in 2024 against 12 % in 1995: SX‑EW cathode rose ×4.30, from 1.1 to 4.8 Mt Cu.
- **The smelting core grew in step.** Matte, anode and electrorefined copper all grew ×2.12–2.15, as a mass-balanced chain requires; refined cathode grew faster (×2.33, to 27.5 Mt Cu) because SX‑EW cathode joins it without passing the smelters.

### Figure: f02_boundary_inputs | Inputs across the system boundary, 1995–2024 | half
> Copper entering the system each year in Mt Cu, stacked: ore in the ground (in situ) and scrap delivered to smelters and fire refineries. Scrap melted directly by fabricators enters outside this boundary and is not shown.
Alt: Stacked area chart of copper entering the system from 1995 to 2024 in Mt Cu. Ore in situ makes up most of the stack and scrap two thin bands on top; the total rises from 15 to 35 Mt Cu.
- **Inputs grew 2.3-fold.** From 15.0 Mt Cu in 1995 to 34.7 Mt Cu in 2024, of which ore supplied 12.8 and 29.6 Mt.
- **Scrap held its share.** Scrap to smelters and refineries rose from 2.2 to 5.1 Mt Cu, 14.5 % of the input in 1995 and 14.6 % in 2024.

### Figure: f03_boundary_outputs | Outputs across the system boundary, 1995–2024 | half
> Copper leaving the system each year in Mt Cu, stacked: cathode to final use, stock changes (exchange, user and producer stocks) and losses to tailings and slag, on the same vertical scale as Figure 2.
Alt: Stacked area chart of copper leaving the system from 1995 to 2024 in Mt Cu. Final use makes up most of the stack, tailings and slag thinner bands; the stock-change band is thin and dips below zero in some years.
- **Four fifths of the input reaches users.** Final use took 78.6 % of the boundary input in 1995 and 78.8 % in 2024 (11.8 and 27.3 Mt Cu).
- **Losses grew slightly slower than final use.** Tailings rose ×2.27 and slag ×2.09 while final use rose ×2.32, so losses fell from 21.5 % to 20.9 % of the input.

### Table: ledger | Boundary ledger, 1995 and 2024
> Each world copper flow across the system boundary in Mt Cu in the first and the last year, with its growth factor. Stock changes are booked on the output side and can be negative (a draw-down).
- **The ledger closes by construction.** The reconciliation requires it, and inputs equal outputs to within 0.2 kt Cu in every year from 1995 to 2024; the processing chain itself holds no stock.
- **Stock changes are small but volatile.** They never exceed 0.81 Mt Cu in a year (a draw-down of 807 kt in 2004, a build-up of 645 kt in 2001), against 15–35 Mt of input.

## 2 · The chain in one picture: 30 Mt Cu in from the ground, 27 Mt of cathode to final use

The Sankey diagrams follow the world's copper through every process in the first and the last year. Each band runs from the process that makes a commodity to the process that uses it; its colour names the commodity, and full colour marks the part that crosses a border.

### Figure: f04_sankey_1995, f04_sankey_2024 | World copper system, 1995 and 2024 | full
> Reconciled world flows; node labels in Mt Cu. Dark bars are processes, coloured bars are sources and sinks at the system boundary, and band width is proportional to the flow, on one scale for both panels. Pale bands stay within the region that made them; full-colour bands cross a border between two of the model's 47 regions.
Alt: Two Sankey diagrams, 1995 above 2024, drawn to one scale. Copper flows from ore in situ through mining, milling or leaching, smelting, converting and electrorefining to final use, with losses branching off to tailings and slag. The 2024 diagram is the larger of the two, and most of its concentrate band is full colour.
- **Concentrate became the traded form of copper.** In 1995, 26 % of the world's concentrate (2.3 of 9.0 Mt Cu) was smelted outside the region that mined it; in 2024, 60 % (11.3 of 18.8 Mt). The share of cathode used abroad fell from 41 % to 35 %, and matte (under 2 %) and anode (7 %) mostly stay where they are made.
- **Most copper is lost at the mine.** In 2024 milling and flotation lost 4.3 of 23.1 Mt Cu (19 %) to tailings and leaching 1.8 of 6.6 Mt (27 %), against 1.2 Mt lost to slag in smelting, converting and electrorefining together.

## 3 · Scrap versus primary input: 0.22 kt of scrap per kt of primary feed in 2024, back at the 1995 level

Smelters and refineries are fed either with primary material (concentrate and SX‑EW cathode) or with scrap. The ratio of the two shows how much of the refined copper supply is recycled within the modelled boundary. The EU‑14 are the 14 EU member states the model resolves, listed under Data and sources.

### Figure: f05_scrap_ratio | Scrap feed per unit of primary feed, 1995–2024 | full
> Ratio of scrap to primary feed at smelters and refineries for the world and five regions, in kt Cu per kt Cu. Scrap is scrap to smelters and fire refineries; primary is concentrate to smelters plus SX‑EW cathode. The dashed grey line is the world ratio implied by ICSG statistics before reconciliation.
Alt: Line chart of the scrap-to-primary ratio from 1995 to 2024 for the world, China, the EU-14, Japan, the United States and South Korea, with the world's ICSG-based prior as a dashed grey line. China and the EU-14 start above 0.8 and fall; South Korea and Japan rise; the United States falls to 0.05; the world line stays between 0.13 and 0.24 and follows its prior.
- **A trough, then recovery.** The world ratio fell from 0.21 in 1995 to 0.13 in 2003, as primary feed grew 37 % while scrap feed fell 15 %, and returned to 0.22 by 2024, close to the ICSG-based 0.20.
- **The regions diverged.** The EU‑14 stayed scrap-heavy (0.83 → 0.62), China fell from 0.84 to 0.34 while its scrap volume rose 6.5-fold, South Korea rose from 0.17 to 0.40 and Japan from 0.12 to 0.30, and the United States dropped from 0.20 to 0.05.

### Figure: f06_scrap_share_2024 | Scrap share of smelter and refinery feed by region, 2024 | half
> Scrap as a share of total smelter and refinery feed (scrap plus primary) in 2024, for the 20 regions whose feed exceeds 300 kt Cu; EU‑14 members carry the EU‑14 colour.
Alt: Horizontal bar chart of the scrap share of smelter and refinery feed in 2024 for 20 regions, sorted from Germany at 46 % down to Chile, Peru, the DR Congo and Zambia at zero.
- **Germany leads.** Close to half of German smelter and refinery feed is scrap (46 %), followed by South Korea (29 %), Poland (28 %), Spain (27 %), China (25 %) and Japan (23 %).
- **In the reconciled accounts the mining countries refine primary metal only.** Chile, Peru, the DR Congo and Zambia feed no scrap and the United States only 4 %, consistent with US scrap going to direct melt or export, outside this boundary.

### Figure: f07_scrap_volume | Scrap feed volume by region, 1995–2024 | half
> Scrap delivered to smelters and fire refineries each year in kt Cu, stacked by region in the order and colours of Figure 5.
Alt: Stacked area chart of scrap feed by region from 1995 to 2024 in kt Cu. The total more than doubles, and China's band grows from about a fifth of the stack to about three fifths.
- **The growth landed in China.** World scrap feed rose from 2.2 to 5.1 Mt Cu, and China's part of it from 22 % to 62 % (0.5 → 3.1 Mt), more than three times the EU‑14's 0.95 Mt.
- **Read the regional split with care.** In 2024 China's ratio (0.34) and South Korea's (0.40) sit above their ICSG-based priors (0.31 and 0.34), so the reconciliation, not the statistics, set part of their scrap feed; the EU‑14's ratio (0.62) matches its prior.

## 4 · Supply concentration: mining spread out a little, everything downstream concentrated in China

For each step of the chain, the regions' shares of world output show how concentrated supply is. In 1995 the United States led smelting, refining and final use; by 2024 China led all three.

### Figure: f08_share_per_step | Share of world output by region, per chain step, 1995–2024 | full
> Six panels, one per step, each stacked to 100 %: mining and concentrate by producing region, smelting and refined cathode by processing region, final use and scrap feed by consuming region. The rest of the world pools every region not named, the DR Congo included.
Alt: Six stacked area charts, one per chain step, of regional shares of world output from 1995 to 2024. China's band widens strongly in smelting, refined cathode, final use and scrap feed but stays narrow in mining and concentrate, where Chile's band is the widest named one.
- **China's rise is downstream.** Between 1995 and 2024 China went from 8 % to 52 % of smelting, from 9 % to 45 % of refined cathode and from 10 % to 58 % of final use, but only from 4 % to 8 % of mining.
- **Mining spread out.** Chile held 24 % of mining in both years, Peru rose from 4 % to 12 % and the DR Congo reached 14 %, while the United States fell from 19 % to 5 %.

### Figure: f09_hhi | Concentration index by chain step, 1995 and 2024 | half
> Herfindahl–Hirschman index (the sum of squared shares of the 47 regions, rest of the world included) for each step: open circle 1995, filled circle 2024. Shaded bands mark moderate (0.15–0.25) and high (above 0.25) concentration.
Alt: Dot chart of the concentration index for six chain steps, open circles for 1995 and filled circles for 2024. All 1995 values lie below the moderate band; in 2024 smelting, final use and scrap feed lie in the high band and refined cathode in the moderate band, while mining and concentrate stay low.
- **Concentration moved downstream.** In 1995 mining (0.116) was more concentrated than smelting (0.077) or final use (0.092); by 2024 smelting (0.281) and final use (0.352) had overtaken it, while mining eased slightly (0.111).
- **Scrap feed is the most concentrated step.** Its index reached 0.400 in 2024, with China taking 62 % of the world's scrap feed.

### Figure: f10_conc_import_share | Imported share of smelter concentrate intake, 1995–2024 | half
> Share in % of the concentrate smelted in China, the EU‑14 and the United States that comes from outside the region (for the EU‑14, from outside the EU‑14 as a whole).
Alt: Line chart of the imported share of concentrate smelted in China, the EU-14 and the United States from 1995 to 2024. China rises to about 80 %, the EU-14 to about 60 %, and the United States stays near zero.
- **China's smelters run on imports.** Their import share rose from 25 % to 81 %: 7.4 of the 9.2 Mt Cu of concentrate they smelted in 2024 came from abroad.
- **The EU‑14 imports three fifths, the United States almost nothing.** The EU‑14's share doubled from 32 % to 62 %, while US smelters, their intake down from 1.3 to 0.5 Mt Cu, run on domestic concentrate.

### Table: shares | Regional share per chain step, 2024 with 1995
> Share of world output in % for China, the EU‑14, the United States, Chile and Peru at each step in 2024, with 1995 in parentheses, together with the 2024 leader, the share of the three largest regions and the concentration index.
- **The EU‑14 lost weight fastest as a user.** Its share of final use fell from 24.5 % to 10.4 %, faster than its shares of smelting (14.6 % → 9.8 %) and refined cathode (15.7 % → 8.3 %).
- **Three regions dominate downstream.** The three largest regions hold 63 % of smelting and 69 % of final use in 2024, against 40 % and 44 % in 1995; in mining they hold half, as they did in 1995.

## 5 · Trade structure: cross-border trade carries 22.4 Mt Cu a year, 2.8 times 1995, and concentrate is now the form in which copper travels

A tonne of copper may cross several borders on its way from mine to user: as concentrate before smelting, as matte or anode between processing steps, and as cathode after refining. The figures follow these legs region by region, in the trade-linked representation that Sun et al. (2017) developed for lithium.

### Figure: f11_chain_smelting_1995, f11_chain_smelting_2024 | Trade-linked flow of copper along the smelting route, 1995 and 2024 | full
> Each column is a step of the chain and each box a region, as tall as the copper it handles (Mt Cu, one scale for both panels); the ten largest regions over 1995–2024 are shown, the EU‑14 as one, with the rest pooled as other regions. Bands carry concentrate, matte, anode and cathode from the region that makes them to the region that uses them: pale bands stay in the region, full bands cross a border. Stubs on the left of a box are scrap (green) and SX‑EW cathode (dark) entering it.
Alt: Two trade-linked flow diagrams, 1995 above 2024, with five columns from mining to final use and one box per region. Bands between the columns carry concentrate, matte, anode and cathode, pale within a region and full colour across borders. In 2024 China's boxes are the tallest from smelting onwards, and wide full-colour concentrate bands run to them from Chile and Peru.
- **The concentrate funnel.** In 2024 Chile → China (2.4 Mt Cu) and Peru → China (1.9 Mt) are the two largest trade legs of the system; China's smelters take 7.4 of the 11.3 Mt of concentrate traded, including 71 % of Chile's and 75 % of Peru's concentrate exports. In 1995 Japan was the largest buyer.
- **Cathode flows back out of the mining regions.** The largest cathode legs in 2024 are DR Congo → China (1.5 Mt Cu), Chile → United States (0.6 Mt) and Chile → China (0.5 Mt).

### Figure: f12_chain_sxew_1995, f12_chain_sxew_2024 | Trade-linked flow of copper along the SX‑EW route, 1995 and 2024 | full
> Ore leached and electrowon where it is mined, on the same scale as Figure 11. The first gap carries ore to leaching within each region. SX‑EW cathode is pooled with electrorefined cathode and not traded under its own name, so the second gap gives each region's cathode shipments multiplied by the SX‑EW share of its cathode pool, an assumption of this figure, not of the model.
Alt: Two trade-linked flow diagrams of the SX-EW route, 1995 above 2024, with columns for mining, SX-EW and final use. In 1995 the United States and Chile boxes are the largest; in 2024 the DR Congo and Chile boxes are, and most bands into final use are full colour.
- **The leach route moved to the DR Congo.** In 1995 the United States (0.74 Mt Cu of ore leached) and Chile (0.51 Mt) dominated; in 2024 the DR Congo leaches 3.6 Mt and Chile 1.7 Mt, of 6.6 Mt worldwide.
- **SX‑EW cathode is mostly exported.** On this attribution, 80 % of SX‑EW cathode crossed a border in 2024, against 47 % in 1995; the DR Congo shipped 2.3 of its 2.5 Mt Cu abroad.

### Figure: f13_exports_imports_1995, f13_exports_imports_2024 | Exports and imports by region and commodity, 1995 and 2024 | full
> Cross-border trade per region in Mt Cu, imports to the left and exports to the right, stacked by commodity; the black tick marks net trade (exports minus imports). The 20 regions with the largest gross trade in 2024 are ordered by their 2024 net exports, on one horizontal scale for both panels.
Alt: Two diverging bar charts, 1995 above 2024, of imports to the left and exports to the right for 20 regions, stacked by commodity. In 2024 China's import bar is more than twice as long as any other bar, and Chile's export bar is the longest on the export side.
- **China became the importer.** Japan imported 1.5 Mt Cu in 1995 and 1.3 Mt in 2024, the United States 0.8 and 0.9 Mt, while China went from 0.4 to 11.9 Mt, more than half of all copper traded.
- **Chile remained the largest exporter.** Its exports rose from 2.2 to 5.5 Mt Cu, joined in 2024 by the DR Congo and Peru (2.8 Mt each); Japan exports 0.8 Mt of cathode while importing 1.3 Mt of concentrate, the profile of a smelting hub.

### Figure: f14_traded_share | Traded share of world output, per commodity, 1995–2024 | half
> Cross-border trade of concentrate, matte, anode and cathode as a share in % of each commodity's world output; trade between EU‑14 members counts as cross-border.
Alt: Line chart of the traded share of world output from 1995 to 2024 for concentrate, matte, anode and cathode. Concentrate rises from about a quarter to 60 %; cathode peaks near half in 2001 and then eases; matte and anode stay below a tenth.
- **Concentrate more than doubled its traded share.** From 26 % of world output in 1995 to 60 % in 2024.
- **Cathode trade peaked early.** Its traded share reached 48 % in 2001 and eased to 35 % by 2024, while China's share of world refined cathode rose from 9 % in 1995 to 45 % (Figure 8).

### Figure: f15_trade_volume | World cross-border trade by commodity, 1995–2024 | half
> World cross-border trade in Mt Cu per year, stacked by commodity.
Alt: Stacked area chart of world cross-border trade by commodity from 1995 to 2024 in Mt Cu. Concentrate and cathode make up most of the stack, concentrate the larger part from 2017; anode and matte are thin bands.
- **Concentrate overtook cathode in 2017.** It has been the largest traded form of copper every year since, at 11.3 Mt Cu in 2024 against 9.5 Mt of cathode.
- **The intermediates are minor.** Anode trade ranged between 0.65 and 1.5 Mt Cu a year, and matte never exceeded 0.26 Mt.

## Data and sources

All numbers are pySUTO posterior flows of run opt_20261022_112855_928da10c (constraints const_20261022_111531_df61bf5c, imputation imp_20261022_111102_1b49a2b3), valued in contained copper: a tonne is a tonne of copper. The run's data feeds are ICSG statistics, CEPII's BACI trade database (HS 1992) and the materialflows.net crude-ore extraction series. Its uncertainty ranges are computed in run uncprop_20261029_090754_0446e784 and are not drawn here. EU‑14 denotes the 14 EU member states resolved in the model: Germany, Italy, France, Spain, Poland, Belgium, the Netherlands, Austria, Sweden, Bulgaria, Greece, Portugal, Slovakia and Luxembourg. The trade-linked figures of Chapter 5 follow Sun, X., Hao, H., Zhao, F. and Liu, Z. (2017), Tracing global lithium flow: a trade-linked material flow analysis, Resources, Conservation and Recycling 124, 50–61, doi:10.1016/j.resconrec.2017.04.012.
