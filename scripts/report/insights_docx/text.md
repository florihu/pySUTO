# Copper Metabolism 1995–2024

> Where the world's copper is mined, processed, traded and used: a reconciled supply–use view
> Florian Huber · 2 November 2026 · pySUTO run opt_20261022_112855_928da10c · all quantities in contained copper

## What pySUTO does and how to read its results

Copper statistics are plentiful but do not add up. Mine and refinery output is reported by the International Copper Study Group (ICSG) and the US Geological Survey, trade by national customs offices (compiled in the BACI database), and each source has its own gaps, definitions and errors. pySUTO fuses them into one supply–use system for copper: who produces which copper product, who uses it and where it goes, for 46 countries and regions plus a rest of the world, in every year from 1995 to 2024. It follows copper from ore in the ground through concentrate, matte, anode and cathode to its first users, and requires that every tonne entering a process leaves it as a product or a loss. Where sources disagree, it adjusts each reported number as little as possible, weighted by how reliable that number is.

Five points matter when reading the results. They are reconciled estimates, not observations: well-reported flows such as national mine output stay close to their sources, while thinly reported ones move more; the regional split of scrap in Chapter 3 is largely a model outcome. The system ends at refined cathode, so scrap melted directly by fabricators and copper in use lie outside it, which is why direct-melt users such as the United States show little scrap. All quantities are contained copper, in Mt Cu or kt Cu, not the mass of ore or product. Trade comes from customs records with reporting errors; those detected were corrected, but bilateral flows remain the least certain part of the picture. Finally, each year is reconciled on its own and the figures show central estimates: a one-year jump may come from the data, and the uncertainty ranges the model computes are not drawn.

## 1 · System size: 35 Mt of copper a year cross the system boundary, 2.3 times as much as in 1995

The system takes in copper from two sources, ore in the ground and scrap returned from use, and releases it as cathode to its users, as stock changes and as losses to tailings and slag. Between 1995 and 2024 everything grew, but not at the same pace.

### Figure: f01_process_outputs | Process outputs along the chain, 1995–2024 | full
> World output of each step of the copper chain in Mt Cu, from ore mined (lightest) to refined cathode (darkest). The dashed line is cathode from the solvent-extraction electrowinning (SX‑EW) route, which leaches ore and bypasses the smelters. The lines are successive stages of one chain and are not additive.
- **Mined copper grew faster than concentrate.** Ore output rose ×2.31 (12.8 → 29.6 Mt Cu) but concentrate only ×2.07 (9.0 → 18.8 Mt), because a growing share of ore is leached rather than milled: SX‑EW cathode rose ×4.30, from 1.1 to 4.8 Mt Cu.
- **The smelting core grew in step.** Matte, anode and electrorefined copper all grew ×2.12–2.15, as a mass-balanced chain should; refined cathode grew faster (×2.33, to 27.5 Mt Cu) because SX‑EW cathode joins it without passing the smelters.

### Figure: f02_boundary_inputs | Inputs across the system boundary, 1995–2024 | half
> Copper entering the system each year in Mt Cu, stacked: ore in the ground (in situ) and scrap delivered to smelters and fire refineries. Scrap melted directly by fabricators enters outside this boundary and is not shown.
- **Inputs grew 2.3-fold.** From 15.0 Mt Cu in 1995 to 34.7 Mt Cu in 2024, of which ore supplied 12.8 and 29.6 Mt.
- **Scrap held its share.** Scrap to smelters and refineries rose from 2.2 to 5.1 Mt Cu, 14.5 % of the input in 1995 and 14.6 % in 2024.

### Figure: f03_boundary_outputs | Outputs across the system boundary, 1995–2024 | half
> Copper leaving the system each year in Mt Cu, stacked: cathode to final use, stock changes (exchange, user and producer stocks) and losses to tailings and slag, on the same vertical scale as Figure 2.
- **Four fifths of the input reaches users.** Final use took 78.6 % of the boundary input in 1995 and 78.8 % in 2024 (11.8 and 27.3 Mt Cu).
- **Losses grew slightly slower than output.** Tailings rose ×2.27 and slag ×2.09 while final use rose ×2.32, so losses fell from 21.5 % to 20.9 % of the input.

### Table: ledger | Boundary ledger, 1995 and 2024
> Each copper flow across the system boundary in Mt Cu in the first and the last year, with its growth factor. Stock changes are booked on the output side and can be negative (a draw-down).
- **The ledger closes.** Inputs equal outputs to within 0.2 kt Cu in every year from 1995 to 2024, because the processing chain itself holds no stock.
- **Stock changes are small but volatile.** They never exceed 0.81 Mt Cu in a year (a draw-down of 807 kt in 2004, a build-up of 645 kt in 2001), against 15–35 Mt of input.

## 2 · The chain in one picture: 30 Mt Cu in from the ground, 27 Mt of cathode out

The Sankey diagrams follow the world's copper through every process in the first and the last year. Each band runs from the process that makes a commodity to the process that uses it; its colour names the commodity, and full colour marks the part that crosses a border.

### Figure: f04_sankey_1995, f04_sankey_2024 | World copper system, 1995 and 2024 | full
> Reconciled world flows; node labels in Mt Cu. Dark bars are processes, coloured bars are sources and sinks at the system boundary, and band width is proportional to the flow, on one scale for both panels. Pale bands stay within the region that made them, full-colour bands cross a border.
- **Concentrate became the traded form of copper.** In 1995, 26 % of the world's concentrate (2.3 of 9.0 Mt Cu) was smelted outside the region that mined it; in 2024, 60 % (11.3 of 18.8 Mt). The share of cathode used abroad fell from 41 % to 35 %, and matte (under 2 %) and anode (7 %) mostly stay where they are made.
- **The mill is where copper is lost.** In 2024 milling and flotation lost 4.3 of 23.1 Mt Cu (19 %) to tailings and leaching 1.8 of 6.6 Mt (27 %), while smelting, converting and electrorefining together lost 1.2 Mt to slag.

## 3 · Scrap versus primary input: the world uses 0.22 t of scrap per tonne of primary feed, back at its 1995 level

Smelters and refineries are fed either with primary material (concentrate and SX‑EW cathode) or with scrap. The ratio of the two shows how much of the refined copper supply is recycled within the modelled boundary.

### Figure: f05_scrap_ratio | Scrap feed per tonne of primary feed, 1995–2024 | full
> Ratio of scrap to primary feed at smelters and refineries for the world and five regions, in kt Cu per kt Cu. Scrap is scrap to smelters and fire refineries; primary is concentrate to smelters plus SX‑EW cathode. The dashed grey line is the world ratio implied by ICSG statistics before reconciliation.
- **A trough, then recovery.** The world ratio fell from 0.22 in 1995 to 0.13 in 2003, as primary feed grew 37 % while scrap feed fell 15 %, and returned to 0.22 by 2024, close to the ICSG-based 0.20.
- **The regions diverged.** The EU‑14 stayed scrap-heavy (0.83 → 0.62), China fell from 0.84 to 0.34 while its scrap volume rose 6.5-fold, South Korea rose from 0.17 to 0.40 and Japan from 0.12 to 0.30, and the United States dropped from 0.20 to 0.05.

### Figure: f06_scrap_share_2024 | Scrap share of refinery-side feed by region, 2024 | half
> Scrap as a share of total smelter and refinery feed (scrap plus primary) in 2024, for the 20 regions whose feed exceeds 300 kt Cu; EU‑14 members carry the EU‑14 colour.
- **Germany leads.** Close to half of German refinery feed is scrap (46 %), followed by South Korea (29 %), Poland (28 %), Spain (27 %), China (25 %) and Japan (23 %).
- **Mining countries refine primary metal only.** Chile, Peru, the DR Congo and Zambia feed no scrap at all and the United States only 4 %, consistent with US scrap going to direct melt or export, outside this boundary.

### Figure: f07_scrap_volume | Scrap feed volume by region, 1995–2024 | half
> Scrap delivered to smelters and fire refineries each year in kt Cu, stacked by region in the order and colours of Figure 5.
- **The growth landed in China.** World scrap feed rose from 2.2 to 5.1 Mt Cu, and China's part of it from 22 % to 62 % (0.5 → 3.1 Mt), more than three times the EU‑14's 0.95 Mt.
- **Read the regional split with care.** The reconciliation adds scrap where it must close the chain, and it adds it in China: China's 2024 ratio (0.34) sits above its ICSG-based prior (0.31), whereas the EU‑14's (0.62) matches its prior.

## 4 · Supply concentration: mining spread out a little, everything downstream concentrated in China

For each step of the chain, the regions' shares of world output show how concentrated supply is. In 1995 the United States led smelting, refining and final use; by 2024 China led all three.

### Figure: f08_share_per_step | Share of world output by region, per chain step, 1995–2024 | full
> Six panels, one per step, each stacked to 100 %: mining and concentrate by producing region, smelting and refined cathode by processing region, final use and scrap feed by consuming region. The rest of the world pools every region not named, the DR Congo included.
- **China's rise is downstream.** Between 1995 and 2024 China went from 8 % to 52 % of smelting, from 9 % to 45 % of refined cathode and from 10 % to 58 % of final use, but only from 4 % to 8 % of mining.
- **Mining spread out.** Chile held 24 % of mining in both years, Peru rose from 4 % to 12 % and the DR Congo reached 14 %, while the United States fell from 19 % to 5 %.

### Figure: f09_hhi | Concentration index by chain step, 1995 and 2024 | half
> Herfindahl–Hirschman index (the sum of squared regional shares over 47 regions) for each step: open circle 1995, filled circle 2024. Shaded bands mark moderate (0.15–0.25) and high (above 0.25) concentration.
- **The chokepoint moved from mine to smelter.** Smelting concentration more than tripled (0.077 → 0.281) and final use almost quadrupled (0.092 → 0.352), while mining eased slightly (0.116 → 0.111).
- **Secondary supply is the most concentrated step.** Scrap feed reached 0.400 in 2024, with China taking 62 % of the world's refinery-bound scrap.

### Figure: f10_conc_import_share | Imported share of smelter concentrate intake, 1995–2024 | half
> Share of the concentrate smelted in China, the EU‑14 and the United States that comes from outside the region (for the EU‑14, from outside the EU‑14 as a whole).
- **China's smelters run on imports.** Their import share rose from 25 % to 81 %: 7.4 of the 9.2 Mt Cu of concentrate they smelted in 2024 came from abroad.
- **The EU‑14 imports three fifths, the United States almost nothing.** The EU‑14's share doubled from 32 % to 62 %, while US smelters, their intake down from 1.3 to 0.5 Mt Cu, run on domestic concentrate.

### Table: shares | Regional share per chain step, 2024 with 1995
> Share of world output in % for China, the EU‑14, the United States, Chile and Peru at each step in 2024, with 1995 in parentheses, together with the 2024 leader, the share of the three largest regions and the concentration index.
- **The EU‑14 is a processor, not a consumer.** It kept about a tenth of smelting (9.8 %) and 8.3 % of refined cathode but fell from 24.5 % to 10.4 % of final use.
- **Three regions dominate downstream.** The three largest regions hold 63 % of smelting and 69 % of final use in 2024, against 40 % and 44 % in 1995; in mining they hold half, as they did in 1995.

## 5 · Trade structure: cross-border trade carries 22.4 Mt Cu a year, 2.8 times 1995, and concentrate is now the form in which copper travels

A tonne of copper may cross several borders on its way from mine to user: as concentrate before smelting, as matte or anode between processing steps, and as cathode after refining. The figures follow these legs region by region, after the representation of Sun et al. (2017).

### Figure: f11_chain_smelting_1995, f11_chain_smelting_2024 | Trade-linked flow of copper along the smelting route, 1995 and 2024 | full
> Each column is a step of the chain and each box a region, as tall as the copper it handles (Mt Cu, one scale for both panels); the ten largest regions over 1995–2024 are shown, the EU‑14 as one, with the rest pooled as other regions. Bands carry concentrate, matte, anode and cathode from the region that makes them to the region that uses them: pale bands stay in the region, full bands cross a border. Stubs on the left of a box are scrap (green) and SX‑EW cathode (dark) entering it.
- **The concentrate funnel.** In 2024 Chile → China (2.4 Mt Cu) and Peru → China (1.9 Mt) are the two largest legs of the system; China's smelters take 7.4 of the 11.3 Mt of concentrate traded, including 71 % of Chile's and 75 % of Peru's concentrate exports. In 1995 Japan was the largest buyer.
- **Cathode flows back out of the mining regions.** The largest cathode legs in 2024 are DR Congo → China (1.5 Mt Cu), Chile → United States (0.6 Mt) and Chile → China (0.5 Mt).

### Figure: f12_chain_sxew_1995, f12_chain_sxew_2024 | Trade-linked flow of copper along the SX‑EW route, 1995 and 2024 | full
> Ore leached and electrowon where it is mined, on the same scale as Figure 11. The first gap carries ore to leaching within each region. SX‑EW cathode is pooled with electrorefined cathode and not traded under its own name, so the second gap gives each region's cathode shipments multiplied by the SX‑EW share of its cathode pool, an assumption of this figure, not of the model.
- **The leach route moved to Africa.** In 1995 the United States (0.74 Mt Cu of ore leached) and Chile (0.51 Mt) dominated; in 2024 the DR Congo leaches 3.6 Mt and Chile 1.7 Mt, of 6.6 Mt worldwide.
- **SX‑EW cathode is mostly exported.** On this attribution, 80 % of SX‑EW cathode crossed a border in 2024, against 47 % in 1995, led by the DR Congo's 2.5 Mt Cu.

### Figure: f13_exports_imports_1995, f13_exports_imports_2024 | Exports and imports by region and commodity, 1995 and 2024 | full
> Cross-border trade per region in Mt Cu, imports to the left and exports to the right, stacked by commodity; the black tick marks net trade (exports minus imports). The 20 regions with the largest gross trade in 2024 are ordered by their 2024 net exports, on one horizontal scale for both panels.
- **The importers changed, the exporters less so.** Japan imported 1.5 Mt Cu in 1995 and 1.3 Mt in 2024, the United States 0.8 and 0.9 Mt, while China went from 0.4 to 11.9 Mt, more than half of all copper traded.
- **Chile remained the largest exporter.** Its exports rose from 2.2 to 5.5 Mt Cu, joined in 2024 by the DR Congo and Peru (2.8 Mt each); Japan exports 0.8 Mt of cathode while importing 1.3 Mt of concentrate, the profile of a smelting hub.

### Figure: f14_traded_share | Traded share of world output, per commodity, 1995–2024 | half
> Cross-border trade of each commodity as a share of its world output.
- **Concentrate more than doubled its traded share.** From 26 % of world output in 1995 to 60 % in 2024.
- **Cathode trade peaked early.** Its traded share reached 48 % in 2001 and has eased to 35 %, as China's own refining grew with its use.

### Figure: f15_trade_volume | World cross-border trade by commodity, 1995–2024 | half
> World cross-border trade in Mt Cu per year, stacked by commodity.
- **Concentrate overtook cathode in 2017.** It has been the largest traded form of copper every year since, at 11.3 Mt Cu in 2024 against 9.5 Mt of cathode.
- **The intermediates are minor.** Anode trade ranged between 0.65 and 1.5 Mt Cu a year, and matte never exceeded 0.26 Mt.

## Data and sources

pySUTO posterior flows of run opt_20261022_112855_928da10c (constraints const_20261022_111531_df61bf5c; feeds ICSG, BACI and USGS via imputation run imp_20261022_111102_1b49a2b3), copper-content valuation: a tonne is a tonne of contained copper. EU‑14 denotes the 14 EU member states resolved in the model (Germany, Italy, France, Spain, Poland, Belgium, the Netherlands, Austria, Sweden, Bulgaria, Greece, Portugal, Slovakia and Luxembourg). The trade figures follow Sun, Hao, Liu and Zhao (2017), Resources, Conservation and Recycling 124, 50–61.
