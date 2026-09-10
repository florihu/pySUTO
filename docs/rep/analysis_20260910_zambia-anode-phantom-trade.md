# Zambia's phantom anode exports — a raw-data defect the model already handles

*Raw-data forensics, no model state. Sources: `data/input/feed/baci/current/values.csv`
(BACI as loaded) and `data/proc/feedprep/icsg_clean/icsg_flows_comb.csv` (ICSG **before**
`icsg_clean.cl` runs). Reproduce with `python scripts/explo/zambia_anode_diag.py`. Figure:
[`figs/zambia_anode_phantom_trade.html`](figs/zambia_anode_phantom_trade.html). Model behaviour
verified against the shipped run `opt_20260909_094009_49eab1dc` (main\@70f9c62).*

**All tonnages are contained copper.** BACI reports gross mass of the traded commodity; ICSG
already reports contained copper. Everything here is converted with the model's nominal copper
contents (`conc`: anode 0.94, concentrate 0.275, cathode 0.9975) so the two are comparable —
mixing the bases overstates the anode ratio by ~6 %.

## 1 · The defect

From 2016 the trade data books Zambian copper anode to **Switzerland, Singapore and
Luxembourg**. Before 2016 those destinations are absent; after it they take 373–906 kt/yr.

| | 2015 | 2016 | 2017 | 2022 | 2024 |
|---|---|---|---|---|---|
| ICSG smelter output | 648.8 | 698.1 | 787.9 | 727.2 | 683.9 |
| recorded anode exports | 318.5 | 923.0 | **1,484.9** | 686.9 | 752.3 |
| of which to CH / SG / LU | 5.1 | 408.4 | **905.6** | 423.2 | 382.8 |
| exports ÷ smelter output | 0.49 | 1.32 | **1.88** | 0.94 | 1.10 |
| **ex-hub** ÷ smelter output | 0.48 | 0.74 | **0.74** | 0.36 | 0.54 |

In 2017 the data asserts nearly **twice the anode Zambia's smelters made**. Removing the three
hub destinations returns the ratio to 0.25–0.74 in every year from 2016 on — physically sane
throughout.

Three facts fix the interpretation:

- **The hubs never release it.** Over 2016–2024 they absorb **4,541 kt** and re-export **38.8 kt**
  (0.9 %; never more than 13.4 kt in a year). ICSG reports no refined production for any of the
  three, so it was not consumed there either.
- **China's imports from Zambia were not diverted.** They *rise* through the break — 205.6 (2015)
  → 273.1 → **381.3 kt (2017)** — and China books under 1 kt/yr from the hubs. The hub legs are
  additional tonnage, not re-labelled cargo.
- **Congolese transit is ruled out.** Congo→Zambia **anode and cathode are 0.0 kt in every year**.
  Congolese *concentrate* does flow — 692 / 714 / 264 kt gross in 2013 / 2017 / 2022 — which is
  why Zambia's smelter output (788 kt Cu in 2017) exceeds its own mine output (628). But the
  smelting happens in Zambia, so ICSG already counts it: it raises the ceiling rather than
  creating room above it. Zambia also imports essentially no anode (0.00–0.16 kt/yr).

Counted across all products, Zambia's recorded exports exceeded everything its plants turned out
in 2017 by **2.35×** (2,327.5 kt Cu against 989.2). **A second, older anomaly sits on the cathode
line** — 1.59× in 2013, with no hub tonnage in play. The anode story does not explain that one and
Congolese cathode transit cannot either. It is unexplained and deserves its own look.

## 2 · Political economy — three hypotheses, ranked

None is decidable from the data in this repo; the ranking is by fit to the signature.

1. **Merchanting: title and cargo land in different cells.** The trading houses marketing Zambian
   copper are domiciled in Geneva/Zug, Singapore and Luxembourg. They take title at the mine gate
   while the metal sails to Asia. If Zambia's customs populates *destination* with the purchaser's
   country while China's records *origin = Zambia*, both declarations are true about different
   things — and a bilateral matrix assembled from two-sided declarations carries both. BACI
   reconciles *within* a cell, not *across* cells. This is the only hypothesis that explains all
   four signatures at once: the destinations are exactly the trading domiciles, the hubs never
   re-export, China's series is untouched, and the excess equals the hub tonnage.
2. **Profit shifting / trade misinvoicing.** The Zambia–Switzerland copper gap is a documented case
   in the illicit-financial-flows literature; the motive is the royalty and corporate tax base, and
   the destinations are low-tax/secrecy jurisdictions rather than merely convenient intermediaries.
   Ranked second because misinvoicing classically distorts *unit value*, not physical tonnage, and
   would not leave China's records intact — it is better read as the **motive** for (1)'s mechanism.
3. **A reporting-convention change in 2016.** The break is too sharp for behaviour drift (5.1 →
   408.4 kt in one year): a destination field switched from final-destination to consignee, or a new
   export-monitoring regime capturing the sales leg. Explains the **timing**, not the destinations.

**What would discriminate, none of it in this repo:** raw UN Comtrade mirror data (does Switzerland
*declare* these imports, or does BACI carry the cell on Zambia's declaration alone?); BACI's value
field (is $/t on hub legs different from China legs?); whether other producers with Swiss-domiciled
offtakers show the same pattern. The first is decisive and cheap.

## 3 · The model already handles it — verified, no change made

The shipped run separates real from phantom by the physical criterion, with **no hand-list and no
rule about these three countries**:

| Zambia → (2017) | prior | posterior | kept |
|---|---|---|---|
| Switzerland | 596.9 | 3.4 | **1 %** |
| Singapore | 240.5 | 4.0 | **2 %** |
| Luxembourg | 68.2 | 4.3 | **6 %** |
| China | 381.3 | 382.6 | 100 % |
| India | 114.6 | 123.4 | 108 % |

It generalises. World anode trade into regions **with** an ICSG refinery is kept at 99 / 90 / 100 %
(2017 / 2022 / 2024); into regions with **none**, at 2 / 5 / 5 %. Those legs are **99 / 74 / 99 %**
of the entire world anode-trade shortfall. The hubs acquire no fabricated refinery — their `ER→CER`
posterior stays at 3–4 kt.

Most tellingly, **the posterior is already right**: `usetotout_Anode_Zambia` 2017 carries an anchor
of 1,752.3 (phantom included) against a posterior of **859.7**, where the phantom-free anchor would
be **846.7** — 1.5 % away. The z = −29.9 on that row is a *fine levied for being right*.

**Decision: no code change.** A σ screen on the hub legs and their anchors was designed and costed,
then rejected:

- The output is already correct, so σ would change the **price**, not the answer — unlike
  [ADR 0019](../adr/0019-contradicted-flow-nodes-lose-anchor-authority.md), where the posterior had
  nowhere to settle and loosening moved 0.46 % of the model's mass.
- The gain is ~1.5 % of the headline sum-row metric, plus the 441 k of prior-deviation the Zambia
  hub legs carry (180 edge-years, 9,301 → 212 kt). Against a new screen, a full chain re-run, and an
  ADR arguing past two prior σ rejections, that is a poor trade.
- **Suppressing the penalty would hide the finding.** That 441 k is the model visibly refusing bad
  data; leaving it is the cheapest way to keep the defect measurable.

A feed-level capacity gate — dropping `Flow→Process` legs into regions with no ICSG capacity for
that process, via the ADR-0014 `Edge_scope_mask` mechanism with a generated (not hand-written) rule
set — was also rejected for now, on the same "do not hide the finding" ground. The design is worked
out should it be needed.

**Revisit if:** the pattern spreads to more producers and the deviation cost grows materially, or
Zambia's residual starts blocking something. Zambia now carries 57,813 of sum-row z² (2.1 % of
2,815,511), down from 468,682 before ADR 0021/0022 — it is no longer a large block.

## 4 · For the manuscript

Report as a known limitation with a number attached: roughly **380–910 kt/yr of recorded Zambian
anode trade from 2016 is not physical**, the reconciliation refuses it on capacity grounds, and the
refusal is what the residual at Zambia measures. The same capacity test is a reusable diagnostic for
any producer–hub pair.
