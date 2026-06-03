# Preserving flow scale in the pySUTO solver — recommendations from the matrix-balancing literature

*Generated 2026-05-28. Reference run: [`data/proc/opt/opt_20260527_095157_40846da4`](../../data/proc/opt/opt_20260527_095157_40846da4/) (parent const [`const_20260527_094403_4bd9e613`](../../data/proc/const/const_20260527_094403_4bd9e613/), 30 timesteps 1995–2024).*

## 0 — Purpose & scope

The pySUTO solver collapses flow magnitudes: flows that should sit in the megatonne (Mt) range are driven to the tonne range, worst at the first year. The user's framing — *"the penalty is not high enough to preserve the scale of the constraints."* This report mines three references that solved the same balancing problem for monetary SUTs / MRIO databases, and turns their properties into concrete recommendations for pySUTO's **solver setup** and **constraint initialisation**.

Sources:
- **Lenzen, Wood & Gallego (2009)** — *Matrix balancing under conflicting information* (CRAS). [`docs/ref/Lenzen et al. - 2009 …`](../ref/)
- **Geschke, Ugon, Lenzen, Kanemoto & Moran (2019)** — *Balancing and reconciling large MRIO databases using parallel optimisation and HPC* (ACIM). [`docs/ref/Geschke et al. - 2019 …`](../ref/)
- **Lenzen et al. (2014)** — *Compiling and using IO frameworks through collaborative virtual laboratories* (IELab). [`docs/ref/Lenzen et al. - 2014 …`](../ref/)

**In scope:** weighting, penalty, conditioning, anchoring, constraint reliability, and objective reformulation. **Out of scope (assumed fixed):** the Mass/CC unit-incoherence documented in [`analysis_opt_20260521_100428_9d33a0b9_20260526_scaling.md`](analysis_opt_20260521_100428_9d33a0b9_20260526_scaling.md); balance rows are treated as dimensionally coherent here. This report builds on the four prior REPs (`…_20260526_scaling.md`, `…_20260527_flow-collapse.md`, `analysis_opt_20260521_100428_9d33a0b9.md`, `cv_bloat_analysis.md`) and does not re-derive the collapse — it adds the literature lens and the recommendations.

## 1 — Evidence on the reference run

Collapse metric: per timestep, total-mass retention `R(t) = Σ|F*| / Σ|a₀|` (over that year's edges), and the share of high-magnitude edges (`|a₀| > 1000`) that collapse below 10% (`|F*|/|a₀| < 0.1`).

| year | retention `Σ\|F*\|/Σ\|a₀\|` | high-mag edges | collapsed | collapse share |
|---|---:|---:|---:|---:|
| 1995 | **1.62e-06** | 1295 | 1295 | **100%** |
| 1996 | 1.22e-06 | 1271 | 1271 | 100% |
| 2000 | 1.31e-06 | 1600 | 1600 | 100% |
| 2010 | 9.87e-07 | 1795 | 1795 | 100% |
| 2024 | 6.26e-07 | 1590 | 1590 | 100% |

Retention min/median/max across 30 years: **6.3e-7 / 9.3e-7 / 1.6e-6** — flows lose ~6 orders of magnitude, every year. The worst single edges at 1995 (all `Mining → WR`, Mass):

| edge | a₀ (t) | F* (t) | σ_a | CV = σ_a/\|a₀\| |
|---:|---:|---:|---:|---:|
| 3780 | 8.46e6 | 7.5e-12 | 4.23e5 | **0.050** |
| 7740 | 5.40e6 | 2.2e-12 | 2.70e5 | 0.050 |
| 3660 | 2.89e6 | 4.7e-12 | 1.45e5 | 0.050 |

Two structural facts from the parent const run: **53.9%** of edges have `a₀ = 0` (FlatPrior), and the largest, most physically certain edges carry σ_a at the relative floor `CV = 0.05` exactly. (Note: cross-year `ratio_eq` deactivation — flagged in the May-27 REP — *is already implemented* in this run at [`ratio.py:186`](../../src/pysuto/constraints/ratio.py#L186); the collapse persists regardless, so it is not the remaining driver.)

## 2 — Mechanism (the live drivers)

The Stage-4 objective is the weighted least-squares of van der Ploeg / Byron form (see §3.1):
`min (a−a₀)ᵀΣ_a⁻¹(a−a₀) + εᵀΣ_c⁻¹ε  s.t.  Ga−ε = c, l ≤ a ≤ u`.

1. **The collapse penalty is bounded and magnitude-blind.** With relative σ_a ≈ `cv_min·|a₀|`, the prior cost of driving any edge to zero is `(a₀/σ_a)² = (1/cv_min)² ≈ 400` (so `|z_a|` caps at `1/cv_min = 20`), *independent of magnitude*. An 8.46 Mt edge is no better protected than an 8 t edge. ([`engine.py:_lp_regularize` L45](../../src/pysuto/constraints/engine.py#L45); defaults `sigma_a_cv_min=0.05` [L187](../../src/pysuto/constraints/engine.py#L187).)
2. **`a = 0` is feasible and cheap.** Every homogeneous row (mass_balance, ratio_eq, ratio_bound, and the many `c=0` sum_eq) is satisfied at zero. The σ_c absolute floor `1e-3` ([L190](../../src/pysuto/constraints/engine.py#L190)) makes a `c=0` soft-row slack penalty `(1/σ_c)² ≈ 1e6` — dwarfing the ~400 prior. The QP collapses big flows to relieve constraints; warm-start `a₀(t)=F*(t−1)` propagates it forward.
3. **No anchor for >50% of edges.** FlatPrior sets `a₀=0` on 53.9% of edges, so the prior term cannot pin their scale at all.

## 3 — Knowledge bits from the three papers

### 3.1 Lenzen 2009 (CRAS) — pySUTO already uses the right formulation; the collapse is the known weakness of least-squares vs RAS

- **pySUTO's QP is exactly van der Ploeg's constrained-optimisation form.** Byron: `f = (a−a₀)ᵀΣ⁻¹(a−a₀) + λᵀ(Ga−c)`. van der Ploeg adds disturbances ε to the constraints (`Ga = c + ε`), stacks `p = [a; ε]` with variances `Σ_a, Σ_c`, and **exactly-known constraints are the special case `Σ_c = 0`** (Eqs. 5–7). This is pySUTO's hard/soft split verbatim. So pySUTO sits on the constrained-optimisation branch that CRAS was built to match — the formulation is not the problem.
- **The scale-preservation property pySUTO lacks.** In RAS-type methods the reliability matrix `E` plays the role of standard deviations, and because RAS adjusts *multiplicatively* and *preserves sign*, *"the balanced estimate A\* could never go more than one standard deviation below the initial estimate A₀"* (an upper limit does not exist). RAS has a **built-in asymmetric floor against collapse**. The absolute least-squares objective has **no such floor** — it can drive a flow to 0 (or negative; the REPs found `CP→PSC = −7.22`), paying only the bounded quadratic penalty of §2.1.
- **Reliability-proportional conflict resolution.** When constraints conflict, CRAS nudges each constraint value by `±α·σ_i` toward its realisation `Ga`, capped at the actual gap. Reliable constraints (small σ) barely move; unreliable ones absorb the conflict. The σ are *genuine standard errors*, not flat floors.
- Constrained optimisation "did not reach an optimum after several hours" on the 107×107 Australian table — motivating the splitting/parallel work in Geschke 2019.

### 3.2 Geschke 2019 (ACIM) — the canonical scaling preprocessing, and reliability as genuine data quality

- **Restates pySUTO's model** (Eqs. 4/6): `min (p−p₀)ᵀ σ̂_p₀⁻² (p−p₀)  s.t.  Gp=c, l≤p≤u`, with `G=(M_hard 0; M_soft −I)`, soft ⟺ `σ_c≠0`, hard ⟺ `σ_c=0`. pySUTO = ACIM = state of the art.
- **Reliability = the standard deviation of a normal random variable per data point**, for *both* `σ_τ₀` (prior) and `σ_c` (constraints) — a genuine data-quality measure, not a uniform CV.
- **Nondimensionalisation (Eq. 5):** substitute `z = σ⁻¹p`, `A = Gσ`, giving `min (z−z₀)ᵀ(z−z₀) s.t. Az=c, σ⁻¹l ≤ z ≤ σ⁻¹u`, and normalise each constraint row to unit norm (`‖gₘ‖=1`). This is the standard way to handle heterogeneous magnitudes in this exact QP. **Caveat for pySUTO:** it is purely numerical conditioning — the collapse-penalty ceiling `(a₀/σ)² = (1/CV)²` is invariant under it, so nondimensionalisation alone does not stop the collapse.
- Bounds `l ≤ p ≤ u` are first-class; non-negativity is "the most common boundary condition." ACIM splits `G` into blocks (Cimmino/Scolnik projections) for HPC parallelism — relevant to *computational* scale, not magnitude collapse.

### 3.3 Lenzen 2014 (IELab) — always seed a populated initial estimate; enforce sign

- Compilation = **(a) build an initial estimate and constraints, then (b) optimise.** The initial estimate is built via "well-known non-survey methods" from an aggregated **proxy vector** — i.e. it is *populated and non-zero*, never left at zero cells. pySUTO's FlatPrior leaving 53.9% of `a₀` at zero is contrary to this practice.
- Constraint reliability carried as standard deviations (per KRAS, Lenzen 2009/2010); constraints expressed in a constraint language (ALANG) against the root classification.
- **Non-negativity is enforced by default**, with explicit exceptions only for "changes in inventories and subsidies." pySUTO allowing flows to go broadly negative is looser than IELab practice.
- Production solvers are parallel KRAS / Cimmino / Barrier-Penalty — again a computational-scale point.

## 4 — Synthesis: why the literature preserves scale and pySUTO does not

pySUTO uses the **correct objective** (van der Ploeg / ACIM), but mis-parameterises the pieces that the literature relies on to hold scale:

| Property the literature uses to preserve scale | In pySUTO today | Effect |
|---|---|---|
| σ reflects **genuine reliability**; high-confidence values get tight σ (Geschke, Lenzen 2009) | A flat relative *floor* `cv_min·\|a₀\|=0.05·\|a₀\|` **overrides** genuine tight σ on derived/parametric edges (Mining→WR had data CV≈1.7e-5) | Big, certain edges get a weak, bounded anchor → cheapest to collapse |
| **Marginal totals are non-zero constraints** that structurally pin scale; `a=0` is not a RAS solution | Scale-bearing sum_eq are **soft and homogeneous (`c=0`)**; `a=0` satisfies them | Nothing forces the network to carry its true magnitude |
| **Populated, non-zero initial estimate** (Lenzen 2014) | 53.9% of `a₀ = 0` (FlatPrior) | Prior term cannot anchor scale on half the edges |
| RAS **multiplicative floor**: `A* ≥ A₀ − σ`, sign preserved | Absolute least-squares: no floor; negatives allowed | Flows can be driven to ~0 / negative for a bounded cost |
| Variable **nondimensionalisation + row-norm** (Geschke Eq. 5) | Solved in raw units (Clarabel's internal equilibration only) | Conditioning is left to the solver; weighting stays magnitude-blind |

The headline: **the anchor strength must scale with edge magnitude, and the scale must be pinned by reliable non-homogeneous constraints** — exactly the two ingredients RAS/KRAS get for free (multiplicative updates + marginal totals) and that pySUTO's current parameterisation removes.

## 5 — Recommendations

### Track A — retune within the current Clarabel QP (concrete; lowest risk)

**A1. Stop inflating genuinely-tight σ_a (highest leverage).** The Lp regulariser's relative term `cv_min·|a₀|` currently *dominates and inflates* σ_a on the largest, best-determined edges (Mining→WR: data CV≈1.7e-5 → forced to 0.05). Replace the relative **floor** with a relative **ceiling** — *trust σ_a no looser than `cv·|a₀|`, but never inflate a tighter genuine σ* — or lower `sigma_a_cv_min` toward ~0.005. This raises the collapse penalty on big edges from ~400 toward `(1/CV)² ≫ 1e6`, emulating the RAS floor via genuine reliability. Anchor: [`engine.py:_lp_regularize` L45](../../src/pysuto/constraints/engine.py#L45), σ_a block [L409](../../src/pysuto/constraints/engine.py#L409); the existing comment at [L404](../../src/pysuto/constraints/engine.py#L404) already calls `cv_min·|a₀|` a "modeling claim." *Caveat:* tighter σ_a raises `κ(Σ_a⁻¹)`; the solver-side ceiling in [`_conditioning.py:condition_prior` L33](../../src/pysuto/_conditioning.py#L33) (`ceiling_factor=1e5`) caps it — verify Clarabel still converges (`res_dual ~1e-11`).

**A2. Seed FlatPrior edges with a non-zero magnitude (Lenzen 2014; cv_bloat REP strategy 1).** Replace `a₀=0` with a sector/region-median magnitude per valuation so the prior can pin scale, keeping σ wide to express low confidence. Anchor: [`src/pysuto/impute/flat_prior.py`](../../src/pysuto/impute/flat_prior.py).

**A3. Promote reliable aggregate totals to non-homogeneous, tight constraints (the RAS marginal-totals property).** Add production/trade totals as sum_eq with `c ≠ 0` and a small `σ_c`, so `a=0` becomes infeasible/expensive and the network is forced to carry its true scale. Anchor: [`src/pysuto/constraints/sum.py`](../../src/pysuto/constraints/sum.py) (`from_csv` / `from_acc_struct`).

**A4. Calibrate σ_c from real reliability, not the flat floor (CRAS / Geschke).** `sigma_c_abs_floor=1e-3` gives `c=0` rows a slack weight `~1e6` that overpowers the prior; set σ_c from pedigree/data quality and reconsider the homogeneous-row weighting. Anchor: σ_c block [`engine.py:L458`](../../src/pysuto/constraints/engine.py#L458); solver `sigma_c_floor=1e-9` [`solver.py:L130`](../../src/pysuto/solver.py#L130).

**A5. Enforce non-negativity except on designated edges (Lenzen 2014).** Set `l=0` for normal flows; allow negatives only for stock-change/sink edges (the `CP→PSC = −7.22` case). Anchor: bounds in the accounting structure / [`constraints/flow_bound.py`](../../src/pysuto/constraints/flow_bound.py).

### Track B — reformulation options (design-level; effort/risk noted)

**B1. Variable nondimensionalisation (Geschke Eq. 5).** Solve in `z=σ⁻¹a` with unit-norm `G` rows. *Effort: low–med; risk: low* (convex QP preserved). Improves conditioning and makes the prior penalty uniformly O(1). **Not a collapse cure on its own** — pair with A1/A3.

**B2. Log-domain / multiplicative objective.** `min Σ wᵢ(log aᵢ − log a₀ᵢ)²` is scale-invariant and structurally repels zero (the RAS floor analogue). *Effort: high; risk: high* — breaks constraint linearity / convex-QP form, needs zero & negative handling.

**B3. KRAS/CRAS multiplicative pre-solve, then QP refine (hybrid).** Run a RAS-type balance (preserves totals & sign, cannot collapse) to get a scale-consistent seed, then the QP for fine adjustment — mirrors EXIOBASE's multi-step practice noted in Geschke 2019. *Effort: high.*

**B4. Scale-anchoring hard totals.** The minimal subset of A3 promoted to fully hard (`σ_c=0`): the simplest structural guarantee that `a=0` is infeasible. *Effort: low; risk: med* (can create infeasibility if totals conflict — relies on the Stage-4 feasibility pre-check).

**Recommended path:** A1 + A2 + A3 first (they attack all three live drivers directly), with B1 as cheap numerical hygiene. Treat B2/B3 as the longer-term answer if retuning proves insufficient.

## 6 — Validation plan (against `opt_20260527_095157_40846da4`)

1. Apply A1 (and A2/A3) in Stage 3; rebuild the const run; re-solve Stage 4.
2. Recompute the collapse metric of §1 (same script): success = retention `Σ|F*|/Σ|a₀|` ≈ 1.0 at 1995 and high-magnitude collapse share → ~0; the `|z_a|` ceiling (currently 20) no longer the binding constraint; Mining→WR Mass edges retain `F* ≈ a₀` within their (now genuine) σ.
3. Confirm Clarabel still converges cleanly (`res_dual ~1e-11`, no iter-cap) — A1 raises `κ(Σ_a⁻¹)`.
4. Before/after per-year table + plots via [`src/pysuto/diag/optim.py`](../../src/pysuto/diag/optim.py) panels `sd_t_03_solution_vs_prior`, `sd_r_06_f_evolution`.
5. Regression guard: cross-check [`cv_bloat_analysis.md`](cv_bloat_analysis.md) so A1 does not re-introduce the σ-bloat the floor originally fixed.

## 7 — References

- Lenzen, M., Wood, R., Gallego, B. (2009). *RAS matrix balancing under conflicting information.* Economic Systems Research, DOI 10.1080/09535310802688661.
- Geschke, A., Ugon, J., Lenzen, M., Kanemoto, K., Moran, D.D. (2019). *Balancing and reconciling large MRIO databases using parallel optimisation and HPC.* Economic Structures 8:2, DOI 10.1186/s40008-019-0133-7.
- Lenzen, M. et al. (2014). *Compiling and using input–output frameworks through collaborative virtual laboratories.* (IELab.)

Key code anchors: objective [`solver.py:L248`](../../src/pysuto/solver.py#L248); prior conditioning [`_conditioning.py:L33`](../../src/pysuto/_conditioning.py#L33); σ regularisation [`engine.py:L45`](../../src/pysuto/constraints/engine.py#L45) & [L409](../../src/pysuto/constraints/engine.py#L409)/[L458](../../src/pysuto/constraints/engine.py#L458); deactivation sentinel [`sum.py:L28`](../../src/pysuto/constraints/sum.py#L28).
