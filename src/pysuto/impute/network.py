"""
network.py
==========
Process-recipe traversal imputer.

Where the other imputers fill one *edge* at a time, :class:`NetworkImputer`
reasons about *nodes*: it pushes the content valuation (copper: ``CC``) through a
supply-use network until nothing more is determinable, then derives gross-mass
twins (copper: ``Mass``) via the concentration coefficient. The valuations and the
boundary roles are read through ``acc.valuation_roles`` and ``acc.boundary_roles`` (ADR 0034).
Two node kinds carry conservation in content:

* **Process node** — a mass-closing *recipe*: ``product_CC = eff·Σ(inputs_CC)``
  and ``waste_CC = (1−eff)·Σ(inputs_CC)``, where the recovery efficiency ``eff``
  acts on the pooled primary inputs (Scrap included). The ``(1−eff)`` complement
  is routed to the residual-output node, an SB node of boundary role ``secondary``.
* **Flow node** — plain CC balance ``Σ(supply in) = Σ(uses out)``; this joins the
  product of one process to the input of the next, and resolves the Ore branch
  split (MF pyro vs SXEW hydro) from whichever side the data pins.

System-boundary (SB) nodes are open endpoints and are never balanced.

The imputer is the sole writer of **process-incident, same-region** chain edges;
cross-region traded imports are read as boundary terms but never written (BACI /
ICSG own those — see ADR 0002). The flow balance may close a same-region final-demand
leg no feed covers (ADR 0040); it never writes a stock-change leg, and a fan's
observed zero (#76) is covered, so it is never written either. eff/conc are resolved with
the same :class:`~pysuto.relation.param_relation.ParamRelationResolver` the constraint
builders use, so the emitted CC/Mass twins satisfy ``conc_*_supply``/``_use`` by
construction (generalising ADR 0002 option D to the whole chain).

The relations are the objects Stage 3 emits (#134): a Flow balance is a
:func:`~pysuto.relation.balance.node_balance_relations` row and a CC→Mass twin a
:func:`~pysuto.relation.ratio_relation.conc_twin_relation`, each solved for its lone
unknown by :func:`~pysuto.relation.prior.solve_lone_unknown`; a recipe's structure and
efficiency are its :func:`~pysuto.relation.ratio_relation.recipe_relation`, which raises
when an ``agg=true`` eff row pools a different input set than the recipe's. σ stays the
imputer's own rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pysuto.acc import BoundaryRole, boundary_role_values
from pysuto.logging import get_logger
from pysuto.lp import P_DEFAULT, lp_combine
from pysuto.relation.balance import node_balance_relations
from pysuto.relation.param_relation import (
    ParamRelationResolver,
    conc_twin_map,
    get_edge_cache,
)
from pysuto.relation.params import read_params
from pysuto.relation.prior import solve_lone_unknown
from pysuto.relation.ratio_relation import (
    conc_twin_relation,
    pooled_input_sets,
    recipe_relation,
)
from pysuto.relation.relation import Relation
from pysuto.relation.structure import process_recipes

from .base import BaseImputer, _apply_cv_cap, _reg_sensitivity

logger = get_logger(__name__)

# The param_types NetworkImputer reads from its eff and stripping_ratio files; the conc
# and ore_grade files are read by conc_twin_map (concentration).
EFF_ACCEPTS       = frozenset({"efficiency"})
STRIPPING_ACCEPTS = frozenset({"stripping_ratio"})

# The network's two inner convergence budgets, declared once (#139). Both loops are
# monotone (a fill is never undone), so a budget reached without convergence is a
# truncated fixpoint, and impute() logs it at WARNING (ADR 0010 am. 2026-11-12).
STRICT_MAX_SWEEPS  = 64  # recipe + flow-balance sweeps per strict phase
RELAXED_MAX_ROUNDS = 8   # strict -> relaxed -> strict rounds per impute() call

_EPS = 1e-12  # guard for divisions by (1−eff) when eff≈1
_SECONDARY = BoundaryRole.SECONDARY.value  # the residual-output / secondary-input role


# ---------------------------------------------------------------------------
# Resolved recipe (skeleton + attached efficiency)
# ---------------------------------------------------------------------------

@dataclass
class _RRecipe:
    inputs: tuple[int, ...]            # all use-side inputs (for Mass-twin emission)
    primary: tuple[int, ...]           # same-region Flow / SB-Nature inputs (block forward if missing)
    secondary: tuple[int, ...]         # SB-SEM inputs (Scrap): treated as 0 when missing
    bilateral_import: tuple[int, ...]  # cross-region Flow→Process imports: pooled when known,
                                       # treated as 0 when missing, non-blocking (ADR 0010)
    products: tuple[int, ...]          # Flow outputs; co-products close by sum (#130)
    wastes: tuple[int, ...]
    eff: float
    eff_sig: float
    eff_eps: float
    eff_param_id: str | None
    relation: Relation                 # Σ products − eff·Σ inputs = 0 (pooled eff)


@dataclass(frozen=True)
class _Stripping:
    """One Mining region-year's waste-rock relation ``WR_Mass = k·Ore_Mass``."""
    ore: int        # Mining→Ore Mass edge (A-side)
    wr: int         # Mining→WR Mass edge (B-side)
    k: float
    param_id: str


# ---------------------------------------------------------------------------
# NetworkImputer
# ---------------------------------------------------------------------------

class NetworkImputer(BaseImputer):
    """Process-recipe traversal imputer (ADR 0004; see module docstring)."""

    def __init__(
        self,
        eff_csv: str,
        conc_csv: str,
        ore_grade_csv: str | None = None,
        p: float = P_DEFAULT,
        *,
        q: float,
        alpha_min: float,
        cv_cap: float,
        id: str | None = None,
        stock_sectors: tuple = (),
        stripping_ratio_csv: str | None = None,
        stripping_sigma: float = 1.0e6,
        effless_stock_sigma: float = 0.0,
    ) -> None:
        self.id = id or type(self).__name__
        self._eff_csv = eff_csv
        self._conc_csv = conc_csv
        self._ore_grade_csv = ore_grade_csv
        self._stripping_csv = stripping_ratio_csv
        self.stripping_sigma = float(stripping_sigma)
        self.p = float(p)
        self.q = float(q)
        self.alpha_min = float(alpha_min)
        self.cv_cap = float(cv_cap)
        # Signed stock-change legs (edges into these SB sectors) are never written:
        # a stock change is a reported value or a zero-centred seed, never the
        # residual of a node balance (Round X, ADR 0040). They are counted when
        # known and treated as 0 when missing, like cross-region legs.
        self._stock_sectors = tuple(stock_sectors)
        # A process with no `efficiency` row closes at eff = 1.0, so its SEM output is
        # written as an exact 0 at σ 0: a missing parameter, not an observation. On a
        # signed output (structural l < 0: CP→PSC, a stock-change leg) that zero was a
        # hidden pin (Stage 3 floors σ 0 to 1e-6); it gets σ = effless_stock_sigma
        # instead, the prior staying 0 (#36, ADR 0041 am.). 0.0 keeps the pin.
        self.effless_stock_sigma = float(effless_stock_sigma)
        # Waste rock (Mining→WR) is a satellite mass flow (ADR 0038): outside the Mining
        # balance and fixed in Stage 3 by the hard stripping relation WR = k·Ore. The
        # prior carries WR Mass = k·Ore Mass so it satisfies that row, at the flat
        # `stripping_sigma` — WR has no data of its own, and an informative WR prior
        # would count the Ore prior twice through the hard row.
        # populated by prepare()
        self._stripping: list[_Stripping] = []
        self._recipes: list[_RRecipe] = []
        self._relaxed_fills: set[int] = set()
        self._assumed_absent: set[int] = set()   # primaries a relaxed close counted as 0
        self._flow_nodes: list[tuple[Relation, np.ndarray]] = []  # (CC balance, uses)
        self._edge_time: np.ndarray | None = None
        self._content: str = ""
        self._cc_to_mass: dict[int, tuple[int, float, float]] = {}
        self._same_region: np.ndarray | None = None
        self._stock_target: np.ndarray | None = None
        self._writable_mask: np.ndarray | None = None
        self._lower: np.ndarray | None = None  # per-edge bounds (for in-imputer clipping)
        self._upper: np.ndarray | None = None
        self._n_clipped: int = 0
        self._n_written: int = 0  # writes made in this impute() call (progress count)

    @property
    def source_paths(self) -> list[str]:
        out = [self._eff_csv, self._conc_csv]
        if self._ore_grade_csv:
            out.append(self._ore_grade_csv)
        if self._stripping_csv:
            out.append(self._stripping_csv)
        return out

    # -- setup --------------------------------------------------------------

    def prepare(self, acc) -> None:
        get_edge_cache(acc)  # warm the per-acc edge cache used by the resolver
        resolver = ParamRelationResolver(acc)
        # The ADR 0010 guard sets live for the whole Stage 2 run: the engine re-runs
        # impute() in every cascade pass, and a pass that forgot the earlier relaxed
        # closes re-derived 336 of copper's 359 absent primaries as 0 (#153).
        self._relaxed_fills = set()
        self._assumed_absent = set()

        # same-region flag per edge (ownership: write only same-region edges)
        em = acc.edge_meta
        same = (em["from_region"].to_numpy() == em["to_region"].to_numpy())
        self._same_region = np.zeros(acc.n_edges, dtype=bool)
        self._same_region[em["edge_id"].to_numpy()] = same
        # stock-change legs: never a write target (see __init__)
        self._stock_target = np.zeros(acc.n_edges, dtype=bool)
        if self._stock_sectors:
            is_stock = em["to_sector"].isin(self._stock_sectors).to_numpy()
            self._stock_target[em["edge_id"].to_numpy()] = is_stock
        self._writable_mask = self._same_region & ~self._stock_target
        self._edge_time = np.empty(acc.n_edges, dtype=object)
        self._edge_time[em["edge_id"].to_numpy()] = em["time"].astype(str).to_numpy()
        self._content = acc.valuation_roles.content

        # per-edge bounds: clip fills to [l, u] inside _w so a negative residual
        # (e.g. Ore→SXEW = Mining→Ore − Ore→MF) cannot propagate through the
        # internal fixpoint and drag downstream sums below their true values.
        self._lower = acc.bounds.l
        self._upper = acc.bounds.u

        # CC edge -> (Mass twin edge, conc, σ_conc). conc first, ore_grade
        # overlaid (more specific for CO → wins, since it is applied second).
        self._cc_to_mass = conc_twin_map(self._conc_csv, resolver)
        if self._ore_grade_csv:
            conc_twin_map(self._ore_grade_csv, resolver, self._cc_to_mass)

        # product CC edge -> (eff, σ_eff, ε, param_id)
        product_eff = self._load_eff_map(self._eff_csv, resolver)

        if self._stripping_csv:
            self._stripping = self._load_stripping(self._stripping_csv, resolver)

        # input edge -> from-node boundary role (classifies primary vs secondary input)
        role_from = dict(zip(em["edge_id"].to_numpy(),
                             boundary_role_values(acc.boundary_roles, em["from_sector"])))
        # input edge -> from/to region (classifies same-region vs cross-region import)
        reg_from = dict(zip(em["edge_id"].to_numpy(), em["from_region"].to_numpy()))
        reg_to   = dict(zip(em["edge_id"].to_numpy(), em["to_region"].to_numpy()))

        # attach eff to each structural recipe
        self._recipes = []
        pooled_sets: dict = {}  # id(agg eff row) -> {(region, time): input edge set}
        for rec in process_recipes(acc):
            # One pooled efficiency per process: Σ products = eff · Σ inputs (#130). The
            # eff map is keyed by product edge, so co-products must resolve to one row.
            effs = [product_eff.get(e, (1.0, 0.0, 0.0, None, None)) for e in rec.products]
            if len({e[3] for e in effs}) > 1:
                raise ValueError(
                    f"process {rec.process!r} ({rec.region}, {rec.time}) has "
                    f"{len(rec.products)} Flow products {rec.products} with different "
                    "efficiency rows; co-products close by the sum, one pooled row"
                )
            eff, eff_sig, eff_eps, pid, eff_row = (
                effs[0] if effs else (1.0, 0.0, 0.0, None, None))
            # the agg=true row declares the pooling: its input set must be the recipe's
            pooled = None
            if eff_row is not None and eff_row.agg:
                if id(eff_row) not in pooled_sets:
                    pooled_sets[id(eff_row)] = pooled_input_sets(
                        eff_row, acc, self._content)
                pooled = pooled_sets[id(eff_row)]
            relation = recipe_relation(
                rec, float(eff), valuation=self._content, variant=pid or "ratioeq",
                sigma=float(eff_sig) / abs(eff) if eff else 0.0, hard=pid is None,
                pooled=pooled, row_id=pid or "",
            )
            secondary = tuple(i for i in rec.inputs if role_from.get(i) == _SECONDARY)
            non_sem = [i for i in rec.inputs if role_from.get(i) != _SECONDARY]
            # same-region non-SEM inputs block the forward gate; cross-region imports
            # are pooled-when-known / 0-when-missing and never block (ADR 0010).
            primary = tuple(i for i in non_sem if reg_from.get(i) == reg_to.get(i))
            bilateral_import = tuple(i for i in non_sem if reg_from.get(i) != reg_to.get(i))
            self._recipes.append(
                _RRecipe(rec.inputs, primary, secondary, bilateral_import,
                         rec.products, rec.wastes,
                         float(eff), float(eff_sig), float(eff_eps), pid, relation)
            )

        # Flow-node CC balances (relation, its use edges): Σ(supply in) − Σ(uses out) = 0
        self._flow_nodes = [
            (rel, rel.edges[rel.coefficients < 0])
            for rel in node_balance_relations(acc, self._content, entities={"Flow"})
        ]

    def _load_eff_map(self, path, resolver) -> dict:
        # Recovery-efficiency rows only: any other process ratio (the stripping
        # ratio, a Mass output ratio with its own file since ADR 0038) must not
        # pollute the eff map. Filtered by declared param_type, not by a magic param_id.
        best: dict[int, tuple] = {}  # product_cc -> (eff, σ, ε, param_id, row, specificity)
        for row in read_params(path, EFF_ACCEPTS):
            nominal = row.require_nominal()
            sig = row.sigma()
            eps = row.epsilon
            pid = row.param_id
            for rr in resolver.resolve_row(row):
                prod = rr.b_edge_id  # B-side = process→product (the recovered output)
                if prod not in best or rr.specificity > best[prod][5]:
                    best[prod] = (nominal, sig, eps, pid, row, rr.specificity)
        return {k: v[:5] for k, v in best.items()}

    def _load_stripping(self, path, resolver) -> list[_Stripping]:
        best: dict[int, tuple[_Stripping, int]] = {}  # WR Mass edge -> (rel, specificity)
        for row in read_params(path, STRIPPING_ACCEPTS):
            k, pid = row.require_nominal(), row.param_id
            for rr in resolver.resolve_row(row):
                wr = rr.b_edge_id  # B-side = Mining→WR Mass, A-side = Mining→Ore Mass
                if wr not in best or rr.specificity > best[wr][1]:
                    best[wr] = (_Stripping(rr.a_edge_id, wr, k, pid), rr.specificity)
        return [rel for rel, _ in best.values()]

    def _do_stripping(self, f, sigma, ledger, step) -> None:
        """WR Mass = k·Ore Mass at the flat σ, uncapped (see __init__)."""
        for s in self._stripping:
            if np.isnan(f[s.wr]) and not np.isnan(f[s.ore]):
                self._w(s.wr, s.k * f[s.ore], self.stripping_sigma, ledger, step,
                        "network_stripping", [s.ore], [s.k], s.param_id, uncapped=True)

    def writer_scope(self, acc) -> np.ndarray:
        """Same-region edges only, and never a stock-change leg."""
        return self._writable_mask

    def _writable(self, e: int) -> bool:
        """The imputer's own gate: whether ``e`` lies in its declared scope."""
        return bool(self._writable_mask[e])

    # -- imputation ---------------------------------------------------------

    def impute(self, ledger, step, acc_struct) -> None:
        f, sigma = ledger.f, ledger.sigma
        self._n_clipped = 0
        self._n_written = 0
        # Two-phase fixpoint. Strict phase: keep sweeping recipes + flow
        # balances until a full sweep fills nothing new (the engine cascade
        # re-runs us too, but converging here keeps chain propagation within
        # one call). Relaxed phase (ADR 0010 amendment): only at strict
        # convergence may a still-missing pooled primary be treated as an
        # absent parallel route (count 0, ≥1 primary known) — convergence
        # proves "missing" means *underivable*, not "not yet derived". A
        # relaxed close can unlock further strict fills, so re-converge.
        for _outer in range(RELAXED_MAX_ROUNDS):
            for _ in range(STRICT_MAX_SWEEPS):
                n_before = self._n_written
                for rec in self._recipes:
                    self._do_recipe(rec, f, sigma, ledger, step)
                for rel, out in self._flow_nodes:
                    self._do_flow_balance(rel, out, f, sigma, ledger, step)
                if self._n_written == n_before:
                    break
            else:
                logger.warning("NetworkImputer %s: strict phase hit STRICT_MAX_SWEEPS=%d "
                               "without converging", self.id, STRICT_MAX_SWEEPS)
            n_before = self._n_written
            for rec in self._recipes:
                self._do_recipe(rec, f, sigma, ledger, step,
                                relax_primaries=True)
            if self._n_written == n_before:
                break
        else:
            logger.warning("NetworkImputer %s: hit RELAXED_MAX_ROUNDS=%d without "
                           "converging", self.id, RELAXED_MAX_ROUNDS)
        # WR is a leaf (satellite): its fill unlocks nothing, so one pass at the end
        self._do_stripping(f, sigma, ledger, step)
        if self._n_clipped:
            # One summary in place of the engine's per-edge out-of-bounds warnings
            # (now silent, since fills are clipped to bounds before they propagate).
            logger.info("NetworkImputer: clipped %d fill(s) to edge bounds", self._n_clipped)

    def _do_recipe(self, rec, f, sigma, ledger, step,
                   relax_primaries: bool = False) -> None:
        p = self.p
        PS, W = list(rec.products), list(rec.wastes)
        P = PS[0] if len(PS) == 1 else None    # the lone product; None for 0 or 2+
        PR, SE = list(rec.primary), list(rec.secondary)
        BI = list(rec.bilateral_import)
        eff, eff_sig, eps = rec.eff, rec.eff_sig, rec.eff_eps

        miss_PR = [i for i in PR if np.isnan(f[i])]
        miss_P = [i for i in PS if np.isnan(f[i])]
        # missing SEM secondaries (Scrap) AND missing cross-region imports count as
        # absent → 0; neither blocks the forward gate (ADR 0010). In the relaxed
        # phase (post strict-fixpoint only, see impute()), a missing same-region
        # primary also counts as 0 *provided ≥1 primary is known* (ADR 0010
        # amendment: parallel technology routes, e.g. CP's CER+CSXEW — a leg
        # still underivable at convergence means the region doesn't have that
        # route). A recipe with NO known primary keeps the gate closed
        # (no fabricated zeros).
        sec_sum = float(sum(f[i] for i in SE if not np.isnan(f[i])))
        sec_sig = [sigma[i] for i in SE if not np.isnan(f[i])]
        imp_sum = float(sum(f[i] for i in BI if not np.isnan(f[i])))
        imp_sig = [sigma[i] for i in BI if not np.isnan(f[i])]
        known_PR = [i for i in PR if not np.isnan(f[i])]

        # ---- determine total primary input CC (S) and its σ ----
        S = None
        S_src = None
        sig_S = 0.0
        if PR and (not miss_PR or (relax_primaries and known_PR)):
            S = float(sum(f[i] for i in known_PR)) + sec_sum + imp_sum
            S_src = "inputs"
            sig_S = lp_combine([sigma[i] for i in known_PR] + sec_sig + imp_sig, p)
        elif relax_primaries and not known_PR and sec_sum > 0.0:
            # 100 %-SECONDARY process (ADR 0010 amendment, Round P.1). The rule
            # above keeps the gate closed with no known primary so that a pure
            # data gap never fabricates output. But a process that is entirely
            # scrap-fed has no primary *by construction*, and its input is not
            # missing — it is measured, on the SEM leg. Applying the same
            # reasoning the amendment already uses for parallel routes ("a leg
            # still underivable at convergence means the region does not have
            # that route"), at convergence no known primary plus a known
            # secondary means the region has no primary route, not that its
            # inputs are unknown.
            # Slovakia is the case that forced this: ICSG reports ~46.6 kt/yr of
            # scrap into its smelter and the prior carried 0 kt out for 15
            # region-years, so `usetotout Matte` was pinned at c = 0.011 kt with
            # σ = 0.001 and strangled the chain — the scrap level (|z| −58.7) and
            # the BACI anode anchor (|z| −50.7) were both violated to satisfy it.
            # Relaxed phase only, so anything derivable the ordinary way already
            # is; `relaxed_close` below then bars filling the absent primary by
            # subtraction, keeping that route an honest NaN gap.
            S = sec_sum + imp_sum
            S_src = "inputs"
            sig_S = lp_combine(sec_sig + imp_sig, p)
        elif PS and not miss_P and eff > _EPS:
            # Σ products = eff·S: with several products the sum and its lp_combine σ
            # stand in for the lone product's value and σ (single branch exact, #130)
            f_PS = f[P] if P is not None else float(sum(f[i] for i in PS))
            sig_PS = sigma[P] if P is not None else lp_combine([sigma[i] for i in PS], p)
            S = float(f_PS / eff)
            S_src = "product"
            sig_S = lp_combine(
                [_reg_sensitivity(1.0 / eff, self.q, self.alpha_min) * sig_PS,
                 abs(S) / eff * eff_sig], p,
            )
        elif len(W) == 1 and not np.isnan(f[W[0]]) and (1.0 - eff) > _EPS:
            S = float(f[W[0]] / (1.0 - eff))
            S_src = "waste"
            sig_S = lp_combine(
                [_reg_sensitivity(1.0 / (1.0 - eff), self.q, self.alpha_min) * sigma[W[0]],
                 abs(S) / (1.0 - eff) * eff_sig], p,
            )
        if S is None:
            return

        eff_term = eff_sig * abs(S) if S_src == "inputs" else 0.0
        src_P = PS if PS else [None]    # the products as a fill's sources

        # ---- product = eff·S ----
        relaxed_close = bool(relax_primaries and miss_PR and S_src == "inputs")
        tag = dict(relaxed=relaxed_close, assumed_absent=miss_PR if relaxed_close else ())
        if len(miss_P) == 1:
            # one unknown product is the sum's remainder; two or more stay NaN for the
            # later fillers (#130). A lone product keeps the exact expression eff·S.
            P_miss = miss_P[0]
            known_P = [i for i in PS if i != P_miss]
            f_P = eff * S
            s_terms = [eff * sig_S, eff_term]
            if known_P:
                f_P = f_P - float(sum(f[i] for i in known_P))
                s_terms += [sigma[i] for i in known_P]
            s_P = lp_combine(s_terms + [eps * abs(f_P)], p)
            self._w(P_miss, f_P, s_P, ledger, step,
                    "network_forward",
                    known_PR + known_P if S_src == "inputs" else src_P, [eff],
                    rec.eff_param_id, **tag)
            if relaxed_close:
                self._relaxed_fills.add(P_miss)

        # ---- single missing PRIMARY input by subtraction (S from output) ----
        # (secondaries + cross-region imports are pooled when known / 0 when missing;
        # never written here, so they must be subtracted off the recovered S)
        # A relaxed-close output must not circle back into the primary it
        # assumed absent (that would echo the missing=0 assumption as an
        # explicit 0 — the route stays an honest NaN gap, ADR 0010).
        if S_src in ("product", "waste") and len(miss_PR) == 1 \
                and not (any(i in self._relaxed_fills for i in PS) if S_src == "product"
                         else W[0] in self._relaxed_fills):
            i = miss_PR[0]
            f_i = S - float(sum(f[j] for j in known_PR)) - sec_sum - imp_sum
            s_i = lp_combine(
                [sig_S] + [sigma[j] for j in known_PR] + sec_sig + imp_sig + [eps * abs(f_i)], p,
            )
            self._w(i, f_i, s_i, ledger, step,
                    "network_backward", PS if S_src == "product" else list(W),
                    [1.0 / eff if S_src == "product" else 1.0],
                    rec.eff_param_id)

        # ---- waste = (1−eff)·S ----
        miss_W = [w for w in W if np.isnan(f[w])]
        if len(miss_W) == 1:
            w = miss_W[0]
            total = (1.0 - eff) * S
            f_w = total - float(sum(f[x] for x in W if x != w and not np.isnan(f[x])))
            method = "network_forward" if S_src == "inputs" else "network_backward"
            s_w = lp_combine([(1.0 - eff) * sig_S, eff_term, eps * abs(f_w)], p)
            if rec.eff_param_id is None and self._lower[w] < 0.0:   # eff-less stock leg
                s_w = max(s_w, self.effless_stock_sigma)
            self._w(w, f_w, s_w, ledger, step, method,
                    known_PR if S_src == "inputs" else src_P, [1.0 - eff],
                    rec.eff_param_id, **tag)
            if relaxed_close:
                self._relaxed_fills.add(w)

        # ---- Mass twins for this recipe's CC edges (atomic with the CC fills) ----
        self._emit_twins(rec, f, sigma, ledger, step)

    def _emit_twins(self, rec, f, sigma, ledger, step) -> None:
        cc_edges = list(rec.inputs) + list(rec.products) + list(rec.wastes)
        for cc in cc_edges:
            if np.isnan(f[cc]):
                continue
            # WR (conc=0) has no conc twin — its Mass is k·Ore (_do_stripping)
            self._mass_twin(cc, f, sigma, ledger, step)

    def _mass_twin(self, cc, f, sigma, ledger, step) -> None:
        """Mass twin of a known CC edge: solve its concentration relation for the Mass."""
        twin = self._cc_to_mass.get(cc)
        if twin is None or np.isnan(f[cc]):
            return
        m, conc, conc_sig = twin
        if not np.isnan(f[m]):
            return
        rel = conc_twin_relation(cc, m, conc, conc_sig, time=self._edge_time[cc],
                                 valuation=self._content)
        fill = solve_lone_unknown(rel, f)
        f_m = fill.value
        s_m = lp_combine([sigma[cc] / conc, abs(f_m) * conc_sig / conc], self.p)
        self._w(m, f_m, s_m, ledger, step,
                "network_mass_twin", fill.sources, fill.coefficients, None)

    def _do_flow_balance(self, rel, out, f, sigma, ledger, step) -> None:
        """Flow-node CC balance: Σ(supply in) = Σ(uses out); fill a lone gap."""
        # only act if exactly one *writable* (same-region, non-stock) edge is unknown
        fill = solve_lone_unknown(rel, f, self._writable_mask)
        if fill is None:
            return
        e = fill.edge
        # A primary a relaxed close counted as 0 is never re-derived by a balance either:
        # the prior would hold both its 0 (inside the close's output) and the balance's
        # value. The route stays an honest NaN gap, as for network_backward (#153).
        if e in self._assumed_absent:
            return
        # every other *writable* edge is therefore known; missing cross-region legs
        # and missing stock-change legs (non-writable) are treated as 0 (no trade on
        # that route / no stock change) — symmetric with the recipe's
        # bilateral_import tier (ADR 0010). They are counted as 0, never written,
        # so the uncovered edge stays an honest gap (flat prior, or the zero-mean
        # net-stock seed for a stock leg — ADR 0040).
        # A node whose uses include a MISSING stock-change leg has no observed use
        # total (the seed's 0 is an expectation, not a report), so its SUPPLY cannot
        # be derived from its uses: that would pin production to trade and send the
        # cathode contradiction into the prior's production edge instead of into a
        # stock (Round X: 836 CP→Cathode fills, a fabricated German mine chain).
        # Production stays the recipe's; only a lone missing USE may be closed here.
        # Once every stock leg among the uses is reported, the use total is observed
        # and the supply closes from it as for any node (#38).
        if fill.edge_coefficient > 0 and (
                self._stock_target[out] & np.isnan(f[out])).any():
            return
        s_e = lp_combine([sigma[x] for x in fill.sources], self.p)
        # the logged coefficients are the magnitudes: a balance's ±1 are one shared weight
        self._w(e, fill.value, s_e, ledger, step,
                "network_balance", fill.sources, np.abs(fill.coefficients), None)
        # emit the Mass twin for the freshly-balanced CC edge
        self._mass_twin(e, f, sigma, ledger, step)

    # -- write helper -------------------------------------------------------

    def _w(self, e, val, sig, ledger, step, method,
           sources, coeffs, param_id, desc="", uncapped=False,
           relaxed=False, assumed_absent=()) -> bool:
        """Write a fill through the ledger, behind the imputer's own scope gate.

        The value is clipped to the edge's ``[lower, upper]`` bounds *before* it is
        written, so a negative residual (e.g. ``Ore→SXEW = Mining→Ore − Ore→MF`` in
        a no-SXEW region) cannot propagate through the internal fixpoint and drag
        downstream sums below their true values. A clipped fill widens σ to the clip
        gap (a saturated estimate, not a precise bound) and is tagged ``_clipped``.
        """
        if not self._writable(e) or not np.isnan(ledger.f[e]):
            return False
        val = float(val)
        sig = float(sig)
        clipped_val = float(min(max(val, self._lower[e]), self._upper[e]))
        was_clipped = clipped_val != val
        if was_clipped:
            sig = lp_combine([sig, abs(val - clipped_val)], self.p)  # widen σ to the clip gap
            val = clipped_val
            self._n_clipped += 1
        capped = False
        if not uncapped:  # cap self-disables at val==0
            sig, capped = _apply_cv_cap(sig, val, self.cv_cap)
        if not ledger.write(
            e, val, float(sig), step,
            method + ("_clipped" if was_clipped else "") + ("_capped" if capped else ""),
            [int(s) for s in sources], [float(c) for c in coeffs], desc, param_id,
            relaxed, assumed_absent,
        ):
            return False
        self._n_written += 1
        self._assumed_absent.update(assumed_absent)   # its 0s now sit in the prior
        return True
