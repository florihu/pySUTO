"""
network.py
==========
Process-recipe traversal imputer.

Where the other imputers fill one *edge* at a time, :class:`NetworkImputer`
reasons about *nodes*: it pushes copper content (``CC``) through a supply-use
network until nothing more is determinable, then derives ``Mass`` twins via the
concentration coefficient. Two node kinds carry conservation in CC:

* **Process node** — a mass-closing *recipe*: ``product_CC = eff·Σ(inputs_CC)``
  and ``waste_CC = (1−eff)·Σ(inputs_CC)``, where the recovery efficiency ``eff``
  acts on the pooled primary inputs (Scrap included). The ``(1−eff)`` complement
  is routed to the SEM residual-output node.
* **Flow node** — plain CC balance ``Σ(supply in) = Σ(uses out)``; this joins the
  product of one process to the input of the next, and resolves the Ore branch
  split (MF pyro vs SXEW hydro) from whichever side the data pins.

System-boundary (SB) nodes are open endpoints and are never balanced.

The imputer is the sole writer of **process-incident, same-region** chain edges;
cross-region traded imports and final-demand edges are read as boundary terms but
never written (BACI / ICSG own those — see ADR 0002). eff/conc are resolved with
the same :class:`~pysuto.param_relation.ParamRelationResolver` the constraint
builders use, so the emitted CC/Mass twins satisfy ``conc_*_supply``/``_use`` by
construction (generalising ADR 0002 option D to the whole chain).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pysuto.logging import get_logger
from pysuto.lp import lp_combine
from pysuto.param_relation import (
    ParamRelationResolver,
    conc_twin_map,
    float_or,
    get_edge_cache,
    sigma_from_row,
)

from .base import BaseImputer, _apply_cv_cap, _log
from .parametric import _reg_sensitivity

logger = get_logger(__name__)

_EPS = 1e-12  # guard for divisions by (1−eff) when eff≈1


# ---------------------------------------------------------------------------
# Structural recipe skeleton (parameter-free; built from acc only)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProcessRecipe:
    """One Process node's CC supply-use skeleton (edge ids, no parameters).

    inputs   use-side CC edge ids into the process (any region; incl. Scrap).
    product  the single supply CC edge to a Flow node, or None.
    wastes   supply CC edge ids to SB nodes with entity_sub == "SEM".
    """

    process_sector: str
    region: str
    time: str
    inputs: tuple[int, ...]
    product: int | None
    wastes: tuple[int, ...]


def derive_process_recipes(acc) -> list[ProcessRecipe]:
    """Build :class:`ProcessRecipe` skeletons from ``acc`` (masks + entity_sub).

    Reads ``edge_meta`` + the ``supply``/``use`` masks + ``entity_sub`` — all of
    which survive the save/load round-trip — so it works whether or not
    ``acc.ruleset`` is set.
    """
    em = acc.edge_meta
    supply = np.asarray(acc.masks["supply"], dtype=bool)
    use = np.asarray(acc.masks["use"], dtype=bool)
    is_cc = (em["valuation"].to_numpy() == "CC")

    sup = em[supply & is_cc & (em["from_entity"].to_numpy() == "Process")]
    use_e = em[use & is_cc & (em["to_entity"].to_numpy() == "Process")]

    recipes: list[ProcessRecipe] = []
    for (proc, region, time), g in sup.groupby(
        ["from_sector", "from_region", "time"], sort=False
    ):
        flow_out = g[g["to_entity"].to_numpy() == "Flow"]
        product = int(flow_out["edge_id"].iloc[0]) if len(flow_out) else None
        waste_rows = g[
            (g["to_entity"].to_numpy() == "SB")
            & (g["to_entity_sub"].to_numpy() == "SEM")
        ]
        wastes = tuple(int(e) for e in waste_rows["edge_id"])

        ig = use_e[
            (use_e["to_sector"].to_numpy() == proc)
            & (use_e["to_region"].to_numpy() == region)
            & (use_e["time"].to_numpy() == time)
        ]
        inputs = tuple(int(e) for e in ig["edge_id"])
        recipes.append(
            ProcessRecipe(str(proc), str(region), str(time), inputs, product, wastes)
        )
    return recipes


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
    product: int | None
    wastes: tuple[int, ...]
    eff: float
    eff_sig: float
    eff_eps: float
    eff_param_id: str | None
    special: str | None  # "stripping_ratio" for Mining, else None


# ---------------------------------------------------------------------------
# NetworkImputer
# ---------------------------------------------------------------------------

class NetworkImputer(BaseImputer):
    """Process-recipe traversal imputer (see module docstring)."""

    def __init__(
        self,
        eff_csv: str,
        conc_csv: str,
        ore_grade_csv: str | None = None,
        p: float = 1.3,
        q: float = 3.0,
        alpha_min: float = 0.05,
        cv_cap: float = 1.5,
        stripping_ratio: float = 2.0,
        id: str | None = None,
    ) -> None:
        self.id = id or type(self).__name__
        self._eff_csv = eff_csv
        self._conc_csv = conc_csv
        self._ore_grade_csv = ore_grade_csv
        self.p = float(p)
        self.q = float(q)
        self.alpha_min = float(alpha_min)
        self.cv_cap = float(cv_cap)
        # Mining Mass close: waste-rock Mass is the clipped input−output residual
        # WR_Mass = max(0, CO_Mass − Ore_Mass) (Ore stays grade-anchored), so the hard
        # Mining Mass balance holds in the seed without fabricating phantom mass. WR is a
        # free slack excluded from the scale total (ADR 0006); the stripping ratio is NOT
        # used by the imputer anymore — it survives only as the soft `ratio` constraint
        # (built independently in Stage 3 from the eff CSV). See ASSUMPTIONS.md. This
        # constructor value is retained as the documented model default.
        self.stripping_ratio = float(stripping_ratio)
        # populated by prepare()
        self._recipes: list[_RRecipe] = []
        self._relaxed_fills: set[int] = set()
        self._flow_nodes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self._cc_to_mass: dict[int, tuple[int, float, float]] = {}
        # structural CC→Mass edge twin (same from/to/time), conc-independent — locates the
        # WR Mass edge whose conc=0 keeps it out of the conc twin map above.
        self._cc_to_mass_struct: dict[int, int] = {}
        self._same_region: np.ndarray | None = None
        self._lower: np.ndarray | None = None  # per-edge bounds (for in-imputer clipping)
        self._upper: np.ndarray | None = None
        self._n_clipped: int = 0

    @property
    def source_paths(self) -> list[str]:
        out = [self._eff_csv, self._conc_csv]
        if self._ore_grade_csv:
            out.append(self._ore_grade_csv)
        return out

    # -- setup --------------------------------------------------------------

    def prepare(self, acc) -> None:
        get_edge_cache(acc)  # warm the per-acc edge cache used by the resolver
        resolver = ParamRelationResolver(acc)

        # same-region flag per edge (ownership: write only same-region edges)
        em = acc.edge_meta
        same = (em["from_region"].to_numpy() == em["to_region"].to_numpy())
        self._same_region = np.zeros(acc.n_edges, dtype=bool)
        self._same_region[em["edge_id"].to_numpy()] = same

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

        # Structural CC->Mass edge twin (same from/to/time), independent of conc — the
        # conc twin map drops conc=0 edges, so the WR (copperless) Mass edge needs this
        # to be located for the Mining WR = CO − Ore residual fill.
        cc_e   = em[em["valuation"] == "CC"][["from_id", "to_id", "time", "edge_id"]]
        mass_e = em[em["valuation"] == "Mass"][["from_id", "to_id", "time", "edge_id"]]
        pair = cc_e.merge(mass_e, on=["from_id", "to_id", "time"], suffixes=("_cc", "_mass"))
        self._cc_to_mass_struct = dict(zip(pair["edge_id_cc"].to_numpy(),
                                           pair["edge_id_mass"].to_numpy()))

        # product CC edge -> (eff, σ_eff, ε, param_id)
        product_eff = self._load_eff_map(self._eff_csv, resolver)

        # input edge -> from-node entity_sub (classifies primary vs SEM secondary)
        sub_from = dict(zip(em["edge_id"].to_numpy(), em["from_entity_sub"].to_numpy()))
        # input edge -> from/to region (classifies same-region vs cross-region import)
        reg_from = dict(zip(em["edge_id"].to_numpy(), em["from_region"].to_numpy()))
        reg_to   = dict(zip(em["edge_id"].to_numpy(), em["to_region"].to_numpy()))

        # attach eff to each structural recipe
        self._recipes = []
        for rec in acc.process_recipes:
            eff, eff_sig, eff_eps, pid = product_eff.get(
                rec.product, (1.0, 0.0, 0.0, None)
            )
            secondary = tuple(i for i in rec.inputs if sub_from.get(i) == "SEM")
            non_sem = [i for i in rec.inputs if sub_from.get(i) != "SEM"]
            # same-region non-SEM inputs block the forward gate; cross-region imports
            # are pooled-when-known / 0-when-missing and never block (ADR 0010).
            primary = tuple(i for i in non_sem if reg_from.get(i) == reg_to.get(i))
            bilateral_import = tuple(i for i in non_sem if reg_from.get(i) != reg_to.get(i))
            special = "stripping_ratio" if rec.process_sector == "Mining" else None
            self._recipes.append(
                _RRecipe(rec.inputs, primary, secondary, bilateral_import,
                         rec.product, rec.wastes,
                         float(eff), float(eff_sig), float(eff_eps), pid, special)
            )

        # Flow-node CC balances: (incoming supply edges, outgoing use edges)
        self._flow_nodes = self._build_flow_balances(acc)

    def _load_eff_map(self, path, resolver) -> dict:
        df = pd.read_csv(path, sep=None, engine="python", comment="#", dtype=str)
        df = df.dropna(subset=["param_id"])
        df = df[~df["param_id"].str.startswith("#")]
        # Recovery-efficiency rows only; the stripping_ratio row (a Mass output
        # ratio, not a CC recovery fraction) lives in the same file and must not
        # pollute the eff map. Filter by param_type, not by a magic param_id.
        df = df[df["param_type"].astype(str).str.strip() == "efficiency"]
        best: dict[int, tuple] = {}  # product_cc -> (eff, σ, ε, param_id, specificity)
        for _, row in df.iterrows():
            try:
                nominal = float(row["value_nominal"])
                sig = sigma_from_row(row, nominal)
            except Exception as exc:  # noqa: BLE001
                logger.warning("eff map %s: row %s skipped (%s)",
                               path, row.get("param_id"), exc)
                continue
            eps = float_or(row.get("epsilon"), 0.0)
            pid = str(row["param_id"])
            for rr in resolver.resolve_row(row):
                prod = rr.b_edge_id  # B-side = process→product (the recovered output)
                if prod not in best or rr.specificity > best[prod][4]:
                    best[prod] = (nominal, sig, eps, pid, rr.specificity)
        return {k: v[:4] for k, v in best.items()}

    def _build_flow_balances(self, acc):
        """Per **(Flow node, year)**: (incoming CC supply edges, outgoing CC use edges).

        Keyed by ``(node_id, time)``. ``node_index`` carries no time axis, so
        grouping on the bare node id would pool all years onto one node — leaving
        the balance with as many unknowns as years and the single-unknown
        determinacy guard in :meth:`_do_flow_balance` could never fire.
        """
        em = acc.edge_meta
        cc = em[em["valuation"].to_numpy() == "CC"]
        flow_ids = set(acc.node_index.index[acc.node_index["entity"].to_numpy() == "Flow"])
        by_to = {k: g["edge_id"].to_numpy() for k, g in cc.groupby(["to_id", "time"])}
        by_from = {k: g["edge_id"].to_numpy() for k, g in cc.groupby(["from_id", "time"])}
        balances = []
        for key in by_to.keys() & by_from.keys():  # nodes/years with both in- and out-edges
            if key[0] not in flow_ids:
                continue
            inc = tuple(int(e) for e in by_to[key])
            out = tuple(int(e) for e in by_from[key])
            balances.append((inc, out))
        return balances

    # -- imputation ---------------------------------------------------------

    def impute(self, f, sigma, missing, step, acc_struct) -> list[dict]:
        entries: list[dict] = []
        self._n_clipped = 0
        self._relaxed_fills = set()
        # Two-phase fixpoint. Strict phase: keep sweeping recipes + flow
        # balances until a full sweep fills nothing new (the engine cascade
        # re-runs us too, but converging here keeps chain propagation within
        # one call). Relaxed phase (ADR 0010 amendment): only at strict
        # convergence may a still-missing pooled primary be treated as an
        # absent parallel route (count 0, ≥1 primary known) — convergence
        # proves "missing" means *underivable*, not "not yet derived". A
        # relaxed close can unlock further strict fills, so re-converge.
        for _outer in range(8):
            for _ in range(64):
                n_before = len(entries)
                for rec in self._recipes:
                    self._do_recipe(rec, f, sigma, entries, step)
                for inc, out in self._flow_nodes:
                    self._do_flow_balance(inc, out, f, sigma, entries, step)
                if len(entries) == n_before:
                    break
            n_before = len(entries)
            for rec in self._recipes:
                self._do_recipe(rec, f, sigma, entries, step,
                                relax_primaries=True)
            if len(entries) == n_before:
                break
        if self._n_clipped:
            # One summary in place of the engine's per-edge out-of-bounds warnings
            # (now silent, since fills are clipped to bounds before they propagate).
            logger.info("NetworkImputer: clipped %d fill(s) to edge bounds", self._n_clipped)
        return entries

    def _do_recipe(self, rec, f, sigma, entries, step,
                   relax_primaries: bool = False) -> None:
        p = self.p
        P, W = rec.product, list(rec.wastes)
        PR, SE = list(rec.primary), list(rec.secondary)
        BI = list(rec.bilateral_import)
        eff, eff_sig, eps = rec.eff, rec.eff_sig, rec.eff_eps

        miss_PR = [i for i in PR if np.isnan(f[i])]
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
        elif P is not None and not np.isnan(f[P]) and eff > _EPS:
            S = float(f[P] / eff)
            S_src = "product"
            sig_S = lp_combine(
                [_reg_sensitivity(1.0 / eff, self.q, self.alpha_min) * sigma[P],
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

        # ---- product = eff·S ----
        relaxed_close = bool(relax_primaries and miss_PR and S_src == "inputs")
        if P is not None and np.isnan(f[P]):
            f_P = eff * S
            s_P = lp_combine([eff * sig_S, eff_term, eps * abs(f_P)], p)
            self._w(P, f_P, s_P, f, sigma, entries, step,
                    "network_forward", known_PR if S_src == "inputs" else [P], [eff],
                    rec.eff_param_id)
            if relaxed_close:
                self._relaxed_fills.add(P)

        # ---- single missing PRIMARY input by subtraction (S from output) ----
        # (secondaries + cross-region imports are pooled when known / 0 when missing;
        # never written here, so they must be subtracted off the recovered S)
        # A relaxed-close output must not circle back into the primary it
        # assumed absent (that would echo the missing=0 assumption as an
        # explicit 0 — the route stays an honest NaN gap, ADR 0010).
        if S_src in ("product", "waste") and len(miss_PR) == 1 \
                and (P if S_src == "product" else W[0]) not in self._relaxed_fills:
            i = miss_PR[0]
            f_i = S - float(sum(f[j] for j in known_PR)) - sec_sum - imp_sum
            s_i = lp_combine(
                [sig_S] + [sigma[j] for j in known_PR] + sec_sig + imp_sig + [eps * abs(f_i)], p,
            )
            self._w(i, f_i, s_i, f, sigma, entries, step,
                    "network_backward", [P] if S_src == "product" else list(W),
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
            self._w(w, f_w, s_w, f, sigma, entries, step, method,
                    known_PR if S_src == "inputs" else [P], [1.0 - eff], rec.eff_param_id)
            if relaxed_close:
                self._relaxed_fills.add(w)

        # ---- Mass twins for this recipe's CC edges (atomic with the CC fills) ----
        self._emit_twins(rec, f, sigma, entries, step)

    def _emit_twins(self, rec, f, sigma, entries, step) -> None:
        p = self.p
        cc_edges = list(rec.inputs) + ([rec.product] if rec.product is not None else []) \
            + list(rec.wastes)
        for cc in cc_edges:
            if np.isnan(f[cc]):
                continue
            twin = self._cc_to_mass.get(cc)
            if twin is None:
                continue  # WR (conc=0) has no conc twin — its Mass comes from the balance below
            m, conc, conc_sig = twin
            if np.isnan(f[m]):
                f_m = float(f[cc] / conc)
                s_m = lp_combine([sigma[cc] / conc, abs(f_m) * conc_sig / conc], p)
                self._w(m, f_m, s_m, f, sigma, entries, step,
                        "network_mass_twin", [cc], [1.0 / conc], None)

        # ---- Mining: WR_Mass = CO_Mass − Ore_Mass (mass-balance residual) ----
        # Ore stays grade-anchored; WR is the clipped input−output residual that closes
        # the hard Mining Mass balance in the seed. _w clips to [0,u] and widens σ on the
        # clip, so positive-delta regions get WR=CO−Ore (seed residual→0) and negative-delta
        # (grade-implied Ore > crude_ore CO) get WR=0. WR is a free slack excluded from the
        # scale total (ADR 0006); the stripping ratio survives only as the soft constraint.
        # See ASSUMPTIONS.md. WR's conc=0 keeps it out of _cc_to_mass, so locate it
        # structurally (same from/to/time).
        if rec.special == "stripping_ratio" and rec.product is not None and rec.wastes:
            co_cc = rec.primary[0] if rec.primary else (rec.inputs[0] if rec.inputs else None)
            co_twin = self._cc_to_mass.get(co_cc) if co_cc is not None else None
            ore_twin = self._cc_to_mass.get(rec.product)
            wr_m = self._cc_to_mass_struct.get(rec.wastes[0])
            if co_twin and ore_twin and wr_m is not None:
                co_m, ore_m = co_twin[0], ore_twin[0]
                if np.isnan(f[wr_m]) and not np.isnan(f[co_m]) and not np.isnan(f[ore_m]):
                    f_wr = float(f[co_m] - f[ore_m])
                    s_wr = lp_combine([sigma[co_m], sigma[ore_m]], p)
                    self._w(wr_m, f_wr, s_wr, f, sigma, entries, step,
                            "network_balance", [co_m, ore_m], [1.0, -1.0], None)

    def _do_flow_balance(self, inc, out, f, sigma, entries, step) -> None:
        """Flow-node CC balance: Σ(supply in) = Σ(uses out); fill a lone gap."""
        p = self.p
        edges = list(inc) + list(out)
        missing = [e for e in edges if np.isnan(f[e]) and self._same_region[e]]
        # only act if exactly one *writable* (same-region) edge is unknown
        if len(missing) != 1:
            return
        e = missing[0]
        # every other *same-region* edge must be known; missing cross-region
        # (non-writable) legs are treated as 0 (no trade on that route) — symmetric
        # with the recipe's bilateral_import tier (ADR 0010). They are counted as 0,
        # never written, so the uncovered edge stays an honest flat-prior gap.
        if any(np.isnan(f[x]) for x in edges if x != e and self._same_region[x]):
            return
        in_sum = float(sum(f[x] for x in inc if x != e and not np.isnan(f[x])))
        out_sum = float(sum(f[x] for x in out if x != e and not np.isnan(f[x])))
        f_e = (in_sum - out_sum) if e in out else (out_sum - in_sum)
        others = [x for x in edges if x != e and not np.isnan(f[x])]
        s_e = lp_combine([sigma[x] for x in others], p)
        self._w(e, f_e, s_e, f, sigma, entries, step,
                "network_balance", others, [1.0] * len(others), None)
        # emit the Mass twin for the freshly-balanced CC edge
        twin = self._cc_to_mass.get(e)
        if twin is not None and not np.isnan(f[e]):
            m, conc, conc_sig = twin
            if np.isnan(f[m]):
                f_m = float(f[e] / conc)
                s_m = lp_combine([sigma[e] / conc, abs(f_m) * conc_sig / conc], p)
                self._w(m, f_m, s_m, f, sigma, entries, step,
                        "network_mass_twin", [e], [1.0 / conc], None)

    # -- write helper -------------------------------------------------------

    def _w(self, e, val, sig, f, sigma, entries, step, method,
           sources, coeffs, param_id, desc="") -> bool:
        """Write a fill, guarded by ownership (same-region) + no-overwrite.

        The value is clipped to the edge's ``[lower, upper]`` bounds *before* it is
        written, so a negative residual (e.g. ``Ore→SXEW = Mining→Ore − Ore→MF`` in
        a no-SXEW region) cannot propagate through the internal fixpoint and drag
        downstream sums below their true values. A clipped fill widens σ to the clip
        gap (a saturated estimate, not a precise bound) and is tagged ``_clipped``.
        """
        if not self._same_region[e] or not np.isnan(f[e]):
            return False
        val = float(val)
        sig = float(sig)
        clipped_val = float(min(max(val, self._lower[e]), self._upper[e]))
        was_clipped = clipped_val != val
        if was_clipped:
            sig = lp_combine([sig, abs(val - clipped_val)], self.p)  # widen σ to the clip gap
            val = clipped_val
            self._n_clipped += 1
        sig, capped = _apply_cv_cap(sig, val, self.cv_cap)  # cap self-disables at val==0
        f[e] = val
        sigma[e] = float(sig)
        entries.append(_log(
            int(e), step,
            method + ("_clipped" if was_clipped else "") + ("_capped" if capped else ""),
            [int(s) for s in sources], [float(c) for c in coeffs],
            val, float(sig), desc, param_id,
        ))
        return True
