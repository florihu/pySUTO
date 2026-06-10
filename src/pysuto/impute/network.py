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
    inputs: tuple[int, ...]     # all use-side inputs (for Mass-twin emission)
    primary: tuple[int, ...]    # Flow / SB-Nature inputs (block forward if missing)
    secondary: tuple[int, ...]  # SB-SEM inputs (Scrap): treated as 0 when missing
    product: int | None
    wastes: tuple[int, ...]
    eff: float
    eff_sig: float
    eff_eps: float
    eff_param_id: str | None
    special: str | None  # "wr_mass_residual" for Mining, else None


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
        wr_to_ore_ratio: float = 2.0,
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
        # WR_Mass = wr_to_ore_ratio · Ore_Mass. A direct ratio avoids the
        # CO_Mass − Ore_Mass residual, which goes negative when a region's
        # ore_grade exceeds the flat conc_ore (→ spurious negative WR mass).
        self.wr_to_ore_ratio = float(wr_to_ore_ratio)
        # populated by prepare()
        self._recipes: list[_RRecipe] = []
        self._flow_nodes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self._cc_to_mass: dict[int, tuple[int, float, float]] = {}
        self._stripping: dict[int, tuple[float, float]] = {}  # ore_mass_edge -> (ratio, σ)
        self._same_region: np.ndarray | None = None

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

        # CC edge -> (Mass twin edge, conc, σ_conc). conc first, ore_grade
        # overlaid (more specific for CO → wins on specificity).
        self._cc_to_mass = {}
        self._load_twin_map(self._conc_csv, resolver, self._cc_to_mass)
        if self._ore_grade_csv:
            self._load_twin_map(self._ore_grade_csv, resolver, self._cc_to_mass)

        # product CC edge -> (eff, σ_eff, ε, param_id)
        product_eff = self._load_eff_map(self._eff_csv, resolver)

        # Ore Mass edge -> (stripping ratio, σ); single source of truth for the
        # Mining WR_Mass fill (replaces the hard-coded wr_to_ore_ratio constant).
        self._stripping = self._load_stripping_map(self._eff_csv, resolver)

        # input edge -> from-node entity_sub (classifies primary vs SEM secondary)
        sub_from = dict(zip(em["edge_id"].to_numpy(), em["from_entity_sub"].to_numpy()))

        # attach eff to each structural recipe
        self._recipes = []
        for rec in acc.process_recipes:
            eff, eff_sig, eff_eps, pid = product_eff.get(
                rec.product, (1.0, 0.0, 0.0, None)
            )
            secondary = tuple(i for i in rec.inputs if sub_from.get(i) == "SEM")
            primary = tuple(i for i in rec.inputs if sub_from.get(i) != "SEM")
            special = "wr_to_ore" if rec.process_sector == "Mining" else None
            self._recipes.append(
                _RRecipe(rec.inputs, primary, secondary, rec.product, rec.wastes,
                         float(eff), float(eff_sig), float(eff_eps), pid, special)
            )

        # Flow-node CC balances: (incoming supply edges, outgoing use edges)
        self._flow_nodes = self._build_flow_balances(acc)

    def _load_twin_map(self, path, resolver, into: dict) -> None:
        """Resolve a conc-style CSV (A=Mass, B=CC) into CC->(Mass,conc,σ)."""
        df = pd.read_csv(path, sep=";", comment="#", dtype=str)
        df = df.dropna(subset=["param_id"])
        df = df[~df["param_id"].str.startswith("#")]
        best: dict[int, tuple] = {}  # cc_edge -> (mass_edge, conc, σ, specificity)
        for _, row in df.iterrows():
            try:
                nominal = float(row["value_nominal"])
                sig = sigma_from_row(row, nominal)
            except Exception as exc:  # noqa: BLE001
                logger.warning("twin map %s: row %s skipped (%s)",
                               path, row.get("param_id"), exc)
                continue
            for rr in resolver.resolve_row(row):
                cc = rr.b_edge_id
                if cc not in best or rr.specificity > best[cc][3]:
                    best[cc] = (rr.a_edge_id, nominal, sig, rr.specificity)
        for cc, (mass, conc, sig, _) in best.items():
            if conc > 0.0:
                into[cc] = (mass, conc, sig)

    def _load_eff_map(self, path, resolver) -> dict:
        df = pd.read_csv(path, sep=";", comment="#", dtype=str)
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

    def _load_stripping_map(self, path, resolver) -> dict:
        """Resolve stripping_ratio rows (A=Ore Mass, B=WR Mass) → {ore_mass_edge: (ratio, σ)}.

        Keyed by the Ore Mass edge (the A-side denominator) so :meth:`_emit_twins`
        can look the ratio up by a Mining recipe's Ore-Mass twin.
        """
        df = pd.read_csv(path, sep=";", comment="#", dtype=str)
        df = df.dropna(subset=["param_id"])
        df = df[~df["param_id"].str.startswith("#")]
        df = df[df["param_type"].astype(str).str.strip() == "stripping_ratio"]
        best: dict[int, tuple] = {}  # ore_mass_edge -> (ratio, σ, specificity)
        for _, row in df.iterrows():
            try:
                nominal = float(row["value_nominal"])
                sig = sigma_from_row(row, nominal)
            except Exception as exc:  # noqa: BLE001
                logger.warning("stripping map %s: row %s skipped (%s)",
                               path, row.get("param_id"), exc)
                continue
            for rr in resolver.resolve_row(row):
                ore = rr.a_edge_id  # A-side = Ore Mass edge (denominator)
                if ore not in best or rr.specificity > best[ore][2]:
                    best[ore] = (nominal, sig, rr.specificity)
        return {k: v[:2] for k, v in best.items()}

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
        # Internal fixpoint: keep sweeping recipes + flow balances until a full
        # sweep fills nothing new (the engine cascade re-runs us too, but
        # converging here keeps chain propagation within one call).
        for _ in range(64):
            n_before = len(entries)
            for rec in self._recipes:
                self._do_recipe(rec, f, sigma, entries, step)
            for inc, out in self._flow_nodes:
                self._do_flow_balance(inc, out, f, sigma, entries, step)
            if len(entries) == n_before:
                break
        return entries

    def _do_recipe(self, rec, f, sigma, entries, step) -> None:
        p = self.p
        P, W = rec.product, list(rec.wastes)
        PR, SE = list(rec.primary), list(rec.secondary)
        eff, eff_sig, eps = rec.eff, rec.eff_sig, rec.eff_eps

        miss_PR = [i for i in PR if np.isnan(f[i])]
        # missing SEM secondaries (Scrap) count as absent → 0
        sec_sum = float(sum(f[i] for i in SE if not np.isnan(f[i])))
        sec_sig = [sigma[i] for i in SE if not np.isnan(f[i])]
        known_PR = [i for i in PR if not np.isnan(f[i])]

        # ---- determine total primary input CC (S) and its σ ----
        S = None
        S_src = None
        sig_S = 0.0
        if PR and not miss_PR:
            S = float(sum(f[i] for i in PR)) + sec_sum
            S_src = "inputs"
            sig_S = lp_combine([sigma[i] for i in PR] + sec_sig, p)
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
        if P is not None and np.isnan(f[P]):
            f_P = eff * S
            s_P = lp_combine([eff * sig_S, eff_term, eps * abs(f_P)], p)
            self._w(P, f_P, s_P, f, sigma, entries, step,
                    "network_forward", PR if S_src == "inputs" else [P], [eff],
                    rec.eff_param_id)

        # ---- single missing PRIMARY input by subtraction (S from output) ----
        # (secondaries are pooled when known / 0 when missing; never written here)
        if S_src in ("product", "waste") and len(miss_PR) == 1:
            i = miss_PR[0]
            f_i = S - float(sum(f[j] for j in known_PR)) - sec_sum
            s_i = lp_combine(
                [sig_S] + [sigma[j] for j in known_PR] + sec_sig + [eps * abs(f_i)], p,
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
                    PR if S_src == "inputs" else [P], [1.0 - eff], rec.eff_param_id)

        # ---- Mass twins for this recipe's CC edges (atomic with the CC fills) ----
        self._emit_twins(rec, f, sigma, entries, step)

    def _emit_twins(self, rec, f, sigma, entries, step) -> None:
        p = self.p
        wr_cc = rec.wastes[0] if (rec.special == "wr_to_ore" and rec.wastes) else None
        cc_edges = list(rec.inputs) + ([rec.product] if rec.product is not None else []) \
            + list(rec.wastes)
        for cc in cc_edges:
            if cc == wr_cc:
                continue  # WR mass comes from the ore ratio below, not CC/conc
            if np.isnan(f[cc]):
                continue
            twin = self._cc_to_mass.get(cc)
            if twin is None:
                continue
            m, conc, conc_sig = twin
            if np.isnan(f[m]):
                f_m = float(f[cc] / conc)
                s_m = lp_combine([sigma[cc] / conc, abs(f_m) * conc_sig / conc], p)
                self._w(m, f_m, s_m, f, sigma, entries, step,
                        "network_mass_twin", [cc], [1.0 / conc], None)

        # ---- Mining: WR_Mass = wr_to_ore_ratio · Ore_Mass ----
        if wr_cc is not None and rec.product is not None:
            ore_twin = self._cc_to_mass.get(rec.product)
            wr_twin = self._cc_to_mass.get(wr_cc)
            if ore_twin and wr_twin:
                ore_m, wr_m = ore_twin[0], wr_twin[0]
                if not np.isnan(f[ore_m]) and np.isnan(f[wr_m]):
                    # stripping ratio from the eff CSV (param_type=stripping_ratio);
                    # fall back to the constructor default if no row resolves.
                    r, _sig_r = self._stripping.get(ore_m, (self.wr_to_ore_ratio, 0.0))
                    f_wr = float(r * f[ore_m])
                    s_wr = lp_combine([r * sigma[ore_m]], p)
                    self._w(wr_m, f_wr, s_wr, f, sigma, entries, step,
                            "network_wr_ratio", [ore_m], [r], None)

    def _do_flow_balance(self, inc, out, f, sigma, entries, step) -> None:
        """Flow-node CC balance: Σ(supply in) = Σ(uses out); fill a lone gap."""
        p = self.p
        edges = list(inc) + list(out)
        missing = [e for e in edges if np.isnan(f[e]) and self._same_region[e]]
        # only act if exactly one *writable* edge is unknown and everything else known
        if len(missing) != 1:
            return
        if any(np.isnan(f[e]) for e in edges if e != missing[0]):
            return
        e = missing[0]
        in_sum = float(sum(f[x] for x in inc if x != e))
        out_sum = float(sum(f[x] for x in out if x != e))
        f_e = (in_sum - out_sum) if e in out else (out_sum - in_sum)
        others = [x for x in edges if x != e]
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
        """Write a fill, guarded by ownership (same-region) + no-overwrite."""
        if not self._same_region[e] or not np.isnan(f[e]):
            return False
        sig, capped = _apply_cv_cap(float(sig), float(val), self.cv_cap)
        f[e] = float(val)
        sigma[e] = float(sig)
        entries.append(_log(
            int(e), step, method + ("_capped" if capped else ""),
            [int(s) for s in sources], [float(c) for c in coeffs],
            float(val), float(sig), desc, param_id,
        ))
        return True
