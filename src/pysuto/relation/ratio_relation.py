"""
ratio_relation.py
=================
Ratio relations — ``Σ numerator − r · Σ denominator = 0`` — as one object for both
stages: the pooled efficiency of a process recipe and the concentration twin of a CC edge
are built here, so ``RatioConstraintBuilder`` (Stage 3, emits them into ``G``) and the
NetworkImputer (Stage 2, solves them for a lone unknown) read the same coefficients.

Also the ``agg=true`` bucketing of a params row (the declaration of pooling, ADR 0005):
which edges a pooled row matches per ``(region_to, time, valuation)``.

Public API
----------
match_side_edges(row, prefix, acc)          Edge ids matching one side's selectors
agg_buckets(row, acc, scope)                Both sides of an agg row, per bucket
ratio_relation(b_edges, a_edges, r, ...)    The ratio relation (numerator first)
conc_twin_relation(cc, mass, conc, ...)     CC − conc·Mass = 0, a CC edge's Mass twin
pooled_input_sets(row, acc, valuation)      ``{(region, time): input edge set}`` of an agg row
recipe_relation(rec, eff, ...)              A process recipe's pooled-efficiency relation
"""

from __future__ import annotations

import numpy as np

from .param_relation import get_edge_cache
from .params import A_FIELDS, B_FIELDS, WILDCARD
from .relation import TYPE_RATIO, Relation
from .structure import ProcessRecipe


def match_side_edges(row, prefix: str, acc_struct) -> np.ndarray:
    """Return edge_ids matching the ``{a_|b_}``-prefixed selectors on `row` (a ParamRow).

    Wildcards / "same" are treated as 'match any value' — `agg` rows do not
    pair sides per-edge, so the cross-side "same" constraint has no meaning
    here. Time and valuation are returned alongside the edge ids by callers
    via ``acc_struct.edge_index`` for bucketing.
    """
    ei_exp, _el, _itt, _fv, _mc, edge_ids_arr = get_edge_cache(acc_struct)
    fields = A_FIELDS if prefix == "a" else B_FIELDS
    mask = np.ones(len(ei_exp), dtype=bool)
    for col_key, ei_col, _ in fields:
        val = row.selectors[col_key]
        if val in WILDCARD or val == "same":
            continue
        mask &= (ei_exp[ei_col].astype(str).to_numpy() == val)
    return edge_ids_arr[mask]


def agg_buckets(row, acc_struct, scope):
    """Match a/b selectors and bucket both sides by ``(region_to, time, valuation)``.

    Returns ``(buckets, a_size, b_size)`` where each bucket is
    ``{"region", "time", "valuation", "a_edges": [...], "b_edges": [...]}`` for
    every key present on *both* sides. Bucketing on ``region_to`` scopes the
    pool per process region: input use-edges carry ``region_to = process region``
    and the product supply-edge carries ``region_to = product region`` (the
    same-region recipe makes the two equal), so cross-region pooling cannot leak.
    For stock-bounds rows ``region_to`` is pinned by the selector, so adding it to
    the key leaves their bucketing unchanged.
    """
    ei_exp = get_edge_cache(acc_struct)[0]
    r_all = ei_exp["region_to"].astype(str).to_numpy()
    t_all = ei_exp["time"].astype(str).to_numpy()
    v_all = ei_exp["valuation"].astype(str).to_numpy()
    a_edges = match_side_edges(row, "a", acc_struct)
    b_edges = match_side_edges(row, "b", acc_struct)
    if scope is not None:
        a_edges = a_edges[scope[a_edges]]
        b_edges = b_edges[scope[b_edges]]
    asz, bsz = int(a_edges.size), int(b_edges.size)
    if asz == 0 or bsz == 0:
        return [], asz, bsz
    ra, ta, va = r_all[a_edges], t_all[a_edges], v_all[a_edges]
    rb, tb, vb = r_all[b_edges], t_all[b_edges], v_all[b_edges]
    keys = sorted(set(zip(ra, ta, va)) & set(zip(rb, tb, vb)))
    buckets = []
    for r, t, v in keys:
        a_sub = a_edges[(ra == r) & (ta == t) & (va == v)]
        b_sub = b_edges[(rb == r) & (tb == t) & (vb == v)]
        buckets.append({"region": r, "time": t, "valuation": v,
                        "a_edges": a_sub.tolist(), "b_edges": b_sub.tolist()})
    return buckets, asz, bsz


def ratio_relation(b_edges, a_edges, r: float, *, variant: str, specs: tuple,
                   time: str, valuation: str, hard: bool = False, sigma: float = 0.0,
                   scale_mode: str = "symmetric", meta: dict | None = None,
                   agg: bool = False, source: str = "") -> Relation:
    """``Σ b − r · Σ a = 0``: the ``b`` edges (numerator) at ``+1`` first, the ``a`` edges
    (denominator) at ``−r``; ``n_num = len(b_edges)``. The one place the coefficient
    layout of a ratio row is written (``sigma`` is the relative σ_r, see
    :class:`~pysuto.relation.relation.Relation`)."""
    n_b, n_a = len(b_edges), len(a_edges)
    return Relation(
        edges        = np.array(list(b_edges) + list(a_edges), dtype=np.int64),
        coefficients = np.array([1.0] * n_b + [-r] * n_a),
        type         = TYPE_RATIO,
        variant      = variant,
        specs        = specs,
        time         = time,
        valuation    = valuation,
        hard         = hard,
        sigma        = sigma,
        scale_mode   = scale_mode,
        n_num        = n_b,
        meta         = {} if meta is None else meta,
        agg          = agg,
        source       = source,
    )


def conc_twin_relation(cc_edge: int, mass_edge: int, conc: float, conc_sigma: float = 0.0,
                       *, time: str, valuation: str) -> Relation:
    """The concentration twin of a CC edge: ``CC − conc · Mass = 0``.

    The relation Stage 3 emits for a conc-family params row (A = Mass, B = CC), so the
    CC edge is the numerator. ``conc_sigma`` is the row's σ in ratio units; the relation
    carries the relative σ_r = σ/|conc| of every ratio relation.
    """
    return ratio_relation(
        [cc_edge], [mass_edge], conc,
        variant = "conc", specs = (), time = str(time), valuation = valuation,
        sigma   = conc_sigma / abs(conc) if conc else 0.0,
    )


def pooled_input_sets(row, acc_struct, valuation: str) -> dict:
    """``{(region, time): frozenset of input edge ids}`` of an ``agg=true`` row.

    The a-side (input) edges the row pools in each bucket of ``valuation`` — the set a
    process recipe's inputs must equal for the row to declare that recipe's pooling
    (ER pools its anode input over all source regions: ADR 0022).
    """
    out = {}
    for bk in agg_buckets(row, acc_struct, None)[0]:
        if bk["valuation"] == valuation:
            out[(bk["region"], bk["time"])] = frozenset(int(e) for e in bk["a_edges"])
    return out


def recipe_relation(rec: ProcessRecipe, eff: float, *, valuation: str,
                    variant: str = "ratioeq", sigma: float = 0.0, hard: bool = False,
                    pooled: dict | None = None, row_id: str = "") -> Relation:
    """A process recipe's pooled-efficiency relation ``Σ products − eff · Σ inputs = 0``.

    The recipe supplies the structure (its product and input edge ids); ``eff`` is the
    row's nominal, ``1.0`` and ``hard`` for a process no row declares (it closes through
    its hard balance). ``pooled`` is :func:`pooled_input_sets` of the declaring ``agg``
    row: the row declares pooling (ADR 0005), so when the input set it pools in this
    recipe's region and period differs from the recipe's, this raises ``ValueError`` —
    the structure never silently overrules the declaration.
    """
    if pooled is not None:
        theirs = pooled.get((rec.region, rec.time), frozenset())
        if theirs != frozenset(rec.inputs):
            raise ValueError(
                f"process {rec.process!r} ({rec.region}, {rec.time}): agg row "
                f"{row_id!r} pools input edges {sorted(theirs)} but the recipe's inputs "
                f"are {sorted(rec.inputs)}; the agg=true row declares the pooling, so "
                "its input set must equal the recipe's"
            )
    return ratio_relation(
        rec.products, rec.inputs, eff,
        variant   = variant,
        specs     = (rec.process, rec.region, valuation),
        time      = rec.time,
        valuation = valuation,
        hard      = hard,
        sigma     = sigma,
        meta      = {"region": rec.region},
        agg       = pooled is not None,
        source    = row_id,
    )
