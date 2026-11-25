"""reexport.py
============
Re-attribute **re-exported** bilateral trade to its origin and destination at
feed time (ADR 0027): the volumes, the waterfill netting, the σ multiplier and the
rewrite of the raw rows, as library functions with the case's labels as parameters.

The rule is in ``scripts/build_baci_reexport.py``'s docstring and ADR 0027: four observed
numbers per (product s, region h, year t), all in contained copper (CC),

    I = Σ_A BACI(A→h)    imports      X = Σ_Z BACI(h→Z)    exports
    U = ICSG intake      P = ICSG production
    R = max(0, I − U)    E = max(0, X − P)    T = min(X, I, max(R, E))

and ``T ≤ X``, ``T ≤ I`` make it safe by construction: producers are never netted.
Reading the registered ICSG input (``U``, ``P``) is the case's business,
``scripts/copper_feed.py``; the script is the CLI around these functions.

Public API
----------
screened_origins, reexport_volumes   The origins left alone and the per-hub ``T``
icsg_aggregate_members               Region-years ICSG reports inside an aggregate row
waterfill, net_one_hub, net_reexports  The netting
cv_scale                             The σ multiplier of the inference
label_row                            The ledger's population label
check_region_map, check_icsg_baci_version  Refusals of a stale source (#77, #63)
load_root_trade, rebuild_values      Raw BACI rows <-> root legs in CC
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from pysuto.relation.param_relation import conc_for

# ---------------------------------------------------------------------------
# Knobs of the netting: a change is a measured change (ADR 0027)
# ---------------------------------------------------------------------------

# Mirrors the copper case's Stage 2 `OUTLIER_LEG_K` (reference = production +
# imports). Kept as its own constant: this is a Stage-1 restatement, and the
# two are free to diverge if the Stage-2 screen is retuned.
OUTLIER_K = 3.0

# kt CC. Below this a hub-year is not worth netting (and `FeedCollector`'s
# min_flow_kt = 0.01 would drop the resulting legs from coverage anyway).
MIN_T_CC = 0.01

# kt CC. A pair with no prior direct trade is created only at or above this.
MIN_NEW_LEG_CC = 1.0

# σ compounding: the adjustment derives from two observed legs.
K_INFER = float(np.sqrt(2.0))

# Ceiling on that compounding. The pedigree cv of a BACI leg is ~0.209, and
# ADR 0017 caps a damped leg's cv at 1.0, so a multiplier beyond ~4.8 is
# already inert — while an *uncapped* ratio explodes on a leg the netting drove
# to a rounding remainder (v_new → 0 ⇒ scale → ∞, measured at 1e48 before this
# cap). A leg that far gone is "no trade on this route" (ADR 0010), not a
# precisely-known tiny flow.
MAX_CV_SCALE = 5.0

# kt CC. Below this a leg is not a flow: `FeedCollector(min_flow_kt=0.01)`
# drops it from coverage, so carrying a σ for it is meaningless. Netting
# remainders under this floor are zeroed rather than left as dust.
LEG_ZERO_FLOOR_CC = 1.0e-4

# Hub→hub chains (Chile→NL→Germany) need the first hub netted before the
# second sees its inflow, so the sweep repeats until nothing moves.
MAX_PASSES = 3


# ---------------------------------------------------------------------------
# Pure computation (unit-tested)
# ---------------------------------------------------------------------------

def screened_origins(trade: pd.DataFrame, production: pd.DataFrame,
                     k: float = OUTLIER_K) -> set:
    """``(sector, region, time)`` whose exports no production+imports explains.

    The Stage-2 plausibility screen (``_screen_outlier_trade_legs``) inflates
    σ on exactly this population; at Stage 1 the consequence is stronger — such
    an origin's tonnage must not be *moved*, or a hub re-attributes a source
    defect onto plausible partners. Groups with no production stay unjudged,
    exactly as the Stage-2 screen leaves them (`outlier_ref` = production +
    imports, `prod_cc > 0` required).
    """
    exp = trade.groupby(["sector", "region_from", "time"])["cc"].sum()
    imp = trade.groupby(["sector", "region_to", "time"])["cc"].sum()
    out = set()
    for (s, a, t), x in exp.items():
        p = production.get((s, a, t), np.nan)
        if np.isnan(p) or p <= 0.0:
            continue
        ref = p + float(imp.get((s, a, t), 0.0))
        if ref > 0.0 and x > k * ref:
            out.add((s, a, t))
    return out


def reexport_volumes(trade: pd.DataFrame, intake: dict, production: dict,
                     reported: set, aggregate_of: dict,
                     hub_exclude: set) -> pd.DataFrame:
    """Per (sector, region, year) the four observed numbers and ``T``.

    ``trade``       cross-region root legs: sector, region_from, region_to, time, cc
    ``intake``      {(sector, region, time): U}   — ICSG, may be absent
    ``production``  {(sector, region, time): P}   — ICSG, may be absent
    ``reported``    {(sector, region)} ICSG reports at all, in any year
    ``aggregate_of``{(region, time): ICSG aggregate row} — region-years whose
                    flows ICSG reports inside that row (``icsg_aggregate_members``)
    """
    imp = trade.groupby(["sector", "region_to", "time"])["cc"].sum()
    exp = trade.groupby(["sector", "region_from", "time"])["cc"].sum()
    keys = sorted(set(imp.index) | set(exp.index))

    rows = []
    for s, h, t in keys:
        i_h = float(imp.get((s, h, t), 0.0))
        x_h = float(exp.get((s, h, t), 0.0))
        u   = intake.get((s, h, t), np.nan)
        p   = production.get((s, h, t), np.nan)
        excluded = h in hub_exclude
        # A region that reports the product elsewhere but not this year has a
        # reporting gap, not a zero: netting it would delete real exports.  So
        # has a region-year whose flows sit inside an ICSG aggregate row, even
        # where its own stock rows give it a U (#67).
        aggregate = aggregate_of.get((h, t))
        gap = aggregate is not None or (np.isnan(u) and np.isnan(p)
                                        and (s, h) in reported)
        u_f = 0.0 if np.isnan(u) else max(u, 0.0)
        p_f = 0.0 if np.isnan(p) else max(p, 0.0)
        r   = max(0.0, i_h - u_f)
        e   = max(0.0, x_h - p_f)
        t_v = 0.0 if (excluded or gap or i_h <= 0.0 or x_h <= 0.0) \
            else min(x_h, i_h, max(r, e))
        rows.append(dict(
            sector=s, region=h, time=t, I=i_h, X=x_h, U=u, P=p, R=r, E=e, T=t_v,
            T_surplus=min(x_h, i_h, r), T_ceiling=min(x_h, i_h, e),
            resid=r - t_v, dom_exports=x_h - t_v,
            excluded_aggregate=excluded, icsg_gap=gap, icsg_aggregate=aggregate,
            U_missing=bool(np.isnan(u)), P_missing=bool(np.isnan(p)),
            imports_exhausted=bool(t_v > 0 and np.isclose(t_v, i_h) and u_f > p_f),
        ))
    return pd.DataFrame(rows)


def icsg_aggregate_members(path, label: str) -> dict:
    """``{(root_region, year): aggregate}`` from the hand-authored ``icsg_aggregates`` input.

    A row declares that ICSG reports ``root_region``'s flows inside its aggregate
    row ``src_region`` in every year from ``year_from`` to ``year_to`` inclusive.
    Declared, never inferred from region names (#67). Years are strings. A
    reversed range, or a region-year claimed by two aggregates, raises.
    """
    df = pd.read_csv(path, sep=None, engine="python", dtype=str)
    out: dict = {}
    for r in df.itertuples(index=False):
        y0, y1 = int(r.year_from), int(r.year_to)
        if y0 > y1:
            raise ValueError(f"{label}: {r.src_region} ⊃ {r.root_region} has "
                             f"year_from {y0} > year_to {y1}")
        for y in range(y0, y1 + 1):
            key = (r.root_region, str(y))
            if key in out:
                raise ValueError(f"{label}: {r.root_region} {y} is claimed by "
                                 f"both {out[key]} and {r.src_region}")
            out[key] = r.src_region
    return out


def waterfill(want: dict, cap: dict) -> dict:
    """Allocate ``Σ want`` across keys without exceeding ``cap`` per key.

    ``Σ want ≤ Σ cap`` is guaranteed by the caller (the total is ``T``, capped
    at the hub's own imports and exports), so a feasible allocation exists.
    Keys that hit their cap are frozen and their excess is redistributed over
    the rest in proportion to what they still wanted.
    """
    alloc = dict(want)
    for _ in range(len(want) + 1):
        over = {k: v - cap[k] for k, v in alloc.items() if v > cap[k] + 1e-12}
        if not over:
            break
        excess = sum(over.values())
        for k in over:
            alloc[k] = cap[k]
        room = {k: cap[k] - alloc[k] for k in alloc if k not in over}
        head = sum(alloc[k] for k in room)
        if head <= 0:
            base = {k: room[k] for k in room}
            head = sum(base.values())
            if head <= 0:
                break
            for k in room:
                alloc[k] += excess * base[k] / head
        else:
            for k in room:
                alloc[k] += excess * alloc[k] / head
    return alloc


def net_one_hub(mat: dict, h: str, t_want: float, screened_from: set,
                min_new_leg: float = MIN_NEW_LEG_CC) -> tuple:
    """Re-attribute ``t_want`` through hub ``h`` in one (sector, year) matrix.

    ``mat`` maps ``(origin, destination) → cc`` and is mutated in place.
    Returns ``(t_applied, self_dropped, n_new_legs)``.
    """
    ins  = {a: c for (a, z), c in mat.items()
            if z == h and c > 0.0 and a not in screened_from}
    outs = {z: c for (a, z), c in mat.items() if a == h and c > 0.0}
    if not ins or not outs:
        return 0.0, 0.0, 0

    i_ns = sum(ins.values())
    x_h  = sum(outs.values())
    t_v  = min(t_want, i_ns, x_h)
    if t_v <= 0.0:
        return 0.0, 0.0, 0

    a_share = {a: c / i_ns for a, c in ins.items()}
    b_share = {z: c / x_h for z, c in outs.items()}

    placed:  dict = defaultdict(float)  # (origin, destination) → cc to add
    moved:   dict = {}                  # origin → cc to remove from origin→h
    returned: dict = {}                 # origin → cc that came back to it
    n_new = 0

    for a, a_s in a_share.items():
        from_a   = t_v * a_s
        self_amt = from_a * b_share.get(a, 0.0)     # A → h → A: a returned good
        others = {z: from_a * b for z, b in b_share.items() if z != a}
        kept   = {z: v for z, v in others.items()
                  if (a, z) in mat or v >= min_new_leg}
        if kept:
            # Sub-threshold pairs would be sub-kilotonne inferred legs; fold
            # them into this origin's kept destinations instead of creating them.
            crumbs = sum(v for z, v in others.items() if z not in kept)
            scale  = (sum(kept.values()) + crumbs) / sum(kept.values())
            for z, v in kept.items():
                if (a, z) not in mat:
                    n_new += 1
                placed[(a, z)] += v * scale
            moved[a]    = from_a
            returned[a] = self_amt
        elif self_amt > 0.0:
            # Nothing of this origin's tonnage can be placed, but the part that
            # came straight back to it is still a re-import: drop just that.
            moved[a]    = self_amt
            returned[a] = self_amt
        # else: this origin does not participate at all.

    if not moved:
        return 0.0, 0.0, 0

    # Origin side: a leg can never give back more than it carries.
    moved = waterfill(moved, {a: ins[a] for a in moved})
    t_applied = sum(moved.values())

    # Destination side: the same total, capped by each h→Z leg. What each
    # destination gives up is what the origins place on it, plus any tonnage
    # that returned to an origin (which is itself a destination of h).
    want_z: dict = defaultdict(float)
    for (a, z), v in placed.items():
        want_z[z] += v
    for a, s_amt in returned.items():
        if s_amt > 0.0:
            want_z[a] += s_amt
    tot_want = sum(want_z.values())
    if tot_want <= 0.0:
        return 0.0, 0.0, 0
    corr   = t_applied / tot_want               # 1.0 unless a cap bound above
    red_z  = waterfill({z: v * corr for z, v in want_z.items()},
                        {z: outs[z] for z in want_z})

    # Apply.
    for a, v in moved.items():
        mat[(a, h)] = max(0.0, mat[(a, h)] - v)
    for z, v in red_z.items():
        mat[(h, z)] = max(0.0, mat[(h, z)] - v)
    for (a, z), v in placed.items():
        mat[(a, z)] = mat.get((a, z), 0.0) + v * corr

    return t_applied, sum(returned.values()) * corr, n_new


def net_reexports(trade: pd.DataFrame, volumes: pd.DataFrame, screened: set,
                  min_new_leg: float = MIN_NEW_LEG_CC,
                  max_passes: int = MAX_PASSES) -> tuple:
    """Apply the re-attribution to every (sector, year) matrix.

    Hubs are processed largest-``T``-first within a (sector, year) because
    hub→hub legs exist: the first hub must be netted before the second sees its
    inflow. The sweep repeats until nothing moves (idempotence).

    Returns ``(trade_new, events)``.
    """
    mats: dict = defaultdict(dict)
    for s, a, z, t, cc in trade[["sector", "region_from", "region_to",
                                 "time", "cc"]].itertuples(index=False):
        mats[(s, t)][(a, z)] = mats[(s, t)].get((a, z), 0.0) + float(cc)

    want = {(r.sector, r.region, r.time): r.T
            for r in volumes.itertuples(index=False) if r.T > MIN_T_CC}

    events = []
    for (s, t), mat in mats.items():
        scr  = {a for (ss, a, tt) in screened if ss == s and tt == t}
        hubs = {h: v for (ss, h, tt), v in want.items() if ss == s and tt == t}
        done: dict = defaultdict(float)     # hub → T already applied
        for p in range(max_passes):
            # A hub is worth (re)visiting for whatever of its T it has not yet
            # moved, capped by the legs it *now* carries: a later pass exists
            # only because an upstream hub's re-attribution replenished this
            # one's inflow (the hub→hub chain), never to re-apply its T twice.
            todo = sorted(
                ((min(v - done[h], _flow(mat, h)), h) for h, v in hubs.items()
                 if v - done[h] > MIN_T_CC),
                reverse=True,
            )
            applied_pass = 0.0
            for t_want, h in todo:
                if t_want <= MIN_T_CC:
                    continue
                t_app, self_amt, n_new = net_one_hub(
                    mat, h, t_want, scr, min_new_leg)
                if t_app <= 0.0:
                    continue
                done[h] += t_app
                applied_pass += t_app
                events.append(dict(sector=s, region=h, time=t, pass_no=p,
                                   T_applied=t_app, self_dropped=self_amt,
                                   n_new_legs=n_new))
            if applied_pass <= MIN_T_CC:
                break

    rows = []
    for (s, t), mat in mats.items():
        for (a, z), cc in mat.items():
            # Float remainders of a fully-netted leg are not a flow.
            rows.append((s, a, z, t,
                         0.0 if cc < LEG_ZERO_FLOOR_CC else cc))
    trade_new = pd.DataFrame(
        rows, columns=["sector", "region_from", "region_to", "time", "cc"])
    return trade_new, pd.DataFrame(events)


def _flow(mat: dict, h: str) -> float:
    """``min(imports, exports)`` of ``h`` in the matrix as it now stands."""
    i_h = sum(c for (a, z), c in mat.items() if z == h)
    x_h = sum(c for (a, z), c in mat.items() if a == h)
    return min(i_h, x_h)


def cv_scale(trade_old: pd.DataFrame, trade_new: pd.DataFrame,
             k: float = K_INFER, max_scale: float = MAX_CV_SCALE) -> pd.DataFrame:
    """Per changed root leg, the σ multiplier of the ADR 0027 inference.

    ``σ_new = cv_pedigree · √(v_obs² + (k·Δ)²)`` ⇒
    ``cv_root_scale = √(v_obs² + (k·Δ)²) / |v_new|``, **floored at 1** so a
    netting step can never make a leg more certain than its pedigree says it is,
    and **capped at ``max_scale``** because past that the leg's cv has already
    reached ADR 0017's ceiling of 1.0 and further inflation buys nothing.

    Legs the netting drove below ``LEG_ZERO_FLOOR_CC`` are excluded rather than
    carried at a vast σ: they are not flows, and the collector's ``min_flow_kt``
    would drop them from coverage anyway.
    """
    key = ["sector", "region_from", "region_to", "time"]
    m = trade_old.merge(trade_new, on=key, how="outer",
                        suffixes=("_obs", "_new")).fillna({"cc_obs": 0.0, "cc_new": 0.0})
    m["delta"] = m["cc_new"] - m["cc_obs"]
    m = m[(m["delta"].abs() > 1e-9) & (m["cc_new"] >= LEG_ZERO_FLOOR_CC)].copy()
    m["cv_scale"] = np.clip(
        np.sqrt(m["cc_obs"] ** 2 + (k * m["delta"]) ** 2) / m["cc_new"],
        1.0, max_scale)
    return m[key + ["cc_obs", "cc_new", "delta", "cv_scale"]]


def label_row(r) -> str:
    """The ledger's population label (reporting only — never a gate)."""
    if r.excluded_aggregate:
        return "excluded-aggregate"
    if r.icsg_gap:
        return "icsg-gap"
    if r.T <= MIN_T_CC:
        if r.R > 0.0 and r.I > 0.0:
            return "inbound-only/no-process" if (r.U_missing or r.U <= 0.0) \
                and (r.P_missing or r.P <= 0.0) else "inbound-only"
        return "producer" if (not r.P_missing and r.P > 0.0) else "untouched"
    if r.T >= 0.9 * r.X:
        return "pass-through"
    return "partial"


# ---------------------------------------------------------------------------
# Source checks and raw rows
# ---------------------------------------------------------------------------

def _region_map(d: str) -> dict:
    """BACI code → root region, from the region map of ``d``'s ``conc.xlsx``."""
    m = pd.read_excel(os.path.join(d, "conc.xlsx"), sheet_name="region_map")
    return dict(zip(m["src_region"].astype(int), m["root_region"]))


def check_region_map(src_dir: str, current_dir: str) -> None:
    """Raise unless the source version maps every BACI code as ``current/`` does.

    The netting maps codes through the source version's region map, but
    ``baci_feed.py`` maps the netted ``values.csv`` through ``current/``'s. After a
    region-map edit (#77: code 490 → Taiwan) the two differ until the BACI parse
    is re-run, which also re-keys the ADR 0033 sidecar, and its output registered:
    net that version, not an older one.
    """
    src, cur = _region_map(src_dir), _region_map(current_dir)
    diff = sorted(k for k in src.keys() | cur.keys() if src.get(k) != cur.get(k))
    if diff:
        raise ValueError(
            f"{len(diff)} BACI code(s) map differently in {src_dir} and {current_dir} "
            f"(code: source → current; first {min(len(diff), 5)}): "
            + ", ".join(f"{k}: {src.get(k)} → {cur.get(k)}" for k in diff[:5])
            + " — re-run the BACI parse, register it, and net that version")


def check_icsg_baci_version(icsg_dir: os.PathLike, source_version: int,
                           feed_id: str, parents_name: str = "parents.json") -> None:
    """Raise unless the ICSG input at *icsg_dir* was parsed against baci *source_version*.

    The ICSG parse nets imported anode off the direct-to-refinery route using a
    gross BACI version (#63, ADR 0030 am.), and the netting reads that ICSG input
    back through ``icsg_intake_production``. Both must read the same gross version,
    or the ICSG input carries another release's anode trade, silently.
    """
    path = os.path.join(icsg_dir, parents_name)
    recorded = None
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            recorded = json.load(f).get("inputs", {}).get("feed", {}).get(feed_id)
    if recorded != source_version:
        read = (f"was parsed against baci v{recorded}" if recorded is not None
                else f"records no baci version ({parents_name})")
        raise ValueError(
            f"the ICSG input at {icsg_dir} {read}, but this nets baci v{source_version} — "
            f"re-run python scripts/parse/icsg_parse/icsg_clean.py --baci-version "
            f"{source_version} --to-feed, then this script")



def load_root_trade(src_dir: str, conc_lookup: dict) -> tuple:
    """Raw BACI → cross-region root legs in CC, plus the raw frame and maps."""
    raw  = pd.read_csv(os.path.join(src_dir, "values.csv"))
    smap = pd.read_excel(os.path.join(src_dir, "conc.xlsx"), sheet_name="sector_map")
    s_of = dict(zip(smap["src_sector"].astype(int), smap["root_sector"]))
    r_of = _region_map(src_dir)

    raw["sector"]      = raw["Sector"].astype(int).map(s_of)
    raw["region_from"] = raw["Region_from"].astype(int).map(r_of)
    raw["region_to"]   = raw["Region_to"].astype(int).map(r_of)
    raw["time"]        = raw["Year"].astype(str)
    if raw[["sector", "region_from", "region_to"]].isna().any().any():
        raise SystemExit("unmapped BACI sector or region — check conc.xlsx")

    # Root self-trade is dropped by the feed's cross_region scope (ADR 0014);
    # it is neither trade nor a domestic observation, so it takes no part in
    # the netting — but the rows stay in values.csv untouched.
    cross = raw[raw["region_from"] != raw["region_to"]].copy()
    cross["cc"] = cross["Value"] * conc_for(conc_lookup, cross["sector"],
                                            cross["region_from"])
    legs = (cross.groupby(["sector", "region_from", "region_to", "time"])["cc"]
            .sum().reset_index())
    return raw, legs, s_of, r_of


def canonical_codes(raw: pd.DataFrame) -> tuple:
    """Per (root sector, root region), the heaviest src code on each side.

    A new raw row needs *a* code pair resolving to the root leg; the region map
    folds many codes onto one region, so the choice is cosmetic — but it must be
    deterministic, so it is the largest by traded mass, ties broken by code.
    """
    def pick(side_code, side_root):
        g = (raw.groupby(["sector", side_root, side_code])["Value"].sum()
             .reset_index()
             .sort_values(["sector", side_root, "Value", side_code],
                          ascending=[True, True, False, True]))
        g = g.drop_duplicates(["sector", side_root])
        return {(s, r): int(c) for s, r, c in
                zip(g["sector"], g[side_root], g[side_code])}
    return (pick("Region_from", "region_from"), pick("Region_to", "region_to"))


def rebuild_values(raw: pd.DataFrame, legs_old: pd.DataFrame,
                   legs_new: pd.DataFrame, conc_lookup: dict,
                   hs_of: dict) -> pd.DataFrame:
    """Apply the per-root-leg change to the raw rows.

    Existing rows are scaled by ``v_new / v_obs`` so the source's src-code
    detail survives; a root leg with no rows gets one new row at canonical
    codes. Scaling (rather than replacing) keeps HS 740110 and 740120 folding
    into Matte in their observed proportion.
    """
    key = ["sector", "region_from", "region_to", "time"]
    chg = legs_old.merge(legs_new, on=key, how="outer",
                         suffixes=("_obs", "_new")).fillna({"cc_obs": 0.0, "cc_new": 0.0})
    chg = chg[(chg["cc_new"] - chg["cc_obs"]).abs() > 1e-9]

    grew  = chg[chg["cc_obs"] > 0.0].copy()
    grew["factor"] = grew["cc_new"] / grew["cc_obs"]
    out = raw.merge(grew[key + ["factor"]], on=key, how="left")
    out["Value"] = np.where(out["factor"].notna(),
                            out["Value"] * out["factor"], out["Value"])

    born = chg[chg["cc_obs"] <= 0.0]
    if len(born):
        cf, ct = canonical_codes(raw)
        rows = []
        for r in born.itertuples(index=False):
            a = cf.get((r.sector, r.region_from))
            z = ct.get((r.sector, r.region_to))
            if a is None or z is None:      # region never seen on that side
                continue
            rows.append({"Year": int(r.time), "Sector": hs_of[r.sector],
                         "Region_from": a, "Region_to": z,
                         # a born leg's mass is its CC at the ORIGIN's grade —
                         # the grade the model will read it back with
                         "Value": r.cc_new / float(conc_for(
                             conc_lookup, [r.sector], [r.region_from])[0]),
                         "Unit": "kt",
                         "sector": r.sector, "region_from": r.region_from,
                         "region_to": r.region_to, "time": r.time})
        out = pd.concat([out, pd.DataFrame(rows)], ignore_index=True)

    cols = ["Year", "Sector", "Region_from", "Region_to", "Value", "Unit"]
    out = out[out["Value"] > 0.0]
    # Re-aggregate: a new row may land on a src pair the source already used.
    out = (out.groupby(["Year", "Sector", "Region_from", "Region_to", "Unit"],
                       as_index=False)["Value"].sum())
    return out[cols].sort_values(cols[:4]).reset_index(drop=True)

