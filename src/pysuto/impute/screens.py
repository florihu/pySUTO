"""screens.py
==========
Stage 2 σ screens of the case's pre/post-imputation hooks (#121, #144): size-damped
bilateral-leg σ (ADR 0017), the outlier-leg plausibility screen and the
contradicted-node screen (ADR 0019).

The functions hold no case label: sector labels (``sectors``, the ``a_ref_override``
keys), thresholds (``gamma``, ``cv_cap``, ``k``, ``cv``) and parameter files are
arguments the case passes. The valuations are read as roles
(``acc.valuation_roles.gross`` / ``.content``, ADR 0034); the ``"Flow"`` and
``"Process"`` node entities are the accounting structure's own vocabulary, as in the
rest of ``pysuto.impute``. Each mutates ``agg_result.sigma_agg`` in place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pysuto.flow_node_balance import K_CLOSE as UNCLOSED_K, contradicted_flow_nodes
from pysuto.param_relation import (conc_for, conc_nominal_lookup, conc_sectors,
                                   expand_edge_index)


def damp_leg_sigma(acc, agg_result, gamma, cv_cap, logger, a_ref_override=None):
    """Size-damp σ on covered bilateral Flow-origin Mass legs (ADR 0017).

    σ_e = cv_e · a_ref(sector)^(1−γ) · a_e^γ with a_ref = the sector's median
    covered bilateral leg |f|, pooled over years; resulting cv capped at
    cv_cap. Runs BEFORE the outlier screen (its cv-5 inflation overrides).
    """
    if gamma is None or gamma >= 1.0:
        logger.info("B  leg-σ damping disabled (gamma=%s)", gamma)
        return
    ei = expand_edge_index(acc.edge_index, acc.node_index)
    cov = np.asarray(agg_result.coverage, dtype=bool)
    eid = ei["edge_id"].to_numpy()
    f = np.asarray(agg_result.f_agg, dtype=float)
    # entity_from Flow, any destination — Cathode's bilateral legs are
    # Flow→SB (final demand), matching the outlier screen's population.
    bilat = (cov[eid]
             & ei["entity_from"].eq("Flow").to_numpy()
             & (ei["region_from"] != ei["region_to"]).to_numpy()
             & ei["valuation"].eq(acc.valuation_roles.gross).to_numpy()
             & (np.abs(np.nan_to_num(f[eid])) > 0))
    legs = ei[bilat]
    n_damped = 0
    for sector, grp in legs.groupby("sector_from"):
        ids = grp["edge_id"].to_numpy()
        a = np.abs(f[ids])
        a_ref_run = float(np.median(a))
        # ADR 0027: across a measurement series the reference size is pinned to
        # the predecessor run's, so a change in the leg population cannot
        # re-price every leg in the sector under cover of the structural change.
        a_ref = float((a_ref_override or {}).get(sector, a_ref_run))
        if a_ref_override and sector in a_ref_override:
            logger.info("B  leg-σ damping %-12s a_ref PINNED %.4g kt "
                        "(this run would use %.4g)", sector, a_ref, a_ref_run)
        if a_ref <= 0:
            continue
        cv0 = agg_result.sigma_agg[ids] / a
        sigma = cv0 * a_ref ** (1.0 - gamma) * a ** gamma
        sigma = np.minimum(sigma, cv_cap * a)          # σ-cap on tiny legs
        agg_result.sigma_agg[ids] = sigma
        n_damped += len(ids)
        logger.info(
            "B  leg-σ damping %-12s a_ref=%.1f kt  n=%d  cv %0.3f–%0.3f "
            "(pedigree cv med %.3f)",
            sector, a_ref, len(ids), float(np.min(sigma / a)),
            float(np.max(sigma / a)), float(np.median(cv0)))
    logger.info("B  leg-σ damping: gamma=%.2f cv_cap=%.2f — %d legs damped",
                gamma, cv_cap, n_damped)


def screen_outlier_trade_legs(acc, agg_result, conc_csv: str, logger,
                               grade_csv: str | None = None, *, k: float, cv: float):
    """σ-inflate implausible bilateral Mass legs (``k`` and ``cv`` are
    the case's outlier-leg knobs).

    Per (flow sector, exporter region, year): implied CC exports =
    Σ covered bilateral Flow→* Mass legs · conc_nominal(sector); reference =
    Σ covered Process→Flow CC edges into that exporter's flow node (ICSG
    production) + implied CC imports (Σ covered bilateral Mass legs into the
    region · conc_nominal), so re-exporters are judged against what they
    actually receive. Groups without covered production stay unjudged (the
    transit-hub population; see the case's OUTLIER_LEG_K note).
    Ratio > k ⇒ every covered bilateral Mass leg
    of that group gets sigma_agg = cv·|f_agg|. Returns the report
    DataFrame (flagged groups only; empty frame if none).
    """
    ei = expand_edge_index(acc.edge_index, acc.node_index)
    cov = np.asarray(agg_result.coverage, dtype=bool)
    f = np.nan_to_num(np.asarray(agg_result.f_agg, dtype=float))

    conc_lookup = conc_nominal_lookup(*[c for c in (conc_csv, grade_csv) if c],
                                      gross=acc.valuation_roles.gross)
    sectors = conc_sectors(conc_lookup)

    eid = ei["edge_id"].to_numpy()
    bilat = (cov[eid]
             & ei["entity_from"].eq("Flow").to_numpy()
             & (ei["region_from"] != ei["region_to"]).to_numpy()
             & ei["valuation"].eq(acc.valuation_roles.gross).to_numpy()
             & ei["sector_from"].isin(sectors).to_numpy())
    prodm = (cov[eid]
             & ei["entity_from"].eq("Process").to_numpy()
             & ei["entity_to"].eq("Flow").to_numpy()
             & ei["valuation"].eq(acc.valuation_roles.content).to_numpy())

    legs = ei[bilat].copy()
    # priced with the EXPORTER's concentration — the grade belongs to the origin
    legs["cc"] = f[legs["edge_id"].to_numpy()] \
        * conc_for(conc_lookup, legs["sector_from"], legs["region_from"])
    exp_cc = legs.groupby(["sector_from", "region_from", "time"])["cc"].sum()
    imp_cc = legs.groupby(["sector_from", "region_to", "time"])["cc"].sum()
    imp_cc.index.names = exp_cc.index.names

    prod = ei[prodm].copy()
    prod["cc"] = f[prod["edge_id"].to_numpy()]
    prod_cc = prod.groupby(["sector_to", "region_to", "time"])["cc"].sum()
    prod_cc.index.names = exp_cc.index.names

    both = pd.concat([exp_cc.rename("exports_cc"), prod_cc.rename("prod_cc"),
                      imp_cc.rename("import_cc")], axis=1).fillna(0.0)
    both["ref_cc"] = both["prod_cc"] + both["import_cc"]
    both = both[(both["prod_cc"] > 0) & (both["exports_cc"] > 0)]
    both["ratio"] = both["exports_cc"] / both["ref_cc"]
    flagged = both[both["ratio"] > k].reset_index()

    n_edges = 0
    for _, g in flagged.iterrows():
        m = bilat \
            & ei["sector_from"].eq(g["sector_from"]).to_numpy() \
            & ei["region_from"].eq(g["region_from"]).to_numpy() \
            & ei["time"].eq(g["time"]).to_numpy()
        ids = ei["edge_id"].to_numpy()[m]
        ids = ids[np.abs(f[ids]) > 0]      # a genuine 0 keeps its σ (no 0-pin)
        agg_result.sigma_agg[ids] = cv * np.abs(agg_result.f_agg[ids])
        n_edges += len(ids)

    if len(flagged):
        logger.warning(
            "B  outlier-leg screen: %d exporter-sector-years flagged "
            "(implied CC exports > %.1f× covered production+imports) — σ "
            "inflated to cv=%.1f on %d bilateral Mass legs\n%s",
            len(flagged), k, cv, n_edges,
            flagged.sort_values("ratio", ascending=False).head(10).to_string(index=False))
    else:
        logger.info("B  outlier-leg screen: no exporter-sector-year exceeds "
                    "%.1f× covered production+imports — nothing inflated",
                    k)
    return flagged


def screen_contradicted_nodes(acc, agg_result, logger, *, sectors: tuple, cv):
    """σ-inflate the outflow edges of contradicted Flow nodes (ADR 0019) of
    ``sectors`` to ``cv``·|f| (the case's unclosed-sector labels and σ cv; ``cv=None``
    detects only).

    Detection lives in ``pysuto.flow_node_balance`` so Stage 3 applies the
    identical rule to the scale-total anchors. Returns the report DataFrame
    (failing nodes only; empty frame if none).

    Runs **after** imputation: the inflow side of a Flow node (Smelting→Matte)
    is never fed, so pre-cascade its prior is 0 and no node looks contradicted.
    """
    keys, report = contradicted_flow_nodes(
        acc, agg_result, sectors=sectors)
    if not keys:
        logger.info("B  contradicted-node screen: every %s node's outflow "
                    "prior honours its inflow — nothing inflated",
                    "/".join(sectors))
        return report

    ei = expand_edge_index(acc.edge_index, acc.node_index)
    cov = np.asarray(agg_result.coverage, dtype=bool)
    f = np.nan_to_num(np.asarray(agg_result.f_agg, dtype=float))
    eid = ei["edge_id"].to_numpy()
    outflow = cov[eid] & ei["entity_from"].eq("Flow").to_numpy()
    node_key = list(zip(ei["sector_from"], ei["region_from"], ei["time"]))
    hit = outflow & np.array([k in keys for k in node_key])

    ids = eid[hit]
    ids = ids[np.abs(f[ids]) > 0]          # a genuine 0 keeps its σ (no 0-pin)
    if cv is not None:
        agg_result.sigma_agg[ids] = cv * np.abs(agg_result.f_agg[ids])
    logger.warning(
        "B  contradicted-node screen: %d %s node-years whose outflow prior is "
        "< %.2f× their inflow prior (%d outflow edges) — σ action: %s\n%s",
        len(keys), "/".join(sectors), UNCLOSED_K, len(ids),
        f"cv={cv}" if cv is not None
        else "none (ADR 0019 — detector only)",
        report.head(10).to_string(index=False))
    return report
