"""Unit tests for the Flow-node prior-contradiction check (ADR 0019)."""
import numpy as np
import pandas as pd

from pysuto.flow_node_balance import contradicted_flow_nodes, flow_node_balance

# nodes: (sector, entity, region) — one smelter + matte flow + refinery per region
_REGIONS = ["Russia", "Iran", "Bulgaria", "Kazakhstan", "Chile"]
_NODES, _EDGES = [], []
for r in _REGIONS:
    s = len(_NODES)
    _NODES += [("Smelting", "Process", r), ("Matte", "Flow", r), ("CFR", "Process", r)]
    _EDGES += [(s, s + 1), (s + 1, s + 2)]        # Smelting→Matte, Matte→CFR


class _Acc:
    def __init__(self):
        self.node_index = pd.DataFrame(_NODES, columns=["sector", "entity", "region"])
        rows, eid = [], 0
        for f, t in _EDGES:
            for val in ("Mass", "CC"):
                rows.append((eid, f, t, "1995", val))
                eid += 1
        self.edge_index = pd.DataFrame(
            rows, columns=["edge_id", "from_id", "to_id", "time", "valuation"])
        self.n_edges = len(rows)


class _Agg:
    def __init__(self, n):
        self.f_agg = np.zeros(n)


def _build(pairs):
    """pairs: {region: (inflow, outflow)} in CC."""
    acc, agg = _Acc(), _Agg(len(_REGIONS) * 4)
    ei = acc.edge_index
    for i, r in enumerate(_REGIONS):
        infl, outfl = pairs[r]
        for k, v in ((2 * i, infl), (2 * i + 1, outfl)):
            eid = int(ei[(ei.index // 2 == k) & (ei["valuation"] == "CC")]["edge_id"].iloc[0])
            agg.f_agg[eid] = v
    return acc, agg


# the real region-years this screen was built from (Round-D priors, kt CC)
_REAL = {"Russia": (545.0, 50.8), "Iran": (116.0, 3.76), "Bulgaria": (103.2, 20.3),
         "Kazakhstan": (250.2, 271.3), "Chile": (1460.4, 1318.1)}


def test_flags_contradicted_nodes_and_spares_closing_ones():
    acc, agg = _build(_REAL)
    keys, rep = contradicted_flow_nodes(acc, agg)
    assert ("Matte", "Russia", "1995") in keys
    assert ("Matte", "Iran", "1995") in keys
    assert ("Matte", "Bulgaria", "1995") in keys
    # these close — a healthy node must never be flagged
    assert ("Matte", "Kazakhstan", "1995") not in keys
    assert ("Matte", "Chile", "1995") not in keys
    assert len(keys) == 3
    assert rep["ratio"].is_monotonic_increasing        # worst first


def test_anode_exports_do_not_rescue_the_matte_node():
    # Bulgaria's real signature: it exports anode, which is produced FROM matte
    # by CFR — so anode exports raise the matte requirement, they are not an
    # outlet for matte. The node must still be flagged.
    acc, agg = _build({**_REAL, "Bulgaria": (103.2, 20.3)})
    keys, _ = contradicted_flow_nodes(acc, agg)
    assert ("Matte", "Bulgaria", "1995") in keys


def test_tiny_nodes_are_not_flagged():
    acc, agg = _build({**_REAL, "Chile": (0.4, 0.0)})   # below MIN_INFLOW_KT
    keys, _ = contradicted_flow_nodes(acc, agg)
    assert ("Matte", "Chile", "1995") not in keys


def test_sector_scope_restricts_action_not_measurement():
    acc, agg = _build(_REAL)
    keys, _ = contradicted_flow_nodes(acc, agg, sectors=("Anode",))
    assert keys == set()                                # nothing acted on
    bal = flow_node_balance(acc, agg)                   # still measured
    assert set(bal["sector"]) == {"Matte"}
    assert len(bal) == len(_REGIONS)
