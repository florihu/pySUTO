"""acc_store.py
==============
The persisted form of an :class:`~pysuto.acc.AccountingStructure`: ``save`` writes it
under ``<storage_dir>/<id>/``, ``load`` reads it back, and the build record beside it
names the acc input version it was made from. ``AccountingStructure`` itself only
builds; this module is the one place that knows the file layout.

Layout of ``<storage_dir>/<id>/``::

    edge_index.csv  node_index.csv  root_nodes.csv     the index tables
    bounds_l.npy    bounds_u.npy                       per-edge bounds
    masks_user.npz                                     acc-local user masks (if any)
    metadata.json                                      the header: axes, node ids,
                                                       roles, rule tables
    build_record.json                                  written by write_build_record

An older directory holds ``metadata.pkl`` (a pickle of the same header)
in place of ``metadata.json`` and four files nothing reads (``A.npz``, ``A_nodes.npz``,
``edge_index_all.csv``): ``load`` reads the pickle when there is no JSON header and
never opens the other three; ``save`` removes them.
"""

from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd

from .acc import (DEFAULT_STORAGE_DIR, DEFAULT_UPPER_BOUND, AccountingStructure,
                  EdgeBounds, RuleSet, ValuationRole, _bind_roles,
                  require_compatible_acc)
from .logging import get_logger

HEADER        = "metadata.json"
LEGACY_HEADER = "metadata.pkl"
# Files of the older layout: no consumer reads them; ``save`` deletes stale copies.
_LEGACY_FILES = (LEGACY_HEADER, "A.npz", "A_nodes.npz", "edge_index_all.csv")

# =============================================================================
# Save
# =============================================================================


def _header(acc: AccountingStructure) -> dict:
    """The JSON-able header of ``acc``: node ids as a list in id order, rule-table
    sets as sorted lists."""
    rs = acc.ruleset
    return {
        "time_range": list(acc.time_range),
        "valuation":  list(acc.valuation),
        "nodes":      [list(acc.id_to_node[i]) for i in range(acc.n_nodes)],
        "boundary_roles": {s: r.value for s, r in acc.boundary_roles.items()},
        "valuation_roles": {acc.valuation_roles.content: ValuationRole.CONTENT.value,
                            acc.valuation_roles.gross:   ValuationRole.GROSS.value},
        "rule_tables": {
            "supply_pairs": sorted(map(list, rs.supply_pairs)),
            "use_pairs":    sorted(map(list, rs.use_pairs)),
            "tradable":     sorted(rs.tradable_flows),
            "accounting":   sorted([list(k), v] for k, v in rs.accounting.items()),
        },
    }


def save(acc: AccountingStructure) -> str:
    """Write ``acc`` to ``<acc.storage_dir>/<acc.id>/``; return that directory."""
    path = os.path.join(acc.storage_dir, acc.id)
    os.makedirs(path, exist_ok=True)

    acc.edge_index.to_csv(os.path.join(path, "edge_index.csv"), index=False)
    acc.node_index.to_csv(os.path.join(path, "node_index.csv"), index=False)
    acc.root_nodes.to_csv(os.path.join(path, "root_nodes.csv"), index=False)
    np.save(os.path.join(path, "bounds_l.npy"), acc.bounds.l)
    np.save(os.path.join(path, "bounds_u.npy"), acc.bounds.u)

    # acc-local user masks (masks.csv) — load() rebuilds only the structural
    # masks, so without this file a user mask silently vanishes after load.
    masks_path = os.path.join(path, "masks_user.npz")
    if acc.user_mask_names:
        np.savez(masks_path, **{n: acc.masks[n] for n in acc.user_mask_names})
    elif os.path.exists(masks_path):
        os.remove(masks_path)  # a stale file would resurrect removed masks

    # Roles and rule tables, so load() returns what the build made: the rule-derived
    # masks (supply, use, tradable) are rebuilt from the same tables.
    with open(os.path.join(path, HEADER), "w", encoding="utf-8") as f:
        json.dump(_header(acc), f, indent=1)
    for name in _LEGACY_FILES:      # a rebuilt directory keeps no older-layout file
        if os.path.exists(os.path.join(path, name)):
            os.remove(os.path.join(path, name))
    acc.logger.info(f"Structure saved to {path}")
    return path


# =============================================================================
# Load
# =============================================================================


def _read_header(path: str) -> dict:
    """The header as ``save`` writes it: from ``metadata.json``, or from the
    older ``metadata.pkl`` converted to the same shape."""
    json_path = os.path.join(path, HEADER)
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    with open(os.path.join(path, LEGACY_HEADER), "rb") as f:
        meta = pickle.load(f)
    nodes = meta["id_to_node"]
    meta["nodes"] = [list(nodes[i]) for i in range(len(nodes))]
    if meta.get("rule_tables") is not None:
        rt = meta["rule_tables"]
        rt["accounting"] = [[list(k), v] for k, v in rt["accounting"].items()]
    return meta


def load(id: str, storage_dir: str = DEFAULT_STORAGE_DIR) -> AccountingStructure:
    """Read the structure saved under ``<storage_dir>/<id>/``.

    Raises
    ------
    FileNotFoundError
        The directory does not exist.
    ValueError
        The header records no roles or rule tables (saved by an older version), a persisted
        role fails the build's validation, or a user mask's shape does not match the
        edge index.
    """
    path = os.path.join(storage_dir, id)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Directory '{id}' not found in {storage_dir}: build it with "
            f"scripts/acc.py --id {id}"
        )
    obj = AccountingStructure.__new__(AccountingStructure)
    obj._set_identity(id, storage_dir)

    obj.edge_index = pd.read_csv(os.path.join(path, "edge_index.csv"))
    # Ensure time column is string — pandas infers year integers from CSV
    obj.edge_index["time"] = obj.edge_index["time"].astype(str)
    obj.root_nodes = pd.read_csv(os.path.join(path, "root_nodes.csv"))
    obj.node_index = pd.read_csv(os.path.join(path, "node_index.csv"))
    if "entity_sub" not in obj.node_index.columns:
        obj.node_index["entity_sub"] = None

    meta = _read_header(path)
    missing = [k for k in ("boundary_roles", "valuation_roles", "rule_tables")
               if meta.get(k) is None]
    if missing:
        raise ValueError(
            f"acc {id!r} at {path} records no {missing}: it was saved before roles "
            f"and rule tables were persisted (#119) — rebuild it with "
            f"scripts/acc.py --id {id}"
        )

    obj.time_range  = list(meta["time_range"])
    obj.valuation   = list(meta["valuation"])
    obj.id_to_node  = {i: tuple(n) for i, n in enumerate(meta["nodes"])}
    obj.node_to_id  = {n: i for i, n in obj.id_to_node.items()}
    obj.n_nodes     = len(obj.id_to_node)
    obj.n_timeps    = len(obj.time_range)
    obj.n_valtns    = len(obj.valuation)
    obj.n_edges     = len(obj.edge_index)
    obj.time_to_idx = {t: i for i, t in enumerate(obj.time_range)}
    obj.val_to_idx  = {v: i for i, v in enumerate(obj.valuation)}
    rt = meta["rule_tables"]
    obj.ruleset     = RuleSet.from_tables(
        {tuple(p) for p in rt["supply_pairs"]}, {tuple(p) for p in rt["use_pairs"]},
        set(rt["tradable"]), {tuple(k): v for k, v in rt["accounting"]})
    obj.logger.info("Loading AccountingStructure '%s' from %s", id, path)

    # The same validation as the build: the persisted roles bound to the nodes.
    sectors = obj.node_index.drop_duplicates(["sector", "entity"])
    obj.boundary_roles, obj.valuation_roles = _bind_roles(
        [{"name": s, "entity": e, "role": meta["boundary_roles"].get(s)}
         for s, e in zip(sectors["sector"], sectors["entity"])],
        [{"name": v, "role": meta["valuation_roles"].get(v)}
         for v in obj.valuation],
    )

    l_path = os.path.join(path, "bounds_l.npy")
    u_path = os.path.join(path, "bounds_u.npy")
    bounds = EdgeBounds.__new__(EdgeBounds)
    bounds.n_edges = obj.n_edges
    if os.path.exists(l_path) and os.path.exists(u_path):
        bounds.l = np.load(l_path)
        bounds.u = np.load(u_path)
    else:
        bounds.l = np.zeros(obj.n_edges,  dtype=float)
        bounds.u = np.full(obj.n_edges, DEFAULT_UPPER_BOUND, dtype=float)
    non_default = (
        (bounds.l != EdgeBounds.DEFAULT_LOWER) |
        (bounds.u != EdgeBounds.DEFAULT_UPPER)
    )
    bounds.n_exceptions = int(non_default.sum())
    obj.bounds = bounds

    obj.logger.info(
        "   nodes=%d  edges=%d  time=%s–%s  valuations=%s  bound_exceptions=%d",
        obj.n_nodes, obj.n_edges,
        obj.time_range[0], obj.time_range[-1],
        obj.valuation, bounds.n_exceptions,
    )
    obj._build_edge_masks()

    obj.user_mask_names = []
    masks_path = os.path.join(path, "masks_user.npz")
    if os.path.exists(masks_path):
        with np.load(masks_path) as z:
            for name in z.files:
                arr = z[name]
                if arr.shape != (obj.n_edges,):
                    raise ValueError(
                        f"masks_user.npz: mask {name!r} has shape {arr.shape}, "
                        f"expected ({obj.n_edges},) — rebuild the structure "
                        f"with scripts/acc.py"
                    )
                obj.masks[name] = arr.astype(bool)
                obj.user_mask_names.append(name)
        obj.logger.info("   user masks restored: %s", ", ".join(obj.user_mask_names))
    return obj


# =============================================================================
# Build record — which acc input version a built structure was made from
# =============================================================================

# Written by scripts/acc.py beside the saved structure, in <storage_dir>/<id>/.
BUILD_RECORD = "build_record.json"


def write_build_record(storage_dir: str, entry: dict) -> str:
    """Record ``{id, version, code}`` of the acc input entry the structure saved
    under ``<storage_dir>/<id>/`` was built from; return the record's path."""
    path = os.path.join(storage_dir, entry["id"], BUILD_RECORD)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: entry[k] for k in ("id", "version", "code")}, f, indent=2)
    return path


def check_built_acc(storage_dir: str, acc_ref: dict, acc_registry, keys,
                    require_record: bool = False) -> None:
    """Check the build record of the structure under ``<storage_dir>/<id>/`` against
    ``acc_ref``, the acc input version (``{id, version, ...}``) a run's sidecar names.

    A record naming another version of the same acc passes when the two versions' specs
    agree on ``keys`` (:func:`require_compatible_acc`, read from ``acc_registry``):
    :data:`ALL_SPEC_KEYS` for Stages 2 and 3, :data:`EDGE_UNIVERSE_ROLE_KEYS` for
    Stages 4 (warm) and 6. A structure built before
    the record existed has none: unless ``require_record``, that is logged as a warning
    and the caller goes on, so the runs made from it still load. Reads the record and
    the two specs only, so a caller checks before the (multi-GB) load.

    Raises
    ------
    ValueError
        ``acc_ref`` lacks its id or version; the record names another acc id, or a
        version whose spec differs in a key of ``keys``; or ``require_record`` and the
        structure has no record.
    """
    ref = (acc_ref.get("id"), acc_ref.get("version"))
    if not ref[0] or ref[1] is None:
        raise ValueError(f"the acc ref {acc_ref!r} names no acc id and version, so the "
                         f"structure in {storage_dir} cannot be checked against it")
    path = os.path.join(storage_dir, ref[0], BUILD_RECORD)
    if not os.path.isfile(path):
        msg = (f"{os.path.dirname(path)}: the built structure has no build record "
               f"(built before #53), so it cannot be checked against acc "
               f"{ref[0]} v{ref[1]}")
        if require_record:
            raise ValueError(f"{msg}: rebuild it with scripts/acc.py")
        get_logger("AccountingStructure").warning(msg)
        return
    with open(path, encoding="utf-8") as f:
        rec = json.load(f)
    context = (f"{os.path.dirname(path)} was built from acc {rec['id']} "
               f"v{rec['version']}, but the run names acc {ref[0]} v{ref[1]}")
    remedy  = ("rebuild the structure with scripts/acc.py, or run against the acc it "
               "was built from")
    if rec["id"] != ref[0]:
        raise ValueError(f"{context}: {remedy}")
    require_compatible_acc(acc_registry, ref[0], rec["version"], ref[1], keys,
                           context=context, remedy=remedy)
