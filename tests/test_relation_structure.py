"""
test_relation_structure.py
==========================
Recipes and node balances are derived once in ``pysuto.relation.structure`` (#129): a
co-product process lists every Flow product, Flow and Process nodes both get a balance,
``acc`` imports nothing from ``impute``, and the NetworkImputer refuses a co-product
recipe by name until the co-product ticket lands.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from pysuto.acc import AccountingStructure
from pysuto.tutorial.toy import ACC_SPEC
from pysuto.impute import NetworkImputer
from pysuto.relation.structure import node_balances, process_recipes

REPO = Path(__file__).resolve().parents[1]


def _two_product_spec() -> dict:
    """The toy's spec with a second Flow output, ``Byprod``, on Mining."""
    spec = copy.deepcopy(ACC_SPEC)
    spec["sectors"].append({"name": "Byprod", "entity": "Flow"})
    spec["supply"]["Mining"] = list(spec["supply"]["Mining"]) + ["Byprod"]
    return spec


@pytest.fixture(scope="module")
def toy():
    return AccountingStructure.from_spec(copy.deepcopy(ACC_SPEC), "toy")


@pytest.fixture(scope="module")
def two_product():
    return AccountingStructure.from_spec(_two_product_spec(), "toy2")


def test_a_process_with_two_flow_outputs_lists_both_products(two_product):
    em = two_product.edge_meta.set_index("edge_id")
    mining = [r for r in process_recipes(two_product) if r.process == "Mining"]
    assert mining
    for rec in mining:
        assert len(rec.products) == 2
        assert {em.loc[e, "to_sector"] for e in rec.products} == {"Ore", "Byprod"}


def test_the_toy_has_one_flow_product_per_recipe(toy):
    assert all(len(r.products) <= 1 for r in process_recipes(toy))


def test_recipes_and_balances_are_cached_per_acc(toy):
    assert process_recipes(toy) is process_recipes(toy)
    content = toy.valuation_roles.content
    assert node_balances(toy, content) is node_balances(toy, content)


def test_balances_cover_flow_and_process_nodes_sorted(toy):
    bal = node_balances(toy, toy.valuation_roles.content)
    assert {b.entity for b in bal} == {"Flow", "Process"}
    keys = [(b.node_id, b.time) for b in bal]
    assert keys == sorted(keys)
    em = toy.edge_meta.set_index("edge_id")
    for b in bal:
        assert b.incoming and b.outgoing
        assert (em.loc[list(b.incoming), "to_id"] == b.node_id).all()
        assert (em.loc[list(b.outgoing), "from_id"] == b.node_id).all()
        assert (em.loc[list(b.incoming) + list(b.outgoing), "time"] == b.time).all()


def test_a_recipe_references_its_process_node_balance(toy):
    em = toy.edge_meta.set_index("edge_id")
    incoming = {(b.node_id, b.time): set(b.incoming)
                for b in node_balances(toy, toy.valuation_roles.content)
                if b.entity == "Process"}
    n_checked = 0
    for rec in process_recipes(toy):
        if not rec.inputs:
            continue
        (node_id,) = {int(em.loc[e, "to_id"]) for e in rec.inputs}
        assert incoming[(node_id, rec.time)] == set(rec.inputs)
        n_checked += 1
    assert n_checked > 0


def test_acc_imports_nothing_from_impute():
    tree = ast.parse((REPO / "src" / "pysuto" / "acc.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = ("." * node.level) + (node.module or "")
            assert not mod.startswith(("pysuto.impute", ".impute")), mod
            if mod in ("pysuto", "."):
                assert "impute" not in {a.name for a in node.names}
        elif isinstance(node, ast.Import):
            assert not any(a.name.startswith("pysuto.impute") for a in node.names)


def test_the_network_imputer_accepts_a_co_product_recipe(two_product, tmp_path):
    eff = tmp_path / "eff.csv"
    conc = tmp_path / "conc.csv"
    for p in (eff, conc):
        p.write_text("param_id;param_type\n")
    imputer = NetworkImputer(eff_csv=str(eff), conc_csv=str(conc), q=3.0, alpha_min=0.05, cv_cap=1.5)
    imputer.prepare(two_product)
    mining = [r for r in imputer._recipes if len(r.products) == 2]
    assert mining and len(mining) == len([r for r in process_recipes(two_product)
                                          if r.process == "Mining"])
