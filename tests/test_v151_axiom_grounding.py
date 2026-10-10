"""V151.5 — Ugruntowanie tunelowania aksjomatów (WEKTOR II / condition-skip).

Kontrakt:
  1. `compute_axiom_tunnel_mask(query, axiom_node_names, theta_min=None)` (NOWE API):
     maska per-aksjomat: sim(stems(query), stems(axiom)) > theta_min → 1.0, else 0.0.
     theta_min z AtlasConfig.axiom_theta_min (def 0.05) — config-driven.
  2. `HippoRAG2PPR.run`: u_axioms_eff = u_axioms * mask — aksjomaty bez ugruntowania
     SEMANTYCZNEGO (sim == 0, np. `rule:preferred_editor`, `preference:peptide_diet`)
     mają ZERO energii tunelowania i NIE mogą się pojawić w kandydatach PPR dla
     zapytania o silnik grafowy (brak recyklingu reguł w knapsacku).
  3. Aksjomat ugruntowany (wspólne łodygi z zapytaniem) nadal dostaje energię.
"""
from __future__ import annotations

import pytest

from atlas_memory.cognitive.retrieval.ppr import (
    HippoRAG2PPR,
    compute_axiom_tunnel_mask,
)
from atlas_memory.core.config import AtlasConfig, reset_config
from atlas_memory.l2_semantic.kuzu.ram_multigraph import AtlasMultiDiGraph


@pytest.fixture(autouse=True)
def _config_isolation():
    reset_config(AtlasConfig(axiom_theta_min=0.05))
    yield
    reset_config(None)


def test_v151_5a_mask_zero_for_disjoint_axioms() -> None:
    """Reguły/preferencje bez wspólnych łodyg → zero energii tunelowania."""
    mask = compute_axiom_tunnel_mask(
        "hipporag graph ppr ranking engine",
        ["rule:preferred_editor", "preference:peptide_diet", "rule:kitchen_hygiene"],
    )
    assert mask == [0.0, 0.0, 0.0], mask


def test_v151_5b_mask_one_for_grounded_axiom() -> None:
    """Aksjomat dzielący łodygi z zapytaniem → pełna energia."""
    mask = compute_axiom_tunnel_mask(
        "hipporag graph ppr ranking",
        ["rule:hipporag_ppr_engine_ranking"],
    )
    assert mask == [1.0], mask


def test_v151_5c_theta_min_is_config_driven() -> None:
    """THETA_MIN z config: próg 0.99 wycina nawet częściowe podobieństwo."""
    reset_config(AtlasConfig(axiom_theta_min=0.99))
    strict = compute_axiom_tunnel_mask("ppr engine", ["rule:ppr"])
    assert strict == [0.0], strict
    reset_config(AtlasConfig(axiom_theta_min=0.05))
    loose = compute_axiom_tunnel_mask("ppr engine", ["rule:ppr"])
    assert loose == [1.0], loose


def _dual_node_graph() -> AtlasMultiDiGraph:
    g = AtlasMultiDiGraph()
    g.add_node("doc:engine_notes", type="passage")
    g.add_node("graph engine", type="entity")
    g.add_edge("doc:engine_notes", "graph engine", predicate="CONTAINS", confidence=0.9)
    # Aksjomaty NIEUGRUNTOWANE (zero wspólnych łodyg z zapytaniem o silnik grafowy):
    g.add_node("rule:preferred_editor", type="entity")
    g.add_edge("rule:preferred_editor", "vim", predicate="PREFERS", confidence=0.9)
    g.add_node("vim", type="entity")
    g.add_node("preference:peptide_diet", type="entity")
    return g


def test_v151_5d_ungrounded_rules_receive_zero_energy() -> None:
    """Zapytanie o silnik grafowy NIE rekapituluje reguł edytora/diety (knapsack-safe)."""
    g = _dual_node_graph()
    res = HippoRAG2PPR(backend="numpy").run(
        query="hipporag graph ppr ranking engine", graph_store=g, top_k=15, budget_ms=0.0
    )
    keys = {str(c.get("key", "")) for c in res.candidates}
    assert "rule:preferred_editor" not in keys, f"tunelowanie bez ugruntowania: {keys}"
    assert "preference:peptide_diet" not in keys, f"tunelowanie bez ugruntowania: {keys}"
    scores = res.node_scores
    assert scores.get("rule:preferred_editor", 0.0) == 0.0, scores
    assert scores.get("preference:peptide_diet", 0.0) == 0.0, scores


def test_v151_5e_grounded_rule_still_tunnels() -> None:
    """Aksjomat ugruntowany semantycznie (wspólne łodygi) nadal dostaje energię."""
    g = AtlasMultiDiGraph()
    g.add_node("doc:engine_notes", type="passage")
    g.add_node("graph engine", type="entity")
    g.add_edge("doc:engine_notes", "graph engine", predicate="CONTAINS", confidence=0.9)
    # Izolowany, ALE ugruntowany aksjomat dot. zarzadzania silnikiem grafowym:
    g.add_node("rule:hipporag_graph_engine_ranking_policy", type="entity")
    res = HippoRAG2PPR(backend="numpy").run(
        query="hipporag graph ppr ranking engine", graph_store=g, top_k=15, budget_ms=0.0
    )
    assert res.node_scores.get("rule:hipporag_graph_engine_ranking_policy", 0.0) > 0.0, (
        "ugruntowany aksjomat utracił energię tunelowania"
    )
