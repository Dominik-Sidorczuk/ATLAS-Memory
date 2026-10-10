"""V151.4 — Adaptacyjny budżet PPR (WEKTOR II / QM+Config).

Kontrakt:
  1. `fetch_graph_ppr_candidates` NIE MA twardego kodu `35.0` — sygnatura
     `budget_ms=None` (odczyt z konfiguracji, nie z konstanty).
  2. Budżet liczony Z PRZEDU (przed `run()`) z sygnatury snapshotu grafu:
     `budget_ms = max(ppr_tau_base, ppr_tau_base + ppr_gamma * ln(V + E))`
     (qml: graf 100 Ø → ~76.85ms; graf 1924+2706 → ~107ms; nie 35.0).
  3. Jawny `budget_ms=<float>` kalibruje/wygrywa z adaptacją (hook testu).
  4. Budżet musi być adaptacyjny — mutation M2 (hardcode 35.0) obala test.
"""
from __future__ import annotations

import inspect
import math
import types
from typing import Any, Dict

import pytest

from atlas_memory.cognitive.retrieval.ppr import (
    adaptive_ppr_budget,
    fetch_graph_ppr_candidates,
)
from atlas_memory.core.config import AtlasConfig, reset_config
from atlas_memory.l2_semantic.kuzu.ram_multigraph import AtlasMultiDiGraph


class _FakePPREngine:
    """Kapturny silnik PPR: rejestruje kwargs run() i zwraca pusty wynik."""

    def __init__(self) -> None:
        self.calls: list[Dict[str, Any]] = []

    def run(self, query: str, graph_store: Any, top_k: int = 15, budget_ms: float = 0.0):
        self.calls.append({"query": query, "top_k": top_k, "budget_ms": budget_ms})

        class _R:
            node_scores: Dict[str, float] = {}

            @property
            def candidates(self):  # noqa: D401
                return []

        return _R()


def _big_graph() -> AtlasMultiDiGraph:
    g = AtlasMultiDiGraph()
    g.add_node("doc:root", type="passage")
    for i in range(9258):
        g.add_node(f"doc:passage:{i}", type="passage")
        g.add_edge("doc:root", f"doc:passage:{i}", predicate="CONTAINS", confidence=0.9)
    return g


def test_v151_4a_signature_budget_default_is_none() -> None:
    """RED: `budget_ms` nie może już mieć twardego kodu 35.0."""
    sig = inspect.signature(fetch_graph_ppr_candidates)
    default = sig.parameters["budget_ms"].default
    assert default is None, f"expected budget_ms=None, got {default!r}"


def test_v151_4b_adaptive_budget_formula() -> None:
    """Formula: budget = max(tau_base, tau_base + gamma * ln(V + E)); config-driven.

    V151.2: test czystej formuły adaptacyjnej — podłoga `ppr_budget_floor_ms=0`
    (kontrakt podłogi 200ms testuje test_v151_2_f034_ppr_budget_floor_200).
    """
    reset_config(AtlasConfig(ppr_tau_base=40.0, ppr_gamma=8.0, ppr_budget_floor_ms=0.0))
    try:
        b100 = adaptive_ppr_budget(100, 0)
        assert b100 == pytest.approx(max(40.0, 40.0 + 8.0 * math.log(100)), rel=1e-6), b100
        b_big = adaptive_ppr_budget(1924, 2706)
        assert b_big == pytest.approx(40.0 + 8.0 * math.log(4630), rel=1e-6), b_big
        assert 105.0 <= b_big <= 110.0, f"budget ~107ms, got {b_big}"
        b_tiny = adaptive_ppr_budget(1, 0)
        assert b_tiny >= 40.0, f"floor tau_base: {b_tiny}"
    finally:
        reset_config(None)


def test_v151_4c_budget_computed_from_sig_before_run() -> None:
    """Budżet policzony z sygnatury snapshotu (n_nodes, n_edges) PRZED run(); nie 35.0."""
    g = _big_graph()
    gs = types.SimpleNamespace(nx_graph=g)
    fake = _FakePPREngine()
    gs._ppr_engine = fake
    reset_config(AtlasConfig(hipporag_enabled=True, ppr_tau_base=40.0, ppr_gamma=8.0, ppr_budget_floor_ms=0.0))
    try:
        cands = fetch_graph_ppr_candidates(gs, "hipporag ranking engine", limit=15)
        assert isinstance(cands, list)
        assert len(fake.calls) == 1, "PPR run musiał zostać wywołany dokładnie raz"
        v = g.number_of_nodes()
        e = g.number_of_edges()
        expected = max(40.0, 40.0 + 8.0 * math.log(v + e))
        got = fake.calls[0]["budget_ms"]
        assert got != 35.0, "hardcode 35.0 wykryty (mutation M2 domain)"
        assert got == pytest.approx(expected, rel=1e-6), f"got {got}, expected {expected}"
    finally:
        reset_config(None)


def test_v151_4d_explicit_budget_ms_overrides_adaptive() -> None:
    """Jawny budget_ms=(float) wygrywa z adaptacją (kalibrowany hook)."""
    g = AtlasMultiDiGraph()
    g.add_node("doc:root", type="passage")
    for i in range(20):
        g.add_node(f"doc:passage:{i}", type="passage")
        g.add_edge("doc:root", f"doc:passage:{i}", predicate="CONTAINS", confidence=0.9)
    gs = types.SimpleNamespace(nx_graph=g)
    fake = _FakePPREngine()
    gs._ppr_engine = fake
    reset_config(AtlasConfig(hipporag_enabled=True, ppr_tau_base=40.0, ppr_gamma=8.0, ppr_budget_floor_ms=0.0))
    try:
        fetch_graph_ppr_candidates(gs, "hipporag ranking engine", limit=15, budget_ms=12.5)
        assert len(fake.calls) == 1
        assert fake.calls[0]["budget_ms"] == pytest.approx(12.5), fake.calls[0]
    finally:
        reset_config(None)


def test_v151_4e_adaptive_budget_respects_config_tau_gamma() -> None:
    """Config-driven: tau_base i gamma z AtlasConfig, nie z konstant."""
    reset_config(AtlasConfig(ppr_tau_base=10.0, ppr_gamma=2.0, ppr_budget_floor_ms=0.0))
    try:
        b = adaptive_ppr_budget(100, 0)
        expected = max(10.0, 10.0 + 2.0 * math.log(100))
        assert b == pytest.approx(expected, rel=1e-6), b
        assert b != 35.0
    finally:
        reset_config(None)
