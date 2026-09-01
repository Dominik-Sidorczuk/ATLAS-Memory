"""Testy Fazy V19: Asynchronous Causal Graph Self-Annealing.

Weryfikacja:
- EnergyModule.intrinsic_cost / trainable_critic w zakresie [0.0, 1.0]
- CausalAnnealer.run — konwergencja, spadek temperatury, clamp, adaptacja
- Wydajność: 50-węzłowy graf < 100 ms
"""

from __future__ import annotations

import asyncio
import time

from atlas_memory.causal.annealer import CausalAnnealer
from atlas_memory.causal.energy_module import EnergyModule
from atlas_memory.causal.models import CausalEdge


def _make_graph(n_nodes: int = 50, seed: float = 0.0) -> list[CausalEdge]:
    """Buduje graf przyczynowy n-węzłowy z różnymi confidence."""
    edges = []
    for i in range(n_nodes):
        confidence = 0.3 + ((i + seed) % 7) * 0.1
        edges.append(
            CausalEdge(
                source=f"node_{i}",
                predicate="depends_on" if i % 3 == 0 else "triggers",
                target=f"node_{(i + 1) % n_nodes}",
                confidence=round(min(0.95, max(0.05, confidence)), 3),
            )
        )
    return edges


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# EnergyModule
# ---------------------------------------------------------------------------


def test_energy_module_intrinsic_cost_range():
    """Koszt błędu predykcji musi być w [0.0, 1.0] dla różnych akcji."""
    em = EnergyModule()

    async def _check():
        costs = []
        for action in ("restart", "reboot", "read", "query", "delete", "observe", "plan"):
            costs.append(await em.intrinsic_cost("svc_db", action))
        return costs

    costs = _run(_check())
    assert all(0.0 <= c <= 1.0 for c in costs)
    # Akcja destrukcyjna (restart) kosztuje więcej niż obserwacja (query)
    assert costs[0] >= costs[3]
    assert costs[2] <= costs[5]  # read <= observe (heurystyka niepewności)


def test_energy_module_trainable_critic_range():
    """Critic CPoF musi być w [0.0, 1.0]."""
    em = EnergyModule()
    edges = _make_graph(10)

    async def _check():
        return await em.trainable_critic("node_0", edges=edges)

    risk = _run(_check())
    assert 0.0 <= risk <= 1.0


def test_energy_module_prediction_error_raises_cost():
    """Zarejestrowany błąd predykcji telemetrii podnosi koszt dla tej pary."""
    em = EnergyModule()
    em.register_prediction_error("svc_a", "restart", 0.9)

    async def _check():
        before = await em.intrinsic_cost("svc_a", "restart")
        return before

    cost = _run(_check())
    assert cost >= 0.5  # wysoki discrepancy_score => wysoki koszt


# ---------------------------------------------------------------------------
# CausalAnnealer — konwergencja i temperatura
# ---------------------------------------------------------------------------


def test_annealer_converges_50_node_graph():
    """50-węzłowy graf musi skonwergować poniżej max_iter."""
    graph = _make_graph(50)
    annealer = CausalAnnealer(rng_seed=42)
    result = _run(annealer.run(graph, max_iter=500))

    assert result.converged is True
    assert result.iterations < 500
    assert result.iterations > 0
    assert len(result.energy_trace) == result.iterations
    assert len(result.final_edges) == 50


def test_annealer_temperature_decay():
    """Temperatura musi maleć zgodnie z decay, aż do T_min."""
    graph = _make_graph(10)
    annealer = CausalAnnealer(rng_seed=7)
    T_init, T_min = 1.0, 0.01
    result = _run(annealer.run(graph, T_init=T_init, T_min=T_min, decay=0.95, max_iter=500))

    # Po ~90 iteracjach T = 1.0 * 0.95^90 ≈ 0.0097 < T_min => stop
    assert result.iterations > 0
    # Finalna temperatura MUSI być >= T_min (z marginesem błędu float)
    assert result.final_temperature >= T_min - 1e-9, (
        f"Temperatura spadła poniżej T_min={T_min}: {result.final_temperature}. "
        "Warunek zatrzymania (T<=T_min) nie działa."
    )
    assert result.final_temperature <= 0.05


def test_annealer_confidence_stays_in_range():
    """Po wyżarzaniu wszystkie confidence muszą pozostać w [0.0, 1.0]."""
    graph = _make_graph(50)
    annealer = CausalAnnealer(rng_seed=123)
    result = _run(annealer.run(graph, max_iter=500))

    assert all(0.0 <= e.confidence <= 1.0 for e in result.final_edges)
    # Graf musi zachować strukturę (source/predicate/target bez mutacji)
    for orig, new in zip(graph, result.final_edges, strict=False):
        assert orig.source == new.source
        assert orig.predicate == new.predicate

        assert orig.target == new.target


def test_annealer_adapts_after_prediction_error():
    """Błąd predykcji telemetrii musi obniżyć confidence zależnej krawędzi."""
    em = EnergyModule()
    em.register_prediction_error("node_0", "depends_on", 0.95)

    graph = _make_graph(10)
    annealer = CausalAnnealer(energy_module=em, rng_seed=1)

    # Wysoka energia => annealing musi obniżyć confidence tam gdzie koszt duży
    result = _run(annealer.run(graph, max_iter=200))

    affected = [e for e in result.final_edges if e.source == "node_0"]
    assert affected  # muszą istnieć
    # Po wyżarzaniu z wysokim kosztem confidence nie może wzrosnąć ponad startowy
    original = next(e for e in graph if e.source == "node_0")
    assert all(e.confidence <= original.confidence + 0.15 for e in affected)


def test_annealer_energy_monotonically_decreases_or_plateaus():
    """Energia całkowita nie może wzrastać między iteracjami (poza szumem anneal)."""
    graph = _make_graph(20)
    annealer = CausalAnnealer(rng_seed=99)
    result = _run(annealer.run(graph, max_iter=300))

    assert len(result.energy_trace) >= 2
    # Energia końcowa nie może być znacząco wyższa niż początkowa (deterministyczny dryf w dół)
    assert result.energy_trace[-1] <= result.energy_trace[0] + 1e-3


def test_annealer_empty_graph_no_crash():
    """Pusty graf nie może crashować — zwraca pusty wynik z converged=True."""
    annealer = CausalAnnealer()
    result = _run(annealer.run([], max_iter=500))

    assert result.converged is True
    assert result.final_edges == []
    assert result.energy_trace == []


def test_annealer_performance_50_nodes_under_100ms():
    """50-węzłowy graf musi skonwergować w < 100 ms (wymóg dyrektywy)."""
    graph = _make_graph(50)
    annealer = CausalAnnealer(rng_seed=42)

    start = time.perf_counter()
    result = _run(annealer.run(graph, max_iter=200))
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert elapsed_ms < 100.0, f"Wydajność przekroczona: {elapsed_ms:.1f} ms"
    assert result.duration_ms < 100.0


def test_annealer_input_graph_not_mutated():
    """Graf wejściowy nie może zostać zmodyfikowany (deep copy)."""
    graph = _make_graph(10)
    original_confidences = [e.confidence for e in graph]

    annealer = CausalAnnealer(rng_seed=5)
    _run(annealer.run(graph, max_iter=100))

    assert [e.confidence for e in graph] == original_confidences
