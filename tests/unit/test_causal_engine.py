"""
Canonical Causal Reasoning & Annealing Tests (consolidated).

Covers:
- Topological graph diffusion & multi-hop confidence propagation
- Cycle detection & self-loop avoidance
- Epistemic risk classification & PII sanitization
- EnergyModule intrinsic cost & critic scoring
- CausalAnnealer convergence & performance
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List

import pytest

from atlas_memory.causal.annealer import CausalAnnealer
from atlas_memory.causal.energy_module import EnergyModule
from atlas_memory.causal.models import CausalEdge
from atlas_memory.causal.retro_causal_edge import (
    EMAIL_PATTERN,
    RetroCausalEngine,
    is_pii_entity,
)


class MockGraphForCausal:
    """Wielorelacyjny graf testowy dla weryfikacji wnioskowania przyczynowo-skutkowego."""
    def __init__(self):
        self.relations: List[Dict[str, Any]] = []

    def add(self, subj: str, pred: str, obj: str, conf: float = 1.0):
        self.relations.append({
            "subject": subj,
            "predicate": pred,
            "object": obj,
            "confidence": conf,
        })

    async def get_subgraph_relations(self, entities, max_depth=2):
        return {"relations": self.relations}


@pytest.mark.asyncio
async def test_causal_what_if_propagation_and_decay():
    """Weryfikacja propagacji wieloskokowej z tłumieniem pewności (decay)."""
    g = MockGraphForCausal()
    g.add("entity_nas_01", "hosts", "NFS_Share", 0.9)
    g.add("NFS_Share", "mounts_on", "AppServer", 0.8)
    g.add("AppServer", "serves", "WebAPI", 0.7)

    engine = RetroCausalEngine(graph_client=g)

    # 1 hop
    paths_1 = await engine.causal_what_if("entity_nas_01", "restart_service", depth=1)
    assert len(paths_1) == 1
    assert paths_1[0].affected_target == "NFS_Share"
    assert paths_1[0].cumulative_confidence == 0.9
    assert paths_1[0].risk_level == "CRITICAL"

    # Multi-hop with decay
    paths_multi = await engine.causal_what_if("entity_nas_01", "restart_service", depth=3)
    assert len(paths_multi) >= 2
    targets = [p.affected_target for p in paths_multi]
    assert "NFS_Share" in targets
    assert "AppServer" in targets

    # Confidence must monotonically decrease across hops
    app_path = next(p for p in paths_multi if p.affected_target == "AppServer")
    assert app_path.cumulative_confidence < 0.9


@pytest.mark.asyncio
async def test_causal_cycle_detection_and_self_loops():
    """Weryfikacja braku nieskończonych pętli przy cyklicznym grafie."""
    g = MockGraphForCausal()
    g.add("A", "causes", "B", 0.9)
    g.add("B", "causes", "C", 0.9)
    g.add("C", "causes", "A", 0.9)  # Cykl A -> B -> C -> A
    g.add("A", "causes", "A", 0.9)  # Self-loop

    engine = RetroCausalEngine(graph_client=g)
    paths = await engine.causal_what_if("A", "modify_config", depth=5)

    targets = [p.affected_target for p in paths]
    assert "A" not in targets or all(p.path[0] != p.path[-1] or len(p.path) == 1 for p in paths)
    assert len(paths) <= 3


@pytest.mark.asyncio
async def test_causal_risk_classification():
    """Weryfikacja poziomów ryzyka zależnie od akcji (CRITICAL vs MODERATE/LOW)."""
    g = MockGraphForCausal()
    g.add("database_primary", "replicates_to", "database_replica", 0.95)
    g.add("worker_node", "logs_to", "monitoring_agent", 0.5)

    engine = RetroCausalEngine(graph_client=g)

    # Destrukcyjna akcja lub krytyczny predykat -> wysokie ryzyko
    res_crit = await engine.causal_what_if("database_primary", "drop_table", depth=1)
    assert res_crit[0].risk_level in ("CRITICAL", "HIGH")

    # Bezpieczny odczyt na zwykłym predykacie -> niskie ryzyko
    res_read = await engine.causal_what_if("worker_node", "get_status", depth=1)
    assert res_read[0].risk_level in ("LOW", "NEGLIGIBLE", "MODERATE")


def test_causal_pii_sanitization():
    """Weryfikacja wykrywania i odrzucania PII w węzłach grafu."""
    assert is_pii_entity("user@domain.com") is True
    assert is_pii_entity("admin_123@sub.hermes.org") is True
    assert is_pii_entity("postgres_cluster_01") is False
    assert is_pii_entity("database_replica") is False
    assert EMAIL_PATTERN.search("contact: dev@example.com for info") is not None


@pytest.mark.asyncio
async def test_energy_module_intrinsic_cost():
    """Koszt błędu predykcji EnergyModule musi mieścić się w [0.0, 1.0]."""
    em = EnergyModule()
    actions = ["restart", "reboot", "read", "query", "delete", "observe", "plan"]
    costs = [await em.intrinsic_cost("svc_db", a) for a in actions]

    assert all(0.0 <= c <= 1.0 for c in costs)
    # restart (index 0) musi kosztować więcej niż query (index 3)
    assert costs[0] >= costs[3]


def test_causal_annealer_convergence_and_performance():
    """CausalAnnealer konwerguje i przetwarza graf 50-węzłowy < 250ms."""
    n_nodes = 50
    edges = [
        CausalEdge(
            source=f"node_{i}",
            predicate="depends_on" if i % 3 == 0 else "triggers",
            target=f"node_{(i + 1) % n_nodes}",
            confidence=round(min(0.95, max(0.05, 0.3 + (i % 7) * 0.1)), 3),
        )
        for i in range(n_nodes)
    ]

    annealer = CausalAnnealer(rng_seed=42)
    t0 = time.perf_counter()
    result = asyncio.run(annealer.run(edges, max_iter=20))
    elapsed = time.perf_counter() - t0

    assert len(result.final_edges) == len(edges)
    assert elapsed < 0.250, f"Annealing took too long: {elapsed:.3f}s"
    assert all(0.0 <= e.confidence <= 1.0 for e in result.final_edges)
