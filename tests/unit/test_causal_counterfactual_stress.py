from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from atlas_memory.causal.annealer import CausalAnnealer
from atlas_memory.causal.energy_module import EnergyModule
from atlas_memory.causal.models import CausalEdge


def test_causal_edge_immutable_frozen_validation():
    edge = CausalEdge(
        source="server_a",
        predicate="depends_on",
        target="db_main",
        confidence=0.9,
    )

    # Mutation should raise an error because model is frozen
    with pytest.raises(ValidationError):
        edge.confidence = 0.5  # type: ignore

    assert edge.source == "server_a"
    assert edge.confidence == 0.9


@pytest.mark.asyncio
async def test_causal_annealer_convergence():
    annealer = CausalAnnealer(rng_seed=42)
    edges = [
        CausalEdge(source="server_a", predicate="depends_on", target="db_main", confidence=0.8),
        CausalEdge(source="db_main", predicate="powers", target="cache_redis", confidence=0.6),
    ]

    result = await annealer.run(graph=edges, T_init=0.5, T_min=0.01, max_iter=20)
    assert result.iterations > 0
    assert len(result.final_edges) == 2
    assert all(0.0 <= e.confidence <= 1.0 for e in result.final_edges)


@pytest.mark.asyncio
async def test_energy_module_edge_energy_and_gradient():
    module = EnergyModule()
    edge = CausalEdge(source="A", predicate="powers", target="B", confidence=0.8)

    energy, grad = await module.compute_edge_energy_and_gradient(edge)
    assert isinstance(energy, float)
    assert not np.isnan(energy)
    assert isinstance(grad, float)
    assert not np.isnan(grad)
