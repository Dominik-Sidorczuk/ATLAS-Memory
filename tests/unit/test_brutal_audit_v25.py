"""
Unit tests for V25 Brutal Audit (Determinizm hashy, bufor stanów JEPA, czyszczenie Kuzu, Causal $O(1)$).
"""

from __future__ import annotations

import numpy as np

from atlas_memory.causal.annealer import CausalAnnealer
from atlas_memory.causal.energy_module import EnergyModule
from atlas_memory.causal.models import CausalEdge
from atlas_memory.l1_working.jepa_latent import JEPALatentBuffer
from atlas_memory.l2_semantic.kuzu_graph import KuzuGraphStore
from atlas_memory.l2_semantic.qdrant_store import QdrantVectorStore
from atlas_memory.models import ActionPlan


def test_jepa_action_encoding_deterministic():
    """Weryfikacja że encode_action jest 100% deterministyczny na tym samym procesie i między restartami."""
    buf1 = JEPALatentBuffer(state_dim=16, action_dim=8, seed=42)
    buf2 = JEPALatentBuffer(state_dim=16, action_dim=8, seed=42)

    act = ActionPlan(name="db_query", parameters={"table": "users", "limit": 10})
    enc1 = buf1.encode_action(act)
    enc2 = buf2.encode_action(act)

    assert np.allclose(enc1, enc2)


def test_jepa_history_rolling_buffer():
    """Weryfikacja że historia stanów w JEPA nie rośnie w sposób nieograniczony (ochrona przed memory leak)."""
    buf = JEPALatentBuffer(state_dim=16, action_dim=8, seed=42, max_history=5)
    for i in range(20):
        buf.commit_state_transition(ActionPlan(name=f"step_{i}"))

    assert len(buf.history) == 5
    assert buf.history[-1].context_tags == ["step_19"]


async def test_causal_annealer_dict_index_convergence():
    """Weryfikacja że CausalAnnealer poprawnie działa z indeksem słownikowym i szybko konwerguje."""
    energy_mod = EnergyModule()
    annealer = CausalAnnealer(energy_module=energy_mod, rng_seed=42)

    edges = [
        CausalEdge(source="web", predicate="depends_on", target="api", confidence=0.9),
        CausalEdge(source="api", predicate="depends_on", target="db", confidence=0.85),
        CausalEdge(source="db", predicate="hosted_on", target="nas", confidence=0.95),
    ]

    res = await annealer.run(edges, T_init=0.5, T_min=0.01, max_iter=20)
    assert res.converged
    assert len(res.final_edges) == 3
    for e in res.final_edges:
        assert 0.0 <= e.confidence <= 1.0


def test_qdrant_ngram_hashing_deterministic():
    """Weryfikacja deterministycznego rzutowania n-gramów w QdrantVectorStore."""
    store = QdrantVectorStore(dimension=32)
    vec1 = store.encoder.encode("PostgreSQL configuration database port 5432")
    vec2 = store.encoder.encode("PostgreSQL configuration database port 5432")

    assert np.allclose(vec1, vec2)
    assert len(vec1) == 32



def test_kuzu_graph_store_close_cleans_resources():
    """Weryfikacja że KuzuGraphStore.close() czyści uchwyty i usuwa katalog tymczasowy."""
    store = KuzuGraphStore(db_path=":memory:")
    assert store.nx_graph is not None
    store.close()
    assert store.conn is None
    assert store.db is None
    assert store._temp_dir is None
