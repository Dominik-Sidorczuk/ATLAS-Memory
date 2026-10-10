"""
Unit tests for v145.0 Cognitive Singularity Remediation:
Homomorphism, Metric Continuity, Associative Bridging, and RAM Deep Immutability.
"""

import copy
import math
import threading
from unittest.mock import MagicMock

from atlas_memory.cognitive.retrieval.ppr import HippoRAG2PPR, expand_associative_seeds_2hop
from atlas_memory.cognitive.retrieval.render import pack_knapsack_01
from atlas_memory.cognitive.retrieval.strategies.fusion import (
    EntropySalienceFilter,
    StaticAnchorDbsfFusion,
)
from atlas_memory.cognitive.write_path import EpistemicWritePath
from atlas_memory.core.models import MemoryRecord
from atlas_memory.l1_working.hot_buffer import HotBufferPolicy
from atlas_memory.l2_semantic.kv_store import VerifiedKVStore
from atlas_memory.l2_semantic.qdrant_store import QdrantVectorStore

# ==============================================================================
# WEKTOR I: Homomorfizm i Odwracalność Przestrzeni Zapisu (Savepoint & Key Mapping)
# ==============================================================================

def test_wektor_i_savepoint_rollback_and_key_symmetry(tmp_path):
    """
    Testuje wycofanie transakcji w sadze przez SAVEPOINT saga_tx:
    1. Pre-state hash w audycie jest zapamiętany.
    2. Błąd wektora lub grafu po operacji set_sync w KV wycofuje stan bazy bez wpisywania akcji DELETE.
    3. Hash audytu po rollbacku jest identyczny z pre-state hash (zero histerezy).
    4. Qdrant defensywnie usuwa i nie odnajduje klucza ani w postaci surowej, ani z prefiksem kv:.
    """
    db_path = tmp_path / "atlas_state.db"
    kv = VerifiedKVStore(str(db_path))
    
    # Inicjalizacja bazy jakimś stanem bazowym
    kv.set_sync("init_key", "init_val", metadata={"test": True})
    pre_audit_hash = kv.get_latest_audit_hash()
    assert pre_audit_hash != ""

    # Mock Qdrant
    qdrant = QdrantVectorStore(location=str(tmp_path / "qdrant"), dimension=384)
    rec = MemoryRecord(subject="test_topic", predicate="has_value", object="v", confidence=1.0)
    qdrant.insert_sync(rec, raw_text="v", upsert_key="kv:test_topic")
    assert qdrant.get_by_key("test_topic") is not None
    assert qdrant.get_by_key("kv:test_topic") is not None

    # Tworzymy EpistemicWritePath z mockiem grafu rzucającym błąd, aby wymusić rollback Sagi
    failing_graph = MagicMock()
    failing_graph.add_relation.side_effect = RuntimeError("Simulated Graph Explosion")

    write_path = EpistemicWritePath(
        kv_store=kv,
        vector_store=qdrant,
        graph_store=failing_graph,
    )

    # Wywołanie write powinno przechwycić błąd i wycofać stan lub rzucić wyjątek po rollbacku
    try:
        write_path.write(
            key="test_topic",
            value="new_value",
            metadata={"subject": "test_topic", "predicate": "links", "object": "exploding_target"},
        )
    except Exception:
        pass

    # 1. Sprawdzenie KV: klucz test_topic nie powinien istnieć w KV (został wycofany przez rollback savepointa)
    assert kv.get_sync("test_topic") is None

    # 2. Sprawdzenie Hash Chain: brak wpisu DELETE, hash powrócił do stanu pre-state
    post_audit_hash = kv.get_latest_audit_hash()
    assert post_audit_hash == pre_audit_hash, "Łańcuch audytu doznał histerezy podczas wycofania savepointa!"

    # 3. Sprawdzenie Qdrant: punkt został usunięty przez kompensację wektora
    assert qdrant.get_by_key("test_topic") is None
    assert qdrant.get_by_key("kv:test_topic") is None


# ==============================================================================
# WEKTOR II: Ciągłość Metryczna Fuzji i Stabilizacja Entropijna
# ==============================================================================

def test_wektor_ii_fusion_continuity_around_threshold():
    """
    Weryfikuje ciągłość gładkiej funkcji fuzji C^1 w otoczeniu punktu S_dense = 0.75:
    brak skokowej bariery delta Diraca.
    """
    fusion = StaticAnchorDbsfFusion()
    points = [0.7499, 0.7500, 0.7501]
    scores = []
    
    for s_dense in points:
        cand = {
            "key": f"c_{s_dense}",
            "subject": "test_concept",
            "object": "daemon_probe",
            "vector_score": s_dense,
            "bm25_score": 0.0,
            "graph_score": 0.0,
            "is_daemon_no_lex": True,
            "metadata": {"is_daemon_no_lex": True},
        }
        fusion.score(query="test query", candidates=[cand])
        scores.append(cand["score"])

    # Sprawdzamy monotoniczność i małą deltę (brak skoku dyskretnego rzędu > 0.05)
    assert scores[0] <= scores[1] <= scores[2]
    diff1 = abs(scores[1] - scores[0])
    diff2 = abs(scores[2] - scores[1])
    assert diff1 < 0.01, f"Skok fuzji na progu: {diff1}"
    assert diff2 < 0.01, f"Skok fuzji na progu: {diff2}"


def test_wektor_ii_entropy_small_sample_asymptotics():
    """
    Testuje filtr entropijny dla małych prób (K=2):
    Dla niemal równomiernego rozkładu p=[0.500001, 0.499999], H_eff > 0.95,
    co zapobiega fałszywemu odrzuceniu kandydata.
    """
    entropy_filter = EntropySalienceFilter(threshold=0.80)
    
    # Tworzymy 2 kandydatów z niemal identycznymi wynikami
    cand1 = {
        "key": "c1",
        "subject": "Entity One",
        "object": "detail",
        "score": 0.500001,
        "dense_score": 0.5,
        "sparse_score": 0.5,
        "graph_score": 0.0,
    }
    cand2 = {
        "key": "c2",
        "subject": "Entity Two",
        "object": "detail",
        "score": 0.499999,
        "dense_score": 0.5,
        "sparse_score": 0.5,
        "graph_score": 0.0,
    }
    
    scores = [cand1["score"], cand2["score"]]
    h_norm = entropy_filter.compute_normalized_entropy(scores)
    assert h_norm > 0.95, f"Znormalizowana entropia dla K=2 powinna być > 0.95, otrzymano: {h_norm}"

    # Zgodnie z naprawą, dla K=2 h_eff = h_norm, a cliff ani próg salience cliff nie ucinają kandydatów
    filtered, eff_thr = entropy_filter.filter([cand1, cand2], base_threshold=0.30)
    assert len(filtered) == 2


# ==============================================================================
# WEKTOR III: Likwidacja Membrany Asocjacyjnej i Cienia Incydencji (PPR)
# ==============================================================================

def test_wektor_iii_associative_bridging_and_laplace_damping():
    """
    1. Brak nasion frazowych, lecz obecność pasażu semantycznego o wysokim podobieństwie
       aktywuje encje przez relację CONTAINS i produkuje kandydatów PPR.
    2. Węzeł o 30 incydentnych krawędziach zachowuje wszystkie 30 krawędzi (brak cięcia do 8)
       oraz stosuje Laplace degree damping.
    """
    # 1. Asocjacyjne mostkowanie bez phrase_seeds
    mock_nx = MagicMock()
    # Mock grafu NetworkX zwracający krawędzie out_edges dla pasażu
    mock_nx.out_edges.return_value = [
        ("doc_passage_1", "RustMemorySafety", {"relation": "CONTAINS", "weight": 1.0})
    ]
    mock_nx.in_edges.return_value = []

    passage_seeds = {"doc_passage_1": 0.90}
    phrase_seeds = {}  # Puste nasiona frazowe

    expanded = expand_associative_seeds_2hop(
        nx_graph=mock_nx,
        phrase_seeds=phrase_seeds,
        passage_seeds=passage_seeds,
        max_expanded=24,
    )

    assert "RustMemorySafety" in expanded
    assert expanded["RustMemorySafety"] > 0.0

    # 2. HippoRAG2PPR zachowuje ponad 8 krawędzi węzła i aplikuje Laplace degree damping
    import networkx as nx
    g = nx.DiGraph()
    g.add_node("RootNode", type="entity")
    for i in range(30):
        g.add_node(f"Neighbor_{i}", type="entity")
        g.add_edge("RootNode", f"Neighbor_{i}", predicate="CONNECTED", confidence=1.0)

    ppr = HippoRAG2PPR()
    ppr._build_graph_structures(g)

    root_edges = [e for e in ppr._cached_entity_rel_edges if e[1] == "RootNode"]
    assert len(root_edges) == 30, f"Obcięto krawędzie do {len(root_edges)} zamiast 30!"

    deg_root = g.degree("RootNode")
    deg_nbr = g.degree("Neighbor_0")
    expected_w = 1.0 / math.sqrt(float(deg_root * deg_nbr))
    assert abs(root_edges[0][4] - expected_w) < 1e-4, f"Niepoprawna waga Laplace'a: {root_edges[0][4]} != {expected_w}"


# ==============================================================================
# WEKTOR IV: Głęboka Niezmienniczość RAM i Atomowy Read-Through
# ==============================================================================

def test_wektor_iv_ram_immutability_and_hot_buffer_lock():
    """
    1. pack_knapsack_01 z opcją compact=True nie modyfikuje obiektów wejściowych (deep immutability).
    2. HotBufferPolicy posiada reentrantną blokadę threading.RLock() gwarantującą atomowy read-through.
    """
    original_metadata = {"steps": [1, 2, 3], "tags": {"priority": "high"}}
    cand = {
        "key": "c_immut",
        "subject": "Mutable Candidate",
        "object": "A" * 300,
        "metadata": copy.deepcopy(original_metadata),
    }

    candidates = [cand, copy.deepcopy(cand), copy.deepcopy(cand)]
    packed = pack_knapsack_01(candidates, budget=20, compact=True)

    # Oryginalny kandydat nie powinien mieć zmodyfikowanych metadanych ani skróconego tekstu
    assert cand["metadata"] == original_metadata
    assert cand["metadata"]["steps"] == [1, 2, 3]
    assert len(cand["object"]) == 300, "Oryginalny obiekt w liście wejściowej został zmutowany!"

    # Modyfikacja obiektu spakowanego nie wpływa na wejściowy
    if packed:
        if "steps" in packed[0].get("metadata", {}):
            packed[0]["metadata"]["steps"].append(4)
            assert cand["metadata"]["steps"] == [1, 2, 3], "Wyciek mutowalności RAM między knapsack a kandydatem wejściowym!"

    # Weryfikacja blokady HotBufferPolicy
    hot_buffer = HotBufferPolicy()
    assert hasattr(hot_buffer, "_lock")
    assert isinstance(hot_buffer._lock, type(threading.RLock()))

    # Weryfikacja współbieżnej ochrony
    with hot_buffer._lock:
        hot_buffer.append({"key": "k1", "value": "v1"})
        active = hot_buffer.get_active()
        assert len(active) == 1

