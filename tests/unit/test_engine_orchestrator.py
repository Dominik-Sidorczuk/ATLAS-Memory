"""
Unit Tests for Hybrid Memory Engine, Memory Orchestrator & Distributed Sagas.
"""
from __future__ import annotations

import asyncio

import pytest

from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.models import MemoryRecord


@pytest.mark.asyncio
async def test_hybrid_memory_engine_recall_and_commit():
    engine = HybridMemoryEngine.create_default(db_path=":memory:")

    # 1. Commit obserwacji
    r1 = MemoryRecord(
        subject="ServiceX",
        predicate="depends_on",
        object="ServiceY",
        confidence=0.98,
    )
    r2 = MemoryRecord(
        subject="ServiceX",
        predicate="is_active",
        object="true",
        confidence=1.0,
        is_state_variable=True,
    )

    await engine.commit_observation(r1)
    await engine.commit_observation(r2)

    # Przetwórz kolejkę audytora
    await engine.process_all_pending()

    # 2. Równoległe odpytanie recall
    result = await engine.recall(query="Service dependencies", active_entities=["ServiceX"])

    assert "semantic_context" in result
    assert "graph_topology" in result
    assert "verified_state" in result
    assert "retrieval_latency_ms" in result
    assert result["retrieval_latency_ms"] < 300.0  # Warunek architektury: 100-300 ms

    # Weryfikacja topologii
    assert result["graph_topology"]["matched_nodes_count"] >= 1
    canon_subj = engine.canonicalizer.canonicalize("ServiceX")
    assert canon_subj in result["graph_topology"]["nodes"]

    # Weryfikacja stanu
    assert canon_subj in result["verified_state"]
    assert result["verified_state"][canon_subj]["value"] == "true"

    await engine.kv.close()


@pytest.mark.asyncio
async def test_engine_background_worker_lifecycle():
    engine = HybridMemoryEngine.create_default(db_path=":memory:")
    worker = engine.start_worker()
    assert not worker.done()

    r = MemoryRecord(subject="Env", predicate="mode", object="production", is_state_variable=True)
    await engine.commit_observation(r)

    # Dajmy pętli chwilę na przetworzenie w tle
    await asyncio.sleep(0.15)

    canon_subj = engine.canonicalizer.canonicalize("Env")
    state = await engine.kv.get_state(canon_subj)
    assert state is not None
    assert state["value"] == "production"

    await engine.stop_worker()
    assert worker.done()
    await engine.kv.close()
import time

import pytest

from atlas_memory.models import EpistemicSource
from atlas_memory.orchestrator import MemoryOrchestrator


class MockMnemosyneClient:
    """Mock klienta Mnemosyne do testów jednostkowych."""

    def __init__(self):
        self.triples = []
        self.recall_calls = 0

    async def recall(self, query: str, active_entities=None):
        self.recall_calls += 1
        return {
            "graph_topology": {
                "relations": [
                    {"subject": "entity_nas_01", "predicate": "ip_address", "object": "192.168.1.100", "confidence": 1.0}
                ]
            },
            "semantic_context": []
        }

    async def triple_add(self, subject: str, predicate: str, object_: str, confidence: float, source: str, supersede: bool):
        self.triples.append({
            "subject": subject,
            "predicate": predicate,
            "object": object_,
            "confidence": confidence,
            "source": source,
            "supersede": supersede,
        })


@pytest.mark.asyncio
async def test_retrieval_policy_gate_with_mnemosyne_mock():
    mock_mne = MockMnemosyneClient()
    orchestrator = MemoryOrchestrator(mnemosyne_client=mock_mne)

    # 1. Zwykły czat -> nie woła mnemosyne_recall()
    res_1 = await orchestrator.orchestrated_recall("Cześć, jak leci?")
    assert res_1["retrieval_skipped"] is True
    assert mock_mne.recall_calls == 0

    # 2. Pytanie o NAS -> woła mnemosyne_recall()
    res_2 = await orchestrator.orchestrated_recall("Jaki jest IP mojego NAS?")
    assert res_2["retrieval_skipped"] is False
    assert mock_mne.recall_calls == 1
    assert "entity_nas_01" in res_2["context_block"]


@pytest.mark.asyncio
async def test_shadow_reconcile_calls_mnemosyne_triple_add_supersede():
    mock_mne = MockMnemosyneClient()
    orchestrator = MemoryOrchestrator(mnemosyne_client=mock_mne)

    # Ekstrakcja z tury i zapis do Mnemosyne
    records = await orchestrator.shadow_reconcile(
        user_msg="Mój NAS to 192.168.1.200",
        agent_response="Zanotowałem.",
    )
    assert len(records) >= 1
    assert len(mock_mne.triples) >= 1

    last_triple = mock_mne.triples[-1]
    assert last_triple["subject"] == "entity_nas_01"
    assert last_triple["object"] == "192.168.1.200"
    assert last_triple["supersede"] is True
    assert last_triple["source"] == "user_explicit"


def test_prune_stale_facts_salience_decay():
    orchestrator = MemoryOrchestrator()
    t0 = time.time()

    # Ważny fakt
    f_imp = MemoryRecord(
        subject="root_pwd",
        predicate="val",
        object="secret",
        importance_score=1.0,
        timestamp=t0 - 100000.0,
    )
    # Błahy stary fakt
    f_junk = MemoryRecord(
        subject="temp_note",
        predicate="note",
        object="test",
        importance_score=0.1,
        timestamp=t0 - 100000.0,
    )

    active, pruned = orchestrator.prune_stale_facts([f_imp, f_junk], threshold=0.20, current_time=t0)
    assert len(active) == 1
    assert active[0].subject == "root_pwd"
    assert len(pruned) == 1
    assert pruned[0].subject == "temp_note"
import pytest

from atlas_memory.l2_semantic.kuzu_graph import KuzuGraphStore
from atlas_memory.l2_semantic.kv_store import VerifiedKVStore
from atlas_memory.l2_semantic.qdrant_store import QdrantVectorStore
from atlas_memory.l3_procedural.auditor import MemoryAuditor


@pytest.mark.asyncio
async def test_saga_successful_transaction():
    kv = VerifiedKVStore(db_path=":memory:")
    graph = KuzuGraphStore(db_path=":memory:")
    vector = QdrantVectorStore(location=":memory:", dimension=64)
    auditor = MemoryAuditor(graph, kv, vector)

    rec = MemoryRecord(
        subject="PaymentService",
        predicate="depends_on",
        object="StripeAPI",
        confidence=1.0,
        is_state_variable=True,
    )

    success = await auditor.atomic_insert_with_saga(rec)
    assert success is True

    # Sprawdzenie w KV
    state = await kv.get_state("PaymentService")
    assert state is not None
    assert state["value"] == "StripeAPI"

    # Sprawdzenie w intencjach transakcyjnych
    dangling = await kv.get_dangling_intents()
    assert len(dangling) == 0, "Wszystkie intencje powinny mieć status COMMITTED"

    await kv.close()
    graph.close()


@pytest.mark.asyncio
async def test_saga_crash_recovery():
    kv = VerifiedKVStore(db_path=":memory:")
    graph = KuzuGraphStore(db_path=":memory:")
    auditor = MemoryAuditor(graph, kv)

    # Sztuczne utworzenie wiszącej intencji PENDING (symulacja nagłego zrestartowania procesu)
    rec_pending = MemoryRecord(
        subject="GhostService",
        predicate="status",
        object="zombie",
    )
    await kv.create_transaction_intent("tx_crash_999", rec_pending)
    await graph.add_record(rec_pending)

    dangling_before = await kv.get_dangling_intents()
    assert len(dangling_before) == 1

    # Uruchomienie procedury odzyskiwania stanu
    recovered = await auditor.recover_dangling_transactions()
    assert recovered == 1

    dangling_after = await kv.get_dangling_intents()
    assert len(dangling_after) == 0

    await kv.close()
    graph.close()

import pytest

from atlas_memory.extensions.canonicalizer import EntityCanonicalizer
from atlas_memory.extensions.compactor import ContextCompactor
from atlas_memory.extensions.decay_scorer import SalienceDecayEngine
from atlas_memory.extensions.epistemic import EpistemicCalibrator


def test_decay_scorer_formula_and_pruning():
    engine = SalienceDecayEngine(decay_lambda=0.1, prune_threshold=0.25)
    t0 = time.time()

    # Ważny fakt (I=1.0)
    critical_rec = MemoryRecord(
        subject="admin_key",
        predicate="val",
        object="secret",
        importance_score=1.0,
        timestamp=t0 - 1000.0,
    )
    assert engine.should_prune(critical_rec, current_time=t0) is False

    # Błahy fakt (I=0.1, stare t0)
    trivial_rec = MemoryRecord(
        subject="weather",
        predicate="temp",
        object="22C",
        importance_score=0.1,
        timestamp=t0 - 500.0,
    )
    assert engine.should_prune(trivial_rec, current_time=t0) is True

    # Zwiększenie liczby odpytań podnosi salience
    engine.record_access(trivial_rec, access_time=t0)
    score = engine.calculate_salience(trivial_rec, similarity_score=0.9, current_time=t0)
    assert score > 0.35


def test_entity_canonicalizer_aliases():
    canon = EntityCanonicalizer()

    # Domyślne encje
    assert canon.canonicalize("mój NAS") == "entity_nas_01"
    assert canon.canonicalize("TrueNAS") == "entity_nas_01"
    assert canon.canonicalize("Advantech") == "entity_nas_01"
    assert canon.canonicalize("Hermes") == "entity_agent_core"

    # Rejestracja nowej encji
    canon.register_entity(
        canonical_id="entity_db_pg",
        canonical_name="PostgreSQL Cluster",
        aliases=["baza", "postgres", "pg_main"],
    )
    assert canon.canonicalize("pg_main") == "entity_db_pg"
    assert canon.canonicalize("baza") == "entity_db_pg"


def test_context_compactor_window():
    compactor = ContextCompactor(session_window_size=3)

    assert compactor.add_interaction_turn("user", "Ustaw serwer to 192.168.1.50") is False
    assert compactor.add_interaction_turn("agent", "Zrozumiałem, serwer ustawiony.") is False
    # Trzecia tura wyzwala kompakcję
    assert compactor.add_interaction_turn("user", "Zmień port na 8080") is True

    level = compactor.compact_working_window(episode_id="ep_001")
    assert level.source_items_count == 3
    assert len(level.extracted_facts) >= 1
    assert "192.168.1.50" in level.compressed_text


def test_epistemic_calibration_and_arbitration():
    calibrator = EpistemicCalibrator()

    rec_user = MemoryRecord(
        subject="host_ip",
        predicate="is",
        object="10.0.0.1",
        confidence=1.0,
        source_type=EpistemicSource.USER_EXPLICIT,
        timestamp=100.0,
    )
    rec_agent_guess = MemoryRecord(
        subject="host_ip",
        predicate="is",
        object="10.0.0.99",
        confidence=1.0,
        source_type=EpistemicSource.AGENT_INFERENCE,
        timestamp=200.0,  # Nowszy timestamp, ale gorsze źródło
    )

    # Użytkownik musi wygrać z inferencją agenta mimo starszego timestampu
    override, reason = calibrator.arbitrate_conflict(rec_agent_guess, rec_user)
    assert override is True
    assert "epistemic_override" in reason

    # Inferencja agenta nie może unieważnić deklaracji usera
    override2, reason2 = calibrator.arbitrate_conflict(rec_user, rec_agent_guess)
    assert override2 is False
    assert "epistemic_rejected" in reason2

def test_public_api_imports() -> None:
    """Weryfikacja że wszystkie publiczne klasy i funkcje V10-V23 są eksportowane z atlas_memory."""
    import atlas_memory

    symbols = [
        "HybridMemoryEngine",
        "MemoryOrchestrator",
        "MemoryRecord",
        "SyncCrypto",
        "VectorClock",
        "LWWElementSet",
        "DeltaCRDT",
        "GossipProtocol",
        "GossipTransport",
        "UDPGossipTransport",
        "InMemoryGossipTransport",
        "create_transport",
        "MIBQuantizer",
        "SIMDHamming",
        "QuantizationConfig",
        "QuantizedVector",
        "compile_sop_to_skill",
        "ASTSafetyScanner",
        "SafetyViolationError",
        "SleepBaker",
        "StandardProcedure",
        "AtlasDaemon",
        "AtlasDaemonClient",
    ]
    for name in symbols:
        sym = getattr(atlas_memory, name, None)
        assert sym is not None, f"Brak symbolu {name} w atlas_memory"
        assert callable(sym), f"Symbol {name} nie jest callable"

