"""
Unit Tests for Hybrid Memory Engine, Memory Orchestrator & Distributed Sagas.
"""
from __future__ import annotations

import asyncio
import signal
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.extensions.decay_scorer import SalienceDecayEngine
from atlas_memory.models import EpistemicSource, MemoryRecord
from atlas_memory.orchestrator import MemoryOrchestrator
from atlas_memory.server.atlas_daemon import AtlasDaemon, clamp_rpc_limit


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

import pytest


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
    assert res_1["records"] == []

    # 2. Pytanie o NAS -> woła mnemosyne_recall()
    res_2 = await orchestrator.orchestrated_recall("Jaki jest IP mojego NAS?", session_id="test_session_1")
    assert res_2["retrieval_skipped"] is False
    assert mock_mne.recall_calls == 1
    assert "entity_nas_01" in res_2["context_block"]
    assert "records" in res_2
    assert len(res_2["records"]) == 1
    assert isinstance(res_2["records"][0], MemoryRecord)
    assert res_2["records"][0].subject == "entity_nas_01"
    assert "selected_facts" in res_2


def test_entity_detection_heuristics_t3():
    orchestrator = MemoryOrchestrator()

    # 1. Colon-separated entity keys
    should_run, entities, _ = orchestrator.should_retrieve("status test:hop:p1 teraz")
    assert should_run is True
    assert "test:hop:p1" in entities

    should_run, entities, _ = orchestrator.should_retrieve("podgląd chain:sep04:chain_a")
    assert should_run is True
    assert "chain:sep04:chain_a" in entities

    # 2. Alphanumeric node names with digits
    should_run, entities, _ = orchestrator.should_retrieve("relacja między p1 a p2")
    assert should_run is True
    assert "p1" in entities
    assert "p2" in entities

    should_run, entities, _ = orchestrator.should_retrieve("sprawdź node1 oraz port8080")
    assert should_run is True
    assert "node1" in entities
    assert "port8080" in entities

    # 3. Knowledge / retrieval intent keywords
    keywords = {"chain", "hop", "fakt", "dawka", "port", "token", "rule", "config", "szukaj", "znajdź", "pokaż", "sprawdź"}
    for kw in keywords:
        should_run, entities, _ = orchestrator.should_retrieve(f"jaka jest wartość {kw} w bazie?")
        assert should_run is True, f"Keyword '{kw}' did not trigger retrieval"
        assert kw in entities, f"Keyword '{kw}' not in detected_entities: {entities}"



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


def test_epistemic_knapsack_packing_optimality():
    """Weryfikacja że Epistemic Knapsack Packing maksymalizuje gęstość w zadanym budżecie tokenów."""
    orchestrator = MemoryOrchestrator()

    records = [
        (MemoryRecord(subject="auth", predicate="ip", object="10.0.0.1", source_type=EpistemicSource.USER_EXPLICIT), 1.0),
        (MemoryRecord(subject="log", predicate="msg", object="a" * 400, source_type=EpistemicSource.AGENT_INFERENCE), 0.5),
        (MemoryRecord(subject="db", predicate="port", object="5432", source_type=EpistemicSource.TOOL_OUTPUT), 0.85),
    ]

    res = orchestrator.apply_token_budget(records, max_tokens=50, strategy="knapsack")
    assert res["estimated_tokens"] <= 50
    selected_subjects = [f["subject"] for f in res["selected_facts"]]
    assert "auth" in selected_subjects
    assert "db" in selected_subjects
    assert "log" not in selected_subjects


@pytest.mark.asyncio
async def test_v37_daemon_trigger_sleep_consolidation_rpc(tmp_path: Path):
    """Test AtlasDaemon _handle_trigger_sleep_consolidation RPC method."""
    mock_engine = MagicMock()
    mock_engine.kv = MagicMock()
    mock_engine.auditor = MagicMock()
    mock_engine.auditor.run_sleep_cycle_consolidation = AsyncMock(
        return_value=MagicMock(consolidated_records=3)
    )
    mock_engine.trajectory_buffer = MagicMock(trajectories=[])

    daemon = AtlasDaemon(engine=mock_engine)

    res = await daemon._handle_trigger_sleep_consolidation({
        "skills_dir": str(tmp_path / "skills"),
    })

    assert res["status"] == "ok"
    assert res["consolidated_records"] == 3
    assert res["baked_sops_count"] == 0


def test_v38_vectorized_salience_decay_batch():
    """Test SalienceDecayEngine vectorized batch calculation vs scalar calculations."""
    engine = SalienceDecayEngine(decay_lambda=0.001)
    now = 10000.0

    records = [
        MemoryRecord(
            subject=f"entity_{i}",
            predicate="relates_to",
            object=f"target_{i}",
            confidence=0.9,
            importance_score=0.5 + (i % 5) * 0.1,
            timestamp=now - i * 100.0,
            access_count=i,
        )
        for i in range(100)
    ]

    sims = np.linspace(0.1, 0.9, 100, dtype=np.float32)

    scalar_scores = [
        engine.calculate_salience(rec, similarity_score=float(sims[i]), current_time=now)
        for i, rec in enumerate(records)
    ]

    batch_scores = engine.calculate_salience_batch(records, similarity_scores=sims, current_time=now)

    assert len(batch_scores) == 100
    np.testing.assert_allclose(batch_scores, scalar_scores, rtol=1e-5, atol=1e-5)


@pytest.mark.asyncio
async def test_prefetch_orchestrator_dict_and_list_handling(tmp_path: Path):
    """Verify prefetch handles orchestrator returning dict (records/selected_facts) or list."""
    sock_path = tmp_path / "test_atlas.sock"
    pid_path = tmp_path / "test_atlas.pid"

    class MockOrchestratorDict:
        async def orchestrated_recall(self, query: str, session_id: str = ""):
            return {
                "records": [
                    {"subject": "cluster_node", "predicate": "status", "object": "healthy", "veracity": 1.0}
                ],
                "selected_facts": [],
            }

    class MockOrchestratorSelectedFacts:
        async def orchestrated_recall(self, query: str, session_id: str = ""):
            return {
                "records": [],
                "selected_facts": [
                    {"subject": "backup_node", "predicate": "status", "object": "standby", "veracity": 0.9}
                ],
            }

    class MockOrchestratorList:
        async def orchestrated_recall(self, query: str, session_id: str = ""):
            return [
                {"subject": "gateway_node", "predicate": "status", "object": "active", "veracity": 0.95}
            ]

    daemon_dict = AtlasDaemon(socket_path=sock_path, pid_path=pid_path, orchestrator=MockOrchestratorDict())
    res_dict = await daemon_dict._handle_prefetch({"query": "cluster status"})
    assert any(r["subject"] == "cluster_node" for r in res_dict.get("records", []))

    daemon_sf = AtlasDaemon(socket_path=sock_path, pid_path=pid_path, orchestrator=MockOrchestratorSelectedFacts())
    res_sf = await daemon_sf._handle_prefetch({"query": "backup status"})
    assert any(r["subject"] == "backup_node" for r in res_sf.get("records", []))

    daemon_list = AtlasDaemon(socket_path=sock_path, pid_path=pid_path, orchestrator=MockOrchestratorList())
    res_list = await daemon_list._handle_prefetch({"query": "gateway status"})
    assert any(r["subject"] == "gateway_node" for r in res_list.get("records", []))


@pytest.mark.asyncio
async def test_orchestrator_vector_score_epistemic_ranking():
    """Test that records with high vector score outrank unrelated global user rules."""
    orchestrator = MemoryOrchestrator()

    unrelated_rule = MemoryRecord(
        subject="Prompt dla Delegated Agent (LOOP)",
        predicate="stated_memory",
        object="NA GÓRZE rola BOT w protokole COGNITIVE PIPELINE V43",
        importance_score=0.99,
        source_type=EpistemicSource.USER_EXPLICIT,
    )

    relevant_fact = MemoryRecord(
        subject="Peptide research",
        predicate="stated_memory",
        object="Rekomendacja = protokół BEZPIECZNEGO UŻYCIA po screeningu onkologicznym",
        importance_score=0.85,
        source_type=EpistemicSource.TOOL_OUTPUT,
        metadata={"vector_score": 0.88},
    )

    ranked = orchestrator.epistemic_rank(
        [unrelated_rule, relevant_fact],
        query="protokół bezpieczeństwa dawek peptydów",
    )

    assert len(ranked) == 2
    assert ranked[0][0].subject == "Peptide research"
    assert ranked[1][0].subject == "Prompt dla Delegated Agent (LOOP)"


def test_sighup_signal_ignored():
    """Finding #6: SIGHUP is ignored so daemon survives parent terminal closure."""
    if hasattr(signal, "SIGHUP"):
        daemon = AtlasDaemon()
        daemon.register_signal_handlers()
        handler = signal.getsignal(signal.SIGHUP)
        assert handler == signal.SIG_IGN


@pytest.mark.asyncio
async def test_f5_delete_truthfulness_existing_vs_nonexistent(tmp_path: Path):
    """
    F5: Verifies truthful reporting in _handle_delete:
    returns deleted=True, found=True for existing keys,
    and deleted=False, found=False for non-existent or already-deleted keys.
    """
    db_path = str(tmp_path / "test_atlas_f5.db")
    engine = HybridMemoryEngine.create_default(
        db_path=db_path,
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )
    daemon = AtlasDaemon(engine=engine)

    engine.kv.set_sync("user:preference:lang", "pl")

    res_del1 = await daemon._handle_delete({"key": "user:preference:lang"})
    assert res_del1["status"] == "ok"
    assert res_del1["key"] == "user:preference:lang"
    assert res_del1["deleted"] is True
    assert res_del1["found"] is True

    res_del2 = await daemon._handle_delete({"key": "user:preference:lang"})
    assert res_del2["status"] == "ok"
    assert res_del2["key"] == "user:preference:lang"
    assert res_del2["deleted"] is False
    assert res_del2["found"] is False

    res_del3 = await daemon._handle_delete({"key": "nonexistent_key_9999"})
    assert res_del3["status"] == "ok"
    assert res_del3["key"] == "nonexistent_key_9999"
    assert res_del3["deleted"] is False
    assert res_del3["found"] is False


@pytest.mark.asyncio
async def test_f5_limit_clamping(tmp_path: Path):
    """
    F5: Verifies safe parsing and clamping of limit parameter to [1, 100].
    """
    assert clamp_rpc_limit(500) == 100
    assert clamp_rpc_limit(-10) == 1
    assert clamp_rpc_limit(0) == 1
    assert clamp_rpc_limit("50") == 50
    assert clamp_rpc_limit("invalid", default=15) == 15
    assert clamp_rpc_limit(None, default=15) == 15

    db_path = str(tmp_path / "test_atlas_clamp.db")
    engine = HybridMemoryEngine.create_default(
        db_path=db_path,
        qdrant_location=str(tmp_path / "qdrant"),
        kuzu_path=str(tmp_path / "kuzu"),
    )
    daemon = AtlasDaemon(engine=engine)

    now = time.time()
    for i in range(10):
        daemon._hot_kv_buffer.append({
            "key": f"item:{i}",
            "value": f"value_{i}",
            "timestamp": now,
            "metadata": {"session_id": "hermes_default"},
        })

    res_high = await daemon._handle_prefetch({
        "query": "",
        "session_id": "hermes_default",
        "limit": 500,
    })
    assert res_high["count"] == 10

    res_low = await daemon._handle_prefetch({
        "query": "",
        "session_id": "hermes_default",
        "limit": -5,
    })
    assert res_low["count"] == 1


@pytest.mark.asyncio
async def test_h13_commit_observation_rejects_empty_subject(tmp_path: Path):
    """
    H13 (F8): Verifies commit_observation rejects empty or whitespace-only subjects.
    """
    daemon = AtlasDaemon.create_default(
        atlas_dir=tmp_path / "atlas",
        socket_path=tmp_path / "daemon.sock",
        ingest_mnemosyne_on_start=False,
    )

    res_empty = await daemon._handle_commit_observation({"subject": "", "object": "val1"})
    assert res_empty["status"] == "error"
    assert res_empty["error"] == "empty_subject"

    res_ws = await daemon._handle_commit_observation({"subject": "   \t \n", "object": "val2"})
    assert res_ws["status"] == "error"
    assert res_ws["error"] == "empty_subject"

    res_none = await daemon._handle_commit_observation({"subject": None, "object": "val3"})
    assert res_none["status"] == "error"
    assert res_none["error"] == "empty_subject"


@pytest.mark.asyncio
async def test_h13_commit_observation_clamps_confidence(tmp_path: Path):
    """
    H13 (F8): Verifies commit_observation clamps confidence to [0.0, 1.0] and handles malformed values.
    """
    daemon = AtlasDaemon.create_default(
        atlas_dir=tmp_path / "atlas",
        socket_path=tmp_path / "daemon.sock",
        ingest_mnemosyne_on_start=False,
    )

    res_neg = await daemon._handle_commit_observation({
        "subject": "service_a",
        "predicate": "status",
        "object": "degraded",
        "confidence": -5.0,
    })
    assert res_neg["status"] == "ok"
    state_neg = await daemon.engine.kv.get_state("service_a:status")
    assert state_neg is not None
    assert state_neg["confidence"] == 0.0

    res_hi = await daemon._handle_commit_observation({
        "subject": "service_b",
        "predicate": "status",
        "object": "healthy",
        "confidence": 999.0,
    })
    assert res_hi["status"] == "ok"
    state_hi = await daemon.engine.kv.get_state("service_b:status")
    assert state_hi is not None
    assert state_hi["confidence"] == 1.0

    res_nan = await daemon._handle_commit_observation({
        "subject": "service_c",
        "predicate": "status",
        "object": "unknown",
        "confidence": "invalid_number",
    })
    assert res_nan["status"] == "ok"
    state_nan = await daemon.engine.kv.get_state("service_c:status")
    assert state_nan is not None
    assert state_nan["confidence"] == 1.0


