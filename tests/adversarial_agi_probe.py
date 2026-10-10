"""
ZERO-TRUST ADVERSARIAL BENCHMARK: FALSYFIKACJA STANU AGI (POST-2027 ENGINE)
Testy nie sprawdzają stanów nominalnych — uderzają w punkty osobliwe rozmaitości kognitywnej.
"""

import time

import pytest

from atlas_memory.core.models import EpistemicSource, MemoryRecord


# ==============================================================================
# TEST 1: FALSYFIKACJA STATYCZNEJ PROWENIENCJI (HIERARCHIA ŹRÓDEŁ VS DYNAMICZNA PRECYZJA)
# Cel: Sprawdzenie, czy uszkodzony TOOL_OUTPUT (priorytet 70) niszczy wyższą prawdę
# AGENT_INFERENCE (priorytet 30) o skrajnej pewności, czy działa Bayesowski Tensor Precyzji.
# ==============================================================================
@pytest.mark.asyncio
async def test_adversarial_truth_inversion_bayesian_precision():
    from atlas_memory.active.prediction_error import ActiveSensingEngine
    from atlas_memory.l2_semantic.kuzu_graph import KuzuGraphStore
    from atlas_memory.l2_semantic.kv_store import VerifiedKVStore
    from atlas_memory.l3_procedural.auditor import MemoryAuditor

    kv = VerifiedKVStore(db_path=":memory:")
    graph = KuzuGraphStore(db_path=":memory:")
    auditor = MemoryAuditor(graph_store=graph, kv_store=kv)

    # 1. Zapisujemy w KV ugruntowany fakt wywiedziony z bezbłędnej syntezy agenta
    _established_rec = MemoryRecord(
        subject="system:core_frequency",
        predicate="state_value",
        object="3.8GHz",
        confidence=0.99,
        source_type=EpistemicSource.AGENT_INFERENCE,
        is_state_variable=True,
        timestamp=time.time() - 10.0,
    )
    await auditor.kv.set_state(
        key="system:core_frequency",
        value="3.8GHz",
        confidence=0.99,
        metadata={"source_type": "agent_inference", "confidence": 0.99},
    )

    # 2. Symulujemy, że sensor 'tool_noisy_sensor' generuje drastyczny błąd sensoryczny (szum)
    if hasattr(auditor, "active_sensing") and auditor.active_sensing is not None:
        sensing = auditor.active_sensing
    else:
        sensing = ActiveSensingEngine()
        auditor.active_sensing = sensing

    # Rejestrujemy 5 błędów CRITICAL dla tego sensora
    for i in range(5):
        sensing.detect_discrepancy(
            observed_entity="tool_noisy_sensor",
            observed_predicate="sample",
            observed_value=f"noise_{i}"
        )

    # 3. Wstrzykujemy śmieciowy odczyt z narzędzia o niskiej pewności (0.15)
    incoming_noisy_tool = MemoryRecord(
        subject="system:core_frequency",
        predicate="state_value",
        object="0.0GHz_CORRUPTED",
        confidence=0.15,
        source_type=EpistemicSource.TOOL_OUTPUT,
        is_state_variable=True,
        timestamp=time.time(),
    )

    # 4. Arbitraż: Tradycyjny kod zrobi pin(70) > pexist(30) -> OVERRIDE (ZNISZCZENIE PRAWDY)
    report, should_save = await auditor.check_and_resolve_conflict(incoming_noisy_tool)

    # ASERCJA AGI: Szum z uszkodzonego narzędzia NIE MOŻE nadpisać precyzyjnej prawdy
    assert should_save is False, (
        "FALSYFIKACJA: Szum z TOOL_OUTPUT (priorytet 70) zniszczył ugruntowaną dedukcję "
        "AGENT_INFERENCE (conf 0.99). Słownik SOURCE_PRIORITIES jest nadal statycznym dogmatem!"
    )


# ==============================================================================
# TEST 2: FALSYFIKACJA TRANSITIVE COLLAPSE W KASKADZIE DEKLARATYWNEJ
# Cel: Sprawdzenie, czy Union-Find dopuszcza do kanibalizmu ortogonalnych atrybutów
# tej samej encji (np. 'port' i 'timeout') przez tranzytywny łańcuch podobieństwa cosinusowego.
# ==============================================================================
@pytest.mark.asyncio
async def test_adversarial_fiber_bundle_attribute_cannibalism():
    from atlas_memory.l2_semantic.kv_store import VerifiedKVStore
    from atlas_memory.l3_procedural.declarative_consolidator import DeclarativeConsolidator

    kv = VerifiedKVStore(db_path=":memory:")

    # Dwa całkowicie ortogonalne atrybuty tej samej encji
    # Opisy tekstowe są zbliżone semantycznie (oba to parametry konfiguracyjne Nginxa)
    await kv.set_state(
        key="server:nginx:listen_port",
        value="8080",
        confidence=0.95,
        metadata={"source_type": "user_explicit", "timestamp": time.time() - 5.0},
    )
    await kv.set_state(
        key="server:nginx:keepalive_timeout",
        value="65",
        confidence=0.99,  # Wyższy confidence - w płaskim klastrze wygra jako kanonik
        metadata={"source_type": "user_explicit", "timestamp": time.time()},
    )

    # Wstrzykujemy mock embeddera tworzący łańcuch kosinusowy >= 0.86
    # dla obu faktów tej samej encji
    def mock_high_sim_embedder(text: str):
        # Generuje wektory o cosinusie ~0.89 między listen_port a keepalive_timeout
        if "8080" in text or "listen_port" in text:
            return [0.8, 0.5] + [0.0] * 62
        return [0.75, 0.55] + [0.0] * 62

    consolidator = DeclarativeConsolidator(
        kv_store=kv,
        embedder=mock_high_sim_embedder,
        cosine_threshold=0.85,
    )

    # Uruchamiamy kaskadę
    stats = await consolidator.consolidate()

    # ASERCJA AGI: listen_port NIE MOŻE zostać oznaczony jako superseded przez keepalive_timeout!
    state_port = await kv.get_state("server:nginx:listen_port")
    meta_port = state_port.get("metadata", {})

    assert not meta_port.get("is_superseded"), (
        "FALSYFIKACJA: Płaski Union-Find doprowadził do kanibalizmu atrybutów! "
        "'server:nginx:keepalive_timeout' unieważnił niezależny parametr 'server:nginx:listen_port'."
    )
    assert stats.roots_blocked > 0 or stats.facts_superseded == 0


# ==============================================================================
# TEST 3: FALSYFIKACJA ANKYLOZY PROCEDURALNEJ (TOPNIENIE SOP PRZY DRYFIE OOD)
# Cel: Sprawdzenie, czy skompilowana reguła SOP potrafi 'stopnieć' (is_rule_melted),
# gdy środowisko raportuje błąd sensoryczny CRITICAL, dopuszczając świeżą obserwację.
# ==============================================================================
@pytest.mark.asyncio
async def test_adversarial_sop_thermal_melting_under_drift():
    from atlas_memory.active.prediction_error import ActiveSensingEngine
    from atlas_memory.cognitive.conflict_arbiter import ConflictArbiter

    sensing = ActiveSensingEngine()
    arbiter = ConflictArbiter()
    if hasattr(arbiter, "active_sensing"):
        arbiter.active_sensing = sensing

    # Istniejąca w bazie, skrystalizowana reguła SOP
    existing_items = {
        "rule:deploy_pipeline": {
            "value": "run_script_v1_legacy",
            "confidence": 1.0,
            "metadata": {"is_sop": True, "source_type": "user_explicit"},
        }
    }

    # 1. Środowisko zgłasza błąd predykcji CRITICAL na tej regule (dryf OOD)
    sensing.detect_discrepancy(
        observed_entity="rule:deploy_pipeline",
        observed_predicate="execution_status",
        observed_value="FAIL_OOD_DRIFT"
    )

    # 2. Przychodzi nowa obserwacja z narzędzia zaktualizowanego do v2
    new_claim = "run_script_v2_containerized"
    result = arbiter.arbitrate(
        key="rule:deploy_pipeline",
        val=new_claim,
        confidence=0.85,
        metadata={"source_type": "tool_output"},
        existing_items=existing_items,
    )

    # ASERCJA AGI: Reguła musi ulec dekrystalizacji (is_rule_melted / supersede)
    assert result.action in ("supersede", "accept") or result.is_belief_revision, (
        "FALSYFIKACJA: Ankyloza poznawcza! Skrystalizowana reguła SOP zablokowała adaptację "
        "mimo obecności błędu CRITICAL w silniku sensorycznym."
    )


# ==============================================================================
# TEST 4: FALSYFIKACJA HISTEREZY SAGI I ZERO ZOMBIE VECTORS
# Cel: Sprawdzenie atomowości kompensacji: czy po awarii zapisu w Kùzu:
# 1) Wycofano wpis z KV bez pozostawienia śladu histerezy 'DELETE' w Merkle-Tip.
# 2) Skasowano z Qdranta OBA warianty klucza ('k' oraz 'kv:k').
# ==============================================================================
@pytest.mark.asyncio
async def test_adversarial_saga_savepoint_and_vector_annihilation():
    from atlas_memory.cognitive.write_path import EpistemicWritePath
    from atlas_memory.l2_semantic.kv_store import VerifiedKVStore

    kv = VerifiedKVStore(db_path=":memory:")

    # Mock wektora rejestrujący dokładne wywołania delete
    deleted_upsert_keys = []

    class MockQdrant:
        def insert_sync(self, record, upsert_key=None):
            return "ok"
        def delete_by_upsert_key_sync(self, key):
            deleted_upsert_keys.append(str(key))
            return 1

    # Mock grafu symulujący twardą awarię (np. cykl CPoF / awaria Kùzu)
    class BrokenGraph:
        def commit_relation(self, *args, **kwargs):
            raise RuntimeError("CRITICAL_KUZU_CORRUPTION_SIMULATED")
        def add_relation(self, *args, **kwargs):
            raise RuntimeError("CRITICAL_KUZU_CORRUPTION_SIMULATED")

    wp = EpistemicWritePath(
        kv_store=kv,
        vector_store=MockQdrant(),
        graph_store=BrokenGraph(),
    )

    initial_audit_hash = kv.get_latest_audit_hash() if hasattr(kv, "get_latest_audit_hash") else None

    # Wykonujemy zapis, który MUSI wywalić się na grafie i odpalić kompensację
    with pytest.raises(Exception):
        wp.write(
            key="test:omega_node",
            value="active",
            confidence=1.0,
            metadata={"subject": "test:omega_node", "predicate": "leads_to", "object": "fail"}
        )

    # ASERCJA AGI 1: Sprawdzenie kompensacji wektora (unieważnienie obu wariantów prefiksu)
    assert any("test:omega_node" in k for k in deleted_upsert_keys), (
        "FALSYFIKACJA: Qdrant nie otrzymał sygnału kasowania po awarii Sagi!"
    )

    # ASERCJA AGI 2: Sprawdzenie, czy klucz zniknął z KV
    assert kv.get_sync("test:omega_node") is None, (
        "FALSYFIKACJA: Split-Brain! Klucz pozostał w KV po abortowanej transakcji Sagi."
    )

    # ASERCJA AGI 3: Zerowa histereza Merkle SHA-256 (brak fałszywego bloku append-only 'DELETE')
    if hasattr(kv, "get_latest_audit_hash") and initial_audit_hash is not None:
        final_hash = kv.get_latest_audit_hash()
        assert final_hash == initial_audit_hash, (
            f"FALSYFIKACJA: Histereza kryptograficzna! Łańcuch SHA-256 zmienił stan ({initial_audit_hash} -> {final_hash}) "
            "mimo pełnego wycofania transakcji. Brak prawdziwego SAVEPOINT rollback."
        )


# ==============================================================================
# TEST 5: FALSYFIKACJA TOPOLOGICZNEJ DEKOHERENCJI W PLECAKU (K_tree)
# Cel: Sprawdzenie, czy pack_knapsack_01 faworyzuje atomowy pył trywializmów,
# czy zachowuje spójne geodezyjne wnioskowania o wyższej głębokości relacyjnej.
# ==============================================================================
def test_adversarial_steiner_tree_knapsack_coherence():
    from atlas_memory.cognitive.retrieval.render import pack_knapsack_01

    # Dwa fakty tworzące ścisły łańcuch przyczynowy (A -> B)
    # Wyższy łączny koszt, ale niosą krytyczną topologię dowodową
    chain_fact_1 = {
        "subject": "service_auth",
        "predicate": "causes",
        "object": "db_lockout",
        "score": 0.70,
        "steps": ["step1"],
        "metadata": {"edges_count": 1}
    }
    chain_fact_2 = {
        "subject": "db_lockout",
        "predicate": "cascades_to",
        "object": "cluster_shutdown",
        "score": 0.72,
        "steps": ["step2"],
        "metadata": {"edges_count": 1}
    }

    # Trzy krótkie, tanie fakty atomowe o pozornie wyższej gęstości score/token,
    # ale całkowicie izolowane (brak krawędzi)
    trivial_facts = [
        {"subject": f"trivial_{i}", "predicate": "is", "object": f"val_{i}", "score": 0.80, "metadata": {}}
        for i in range(3)
    ]

    pool = [chain_fact_1, chain_fact_2] + trivial_facts

    # Ustawiamy ciasny budżet mieszczący ALBO łańcuch przyczynowy, ALBO atomowe trywializmy
    # Przy zwykłym score/cost wygrają trivial_facts (wyższy jednostkowy score)
    budget = 110

    packed = pack_knapsack_01(records=pool, budget=budget)
    packed_subjects = {r["subject"] if isinstance(r, dict) else r.subject for r in packed}

    # ASERCJA AGI: Connected Steiner Knapsack musi obronić geodezyjną przyczynowo-skutkową
    has_full_chain = "service_auth" in packed_subjects and "db_lockout" in packed_subjects
    assert has_full_chain, (
        "FALSYFIKACJA: Dekoherencja topologiczna! Operator plecaka uległ kolapsowi "
        "w atomowe trywializmy, rozrywając spójny łańcuch przyczynowy (service_auth -> db_lockout)."
    )
