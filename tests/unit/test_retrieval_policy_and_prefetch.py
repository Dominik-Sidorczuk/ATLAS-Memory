"""
Tests verifying resolution of audit findings P1-P4 (V81 follow-up):
P1: Policy gate in provider/gate does not skip semantic turns as conversational.
P2: Continuation query without active session hot-buffer does NOT dump unrelated rules (0 noise tokens).
P3: Vector or graph hits without existence in KV store (orphans) are strictly rejected.
P4: session/end honestly reports consolidated_records > 0 when declarative cascade supersedes facts.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from atlas_memory.cognitive.retrieval.candidates import gather_candidates
from atlas_memory.cognitive.retrieval.gate import should_retrieve_query
from atlas_memory.cognitive.retrieval.prefetch import PrefetchEngine


def test_p1_semantic_turns_not_skipped():
    """P1: Natural user inputs are recognized as semantic turns, not conversational chit-chat."""
    should_run, ents, reason = should_retrieve_query("Wprowadziłem znaczące poprawki. W związku z tym...")
    assert should_run is True
    assert reason == "semantic_turn"

    should_run2, ents2, reason2 = should_retrieve_query("bug ledger")
    assert should_run2 is True
    assert reason2 == "semantic_turn"

    # Only casual greetings are skipped
    should_run3, _, reason3 = should_retrieve_query("Cześć, jak się dziś masz?")
    assert should_run3 is False
    assert reason3 == "conversational_turn_no_entity"


@pytest.mark.asyncio
async def test_p2_continuation_without_hot_buffer_is_clean_skip():
    """P2: Continuation query without hot-buffer items returns 0 records, never dumping unrelated rules."""
    engine = PrefetchEngine()

    class MockEngine:
        kv = None
        vector_store = None
        graph = None
        hot_buffer = None

    res = await engine.execute(
        {"query": "Kontynuuj działanie", "session_id": "sess_p2"},
        engine=MockEngine(),
    )
    assert res.get("skipped") is True
    assert res.get("count") == 0
    assert res.get("reason") == "continuation_prompt"
    assert res.get("records") == []
    assert res.get("context_block") == ""


@pytest.mark.asyncio
async def test_p3_vector_orphans_without_kv_are_filtered():
    """P3: Vector store hits whose keys do not exist in KV are discarded as orphans."""
    class DummyKV:
        def __init__(self, existing_keys):
            self._keys = set(existing_keys)

        def get_all_sync(self, **kwargs):
            return {k: {"value": f"val_{k}"} for k in self._keys}

        def get_sync(self, k):
            return {"value": f"val_{k}"} if k in self._keys else None

    class DummyVectorHit:
        def __init__(self, key, text, score=0.95):
            self.id = key
            self.text = text
            self.score = score
            self.payload = {"metadata": {"kv_key": key}, "record": {"subject": key, "object": text}}

    class DummyVectorStore:
        async def search(self, query, top_k=15):
            return [
                DummyVectorHit("valid_rule:git", "Valid rule in KV", score=0.95),
                DummyVectorHit("daemon 47780 death", "Orphaned test point", score=0.96),
                DummyVectorHit("hot-KV loss", "Orphaned test point 2", score=0.94),
            ]

    class DummyOrchestrator:
        async def orchestrated_recall(self, query, session_id=None):
            return {
                "records": [
                    {"subject": "kv:valid_rule:git", "object": "Valid rule in KV", "metadata": {}},
                    {"subject": "daemon 47780 death", "object": "Orphaned test point", "metadata": {}},
                    {"subject": "agi:probe:vr", "object": "Orphaned test point 3", "metadata": {}},
                ]
            }

    class MockEngine:
        kv = DummyKV(["valid_rule:git"])
        vector_store = DummyVectorStore()
        orchestrator = DummyOrchestrator()

    mock_engine = MockEngine()
    cands = await gather_candidates(
        query="test query",
        session_id="default",
        engine=mock_engine,
        orchestrator=mock_engine.orchestrator,
        enable_calibrated_thresholds=True,
    )

    cand_keys = [c["subject"] for c in cands]
    assert "valid_rule:git" in cand_keys
    assert "daemon 47780 death" not in cand_keys
    assert "hot-KV loss" not in cand_keys
    assert "agi:probe:vr" not in cand_keys

    # Also verify full prefetch pipeline does not admit orphans
    prefetch_engine = PrefetchEngine()
    pref_res = await prefetch_engine.execute(
        {"query": "test query", "session_id": "default"},
        engine=mock_engine,
        orchestrator=mock_engine.orchestrator,
    )
    admitted_keys = [r.get("subject") for r in pref_res.get("records", [])]
    assert "daemon 47780 death" not in admitted_keys
    assert "hot-KV loss" not in admitted_keys
    assert "agi:probe:vr" not in admitted_keys


@pytest.mark.asyncio
async def test_p4_session_end_honest_reporting():
    """P4: session/end reports status='ok' and honest consolidated_records count."""
    from atlas_memory.server.atlas_daemon import AtlasDaemon

    daemon = AtlasDaemon.__new__(AtlasDaemon)
    mock_engine = MagicMock()
    daemon.engine = mock_engine

    # Simulate hook returning 9 superseded facts
    class MockStats:
        consolidated_records = 9
        facts_superseded = 9
        pruned_stale_facts = 0
        status = "ok"

    mock_hook = MagicMock()
    mock_hook.on_session_end = AsyncMock(return_value=MockStats())

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("atlas_memory.hermes.prefix_guard.HermesSessionHook", lambda eng: mock_hook)
        res = await daemon._handle_session_end({"session_id": "test_p4", "trigger_sleep": True})

    assert res["status"] == "ok"
    assert res["reason"] == "consolidated"
    assert res["consolidated_records"] == 9
    assert res["facts_superseded"] == 9


@pytest.mark.asyncio
async def test_p9_entity_root_guard_prevents_false_cross_entity_supersession():
    """P9: Distinct entities with identical values (e.g. 'set') are NEVER falsely superseded."""
    from atlas_memory.cognitive.conflict_arbiter import entity_root
    from atlas_memory.l3_procedural.declarative_consolidator import CandidateFact, DeclarativeConsolidator

    # 1. entity_root contract
    assert entity_root("fact:cors:is") == "fact:cors"
    assert entity_root("fact:api_key:is") == "fact:api_key"
    assert entity_root("fact:cors:is") != entity_root("fact:api_key:is")
    assert entity_root("rule:obsidian:edit_method") == entity_root("rule:obsidian:allow_disk")
    assert entity_root("hermes:memory_md:Obsidian") != entity_root("hermes:memory_md:Crawl4AI")

    # 2. Declarative consolidation with identical short value "set"
    candidates = [
        CandidateFact(key="fact:cors:is", value="set", text="set", confidence=0.7, timestamp=100.0, metadata={}),
        CandidateFact(key="fact:api_key:is", value="set", text="set", confidence=0.7, timestamp=200.0, metadata={}),
    ]

    class FakeKV:
        def __init__(self):
            self.saved = {}

        async def put_state(self, key, value, confidence=1.0, metadata=None):
            self.saved[key] = (value, confidence, metadata)

    def mock_embed(text):
        return [1.0, 0.0]

    cons = DeclarativeConsolidator(kv_store=FakeKV(), cosine_threshold=0.5, embedder=mock_embed)
    pairs = cons.find_pairs(candidates, mock_embed)

    # Pairs must be empty because roots are different: fact:cors != fact:api_key
    assert pairs == []
    assert cons._blocked_roots == 1


# ---------------------------------------------------------------------------
# V100 F2 (BUG-CAL41): realna funkcja gate oceniajaca (gate.py:94) — granice progu
# MUSZA wiazac. Test wykonuje `evaluate_gate` z produkcyjnym fallbackiem progowym
# (engine=None / engine enabled=False) i odtwarza decyzje konsumenta tak jak robi to
# prefetch.py (`score >= gate.threshold`). Mutacja literala 0.50 -> 0.42 MUSI wywalic
# ten test (stary reliktem V99 wskazany w STATE.md jako luka).
# ---------------------------------------------------------------------------

def _gate_accepts(gate_decision, score: float) -> bool:
    """Produkcyjna semantyka bramki (prefetch.py): kandydat przechodzi, gdy score >= threshold."""
    return score >= gate_decision.threshold


class _DisabledCalibration:
    """Engine kalibracji z kalibratorem, ale enabled=False — bramka NIE wolno go uzyc."""

    enabled = False
    calibrator = object()

    def get_threshold(self, query_type: str, default: float) -> float:  # pragma: no cover
        raise AssertionError("gate nie wolno pytac o prog, gdy enabled=False")


def test_v100_f2_gate_threshold_boundaries_bind_decision_symbolic_and_prose():
    """V100 F2/CAL41: score tuz PONAD i tuz PONIZEJ 0.50 (symboliczny) / 0.25 (prose)
    odwraca decyzje bramki; wstrzykniety engine=None oraz enabled=False obie sciezki
    musza dawac ten sam produktowy fallback."""
    from atlas_memory.cognitive.retrieval.gate import evaluate_gate

    symbolic_query = "node:main"          # izomorf z produkcyjnymi identyfikatorami (':')
    prose_query = "jak dziala proces kalibracji w systemie"

    for engine in (None, _DisabledCalibration()):
        gate_sym = evaluate_gate(symbolic_query, [], calibration_engine=engine)
        assert gate_sym.threshold == 0.50, (
            f"prog symboliczny = {gate_sym.threshold} != 0.50 — literat fallbacku zostal zmieniony"
        )
        assert gate_sym.threshold_selection_reason == "no_candidate_fallback"
        # granica: tuz ponad / tuz ponizej — decyzja MUSI sie odwracac
        assert _gate_accepts(gate_sym, 0.50 + 1e-9) is True
        assert _gate_accepts(gate_sym, 0.50 - 1e-9) is False, (
            "mutacja progu 0.50 w dol (np. 0.42) musialaby tu wypic — decision nie moze sie odwracic"
        )

        gate_prose = evaluate_gate(prose_query, [], calibration_engine=engine)
        assert gate_prose.threshold == 0.25, (
            f"prog prose = {gate_prose.threshold} != 0.25 — literat fallbacku zostal zmieniony"
        )
        assert gate_prose.threshold_selection_reason == "no_candidate_fallback"
        assert _gate_accepts(gate_prose, 0.25 + 1e-9) is True
        assert _gate_accepts(gate_prose, 0.25 - 1e-9) is False
