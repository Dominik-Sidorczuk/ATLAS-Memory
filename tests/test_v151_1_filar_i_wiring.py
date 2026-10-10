"""V151.1 FILAR I — ConversationalTurn/EpistemicAssertion + bramka uprawnień shadow.

Kontrakt:
  (1) shadow_reconcile: trójka z tematem-bedącym-aktem-mowy-konwersacyjnym
      (meta-pytanie / discourse garbage wg core.linguistics) NIE jest
      kanonizowana jako encja ani commitowana.
  (2) shadow_reconcile: zmienna stanu ze shadow, którą arbiter ROZSTRZYGNAŁ
      czysto (accept/supersede/promote/accept_conflict) →
      metadata.shadow_promotion='arbitration_promoted' i is_state_variable=True.
  (3) shadow_reconcile: werdykt z NIEROZSTRZYGNIĘTYM konfliktem (warn_conflict) →
      rekord zostaje ZWYKŁYM faktem (is_state_variable=False) z
      metadata.shadow_promotion='pending_arbitration' — NIE znika, ale traci
      uprawnienie state_variable.
  (4) engine.commit_observation: bramka state_variable_authority — shadow z
      is_state_variable=True bez 'arbitration_promoted' zostaje ZDEMOTEOWANY
      PRZED zapisem do write_path (KV widzi zwykły fakt).
"""
from __future__ import annotations

import time

import pytest

from atlas_memory.core.models import (
    EpistemicSource,
    MemoryRecord,
    SpeechAct,
    classify_speech_act,
    state_variable_authority,
)


class StubArbiter:
    """Arbiter-zwracający wskazany werdykt (bez konfliktu dla accept/supersede)."""

    def __init__(self, action: str) -> None:
        self._action = action
        self.calls: list[dict] = []

    def arbitrate(self, **kwargs):
        from atlas_memory.cognitive.models import ArbitrationResult

        self.calls.append(kwargs)
        return ArbitrationResult(
            action=self._action,
            is_conflict=self._action == "warn_conflict",
            is_belief_revision=self._action == "supersede",
        )


class SpyEngine:
    def __init__(self) -> None:
        self.observed: list = []
        self.kv = None

    async def commit_observation(self, record) -> bool:
        self.observed.append(record)
        return True


def _mk_orch(extractor, arbiter=None, engine=None):
    from atlas_memory.orchestrator import MemoryOrchestrator

    orch = MemoryOrchestrator(engine=engine or SpyEngine(), shadow_extractor=extractor)
    orch.arbiter = arbiter if arbiter is not None else StubArbiter("accept")
    return orch


def _state_triple(subject: str = "port_wejscia") -> dict:
    return {
        "subject": subject,
        "predicate": "state_value",
        "object": "8443",
        "source_type": EpistemicSource.SHADOW_INFERENCE.value,
        "confidence": 0.9,
        "is_state_variable": True,
    }


# ── (1) akt mowy: meta-pytanie nie staje się encją ────────────────────────────

def test_classify_speech_act_typology():
    assert classify_speech_act("jak_obecnie", "state_value", "x") == SpeechAct.CONVERSATIONAL_TURN
    assert classify_speech_act("port_wejscia", "state_value", "czy otwarty?") == SpeechAct.CONVERSATIONAL_TURN
    assert classify_speech_act("port_wejscia", "state_value", "8443") == SpeechAct.EPISTEMIC_ASSERTION


@pytest.mark.asyncio
async def test_conversational_turn_never_committed_by_shadow():
    """Trójka 'jak_obecnie' (akt konwersacyjny) NIE przechodzi kanonizacji i nie jest commitowany."""
    async def extractor(user_msg, agent_response):
        return [_state_triple("jak_obecnie"), _state_triple("port_wejscia")]

    orch = _mk_orch(extractor=extractor, arbiter=StubArbiter("accept"))
    committed = await orch.shadow_reconcile("jak obecnie?", "a")
    subjects = {r.subject for r in committed}
    assert "jak_obecnie" not in subjects
    assert "port_wejscia" in subjects


# ── (2)+(3) arbitraż podnosi / demotuje uprawnienie state_variable ─────────────

@pytest.mark.asyncio
async def test_shadow_state_promoted_gets_arbitration_promoted():
    """Werdykt czysty (accept) → is_state_variable=True + shadow_promotion='arbitration_promoted'."""
    async def extractor(user_msg, agent_response):
        return [_state_triple()]

    orch = _mk_orch(extractor=extractor, arbiter=StubArbiter("accept"))
    committed = await orch.shadow_reconcile("q", "a")
    assert len(committed) == 1
    rec = committed[0]
    assert rec.is_state_variable is True
    assert rec.metadata.get("shadow_promotion") == "arbitration_promoted"
    assert state_variable_authority(rec) is True


@pytest.mark.asyncio
async def test_shadow_state_unresolved_conflict_demoted_to_plain_fact():
    """Werdykt warn_conflict → rekord ZOSTAJE zwykłym faktem (is_state_variable=False),
    metadata.shadow_promotion='pending_arbitration' — ślad do arbitrażu, bez uprawnień."""
    async def extractor(user_msg, agent_response):
        return [_state_triple()]

    orch = _mk_orch(extractor=extractor, arbiter=StubArbiter("warn_conflict"))
    committed = await orch.shadow_reconcile("q", "a")
    assert len(committed) == 1
    rec = committed[0]
    assert rec.is_state_variable is False
    assert rec.metadata.get("shadow_promotion") == "pending_arbitration"
    assert state_variable_authority(rec) is False
    # ślad arbitrażu zachowany dla audytu
    assert rec.metadata.get("shadow_arbitration") == "warn_conflict"


@pytest.mark.asyncio
async def test_shadow_reject_still_dropped():
    """Regres: werdykt reject blokowany (istniejący kontrakt V151/H1)."""
    async def extractor(user_msg, agent_response):
        return [_state_triple()]

    orch = _mk_orch(extractor=extractor, arbiter=StubArbiter("reject"))
    committed = await orch.shadow_reconcile("q", "a")
    assert committed == []


# ── (4) silnik: bramka state_variable_authority przed write_path ──────────────

@pytest.mark.asyncio
async def test_engine_gate_demotes_unauthorised_shadow_state():
    """commit_observation z shadow state_variable bez 'arbitration_promoted' →
    rekord DEMOTEOWANY przed zapisem; write_path dostaje metadata.is_state_variable=False."""
    from atlas_memory.engine import HybridMemoryEngine

    engine = HybridMemoryEngine.create_default()
    captured: dict = {}

    class SpyWP:
        async def write_async(self, key, value, confidence, metadata, source_type, reason, vector_record):
            captured["metadata"] = dict(metadata)
            captured["record"] = vector_record
            return {"action": "written"}

    engine.write_path = SpyWP()

    sneaky = MemoryRecord(
        subject="port_wejscia",
        canonical_entity_id="port_wejscia",
        predicate="state_value",
        object="8443",
        confidence=0.9,
        source_type=EpistemicSource.SHADOW_INFERENCE,
        timestamp=time.time(),
        is_state_variable=True,  # shadow ROŚCZI na state_variable bez arbitrażu
        metadata={"shadow_promotion": "pending_arbitration"},
    )
    await engine.commit_observation(sneaky)
    assert sneaky.is_state_variable is False, "bramka silnika nie zdemotowała shadow bez uprawnień"
    assert captured.get("metadata", {}).get("is_state_variable") is False
    assert captured.get("metadata", {}).get("shadow_promotion") == "pending_arbitration"


@pytest.mark.asyncio
async def test_engine_gate_passes_promoted_and_explicit_sources():
    """Regres: shadow Z PROMOCJĄ oraz źródła nie-shadow (user_explicit) zachowują flagę."""
    from atlas_memory.engine import HybridMemoryEngine

    engine = HybridMemoryEngine.create_default()
    captured: list = []

    class SpyWP:
        async def write_async(self, key, value, confidence, metadata, source_type, reason, vector_record):
            captured.append(dict(metadata))
            return {"action": "written"}

    engine.write_path = SpyWP()

    promoted = MemoryRecord(
        subject="port_wejscia", canonical_entity_id="port_wejscia", predicate="state_value",
        object="8443", confidence=0.9, source_type=EpistemicSource.SHADOW_INFERENCE,
        timestamp=time.time(), is_state_variable=True,
        metadata={"shadow_promotion": "arbitration_promoted"},
    )
    await engine.commit_observation(promoted)
    assert promoted.is_state_variable is True
    assert captured[-1].get("is_state_variable") is True

    explicit = MemoryRecord(
        subject="kolumna_a", canonical_entity_id="kolumna_a", predicate="state_value",
        object="v1", confidence=1.0, source_type=EpistemicSource.USER_EXPLICIT,
        timestamp=time.time(), is_state_variable=True, metadata={},
    )
    await engine.commit_observation(explicit)
    assert explicit.is_state_variable is True
    assert captured[-1].get("is_state_variable") is True
