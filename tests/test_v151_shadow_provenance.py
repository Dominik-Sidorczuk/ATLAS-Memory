"""V151 Wektor I / H1 — kontrakt proweniencji epistemicznej Shadow Workera.

Falsyfikuje hipotezę H1: `orchestrator.shadow_reconcile` (dawniej linia 134)
fałszował dowodzenie użytkownika — brak/invalid `source_type` w trójce z
ekstrakcji cieniowej dostawał fallback `EpistemicSource.USER_EXPLICIT`
(źródło najwyższej wiarygodności) oraz `is_state_variable=True` omijało
ConflictArbiter (zmienna stanu zapisywana bez arbitrażu).

Kontrakt V151:
  1. Fallback/claim 'user_explicit' w ścieżce cieniowej ZDEMOTOWANY do
     EpistemicSource.SHADOW_INFERENCE (LLM-a ekstrakcja nigdy nie jest
     bezpośrednim dowodzeniem użytkownika).
  2. 'tool_output' pozostaje 'tool_output' (nie nad-demotowujemy).
  3. Każda trójka z is_state_variable=True MUSI przejść ConflictArbiter
     (atrybut `orchestrator.arbiter`, domyślnie ConflictArbiter());
     werdykt reject/block NIE jest commitowany.
"""
from __future__ import annotations

import pytest

from atlas_memory.core.models import EpistemicSource
from atlas_memory.orchestrator import MemoryOrchestrator


class RecordingArbiter:
    """Stub ConflictArbitera rejestrujący wywołania .arbitrate(**kwargs)."""

    def __init__(self, action: str = "accept") -> None:
        self.calls: list[dict] = []
        self._action = action

    def arbitrate(self, **kwargs):
        from atlas_memory.cognitive.models import ArbitrationResult

        self.calls.append(kwargs)
        return ArbitrationResult(action=self._action, is_conflict=self._action != "accept")


class SpyEngine:
    """Fałszywy silnik: liczy commit_observation, nie dotyka dysku."""

    def __init__(self) -> None:
        self.observed: list = []
        self.kv = None

    async def commit_observation(self, record) -> bool:
        self.observed.append(record)
        return True


def _mk_orch(extractor=None, engine=None, arbiter=None):
    orch = MemoryOrchestrator(engine=engine, shadow_extractor=extractor)
    if arbiter is not None:
        orch.arbiter = arbiter
    return orch


def _triples():
    # Trójki ze struktury LISTY zwracanej przez shadow extractora:
    # brak source_type → musi być shadow_inference (NIE user_explicit!).
    return [
        {"subject": "IP_SerweraNAS", "predicate": "state_value", "object": "10.0.0.2",
         "is_state_variable": True, "confidence": 0.9},
        # jawny claim 'user_explicit' z LLM-a → democja do shadow_inference
        {"subject": "kolor", "predicate": "lubi", "object": "niebieski",
         "source_type": "user_explicit", "is_state_variable": False, "confidence": 0.7},
        # tool_output nie jest demowany (brak nad-interpretacji bariery)
        {"subject": "wynik", "predicate": "wartosc", "object": "15ms",
         "source_type": "tool_output", "is_state_variable": False, "confidence": 0.8},
    ]


@pytest.mark.asyncio
async def test_v151_1_shadow_reconcile_never_fakes_user_explicit():
    """H1: fallback/claim user_explicit w shadow_reconcile → shadow_inference."""
    async def extractor(user_msg: str, agent_response: str):
        return _triples()

    spy = RecordingArbiter(action="accept")
    engine = SpyEngine()
    orch = _mk_orch(extractor=extractor, engine=engine, arbiter=spy)

    committed = await orch.shadow_reconcile("Zmień IP SerweraNAS na 10.0.0.2", "OK")

    assert len(committed) == 3
    srcs = {r.subject: r.source_type for r in committed}
    # Kluczowa falsyfikacja H1: ŻADEN rekord nie ma user_explicit.
    assert EpistemicSource.USER_EXPLICIT not in set(srcs.values())
    # Brak source_type → shadow_inference (fallback V151, nie USER_EXPLICIT).
    assert srcs["IP_SerweraNAS"] == EpistemicSource.SHADOW_INFERENCE
    # Claim user_explicit z LLM → zdemowany do shadow_inference.
    assert srcs["kolor"] == EpistemicSource.SHADOW_INFERENCE
    # tool_output pozostaje tool_output.
    assert srcs["wynik"] == EpistemicSource.TOOL_OUTPUT
    # Warstwa serializacji: .value == 'shadow_inference'
    assert srcs["IP_SerweraNAS"].value == "shadow_inference"


@pytest.mark.asyncio
async def test_v151_2_state_variable_must_pass_conflict_arbiter():
    """H1: is_state_variable=True MUSI przejść ConflictArbiter przed commitem."""
    async def extractor(user_msg: str, agent_response: str):
        return _triples()

    spy = RecordingArbiter(action="accept")
    engine = SpyEngine()
    orch = _mk_orch(extractor=extractor, engine=engine, arbiter=spy)

    committed = await orch.shadow_reconcile("q", "a")

    state_recs = [r for r in committed if r.is_state_variable]
    assert len(state_recs) == 1
    # Arbiter wywołany DOKŁAZNIE dla zmiennej stanu (1 trójka = is_state_variable).
    assert len(spy.calls) == 1
    call = spy.calls[0]
    assert call["key"] == state_recs[0].canonical_entity_id
    assert call["val"] == "10.0.0.2"
    # Ślad arbitrażu na rekordzie (audytowalność werdyktu).
    assert state_recs[0].metadata.get("shadow_arbitration") == "accept"
    assert engine.observed and engine.observed[0] is state_recs[0]


@pytest.mark.asyncio
async def test_v151_3_arbiter_reject_blocks_shadow_state_commit():
    """H1: werdykt reject → zmienna stanu NIE trafia do silnika ani do committed."""
    async def extractor(user_msg: str, agent_response: str):
        return [_triples()[0]]

    spy = RecordingArbiter(action="reject")
    engine = SpyEngine()
    orch = _mk_orch(extractor=extractor, engine=engine, arbiter=spy)

    committed = await orch.shadow_reconcile("q", "a")

    assert committed == []
    assert engine.observed == []
    assert len(spy.calls) == 1
    assert orch.stats["shadow_facts_extracted"] == 0


@pytest.mark.asyncio
async def test_v151_4_heuristic_fallback_provenance_is_shadow_inference():
    """H1: ścieżka bez shadow extractora (regex fallback) też nie podaje się za user_explicit."""
    orch = _mk_orch(extractor=None, engine=SpyEngine(), arbiter=RecordingArbiter(action="accept"))

    committed = await orch.shadow_reconcile("Zmień port SerweraNAS na 8080", "Zmieniono")

    assert committed, "fallback regex powinien wyciągnć przynajmniej jedną zmienną stanu"
    for rec in committed:
        assert rec.source_type == EpistemicSource.SHADOW_INFERENCE
        assert rec.source_type != EpistemicSource.USER_EXPLICIT
