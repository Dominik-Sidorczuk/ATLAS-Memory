"""
Integration Tests for Hermes Agent Plugin, MemoryProvider ABC & Mnemosyne Delegation.
"""
from __future__ import annotations

import pytest

from atlas_memory.active.prediction_error import PredictionCheck
from atlas_memory.hermes_integration import HermesMemoryAdapter


@pytest.mark.asyncio
async def test_hermes_adapter_orchestrated_search_and_gate():
    adapter = HermesMemoryAdapter.create_default()

    # 1. Zwykłe powitanie -> skip
    res_1 = await adapter.orchestrated_search("Cześć, witaj!")
    assert res_1["retrieval_skipped"] is True
    assert res_1["matched_facts_count"] == 0

    # 2. Pytanie o NAS -> gate trigger
    res_2 = await adapter.orchestrated_search("Jaki jest IP mojego NAS?")
    assert res_2["retrieval_skipped"] is False
    assert "entity_nas_01" in res_2["canonical_entities"]


@pytest.mark.asyncio
async def test_hermes_adapter_ingest_and_cache_prefix():
    adapter = HermesMemoryAdapter.create_default()

    # 1. Test cache prefix
    prefix = adapter.get_cache_prefix(
        profile_state={"admin_user": "Dominik"},
        rules=["Rule: Fast execution"],
        model_name="deepseek-chat",
    )
    assert "Dominik" in prefix
    assert "Rule: Fast execution" in prefix

    # 2. Ingest
    await adapter.orchestrated_ingest(
        user_msg="Moja baza danych to PostgreSQL 16 na porcie 5432",
        agent_response="Zapamiętane.",
    )
    processed = await adapter.shadow_worker.process_all_pending()
    assert processed == 1

    # 3. Telemetry report
    rep = adapter.get_telemetry_report()
    assert rep["cache_stats"]["total_monitored_turns"] == 1


@pytest.mark.asyncio
async def test_hermes_adapter_what_if_and_telemetry():
    adapter = HermesMemoryAdapter.create_default()

    # Dodanie relacji do grafu
    from atlas_memory.models import MemoryRecord
    await adapter.orchestrator.engine.graph.add_record(MemoryRecord(
        subject="entity_nas_01",
        predicate="hosts",
        object="BackupVolume",
        confidence=0.95,
    ))

    # Symulacja what-if
    what_if = await adapter.what_if_analysis("entity_nas_01", "restart_array", depth=1)
    assert what_if["source_entity"] == "entity_nas_01"
    assert len(what_if["paths"]) >= 1
    assert what_if["paths"][0]["affected_target"] == "BackupVolume"

    # Rejestracja i sprawdzenie Active Sensing
    adapter.active_sensing.register_expectation(PredictionCheck(
        check_id="nas_temp",
        target_entity="entity_nas_01",
        expected_predicate="temperature_c",
        expected_value="42",
        tolerance=5.0,
    ))

    err = await adapter.process_telemetry_observation("entity_nas_01", "temperature_c", "65")
    assert err is not None
    assert err.severity == "CRITICAL"

import pytest

from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.hermes.prefix_guard import HermesSessionHook, PrefixCacheGuard
from atlas_memory.hermes.tools import create_hermes_tool_handlers


@pytest.mark.asyncio
async def test_hermes_tool_handlers():
    engine = HybridMemoryEngine.create_default(db_path=":memory:")
    handlers = create_hermes_tool_handlers(engine)

    # 1. Commit przez handler narzędzia Hermesa
    commit_res = await handlers["commit_observation"]({
        "subject": "mój NAS",  # Alias, który powinien ulec kanonizacji
        "predicate": "ip_address",
        "object": "192.168.1.200",
        "is_state_variable": True,
        "source_type": "user_explicit",
    })
    assert commit_res["status"] == "queued_for_validation"

    # Przetworzenie kolejki w silniku
    await engine.process_all_pending()

    # 2. Search przez handler narzędzia Hermesa
    search_res = await handlers["search_memory"]({
        "query": "NAS IP address",
        "entities": ["TrueNAS"],
    })
    assert "verified_state" in search_res
    assert "entity_nas_01" in search_res["verified_state"]
    assert search_res["verified_state"]["entity_nas_01"]["value"] == "192.168.1.200"

    await engine.kv.close()


def test_prefix_cache_guard_stability():
    verified_state = {
        "zeta_key": {"value": 123},
        "alpha_key": {"value": "first"},
    }
    rules = ["Rule B", "Rule A"]

    prefix_1 = PrefixCacheGuard.build_immutable_prefix(verified_state, system_rules=rules)
    prefix_2 = PrefixCacheGuard.build_immutable_prefix(verified_state, system_rules=rules)

    # Niezmienność i determinizm kolejności
    assert prefix_1 == prefix_2
    assert prefix_1.index("Rule A") < prefix_1.index("Rule B")
    assert prefix_1.index("alpha_key") < prefix_1.index("zeta_key")
    assert "KV-CACHE-PREFIX-GUARD" in prefix_1


@pytest.mark.asyncio
async def test_hermes_session_hook_lifecycle():
    engine = HybridMemoryEngine.create_default(db_path=":memory:")
    session_hook = HermesSessionHook(engine)

    # Ustawienie zmiennej w KV
    await engine.kv.set_state("entity_nas_01", "online")

    # Start sesji
    session_start = await session_hook.on_session_start(
        session_id="session_test_42",
        active_entities=["entity_nas_01"],
    )
    assert "prefix_system_prompt" in session_start
    assert "entity_nas_01" in session_start["prefix_system_prompt"]
    assert "session_test_42" in session_hook.active_sessions

    # Koniec sesji
    stats = await session_hook.on_session_end(
        session_id="session_test_42",
        trigger_sleep_cycle=True,
    )
    assert stats.duration_ms > 0.0
    assert "session_test_42" not in session_hook.active_sessions

    await engine.kv.close()

"""Test parsera formatu Mnemosyne Context w AtlasMemoryProvider (fix atlas-v5).

Regression z benchmarku 2026-08-30: AtlasMemoryProvider.prefetch() zwracał
0/20 trafień w Hermes runtime, mimo że MnemosyneMemoryProvider.prefetch()
zwracał realne dane (14/20 hitów). Root cause: krucha heurystyka w
atlas_provider.py (`":" in line or "- " in line`) gubiła rekordy, których
content nie zawiera dwukropka ani myślnika → records=[] →
{"skipped": True, "reason": "no_records_from_backend"} → return "".

Format linii Mnemosyne (1 linia = 1 rekord):
    [TIMESTAMP] (importance X, source Y) [TRUST] CONTENT
np.:
    [2026-08-30T10:47:00] (importance 0.7, source task) [trust:stated] treść...
"""
import pytest

from atlas_memory.hermes.atlas_provider import (
    AtlasMemoryProvider,
    _parse_mnemosyne_context,
)
from atlas_memory.models import EpistemicSource
from atlas_memory.orchestrator import MemoryOrchestrator

MNEMOSYNE_RAW = """## Mnemosyne Context
 [2026-08-30T10:47:00] (importance 0.7, source task) [trust:stated] Napraw parser AtlasMemoryProvider w atlas_provider.py
 [2026-08-29T14:10] (importance 0.95, source canonical:procedure) [CANONICAL] Loop Engineering cykl działa poprawnie od atlas-v4
 [2026-08-29T14:10] (importance 0.70, source task) Delegacja Loop Engineering do izolowanego worktree
"""


class _FakeMnemosyne:
    """Mock backendu: zwraca tekst w formacie Mnemosyne Context."""

    def __init__(self, raw: str):
        self._raw = raw

    def prefetch(self, query: str, session_id: str = "") -> str:
        return self._raw


def _make_provider(raw: str) -> AtlasMemoryProvider:
    provider = AtlasMemoryProvider(orchestrator=MemoryOrchestrator())
    provider._mnemosyne = _FakeMnemosyne(raw)
    return provider


def test_provider_prefetch_extracts_mnemosyne_hits():
    """AtlasMemoryProvider z mockowanym Mnemosyne wyciąga >0 MemoryRecords."""
    provider = _make_provider(MNEMOSYNE_RAW)

    # "gdzie" = keyword intent → przechodzi Retrieval Policy Gate
    context = provider.prefetch("gdzie naprawić parser atlas_provider", session_id="test")

    assert provider._last_prefetch is not None
    assert provider._last_prefetch["skipped"] is False
    assert provider._last_prefetch["count"] > 0
    assert provider._last_prefetch["source"] == "atlas_orchestrator"
    assert "Loop Engineering" in context
    assert "w atlas_provider.py" in context


def test_parser_extracts_3_records_with_metadata():
    """Parser wyciąga z każdej linii: timestamp, importance, source, content."""
    records = _parse_mnemosyne_context(MNEMOSYNE_RAW)

    assert len(records) == 3

    rec0 = records[0]
    assert rec0.predicate == "context"
    assert rec0.metadata["importance"] == pytest.approx(0.7)
    assert rec0.metadata["source"] == "task"
    assert rec0.metadata["timestamp"] == "2026-08-30T10:47:00"
    assert "parser" in rec0.object.lower()
    assert rec0.importance_score == pytest.approx(0.7)

    rec1 = records[1]
    assert rec1.metadata["importance"] == pytest.approx(0.95)
    assert rec1.metadata["source"] == "canonical:procedure"
    assert "Loop Engineering" in rec1.object


def test_parser_handles_record_without_trust_tag_and_header():
    """Linia bez [TRUST] oraz nagłówek '##' muszą być obsłużone."""
    raw = (
        "## Mnemosyne Context\n"
        "[2026-08-29T14:10] (importance 0.70, source task) Delegacja bez tagu trust\n"
    )
    records = _parse_mnemosyne_context(raw)

    assert len(records) == 1
    assert records[0].metadata["source"] == "task"
    assert "Delegacja bez tagu trust" in records[0].object
    assert records[0].source_type == EpistemicSource.EXTERNAL_DOC


def test_parser_empty_raw_returns_empty_list():
    """Pusty backend (brak hitów) → [] → prefetch zachowuje skip bez szumu."""
    assert _parse_mnemosyne_context("") == []
    assert _parse_mnemosyne_context("   \n  \n") == []


def test_parser_unrecognized_format_falls_back_to_single_record():
    """Nieznany format → fallback: cały raw jako jeden rekord EXTERNAL_DOC."""
    weird = "zupełnie inny format bez znaczników czasu"
    records = _parse_mnemosyne_context(weird, fallback_query="moje zapytanie")

    assert len(records) == 1
    assert records[0].source_type == EpistemicSource.EXTERNAL_DOC
    assert records[0].subject.startswith("moje zapytanie")
    assert weird in records[0].object
import inspect
from types import NoneType
from unittest.mock import MagicMock

import pytest


def test_atlas_provider_has_on_session_end():
    provider = AtlasMemoryProvider(orchestrator=MagicMock())
    assert hasattr(provider, "on_session_end") is True


def test_atlas_provider_on_session_end_callable():
    provider = AtlasMemoryProvider(orchestrator=MagicMock())
    assert callable(provider.on_session_end) is True
    res = provider.on_session_end([{"session_id": "test_sess"}, {}, {}])
    assert res is None


def test_atlas_provider_on_session_end_matches_abc_signature():
    sig = inspect.signature(AtlasMemoryProvider.on_session_end)
    params = list(sig.parameters.keys())
    assert "self" in params
    assert "messages" in params
    assert params[1] == "messages"
    assert sig.return_annotation is None or sig.return_annotation is NoneType or sig.return_annotation == "None"
