"""
Integration Tests for Hermes Agent Plugin, MemoryProvider ABC & Mnemosyne Delegation.
"""
from __future__ import annotations

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

from atlas_memory.core.models import EpistemicSource
from atlas_memory.hermes.atlas_provider import (
    AtlasMemoryProvider,
    _parse_mnemosyne_context,
)
from atlas_memory.orchestrator import MemoryOrchestrator

MNEMOSYNE_RAW = """## Mnemosyne Context
 [2026-08-30T10:47:00] (importance 0.7, source task) [trust:stated] Napraw parser AtlasMemoryProvider w atlas_provider.py
 [2026-08-29T14:10] (importance 0.95, source canonical:procedure) [CANONICAL] Cognitive Pipeline cykl działa poprawnie od atlas-v4
 [2026-08-29T14:10] (importance 0.70, source task) Delegacja Cognitive Pipeline do izolowanego worktree
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
    assert "Cognitive Pipeline" in context
    assert "w atlas_provider.py" in context


def test_parser_extracts_3_records_with_metadata():
    """Parser wyciąga z każdej linii: timestamp, importance, source, content."""
    records = _parse_mnemosyne_context(MNEMOSYNE_RAW)

    assert len(records) == len([line for line in MNEMOSYNE_RAW.splitlines() if line.strip() and not line.startswith("##")])

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
    assert "Cognitive Pipeline" in rec1.object


def test_parser_handles_record_without_trust_tag_and_header():
    """Linia bez [TRUST] oraz nagłówek '##' muszą być obsłużone."""
    raw = (
        "## Mnemosyne Context\n"
        "[2026-08-29T14:10] (importance 0.70, source task) Delegacja bez tagu trust\n"
    )
    records = _parse_mnemosyne_context(raw)

    assert len(records) == len([line for line in raw.splitlines() if line.startswith("[")])
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

    assert len(records) == len([weird])
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


def test_atlas_provider_get_tool_schemas():
    from atlas_memory.hermes.tools import ATLAS_TOOL_SCHEMAS
    provider = AtlasMemoryProvider()
    schemas = provider.get_tool_schemas()
    assert {s["name"] for s in schemas} == {s["function"]["name"] for s in ATLAS_TOOL_SCHEMAS}
    assert len(schemas) == len(ATLAS_TOOL_SCHEMAS)
    for s in schemas:
        assert "description" in s
        assert "parameters" in s


def test_atlas_provider_handle_tool_call_in_process():
    import json

    from atlas_memory.engine import HybridMemoryEngine

    engine = HybridMemoryEngine.create_default(db_path=":memory:")
    orchestrator = MemoryOrchestrator(engine=engine)
    provider = AtlasMemoryProvider(orchestrator=orchestrator, socket_path="/tmp/non_existent_atlas.sock")

    # 1. stats (degraded in-process)
    stats_raw = provider.handle_tool_call("atlas_stats", {})
    stats = json.loads(stats_raw)
    assert stats["status"] == "ok"
    assert stats["mode"] == "in_process"
    assert stats["serving"] is False

    # 2. remember (in-process zero-drop fallback)
    rem_raw = provider.handle_tool_call("atlas_remember", {"key": "test:k", "value": "test:v"})
    rem = json.loads(rem_raw)
    assert rem["status"] == "ok"
    assert rem["mode"] == "in_process"

    # 3. recall (in-process zero-drop fallback)
    rec_raw = provider.handle_tool_call("atlas_recall", {"query": "test:k"})
    rec = json.loads(rec_raw)
    assert rec["status"] == "ok"
    assert rec["mode"] == "in_process"

    # 4. delete protected key
    del_prot_raw = provider.handle_tool_call("atlas_delete", {"key": "hermes:protected_key"})
    del_prot = json.loads(del_prot_raw)
    assert del_prot["status"] == "rejected"
    assert del_prot["reason"] == "protected_key"

    # 5. delete unprotected key (in-process zero-drop fallback)
    del_unprot_raw = provider.handle_tool_call("atlas_delete", {"key": "unprotected_key"})
    del_unprot = json.loads(del_unprot_raw)
    assert del_unprot["status"] == "ok"
    assert del_unprot["mode"] == "in_process"

    # 6. unknown tool
    with pytest.raises(NotImplementedError):
        provider.handle_tool_call("unknown_atlas_tool", {})


def test_hermes_memory_manager_tool_injection():
    import os
    import sys
    sys.path.insert(0, os.path.expanduser(os.environ.get("HERMES_AGENT_PATH", "~/.hermes/hermes-agent")))
    try:
        from agent.memory_manager import MemoryManager, inject_memory_provider_tools
    except ImportError:
        pytest.skip("Hermes agent package not installed or path not found")

    provider = AtlasMemoryProvider()
    mm = MemoryManager()
    mm.add_provider(provider)

    tool_names = mm.get_all_tool_names()
    assert "atlas_recall" in tool_names
    assert "atlas_remember" in tool_names
    assert "atlas_what_if" in tool_names
    assert "atlas_active_sensing" in tool_names
    assert "atlas_stats" in tool_names
    assert "atlas_get" in tool_names
    assert "atlas_delete" in tool_names
    assert "atlas_task_feedback" in tool_names
    assert "atlas_verify_chain" in tool_names
    assert "atlas_sleep_sync" in tool_names

    mock_agent = MagicMock()
    mock_agent._memory_manager = mm
    mock_agent.tools = []
    mock_agent.valid_tool_names = set()
    mock_agent.enabled_toolsets = ["memory"]
    mock_agent.disabled_toolsets = []
    mock_agent._agent_cfg = {}

    from atlas_memory.hermes.tools import ATLAS_TOOL_SCHEMAS

    injected = inject_memory_provider_tools(mock_agent)
    assert injected == len(ATLAS_TOOL_SCHEMAS)
    assert len(mock_agent.tools) == len(ATLAS_TOOL_SCHEMAS)
    assert "atlas_recall" in mock_agent.valid_tool_names
    assert "atlas_get" in mock_agent.valid_tool_names

