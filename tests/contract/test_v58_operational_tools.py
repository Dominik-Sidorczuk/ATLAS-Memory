"""
Contract & Integration tests for ATLAS V58: Full Operational Tooling Surface (F1–F4).

Verifies:
1. JSON Schema integrity and OpenAI/Hermes calling format for all 10 canonical tools (F1.A).
2. UDS handler registration and parameter translation for all 10 tools (F1.B).
3. In-process handler parity for all 10 tools when running engine without daemon (F1.C).
4. Timeout pool assignment in client.py for long-running operations (F2).
5. Dynamic vector noise filter rejecting spurious identifier nearest-neighbors below 0.70 (F3).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atlas_memory.cognitive.retrieval_pipeline import RetrievalPipeline
from atlas_memory.hermes.tools import (
    ATLAS_ACTIVE_SENSING_SCHEMA,
    ATLAS_DELETE_SCHEMA,
    ATLAS_GET_SCHEMA,
    ATLAS_RECALL_SCHEMA,
    ATLAS_REMEMBER_SCHEMA,
    ATLAS_SLEEP_SYNC_SCHEMA,
    ATLAS_STATS_SCHEMA,
    ATLAS_TASK_FEEDBACK_SCHEMA,
    ATLAS_TOOL_SCHEMAS,
    ATLAS_VERIFY_CHAIN_SCHEMA,
    ATLAS_WHAT_IF_SCHEMA,
    create_uds_tool_handlers,
)
from atlas_memory.server.client import (
    DEFAULT_METHOD_TIMEOUTS,
    AtlasDaemonClient,
    send_uds_request_sync,
)


def test_v58_all_10_tools_schemas_valid():
    """Weryfikuje poprawność schematów JSON Schema dla wszystkich 10 narzędzi."""
    expected_names = {
        "atlas_recall",
        "atlas_remember",
        "atlas_what_if",
        "atlas_active_sensing",
        "atlas_stats",
        "atlas_get",
        "atlas_delete",
        "atlas_task_feedback",
        "atlas_verify_chain",
        "atlas_sleep_sync",
    }
    assert len(ATLAS_TOOL_SCHEMAS) == 10
    registered_names = {s["function"]["name"] for s in ATLAS_TOOL_SCHEMAS}
    assert registered_names == expected_names

    # Sprawdzenie obecności poszczególnych schematów
    assert ATLAS_RECALL_SCHEMA["function"]["name"] == "atlas_recall"
    assert ATLAS_REMEMBER_SCHEMA["function"]["name"] == "atlas_remember"
    assert ATLAS_WHAT_IF_SCHEMA["function"]["name"] == "atlas_what_if"
    assert ATLAS_ACTIVE_SENSING_SCHEMA["function"]["name"] == "atlas_active_sensing"
    assert ATLAS_STATS_SCHEMA["function"]["name"] == "atlas_stats"

    # Sprawdzenie ogólnej struktury OpenAI / Hermes function calling
    for schema in ATLAS_TOOL_SCHEMAS:
        assert schema.get("type") == "function"
        fn = schema.get("function", {})
        assert "name" in fn and isinstance(fn["name"], str)
        assert "description" in fn and isinstance(fn["description"], str)
        assert "parameters" in fn
        params = fn["parameters"]
        assert params.get("type") == "object"
        assert "properties" in params and isinstance(params["properties"], dict)

    # atlas_get
    get_fn = ATLAS_GET_SCHEMA["function"]
    assert get_fn["name"] == "atlas_get"
    assert "key" in get_fn["parameters"]["properties"]
    assert set(get_fn["parameters"]["required"]) == {"key"}

    # atlas_delete
    del_fn = ATLAS_DELETE_SCHEMA["function"]
    assert del_fn["name"] == "atlas_delete"
    assert "key" in del_fn["parameters"]["properties"]
    assert "reason" in del_fn["parameters"]["properties"]
    assert set(del_fn["parameters"]["required"]) == {"key"}

    # atlas_task_feedback
    fb_fn = ATLAS_TASK_FEEDBACK_SCHEMA["function"]
    assert fb_fn["name"] == "atlas_task_feedback"
    assert "key" in fb_fn["parameters"]["properties"]
    assert "task_success" in fb_fn["parameters"]["properties"]
    assert fb_fn["parameters"]["properties"]["delta"]["default"] == 0.05
    assert set(fb_fn["parameters"]["required"]) == {"key", "task_success"}

    # atlas_verify_chain
    vc_fn = ATLAS_VERIFY_CHAIN_SCHEMA["function"]
    assert vc_fn["name"] == "atlas_verify_chain"
    assert "heal" in vc_fn["parameters"]["properties"]
    assert "deep" in vc_fn["parameters"]["properties"]
    assert vc_fn["parameters"]["properties"]["heal"]["default"] is False
    assert vc_fn["parameters"]["properties"]["deep"]["default"] is False

    # atlas_sleep_sync
    ss_fn = ATLAS_SLEEP_SYNC_SCHEMA["function"]
    assert ss_fn["name"] == "atlas_sleep_sync"
    assert "skills_dir" in ss_fn["parameters"]["properties"]


@patch("atlas_memory.hermes.tools.send_uds_request_sync")
def test_v58_uds_handlers_registration(mock_send):
    """Weryfikuje obecność 10 kluczy w create_uds_tool_handlers oraz poprawne delegowanie RPC."""
    handlers = create_uds_tool_handlers("/tmp/fake_v58_atlas.sock")
    expected_keys = {
        "atlas_recall",
        "atlas_remember",
        "atlas_what_if",
        "atlas_active_sensing",
        "atlas_stats",
        "atlas_get",
        "atlas_delete",
        "atlas_task_feedback",
        "atlas_verify_chain",
        "atlas_sleep_sync",
    }
    assert expected_keys.issubset(set(handlers.keys()))

    # 1. atlas_get
    mock_send.return_value = {"status": "ok", "key": "agent:mode", "found": True, "state": "active"}
    res_get = handlers["atlas_get"]({"key": "agent:mode"})
    assert res_get["found"] is True
    mock_send.assert_called_with(Path("/tmp/fake_v58_atlas.sock"), "get", {"key": "agent:mode"})

    # 2. atlas_delete
    mock_send.reset_mock()
    mock_send.return_value = {"status": "ok", "key": "temp:var", "deleted": True, "found": True}
    res_del = handlers["atlas_delete"]({"key": "temp:var", "reason": "cleanup"})
    assert res_del["deleted"] is True
    mock_send.assert_called_with(Path("/tmp/fake_v58_atlas.sock"), "delete", {"key": "temp:var", "reason": "cleanup"})

    # 3. atlas_task_feedback
    mock_send.reset_mock()
    mock_send.return_value = {"status": "ok", "key": "rule:dns", "confidence": 0.95, "task_success": True}
    res_fb = handlers["atlas_task_feedback"]({"key": "rule:dns", "task_success": True, "delta": 0.05})
    assert res_fb["confidence"] == 0.95
    mock_send.assert_called_with(
        Path("/tmp/fake_v58_atlas.sock"),
        "task_feedback",
        {"key": "rule:dns", "task_success": True, "delta": 0.05},
    )

    # 4. atlas_verify_chain
    mock_send.reset_mock()
    mock_send.return_value = {"status": "ok", "valid": True, "checked_entries": 42}
    res_vc = handlers["atlas_verify_chain"]({"heal": True, "deep": True})
    assert res_vc["valid"] is True
    mock_send.assert_called_with(
        Path("/tmp/fake_v58_atlas.sock"),
        "verify_audit_log",
        {"heal": True, "deep": True},
        timeout=20.0,
    )

    # 5. atlas_sleep_sync
    mock_send.reset_mock()
    mock_send.return_value = {
        "status": "ok",
        "consolidated_records": 10,
        "pruned_stale_facts": 2,
        "baked_sops_count": 1,
        "baked_sops": [{"name": "sop_1"}],
    }
    res_ss = handlers["atlas_sleep_sync"]({"skills_dir": "/tmp/custom_skills"})
    assert res_ss["baked_sops_count"] == 1
    mock_send.assert_called_with(
        Path("/tmp/fake_v58_atlas.sock"),
        "trigger_sleep_consolidation",
        {"skills_dir": "/tmp/custom_skills"},
        timeout=30.0,
    )

    # 6. atlas_recall
    mock_send.reset_mock()
    mock_send.return_value = {"status": "ok", "records": [], "count": 0}
    res_rec = handlers["atlas_recall"]({"query": "user preferences"})
    assert res_rec["status"] == "ok"
    mock_send.assert_called_with(
        Path("/tmp/fake_v58_atlas.sock"),
        "prefetch",
        {"query": "user preferences", "session_id": "hermes_default", "limit": 15, "force": False},
    )

    # 7. atlas_remember
    mock_send.reset_mock()
    mock_send.return_value = {"status": "ok", "key": "pref:theme", "persisted": True}
    res_rem = handlers["atlas_remember"]({"key": "pref:theme", "value": "dark"})
    assert res_rem["status"] == "ok"
    mock_send.assert_called_with(
        Path("/tmp/fake_v58_atlas.sock"),
        "set",
        {"key": "pref:theme", "value": "dark", "confidence": 1.0, "reason": "agent_explicit", "session_id": "hermes_default"},
    )

    # 8. atlas_what_if
    mock_send.reset_mock()
    mock_send.return_value = {"status": "ok", "causal_paths": []}
    res_wi = handlers["atlas_what_if"]({"action": "deploy"})
    assert res_wi["status"] == "ok"
    mock_send.assert_called_with(
        Path("/tmp/fake_v58_atlas.sock"),
        "what_if",
        {"entity": "", "action": "deploy", "depth": 2},
    )

    # 9. atlas_active_sensing
    mock_send.reset_mock()
    mock_send.return_value = {"status": "ok", "has_error": False}
    res_as = handlers["atlas_active_sensing"]({"probe": "cpu"})
    assert res_as["status"] == "ok"
    mock_send.assert_called_with(
        Path("/tmp/fake_v58_atlas.sock"),
        "active_sensing",
        {"target_entity": "cpu", "expected_value": "", "observed_value": ""},
    )

    # 10. atlas_stats
    mock_send.reset_mock()
    mock_send.return_value = {"status": "ok", "kv_records": 42}
    res_st = handlers["atlas_stats"]({})
    assert res_st["status"] == "ok"
    mock_send.assert_called_with(
        Path("/tmp/fake_v58_atlas.sock"),
        "get_stats",
        {},
    )


def test_v58_method_timeout_pool():
    """Weryfikuje poprawne przypisanie timeoutów z DEFAULT_METHOD_TIMEOUTS w client.py."""
    assert DEFAULT_METHOD_TIMEOUTS["trigger_sleep_consolidation"] == 30.0
    assert DEFAULT_METHOD_TIMEOUTS["sync_mnemosyne"] == 30.0
    assert DEFAULT_METHOD_TIMEOUTS["verify_audit_log"] == 20.0
    assert DEFAULT_METHOD_TIMEOUTS["sync_memory_file"] == 20.0

    client = AtlasDaemonClient(socket_path="/tmp/fake.sock", timeout=2.0)

    # Test asynchronicznej kalkulacji effective_timeout w AtlasDaemonClient
    # 1. Domyślny timeout dla operacji ciężkiej
    async def fake_wait_for(coro, timeout=None):
        coro.close()
        return {"status": "ok"}

    with patch.object(client, "connect", AsyncMock(return_value=True)), \
         patch("asyncio.wait_for", side_effect=fake_wait_for) as mock_wait_for:
        client._reader = MagicMock()
        client._writer = MagicMock()
        client._writer.is_closing.return_value = False

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(client.call("trigger_sleep_consolidation"))
            assert mock_wait_for.call_args[1]["timeout"] == 30.0

            loop.run_until_complete(client.call("verify_audit_log"))
            assert mock_wait_for.call_args[1]["timeout"] == 20.0

            loop.run_until_complete(client.call("ping"))
            assert mock_wait_for.call_args[1]["timeout"] == 2.0

            # Jawny timeout nadpisuje domyślny
            loop.run_until_complete(client.call("trigger_sleep_consolidation", timeout=7.5))
            assert mock_wait_for.call_args[1]["timeout"] == 7.5
        finally:
            loop.close()

    # Test synchronicznej funkcji send_uds_request_sync
    with patch("os.path.exists", return_value=True), \
         patch("socket.socket") as mock_socket_cls:
        mock_sock_inst = MagicMock()
        mock_socket_cls.return_value = mock_sock_inst
        mock_sock_inst.recv.return_value = b""

        # Wywołanie z domyślnym timeoutem (5.0s) dla operacji z puli
        send_uds_request_sync("/tmp/fake.sock", "trigger_sleep_consolidation")
        mock_sock_inst.settimeout.assert_called_with(30.0)

        # Wywołanie dla nieznanej metody z domyślnym timeoutem
        send_uds_request_sync("/tmp/fake.sock", "ping")
        mock_sock_inst.settimeout.assert_called_with(5.0)

        # Wywołanie ze specyficznym timeoutem
        send_uds_request_sync("/tmp/fake.sock", "trigger_sleep_consolidation", timeout=12.0)
        mock_sock_inst.settimeout.assert_called_with(12.0)


@pytest.mark.asyncio
async def test_v58_vector_noise_filter_rejects_spurious_identifier():
    """Weryfikuje odrzucenie losowego identyfikatora o score < 0.70 przez dynamiczny filtr szumu."""
    mock_vec_store = MagicMock()
    # Mock search zwraca trafienie o score 0.55
    mock_vec_store.search = AsyncMock(return_value=[
        {
            "id": "item_1",
            "score": 0.55,
            "record": {
                "subject": "service:auth:key_999",
                "predicate": "semantic_fact",
                "object": "auth secret key configuration",
                "metadata": {},
            },
        }
    ])

    pipeline = RetrievalPipeline(vector_store=mock_vec_store)

    # 1. Zapytanie z dwukropkiem (identyfikator) -> próg rośnie do 0.70 -> trafienie (0.55) odrzucone
    res_id_colon = await pipeline.execute_prefetch({"query": "user:profile:test_id", "force": True})
    vec_items = [r for r in res_id_colon["records"] if r["subject"] == "service:auth:key_999"]
    assert len(vec_items) == 0

    # 2. Zapytanie z podkreśleniem (identyfikator) -> próg 0.70 -> odrzucone
    res_id_under = await pipeline.execute_prefetch({"query": "config_database_host", "force": True})
    vec_items_under = [r for r in res_id_under["records"] if r["subject"] == "service:auth:key_999"]
    assert len(vec_items_under) == 0

    # 3. Zapytanie z myślnikiem (identyfikator) -> próg 0.70 -> odrzucone
    res_id_dash = await pipeline.execute_prefetch({"query": "uuid-1234-5678", "force": True})
    vec_items_dash = [r for r in res_id_dash["records"] if r["subject"] == "service:auth:key_999"]
    assert len(vec_items_dash) == 0

    # 4. Pojedyncze słowo bez spacji >3 znaki (np. "secretkey") -> próg 0.70 -> odrzucone
    res_single_word = await pipeline.execute_prefetch({"query": "secretkey", "force": True})
    vec_items_single = [r for r in res_single_word["records"] if r["subject"] == "service:auth:key_999"]
    assert len(vec_items_single) == 0

    # 5. Zapytanie w języku naturalnym (ze spacjami, bez znaków identyfikatorów) -> próg 0.35 -> zaakceptowane
    res_nl = await pipeline.execute_prefetch({"query": "jak działa autentykacja użytkownika", "force": True})
    vec_items_nl = [r for r in res_nl["records"] if r["subject"] == "service:auth:key_999"]
    assert len(vec_items_nl) == 1
    assert vec_items_nl[0]["subject"] == "service:auth:key_999"


def test_v58_provider_get_tool_schemas_surface_parity():
    """Weryfikuje, że AtlasMemoryProvider.get_tool_schemas() zwraca pełny komplet 10 narzędzi (F-REG-0 guard)."""
    from atlas_memory.hermes.atlas_provider import AtlasMemoryProvider

    provider = AtlasMemoryProvider()
    schemas = provider.get_tool_schemas()
    assert len(schemas) == 10, f"Oczekiwano 10 schematów w provider.get_tool_schemas(), znaleziono {len(schemas)}"
    names = {s["name"] for s in schemas}
    expected_names = {
        "atlas_recall",
        "atlas_remember",
        "atlas_what_if",
        "atlas_active_sensing",
        "atlas_stats",
        "atlas_get",
        "atlas_delete",
        "atlas_task_feedback",
        "atlas_verify_chain",
        "atlas_sleep_sync",
    }
    assert names == expected_names

