"""
Contract tests for Hermes Tools & AtlasDaemon RPC Parity.

Verifies:
1. Tool schemas and parameter constraints for all 5 ATLAS tools.
2. Handler parameter translation and payload contract with AtlasDaemon RPC methods:
   - atlas_recall -> 'prefetch' with default force=False, query, session_id, limit.
   - atlas_remember -> 'set' with key/subject/name, value/object/content, confidence, reason.
   - atlas_what_if -> 'what_if' with entity/target, action, depth.
   - atlas_active_sensing -> 'active_sensing' with probe/target_entity, observed_value, expected_value, tolerance.
   - atlas_stats -> 'get_stats'.
3. Daemon RPC method parity with AtlasDaemon._handlers.
4. Offline daemon fallback behavior across all handlers.
5. Cognitive policy gate integration: continuation queries pass with force=False.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from atlas_memory.hermes.tools import (
    ATLAS_ACTIVE_SENSING_SCHEMA,
    ATLAS_RECALL_SCHEMA,
    ATLAS_REMEMBER_SCHEMA,
    ATLAS_TOOL_SCHEMAS,
    ATLAS_WHAT_IF_SCHEMA,
    create_uds_tool_handlers,
)
from atlas_memory.server.atlas_daemon import AtlasDaemon


def test_atlas_tool_schemas_integrity():
    """Validates that all 5 canonical tool schemas conform to OpenAI/Hermes function calling structure."""
    expected_tool_names = {
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
    registered_names = {s["function"]["name"] for s in ATLAS_TOOL_SCHEMAS}
    assert registered_names == expected_tool_names

    # Check atlas_recall schema
    recall_fn = ATLAS_RECALL_SCHEMA["function"]
    assert "query" in recall_fn["parameters"]["properties"]
    assert "query" in recall_fn["parameters"]["required"]
    assert recall_fn["parameters"]["properties"]["limit"]["default"] == 15

    # Check atlas_remember schema
    remember_fn = ATLAS_REMEMBER_SCHEMA["function"]
    assert "key" in remember_fn["parameters"]["properties"]
    assert "value" in remember_fn["parameters"]["properties"]
    assert set(remember_fn["parameters"]["required"]) == {"key", "value"}

    # Check atlas_what_if schema
    what_if_fn = ATLAS_WHAT_IF_SCHEMA["function"]
    assert "entity" in what_if_fn["parameters"]["properties"]
    assert "action" in what_if_fn["parameters"]["properties"]
    assert set(what_if_fn["parameters"]["required"]) == {"entity", "action"}

    # Check atlas_active_sensing schema
    sensing_fn = ATLAS_ACTIVE_SENSING_SCHEMA["function"]
    assert "probe" in sensing_fn["parameters"]["properties"]
    assert "observed_value" in sensing_fn["parameters"]["properties"]
    assert "expected_value" in sensing_fn["parameters"]["properties"]
    assert "tolerance" in sensing_fn["parameters"]["properties"]
    assert set(sensing_fn["parameters"]["required"]) == {"probe", "observed_value"}


def test_contract_parity_with_daemon_rpc_methods():
    """Verifies that all tool handlers map to valid, existing RPC handlers in AtlasDaemon."""
    daemon = AtlasDaemon()
    registered_rpc_methods = set(daemon._handlers.keys())

    tool_to_rpc_mapping = {
        "atlas_recall": "prefetch",
        "atlas_remember": "set",
        "atlas_what_if": "what_if",
        "atlas_active_sensing": "active_sensing",
        "atlas_stats": "get_stats",
        "atlas_get": "get",
        "atlas_delete": "delete",
        "atlas_task_feedback": "task_feedback",
        "atlas_verify_chain": "verify_audit_log",
        "atlas_sleep_sync": "trigger_sleep_consolidation",
    }

    for tool_name, rpc_method in tool_to_rpc_mapping.items():
        assert rpc_method in registered_rpc_methods, (
            f"RPC method '{rpc_method}' mapped from '{tool_name}' not registered in AtlasDaemon handlers!"
        )


@patch("atlas_memory.hermes.tools.send_uds_request_sync")
def test_atlas_recall_contract_mapping(mock_send):
    """Verifies atlas_recall maps parameters and defaults force to False."""
    mock_send.return_value = {"records": [{"subject": "test", "object": "val"}], "count": 1}
    handlers = create_uds_tool_handlers("/tmp/fake_atlas.sock")

    # 1. Default invocation: force must be False
    res_default = handlers["atlas_recall"]({"query": "Kontynuuj zadanie"})
    assert res_default["count"] == 1
    mock_send.assert_called_with(
        Path("/tmp/fake_atlas.sock"),
        "prefetch",
        {
            "query": "Kontynuuj zadanie",
            "session_id": "hermes_default",
            "limit": 15,
            "force": False,
        },
    )

    # 2. Explicit force=True override
    mock_send.reset_mock()
    res_forced = handlers["atlas_recall"]({"query": "Sprawdź port", "session_id": "sess_1", "limit": 5, "force": True})
    assert res_forced["count"] == 1
    mock_send.assert_called_with(
        Path("/tmp/fake_atlas.sock"),
        "prefetch",
        {
            "query": "Sprawdź port",
            "session_id": "sess_1",
            "limit": 5,
            "force": True,
        },
    )


@patch("atlas_memory.hermes.tools.send_uds_request_sync")
def test_atlas_remember_contract_mapping(mock_send):
    """Verifies atlas_remember correctly maps alias parameters (key/subject/name, value/object/content)."""
    mock_send.return_value = {"status": "ok", "key": "user:pref", "persisted": True}
    handlers = create_uds_tool_handlers("/tmp/fake_atlas.sock")

    # Standard call
    res = handlers["atlas_remember"]({
        "key": "user:pref",
        "value": "dark_mode",
        "confidence": 0.95,
        "reason": "user_input",
    })
    assert res["persisted"] is True
    mock_send.assert_called_with(
        Path("/tmp/fake_atlas.sock"),
        "set",
        {"key": "user:pref", "value": "dark_mode", "confidence": 0.95, "reason": "user_input", "session_id": "hermes_default"},
    )

    # Alias call: subject + object
    mock_send.reset_mock()
    handlers["atlas_remember"]({
        "subject": "service:db",
        "object": "postgres",
    })
    mock_send.assert_called_with(
        Path("/tmp/fake_atlas.sock"),
        "set",
        {"key": "service:db", "value": "postgres", "confidence": 1.0, "reason": "agent_explicit", "session_id": "hermes_default"},
    )


@patch("atlas_memory.hermes.tools.send_uds_request_sync")
def test_atlas_what_if_contract_mapping(mock_send):
    """Verifies atlas_what_if parameter translation."""
    mock_send.return_value = {"status": "ok", "causal_paths": [{"entity": "DatabaseCluster"}]}
    handlers = create_uds_tool_handlers("/tmp/fake_atlas.sock")

    res = handlers["atlas_what_if"]({
        "entity": "DatabaseCluster",
        "action": "restart_service",
        "depth": 3,
    })
    assert res["status"] == "ok"
    mock_send.assert_called_with(
        Path("/tmp/fake_atlas.sock"),
        "what_if",
        {"entity": "DatabaseCluster", "action": "restart_service", "depth": 3},
    )


@patch("atlas_memory.hermes.tools.send_uds_request_sync")
def test_atlas_active_sensing_contract_mapping(mock_send):
    """Verifies atlas_active_sensing parameter translation including numeric tolerance."""
    mock_send.return_value = {"status": "ok", "has_error": False}
    handlers = create_uds_tool_handlers("/tmp/fake_atlas.sock")

    res = handlers["atlas_active_sensing"]({
        "probe": "agent_latency",
        "expected_value": "120",
        "observed_value": "125",
        "tolerance": 10.0,
    })
    assert res["has_error"] is False
    mock_send.assert_called_with(
        Path("/tmp/fake_atlas.sock"),
        "active_sensing",
        {
            "target_entity": "agent_latency",
            "expected_value": "120",
            "observed_value": "125",
            "tolerance": 10.0,
        },
    )


@patch("atlas_memory.hermes.tools.send_uds_request_sync")
def test_atlas_stats_contract_mapping(mock_send):
    """Verifies atlas_stats sends get_stats RPC request."""
    mock_send.return_value = {"status": "ok", "serving": True, "kv_records": 42}
    handlers = create_uds_tool_handlers("/tmp/fake_atlas.sock")

    res = handlers["atlas_stats"]()
    assert res["serving"] is True
    assert res["kv_records"] == 42
    mock_send.assert_called_with(Path("/tmp/fake_atlas.sock"), "get_stats", {})


@patch("atlas_memory.hermes.tools.send_uds_request_sync")
def test_daemon_offline_fallback(mock_send):
    """Verifies graceful fallback payloads when daemon socket is unreachable or returns None."""
    mock_send.return_value = None
    handlers = create_uds_tool_handlers("/tmp/offline.sock")

    res_recall = handlers["atlas_recall"]({"query": "test"})
    assert res_recall["status"] == "daemon_offline"
    assert res_recall["records"] == []

    res_remember = handlers["atlas_remember"]({"key": "a", "value": "b"})
    assert res_remember["status"] == "error"
    assert res_remember["persisted"] is False

    res_what_if = handlers["atlas_what_if"]({"entity": "a", "action": "b"})
    assert res_what_if["status"] == "error"
    assert res_what_if["causal_paths"] == []

    res_sensing = handlers["atlas_active_sensing"]({"probe": "a", "observed_value": "1"})
    assert res_sensing["status"] == "error"
    assert res_sensing["has_error"] is False

    res_stats = handlers["atlas_stats"]()
    assert res_stats["status"] == "error"
    assert res_stats["serving"] is False


def test_provider_and_daemon_rpc_method_parity():
    """Verifies that all methods called by AtlasMemoryProvider exist in AtlasDaemon."""
    daemon = AtlasDaemon()
    registered_rpc_methods = set(daemon._handlers.keys())

    provider_rpc_methods = {
        "prefetch",
        "session/end",
        "session_end",
        "set",
        "get",
        "delete",
        "get_stats",
    }
    for method in provider_rpc_methods:
        assert method in registered_rpc_methods, (
            f"Provider RPC method '{method}' not registered in AtlasDaemon handlers!"
        )


async def test_daemon_session_end_dispatch():
    """Verifies AtlasDaemon handles session/end and session_end RPC calls gracefully."""
    daemon = AtlasDaemon()
    res = await daemon._handle_session_end({
        "session_id": "contract_sess_42",
        "turn_count": 3,
        "trigger_sleep": False,
    })
    assert res["status"] == "ok"
    assert res["session_id"] == "contract_sess_42"
    assert res["turn_count"] == 3

