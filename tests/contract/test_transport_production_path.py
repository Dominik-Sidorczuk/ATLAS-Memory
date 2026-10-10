"""
Production-path tests for ATLAS UDS transport error handling.

Tests run WITHOUT mocks or monkeypatching on client.py/tools.py.
Verifies:
1. Real transport failure with nonexistent socket raises AtlasTransportError in client.py.
2. create_uds_tool_handlers handles AtlasTransportError gracefully:
   - atlas_get does NOT return {"found": False} (preventing false-negative rule dismissal)
   - atlas_get returns status="error" and transport_reason="socket_missing"
   - all other handlers return transport_reason="socket_missing"
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas_memory.hermes.tools import create_uds_tool_handlers
from atlas_memory.ipc.uds_client import AtlasTransportError, send_uds_request_sync


def test_send_uds_request_sync_raises_transport_error_on_missing_socket(tmp_path: Path):
    """Direct invocation of send_uds_request_sync on missing socket unconditionally raises AtlasTransportError."""
    missing_sock = tmp_path / "nonexistent.sock"
    with pytest.raises(AtlasTransportError) as exc_info:
        send_uds_request_sync(missing_sock, "get", {"key": "rule:git:branch_strategy"})
    
    assert exc_info.value.reason == "socket_missing"
    assert "does not exist" in exc_info.value.detail


def test_atlas_get_does_not_return_found_false_on_transport_failure(tmp_path: Path):
    """
    Critical contract test (F-110, F-112):
    When daemon is unreachable, atlas_get MUST NOT claim found=False.
    Claiming found=False would cause the agent to believe an active rule does not exist.
    """
    missing_sock = tmp_path / "nonexistent.sock"
    handlers = create_uds_tool_handlers(missing_sock)

    res = handlers["atlas_get"]({"key": "rule:git:branch_strategy"})

    # Must NOT report found=False
    assert "found" not in res or res["found"] is not False
    assert res.get("found") is None

    # Must report error status and transport_reason
    assert res["status"] == "error"
    assert res["transport_reason"] == "socket_missing"
    assert res["key"] == "rule:git:branch_strategy"
    assert res["state"] is None


def test_all_uds_handlers_report_transport_reason_on_missing_socket(tmp_path: Path):
    """All 11 UDS handlers report transport_reason='socket_missing' on transport failure."""
    missing_sock = tmp_path / "nonexistent.sock"
    handlers = create_uds_tool_handlers(missing_sock)

    # 1. atlas_recall
    res_recall = handlers["atlas_recall"]({"query": "architecture guidelines"})
    assert res_recall["status"] == "daemon_offline"
    assert res_recall["transport_reason"] == "socket_missing"
    assert res_recall["records"] == []

    # 2. atlas_remember
    res_remember = handlers["atlas_remember"]({"key": "pref:lang", "value": "pl"})
    assert res_remember["status"] == "error"
    assert res_remember["transport_reason"] == "socket_missing"
    assert res_remember["persisted"] is False

    # 3. commit_observation
    res_commit = handlers["commit_observation"]({"subject": "test", "predicate": "obs", "object": "val"})
    assert res_commit["status"] == "error"
    assert res_commit["transport_reason"] == "socket_missing"
    assert res_commit["committed"] is False

    # 4. atlas_what_if
    res_what_if = handlers["atlas_what_if"]({"entity": "Database", "action": "restart"})
    assert res_what_if["status"] == "error"
    assert res_what_if["transport_reason"] == "socket_missing"
    assert res_what_if["causal_paths"] == []

    # 5. atlas_active_sensing
    res_sensing = handlers["atlas_active_sensing"]({"probe": "latency", "expected_value": "100", "observed_value": "110"})
    assert res_sensing["status"] == "error"
    assert res_sensing["transport_reason"] == "socket_missing"
    assert res_sensing["has_error"] is False

    # 6. atlas_stats
    res_stats = handlers["atlas_stats"]()
    assert res_stats["status"] == "error"
    assert res_stats["transport_reason"] == "socket_missing"
    assert res_stats["serving"] is False

    # 7. atlas_delete
    res_delete = handlers["atlas_delete"]({"key": "temp:var", "reason": "cleanup"})
    assert res_delete["status"] == "error"
    assert res_delete["transport_reason"] == "socket_missing"
    assert res_delete["deleted"] is False

    # 8. atlas_task_feedback
    res_fb = handlers["atlas_task_feedback"]({"key": "rule:dns", "task_success": True})
    assert res_fb["status"] == "error"
    assert res_fb["transport_reason"] == "socket_missing"

    # 9. atlas_verify_chain
    res_vc = handlers["atlas_verify_chain"]({"heal": False})
    assert res_vc["status"] == "error"
    assert res_vc["transport_reason"] == "socket_missing"
    assert res_vc["valid"] is False

    # 10. atlas_sleep_sync
    res_sleep = handlers["atlas_sleep_sync"]()
    assert res_sleep["status"] == "error"
    assert res_sleep["transport_reason"] == "socket_missing"
    assert res_sleep["consolidated_records"] == 0
