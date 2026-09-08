"""Tests for AtlasDaemon IPC over Unix Domain Socket (JSON-RPC 2.0)."""

from __future__ import annotations

import asyncio
import json
import struct
from pathlib import Path
from typing import Any, Dict, List

import pytest

from atlas_memory.hermes.atlas_provider import AtlasMemoryProvider
from atlas_memory.server.atlas_daemon import AtlasDaemon
from atlas_memory.server.client import AtlasDaemonClient


class DummyOrchestrator:
    """Mock orchestrator for server IPC testing."""

    def __init__(self) -> None:
        self.stats: Dict[str, Any] = {"tokens_saved_estimate": 42}

    async def orchestrated_recall(self, query: str, session_id: str = "") -> List[Dict[str, Any]]:
        return [{"id": "rec_1", "content": f"Memory about {query}", "veracity": 0.9}]

    def _fallback_extract_facts(self, user_msg: str, agent_response: str) -> List[str]:
        return [f"Fact: {user_msg} -> {agent_response}"]


class DummyGraphClient:
    """Mock causal graph client for what_if testing."""

    def what_if(self, action: str) -> Dict[str, Any]:
        return {"action": action, "nodes": ["N1", "N2"], "confidence": 0.88}


@pytest.mark.asyncio
async def test_daemon_ping(tmp_path: Path) -> None:
    """Test 1: Start daemon on temp socket, send ping, receive status ok."""
    sock_path = tmp_path / "atlas_test.sock"
    pid_path = tmp_path / "atlas_test.pid"

    daemon = AtlasDaemon(socket_path=sock_path, pid_path=pid_path)
    await daemon.start()
    try:
        client = AtlasDaemonClient(socket_path=sock_path)
        is_ok = await client.ping()
        assert is_ok is True
        await client.close()
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_daemon_prefetch(tmp_path: Path) -> None:
    """Test 2: Daemon with mock orchestrator responds to prefetch RPC."""
    sock_path = tmp_path / "atlas_test.sock"
    pid_path = tmp_path / "atlas_test.pid"

    orchestrator = DummyOrchestrator()
    daemon = AtlasDaemon(socket_path=sock_path, pid_path=pid_path, orchestrator=orchestrator)
    await daemon.start()
    try:
        client = AtlasDaemonClient(socket_path=sock_path)
        resp = await client.call("prefetch", {"query": "vector databases", "session_id": "s1"})
        assert isinstance(resp, dict)
        assert resp["query"] == "vector databases"
        assert len(resp["records"]) == 1
        assert "Memory about vector databases" in resp["records"][0]["content"]
        await client.close()
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_daemon_what_if(tmp_path: Path) -> None:
    """Test 3: Daemon with mock graph client handles what_if RPC query."""
    sock_path = tmp_path / "atlas_test.sock"
    pid_path = tmp_path / "atlas_test.pid"

    graph = DummyGraphClient()
    daemon = AtlasDaemon(socket_path=sock_path, pid_path=pid_path, graph_client=graph)
    await daemon.start()
    try:
        client = AtlasDaemonClient(socket_path=sock_path)
        resp = await client.call("what_if", {"action": "delete_cache"})
        assert isinstance(resp, dict)
        assert resp["action"] == "delete_cache"
        assert resp["causal_path"]["confidence"] == 0.88
        await client.close()
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_client_fallback_when_socket_missing(tmp_path: Path) -> None:
    """Test 4: Client and AtlasMemoryProvider fallback gracefully when socket is missing."""
    sock_path = tmp_path / "non_existent.sock"
    client = AtlasDaemonClient(socket_path=sock_path)

    # Direct client ping returns False on missing socket
    assert await client.ping() is False

    # Calling client on missing socket raises ConnectionError
    with pytest.raises(ConnectionError):
        await client.call("ping")

    # AtlasMemoryProvider sync fallback
    class MockOrch:
        def should_retrieve(self, q: str, explicit_entities: Any = None):
            return False, [], "no_entities"
        stats: Dict[str, Any] = {}

    provider = AtlasMemoryProvider(orchestrator=MockOrch(), socket_path=str(sock_path))  # type: ignore[arg-type]
    res = provider._call_uds_sync("ping")
    assert res is None


@pytest.mark.asyncio
async def test_client_reconnect(tmp_path: Path) -> None:
    """Test 5: Client reconnects when daemon restarts."""
    sock_path = tmp_path / "atlas_test.sock"
    pid_path = tmp_path / "atlas_test.pid"

    daemon = AtlasDaemon(socket_path=sock_path, pid_path=pid_path)
    await daemon.start()

    client = AtlasDaemonClient(socket_path=sock_path)
    assert await client.ping() is True

    # Stop daemon & close client
    await daemon.stop()
    await client.close()
    assert await client.ping() is False

    # Restart daemon
    daemon2 = AtlasDaemon(socket_path=sock_path, pid_path=pid_path)
    await daemon2.start()
    try:
        # Client should reconnect
        assert await client.ping() is True
        await client.close()
    finally:
        await daemon2.stop()


@pytest.mark.asyncio
async def test_concurrent_requests(tmp_path: Path) -> None:
    """Test 6: Multiple concurrent requests over UDS client succeed without race conditions."""
    sock_path = tmp_path / "atlas_test.sock"
    pid_path = tmp_path / "atlas_test.pid"

    orchestrator = DummyOrchestrator()
    daemon = AtlasDaemon(socket_path=sock_path, pid_path=pid_path, orchestrator=orchestrator)
    await daemon.start()
    try:
        client = AtlasDaemonClient(socket_path=sock_path)

        async def _req(idx: int) -> Dict[str, Any]:
            return await client.call("prefetch", {"query": f"query_{idx}"})

        results = await asyncio.gather(*[_req(i) for i in range(5)])
        assert len(results) == 5
        for i, res in enumerate(results):
            assert res["query"] == f"query_{i}"

        await client.close()
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_pid_file_prevents_multi_instance(tmp_path: Path) -> None:
    """Test 7: Attempting to start daemon with active PID file raises RuntimeError."""
    sock_path = tmp_path / "atlas_test.sock"
    pid_path = tmp_path / "atlas_test.pid"

    daemon1 = AtlasDaemon(socket_path=sock_path, pid_path=pid_path)
    await daemon1.start()

    daemon2 = AtlasDaemon(socket_path=sock_path, pid_path=pid_path)
    with pytest.raises(RuntimeError, match="AtlasDaemon is already running"):
        await daemon2.start()

    await daemon1.stop()


@pytest.mark.asyncio
async def test_daemon_notifications_send_no_response(tmp_path: Path) -> None:
    """Test 8: Notification (no 'id' or id=None) executes handler without sending response."""
    sock_path = tmp_path / "atlas_test.sock"
    pid_path = tmp_path / "atlas_test.pid"

    daemon = AtlasDaemon(socket_path=sock_path, pid_path=pid_path)
    await daemon.start()
    try:
        reader, writer = await asyncio.open_unix_connection(path=str(sock_path))

        # Send notification (missing "id")
        notif = {"jsonrpc": "2.0", "method": "ping"}
        payload = json.dumps(notif).encode("utf-8")
        writer.write(struct.pack(">I", len(payload)) + payload)
        await writer.drain()

        # Notification must NOT send any response - read should timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(reader.read(4), timeout=0.3)

        # Send notification with explicit id=None
        notif_none = {"jsonrpc": "2.0", "method": "ping", "id": None}
        payload_none = json.dumps(notif_none).encode("utf-8")
        writer.write(struct.pack(">I", len(payload_none)) + payload_none)
        await writer.drain()

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(reader.read(4), timeout=0.3)

        # Send regular call with id to verify connection is still alive and responsive
        req = {"jsonrpc": "2.0", "method": "ping", "id": 100}
        req_bytes = json.dumps(req).encode("utf-8")
        writer.write(struct.pack(">I", len(req_bytes)) + req_bytes)
        await writer.drain()

        len_bytes = await asyncio.wait_for(reader.readexactly(4), timeout=1.0)
        (resp_len,) = struct.unpack(">I", len_bytes)
        resp_bytes = await reader.readexactly(resp_len)
        resp_dict = json.loads(resp_bytes.decode("utf-8"))
        assert resp_dict["id"] == 100
        assert resp_dict["result"] == {"status": "ok"}

        writer.close()
        await writer.wait_closed()
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_daemon_batch_requests(tmp_path: Path) -> None:
    """Test 9: Batch requests handling: empty list, valid batches, mixed with notifications."""
    sock_path = tmp_path / "atlas_test.sock"
    pid_path = tmp_path / "atlas_test.pid"

    daemon = AtlasDaemon(socket_path=sock_path, pid_path=pid_path)
    await daemon.start()
    try:
        reader, writer = await asyncio.open_unix_connection(path=str(sock_path))

        # 1. Empty batch [] -> returns JSONRPCResponse with error invalid_request
        empty_batch: List[Any] = []
        payload = json.dumps(empty_batch).encode("utf-8")
        writer.write(struct.pack(">I", len(payload)) + payload)
        await writer.drain()

        len_bytes = await asyncio.wait_for(reader.readexactly(4), timeout=1.0)
        (resp_len,) = struct.unpack(">I", len_bytes)
        resp = json.loads((await reader.readexactly(resp_len)).decode("utf-8"))
        assert "error" in resp
        assert resp["error"]["code"] == -32600
        assert "Batch cannot be empty" in resp["error"]["message"]

        # 2. Valid batch with multiple requests
        batch = [
            {"jsonrpc": "2.0", "method": "ping", "id": 1},
            {"jsonrpc": "2.0", "method": "ping", "id": 2},
        ]
        payload = json.dumps(batch).encode("utf-8")
        writer.write(struct.pack(">I", len(payload)) + payload)
        await writer.drain()

        len_bytes = await asyncio.wait_for(reader.readexactly(4), timeout=1.0)
        (resp_len,) = struct.unpack(">I", len_bytes)
        responses = json.loads((await reader.readexactly(resp_len)).decode("utf-8"))
        assert isinstance(responses, list)
        assert len(responses) == 2
        assert responses[0]["id"] == 1 and responses[0]["result"] == {"status": "ok"}
        assert responses[1]["id"] == 2 and responses[1]["result"] == {"status": "ok"}

        # 3. Batch mixed with requests and notifications
        mixed_batch = [
            {"jsonrpc": "2.0", "method": "ping", "id": 10},
            {"jsonrpc": "2.0", "method": "ping"},  # notification
            {"jsonrpc": "2.0", "method": "ping", "id": 11},
        ]
        payload = json.dumps(mixed_batch).encode("utf-8")
        writer.write(struct.pack(">I", len(payload)) + payload)
        await writer.drain()

        len_bytes = await asyncio.wait_for(reader.readexactly(4), timeout=1.0)
        (resp_len,) = struct.unpack(">I", len_bytes)
        mixed_resp = json.loads((await reader.readexactly(resp_len)).decode("utf-8"))
        assert isinstance(mixed_resp, list)
        assert len(mixed_resp) == 2
        assert [r["id"] for r in mixed_resp] == [10, 11]

        # 4. Batch of ONLY notifications -> server writes nothing
        all_notifs = [
            {"jsonrpc": "2.0", "method": "ping"},
            {"jsonrpc": "2.0", "method": "ping", "id": None},
        ]
        payload = json.dumps(all_notifs).encode("utf-8")
        writer.write(struct.pack(">I", len(payload)) + payload)
        await writer.drain()

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(reader.read(4), timeout=0.3)

        writer.close()
        await writer.wait_closed()
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_daemon_verify_audit_log_rpc(tmp_path: Path) -> None:
    """Test 10: verify_audit_log RPC method checks audit log and returns expected dict."""
    sock_path = tmp_path / "atlas_test.sock"
    pid_path = tmp_path / "atlas_test.pid"

    class MockKV:
        def __init__(self) -> None:
            self.deep_called = False

        async def verify_audit_log(self, deep: bool = False):
            self.deep_called = deep
            return True, 7

    class MockEngine:
        def __init__(self) -> None:
            self.kv = MockKV()

    engine = MockEngine()
    daemon = AtlasDaemon(socket_path=sock_path, pid_path=pid_path, engine=engine)
    await daemon.start()
    try:
        client = AtlasDaemonClient(socket_path=sock_path)
        resp = await client.call("verify_audit_log", {"deep": True})
        assert isinstance(resp, dict)
        assert resp["status"] == "ok"
        assert resp["valid"] is True
        assert resp["checked_entries"] == 7
        assert engine.kv.deep_called is True
        await client.close()
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_daemon_sync_turn_status_ok(tmp_path: Path) -> None:
    """Test 11: sync_turn RPC returns status: ok in response dict."""
    sock_path = tmp_path / "atlas_test.sock"
    pid_path = tmp_path / "atlas_test.pid"

    daemon = AtlasDaemon(socket_path=sock_path, pid_path=pid_path)
    await daemon.start()
    try:
        client = AtlasDaemonClient(socket_path=sock_path)
        res = await client.call("sync_turn", {
            "user_content": "Remember that Alice lives in Warsaw",
            "assistant_content": "Understood, noted.",
            "session_id": "test_session",
        })
        assert isinstance(res, dict)
        assert res["status"] == "ok"
        assert res["synced"] is True
        assert "extracted_facts" in res
        assert res["session_id"] == "test_session"
        await client.close()
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_daemon_get_all_sync_used_in_prefetch_and_stats(tmp_path: Path) -> None:
    """Test 12: Verify get_all_sync is invoked for prefetch and get_stats."""
    sock_path = tmp_path / "atlas_test.sock"
    pid_path = tmp_path / "atlas_test.pid"

    class SpyKV:
        def __init__(self) -> None:
            self.get_all_sync_called = 0

        def get_all_sync(self) -> Dict[str, Any]:
            self.get_all_sync_called += 1
            return {
                "pref_key": {
                    "value": "special vector index content",
                    "confidence": 1.0,
                    "metadata": {"session_id": "global"},
                }
            }

    class SpyEngine:
        def __init__(self) -> None:
            self.kv = SpyKV()

    engine = SpyEngine()
    daemon = AtlasDaemon(socket_path=sock_path, pid_path=pid_path, engine=engine)
    await daemon.start()
    try:
        client = AtlasDaemonClient(socket_path=sock_path)

        # 1. get_stats
        stats = await client.call("get_stats")
        assert stats["status"] == "ok"
        assert stats["kv_records"] == 1
        assert engine.kv.get_all_sync_called == 1

        # 2. prefetch
        res = await client.call("prefetch", {"query": "special vector index"})
        assert engine.kv.get_all_sync_called >= 2
        assert len(res["records"]) >= 1

        await client.close()
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_live_daemon_d2_sha_chain_auto_healed() -> None:
    """P5: Live Daemon D2 SHA Chain is verified and auto-healed on startup.

    Verifies that the running daemon on ~/.hermes/atlas.sock returns
    valid=True for audit log check.
    """
    sock_path = Path.home() / ".hermes" / "atlas.sock"
    if not sock_path.exists():
        pytest.skip("Atlas daemon socket not running")

    client = AtlasDaemonClient(socket_path=sock_path)
    try:
        res = await client.call("verify_audit_log", {"deep": False, "heal": False})
        assert isinstance(res, dict)
        assert res.get("valid") is True
    finally:
        await client.close()


def test_h5_uds_e2e_prefetch_continuation():
    """
    H5 UDS E2E test: verifies atlas_recall tool handler executes prefetch via Unix Domain Socket.
    Skips cleanly with pytest.skip if daemon socket does not exist.
    """
    from atlas_memory.server.models import DEFAULT_SOCKET_PATH
    sock_path = Path(DEFAULT_SOCKET_PATH)
    if not sock_path.exists():
        pytest.skip(f"ATLAS daemon UDS socket not found at {sock_path}")

    from atlas_memory.hermes.tools import create_uds_tool_handlers

    handlers = create_uds_tool_handlers(socket_path=sock_path)
    recall_fn = handlers.get("atlas_recall")
    assert recall_fn is not None

    res = recall_fn({
        "query": "Kontynuuj działanie",
        "session_id": "hermes_default",
        "force": False,
    })
    assert isinstance(res, dict)
    assert "records" in res or "count" in res


@pytest.mark.asyncio
async def test_live_daemon_uds_trajectory_e2e(tmp_path: Path):
    """Uruchamia daemona UDS, wykonuje 3 wywołania RPC i weryfikuje zliczanie trajektorii w get_stats."""
    sock_path = tmp_path / "test_v51.sock"
    pid_path = tmp_path / "test_v51.pid"
    atlas_dir = tmp_path / "atlas_vault"

    daemon = AtlasDaemon.create_default(
        socket_path=sock_path,
        pid_path=pid_path,
        atlas_dir=atlas_dir,
    )
    await daemon.start()
    try:
        assert sock_path.exists()
        client = AtlasDaemonClient(socket_path=sock_path)

        res1 = await client.call("commit_observation", {
            "subject": "SystemArchitecture",
            "predicate": "version",
            "object": "V51",
            "confidence": 0.99,
            "session_id": "test_e2e_session",
        })
        assert res1["status"] == "ok"

        res2 = await client.call("commit_observation", {
            "subject": "OptimizationStrategy",
            "predicate": "policy",
            "object": "Connect_Zero_Rm",
            "confidence": 0.95,
            "session_id": "test_e2e_session",
        })
        assert res2["status"] == "ok"

        res3 = await client.call("sync_turn", {
            "user_content": "Jaki jest status implementacji V51?",
            "assistant_content": "Architektura V51 została wdrożona bez usuwania kodu.",
            "session_id": "test_e2e_session",
        })
        assert res3["status"] == "ok"

        stats = await client.call("get_stats")
        assert stats["status"] == "ok"
        assert stats["trajectories_count"] >= 3
        assert stats["active_trajectories"] >= 1
    finally:
        await daemon.stop()



