"""Tests for AtlasDaemon IPC over Unix Domain Socket (JSON-RPC 2.0)."""

from __future__ import annotations

import asyncio
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
