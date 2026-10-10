"""Unit test reproducing and verifying F-031 sync_peer_status serialization crash."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from atlas_memory.server.atlas_daemon import AtlasDaemon, JSONRPCResponse
from atlas_memory.sync.crdt import VectorClock
from atlas_memory.sync.protocol import PeerStatus


class StubPeerWithLock:
    """Peer stub holding a threading.Lock directly in model_dump or attributes."""

    def __init__(self, peer_id: str = "peer_stub") -> None:
        self.peer_id = peer_id
        self.lock = threading.Lock()
        self.vector_clock = VectorClock(clocks={"node_b": 5})
        self.last_seen = 1234567.0
        self.rtt_ms = 4.2
        self.status = "active"

    def model_dump(self) -> Dict[str, Any]:
        return {
            "peer_id": self.peer_id,
            "lock": self.lock,
            "vector_clock": {"clocks": {"node_b": 5}, "_lock": self.lock},
            "last_seen": self.last_seen,
            "rtt_ms": self.rtt_ms,
            "status": self.status,
        }


@pytest.mark.asyncio
async def test_sync_peer_status_production_peer_status_json_serializable() -> None:
    """F-031 Reproduction: PeerStatus has CompactVectorClock which contains _lock.

    Calling _handle_sync_peer_status must produce a JSON-safe payload
    that can be serialized via json.dumps and JSONRPCResponse.model_dump_json.
    """
    daemon = AtlasDaemon.__new__(AtlasDaemon)
    mock_gossip = MagicMock()

    peer = PeerStatus(peer_id="node_prod_1")
    peer.vector_clock.clocks["node_prod_1"] = 10
    mock_gossip.peers = {"node_prod_1": peer}

    mock_bridge = MagicMock()
    mock_bridge.clock.clocks = {"node_local": 42}
    mock_gossip.bridge = mock_bridge
    daemon.gossip = mock_gossip

    result = await daemon._handle_sync_peer_status({})
    assert result["status"] == "ok"

    # Must serialize with standard json
    serialized = json.dumps(result)
    assert "node_prod_1" in serialized

    # Must serialize via Pydantic JSONRPCResponse
    rpc_resp = JSONRPCResponse(id="test-1", result=result)
    resp_bytes = rpc_resp.model_dump_json().encode("utf-8")
    assert b"node_prod_1" in resp_bytes


@pytest.mark.asyncio
async def test_sync_peer_status_stub_with_lock_sanitized() -> None:
    """F-031: Stub peer holding threading.Lock in model_dump must not crash serialization."""
    daemon = AtlasDaemon.__new__(AtlasDaemon)
    mock_gossip = MagicMock()

    peer = StubPeerWithLock("node_stub_lock")
    mock_gossip.peers = {"node_stub_lock": peer}
    mock_gossip.bridge = None
    mock_crdt = MagicMock()
    mock_crdt.clock.clocks = {"node_local": 1}
    mock_gossip.crdt = mock_crdt
    daemon.gossip = mock_gossip

    result = await daemon._handle_sync_peer_status({})
    assert result["status"] == "ok"

    serialized = json.dumps(result)
    assert "node_stub_lock" in serialized

    rpc_resp = JSONRPCResponse(id="test-2", result=result)
    resp_bytes = rpc_resp.model_dump_json().encode("utf-8")
    assert b"node_stub_lock" in resp_bytes


@pytest.mark.asyncio
async def test_sync_peer_status_via_daemon_uds_client(uds_dir: Path) -> None:
    """End-to-end UDS IPC test: client.call('sync_peer_status') returns clean dict, no IncompleteReadError."""
    from atlas_memory.ipc.uds_client import AtlasDaemonClient

    sock_path = uds_dir / "t1_peer.sock"
    pid_path = uds_dir / "t1_peer.pid"

    daemon = AtlasDaemon(socket_path=sock_path, pid_path=pid_path)

    # Attach gossip with real PeerStatus
    mock_gossip = MagicMock()
    peer = PeerStatus(peer_id="agent_peer_xyz")
    peer.vector_clock.clocks["agent_peer_xyz"] = 7
    mock_gossip.peers = {"agent_peer_xyz": peer}
    mock_gossip.bridge = None
    mock_crdt = MagicMock()
    mock_crdt.clock.clocks = {"local_node": 15}
    mock_gossip.crdt = mock_crdt
    daemon.gossip = mock_gossip

    await daemon.start()
    try:
        client = AtlasDaemonClient(socket_path=sock_path, timeout=5.0)
        res = await client.call("sync_peer_status", {})
        assert isinstance(res, dict)
        assert res["status"] == "ok"
        assert "agent_peer_xyz" in res["peers"]
        assert res["peers"]["agent_peer_xyz"]["vector_clock"] == {"agent_peer_xyz": 7}
        assert res["local_clock"] == {"local_node": 15}
        await client.close()
    finally:
        await daemon.stop()

