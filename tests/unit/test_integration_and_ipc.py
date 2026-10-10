"""
Unit tests for ATLAS V83: Architectural Audit & Refactoring Plan for Integration, Daemon & IPC.
Tests:
1. SyncUDSConnectionPool reuse, health checks, and lifecycle.
2. AtlasDaemonClient dynamic event loop binding across loop hops.
3. KuzuGraphStore get_stats lock hierarchy (_lock -> _graph_lock).
5. HermesSessionHook stale session TTL pruning.
6. TrajectoryExtractor strict call_id matching without positional desynchronization.
7. AtlasMemoryProvider BackgroundWorker queue execution and clean shutdown.
8. Daemon build stamp caching.
"""
from __future__ import annotations

import asyncio
import socket
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from atlas_memory.hermes.atlas_provider import BackgroundWorker
from atlas_memory.hermes.prefix_guard import HermesSessionHook
from atlas_memory.hermes.trajectory_extractor import extract_trajectory
from atlas_memory.ipc.uds_client import AtlasDaemonClient, SyncUDSConnectionPool
from atlas_memory.l2_semantic.kuzu_graph import KuzuGraphStore
from atlas_memory.server.atlas_daemon import _daemon_build_stamp

# ==============================================================================
# 1. SyncUDSConnectionPool
# ==============================================================================

def test_sync_uds_connection_pool_lifecycle():
    with tempfile.TemporaryDirectory() as tmp_dir:
        sock_path = str(Path(tmp_dir) / "test.sock")

        # Create dummy server
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(sock_path)
        server.listen(5)

        pool = SyncUDSConnectionPool(max_idle=2, max_lifetime=10.0)

        # Acquire 1
        s1, from_pool = pool.acquire(sock_path, timeout=2.0)
        assert not from_pool
        assert isinstance(s1, socket.socket)

        # Release to pool
        pool.release(sock_path, s1)
        assert len(pool._pools[sock_path]) == 1

        # Acquire 2 (should be from pool)
        s2, from_pool2 = pool.acquire(sock_path, timeout=2.0)
        assert from_pool2
        assert s2 is s1

        # Release again
        pool.release(sock_path, s2)
        assert len(pool._pools[sock_path]) == 1

        # Close all
        pool.close_all()
        assert len(pool._pools) == 0
        server.close()


# ==============================================================================
# 2. AtlasDaemonClient Dynamic Event Loop Binding
# ==============================================================================

def test_atlas_daemon_client_loop_hop():
    client = AtlasDaemonClient()

    async def _run_in_loop_1():
        lock1 = client._ensure_lock()
        assert lock1 is not None
        assert client._bound_loop is asyncio.get_running_loop()
        return lock1

    async def _run_in_loop_2():
        lock2 = client._ensure_lock()
        assert lock2 is not None
        assert client._bound_loop is asyncio.get_running_loop()
        return lock2

    l1 = asyncio.run(_run_in_loop_1())
    l2 = asyncio.run(_run_in_loop_2())
    assert l1 is not l2  # Distinct locks created for distinct loops, preventing loop contamination


# ==============================================================================
# 3. KuzuGraphStore Lock Hierarchy
# ==============================================================================

@pytest.mark.asyncio
async def test_kuzu_graph_store_get_stats_hierarchy():
    store = KuzuGraphStore(db_path=None)  # In-memory store
    sync_stats = store.get_stats_sync()
    assert "ram_nodes" in sync_stats
    assert "ram_edges" in sync_stats
    assert "kuzu_available" in sync_stats
    assert sync_stats["kuzu_available"] is True
    assert sync_stats["kuzu_nodes"] == 0
    assert sync_stats["kuzu_edges"] == 0

    async_stats = await store.get_stats()
    assert async_stats == sync_stats

    # Test fallback when conn is None
    store.conn = None
    fallback_stats = store.get_stats_sync()
    assert fallback_stats["kuzu_available"] is False
    assert fallback_stats["kuzu_nodes"] is None


# ==============================================================================
# 5. HermesSessionHook Stale Session TTL Pruning
# ==============================================================================

@pytest.mark.asyncio
async def test_hermes_session_hook_ttl_pruning():
    mock_engine = MagicMock()
    mock_engine.kv.get_states = AsyncMock(return_value={})

    hook = HermesSessionHook(mock_engine)

    # Populate active sessions with one fresh and one ancient
    now = time.time()
    hook.active_sessions["session_fresh"] = {"start_time": now, "session_id": "session_fresh"}
    hook.active_sessions["session_stale"] = {"start_time": now - 10000.0, "session_id": "session_stale"}

    assert len(hook.active_sessions) == 2
    pruned = hook._cleanup_stale_sessions(ttl_seconds=7200.0)
    assert pruned == 1
    assert "session_stale" not in hook.active_sessions
    assert "session_fresh" in hook.active_sessions

    # on_session_start automatically triggers pruning
    hook.active_sessions["session_old2"] = {"start_time": now - 9000.0, "session_id": "session_old2"}
    await hook.on_session_start("session_new")
    assert "session_old2" not in hook.active_sessions
    assert "session_new" in hook.active_sessions


# ==============================================================================
# 6. TrajectoryExtractor Strict call_id Matching
# ==============================================================================

def test_trajectory_extractor_strict_call_id_no_stealing():
    # Scenario: assistant invokes call_1 (missing result) and call_2 (successful result)
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "atlas_remember", "arguments": "{\"key\": \"k1\"}"},
                },
                {
                    "id": "call_2",
                    "function": {"name": "atlas_get", "arguments": "{\"key\": \"k2\"}"},
                },
            ],
        },
        # Tool result only arrives for call_2!
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": "{\"status\": \"ok\", \"found\": true}",
        },
    ]

    steps = extract_trajectory(messages, session_id="test_sess")
    assert len(steps) == 2

    # Step 1 (call_1) should be marked failure due to missing result and NOT steal call_2's result
    assert steps[0]["tool_name"] == "atlas_remember"
    assert steps[0]["success"] is False
    assert steps[0]["error"] == "no_result"

    # Step 2 (call_2) should match call_2 and succeed
    assert steps[1]["tool_name"] == "atlas_get"
    assert steps[1]["success"] is True
    assert steps[1]["error"] is None


# ==============================================================================
# 7. BackgroundWorker Queue & Shutdown
# ==============================================================================

def test_background_worker_fifo_and_shutdown():
    worker = BackgroundWorker(name="TestWorker", maxsize=10)
    results = []

    def _task_1():
        results.append(1)

    def _task_2():
        results.append(2)

    assert worker.submit(_task_1)
    assert worker.submit(_task_2)

    # Wait for queue drain
    time.sleep(0.1)
    assert results == [1, 2]

    worker.shutdown()
    assert worker._closed is True
    assert not worker.submit(_task_1)


# ==============================================================================
# 8. Daemon Build Stamp Caching
# ==============================================================================

def test_daemon_build_stamp_cached():
    stamp1 = _daemon_build_stamp()
    assert isinstance(stamp1, dict)
    assert "version" in stamp1

    stamp2 = _daemon_build_stamp()
    assert stamp1 is stamp2  # Identical cached object reference
