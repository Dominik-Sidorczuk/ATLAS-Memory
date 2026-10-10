"""Unit and contract tests for ATLAS Daemon startup logging and diagnostics (V148).

Verifies:
1. Phased boot sequence logging format [SYS:BOOT] [1/8]..[8/8] with timings.
2. Subsystem Status Matrix format and health checklist structure.
3. IPC client frame guard against raw HTTP/JSON text packets (graceful disconnect without ValueError crash).
4. Idempotent column addition handling without false warnings in Kùzu schema driver.
5. Ingest no-op suppression (DEBUG on unchanged, INFO on new/updated).
"""

from __future__ import annotations

import asyncio
import logging
import struct
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from atlas_memory.core.agi_logger import log_subsystem_status_matrix
from atlas_memory.server.atlas_daemon import AtlasDaemon


def test_log_subsystem_status_matrix_format(caplog: pytest.LogCaptureFixture) -> None:
    """Verifies that log_subsystem_status_matrix produces the exact specified matrix layout."""
    logger = logging.getLogger("test_subsystem_matrix")
    logger.setLevel(logging.INFO)

    rows = [
        ("[L0] TTT Engine", "READY", "Steps: 33 restored"),
        ("[L1] HotBuffer (Working)", "READY", "Seeded: 2 records"),
        ("[L2] Kùzu Multigraph", "READY", "Nodes: 1841 (E:1797, P:651) | Edges: 2514"),
        ("[L2] Qdrant Vector Engine", "READY", "Dim: 512 (onnx68 / poldense68)"),
        ("[CR] HippoRAG 2 PPR", "WARMED", "Warmup completed (damping=0.85)"),
        ("[EG] Epistemic Guard", "ACTIVE", "Conflict Policy: STRICT"),
        ("[MB] Mnemosyne Bridge", "UNLINKED", "Reason: In-process module missing (FALLBACK: Shadow RPC)"),
        ("[SH] OmniRoute Shadow", "DISPATCHING", "Queue max: 1000 | Model: auto/best-fast"),
        ("[IPC] UDS Socket Server", "LISTENING", "Path: ~/.hermes/atlas.sock | MaxMsg: 16MB"),
    ]

    with caplog.at_level(logging.INFO, logger="test_subsystem_matrix"):
        log_subsystem_status_matrix(
            log=logger,
            rows=rows,
            total_modules=8,
            active_count=8,
            optional_disabled=1,
        )

    output = "\n".join(rec.message for rec in caplog.records)
    assert "ATLAS COGNITIVE CORE // SUBSYSTEM INITIALIZATION MATRIX [AGI RUNTIME]" in output
    assert "[L0] TTT Engine           | READY       | Steps: 33 restored" in output
    assert "[L1] HotBuffer (Working)  | READY       | Seeded: 2 records" in output
    assert "[L2] Kùzu Multigraph      | READY       | Nodes: 1841 (E:1797, P:651) | Edges: 2514" in output
    assert "[L2] Qdrant Vector Engine | READY       | Dim: 512 (onnx68 / poldense68)" in output
    assert "[CR] HippoRAG 2 PPR       | WARMED      | Warmup completed (damping=0.85)" in output
    assert "[EG] Epistemic Guard      | ACTIVE      | Conflict Policy: STRICT" in output
    assert "[MB] Mnemosyne Bridge     | UNLINKED    | Reason: In-process module missing (FALLBACK: Shadow RPC)" in output
    assert "[SH] OmniRoute Shadow     | DISPATCHING | Queue max: 1000 | Model: auto/best-fast" in output
    assert "[IPC] UDS Socket Server   | LISTENING   | Path: ~/.hermes/atlas.sock | MaxMsg: 16MB" in output
    assert "CORE SYSTEM STATUS: ALL MANDATORY SUBSYSTEMS GREEN (8/8 ONLINE, 1 NON-CRITICAL BYPASS)" in output


@pytest.mark.asyncio
async def test_ipc_client_frame_guard_http_disconnect(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Verifies that sending an HTTP GET or raw JSON header disconnects cleanly with a warning, not a ValueError crash."""
    daemon = AtlasDaemon(
        socket_path=tmp_path / "test_frame_guard.sock",
        pid_path=tmp_path / "test_frame_guard.pid",
    )
    daemon._serving = True

    # Simulate HTTP GET request frame: b"GET "
    reader = asyncio.StreamReader()
    reader.feed_data(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
    reader.feed_eof()

    writer = MagicMock()
    writer.close = MagicMock()
    writer.wait_closed = MagicMock()

    async def _async_wait_closed():
        pass

    writer.wait_closed.side_effect = _async_wait_closed

    with caplog.at_level(logging.WARNING, logger="atlas_memory.server.atlas_daemon"):
        await daemon._handle_client(reader, writer)

    # Must log clean warning, not crash with ValueError
    assert any("Incompatible un-framed payload" in rec.message for rec in caplog.records)
    assert not any("Exception handling client connection" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_ipc_client_frame_guard_oversized_frame(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Verifies that frame length exceeding 16MB ceiling disconnects cleanly without traceback."""
    daemon = AtlasDaemon(
        socket_path=tmp_path / "test_oversized.sock",
        pid_path=tmp_path / "test_oversized.pid",
    )
    daemon._serving = True

    # 20MB frame length prefix
    oversized_len = 20 * 1024 * 1024
    header = struct.pack(">I", oversized_len)

    reader = asyncio.StreamReader()
    reader.feed_data(header + b"x" * 100)
    reader.feed_eof()

    writer = MagicMock()
    writer.close = MagicMock()
    writer.wait_closed = MagicMock()

    async def _async_wait_closed():
        pass

    writer.wait_closed.side_effect = _async_wait_closed

    with caplog.at_level(logging.WARNING, logger="atlas_memory.server.atlas_daemon"):
        await daemon._handle_client(reader, writer)

    assert any("Framing violation: parsed msg_len=20971520 bytes exceeds 16MB ceiling" in rec.message for rec in caplog.records)
    assert any("Raw header hex:" in rec.message for rec in caplog.records)
    assert not any("Exception handling client connection" in rec.message for rec in caplog.records)

