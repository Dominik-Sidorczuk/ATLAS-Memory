"""End-to-end test: ATLAS daemon + Hermes memory provider."""
from __future__ import annotations

import asyncio
import os
import socket
from pathlib import Path

import pytest

from atlas_memory.hermes.atlas_provider import AtlasMemoryProvider
from atlas_memory.server.atlas_daemon import AtlasDaemon
from atlas_memory.server.client import AtlasDaemonClient


def test_atlas_socket_reachable():
    """Verify AtlasDaemon is running and socket reachable."""
    sock = Path(os.environ.get("ATLAS_SOCKET_PATH", str(Path.home() / ".hermes" / "atlas.sock")))
    if not sock.exists():
        pytest.skip("AtlasDaemon not running — run scripts/atlas_daemon_launcher.py first")
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(str(sock))
        s.close()
    except (ConnectionRefusedError, FileNotFoundError, OSError):
        pytest.skip("AtlasDaemon socket exists but daemon is not actively listening")
    except Exception as e:
        pytest.fail(f"Socket not reachable: {e}")


def test_atlas_memory_provider_import():
    """Verify AtlasMemoryProvider can be imported."""
    provider = AtlasMemoryProvider()
    assert provider.name == "atlas"


def test_atlas_daemon_client_roundtrip(tmp_path: Path):
    """Verify AtlasDaemon and AtlasDaemonClient can communicate over UDS."""
    sock_path = tmp_path / "test_atlas.sock"
    pid_path = tmp_path / "test_atlas.pid"

    async def _test():
        daemon = AtlasDaemon(socket_path=sock_path, pid_path=pid_path)
        await daemon.start()
        try:
            client = AtlasDaemonClient(socket_path=sock_path)
            res = await client.ping()
            assert res is True
        finally:
            await daemon.stop()

    asyncio.run(_test())
