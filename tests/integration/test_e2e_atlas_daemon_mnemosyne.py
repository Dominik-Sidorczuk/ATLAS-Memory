"""E2E Integration Test: ATLAS Daemon with Mnemosyne Ingestion and UDS IPC."""

import asyncio
import concurrent.futures
from pathlib import Path
import pytest

from atlas_memory.ingest.mnemosyne_ingest import MnemosyneIngestEngine
from atlas_memory.server.atlas_daemon import AtlasDaemon
from atlas_memory.server.client import AtlasDaemonClient
from atlas_memory.hermes.atlas_provider import AtlasMemoryProvider


@pytest.mark.asyncio
async def test_mnemosyne_ingestion_and_daemon_rpc(tmp_path: Path):
    sock_path = tmp_path / "test_atlas.sock"
    pid_path = tmp_path / "test_atlas.pid"
    atlas_dir = tmp_path / "atlas_vault"

    daemon = AtlasDaemon.create_default(
        socket_path=sock_path,
        pid_path=pid_path,
        atlas_dir=atlas_dir,
        sync_legacy_mnemosyne=True,
    )
    await daemon.start()
    try:
        assert sock_path.exists()

        client = AtlasDaemonClient(socket_path=sock_path)
        ping = await client.call("ping")
        assert ping == {"status": "ok"}

        stats = await client.call("get_stats")
        assert stats["status"] == "ok"
        assert stats["serving"] is True

        # Commit an observation via UDS
        commit_res = await client.call("commit_observation", {
            "subject": "Peptide_GHK_Cu",
            "predicate": "stimulates",
            "object": "Collagen synthesis and tissue remodeling",
            "confidence": 0.95,
            "source": "user_explicit",
        })
        assert commit_res["status"] == "ok"

        # Prefetch query via UDS client
        prefetch_res = await client.call("prefetch", {"query": "Peptide_GHK_Cu collagen synthesis"})
        assert prefetch_res["count"] >= 1
        assert "Peptide_GHK_Cu" in prefetch_res["context_block"]

        # Run sync provider prefetch in a separate thread so asyncio loop continues serving
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            def _sync_call():
                provider = AtlasMemoryProvider(socket_path=str(sock_path))
                return provider.prefetch("Peptide_GHK_Cu collagen")
            
            ctx = await asyncio.get_running_loop().run_in_executor(pool, _sync_call)
            assert len(ctx) > 0
            assert "Peptide_GHK_Cu" in ctx

    finally:
        await daemon.stop()
