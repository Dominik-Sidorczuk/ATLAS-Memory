"""FIX-4 (F-005 & F-007): Default `ATLAS_HIPPORAG=0` & `get_stats` Observability Tests.

Scenarios:
  a) `ATLAS_HIPPORAG=0` + `get_stats` -> `hybrid_channels_log` does NOT contain "graph_ppr";
     `dual_node_coverage == 0.0` before migration and `1.0` after migration.
  b) `ATLAS_HIPPORAG=1` + `get_stats` -> `hybrid_channels_log["graph_ppr"]` is a list of length >= 1
     after calling recall/prefetch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas_memory.cognitive.retrieval.ppr import (
    is_hipporag_enabled,
    reset_hipporag_telemetry,
)
from atlas_memory.server.atlas_daemon import AtlasDaemon


@pytest.mark.asyncio
async def test_get_stats_observability_off_and_on(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ATLAS_HIPPORAG", raising=False)
    assert is_hipporag_enabled() is False, "Default ATLAS_HIPPORAG must be 0 (OFF)"

    daemon = AtlasDaemon.create_default(
        atlas_dir=tmp_path / "atlas_obs",
        socket_path=tmp_path / "obs.sock",
        ingest_mnemosyne_on_start=False,
    )
    try:
        # Scenario (a) before migration: ATLAS_HIPPORAG=0 -> dual_node_coverage == 0.0, no "graph_ppr"
        reset_hipporag_telemetry()
        monkeypatch.setenv("ATLAS_HIPPORAG", "0")
        stats_pre = await daemon._handle_get_stats({})
        assert stats_pre["ppr_backend"] == "igraph"
        assert "graph_ppr" not in stats_pre["hybrid_channels_log"]
        assert stats_pre["dual_node_coverage"] == 0.0

        # Populate a dual-node record via _handle_set (uses EpistemicWritePath FIX-1)
        await daemon._handle_set(
            {
                "key": "fact:obs_test",
                "value": "System telemetrii HippoRAG monitoruje pokrycie węzłów dual-node.",
                "session_id": "obs_session",
                "confidence": 0.95,
            }
        )
        await daemon._handle_prefetch({"query": "telemetria HippoRAG dual-node", "session_id": "obs_session"})
        stats_off = await daemon._handle_get_stats({})
        assert "graph_ppr" not in stats_off["hybrid_channels_log"]
        assert stats_off["dual_node_coverage"] == 1.0

        # Scenario (b): ATLAS_HIPPORAG=1 -> after recall/prefetch, hybrid_channels_log["graph_ppr"] has >= 1 entry
        monkeypatch.setenv("ATLAS_HIPPORAG", "1")
        await daemon._handle_prefetch({"query": "telemetria HippoRAG dual-node", "session_id": "obs_session"})
        stats_on = await daemon._handle_get_stats({})
        assert "graph_ppr" in stats_on["hybrid_channels_log"]
        assert isinstance(stats_on["hybrid_channels_log"]["graph_ppr"], list)
        assert len(stats_on["hybrid_channels_log"]["graph_ppr"]) >= 1
        assert stats_on["dual_node_coverage"] == 1.0
    finally:
        await daemon.stop()
