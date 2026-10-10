"""Unit tests verifying F-036 active sensing writeback TTL."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from atlas_memory.cognitive.predictive_coder import PredictiveCoder
from atlas_memory.l2_semantic.kv_store import VerifiedKVStore


@pytest.mark.asyncio
async def test_active_sensing_writeback_calls_kv_with_ttl_3600() -> None:
    """F-036: Probe writeback must supply ttl=3600 and metadata['ttl']=3600."""
    mock_kv = MagicMock()
    mock_engine = MagicMock()
    mock_engine.kv = mock_kv
    mock_engine.graph = None

    coder = PredictiveCoder(engine=mock_engine)

    # Active sensing probe without expectations (triggers moderate error writeback)
    res = await coder.execute_sensing({
        "probe": "retrieval_health",
        "observed": "OK",
    })

    assert res["has_error"] is True
    assert res["severity"] in ("MODERATE", "CRITICAL")
    assert res["world_model_updated"] is True

    # Assert kv_store.set_sync was called with ttl=3600
    mock_kv.set_sync.assert_called_once()
    call_args, call_kwargs = mock_kv.set_sync.call_args
    assert call_args[0] == "probe:retrieval_health:last_observed"
    assert call_args[1] == "OK"
    assert call_kwargs.get("ttl") == 3600
    assert call_kwargs.get("metadata", {}).get("ttl") == 3600


@pytest.mark.asyncio
async def test_active_sensing_writeback_persists_ttl_in_real_kv(tmp_path: Path) -> None:
    """F-036: Probe writeback into real VerifiedKVStore must retain TTL metadata."""
    db_file = tmp_path / "test_ttl.db"
    kv = VerifiedKVStore(db_path=str(db_file))

    mock_engine = MagicMock()
    mock_engine.kv = kv
    mock_engine.graph = None

    coder = PredictiveCoder(engine=mock_engine)

    res = await coder.execute_sensing({
        "probe": "retrieval_health",
        "observed": "OK",
    })
    assert res["world_model_updated"] is True

    record = kv.get("probe:retrieval_health:last_observed")
    assert record is not None
    assert record["metadata"].get("ttl") == 3600
