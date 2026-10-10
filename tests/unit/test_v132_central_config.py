"""Unit tests for centralized runtime configuration (AtlasConfig / Pydantic v2)."""
from __future__ import annotations

import pytest

from atlas_memory.cognitive.retrieval.ppr import is_hipporag_enabled
from atlas_memory.core.config import AtlasConfig, get_config, parse_bool, reset_config
from atlas_memory.orchestrator import MemoryOrchestrator
from atlas_memory.server.atlas_daemon import AtlasDaemon


def test_parse_bool_variants() -> None:
    assert parse_bool("1") is True
    assert parse_bool("true") is True
    assert parse_bool("TRUE") is True
    assert parse_bool("yes") is True
    assert parse_bool("on") is True
    assert parse_bool(True) is True
    assert parse_bool(1) is True

    assert parse_bool("0") is False
    assert parse_bool("false") is False
    assert parse_bool("FALSE") is False
    assert parse_bool("no") is False
    assert parse_bool("off") is False
    assert parse_bool(False) is False
    assert parse_bool(0) is False

    assert parse_bool(None, default=True) is True
    assert parse_bool("", default=False) is False
    assert parse_bool("unknown", default=True) is True


def test_atlas_config_defaults() -> None:
    cfg = AtlasConfig()
    assert cfg.hipporag_enabled is False
    assert cfg.hipporag_rm_enabled is False
    assert cfg.ppr_backend == "igraph"
    assert cfg.fusion_strategy == "dbsf"
    assert cfg.enable_calibrated_thresholds is True
    assert cfg.shadow_rpc_enabled is True
    assert cfg.shadow_worker_enabled is True
    assert cfg.omniroute_url == "http://localhost:20128/v1"
    assert cfg.kuzu_buffer_mb == 1024


def test_atlas_config_from_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATLAS_HIPPORAG", "0")
    monkeypatch.setenv("ATLAS_KUZU_BUFFER_MB", "512")
    monkeypatch.setenv("ATLAS_FUSION_STRATEGY", "rrf")
    monkeypatch.setenv("ATLAS_ENABLE_CALIBRATED_THRESHOLDS", "false")
    monkeypatch.setenv("ATLAS_OMNIRUTE_URL", "http://localhost:9999/v1")

    cfg = AtlasConfig.from_env()
    assert cfg.hipporag_enabled is False
    assert cfg.kuzu_buffer_mb == 512
    assert cfg.fusion_strategy == "rrf"
    assert cfg.enable_calibrated_thresholds is False
    assert cfg.omniroute_url == "http://localhost:9999/v1"


def test_singleton_get_and_reset_config() -> None:
    reset_config(None)
    c1 = get_config()
    c2 = get_config()
    assert c1 is c2

    custom = AtlasConfig(kuzu_buffer_mb=2048)
    reset_config(custom)
    assert get_config().kuzu_buffer_mb == 2048
    reset_config(None)


def test_wiring_parity_between_daemon_and_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test that both daemon and orchestrator read the exact same canonical configuration
    reset_config(None)
    monkeypatch.setenv("ATLAS_ENABLE_CALIBRATED_THRESHOLDS", "1")
    cfg = get_config()
    assert cfg.enable_calibrated_thresholds is True

    orc = MemoryOrchestrator()
    assert orc.enable_calibrated_thresholds is True

    daemon = AtlasDaemon(engine=None, orchestrator=orc)
    assert daemon.enable_calibrated_thresholds is True


def test_hipporag_enabled_resolves_via_config(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_config(None)
    monkeypatch.setenv("ATLAS_HIPPORAG", "1")
    reset_config(None)
    assert is_hipporag_enabled() is True

    monkeypatch.setenv("ATLAS_HIPPORAG", "0")
    reset_config(None)
    assert is_hipporag_enabled() is False
    reset_config(None)
