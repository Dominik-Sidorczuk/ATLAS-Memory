"""V89.1 — konfiguracja LLM ekstraktora cieni (regresja: model 'qwen-mini' nie istnial
na OmniRoute -> kazde wywolanie 400 -> cichy regex_fallback w produkcji).

Testy OFFLINE (bez sieci): spawnuja worker i sprawdzaja rezołwianie configu.
Dowod zycia LLM jest osobno: live RPC shadow_drain na daemonie (raport V89).
"""
from __future__ import annotations

import pytest

from atlas_memory.active.shadow_worker import OmniRouteShadowWorker


class _StubOrch:
    def _fallback_extract_facts(self, *_a, **_k):
        return []


def _worker(**kw) -> OmniRouteShadowWorker:
    return OmniRouteShadowWorker(orchestrator=_StubOrch(), **kw)  # type: ignore[arg-type]


def test_default_model_is_a_live_omniroute_alias_not_dead_qwen_mini() -> None:
    w = _worker()
    assert w.model != "qwen-mini", "martwy id modelu => 400 => cichy fallback regex"
    assert w.model, "model nie moze byc pusty"


def test_env_shadow_model_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATLAS_SHADOW_MODEL", "auto/best-coding")
    assert _worker().model == "auto/best-coding"


def test_env_api_base_typo_alias_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ATLAS_OMNIRROUTE_URL", raising=False)
    monkeypatch.setenv("ATLAS_OMNIRUTE_URL", "http://127.0.0.1:9999/v1/")
    assert _worker().api_base == "http://127.0.0.1:9999/v1"


def test_env_api_base_prefers_spelled_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATLAS_OMNIRROUTE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("ATLAS_OMNIRUTE_URL", "http://127.0.0.1:2/v1")
    assert _worker().api_base == "http://127.0.0.1:1/v1"


def test_explicit_api_base_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATLAS_OMNIRROUTE_URL", "http://127.0.0.1:1/v1")
    assert _worker(api_base="http://127.0.0.1:3/v1").api_base == "http://127.0.0.1:3/v1"


def test_shadow_rpc_handlers_registered_on_daemon() -> None:
    from atlas_memory.server.atlas_daemon import AtlasDaemon

    daemon = AtlasDaemon()
    for method in ("shadow_enqueue", "shadow_drain", "shadow_status"):
        assert method in daemon._handlers
