"""V112 F-028+ fail-loud-or-evidence: ciche polkniecia zostawiaja dowod.

Bramka mutacyjna: deprecated `except Exception: pass` w hot-path MUSI wywołać
record_swallow (event 'swallow' zarejestrowany), a record_swallow NIGDY nie
może propagować.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from atlas_memory.l2_semantic.audit_ledger import (
    SWALLOW_EVENT_TYPE,
    record_swallow,
    reset_swallow_state,
    set_swallow_sink,
)
from atlas_memory.l2_semantic.kv_store import VerifiedKVStore


@pytest.fixture(autouse=True)
def _reset_swallow():
    reset_swallow_state()
    yield
    reset_swallow_state()


class _SinkCollector:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def __call__(self, payload: dict) -> None:
        self.events.append(payload)


def test_t1_kv_swallow_emits_evidence_event() -> None:
    """T1: KV write-path (json.loads != JSONDecodeError) rejestruje event 'swallow'."""
    sink = _SinkCollector()
    set_swallow_sink(sink)

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE t (key TEXT, value TEXT, value_hash TEXT, metadata TEXT)")
    conn.execute("INSERT INTO t VALUES (?, ?, ?, ?)", ("hermes:test:swallow", "{}", None, "{}"))
    real_row = conn.execute("SELECT * FROM t").fetchone()

    import atlas_memory.l2_semantic.kv_store as kvmod
    orig = kvmod.json.loads

    def boom(_s, *a, **k):
        raise RuntimeError("disk fault simulation")

    kvmod.json.loads = boom
    try:
        VerifiedKVStore._row_to_state_dict(real_row)
    except Exception:
        pass
    finally:
        kvmod.json.loads = orig

    swallow_events = [
        e for e in sink.events
        if e.get("event") == SWALLOW_EVENT_TYPE and "kv_store" in str(e.get("module", ""))
    ]
    assert swallow_events, f"swallow nie zarejestrowany: {sink.events!r}"
    # oba swallowe w _row_to_state_dict (value + meta) muszą zostawić dowód
    assert {e["module"] for e in swallow_events} == {
        "l2_semantic.kv_store._row_to_state_dict",
        "l2_semantic.kv_store._row_to_state_dict.meta",
    }


def test_t2_record_swallow_never_propagates() -> None:
    """T2: record_swallow NIE MOŻE wywalić hot-path przy awarii sinka ani pliku."""

    def broken_sink(_payload: dict) -> None:
        raise RuntimeError("ledger dropped")

    set_swallow_sink(broken_sink)
    record_swallow("test.module.a", "k1", RuntimeError("boom"))
    record_swallow("test.module.b", "k2", ValueError("valexc"))
    record_swallow("test.module.c", "k3", None)  # exc=None


def test_t3_record_swallow_default_jsonl_fallback(tmp_path, monkeypatch) -> None:
    """T3: bez sinka — JSONL fallback na dysku (env-var konfigurowalne)."""
    monkeypatch.setenv("ATLAS_SWALLOW_LEDGER", str(tmp_path / "swallow.jsonl"))
    record_swallow("t3.module", "k3", KeyError("nope"))
    data = (tmp_path / "swallow.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(data) >= 1
    ev = json.loads(data[-1])
    assert ev["event"] == SWALLOW_EVENT_TYPE and ev["module"] == "t3.module"


def test_t4_happy_path_zero_swallow_events() -> None:
    """T4-regresja: normalny zapis/odczyt bez błędów = 0 zdarzeń swallow w KV."""
    sink = _SinkCollector()
    set_swallow_sink(sink)
    store = VerifiedKVStore(db_path=":memory:")
    store.set_sync(key="hermes:test:plain", value={"a": 1}, confidence=0.9)
    stored = store.get_sync("hermes:test:plain")
    assert stored is not None and stored.get("value") == {"a": 1}
    kv_swallow = [
        e for e in sink.events
        if e.get("event") == SWALLOW_EVENT_TYPE and "kv_store" in str(e.get("module", ""))
    ]
    assert not kv_swallow, f"unexpected swallow on happy path: {kv_swallow!r}"


def test_t5_regression_v111_empty_pool_still_rejected() -> None:
    """Regresja V111: bramka empty_pool przy pustej puli kandydatów nietknięta."""
    from atlas_memory.cognitive.retrieval import gate as gate_mod

    d = gate_mod.evaluate_gate("jak działa proces kalibracji ATLAS", [], calibration_engine=None)
    assert d.should_retrieve is False
    assert d.reason == "empty_pool"


def test_t6_dedupe_same_signature_one_event() -> None:
    """T6: identyczny (module, key, exc_type) NIE jest dublowany w ledgerze."""
    sink = _SinkCollector()
    set_swallow_sink(sink)
    for _ in range(5):
        record_swallow("dedup.module", "same-key", ValueError("dup"))
    relevant = [e for e in sink.events if e.get("module") == "dedup.module"]
    assert len(relevant) == 1, f"dedupe nie zadzialal: {relevant!r}"


def test_t7_internal_ledger_failure_no_propagation(tmp_path, monkeypatch) -> None:
    """T7: gdy oba (sink i JSONL) zawiodą, record_swallow NIE propaguje."""
    monkeypatch.setenv("ATLAS_SWALLOW_LEDGER", "/proc/cannot/write/at/all/swallow.jsonl")

    def broken(_payload: dict) -> None:
        raise RuntimeError("both fail")

    set_swallow_sink(broken)
    record_swallow("t7.module", "k", OSError("disk full"))  # nie powinno rzucic
