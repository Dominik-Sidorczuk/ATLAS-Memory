"""V123 (T2) — Kùzu GC TOCTOU Exception Logging & Graceful Isolation.

Contracts tested:
  1. When batch UNWIND raises 'has connected edges', it is logged as INFO (expected MVCC TOCTOU race) without emitting a scary WARNING.
  2. Unrelated batch exceptions continue to log WARNING.
  3. Orphan deletion continues to work cleanly via per-node fallback.
"""
from __future__ import annotations

import logging
from typing import Any, List

import pytest

from atlas_memory.l2_semantic.kuzu.schema_driver import KuzuSchemaDriverMixin


class _FakeResult:
    def __init__(self, rows: List[List[Any]]) -> None:
        self._rows = list(rows)

    def has_next(self) -> bool:
        return bool(self._rows)

    def get_next(self) -> List[Any]:
        return self._rows.pop(0)


class _KuzuConnMock:
    def __init__(self, batch_error_msg: str) -> None:
        self.batch_error_msg = batch_error_msg
        self.execute_calls: List[str] = []

    def execute(self, query: str, params: Any = None) -> _FakeResult:
        self.execute_calls.append(query)
        if "MATCH (e:Entity) WHERE NOT (e)-[:RELATION]-()" in query:
            return _FakeResult([["ORPHAN_A"], ["ORPHAN_B"]])
        if "UNWIND $names AS nm" in query:
            raise RuntimeError(self.batch_error_msg)
        if "RETURN count(r)" in query:
            return _FakeResult([[0]])
        if "DELETE e RETURN nm" in query:
            return _FakeResult([[params.get("name")]])
        return _FakeResult([])


def _make_driver(conn: Any) -> KuzuSchemaDriverMixin:
    drv = KuzuSchemaDriverMixin()
    drv.conn = conn
    return drv


def test_t2_connected_edges_logged_as_info(caplog: pytest.LogCaptureFixture) -> None:
    conn = _KuzuConnMock("Runtime exception: Node(nodeOffset: 527) has connected edges in table RELATION in the fwd direction, which cannot be deleted.")
    drv = _make_driver(conn)

    with caplog.at_level(logging.INFO):
        deleted = drv._collect_and_delete_orphans_on_disk()

    assert deleted == ["ORPHAN_A", "ORPHAN_B"]

    # Verify INFO log was emitted
    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert any("napotkał wstrzyknięcie współbieżnej krawędzi" in r.message for r in info_records)

    # Verify NO WARNING was emitted for the batch failure
    warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("batch GC nieudany" in r.message for r in warning_records)


def test_t2_unrelated_exception_logged_as_warning(caplog: pytest.LogCaptureFixture) -> None:
    conn = _KuzuConnMock("Disk full / I/O device error")
    drv = _make_driver(conn)

    with caplog.at_level(logging.INFO):
        deleted = drv._collect_and_delete_orphans_on_disk()

    assert deleted == ["ORPHAN_A", "ORPHAN_B"]

    # Verify WARNING was emitted for unexpected error
    warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("batch GC nieudany" in r.message for r in warning_records)
