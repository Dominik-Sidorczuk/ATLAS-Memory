"""V121 (T3 / F-034) — RED-first test: GC skip non-orphan before delete (kills 1130 retry storm).

Kontrakt:
  - candidate set zawiera NIE-orphan (węzeł z krawędziami wstrzykniętymi między SELECT a DELETE — TOCTOU).
  - przed per-node DELETE wykonywany jest edge-count re-check; jeśli edges>0 → skip z logiem 'skipped_non_orphan'.
  - per-node fallback NIE propaguje RuntimeError dla takich węzłów (V111 invariant).
  - nie używamy DETACH DELETE (żywe krawędzie to dane).
  - orphan jest usuwany; non-orphan zostaje wraz z krawędzią (nietknięty).

RED (przed fixem):
  - 'skipped_non_orphan' brak w logach.
  - dla węzła z krawędziami DELETE rzuca RuntimeError (TOCTOU) → catch + warning 'kasowanie sieroty nieudane'.
  - non-orphan jest nadmiernie raportowany (wliczony do deleted) lub loguje ostrzeżenie.

GREEN (po fixie):
  - 'skipped_non_orphan' obecny w logach (info level).
  - brak 'kasowanie sieroty ... nieudane' warning dla non-orphan.
  - deleted == ['ORPHAN']; non-orphan pozostaje + edge w bazie nietknięta.
"""
from __future__ import annotations

import inspect
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


class _GCv121FakeConn:
    """Sterowany sternik dla GC. Symuluje TOCTOU:
      - SELECT zwraca ORPHAN + NON_ORPHAN (bo edge jeszcze nie istniał w SELECT, ale istnieje w DELETE).
      - batch UNWIND: 'UNWIND ... DELETE e RETURN nm' WHERE filtruje i zwraca tylko ORPHAN → batch_returns_full=False.
      - per-node: dla NON_ORPHAN edge-count query zwraca 1 (TOCTOU: edge wstrzyknięta);
        a DELETE dla NON_ORPHAN rzuca RuntimeError (symulacja realnego zachowania Kùzu).
      - per-node: dla ORPHAN edge-count=0, DELETE nic nie robi (no row), bez wyjątku.

    Implementuje _drain przez konsumpcję wszystkich rows w has_next/get_next.
    """

    def __init__(self) -> None:
        self.execute_count = 0
        self.execute_calls: List[str] = []
        self.orphan_names_in_select = ["ORPHAN", "NON_ORPHAN"]

    def execute(self, query: str, params: Any = None) -> _FakeResult:
        self.execute_count += 1
        first = query.strip().split()[0] if query.strip() else ""
        self.execute_calls.append(first + ":" + ("batch" if "UNWIND" in query else "sel" if "RETURN e.name" in query else "delete" if "DELETE e" in query else "check" if "RETURN count(r)" in query else "?"))
        if "RETURN e.name" in query:
            return _FakeResult([[n] for n in self.orphan_names_in_select])
        if "UNWIND" in query and "DELETE" in query:
            # batch z WHERE — filtruje → zwraca tylko ORPHAN (partial batch)
            return _FakeResult([["ORPHAN"]])
        if "RETURN count(r)" in query and (params or {}).get("name") == "NON_ORPHAN":
            return _FakeResult([[1]])
        if "RETURN count(r)" in query and (params or {}).get("name") == "ORPHAN":
            return _FakeResult([[0]])
        if "DELETE e" in query and "UNWIND" not in query:
            name = str((params or {}).get("name", ""))
            if name == "NON_ORPHAN":
                raise RuntimeError("Node(nodeOffset: 1130) has connected edges in table RELATION in the fwd direction")
            return _FakeResult([])
        raise AssertionError(f"unexpected query: {query[:120]}")


def _driver(conn: Any) -> KuzuSchemaDriverMixin:
    drv = KuzuSchemaDriverMixin.__new__(KuzuSchemaDriverMixin)
    drv.conn = conn  # type: ignore[attr-defined]
    return drv


def test_t3_skip_non_orphan_emits_log_and_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    """RED: obecny kod loguje 'kasowanie sieroty ... nieudane (TOCTOU)' zamiast 'skipped_non_orphan' info."""
    conn = _GCv121FakeConn()
    drv = _driver(conn)

    with caplog.at_level(logging.DEBUG, logger="atlas_memory.l2_semantic.kuzu.schema_driver"):
        deleted = drv._collect_and_delete_orphans_on_disk()

    # After fix: ORPHAN deleted, NON_ORPHAN skipped via edge-count re-check.
    assert "ORPHAN" in deleted, f"ORPHAN musi zostać skasowany; deleted={deleted!r}"
    assert "NON_ORPHAN" not in deleted, f"NON_ORPHAN nie może być raportowany jako skasowany; deleted={deleted!r}"

    # 'skipped_non_orphan' must appear (RED: missing; GREEN: present).
    messages = [r.getMessage() for r in caplog.records]
    assert any("skipped_non_orphan" in m for m in messages), (
        f"oczekiwano info log 'skipped_non_orphan'; caplog={messages}"
    )

    # No 'kasowanie sieroty ... nieudane (TOCTOU?)' warning for NON_ORPHAN — fix pomija DELETE.
    assert not any("kasowanie sieroty" in m and "NON_ORPHAN" in m for m in messages), (
        f"per-node DELETE dla NON_ORPHAN nie powinien być próbowany (TOCTOU pre-check); caplog={messages}"
    )


def test_t3_skip_non_orphan_no_runtime_error(caplog: pytest.LogCaptureFixture) -> None:
    """Per-node fallback NIE propaguje RuntimeError — V111 invariant (bez zmian), ale fix musi
    powodować że DELETE nie jest wołany dla węzła z krawędziami (więc NIE MA wyjątku w ogóle).
    """
    conn = _GCv121FakeConn()
    drv = _driver(conn)

    with caplog.at_level(logging.DEBUG):
        # Brak raise'u na zewnątrz (V111 outer try zachowany).
        result = drv._collect_and_delete_orphans_on_disk()

    assert isinstance(result, list)


def test_t3_existing_v111_fallback_isolation_preserved() -> None:
    """V111 kontrakt: inny niepowiązany wyjątek per-node jest łapany, GC nie przerywa się.
    Ten test sprawdza, że fix nie wprowadza 'propagacji' RuntimeError w nowej ścieżce.
    """
    # Użyj oryginalnego V114-style fake (batch_raises=True, failing_per_node=['BAD']).
    # Po naszym fixie: dla BAD-a, edge-count zwraca 0 (brak krawędzi w fake),
    # więc DELETE jest próbowany → nadal rzuca → łapany → deleted = ['A','C'].
    from tests.test_kuzu_batch_gc_v114 import _BatchFakeConn
    from tests.test_kuzu_batch_gc_v114 import _driver as _v114_driver

    class _CountingFakeConn(_BatchFakeConn):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        def execute(self, query, params=None):
            if "RETURN count(r)" in query:
                # Brak krawędzi w symulowanym DB (wszystkie to orphany).
                return _FakeResult([[0]])
            return super().execute(query, params)

    conn = _CountingFakeConn(orphan_names=["A", "B", "C"], batch_raises=True, failing_per_node=["B"])
    drv = _v114_driver(conn)
    deleted = drv._collect_and_delete_orphans_on_disk()
    assert sorted(deleted) == ["A", "C"], f"V111 izolacja per-node zachowana; deleted={deleted!r}"


def test_t3_signature_unchanged() -> None:
    sig = inspect.signature(KuzuSchemaDriverMixin._collect_and_delete_orphans_on_disk)
    assert list(sig.parameters) == ["self"]
    assert sig.return_annotation == "List[str]"
