"""V111 F-021 — GC sierot Kùzu z izolacją wyjątku per-node.

Root cause (udowodniony na klonie Kùzu): węzeł 'CORS' offset 1130 = TOCTOU —
krawędź wstrzyknięta między zbieranie sierot a DELETE → RuntimeError z jednego
DELETE przerywał CAŁĄ pętlę pod jednym outer try, reszta sierot zostawała
nietknięta, GC był cichy (zwracał []).

V121 (T3 / F-034): przed per-node DELETE wykonywany jest edge-count re-check
(`MATCH (e:Entity {name: $name})-[r]-() RETURN count(r)`). W tym teście
wszystkie węzły w `orphan_names` są sierotami (brak krawędzi), więc re-check
zwraca 0 i pętla przechodzi do DELETE.

Kontrakty testowane (unit, fake conn — bez demona, bez.socketa, bez bazy):
 (a) funkcja NIE propaguje wyjątku z pojedynczego DELETE;
 (b) zwraca listę faktycznie skasowanych ['A', 'C'] (NIE []);
 (c) sygnatura publiczna metody bez zmian;
 (d) awaria SELECT (silnik martwy) nadal abortuje -> [] (outer try zachowany);
 (e) per-node porażka jest logowana (telemetria nie może być cicha).
"""
from __future__ import annotations

import inspect
import logging
from typing import Any, List

from atlas_memory.l2_semantic.kuzu.schema_driver import KuzuSchemaDriverMixin


class FakeResult:
    """Udaje kursor Kùzu zużywany przez _drain: has_next()/get_next()."""

    def __init__(self, rows: List[List[Any]]) -> None:
        self._rows = list(rows)

    def has_next(self) -> bool:
        return bool(self._rows)

    def get_next(self) -> List[Any]:
        return self._rows.pop(0)


class FakeConn:
    """Sterowany awariami sternik połączenia Kùzu.

    SELECT sierot zwraca `orphan_names`; DELETE dla nazwy z `failing_deletes`
    rzuca RuntimeError('connected edges fwd'); pozostałe DELETE przechodzą.
    Awaria SELECT (select_raises) modeluje martwy silnik.
    """

    def __init__(
        self,
        orphan_names: List[str],
        failing_deletes: List[str] | None = None,
        select_raises: bool = False,
    ) -> None:
        self.orphan_names = orphan_names
        self.failing_deletes = failing_deletes or []
        self.select_raises = select_raises
        self.delete_calls: List[str] = []

    def execute(self, query: str, params: Any = None) -> FakeResult:
        if "RETURN e.name" in query:
            if self.select_raises:
                raise RuntimeError("database is closed")
            return FakeResult([[name] for name in self.orphan_names])
        if "RETURN count(r)" in query:
            # V121 (T3): edge re-check przed DELETE. W tym scenariuszu TOCTOU jest
            # tak szybkie, że krawędź pojawia się MIĘDZY re-check a DELETE — re-check
            # odpowiada 0 i pętla idzie do DELETE, który rzuca dla 'BAD'. Izolacja
            # V111 (per-node try) nadal musi łapać.
            return FakeResult([[0]])
        if "DELETE e" in query:
            name = str((params or {}).get("name", ""))
            self.delete_calls.append(name)
            if name in self.failing_deletes:
                raise RuntimeError("connected edges fwd")
            return FakeResult([])
        raise AssertionError(f"nieoczekiwany zapytanie w teście: {query[:60]}")


def _driver(conn: FakeConn) -> KuzuSchemaDriverMixin:
    drv = KuzuSchemaDriverMixin.__new__(KuzuSchemaDriverMixin)
    drv.conn = conn  # type: ignore[attr-defined]
    return drv


def test_b1_single_bad_delete_does_not_abort_whole_gc_loop():
    """(a)+(b): 'BAD' rzuca, 'A' i 'C' przechodzą — deleted == ['A','C'], brak propagacji."""
    conn = FakeConn(orphan_names=["A", "BAD", "C"], failing_deletes=["BAD"])
    drv = _driver(conn)

    deleted = drv._collect_and_delete_orphans_on_disk()  # (c) wywołanie bez zmian sygnatury

    # V114: batch UNWIND idzie pierwszy; ten fake zwraca pusty wynik dla batcha
    # (0 nazw != 3 orphans) -> driver PADNA do fallbacku per-node V111, który
    # izoluje 'BAD'. Kontrakt izolacji per-node więc NADAL obowiązuje — tylko
    # ścieżka dojścia do niego zmieniła się (batch-attempt → fallback).
    assert deleted == ["A", "C"], f"per-node izolacja wymagana; dostałem {deleted!r}"
    assert "BAD" in conn.delete_calls, "fallback musiał spróbować każdej sieroty per-node"


def test_b2_select_failure_still_aborts_with_empty_list():
    """(d): awaria SELECT = martwy silnik — outer try zachowany, zwrot []."""
    conn = FakeConn(orphan_names=[], select_raises=True)
    drv = _driver(conn)

    deleted = drv._collect_and_delete_orphans_on_disk()

    assert deleted == []
    assert conn.delete_calls == [], "przy martwym SELECT nie wolno nic kasować"


def test_b3_per_node_failure_is_logged_not_silent(caplog):
    """(e): porażka per-node MUSI być zalogowana warningiem z nazwą (koniec z cichym GC)."""
    conn = FakeConn(orphan_names=["A", "BAD", "C"], failing_deletes=["BAD"])
    drv = _driver(conn)

    with caplog.at_level(logging.WARNING):
        deleted = drv._collect_and_delete_orphans_on_disk()

    assert deleted == ["A", "C"]
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("BAD" in m for m in warns), f"oczekiwano warningu o 'BAD', jest: {warns}"


def test_b4_public_signature_unchanged():
    """(c): sygnatura metody bez zmian — tylko self, zwraca List[str]."""
    sig = inspect.signature(KuzuSchemaDriverMixin._collect_and_delete_orphans_on_disk)
    assert list(sig.parameters) == ["self"]
    # 'from __future__ import annotations' powoduje stringową annotację 'List[str]'
    assert sig.return_annotation == "List[str]"
