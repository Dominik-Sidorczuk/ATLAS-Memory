"""V114 F-021-bis — Batch GC Kùzu z UNWIND + fallback per-node V111.

Kontrakty:
 (a) happy path: JEDEN batch UNWIND + jeden SELECT -> execute_count == 2,
 (b) partial batch (batch zwraca mniej nazw niż orphans) -> fallback per-node,
 (c) batch rzuca (TOCTOU / martwy silnik) -> fallback per-node V111,
 (d) fallback NIE propaguje wyjątków per-node (V111 kontrakt zachowany),
 (e) V111 regression: 'BAD' DELETE rzuca, 'A' i 'C' przechodzą,
 (f) awaria SELECT abortuje -> [] (outer try zachowany).
"""
from __future__ import annotations

import inspect
import logging
from typing import Any, List

from atlas_memory.l2_semantic.kuzu.schema_driver import KuzuSchemaDriverMixin


class _FakeResult:
    def __init__(self, rows: List[List[Any]]) -> None:
        self._rows = list(rows)

    def has_next(self) -> bool:
        return bool(self._rows)

    def get_next(self) -> List[Any]:
        return self._rows.pop(0)


class _BatchFakeConn:
    """Sterowany sternik dla batch GC. Zachowuje licznik execute() dla testu (a)."""

    def __init__(
        self,
        orphan_names: List[str],
        *,
        select_raises: bool = False,
        batch_raises: bool = False,
        batch_returns_full: bool = True,
        failing_per_node: List[str] | None = None,
    ) -> None:
        self.orphan_names = list(orphan_names)
        self.select_raises = select_raises
        self.batch_raises = batch_raises
        self.batch_returns_full = batch_returns_full
        self.failing_per_node = failing_per_node or []
        self.execute_calls: List[str] = []      # zapytania (do inspekcji)
        self.execute_count = 0
        self.unwind_seen: List[str] | None = None

    def execute(self, query: str, params: Any = None) -> _FakeResult:
        self.execute_count += 1
        self.execute_calls.append(query.strip().split()[0] if query.strip() else "")
        if "RETURN e.name" in query:
            if self.select_raises:
                raise RuntimeError("database is closed")
            return _FakeResult([[n] for n in self.orphan_names])
        if "RETURN count(r)" in query:
            # V121 (T3): edge re-check przed per-node DELETE. W tych scenariuszach
            # węzły są sierotami w DB — TOCTOU jest modelowany jako edge wstrzyknięta
            # MIĘDZY re-check a DELETE (per-node DELETE nadal rzuca dla failing_per_node).
            return _FakeResult([[0]])
        if "UNWIND" in query and "DELETE" in query:
            self.unwind_seen = list((params or {}).get("names", []))
            if self.batch_raises:
                raise RuntimeError("edges exist (TOCTOU)")
            # batch zwraca wiersze NM; jeśli full, wszystkie, jeśli partial, tylko część
            if self.batch_returns_full:
                return _FakeResult([[n] for n in self.orphan_names])
            return _FakeResult([[n] for n in self.orphan_names[:1]])
        if "DELETE e" in query and "UNWIND" not in query:
            name = str((params or {}).get("name", ""))
            if name in self.failing_per_node:
                raise RuntimeError("connected edges fwd")
            return _FakeResult([])
        raise AssertionError(f"unexpected query: {query[:80]}")


def _driver(conn: _BatchFakeConn) -> KuzuSchemaDriverMixin:
    drv = KuzuSchemaDriverMixin.__new__(KuzuSchemaDriverMixin)
    drv.conn = conn  # type: ignore[attr-defined]
    return drv


def test_a1_happy_path_exactly_two_executes() -> None:
    """(a) happy path: 1 SELECT + 1 batch UNWIND = 2 execute() total."""
    conn = _BatchFakeConn(orphan_names=["A", "B", "C"])
    drv = _driver(conn)

    deleted = drv._collect_and_delete_orphans_on_disk()

    assert deleted == ["A", "B", "C"], f"batch powinien zwrócić wszystkie, dostałem {deleted!r}"
    assert conn.execute_count == 2, f"happy path wymaga dokładnie 2 execute(); było {conn.execute_count}"
    assert conn.unwind_seen == ["A", "B", "C"], "batch UNWIND nie otrzymał listy orphans"


def test_a2_batch_returns_partial_triggers_fallback() -> None:
    """(b) batch zwrócił 1/3 nazw -> fallback per-node 3 delete (2 zostaną już skasowane, ale per-node DELETE nic nie rzuci)."""
    conn = _BatchFakeConn(orphan_names=["A", "B", "C"], batch_returns_full=False)
    drv = _driver(conn)

    deleted = drv._collect_and_delete_orphans_on_disk()

    # wszystkie 3 orphans faktycznie zniknęły z bazy (batch usunął część, reszta fallback)
    assert sorted(deleted) == ["A", "B", "C"], f"deleted powinien zawierać wszystkie sieroty: {deleted!r}"
    # V121 (T3): 1 SELECT + 1 batch (partial) + 3×(edge re-check + DELETE) = 8
    assert conn.execute_count == 8, f"partial wymaga 8 execute(); było {conn.execute_count}"


def test_a3_batch_raises_falls_back_per_node() -> None:
    """(c) batch rzuca (TOCTOU/martwy silnik) -> fallback per-node, izolacja V111."""
    conn = _BatchFakeConn(
        orphan_names=["A", "B", "C"],
        batch_raises=True,
        failing_per_node=["B"],
    )
    drv = _driver(conn)

    deleted = drv._collect_and_delete_orphans_on_disk()

    # V111 kontrakt: A i C przechodzą, B rzuca ale NIE przerywa pętli
    assert sorted(deleted) == ["A", "C"], f"per-node fallback musi izolować B; dostałem {deleted!r}"
    # V121 (T3): 1 SELECT + 1 batch (rzuca) + 3×(edge re-check + DELETE) = 8
    assert conn.execute_count == 8, f"fallback wymaga 8 execute.; było {conn.execute_count}"


def test_a4_select_failure_aborts_with_empty_list() -> None:
    """(f) awaria SELECT = martwy silnik, outer try zachowany."""
    conn = _BatchFakeConn(orphan_names=["A", "B", "C"], select_raises=True)
    drv = _driver(conn)

    deleted = drv._collect_and_delete_orphans_on_disk()

    assert deleted == []
    # SELECT rzucił, więc batch nawet nie był próbowany
    assert conn.execute_count == 1, f"przy martwym SELECT nie wolno iść dalej; było {conn.execute_count}"


def test_a5_v111_regression_per_node_isolation() -> None:
    """(e) bez V114 batch (rollback fallback) zachowuje się jak V111."""
    # symulujemy fallback: batch rzuca, a per-node A/B/C z BAD w środku
    conn = _BatchFakeConn(
        orphan_names=["A", "BAD", "C"],
        batch_raises=True,
        failing_per_node=["BAD"],
    )
    drv = _driver(conn)

    deleted = drv._collect_and_delete_orphans_on_disk()

    assert sorted(deleted) == ["A", "C"], f"V111 kontrakt złamany: {deleted!r}"


def test_a6_partial_fallback_does_not_double_count(caplog) -> None:
    """(b) szczegóły: po partial batch per-node pętla NIE powinna duplikować w deleted
    nazw już skasowanych przez batch — Kùzu `MATCH` nie rzuca gdy węzeł nie istnieje,
    więc per-node DELETE nic nie robi, ale pętla i tak go dorzuca do deleted.

    W tej wersji (V114 partial semantics) deleted = wszystkie orphans (batch + fallback
    razem wzięte), pod warunkiem że nie propagują błędów — liczba orphanów == liczba
    skasowanych faktycznie w bazie.
    """
    conn = _BatchFakeConn(orphan_names=["A", "B", "C"], batch_returns_full=False)
    drv = _driver(conn)

    with caplog.at_level(logging.WARNING):
        deleted = drv._collect_and_delete_orphans_on_disk()

    assert sorted(deleted) == ["A", "B", "C"]
    # fallback path powinien ostrzec
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("batch" in m.lower() and "fallback" in m.lower() for m in warns), (
        f"oczekiwano warninga o fallbacku; jest: {warns}"
    )


def test_a7_public_signature_unchanged() -> None:
    """Sygnatura metody bez zmian (V111 invariant)."""
    sig = inspect.signature(KuzuSchemaDriverMixin._collect_and_delete_orphans_on_disk)
    assert list(sig.parameters) == ["self"]
    assert sig.return_annotation == "List[str]"
