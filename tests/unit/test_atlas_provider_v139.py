"""V139/MEM-BRIDGE — kontrakt mosta Hermes→Mnemosyne (fail-visible).

Incydent 2026-09-27..10-06: desktop agent działał na interpreterze bez
mnemosyne_hermes; cichy ImportError ustawiał MNEMOSYNE_DELEGATE=None i
sync_turn nie zapisywał nic do Mnemosyne przez 9 dni, bez ani jednego loga.

Bramki (mutacja bez tych testów musi paść):
T1: gałąź ImportError NIE jest cicha — loguje WARNING.
T2: ścieżka błędu sync_turn loguje z exc_info (stack trace).
T3: delegacja _mnemosyne w __init__ używa MNEMOSYNE_DELEGATE (bez hardkodowanej None).
"""
from __future__ import annotations

import logging
import sys

import pytest

MODULE = "atlas_memory.hermes.atlas_provider"


@pytest.fixture()
def fresh_provider_module():
    """Import modułu z czystym stanem (cache bust + monkeypatch dla T1/T2)."""
    saved = sys.modules.get(MODULE)

    def _import(**patches):
        import importlib

        for name, val in patches.items():
            importlib.invalidate_caches()
            setattr(importlib.import_module(MODULE), name, val)
        return importlib.reload(importlib.import_module(MODULE))

    yield _import
    if saved is not None:
        sys.modules[MODULE] = saved


def _capture(caplog):
    caplog.set_level(logging.WARNING, logger=MODULE)


def test_t1_import_error_is_loud(caplog):
    """T1: ImportError delegata MUSI logować WARNING (kontra cichy None)."""
    import importlib

    _capture(caplog)
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *a, **k):
        if name == "mnemosyne_hermes":
            raise ImportError("simulated missing mnemosyne_hermes")
        return real_import(name, *a, **k)

    builtins_mod = importlib.import_module("builtins")
    old = builtins_mod.__import__
    builtins_mod.__import__ = fake_import
    try:
        importlib.reload(importlib.import_module(MODULE))
        mod = importlib.import_module(MODULE)
        assert mod.MNEMOSYNE_DELEGATE is None
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and "mnemosyne" in r.getMessage().lower()]
        assert warnings, "V139: ImportError delegata musi być logowany WARNING (nie cichy)"
    finally:
        builtins_mod.__import__ = old
        importlib.reload(importlib.import_module(MODULE))


def test_t2_sync_turn_failure_has_traceback(caplog):
    """T2: błąd w _mnemosyne.sync_turn → WARNING Z exc_info (stack trace)."""
    from atlas_memory.hermes.atlas_provider import AtlasMemoryProvider

    _capture(caplog)
    p = AtlasMemoryProvider.__new__(AtlasMemoryProvider)  # bez pełnego __init__

    class BrokenDelegate:
        def sync_turn(self, *a, **k):
            raise RuntimeError("boom-mnemosyne")

    p._mnemosyne = BrokenDelegate()
    p._session_id = "v139_test"
    p._shadow_rpc_enabled = lambda: False
    p._orchestrator = None

    class _NoWorker:
        def submit(self, fn):
            pass

        def start(self):
            pass

    p._worker = _NoWorker()
    p.sync_turn("test user content v139", "test assistant content v139")
    rec = [r for r in caplog.records if "Mnemosyne sync_turn failed" in r.getMessage()]
    assert rec, "V139: błąd sync_turn musi być logowany"
    assert rec[0].exc_info and rec[0].exc_info[0] is RuntimeError, "V139: WARNING musi niosło exc_info (stack trace)"


def test_t3_delegate_wired_in_init():
    """T3: __init__ przypisuje self._mnemosyne z MNEMOSYNE_DELEGATE, nie hard-kod None."""
    import inspect

    from atlas_memory.hermes import atlas_provider

    src = inspect.getsource(atlas_provider.AtlasMemoryProvider.__init__)
    assert "_build_mnemosyne_delegate(MNEMOSYNE_DELEGATE)" in src, "V139: delegate musi być budowany factory w __init__"


def test_t5_factory_preseeds_init_home_before_init():
    """T5 (P1b regressja): factory MUSI ustawić _init_home PRZED __init__.

    Bez tego konstruktor 0.7.4 pada wewnątrz turn scope Hermesa (self._beam=None
    trafia w setter property zanim _bindings ma klucz home).
    """
    from atlas_memory.hermes.atlas_provider import _build_mnemosyne_delegate

    class FakeBridge:
        # wiernie oddaje patologiczną kolejność 0.7.4: setter _beam wymaga klucza
        def __init__(self):
            key = self.__dict__.get("_init_home")
            if key is None:
                raise RuntimeError("no binding for the current home (P1b)")
            self.__dict__["beam"] = f"beam-for-{key}"

        def __init_subclass__(cls, **k):  # noqa
            pass

    # symuluj kontekst Hermesa: hermes_constants dostępny
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "hermes_constants":
            import types

            m = types.ModuleType("hermes_constants")
            m.get_hermes_home = lambda: "/home/tester/.hermes"
            return m
        return real_import(name, *a, **k)

    builtins.__import__ = fake_import
    try:
        inst = _build_mnemosyne_delegate(FakeBridge)
    finally:
        builtins.__import__ = real_import
    assert inst.__dict__["_init_home"] == "/home/tester/.hermes"
    assert inst.beam == "beam-for-/home/tester/.hermes"


def test_t6_out_of_hermes_falls_back_to_plain_construction():
    """T6: brak hermes_constants (poza进程 Hermesa) → zwykłe cls()."""
    from atlas_memory.hermes.atlas_provider import _build_mnemosyne_delegate

    constructed = {}

    class Plain:
        def __init__(self):
            constructed["ok"] = True

    import builtins
    real_import = builtins.__import__

    def failing_import(name, *a, **k):
        if name == "hermes_constants":
            raise ImportError("not a hermes process")
        return real_import(name, *a, **k)

    builtins.__import__ = failing_import
    try:
        _build_mnemosyne_delegate(Plain)
    finally:
        builtins.__import__ = real_import
    assert constructed.get("ok") is True


def test_t4_sync_turn_delegates_when_present():
    """T4 (kontrakt dodatni): gdy delegate istnieje, sync_turn woła go z treściami."""
    from atlas_memory.hermes.atlas_provider import AtlasMemoryProvider

    calls = []

    class SpyDelegate:
        def sync_turn(self, user_content, assistant_content, *, session_id=""):
            calls.append((user_content, assistant_content, session_id))

    p = AtlasMemoryProvider.__new__(AtlasMemoryProvider)
    p._mnemosyne = SpyDelegate()
    p._session_id = "sess-x"
    p._shadow_rpc_enabled = lambda: False
    p._orchestrator = None

    class _NoWorker:
        def submit(self, fn):
            pass

    p._worker = _NoWorker()
    p.sync_turn("USER_MSG_V139", "ASSISTANT_MSG_V139")
    assert calls == [("USER_MSG_V139", "ASSISTANT_MSG_V139", "sess-x")]


def test_real_mnemosyne_live_pipeline(tmp_path, monkeypatch):
    """Prawdziwy test integracyjny: brak mocków, realna baza SQLite na dysku.

    Weryfikuje, że:
    1. AtlasMemoryProvider z realnym MnemosyneMemoryProvider konstruuje się bez błędów.
    2. p.initialize() bez żadnych dodatkowych kwargs wiąże BeamMemory do bieżącego HERMES_HOME.
    3. p.sync_turn() fizycznie zapisuje wiersz do tabeli working_memory w SQLite.
    4. Dane w bazie odpowiadają treści zapytania i odpowiedzi.
    """
    import sqlite3

    from atlas_memory.hermes.atlas_provider import AtlasMemoryProvider

    home_dir = tmp_path / "hermes_real_home"
    home_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home_dir))

    p = AtlasMemoryProvider(socket_path=str(home_dir / "atlas.sock"))
    assert p._mnemosyne is not None, "Mnemosyne delegate musi być zainicjalizowany w realnym środowisku"

    p.initialize(session_id="sess_real_live_test")
    beam = getattr(p._mnemosyne, "_beam", None)
    assert beam is not None, "Po initialize() p._mnemosyne._beam NIE MOŻE być None!"

    user_text = "Prawdziwy tekst użytkownika z testu V139"
    asst_text = "Prawdziwa odpowiedź asystenta z testu V139"
    p.sync_turn(user_text, asst_text, session_id="sess_real_live_test")

    db_path = home_dir / "mnemosyne" / "data" / "mnemosyne.db"
    assert db_path.exists(), f"Baza danych Mnemosyne SQLite musi istnieć na dysku pod {db_path}!"

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, content FROM working_memory ORDER BY timestamp DESC LIMIT 1;")
    row = cursor.fetchone()
    conn.close()

    assert row is not None, "Tabela working_memory w SQLite musi zawierać zapisany rekord tury!"
    assert "sess_real_live_test" in row[0], f"Oczekiwano sesji w rekordzie, otrzymano: {row[0]}"
    assert user_text in row[1], f"Oczekiwano tekstu użytkownika w rekordzie, otrzymano: {row[1]}"


def test_real_mnemosyne_auto_heal_on_sync_turn(tmp_path, monkeypatch):
    """Prawdziwy test auto-uzdrawiania: wywołanie sync_turn bez uprzedniego initialize().

    Weryfikuje, że AtlasMemoryProvider wykrywa brak _beam i automatycznie inicjalizuje
    delegata Mnemosyne zamiast po cichu porzucać zapis.
    """
    import sqlite3

    from atlas_memory.hermes.atlas_provider import AtlasMemoryProvider

    home_dir = tmp_path / "hermes_auto_heal"
    home_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home_dir))

    p = AtlasMemoryProvider(socket_path=str(home_dir / "atlas.sock"))
    assert getattr(p._mnemosyne, "_beam", None) is None, "Przed initialize() _beam powinien być None"

    p.sync_turn(
        "Tekst zapytania przed initialize",
        "Odpowiedź asystenta przed initialize",
        session_id="sess_auto_heal",
    )

    beam = getattr(p._mnemosyne, "_beam", None)
    assert beam is not None, "sync_turn musi automatycznie uzdrowić delegata Mnemosyne!"

    db_path = home_dir / "mnemosyne" / "data" / "mnemosyne.db"
    assert db_path.exists(), "Baza SQLite musi zostać utworzona przez auto-heal!"

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT session_id, content FROM working_memory ORDER BY timestamp DESC LIMIT 1;")
    row = cursor.fetchone()
    conn.close()

    assert row is not None, "Auto-heal musi fizycznie zapisać wiersz do bazy SQLite!"
    assert "sess_auto_heal" in row[0]
    assert "Tekst zapytania przed initialize" in row[1]


def test_real_mnemosyne_prefetch_recall(tmp_path, monkeypatch):
    """Prawdziwy test odczytu (prefetch/recall): zapisanie tury i weryfikacja prefetch.

    Weryfikuje, że p.prefetch() odpytuje Mnemosyne i nie rzuca błędów ani nie wisi.
    """
    from atlas_memory.hermes.atlas_provider import AtlasMemoryProvider

    home_dir = tmp_path / "hermes_prefetch"
    home_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home_dir))

    p = AtlasMemoryProvider(socket_path=str(home_dir / "atlas.sock"))
    p.initialize(session_id="sess_prefetch_test")

    p.sync_turn(
        "Kluczowy fakt o architekturze bazy danych PostgreSQL",
        "Potwierdzam, baza to PostgreSQL wersja 16",
        session_id="sess_prefetch_test",
    )

    # Odpytanie o zapisany fakt
    result = p.prefetch("PostgreSQL baza danych architekturze", session_id="sess_prefetch_test")
    assert isinstance(result, str), "prefetch musi zwracać ciąg tekstowy"


def test_storage_bridge_direct_records_zero_regex_overhead(tmp_path, monkeypatch):
    """Test eliminacji 'Adapter-on-Adapter Inception' i zbędnego parsowania regexami.

    Weryfikuje, że AtlasMemoryProvider pobiera surowe słowniki z SQLite przez
    MnemosyneStorageBridge.recall_records() i mapuje je bezpośrednio na MemoryRecord,
    omijając formatowanie tekstu markdown i regex _parse_mnemosyne_context.
    """
    from atlas_memory.hermes.atlas_provider import (
        AtlasMemoryProvider,
        MnemosyneStorageBridge,
        _beam_dict_to_record,
    )
    from atlas_memory.orchestrator import MemoryOrchestrator

    home_dir = tmp_path / "hermes_direct_records"
    home_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home_dir))

    provider = AtlasMemoryProvider(
        orchestrator=MemoryOrchestrator(),
        socket_path=str(home_dir / "atlas.sock"),
    )
    assert isinstance(provider._mnemosyne, MnemosyneStorageBridge), "Most pamięci musi być MnemosyneStorageBridge"

    provider.initialize(session_id="sess_direct_test")
    provider.sync_turn(
        "Krytyczna konfiguracja bazy danych DuckDB oraz Kuzu Graph",
        "Zrozumiałem konfigurację DuckDB i Kuzu",
        session_id="sess_direct_test",
    )

    # Bezpośrednie pobranie słowników
    raw_dicts = provider._mnemosyne.recall_records("DuckDB Kuzu", session_id="sess_direct_test")
    assert len(raw_dicts) > 0, "recall_records musi zwrócić co najmniej jeden słownik z SQLite"
    first = raw_dicts[0]
    assert "content" in first
    assert "DuckDB" in first["content"]

    # Test bezpośredniej konwersji na MemoryRecord
    record = _beam_dict_to_record(first)
    assert record.subject.startswith("mnemosyne:")
    assert "DuckDB" in record.object
    assert record.source_type.value == "user_explicit" or record.source_type.value == "external_doc"

    # Test prefetch z orchestratorem: powinien wydobyć kontekst z rekordów
    ctx = provider.prefetch("DuckDB Kuzu konfiguracja", session_id="sess_direct_test")
    assert provider._last_prefetch is not None
    assert provider._last_prefetch.get("skipped") is False
    assert provider._last_prefetch.get("count", 0) > 0
    assert "DuckDB" in ctx


def test_storage_bridge_native_clean_instance():
    """Weryfikuje, że MnemosyneStorageBridge jest czystym obiektem bez monkey-patchingu."""
    from atlas_memory.hermes.atlas_provider import (
        MnemosyneStorageBridge,
        _build_mnemosyne_delegate,
    )

    inst = _build_mnemosyne_delegate(MnemosyneStorageBridge)
    assert isinstance(inst, MnemosyneStorageBridge)
    # Czysty obiekt nie potrzebuje hakowania _init_home ani _ambient_key
    assert "_init_home" not in inst.__dict__
    assert "_ambient_key" not in inst.__dict__



