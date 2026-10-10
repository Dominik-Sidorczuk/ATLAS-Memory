"""Testy własności magazynu wektorów (V70.1: jeden właściciel na ścieżce dyskowej).

Regresja, którą te testy blokują: starsza wersja `QdrantVectorStore` przy KAŻDYM
wyjątku zawierającym słowo "lock" bezwarunkowo kasowała plik `.lock` i ponawiała
otwarcie. Przy ŻYWYM właścicielu (np. daemon ATLAS) dopuszczało to drugiego pisarza
do jednego katalogu Qdranta — czyli ryzyko korupcji magazynu wektorowego.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from atlas_memory.l2_semantic.qdrant_store import (  # noqa: E402
    OWNER_FILENAME,
    VectorStoreLockedError,
    pid_is_alive,
    read_vector_store_owner,
)

COLLECTION = "test_ownership"
DIMENSION = 8


class _FakeEncoder:
    """Enkoder zastępczy: testy własności nie potrzebują modelu embeddingów."""

    def __init__(self, dimension: int = DIMENSION) -> None:
        self.dimension = dimension

    def encode(self, text: str) -> list[float]:
        base = float(len(text) % 7 + 1)
        return [base / 10.0] * self.dimension


def _store(location: str):
    from atlas_memory.l2_semantic.qdrant_store import QdrantVectorStore

    return QdrantVectorStore(
        collection_name=COLLECTION,
        location=location,
        dimension=DIMENSION,
        encoder=_FakeEncoder(),
        owner="pytest-ownership",
    )


def _dead_pid() -> int:
    """Pid procesu, który na pewno już nie żyje (przez waitpid, nie przez zgadywanie)."""
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def test_pid_is_alive_uses_real_process_state() -> None:
    assert pid_is_alive(os.getpid()) is True
    assert pid_is_alive(_dead_pid()) is False
    assert pid_is_alive(0) is False
    assert pid_is_alive(-1) is False
    bogus: Any = "nie-liczba"  # dane z uszkodzonego .lock bywają nie-liczbą
    assert pid_is_alive(bogus) is False


def test_zaden_test_nie_otwiera_produkcyjnej_sciezki() -> None:
    """Meta-test: `QdrantVectorStore` bez `location` pisze do PRODUKCJI.

    Regresja zmierzona 2026-09-10 18:01:19: `test_l2_semantic.py` konstruowal
    store bez `location`, przez co pelny gate podmienil `owner.json` w
    `~/.hermes/atlas/qdrant` na pid procesu pytest (72918, martwy).
    """
    import ast

    offenders: list[str] = []
    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name != "QdrantVectorStore":
                continue
            if not any(kw.arg == "location" for kw in node.keywords):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"QdrantVectorStore bez `location` = zapis do produkcji: {offenders}"


def test_open_writes_owner_token_with_pid_and_host(tmp_path: Path) -> None:
    store = _store(str(tmp_path))
    try:
        token = read_vector_store_owner(tmp_path)
        assert token is not None, "brak tokenu właściciela po otwarciu magazynu"
        assert token["pid"] == os.getpid()
        assert token["owner"] == "pytest-ownership"
        assert token["host"], "token bez nazwy hosta nie pozwala odróżnić właściciela zdalnego"
    finally:
        store.close()


def test_close_releases_owner_token(tmp_path: Path) -> None:
    store = _store(str(tmp_path))
    assert (tmp_path / OWNER_FILENAME).is_file()
    store.close()
    assert not (tmp_path / OWNER_FILENAME).exists(), "close() nie zwolnił tokenu właściciela"


def test_live_owner_is_refused_and_lock_file_survives(tmp_path: Path) -> None:
    """Drugi magazyn na tej samej ścieżce przy ŻYWYM właścicielu = odmowa, pliki nietknięte."""
    first = _store(str(tmp_path))
    lock_path = tmp_path / ".lock"
    try:
        with pytest.raises(VectorStoreLockedError) as excinfo:
            _store(str(tmp_path))
        assert str(os.getpid()) in str(excinfo.value)
        assert lock_path.exists(), "cudzy ŻYWY .lock został skasowany — to była regresja V70.1"
        token = read_vector_store_owner(tmp_path)
        assert token is not None and token["pid"] == os.getpid()
    finally:
        first.close()


def test_dead_owner_is_provable_and_thus_recoverable(tmp_path: Path) -> None:
    """Warunek odzyskania blokady: właściciel z TEGO hosta i jego pid NIE żyje."""
    dead = _dead_pid()
    token = {
        "owner": "crash-victim",
        "pid": dead,
        "started_at": 1_700_000_000.0,
        "host": os.uname().nodename,
    }
    (tmp_path / OWNER_FILENAME).write_text(json.dumps(token), encoding="utf-8")
    (tmp_path / ".lock").write_text(str(dead), encoding="utf-8")

    read_back = read_vector_store_owner(tmp_path)
    assert read_back is not None and read_back["pid"] == dead
    assert read_back["host"] == os.uname().nodename
    assert pid_is_alive(dead) is False, "odzyskanie blokady bez dowodu śmierci = zakazane"


def test_foreign_host_owner_is_never_recovered(tmp_path: Path) -> None:
    """Właściciel z innego hosta: brak dowodu śmierci → token czytelny, ale nie wolno go kasować."""
    token = {
        "owner": "other-machine",
        "pid": _dead_pid(),
        "started_at": 1_700_000_000.0,
        "host": "jakis-inny-host",
    }
    (tmp_path / OWNER_FILENAME).write_text(json.dumps(token), encoding="utf-8")
    read_back = read_vector_store_owner(tmp_path)
    assert read_back is not None and read_back["host"] == "jakis-inny-host"
    assert read_back["host"] != os.uname().nodename
    assert (tmp_path / OWNER_FILENAME).exists(), "token zdalnego właściciela nie może zniknąć"


def test_memory_location_does_not_create_owner_files(tmp_path: Path) -> None:
    """Tryb `:memory:` nie dotyka dysku, więc nie tworzy tokenu własności."""
    store = _store(":memory:")
    try:
        assert read_vector_store_owner(tmp_path) is None
        assert not (tmp_path / OWNER_FILENAME).exists()
    finally:
        store.close()
