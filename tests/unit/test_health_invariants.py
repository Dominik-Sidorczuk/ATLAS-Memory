"""
V72.2 (W5) — brama zdrowia mierzy to, co naprawdę trzeba: H3 w BAJTACH i H6 jako
pokrycie KV wektorami (oba kierunki). H1-H4 zostają nietknięte.

Dlaczego H3 zmieniło miarę: stara wersja patrzyła na PREFIKS KLUCZA
(``mnemosyne:*``/``turn:*``) i w chwili incydentu SEC-001 raportowała 0%, bo
wielokilobajtowy blob surowej rozmowy (z żywym tokenem API) leżał pod ``fact:*``.
Nowa miara rozpoznaje surowy czat WŁAŚCIWOŚCIOWO: długość >= 1024 znaki ORAZ
markery tur (``[USER]``, ``[conversation]``, ``Human:``...) albo >= 2 separatory.

IZOLACJA: testy są czysto funkcyjne — nie czytają ~/.hermes/atlas, nie pytają
żywego daemona. ``main()`` (I/O produkcyjne) nie jest tu dotykane.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "atlas_health_invariants.py"
_spec = importlib.util.spec_from_file_location("atlas_health_invariants_w5", _SCRIPT)
assert _spec and _spec.loader
health = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = health
_spec.loader.exec_module(health)

# --- Surowa rozmowa: ~5 kB, markery tur, pod kluczem NIE-czatowym (sedno SEC-001). ---
RAW_CHAT_KEY = "fact:session:2026-08-20"
RAW_CHAT_TEXT = (
    "[USER] oto moj klucz API sk-proj-live-ABC123 i prosba o analize\n"
    + ("[conversation] przetwarzam dane uzytkownika i loguje token. " * 90).strip()
    + "\n[Hermes] zapisuje odpowiedz do pliku"
)
SHORT_FACTS = [
    ("fact:user:prefs", "Uzytkownik preferuje jezyk polski i krotkie odpowiedzi."),
    ("fact:project:name", "Projekt nazywa sie ATLAS i trzyma pamiec dlugoterminowa."),
    ("procedura:sop", "SOP: najpierw test, potem commmit."),
]


def _decode(text: str) -> str:
    """Symuluje podwojnie kodowany JSON w `state_variables.value` (V70.2)."""
    return json.dumps(json.dumps(text, ensure_ascii=False), ensure_ascii=False)


def test_h3_liczy_bajty_i_lapie_surowy_czat_pod_kluczem_fact() -> None:
    rows = [(RAW_CHAT_KEY, _decode(RAW_CHAT_TEXT))] + [(k, _decode(v)) for k, v in SHORT_FACTS]

    res = health.h3_raw_chat_bytes(rows)

    assert res["n_kv"] == len(SHORT_FACTS) + 1
    assert res["raw_records"] == 1, "surowa rozmowa musi byc rozpoznana po TRESCI, nie klucza"
    assert res["raw_bytes"] == len(RAW_CHAT_TEXT.encode("utf-8")), "H3 musi liczyć BAJTY"
    assert res["total_bytes"] > res["raw_bytes"]
    assert res["share"] > health.HARD["H3_raw_chat_bytes"], res
    assert res["ok"] is False, f"5 kB surowej rozmowy pod fact:* MUSI krzyknac: {res}"


def test_h3_milczy_na_samych_krotkich_faktach() -> None:
    rows = [(k, _decode(v)) for k, v in SHORT_FACTS]

    res = health.h3_raw_chat_bytes(rows)

    assert res["raw_records"] == 0
    assert res["raw_bytes"] == 0
    assert res["ok"] is True, res


def test_h3_nie_zalezy_od_prefiksu_klucza() -> None:
    """Ten sam blob pod kluczem `mnemosyne:*` i `fact:*` daje IDENTYCZNY pomiar.

    To wlasnie ta wlasciwosc łamie stara miare prefiksowa: klasyfikacja jest funkcja
    tresci, wiec atakujacy nie ucieknie zmieniajac nazwe klucza.
    """
    rows_prefix = [("mnemosyne:turn:1", _decode(RAW_CHAT_TEXT))]
    rows_plain = [("fact:zwykly", _decode(RAW_CHAT_TEXT))]

    a = health.h3_raw_chat_bytes(rows_prefix)
    b = health.h3_raw_chat_bytes(rows_plain)

    assert a["raw_bytes"] == b["raw_bytes"] > 0
    assert a["share"] == b["share"] == pytest.approx(1.0)
    assert a["ok"] is False and b["ok"] is False


def test_is_raw_chat_text_wymaga_dlugosci_i_sladu_tur() -> None:
    long_but_plain = "zwykly fakt " * 200  # dlugie, ale bez markerow i separatorow
    assert health.is_raw_chat_text(long_but_plain) is False
    assert health.is_raw_chat_text("[USER] krotka tura") is False, "krotkie nie jest czatem"
    assert health.is_raw_chat_text(RAW_CHAT_TEXT) is True
    assert health.is_raw_chat_text("a | b | c | " + ("x" * 1100)) is True, "separatory tur"


def test_h5_lapie_mnozenie_punktow() -> None:
    assert health.h5_points_ratio(100, 400)["ok"] is False, "4x na rekord KV = naruszenie"
    assert health.h5_points_ratio(100, 300)["ok"] is True, "granica progu (3.0) jest domknieta"
    assert health.h5_points_ratio(100, None)["ok"] is True, "brak danych to nie dowod"


def test_h6_krzyczy_przy_dwoch_punktach_na_jeden_rekord() -> None:
    """Górna granica H6: 2 punkty na 1 rekord KV = warstwa wektorowa mnozy punkty."""
    res = health.h6_vector_coverage(1, 2)
    assert res["coverage"] == pytest.approx(2.0)
    assert res["ok"] is False, f"2 punkty / 1 rekord MUSI krzyknac: {res}"


def test_h6_lapie_oba_kierunki_i_granice() -> None:
    assert health.h6_vector_coverage(100, 59)["ok"] is False, "ciche niezaindeksowanie"
    assert health.h6_vector_coverage(100, 60)["ok"] is True, "dolna granica 0.60 domknieta"
    assert health.h6_vector_coverage(100, 100)["ok"] is True
    assert health.h6_vector_coverage(100, 106)["ok"] is False, "mnozenie punktow"
    assert health.h6_vector_coverage(100, None)["ok"] is True, "brak danych = brak wniosku"


def test_h1_h4_zostaly_na_miejscu() -> None:
    """W5 nie moze zlagodzic wczesniejszych bram: progi H1-H4 bez zmian."""
    assert health.HARD["H1_duplication"] == 1.05
    assert health.HARD["H2_ram_disk_parity"] == 0.05
    assert health.HARD["H3_raw_chat_bytes"] == 0.20
    assert health.HARD["H5_points_ratio"] == 3.0
    assert health.HARD["H6_vector_coverage_min"] == 0.60
    assert health.HARD["H6_vector_coverage_max"] == 1.05
