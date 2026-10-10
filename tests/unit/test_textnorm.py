"""Testy kontraktowe normalizacji tekstu pamięci (V70.2)."""

from __future__ import annotations

import pytest

from atlas_memory.l2_semantic.textnorm import normalize_stored_text


def test_literal_unicode_escape_becomes_national_char() -> None:
    raw = "Zażółć gęślą jaźń \\u0119 \\u0105 \\u015b"
    assert normalize_stored_text(raw) == "Zażółć gęślą jaźń ę ą ś"


def test_literal_newline_becomes_real_newline() -> None:
    assert normalize_stored_text("pierwsza\\ndruga") == "pierwsza\ndruga"


def test_literal_tab_becomes_real_tab() -> None:
    assert normalize_stored_text("kol1\\tkol2") == "kol1\tkol2"


def test_idempotent_on_escaped_text() -> None:
    raw = '{"text": "\\u017c\\u00f3\\u0142w\\n\\u0119"}'
    once = normalize_stored_text(raw)
    assert normalize_stored_text(once) == once


def test_idempotent_on_double_encoded_text() -> None:
    raw = 'a\\\\u0119b'
    once = normalize_stored_text(raw)
    assert once == normalize_stored_text(once)


def test_clean_text_returned_bit_for_bit() -> None:
    clean = "Łódź, ul. Świętokrzyska 12\nMieszkanie z ogrodem\tdostępne od zaraz 😀"
    assert normalize_stored_text(clean) == clean
    assert normalize_stored_text(clean) is not None


def test_clean_text_with_windows_path_untouched() -> None:
    # "\\U" bez 8 cyfr hex jest niepoprawnym escape - nie ruszamy danych.
    path = "Kopia w C:\\Users\\user\\dane"
    assert normalize_stored_text(path) == path


def test_empty_string_unchanged() -> None:
    assert normalize_stored_text("") == ""


def test_lone_surrogate_is_utf8_encodable() -> None:
    raw = "tekst z emoji \ud83d bez pary"
    result = normalize_stored_text(raw)
    assert isinstance(result, str)
    assert result.encode("utf-8")  # nie rzuca TypeError/UnicodeEncodeError
    assert "\ud83d" not in result
    assert "tekst z emoji" in result


def test_surrogate_pair_is_recombined() -> None:
    raw = "para \ud83d\ude00 złożona"
    result = normalize_stored_text(raw)
    assert result == "para 😀 złożona"
    assert result.encode("utf-8")


def test_invalid_unicode_escape_returns_original() -> None:
    raw = "niepoprawne \\uZZZZ i \\u12 sekwencje"
    assert normalize_stored_text(raw) == raw


def test_empty_and_plain_texts_do_not_raise() -> None:
    for sample in ("", "a", "  ", "bez escape", "\\", "\\\\", '"', 'a"b'):
        result = normalize_stored_text(sample)
        assert isinstance(result, str)
        result.encode("utf-8")
        assert normalize_stored_text(result) == result


def test_quoted_json_body_is_unwrapped_once() -> None:
    raw = '"\\u0142\\u0105ka"'
    assert normalize_stored_text(raw) == "łąka"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
