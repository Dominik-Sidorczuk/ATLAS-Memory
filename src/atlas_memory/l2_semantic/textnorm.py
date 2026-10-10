"""Deterministyczna normalizacja tekstu pobranego z bazy pamięci (L2).

Problem produkcyjny: część rekordów została zapisana PODWÓJNIE zakodowanym JSON-em,
więc w bazie zamiast znaków narodowych leżą LITERALNE sekwencje ``\\uXXXX``
(np. ``\\u0119`` zamiast ``ę``), a zamiast nowej linii / tabulatora literalne
``\\n`` / ``\\t``. Takie teksty psują tokenizer HuggingFace (tokenizacja ``\\u0119``
zamiast ``ę`` = inny embedding) oraz — w skrajnym przypadku — wywalają go wyjątkiem
``TypeError: TextEncodeInput must be Union[...]`` gdy w łańcuchu zostanie
osamotniony *surrogate* (``\\ud83d`` bez pary nie da się zakodować do UTF-8).

Moduł jest czysty i deterministyczny: żadnych zależności poza stdlib, żadnego
stanu globalnego, żadnych efektów ubocznych.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

# Sekwencje, których obecność oznacza, że tekst wymaga odwiniecia JSON-a.
# Uwaga: sprawdzamy tylko backslash-escape ("\"), a nie znaki kontrolne —
# tekst zapisany poprawnie (z prawdziwym ę / prawdziwą nową linią) musi wrócić
# bez zmian, bit w bit.
_LITERAL_ESCAPE_RE = re.compile(r"\\(?:u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|x[0-9a-fA-F]{2}|n|t|r|b|f|/|\\|')")

# Deterministic unescape table (JSON-style only). Replaces ast.literal_eval
# which used the Python parser as a data decoder and emitted SyntaxWarning
# (`<unknown>:1`) for arbitrary input like `\m` (self-feeding contamination
# when the daemon's own log line was round-tripped through trajectory/ingest).
# v151.2: literal_eval is FORBIDDEN on user data — only the regex below.
_UNESCAPE_TABLE: dict[str, str] = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "b": "\b",
    "f": "\f",
    "/": "/",
    "\\": "\\",
    "'": "'",
    '"': '"',
}

_UNICODE_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")
_LONG_UNICODE_ESCAPE_RE = re.compile(r"\\U([0-9a-fA-F]{8})")
_HEX_ESCAPE_RE = re.compile(r"\\x([0-9a-fA-F]{2})")
_SIMPLE_ESCAPE_RE = re.compile(r"\\([ntrbf/\\'\"])")
_KNOWN_ESCAPE_RE = re.compile(r"\\(?:u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|x[0-9a-fA-F]{2}|[ntrbf/\\'\"])")
_UNKNOWN_ESCAPE_RE = re.compile(r"\\([^uUx0-9a-zA-Z\\'\"/])")

# Limit odwijania: zabezpiecza przed patologicznym wejściem (np. tekstem
# złożonym wyłącznie z backslashy), a jednocześnie domyka PODWÓJNE kodowanie.
_MAX_UNESCAPE_PASSES = 8


def _has_literal_escape(text: str) -> bool:
    return _LITERAL_ESCAPE_RE.search(text) is not None


def _surrogate_free(text: str) -> bool:
    return not any(0xD800 <= ord(ch) <= 0xDFFF for ch in text)


def _repair_surrogates(text: str) -> str:
    """Składa pary surrogate i usuwa osamotnione.

    W Pythonie 3 poprawny znak spoza BMP (``😀``) to JEDEN kod — jeśli w
    łańcuchu leżą dwie jednostki ``\\ud83d\\ude00``, pochodzą z uszkodzonego
    JSON-a. Składamy je w poprawny kod i zapisujemy z powrotem, a jednostki
    bez pary (high bez następnego low albo osamotniony low) usuwamy — inaczej
    ``tokenizer.encode()`` pada z ``TypeError`` przy kodowaniu do UTF-8.
    """
    if _surrogate_free(text):
        return text
    repaired: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        code = ord(text[index])
        if 0xD800 <= code <= 0xDBFF and index + 1 < length:
            next_code = ord(text[index + 1])
            if 0xDC00 <= next_code <= 0xDFFF:
                pair = text[index : index + 2]
                try:
                    combined = pair.encode("utf-16", "surrogatepass").decode("utf-16")
                except UnicodeError:  # pragma: no cover - defensywnie
                    index += 1
                    continue
                if len(combined) == 1:
                    repaired.append(combined)
                    index += 2
                    continue
        if 0xD800 <= code <= 0xDFFF:
            index += 1  # osamotniony surrogate - wycinamy
            continue
        repaired.append(text[index])
        index += 1
    return "".join(repaired)


def _unescape_one_pass(text: str) -> str:
    """Jedno odwiniecie JSON-a; przy niepoprawnym wejściu zwraca tekst wejściowy.

    Obsługuje OBA kształty spotkane w produkcyjnym ``state_variables.value``:
      * pełny literał JSON-owy w cudzysłowach (``"\\u0142\\u0105ka"``) — dekodowany
        ``json.loads`` (jedyne wierne narzędzie: rozumie ``\\"`` i ``\\uXXXX``),
      * samo ciało bez cudzysłowów (``\\u0119``) — dekodowane deterministycznym
        regexem (v151.2: NIE ast.literal_eval; kompilator Pythona nie jest
        dekoderem danych użytkownika; w v151.1 literal_eval emitowało
        ``SyntaxWarning: \\m is invalid`` z filename ``<unknown>`` przy
        obecności sekwencji nierozpoznanej, w tym gdy log daemona
        round-tripowany był przez trajectory/ingest).

    Nierozpoznane backslash-sequence (np. ``\\m``, ``\\k``) traktowane jako
    DOSŁOWNE — znak poprzedzony backslashem pozostaje w tekście. To jest
    kompatybilne z Pythonem 3.12 ``DeprecationWarning`` (który w 3.14 stanie
    się ``SyntaxError``) i nie powoduje generowania ostrzeżeń z
    kompilatora — kompilator NIE jest tu używany.
    """
    stripped = text.strip()
    if len(stripped) >= 2 and stripped.startswith('"') and stripped.endswith('"'):
        try:
            result = json.loads(stripped)
        except (ValueError, TypeError) as exc:
            logger.debug("textnorm: json.loads failed (%s), falling back to regex unescape", exc)
        else:
            if isinstance(result, str):
                return result
    return _regex_unescape(text)


def _regex_unescape(text: str) -> str:
    """Deterministyczny JSON-unescape regexem. Idempotentny w swojej dziedzinie.

    Obsługiwane sekwencje: ``\\n \\t \\r \\b \\f \\/ \\\\ \\' \\" \\uXXXX \\UXXXXXXXX \\xNN``.
    Sekwencje spoza tej listy (np. ``\\m``) pozostawiane DOSŁOWNIE — bez
    generowania SyntaxWarning, bez korzystania z kompilatora Pythona.
    """
    def _u(m: re.Match[str]) -> str:
        return chr(int(m.group(1), 16))
    def _U(m: re.Match[str]) -> str:
        return chr(int(m.group(1), 16))
    def _x(m: re.Match[str]) -> str:
        return chr(int(m.group(1), 16))
    def _simple(m: re.Match[str]) -> str:
        return _UNESCAPE_TABLE[m.group(1)]
    def _unknown(m: re.Match[str]) -> str:
        return m.group(0)  # literal — zachowaj ``\X`` jak leci

    out = _UNICODE_ESCAPE_RE.sub(_u, text)
    out = _LONG_UNICODE_ESCAPE_RE.sub(_U, out)
    out = _HEX_ESCAPE_RE.sub(_x, out)
    out = _SIMPLE_ESCAPE_RE.sub(_simple, out)
    out = _UNKNOWN_ESCAPE_RE.sub(_unknown, out)
    return out


def normalize_stored_text(text: str) -> str:
    """Odwija literalne sekwencje JSON-a i naprawia surrogaty.

    Kontrakt:
      * idempotencja: ``normalize(normalize(x)) == normalize(x)``,
      * tekst bez literalnych sekwencji wraca bez zmian (bit w bit),
      * pusty string wraca jako pusty string,
      * przy niepoprawnym wejściu zwracany jest tekst oryginalny (bez wyjątku).
    """
    if not isinstance(text, str):
        logger.debug("textnorm: non-str input (%s), returning as-is", type(text).__name__)
        return text
    if not text:
        return text

    try:
        normalized = text
        if _has_literal_escape(normalized):
            for _ in range(_MAX_UNESCAPE_PASSES):
                candidate = _unescape_one_pass(normalized)
                if candidate == normalized:
                    break
                normalized = candidate
        normalized = _repair_surrogates(normalized)
        return normalized
    except Exception as exc:  # pragma: no cover - ostatnia linia obrony
        logger.debug("textnorm: normalization failed (%s), returning original text", exc)
        return text


__all__ = ["normalize_stored_text"]
