"""V151.2 (2026-10-09): kontrakty użytkownika.

1. SyntaxWarning ``\\m`` z textnorm: ast.literal_eval USUNIĘTY z decode-path;
   nierozpoznane escapes (np. ``\\m``) pozostają dosłowne — NIE przechodzą przez
   kompilator Pythona (zero SyntaxWarning, zero ``<unknown>:1``).
2. Budżet PPR: podłoga 200 ms (``ppr_budget_floor_ms``), formuła adaptacyjna
   skaluje WZGÓRZ z rozmiarem grafu (FILAR III, ADR-V152).
"""
from __future__ import annotations

import warnings

from atlas_memory.cognitive.retrieval.ppr import adaptive_ppr_budget
from atlas_memory.core.config import AtlasConfig, reset_config
from atlas_memory.l2_semantic.textnorm import _regex_unescape, normalize_stored_text


def test_v151_2_f033_literal_m_input_never_compiles() -> None:
    r"""Tekst zawierający control-escape-resemblance `\m` NIE wywoła SyntaxWarning.

    Kontrakt (FINDING-034): unescape jest deterministyczny, bez parsera Pythona.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        text = r'SyntaxWarning: "\m" is an invalid escape sequence.'
        out = normalize_stored_text(text)
        assert out == text  # brak znanego escape w tekście → bez zmian, bit w bit
        out2 = normalize_stored_text(r"\m" + "\n" + r"\end\tline")
        assert "\n" in out2 and "\t" in out2 and r"\m" in out2 and r"\end" in out2


def test_v151_2_f033_garbage_round_trip_is_inert() -> None:
    """Cykl: wstrzyknięcie śmieciowego escape → decode → NIE generuje warningu.

    To jest niezmiennik 'unescape niczego nie przepuszcza do parsera' —
    warstwa unescape nie jest kompilatorem.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", Warning)  # zero warningów JWOGÓLE
        text = r"jak_\m_obecnie:state_value \u0142\u0105ka \x41 \n"
        out = normalize_stored_text(text)
        # unicode + hex + \n zostają zdekodowane; `\m` pozostaje dosłowne
        assert "\u0142\u0105ka" in out and "A" in out and "\n" in out
        assert r"\m" in out and not r"\mm".replace(r"\m", "") == ""


def test_v151_2_f033_regex_unescape_idempotent() -> None:
    """Idempotencja: unescape(unescape(x)) == unescape(x)."""
    inputs = [
        r"\u0142\u0105ka",
        r"\x41\n\t",
        r"\m\k\zzz",
        r"\U0001F600",
        "",
        r"plain\ step",
    ]
    for t in inputs:
        once = _regex_unescape(t)
        twice = _regex_unescape(once)
        assert once == twice, (t, once, twice)


def test_v151_2_f034_ppr_budget_floor_200() -> None:
    """Budżet PPR: podłoga 200 ms (config), zawsze rampuje; skalowanie WZGÓRZ."""
    reset_config(AtlasConfig(ppr_tau_base=40.0, ppr_gamma=8.0, ppr_budget_floor_ms=200.0))
    try:
        b_small = adaptive_ppr_budget(100, 0)
        assert b_small == 200.0, f"floor dominates: {b_small}"
        b_current = adaptive_ppr_budget(1924, 2706)
        assert b_current == 200.0, f"current graph: floor 200 (adaptive ~108 < 200): {b_current}"
        b_huge = adaptive_ppr_budget(19_000, 27_000)
        # adaptive = 40 + 8*ln(46000) ≈ 40+8*10.7 ≈ 126 — pod floor; sprawdź skalowanie ponad:
        assert b_huge >= 200.0
        # gamma nowe: graf na tyle duży (ln(V+E)>20), że adaptive > floor
        b_vast = adaptive_ppr_budget(10**9, 10**9)
        assert b_vast > 200.0, f"adaptive skaluje ponad podłogę: {b_vast}"
    finally:
        reset_config(None)


def test_v151_2_f034_floor_is_config_driven() -> None:
    """Podłoga jest config-driven (FILAR III), NIE stałą w kodzie."""
    reset_config(AtlasConfig(ppr_budget_floor_ms=150.0))
    try:
        assert adaptive_ppr_budget(100, 0) == 150.0
    finally:
        reset_config(None)
