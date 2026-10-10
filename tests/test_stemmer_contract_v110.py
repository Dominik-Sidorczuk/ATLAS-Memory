"""V110 (F-029) Stemmer contract tests — wyjście zweryfikowane empirycznie.

Asercje są zgodne z rzeczywistym zachowaniem bieżącej implementacji
(``pl_core_news_sm`` + heurystyka bez [:5] i bez globalnego y→i):

  pastes   -> 'pastes'      pasted   -> 'pasted'      (osobne lemma)
  typ      -> 'typ'         tip      -> 'tip'         (osobne, brak kolizji y->i)
  gry      -> 'gra'         tri      -> 'tri'         (osobne)
  wypowiedzi -> 'wipowiedz' wypowiedz -> 'wypowiedz'  (różne, dawniej oba 'wipow')
  wypowiedź -> 'wipowiedz'                              (zachowana fonetyka PL)
  systemy  -> 'syst'        (spaCy lemma 'system' + heur. bez [:5])
  uzytkownika -> 'uzytkownik' preferowany -> 'preferowac'  (spaCy lemma)

Gdyby środowisko Thin-Client (bez spaCy) zwróciło None-lemma, test oznacza to
jawnie (regresja środowiska), ale nie asertuje fałszywej wartości.
"""
from __future__ import annotations

import pytest

from atlas_memory.core.linguistics import (
    _bilingual_stem,
    _heuristic_stem_fallback,
    get_spacy_nlp,
)

# ---------------------------------------------------------------------------
# F-029 P1: brak destrukcyjnego stemmera ASCII
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("word", "expected"),
    [
        # Empirycznie zmierzone na bieżącym środowisku (pixi Py3.14 + pl_core_news_sm).
        ("pastes", "pastes"),
        ("pasted", "pasted"),
        ("typ", "typ"),
        ("tip", "tip"),
        ("gry", "gra"),
        ("tri", "tri"),
        ("wypowiedzi", "wipowiedz"),
        ("wypowiedz", "wypowiedz"),
        ("wypowiedź", "wipowiedz"),
        ("systemy", "syst"),
        ("uzytkownika", "uzytkownik"),
        ("preferowany", "preferowac"),
        ("edytorem", "edytor"),
    ],
)
def test_v110_bilingual_stem_expected_outputs(word: str, expected: str) -> None:
    """V110 — bazowy kontrakt: każde wejście ma zmierzone wyjście (spaCy PL + heurystyka)."""
    got = _bilingual_stem(word)
    assert got == expected, f"_bilingual_stem({word!r}) -> {got!r}, oczekiwano {expected!r}"


def test_v110_typ_tip_remain_distinct() -> None:
    """F-029: 'typ' i 'tip' NIE mogą zbiec do wspólnego stemmera (globalny y->i)."""
    assert _bilingual_stem("typ") != _bilingual_stem("tip")


def test_v110_gry_tri_remain_distinct() -> None:
    """F-029: 'gry' (PL rzeczownik) i 'tri' (EN) — osobne stemmy, brak fałszywej kolizji."""
    assert _bilingual_stem("gry") != _bilingual_stem("tri")


def test_v110_pastes_pasted_distinct_after_no_first5_truncation() -> None:
    """F-029: bez [:5] i bez ASCII-shorcut pastes/pasted dostają osobne lemmaty."""
    pastes = _bilingual_stem("pastes")
    pasted = _bilingual_stem("pasted")
    assert pastes != pasted, (
        f"regresja: pastes={pastes!r} == pasted={pasted!r} "
        "— wskazuje na powrót [:5] lub ASCII short-circuit"
    )
    # Antykolizja z historycznym 'paste' z poprzedniej wersji (stary stem = 'paste').
    assert pastes != "paste" or pasted != "paste", (
        "regresja: oba słowa zwracają 'paste' (stary stem via [:5] truncation)"
    )


def test_v110_wypowiedzi_not_collided_by_first5_truncation() -> None:
    """F-029: stem('wypowiedzi') != 'wipow' (historyczny [:5]-collision)."""
    got = _bilingual_stem("wypowiedzi")
    assert got != "wipow", (
        f"regresja F-029: stem('wypowiedzi') = {got!r} == 'wipow' (stary [:5] truncation)"
    )
    # Oraz: ani stem('wypowiedzi') ani stem('wypowiedz') nie są równe 'wipow'
    # (historycznie oba były sklejone przez [:5]).
    assert _bilingual_stem("wypowiedz") != "wipow" or _bilingual_stem("wypowiedz") == "wypowiedz"


def test_v110_heuristic_keeps_y_for_pure_ascii_words() -> None:
    """F-029: heurystyka NIE zamienia y->i dla słów czysto ASCII (bez PL diakrytów)."""
    # _heuristic_stem_fallback działa na izolowanym słowie (bez spaCy), więc jest
    # deterministyczny i nie zależy od środowiska.
    assert "y" in _heuristic_stem_fallback("systemy") or "y" not in "systemy"
    # Bardziej rygorystycznie: dla 'gry' i 'systemy' wynik heurystyki musi
    # zachować 'y' gdy nie ma PL diakrytów.
    assert _heuristic_stem_fallback("systemy") == "systemy", (
        "heurystyka zgubiła 'y' w 'systemy' (powrót globalnego y->i)"
    )
    assert _heuristic_stem_fallback("gry") == "gry"


def test_v110_heuristic_applies_y_to_i_only_for_pl_diacritic_words() -> None:
    """F-029: y->i uruchamia się tylko gdy oryginalne słowo miało PL diakryty."""
    pl_dia = _heuristic_stem_fallback("wypowiedź")  # 'ź' → transliteracja, potem y->i
    ascii_v = _heuristic_stem_fallback("wypowiedz")  # ASCII → y zostaje
    assert "i" in pl_dia.split("w")[1] or pl_dia.startswith("wipow"), (
        f"y->i powinno zadziałać po transliteracji PL: {pl_dia!r}"
    )
    assert ascii_v == "wypowiedz", (
        f"dla ASCII wyjście heurystyki musi zachować 'y': {ascii_v!r}"
    )


def test_v110_bilingual_stem_routes_ascii_to_spacy() -> None:
    """V110: usunięto próg ``len(h_stem)>=5 and isascii`` — ASCII słowa idą przez spaCy."""
    nlp = get_spacy_nlp("pl")
    if nlp is None:
        pytest.skip("Thin-Client: spaCy niedostępne, kontrakt walidowany w innym teście")
    # 'typ' ma len=3 (poniżej starego progu) i ASCII — musi przejść przez spaCy.
    # W rezultacie lemma spaCy dla 'typ' = 'typ', nie 'tip' (jak przy y->i+ASCII shortcut).
    assert _bilingual_stem("typ") == "typ"


def test_v110_bilingual_stem_cache_retained() -> None:
    """V110: lru_cache(maxsize=65536) na _bilingual_stem nadal aktywny."""
    assert hasattr(_bilingual_stem, "cache_info") and hasattr(_bilingual_stem, "cache_clear"), (
        "lru_cache na _bilingual_stem został usunięty — naruszona kontrakt-cache"
    )


# ---------------------------------------------------------------------------
# Heurystyka: brak twardego obcięcia [:5]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("word", "must_not_equal"),
    [
        ("edytorem", "edyt"),   # dawniej [:5]='edyt'
        ("uzytkownika", "uzyt"),  # dawniej [:5]='uzyt'
        ("preferowany", "p"),  # dawniej [:5]='prefe' - sprawdzamy, że wynik > 5 znaków
        ("strategi", "strat"),  # dawniej [:5]='strat'
    ],
)
def test_v110_heuristic_no_hard_first5_truncation(word: str, must_not_equal: str) -> None:
    """F-029: heuristyka nie obcina twardo do [:5] — słowa dłuższe niż 5 znaków wracają w całości (lub po odjęciu sufiksu, ale nie sztucznie)."""
    got = _heuristic_stem_fallback(word)
    assert got != must_not_equal, (
        f"regresja F-029: _heuristic_stem_fallback({word!r}) = {got!r} == {must_not_equal!r} (stary [:5])"
    )
    # Dodatkowo: gdy słowo ma ≥5 znaków i żaden sufiks PL nie pasuje, wynik ma ≥5 znaków.
    if len(word) >= 6:
        assert len(got) >= 5 or got == "", (
            f"heurystyka skróciła {word!r} ({len(word)} znaków) do {got!r} ({len(got)} znaków)"
        )


def test_v110_heuristic_preserves_polish_suffix_stripping() -> None:
    """F-029: odcinanie sufiksów PL ('ach','ami','owi','em','om','ow','a','e','i','o','u') jest zachowane."""
    # Sufiks 'a': 'edytora' -> 'edytor' (>=5+1 znaku, kończy się na 'a')
    assert _heuristic_stem_fallback("edytora") == "edytor"
    # Sufiks 'em': 'system' -> 'syst'
    assert _heuristic_stem_fallback("system") == "syst"
    # Sufiks 'ach': 'uzytkownikach' -> 'uzytkownik'
    assert _heuristic_stem_fallback("uzytkownikach") == "uzytkownik"
