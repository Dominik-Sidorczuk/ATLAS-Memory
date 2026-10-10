"""ATLAS Language Asset Manager & Unified spaCy Morphological Lemmatizer (V108 Architecture).

Unifies stopword sets, morphological lemmatization (`_bilingual_stem` / `spacy_lemmatize_word`),
and dynamic POS-based structural token filtering across Qdrant Sparse IDF, BM25 scoring,
and HippoRAG 2 OpenIE/PPR while remaining 100% safe to import in Python 3.11 Thin-Client
environments where `spacy` is absent.
"""
from __future__ import annotations

import json
import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources" / "lang"

_NAMESPACE_SCHEMA_TOKENS = frozenset({
    "hermes", "config", "fact", "rule", "preference", "constraint", "system", "service",
    "user_md", "memory_md", "stated_memory", "working_fact", "episodic_event",
    "canonical_spec", "user", "agent", "default",
})

# Dwa ROZŁĄCZNE semantycznie zbiory (nie scalać — różne kanały retrievalu):
#   * PPR_*   — filtr seedów grafu HippoRAG 2 (ppr._tokenize_and_stem)
#   * QUERY_* — filtr rdzeni zapytania BM25/DBSF (strategies.pipeline.query_stems, candidates)
PPR_META_STOPWORDS: frozenset[str] = frozenset({
    "jest", "są", "był", "była", "było", "będzie", "jaki", "jaka", "jakie", "jakim",
    "gdzie", "kiedy", "czym", "dlaczego", "skąd", "dokąd", "czy", "jak", "co", "kto",
    "pamiętasz", "pamietasz", "wiesz", "powiedz", "opisz", "wyjaśnij", "podaj", "wskaż",
    "moim", "moja", "moje", "mój", "twoim", "twoja", "nasz", "nasze", "tym", "tego",
    "tej", "tych", "ten", "ta", "te", "się", "sie", "nie", "tak", "albo", "lub", "oraz",
    "przecież", "przeciez", "ale", "więc", "wiec", "żeby", "zeby", "sprawdzony", "sprawdzona",
})
PPR_META_STOPSTEMS: frozenset[str] = frozenset({
    "bic", "byl", "jak", "gdzi", "kied", "dlacz", "pamie", "wiedz", "powie",
    "opis", "wyjas", "podac", "wskaz", "moj", "twoj", "nasz", "przec",
})
QUERY_META_STOPWORDS: frozenset[str] = frozenset({
    "jest", "są", "był", "była", "było", "byli", "będzie", "będą", "być",
    "gdzie", "kiedy", "dlaczego", "skąd", "dokąd", "jaki", "jaka", "jakie",
    "jakiego", "jakiej", "jakim", "jakich", "który", "która", "które", "którego",
    "czy", "oraz", "albo", "więc", "żeby", "gdyby", "gdybyś", "jeśli", "jeżeli",
    "mój", "moja", "moje", "moich", "moim", "twój", "twoja", "twoje", "twoich",
    "nasz", "nasza", "nasze", "naszych", "swój", "swoje", "swoich",
    "pamiętasz", "pamiętam", "wiadomo", "wiesz", "powiedz", "opisz", "wyjaśnij",
    "podaj", "pokaż", "znajdź", "sprawdź", "odnośnie", "dotyczące", "dotyczą",
    "kwestii", "temat", "ustalenia", "notatkach", "notatek",
})
QUERY_META_STOPSTEMS: frozenset[str] = frozenset({
    "bic", "gdzi", "kied", "dlacz", "skad", "jak", "ktor", "gdibi", "jesl",
    "moj", "twoj", "nasz", "swoj", "wiado", "wiedz", "powie", "opic", "wyjas",
    # V151 (Wektor I, H2): stemy negacji i partykuły pytającej — head-token
    # tematów śmieciowych ('nie_ma_porównania_jak', 'czy_warto').
    "nie", "czy", "cz",
})

# V151 (Wektor I, H2): stemy pomocnicze bariera dyskursywnej (POZA filtrem
# zapytań BM25/DBSF — 'obecni', 'wart', 'uslug' to legalne słowa zapytań,
# ale jako ENTITY-ROOT są śmieciem; rozszerzenie QUERY_META_STOPSTEMS nie
# może ich zawierać, żeby nie ucinać recallu wyszukiwania).
# V152 (FILAR I, ADR): 'z' — deterministyczny fallback barier aktu mowy dla
# head-tokenu przyimka ('Z jakiej wersji'), gdy POS spaCy jest niedostępny
# (shutdown/slow). Exact-match na stemie 'z' (żadne inne słowo nie stemmed
# do 'z' — 'zet'→'zet', 'zamek'→'zamek'), więc nazwy własne nie są demowane.
# ('ktori'/'ktory' warianty ASCII/KD — _bilingual_stem zwraca 'ktor' tylko
# dla form z diakrytykami (y→i po translacji), formy ASCII omijają tę regułę.)
_DISCOURSE_GARBAGE_EXTRA_STEMS: frozenset[str] = frozenset({"obecni", "wart", "ktori", "ktory", "z"})

# V151: klasy POS spaCy bariera gramatycznej — zaimek/spójnik/wykrzyknik
# jako HEAD tematu = nie-encja (temat gramatycznie nie może być podmiotem).
_DISCOURSE_GARBAGE_POS = frozenset({"PRON", "SCONJ", "INTJ", "DET", "AUX"})


def _discourse_stem(w: str) -> str:
    """Leniwy wrapper nad _bilingual_stem (definiowany niżej w module)."""
    return _bilingual_stem(w)


def is_discourse_garbage_subject(subject: Any) -> bool:
    """Bariera POS/gramatyczna V151 (Wektor I, H2): czy temat (subject trójki)
    jest śmieciem dyskursywnym — zaimkiem/wyrazem pytającym wcielonym w encję?

    Heurystyka head-token (pierwszy token 'jak_obecnie' → 'jak'):
      1. head ∈ QUERY_META_STOPWORDS lub stem(head) ∈ QUERY_META_STOPSTEMS
         ∪ _DISCOURSE_GARBAGE_EXTRA_STEMS → śmieć (deterministyczne, bez spaCy);
      2. uzupełniająco POS spaCy (zero-torch): head POS w
         _DISCOURSE_GARBAGE_POS (PRON/SCONJ/INTJ/DET/AUX) → śmieć;
         brak spaCy/modelu = graceful degradation (reguła 1 wystarcza).
    Bezpieczne do importu w Python 3.11 Thin-Client (spaCy może być nieobecne).
    """
    raw = str(subject or "").strip()
    if not raw:
        return False
    for tok in raw.replace("-", "_").replace(" ", "_").split("_"):
        if not tok:
            continue
        low = tok.lower()
        stem = _discourse_stem(low)
        if stem and stem in _DISCOURSE_GARBAGE_EXTRA_STEMS:
            return True
        break  # tylko head-token decidosuje o klasie gramatycznej tematu
    head = raw.replace("-", " ").replace("_", " ").split()[0].lower()
    if head in QUERY_META_STOPWORDS:
        return True
    if _discourse_stem(head) in QUERY_META_STOPSTEMS:
        return True
    # Uzupełniająca bariera POS (tylko gdy spaCy z taggerem dostępne).
    try:
        nlp = get_spacy_nlp("pl", fast_lemma_only=False)
        if nlp is not None and getattr(nlp, "has_tagger", lambda: "tagger" in getattr(nlp, "pipe_names", []))():
            doc = nlp(head)
            if doc and doc[0].pos_ in _DISCOURSE_GARBAGE_POS and (head in QUERY_META_STOPWORDS or len(head) <= 4):
                return True
    except Exception:
        pass
    return False



def _import_spacy_zero_torch() -> Any:
    """Imports spaCy in daemon (.pixi Py3.14) without allowing thinc to load PyTorch; returns None in Thin-Client."""
    if "spacy" in sys.modules:
        return sys.modules["spacy"]
    had_torch = "torch" in sys.modules
    if not had_torch:
        sys.modules["torch"] = None  # type: ignore[assignment]
    try:
        import spacy as _spacy
        return _spacy
    except ImportError:
        return None
    finally:
        if not had_torch and sys.modules.get("torch") is None:
            sys.modules.pop("torch", None)


@lru_cache(maxsize=4)
def get_spacy_nlp(lang: str = "pl", fast_lemma_only: bool = True) -> Any:
    """Loads and caches spaCy pipeline (`pl_core_news_sm` or `en_core_web_sm`) if available."""
    spacy_mod = _import_spacy_zero_torch()
    if spacy_mod is None:
        return None
    model_name = "pl_core_news_sm" if lang == "pl" else "en_core_web_sm"
    try:
        if fast_lemma_only:
            return spacy_mod.load(model_name, exclude=["ner", "parser", "senter"])
        return spacy_mod.load(model_name)
    except Exception:
        try:
            return spacy_mod.blank(lang)
        except Exception:
            return None


@lru_cache(maxsize=2)
def _get_spacy_lang_stopwords(lang: str = "pl") -> frozenset[str]:
    """Returns official spaCy stopwords for PL or EN when spaCy is available."""
    if _import_spacy_zero_torch() is None:
        return frozenset()
    try:
        if lang == "pl":
            from spacy.lang.pl.stop_words import STOP_WORDS as PL_STOP_WORDS
            return frozenset(w.lower() for w in PL_STOP_WORDS)
        from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP_WORDS
        return frozenset(w.lower() for w in EN_STOP_WORDS)
    except Exception:
        return frozenset()


class LanguageAssetManager:
    """Manages linguistic assets unifying spaCy official stop_words with external asset files."""

    _instance: Optional[LanguageAssetManager] = None

    def __init__(self, resources_dir: Optional[Path] = None) -> None:
        self.resources_dir = resources_dir or _RESOURCES_DIR
        self._stopwords_cache: Dict[str, Set[str]] = {}
        self._phatic_cache: Dict[str, Dict[str, List[str]]] = {}
        self._intent_cache: Dict[str, Tuple[str, ...]] = {}

    @classmethod
    def get_instance(cls) -> LanguageAssetManager:
        if cls._instance is None:
            cls._instance = LanguageAssetManager()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Resets the singleton instance for isolated test execution."""
        cls._instance = None

    def get_stopwords(self, lang: str = "pl") -> Set[str]:
        """Loads and returns stopwords for a given ISO language code (unifying spaCy STOP_WORDS + resources)."""
        lang = lang.lower()
        if lang in self._stopwords_cache:
            return self._stopwords_cache[lang]

        words: Set[str] = set(_get_spacy_lang_stopwords(lang))
        file_path = self.resources_dir / f"stopwords_{lang}.txt"
        if file_path.is_file():
            try:
                content = file_path.read_text(encoding="utf-8")
                words |= {
                    line.strip().lower()
                    for line in content.splitlines()
                    if line.strip() and not line.startswith("#")
                }
            except Exception as exc:
                logger.warning("Failed to load stopwords from %s: %s", file_path, exc)

        if not words:
            if lang == "pl":
                words = {
                    "w", "z", "na", "do", "i", "o", "to", "nie", "się", "jest",
                    "dla", "oraz", "ale", "jak", "że", "co", "po", "za", "ze", "od",
                }
            else:
                words = {
                    "the", "and", "for", "with", "this", "that", "from", "into",
                    "over", "is", "at", "by", "on", "in",
                }

        _DEONTIC_PROTECTED = {"zawsze", "nigdy", "tylko", "wyłącznie", "wylacznie", "musi", "zakaz", "always", "never", "only", "must", "forbidden"}
        words -= _DEONTIC_PROTECTED
        self._stopwords_cache[lang] = words
        return words

    def get_combined_stopwords(self, langs: Tuple[str, ...] = ("pl", "en")) -> Set[str]:
        """Returns union of stopwords across multiple languages."""
        cache_key = "+".join(sorted(langs))
        if cache_key in self._stopwords_cache:
            return self._stopwords_cache[cache_key]

        combined: Set[str] = set()
        for lang in langs:
            combined |= self.get_stopwords(lang)
        self._stopwords_cache[cache_key] = combined
        return combined

    def get_phatic_tokens(self, lang: str = "pl") -> Dict[str, List[str]]:
        """Loads and returns conversational / phatic token dictionaries."""
        lang = lang.lower()
        if lang in self._phatic_cache:
            return self._phatic_cache[lang]

        file_path = self.resources_dir / f"phatic_tokens_{lang}.json"
        data: Dict[str, List[str]] = {"trivial": [], "continuation": []}
        if file_path.is_file():
            try:
                loaded = json.loads(file_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data["trivial"] = [str(x).lower().strip() for x in loaded.get("trivial", [])]
                    data["continuation"] = [str(x).lower().strip() for x in loaded.get("continuation", [])]
            except Exception as exc:
                logger.warning("Failed to load phatic tokens from %s: %s", file_path, exc)

        if not data["trivial"]:
            data["trivial"] = [
                "cześć", "czesc", "siema", "hej", "witaj", "hello", "hi",
                "dzięki", "dzieki", "super", "ok", "dobrze", "jasne", "thanks",
            ]
        if not data["continuation"]:
            data["continuation"] = [
                "kontynuuj", "kontynuacja", "dalej", "nastepny", "następny",
                "continue", "next", "more",
            ]

        self._phatic_cache[lang] = data
        return data

    def get_intent_keywords(self, lang: str = "pl") -> Tuple[str, ...]:
        """Loads and returns knowledge/intent query trigger keywords."""
        lang = lang.lower()
        if lang in self._intent_cache:
            return self._intent_cache[lang]

        file_path = self.resources_dir / f"intent_keywords_{lang}.json"
        keywords: List[str] = []
        if file_path.is_file():
            try:
                loaded = json.loads(file_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and "intent_keywords" in loaded:
                    keywords = [str(x).lower().strip() for x in loaded["intent_keywords"] if str(x).strip()]
            except Exception as exc:
                logger.warning("Failed to load intent keywords from %s: %s", file_path, exc)

        if not keywords:
            keywords = [
                "chain", "hop", "fakt", "dawka", "port", "token", "rule", "config",
                "szukaj", "znajdź", "pokaż", "sprawdź", "notat", "obsidian", "peptyd",
                "skrypt", "baza", "model", "protokół", "badani", "raport", "informac", "pamięć", "wiedza",
            ]

        res = tuple(keywords)
        self._intent_cache[lang] = res
        return res


_PROTECTED_ENGLISH_ROOTS: frozenset[str] = frozenset({
    "node", "code", "cache", "base", "pipeline", "cookie", "zombie", "calorie",
    "movie", "rookie", "queue", "state", "engine", "scope", "table", "file",
    "rule", "mode", "core", "type", "gate", "route", "service", "database",
    "message", "trace", "interface", "profile", "storage", "source", "module",
    "time", "rate", "device", "price", "space", "phase", "case", "size",
    "line", "site", "side", "name", "guideline", "outline", "baseline",
    "timeline", "share", "software", "hardware", "client", "server", "cluster",
    "vector", "tensor",
})

_EXCLUDED_IE_WORDS: frozenset[str] = frozenset({
    "cookie", "cookies", "zombie", "zombies", "calorie", "calories",
    "movie", "movies", "rookie", "rookies", "selfie", "selfies",
    "prairie", "hippie",
})


def _heuristic_stem_fallback(w_low: str) -> str:
    """F-029 (audit v93): brak globalnego y->i i brak twardego obciecia [:5].

    y->i to fonetyczna alternacja PL — sensowna tylko po translacji diakrytow
    (wypowiedz/wypowiedz). Dla slow czysto ASCII (typ/tip, gry/tri, pastes/pasted)
    zamiana byla destrukcyjna, a [:5] sklealo niezalezne lemma po angielsku.
    """
    if w_low in _PROTECTED_ENGLISH_ROOTS:
        return w_low
    tr = str.maketrans("ąćęłńóśźż", "acelnoszz")
    has_pl_diacritics = any(ch in "ąćęłńóśźż" for ch in w_low)
    w_clean = w_low.translate(tr)
    if w_clean in _PROTECTED_ENGLISH_ROOTS:
        return w_clean
    if has_pl_diacritics:
        w_clean = w_clean.replace("y", "i")
    for suf in ("ach", "ami", "owi", "em", "om", "ow", "a", "e", "i", "o", "u"):
        if len(w_clean) > len(suf) + 2 and w_clean.endswith(suf):
            cand = w_clean[:-len(suf)]
            if w_clean in _PROTECTED_ENGLISH_ROOTS:
                break
            w_clean = cand
            break
    return w_clean


_GRAPH_NOISE_ENTITIES = frozenset({
    "w", "z", "na", "do", "i", "o", "to", "nie", "się", "sie", "jest", "być", "byc",
    "dla", "oraz", "ale", "jak", "jaki", "jaka", "jakie", "gdzie", "kiedy", "czy",
    "że", "ze", "co", "po", "za", "od", "przez", "przy", "nad", "pod", "przed",
    "przecież", "przeciez", "ale_jak", "ma", "mam", "masz", "mają", "ten", "ta", "te",
    "tym", "tego", "tej", "tych", "tylko", "już", "juz", "jeszcze", "bardzo",
    "fact", "working", "working_fact", "stated_memory", "episodic_event", "canonical_spec",
    "set", "is_set", "implemented", "is_implemented", "true", "false", "none", "null",
    "user", "assistant", "conversation", "default", "unknown", "test", "ok",
})


def is_valid_graph_entity(name: str) -> bool:
    """V109.7 Canonical graph entity validator: blocks stopwords, bare numbers, schema words, and multiline prose dumps."""
    s = str(name or "").strip()
    if not s or len(s) > 80:
        return False
    if "\n" in s or "\r" in s or "<!--" in s or "[user]" in s.lower() or "[conversation]" in s.lower():
        return False
    s_low = s.lower().strip("._-:;,()[]{}\"' ")
    if not s_low:
        return False
    if s_low in _GRAPH_NOISE_ENTITIES:
        return False
    if s_low.replace(".", "").isdigit():
        return False
    words = s_low.split()
    if len(words) > 7:
        return False
    try:
        if len(s_low) >= 2 and s_low in LanguageAssetManager.get_instance().get_combined_stopwords():
            return False
    except Exception as exc:
        logger.debug("is_valid_query_word stopword check failed: %s", exc)
    return True


@lru_cache(maxsize=65536)
def _bilingual_stem(w: str) -> str:
    """V110 (F-029) Morphological stemmer: spaCy lemma for every word len>=3 (PL i EN), heuristic fallback for len<=2/digits/_/None-lemma."""
    w_low = str(w or "").lower().strip()
    if not w_low:
        return ""
    if w_low in _PROTECTED_ENGLISH_ROOTS:
        return w_low
    if len(w_low) <= 2 or any(ch.isdigit() for ch in w_low) or "_" in w_low:
        return _heuristic_stem_fallback(w_low)
    h_stem = _heuristic_stem_fallback(w_low)
    nlp = get_spacy_nlp("pl", fast_lemma_only=True)
    if nlp is not None:
        try:
            doc = nlp(w_low)
            if len(doc) > 0:
                lemma = (doc[0].lemma_ or w_low).lower().strip()
                if lemma:
                    if lemma in _PROTECTED_ENGLISH_ROOTS:
                        return lemma
                    return _heuristic_stem_fallback(lemma)
        except Exception as exc:
            logger.debug("_bilingual_stem spacy processing failed: %s", exc)
    return h_stem


@lru_cache(maxsize=32768)
def is_non_salient_key_token(token: str) -> bool:
    """Dynamic POS + stopword + namespace-prefix check replacing static GENERIC_KEY_PARTS."""
    t_low = str(token or "").lower().strip()
    if len(t_low) < 3 or t_low in _NAMESPACE_SCHEMA_TOKENS:
        return True
    stops = LanguageAssetManager.get_instance().get_combined_stopwords(("pl", "en"))
    if t_low in stops:
        return True
    nlp = get_spacy_nlp("pl", fast_lemma_only=True)
    if nlp is not None:
        try:
            doc = nlp(t_low)
            if len(doc) > 0:
                tok = doc[0]
                if tok.is_stop or tok.pos_ in {"PRON", "DET", "ADP", "CCONJ", "SCONJ", "PART", "AUX"}:
                    return True
        except Exception as exc:
            logger.debug("is_non_salient_key_token pos check failed: %s", exc)
    return False


@lru_cache(maxsize=65536)
def spacy_lemmatize_word(w: str) -> str:
    """Returns full canonical spaCy lemma for a single word (`peptydami` -> `peptyd`, `Obsidianie` -> `obsidian`)."""
    w_low = str(w or "").lower().strip()
    if not w_low:
        return ""
    if w_low in _PROTECTED_ENGLISH_ROOTS:
        return w_low
    nlp = get_spacy_nlp("pl", fast_lemma_only=True)
    if nlp is not None:
        try:
            doc = nlp(w_low)
            if len(doc) > 0:
                lemma = (doc[0].lemma_ or w_low).lower().strip()
                if (
                    lemma == w_low
                    and w_low not in _EXCLUDED_IE_WORDS
                    and len(w_low) >= 6
                    and w_low.endswith("ie")
                    and w_low[-3] in "nbmrtpld"
                ):
                    lemma = w_low[:-2]
                if lemma:
                    return lemma
        except Exception as exc:
            logger.debug("spacy_lemmatize_word failed: %s", exc)
    return w_low
