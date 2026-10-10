"""V92: natywny kanał sparse dla QdrantVectorStore (named sparse vector, BM25/IDF w silniku).

Stan przed: kandydaci retrievalu pochodzili z JEDNEGO wektora gęstego, a leksykalność
(`cognitive/retrieval/score.py`, Okapi BM25) była liczona w Pythonie PO zwrocie kandydatów.
Termin, którego wektor gęsty nie widzi, nie miał jak wejść do zbioru kandydatów — BM25 mógł
tylko przestawić to, co już przyszło z dense. W `src/` nie było żadnego wystąpienia `sparse`.

Stan po: kolekcja nosi drugi (NAZWANY) wektor sparse; zapytanie idzie do silnika jako
prefetch dense + prefetch sparse, a `Fusion.DBSF` scala je po stronie Qdranta. Ranking
końcowy nadal liczy Static-Anchor DBSF w `score.py` (bez RRF) — ten moduł dostarcza
WYŁĄCZNIE reprezentację leksykalną dla kanału wyszukiwania.

Wartości wektora to surowe (sublinearne) częstości terminów; IDF dokłada silnik przez
`Modifier.IDF` w konfiguracji kolekcji (rekomendacja Qdranta dla BM25) — dzięki temu
nie ma drugiej, rozjeżdżającej się implementacji IDF w Pythonie.

Tokenizacja jest CELOWO ta sama co w `score.py` (`_bilingual_stem` + stopwords PL/EN),
żeby kanał sparse i leksykalny rerank mówiły tym samym słownikiem.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Dict, List

from qdrant_client import models as qmodels

from atlas_memory.core.linguistics import LanguageAssetManager, _bilingual_stem, spacy_lemmatize_word

SPARSE_VECTOR_NAME = "sparse"
# 2**31-1: kubełki haszy mieszczą się w uint32 wymaganym przez Qdrant.
SPARSE_HASH_SPACE = 2**31 - 1
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _stopwords() -> set:
    """Stopwordy PL z assetów (te same, których używa `_query_stems` w score.py)."""
    try:

        raw = LanguageAssetManager.get_instance().get_stopwords("pl") or []
        return {str(w).lower() for w in raw}
    except Exception:
        return set()


def sparse_terms(text: str, *, min_len: int = 3, drop_stopwords: bool = True) -> List[str]:
    """Stemy terminów tekstu — wspólny słownik z leksykalnym rerankiem (`score.py`)."""


    flat = str(text or "").replace("_", " ").replace(":", " ").replace("-", " ")
    stops = _stopwords() if drop_stopwords else set()
    out: List[str] = []
    for w in TOKEN_RE.findall(flat):
        wl = w.lower()
        if len(wl) < min_len:
            continue
        lemma = spacy_lemmatize_word(wl)
        stem = _bilingual_stem(wl)
        if drop_stopwords and (wl in stops or lemma in stops or stem in stops):
            continue
        if lemma:
            out.append(lemma)
        if stem and stem != lemma:
            out.append(stem)
    return out


def term_index(term: str) -> int:
    """Deterministyczny kubełek termu (blake2b, 8 B) w przestrzeni uint32."""
    digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % SPARSE_HASH_SPACE


def _sparse_vector(weights: Dict[int, float]) -> "qmodels.SparseVector":
    idx = sorted(weights)
    return qmodels.SparseVector(indices=idx, values=[float(weights[i]) for i in idx])


def encode_sparse_document(text: str) -> "qmodels.SparseVector":
    """Wektor sparse dokumentu: sublinearne tf (1 + ln tf) per unikalny stemy.

    Sublinearność (a nie surowy count) tłumi dokumenty powtarzające jeden termin —
    to ta sama rodzina dyskontowania co sufiks nasycenia BM25 w `score.py`.
    """
    counts: Dict[int, int] = {}
    for term in sparse_terms(text):
        i = term_index(term)
        counts[i] = counts.get(i, 0) + 1
    return _sparse_vector({i: 1.0 + math.log(c) for i, c in counts.items() if c > 0})


def encode_sparse_query(query: str) -> "qmodels.SparseVector":
    """Wektor sparse zapytania: 1.0 per unikalny stem (IDF dokłada silnik)."""
    uniq = {term_index(t) for t in sparse_terms(query)}
    return _sparse_vector({i: 1.0 for i in uniq})

