"""V131 regression guard: query (BM25) and PPR stopword sets are semantically distinct.

Dowód błędu wykrytego w audycie zero-trust commita dced21c: scalenie
QUERY_META_* z PPR_META_* przestało filtrować m.in. "który", "temat",
"notatkach" w rdzeniach zapytania BM25. Testy nie łapały tej zmiany.
"""
from __future__ import annotations

from atlas_memory.cognitive.retrieval.strategies.pipeline import query_stems
from atlas_memory.core.linguistics import (
    PPR_META_STOPSTEMS,
    PPR_META_STOPWORDS,
    QUERY_META_STOPSTEMS,
    QUERY_META_STOPWORDS,
    _bilingual_stem,
)


def test_query_and_ppr_sets_are_not_aliased() -> None:
    assert QUERY_META_STOPWORDS is not PPR_META_STOPWORDS
    assert QUERY_META_STOPSTEMS is not PPR_META_STOPSTEMS
    assert QUERY_META_STOPWORDS != PPR_META_STOPWORDS


def test_query_only_framing_words_are_filtered_from_query_stems() -> None:
    for word in ("który", "temat", "notatkach", "jeśli", "odnośnie"):
        assert word in QUERY_META_STOPWORDS
        stems = query_stems(f"{word} kubernetes")
        assert _bilingual_stem(word) not in stems or _bilingual_stem(word) == _bilingual_stem("kubernetes")
        assert _bilingual_stem("kubernetes") in stems


def test_ppr_only_words_stay_ppr_only() -> None:
    assert "przecież" in PPR_META_STOPWORDS
    assert "przecież" not in QUERY_META_STOPWORDS
