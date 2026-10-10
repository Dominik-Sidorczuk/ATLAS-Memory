"""BM25 Lexical scoring strategy with single-pass DF calculation and exact phrase boost."""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Set

from atlas_memory.cognitive.retrieval.strategies.base import BaseScoringStrategy
from atlas_memory.core.linguistics import _bilingual_stem

BM25_K1: float = 1.2
BM25_B: float = 0.0
LEX_EXACT_PHRASE_BONUS: float = 1.0


class Bm25LexicalStrategy(BaseScoringStrategy):
    """Okapi BM25 Lexical Scorer with single-pass DF calculation and phrase boost."""

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B) -> None:
        self.k1 = k1
        self.b = b

    @staticmethod
    def tokenize_candidates(candidates: List[Any]) -> List[List[str]]:
        """Extracts token stems (subject + object) for candidates corpus."""
        doc_tokens: List[List[str]] = []
        for c in candidates:
            s = str(c.get("subject", "") if hasattr(c, "get") else getattr(c, "subject", "")).lower()
            o = str(c.get("object", "") if hasattr(c, "get") else getattr(c, "object", "")).lower()
            cand_text = f"{s} {o}".replace("_", " ").replace(":", " ").replace("-", " ")
            doc_tokens.append([_bilingual_stem(w) for w in re.findall(r"[^\W_]+", cand_text) if len(w) >= 3])
        return doc_tokens

    @staticmethod
    def calculate_scores(
        q_stems: List[str],
        doc_tokens: List[List[str]],
        k1: float = BM25_K1,
        b: float = BM25_B,
        total_corpus_size: Optional[int] = None,
    ) -> List[float]:
        """Calculates Okapi BM25 scores across the candidate corpus."""
        if not doc_tokens:
            return []
        df: Dict[str, int] = {qt: 0 for qt in q_stems}
        wanted = set(q_stems)
        for terms in doc_tokens:
            for qt in set(terms) & wanted:
                df[qt] += 1
        n_docs = total_corpus_size if total_corpus_size is not None else len(doc_tokens)
        idf: Dict[str, float] = {
            qt: math.log(1.0 + (n_docs - nq + 0.5) / (nq + 0.5)) for qt, nq in df.items() if nq > 0
        }
        scores: List[float] = []
        for terms in doc_tokens:
            total = 0.0
            for qt in q_stems:
                term_idf = idf.get(qt)
                if term_idf is None:
                    continue
                tf = terms.count(qt)
                total += term_idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b))
            scores.append(total)
        return scores

    @staticmethod
    def apply_phrase_boost(q_lower: str, candidates: List[Any], raw_bm25_scores: List[float]) -> None:
        """Applies exact phrase match boost to raw BM25 scores."""
        for i, c in enumerate(candidates):
            s = str(c.get("subject", "") if hasattr(c, "get") else getattr(c, "subject", "")).lower()
            o = str(c.get("object", "") if hasattr(c, "get") else getattr(c, "object", "")).lower()
            if q_lower and (q_lower in s or q_lower in o):
                raw_bm25_scores[i] += LEX_EXACT_PHRASE_BONUS

    def score(self, query: str, candidates: List[Any], **kwargs: Any) -> List[float]:
        """Scores candidates using BM25 lexical relevance and attaches lexical features."""
        q_lower = query.strip().lower()
        q_clean = q_lower.replace("_", " ").replace(":", " ").replace("-", " ")
        q_stems = [_bilingual_stem(w) for w in re.findall(r"[^\W_]+", q_clean) if len(w) >= 3]

        doc_tokens = self.tokenize_candidates(candidates)
        raw_bm25_scores = self.calculate_scores(q_stems, doc_tokens)
        self.apply_phrase_boost(q_lower, candidates, raw_bm25_scores)

        max_bm25 = max(raw_bm25_scores, default=0.0)
        for i, c in enumerate(candidates):
            raw_bm25 = raw_bm25_scores[i] if i < len(raw_bm25_scores) else 0.0
            norm_bm25 = (raw_bm25 / max_bm25) if max_bm25 > 0.0 else 0.0
            tokens_set: Set[str] = set(doc_tokens[i]) if i < len(doc_tokens) else set()

            if isinstance(c, dict):
                c["_raw_bm25"] = raw_bm25
                c["_lex_score"] = norm_bm25
                c["_doc_tokens_set"] = tokens_set
                c["bm25_score"] = round(norm_bm25, 4)
                c["sparse_score"] = round(norm_bm25, 4)
            else:
                if hasattr(c, "bm25_score"):
                    c.bm25_score = round(norm_bm25, 4)
                if hasattr(c, "features"):
                    c.features.raw_bm25 = raw_bm25
                    c.features.bm25_score = round(norm_bm25, 4)
                    c.features.lexical_score = round(norm_bm25, 4)
                    c.features.doc_tokens = tokens_set

        return raw_bm25_scores
