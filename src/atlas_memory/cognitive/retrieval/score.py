"""V81/V125 Score: Clean facade delegating to ScoringPipeline (SOLID / Strategy Pattern).
Proweniencja stałych BM25: Okapi k1=1.2, b=0.0, bonus=1.0 (V86.1/V87/T2, Robertsona/Sparck Jones).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from atlas_memory.cognitive.retrieval.models import ScoredCandidate
from atlas_memory.cognitive.retrieval.strategies import (
    Bm25LexicalStrategy,
    ScoringPipeline,
    compute_content_fingerprint,
    compute_record_rank,
    get_authority_score,
    get_record_authority_score,
    query_stems,
)
from atlas_memory.cognitive.retrieval.strategies.fusion import (
    DBSF_ALPHA,
    DBSF_ALPHA_BM25,
    DBSF_ALPHA_DENSE,
    DBSF_ALPHA_GRAPH,
    EntropySalienceFilter,
)
from atlas_memory.cognitive.retrieval.strategies.lexical import BM25_B, BM25_K1, LEX_EXACT_PHRASE_BONUS
from atlas_memory.core.linguistics import _bilingual_stem

_DEFAULT_PIPELINE = ScoringPipeline()
_query_stems = query_stems
_tokenize_candidates_for_bm25 = Bm25LexicalStrategy.tokenize_candidates
bm25_scores_for_corpus = Bm25LexicalStrategy.calculate_scores
apply_entropy_salience_filter = EntropySalienceFilter().filter
compute_normalized_shannon_entropy = EntropySalienceFilter.compute_normalized_entropy


def score_candidates(
    query: str,
    candidates: List[Any],
    fusion_strategy: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Ranks candidates using Static-Anchor DBSF (or RRF) with minimal tiebreak via ScoringPipeline."""
    if not candidates:
        return []
    pipeline = ScoringPipeline(fusion_strategy=fusion_strategy) if fusion_strategy else _DEFAULT_PIPELINE
    scored = pipeline.score(query=query, candidates=candidates, fusion_strategy=fusion_strategy, as_models=False)
    return [c.to_dict() if isinstance(c, ScoredCandidate) else c for c in scored]


__all__ = [
    "BM25_B", "BM25_K1", "DBSF_ALPHA", "DBSF_ALPHA_BM25", "DBSF_ALPHA_DENSE", "DBSF_ALPHA_GRAPH",
    "LEX_EXACT_PHRASE_BONUS", "_bilingual_stem",
    "_query_stems", "_tokenize_candidates_for_bm25", "apply_entropy_salience_filter", "bm25_scores_for_corpus",
    "compute_content_fingerprint", "compute_normalized_shannon_entropy", "compute_record_rank",
    "get_authority_score", "get_record_authority_score", "score_candidates",
]
