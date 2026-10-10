"""Canonical scoring pipeline and query stem helpers for multi-signal retrieval."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from atlas_memory.cognitive.retrieval.models import (
    CandidateEntity,
    ScoredCandidate,
)
from atlas_memory.cognitive.retrieval.strategies.base import BaseScoringStrategy
from atlas_memory.cognitive.retrieval.strategies.dense import DenseCosineStrategy
from atlas_memory.cognitive.retrieval.strategies.fusion import (
    RrfStrategy,
    StaticAnchorDbsfFusion,
    compute_content_fingerprint,
    get_record_authority_score,
)
from atlas_memory.cognitive.retrieval.strategies.graph import GraphSynergyStrategy
from atlas_memory.cognitive.retrieval.strategies.lexical import Bm25LexicalStrategy
from atlas_memory.core.linguistics import (
    QUERY_META_STOPSTEMS,
    QUERY_META_STOPWORDS,
    LanguageAssetManager,
    _bilingual_stem,
)

logger = logging.getLogger("atlas.cognitive.retrieval.strategies.pipeline")


def query_stems(query: str) -> List[str]:
    """Stems for query terms after filtering stopwords and interrogatives."""
    polish_stopwords = LanguageAssetManager.get_instance().get_stopwords("pl")
    raw_words = [w.lower() for w in re.findall(r"[^\W_]+", query) if len(w) >= 3]
    q_words = [
        w for w in raw_words
        if w not in polish_stopwords and w not in QUERY_META_STOPWORDS
        and _bilingual_stem(w) not in polish_stopwords and _bilingual_stem(w) not in QUERY_META_STOPSTEMS
    ]
    return [_bilingual_stem(w) for w in (q_words or raw_words)]


class ScoringPipeline:
    """Orchestrates candidate scoring across dense, lexical, graph channels and fusion strategies (Strategy Pattern)."""

    def __init__(
        self,
        strategies: Optional[List[BaseScoringStrategy]] = None,
        fusion: Optional[BaseScoringStrategy] = None,
        fusion_strategy: Optional[str] = None,
    ) -> None:
        self.strategies = strategies if strategies is not None else [
            DenseCosineStrategy(),
            Bm25LexicalStrategy(),
            GraphSynergyStrategy(),
        ]
        from atlas_memory.core.config import get_config
        self.strategy_name = (fusion_strategy or get_config().fusion_strategy).lower().strip()
        self.dbsf_strategy = StaticAnchorDbsfFusion()
        self.rrf_strategy = RrfStrategy()
        self.bm25_strategy = Bm25LexicalStrategy()
        self.fusion = fusion or (self.rrf_strategy if self.strategy_name == "rrf" else self.dbsf_strategy)

    @staticmethod
    def dedup_rank_key(c: Dict[str, Any]) -> Tuple[int, float, float]:
        """Tiebreak ranking key for candidate deduplication."""
        return (
            get_record_authority_score(str(c.get("subject", ""))),
            float(c.get("graph_score", 0.0) or 0.0),
            float(c.get("vector_score", 0.0) or 0.0),
        )

    def dedup_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicates candidates by key and content fingerprint with tiebreak."""
        dedup: Dict[str, Dict[str, Any]] = {}
        for c in candidates:
            cand_key = str(c.get("key") or c.get("subject") or id(c)).strip()
            obj_val = str(c.get("object", "") or c.get("value", "")).strip()
            fp_content = compute_content_fingerprint(obj_val)
            
            # 1. 'key' without 'source' (raw manual candidates) -> preserve per key
            # 2. Short/atomic config value (e.g. port "5432", boolean, short literal) -> preserve distinct keys
            is_atomic_value = len(obj_val) <= 16 or len(obj_val.split()) <= 2
            if "key" in c and "source" not in c:
                fp = f"{fp_content}::{c['key']}" if fp_content else str(c["key"])
            elif is_atomic_value and cand_key and not cand_key.startswith("<"):
                fp = f"{cand_key}::{fp_content}" if fp_content else cand_key
            else:
                fp = fp_content if fp_content else cand_key

            if fp not in dedup:
                dedup[fp] = c
            else:
                prev = dedup[fp]
                cur_vs, prev_vs = float(c.get("vector_score", 0.0) or 0.0), float(prev.get("vector_score", 0.0) or 0.0)
                cur_gs, prev_gs = float(c.get("graph_score", 0.0) or 0.0), float(prev.get("graph_score", 0.0) or 0.0)
                merged_vec = max(prev_vs, cur_vs)
                merged_graph = max(prev_gs, cur_gs)
                merged_ghit = bool(prev.get("graph_hit") or c.get("graph_hit") or merged_graph > 0.0)
                target = c if self.dedup_rank_key(c) > self.dedup_rank_key(prev) else prev
                target["vector_score"] = merged_vec
                target["graph_score"] = merged_graph
                target["graph_hit"] = merged_ghit
                dedup[fp] = target
        return list(dedup.values())

    @staticmethod
    def build_rank_indices(candidates: List[Dict[str, Any]]) -> Tuple[Dict[int, int], Dict[int, int]]:
        """Builds dense and lexical rank mappings for RRF."""
        by_vec = sorted(candidates, key=lambda x: float(x.get("_vec_score", 0.0)), reverse=True)
        by_lex = sorted(candidates, key=lambda x: float(x.get("_lex_score", 0.0)), reverse=True)
        return {id(c): i + 1 for i, c in enumerate(by_vec)}, {id(c): i + 1 for i, c in enumerate(by_lex)}

    @staticmethod
    def attach_scoring_features(
        candidates: List[Dict[str, Any]],
        raw_bm25_scores: List[float],
        max_bm25: float,
        doc_tokens: List[List[str]],
    ) -> None:
        """Attaches intermediate scoring features to candidate dictionaries."""
        for i, c in enumerate(candidates):
            bm_norm = (raw_bm25_scores[i] / max_bm25) if max_bm25 > 0.0 else 0.0
            raw_gs = c.get("graph_score")
            if raw_gs is None and isinstance(c.get("metadata"), dict):
                raw_gs = c["metadata"].get("graph_score", 0.0)
            c["_lex_score"] = round(bm_norm, 4)
            c["_raw_bm25"] = float(raw_bm25_scores[i]) if i < len(raw_bm25_scores) else 0.0
            c["_vec_score"] = float(c.get("vector_score", 0.0))
            c["_graph_score"] = max(0.0, min(1.0, float(raw_gs or 0.0)))
            c["_doc_tokens_set"] = set(doc_tokens[i]) if i < len(doc_tokens) else set()

    def score(
        self,
        query: str,
        candidates: List[Any],
        fusion_strategy: Optional[str] = None,
        as_models: bool = False,
    ) -> List[Any]:
        """Executes the full scoring pipeline over candidates."""
        if not candidates:
            return []

        is_entity_input = bool(isinstance(candidates[0], (CandidateEntity, ScoredCandidate)))
        normalized = [c.to_dict() if isinstance(c, (CandidateEntity, ScoredCandidate)) else dict(c) for c in candidates]
        deduped = self.dedup_candidates(normalized)
        q_stems, q_lower = query_stems(query), query.lower().strip()
        strategy = (fusion_strategy or self.strategy_name).lower().strip()

        doc_tokens = self.bm25_strategy.tokenize_candidates(deduped)
        raw_bm25 = self.bm25_strategy.calculate_scores(q_stems, doc_tokens)
        self.bm25_strategy.apply_phrase_boost(q_lower, deduped, raw_bm25)
        self.attach_scoring_features(deduped, raw_bm25, max(raw_bm25, default=0.0), doc_tokens)

        rank_vec, rank_lex = self.build_rank_indices(deduped)
        active = self.rrf_strategy if strategy == "rrf" else self.dbsf_strategy
        active.score(query=query, candidates=deduped, rank_vec=rank_vec, rank_lex=rank_lex)

        sorted_cands = sorted(deduped, key=lambda x: x["score"], reverse=True)
        if as_models or is_entity_input:
            return [ScoredCandidate.from_dict({**c, "rank": i + 1}) for i, c in enumerate(sorted_cands)]
        return sorted_cands


__all__ = ["ScoringPipeline", "query_stems"]
