"""Domain models for ATLAS Cognitive Retrieval (V27+ Architecture).

Clean, strictly typed Pydantic v2 domain models:
- CandidateEntity: immutable value object representing raw retrieval candidates.
- CandidateFeatures: extracted scoring features (dense, lexical, graph, authority).
- ScoredCandidate: result of the scoring pipeline with typed components.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CandidateEntity(BaseModel):
    """Immutable value object representing an un-scored retrieval candidate."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    key: str
    value: Any = ""
    subject: str = ""
    predicate: Optional[str] = None
    object: Any = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = "user_explicit"
    source_type: str = "user_explicit"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    raw_scores: Dict[str, float] = Field(default_factory=dict)
    vector: Optional[List[float]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CandidateEntity:
        """Construct CandidateEntity from a dictionary with key normalization."""
        raw_key = str(data.get("key") or data.get("subject") or data.get("source_key") or "").strip()
        val = data.get("value") if "value" in data else data.get("object", "")
        raw_sc = {
            k: float(data[k])
            for k in ("vector_score", "dense_score", "sparse_score", "graph_score", "similarity")
            if k in data and isinstance(data[k], (int, float))
        }
        actual_channel = str(data.get("source") or data.get("source_type") or "user_explicit")
        actual_origin = str(data.get("source_type") or "user_explicit")
        return cls(
            key=raw_key,
            value=val,
            subject=str(data.get("subject") or raw_key).strip(),
            predicate=data.get("predicate"),
            object=data.get("object", val),
            confidence=float(data.get("confidence", 1.0)),
            source=actual_channel,
            source_type=actual_origin,
            metadata=dict(data.get("metadata") or {}),
            raw_scores=raw_sc,
            vector=data.get("vector"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert CandidateEntity to dictionary."""
        d: Dict[str, Any] = {
            "key": self.key,
            "subject": self.subject or self.key,
            "value": self.value,
            "object": self.object or self.value,
            "confidence": self.confidence,
            "source": self.source,
            "source_type": self.source_type,
            "metadata": dict(self.metadata),
        }
        if self.predicate:
            d["predicate"] = self.predicate
        if self.vector is not None:
            d["vector"] = self.vector
        d.update(self.raw_scores)
        return d


class CandidateFeatures(BaseModel):
    """Internal features extracted for candidate scoring."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vector_score: float = 0.0
    dense_score: float = 0.0
    bm25_score: float = 0.0
    lexical_score: float = 0.0
    graph_score: float = 0.0
    raw_bm25: float = 0.0
    doc_tokens: Set[str] = Field(default_factory=set)
    is_graph_candidate: bool = False
    is_trusted: bool = False
    rank_vector: int = 0
    rank_lexical: int = 0
    tiebreak: float = 0.0
    ttl_factor: float = 1.0


class ScoredCandidate(BaseModel):
    """Domain model representing a retrieval candidate with scored components."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    candidate: Optional[CandidateEntity] = None
    key: str = ""
    subject: str = ""
    value: Any = ""
    object: Any = ""
    predicate: Optional[str] = None
    dense_score: float = 0.0
    bm25_score: float = 0.0
    graph_score: float = 0.0
    final_score: float = 0.0
    score: float = 0.0
    rank: int = 0
    features: CandidateFeatures = Field(default_factory=CandidateFeatures)
    source: Optional[str] = None
    source_type: str = "user_explicit"
    confidence: float = 1.0
    timestamp: float = Field(default_factory=time.time)
    relevance_reason: str = ""
    is_superseded: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _sync_scores(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "final_score" in data and ("score" not in data or data["score"] == 0.0):
                data["score"] = data["final_score"]
            elif "score" in data and ("final_score" not in data or data["final_score"] == 0.0):
                data["final_score"] = data["score"]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ScoredCandidate:
        """Construct a ScoredCandidate from a raw candidate dictionary."""
        d = dict(data)
        cand_key = str(d.get("key") or d.get("subject") or d.get("source_key") or "")
        cand_val = d.get("value") if "value" in d else d.get("object", "")
        dense_sc = float(d.get("dense_score", d.get("vector_score", 0.0)))
        bm25_sc = float(d.get("bm25_score", d.get("lexical_score", d.get("_bm25", 0.0))))
        graph_sc = float(d.get("graph_score", 0.0))
        final_sc = float(d.get("final_score", d.get("score", 0.0)))

        feats_data = d.get("features")
        if isinstance(feats_data, CandidateFeatures):
            feats = feats_data
        elif isinstance(feats_data, dict):
            feats = CandidateFeatures(**feats_data)
        else:
            feats = CandidateFeatures(
                vector_score=dense_sc, dense_score=dense_sc,
                bm25_score=bm25_sc, graph_score=graph_sc,
            )

        return cls(
            candidate=CandidateEntity.from_dict(d),
            key=cand_key,
            subject=str(d.get("subject") or cand_key),
            value=cand_val,
            object=d.get("object", cand_val),
            predicate=d.get("predicate"),
            dense_score=dense_sc,
            bm25_score=bm25_sc,
            graph_score=graph_sc,
            final_score=final_sc,
            score=final_sc,
            rank=int(d.get("rank", 0)),
            features=feats,
            source=d.get("source"),
            source_type=str(d.get("source_type") or "user_explicit"),
            confidence=float(d.get("confidence", 1.0)),
            timestamp=float(d.get("timestamp", time.time())),
            relevance_reason=str(d.get("relevance_reason", "")),
            is_superseded=bool(d.get("is_superseded", False)),
            metadata=dict(d.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert ScoredCandidate to a dictionary maintaining 100% contract compatibility."""
        return {
            "key": self.key, "subject": self.subject or self.key,
            "value": self.value, "object": self.object or self.value,
            "predicate": self.predicate, "score": self.final_score,
            "final_score": self.final_score, "dense_score": self.dense_score,
            "vector_score": self.dense_score, "bm25_score": self.bm25_score,
            "graph_score": self.graph_score, "rank": self.rank,
            "source": self.source, "source_type": self.source_type,
            "confidence": self.confidence, "timestamp": self.timestamp,
            "relevance_reason": self.relevance_reason,
            "is_superseded": self.is_superseded, "metadata": dict(self.metadata),
        }


__all__ = ["CandidateEntity", "CandidateFeatures", "ScoredCandidate"]
