"""Canonical exports for retrieval scoring strategies."""
from __future__ import annotations

from atlas_memory.cognitive.retrieval.strategies.base import BaseScoringStrategy
from atlas_memory.cognitive.retrieval.strategies.dense import DenseCosineStrategy
from atlas_memory.cognitive.retrieval.strategies.fusion import (
    EntropySalienceFilter,
    RrfStrategy,
    StaticAnchorDbsfFusion,
    compute_content_fingerprint,
    compute_record_rank,
    get_authority_score,
    get_record_authority_score,
)
from atlas_memory.cognitive.retrieval.strategies.graph import GraphSynergyStrategy
from atlas_memory.cognitive.retrieval.strategies.lexical import Bm25LexicalStrategy
from atlas_memory.cognitive.retrieval.strategies.pipeline import (
    ScoringPipeline,
    query_stems,
)

__all__ = [
    "ScoringPipeline",
    "query_stems",
    "BaseScoringStrategy",
    "Bm25LexicalStrategy",
    "DenseCosineStrategy",
    "EntropySalienceFilter",
    "GraphSynergyStrategy",
    "RrfStrategy",
    "StaticAnchorDbsfFusion",
    "compute_content_fingerprint",
    "compute_record_rank",
    "get_authority_score",
    "get_record_authority_score",
]
