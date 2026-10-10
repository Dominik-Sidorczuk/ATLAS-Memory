"""V81/V106 Cognitive Retrieval Package."""
from __future__ import annotations

from atlas_memory.cognitive.retrieval.candidates import gather_candidates
from atlas_memory.cognitive.retrieval.gate import (
    GateDecision,
    evaluate_gate,
    should_retrieve_query,
)
from atlas_memory.cognitive.retrieval.pipeline import RetrievalPipeline
from atlas_memory.cognitive.retrieval.prefetch import PrefetchEngine
from atlas_memory.cognitive.retrieval.render import render_context_block
from atlas_memory.cognitive.retrieval.score import score_candidates

__all__ = [
    "gather_candidates",
    "score_candidates",
    "evaluate_gate",
    "should_retrieve_query",
    "GateDecision",
    "render_context_block",
    "PrefetchEngine",
    "RetrievalPipeline",
]
