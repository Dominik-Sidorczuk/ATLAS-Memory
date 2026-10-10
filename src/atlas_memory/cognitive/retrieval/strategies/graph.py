"""Graph synergy scoring strategy evaluating HippoRAG 2 / PPR associative connections."""
from __future__ import annotations

import re
from typing import Any, Dict, List

from atlas_memory.cognitive.retrieval.strategies.base import BaseScoringStrategy

GENERIC_KEY_PARTS = {
    "hermes", "config", "fact", "rule", "preference", "constraint", "system", "service",
    "user_md", "memory_md", "stated_memory", "working_fact", "episodic_event",
    "canonical_spec", "user", "agent", "default",
}


class GraphSynergyStrategy(BaseScoringStrategy):
    """Evaluates graph synergy scores from HippoRAG 2 / Personalized PageRank."""

    @staticmethod
    def is_ungrounded_scoped_entity(
        s_cand_low: str,
        q_stems: List[str],
        matched_stems: int,
        ls: float,
        is_trusted: bool,
        gs: float,
    ) -> bool:
        """Flags candidates with scoped entities that lack lexical/graph grounding."""
        subj_entities = [p for p in re.findall(r"[a-z0-9]+", s_cand_low) if len(p) >= 3 and p not in GENERIC_KEY_PARTS]
        return (
            ":" in s_cand_low
            and bool(subj_entities)
            and bool(q_stems)
            and matched_stems == 0
            and ls <= 0.0
            and not is_trusted
            and gs <= 0.0
        )

    def score(self, query: str, candidates: List[Any], **kwargs: Any) -> Dict[str, float]:
        """Extracts and evaluates graph synergy scores across candidates."""
        results: Dict[str, float] = {}
        for c in candidates:
            cand_key = c.get("key") if hasattr(c, "get") else getattr(c, "key", id(c))
            gs = 0.0
            if hasattr(c, "raw_scores") and "graph_score" in c.raw_scores:
                gs = float(c.raw_scores["graph_score"])
            elif hasattr(c, "graph_score") and c.graph_score:
                gs = float(c.graph_score)
            elif hasattr(c, "get") and c.get("graph_score") is not None:
                gs = float(c.get("graph_score"))

            is_graph_cand = bool(
                gs > 0.0
                or (hasattr(c, "get") and (c.get("graph_hit") or c.get("source") == "graph"))
                or getattr(c, "source", None) == "graph"
            )
            if is_graph_cand and gs <= 0.0:
                gs = 0.50

            results[str(cand_key)] = gs
            if isinstance(c, dict):
                c["_graph_score"] = gs
                c["graph_score"] = round(gs, 4)
                c["graph_hit"] = is_graph_cand
            else:
                if hasattr(c, "graph_score"):
                    c.graph_score = round(gs, 4)
                if hasattr(c, "features"):
                    c.features.graph_score = round(gs, 4)
                    c.features.is_graph_candidate = is_graph_cand

        return results
