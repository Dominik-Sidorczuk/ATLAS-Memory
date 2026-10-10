"""Dense cosine scoring strategy with Platt scaling normalization."""
from __future__ import annotations

import math
from typing import Any, Dict, List

from atlas_memory.cognitive.retrieval.strategies.base import BaseScoringStrategy

DBSF_DENSE_MU: float = 0.50
DBSF_DENSE_BETA: float = 12.0


class DenseCosineStrategy(BaseScoringStrategy):
    """Vector cosine scoring strategy with ModernBERT/PolDense-68M Platt scaling."""

    def __init__(self, mu: float = DBSF_DENSE_MU, beta: float = DBSF_DENSE_BETA) -> None:
        self.mu = mu
        self.beta = beta

    @staticmethod
    def normalize_dense(vs: float, mu: float = DBSF_DENSE_MU, beta: float = DBSF_DENSE_BETA) -> float:
        """Normalized Platt Scaling sigmoid with zero-shift (C1 smooth on [0, 1])."""
        if vs <= 0.0:
            return 0.0
        clamped_vs = min(1.0, float(vs))

        def _sigma(z: float) -> float:
            if z >= 40.0:
                return 1.0
            if z <= -40.0:
                return 0.0
            return 1.0 / (1.0 + math.exp(-z))

        s_val = _sigma(beta * (clamped_vs - mu))
        s_zero = _sigma(-beta * mu)
        denom = 1.0 - s_zero
        if denom <= 0.0:
            return 0.0
        norm_val = (s_val - s_zero) / denom
        return max(0.0, min(1.0, norm_val))

    def score(self, query: str, candidates: List[Any], **kwargs: Any) -> Dict[str, float]:
        """Extracts and normalizes vector scores for all candidates."""
        results: Dict[str, float] = {}
        for c in candidates:
            cand_key = c.get("key") or c.get("subject") or id(c)
            raw_vs = 0.0
            if hasattr(c, "raw_scores") and "dense_score" in c.raw_scores:
                raw_vs = float(c.raw_scores["dense_score"])
            elif hasattr(c, "dense_score") and c.dense_score:
                raw_vs = float(c.dense_score)
            elif "dense_score" in c:
                raw_vs = float(c["dense_score"])
            elif "vector_score" in c:
                raw_vs = float(c["vector_score"])
            elif "similarity" in c:
                raw_vs = float(c["similarity"])

            norm_vs = self.normalize_dense(raw_vs, mu=self.mu, beta=self.beta)
            results[str(cand_key)] = norm_vs

            if isinstance(c, dict):
                c["_vec_score"] = raw_vs
                c["dense_score"] = round(norm_vs, 4)
            elif hasattr(c, "dense_score"):
                c.dense_score = round(norm_vs, 4)
                if hasattr(c, "features"):
                    c.features.dense_score = round(norm_vs, 4)
                    c.features.vector_score = raw_vs

        return results
