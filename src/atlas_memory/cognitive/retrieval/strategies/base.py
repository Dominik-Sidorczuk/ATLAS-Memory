"""Base interface for retrieval scoring strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List


class BaseScoringStrategy(ABC):
    """Abstract base class for retrieval scoring and fusion strategies."""

    @abstractmethod
    def score(self, query: str, candidates: List[Any], **kwargs: Any) -> Any:
        """Scores or enriches candidates with respect to the query."""
        raise NotImplementedError
