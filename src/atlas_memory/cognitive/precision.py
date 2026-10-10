"""ATLAS Cognitive Precision: Bayesian Precision Tensor (Active Inference & Friston Precision).

Implements dynamic sensory and channel precision weighting:
Pi_{channel}(t+1) = alpha * Pi_{channel}(t) + (1 - alpha) / (Var(epsilon) + sigma_0^2)
Lambda = Pi_{channel} * confidence
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

from atlas_memory.core.models import EpistemicSource, clamp_conf

logger = logging.getLogger(__name__)


class BayesianPrecisionTensor:
    """Bayesowski Tensor Precyzji Informacyjnej Fristona (Active Inference).

    Zastępuje statyczne drabiny wag dynamicznym tensorem precyzji:
    Pi_{channel}(t+1) = alpha * Pi_{channel}(t) + (1 - alpha) / (Var(epsilon) + sigma_0^2)

    Arbitracja konfliktu i tunelowanie asocjacyjne opierają się na potencjale precyzji:
    Lambda = Pi_{channel} * confidence
    """

    DEFAULT_PRECISIONS: Dict[str, float] = {
        EpistemicSource.USER_EXPLICIT.value: 2.5,
        EpistemicSource.TOOL_OUTPUT.value: 1.5,
        EpistemicSource.EXTERNAL_DOC.value: 1.2,
        EpistemicSource.AGENT_INFERENCE.value: 1.0,
    }

    def __init__(
        self,
        alpha: float = 0.85,
        sigma_0_sq: float = 0.04,
        precisions: Optional[Dict[str, float]] = None,
    ) -> None:
        self.alpha = float(alpha)
        self.sigma_0_sq = float(sigma_0_sq)
        self._precisions: Dict[str, float] = dict(self.DEFAULT_PRECISIONS)
        if precisions:
            self._precisions.update(precisions)

    @staticmethod
    def _normalize_channel(channel: EpistemicSource | str) -> str:
        if hasattr(channel, "value"):
            return str(channel.value)
        return str(channel)

    def get_precision(self, channel: EpistemicSource | str) -> float:
        ch = self._normalize_channel(channel)
        return self._precisions.get(ch, 1.0)

    def record_sensory_observation(
        self,
        channel: EpistemicSource | str,
        error_magnitude: float,
    ) -> float:
        """Dynamiczna aktualizacja tensora precyzji po obserwacji sensorycznej / falsyfikacji."""
        ch = self._normalize_channel(channel)
        curr_pi = self.get_precision(ch)
        err = max(0.0, float(error_magnitude))
        var_eps = err ** 2
        sensory_precision = 1.0 / (var_eps + self.sigma_0_sq)
        new_pi = self.alpha * curr_pi + (1.0 - self.alpha) * sensory_precision
        new_pi = max(0.05, min(10.0, new_pi))
        self._precisions[ch] = new_pi
        return new_pi

    def compute_lambda(self, channel: EpistemicSource | str, confidence: float) -> float:
        """Oblicza iloczyn precyzji kanału i pewności: Lambda = Pi * confidence."""
        pi = self.get_precision(channel)
        return float(pi * clamp_conf(confidence))

