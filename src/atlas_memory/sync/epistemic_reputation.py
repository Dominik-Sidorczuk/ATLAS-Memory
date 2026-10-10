"""Epistemic reputation scoring for distributed peers and Byzantine fault mitigation (V83 Refactor).

Bounded memory consumption (max 100 peers) and bidirectional clock drift validation.
"""

from __future__ import annotations

import time
from typing import ClassVar, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class EpistemicReputation(BaseModel):
    """Scoring zaufania i reputacji epistemicznej peerów w rozproszonej pamięci."""

    MAX_TRACKED_PEERS: ClassVar[int] = 100
    MAX_CLOCK_DRIFT_SECONDS: ClassVar[float] = 120.0

    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimalny próg reputacji do uznania peera za zaufanego",
    )
    history: Dict[str, Tuple[int, int]] = Field(
        default_factory=dict,
        description="Historia walidacji peerów: peer_id -> (validated_count, total_count)",
    )

    def record(self, peer_id: str, validated: bool) -> None:
        """Rejestruje wynik walidacji wiedzy lub operacji przesłanej przez peera z limitem pamięci."""
        pid = str(peer_id)
        if len(self.history) >= self.MAX_TRACKED_PEERS and pid not in self.history:
            # Eksmisja najstarszego peera w celu ochrony przed wyciekiem RAM
            oldest_peer = next(iter(self.history))
            del self.history[oldest_peer]

        validated_count, total_count = self.history.get(pid, (0, 0))
        new_validated = validated_count + (1 if validated else 0)
        new_total = total_count + 1
        self.history[pid] = (new_validated, new_total)

    def validate_timestamp(
        self,
        msg_timestamp: float,
        local_timestamp: Optional[float] = None,
        max_drift: Optional[float] = None,
    ) -> bool:
        """Weryfikuje dwukierunkowy dryf zegara: |t_msg - t_local| <= max_drift."""
        now = time.time() if local_timestamp is None else local_timestamp
        drift_limit = self.MAX_CLOCK_DRIFT_SECONDS if max_drift is None else max_drift
        return abs(float(msg_timestamp) - float(now)) <= drift_limit

    def score(self, peer_id: str) -> float:
        """Oblicza wynik reputacji R(p) = validated / total. Zwraca 0.0 w przypadku braku danych."""
        pid = str(peer_id)
        if pid not in self.history:
            return 0.0
        validated_count, total_count = self.history[pid]
        if total_count == 0:
            return 0.0
        return float(validated_count) / float(total_count)

    def is_trusted(self, peer_id: str) -> bool:
        """Sprawdza, czy wynik reputacji peera spełnia próg zaufania (R(p) >= threshold)."""
        return self.score(peer_id) >= self.threshold

    def disconnect_untrusted(self, peer_ids: List[str]) -> List[str]:
        """Zwraca listę peerów, którzy powinni zostać odłączeni z powodu niskiej reputacji (R < threshold)."""
        untrusted = []
        for peer_id in peer_ids:
            if not self.is_trusted(peer_id):
                untrusted.append(peer_id)
        return untrusted
