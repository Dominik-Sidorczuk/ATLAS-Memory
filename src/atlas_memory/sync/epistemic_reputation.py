"""Epistemic reputation scoring for distributed peers and Byzantine fault mitigation."""

from __future__ import annotations

from typing import Dict, List, Tuple

from pydantic import BaseModel, Field


class EpistemicReputation(BaseModel):
    """Scoring zaufania i reputacji epistemicznej peerów w rozproszonej pamięci."""

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
        """Rejestruje wynik walidacji wiedzy lub operacji przesłanej przez peera."""
        validated_count, total_count = self.history.get(peer_id, (0, 0))
        new_validated = validated_count + (1 if validated else 0)
        new_total = total_count + 1
        self.history[peer_id] = (new_validated, new_total)

    def score(self, peer_id: str) -> float:
        """Oblicza wynik reputacji R(p) = validated / total. Zwraca 0.0 w przypadku braku danych."""
        if peer_id not in self.history:
            return 0.0
        validated_count, total_count = self.history[peer_id]
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
