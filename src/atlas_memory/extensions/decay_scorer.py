from __future__ import annotations

import math
import time
from typing import Optional

from atlas_memory.models import MemoryRecord


class SalienceDecayEngine:
    """
    Moduł A: Silnik Wagowania i Zapominania (Salience & Recency Decay).
    
    Wylicza dynamiczną ważność faktu:
        S(f) = w_sim * Sim(q, f) + w_imp * I(f) + w_rec * e^(-lambda * (t - t0)) + w_freq * log(1 + N_access)
    """

    def __init__(
        self,
        weight_similarity: float = 0.40,
        weight_importance: float = 0.30,
        weight_recency: float = 0.20,
        weight_frequency: float = 0.10,
        decay_lambda: float = 0.0001,  # Współczynnik rozpadu (ok. 2h half-life dla testów, konfigurowalny)
        prune_threshold: float = 0.15,
    ):
        self.w_sim = weight_similarity
        self.w_imp = weight_importance
        self.w_rec = weight_recency
        self.w_freq = weight_frequency
        self.decay_lambda = decay_lambda
        self.prune_threshold = prune_threshold
        self._log_norm = math.log1p(50)

    def calculate_salience(
        self,
        record: MemoryRecord,
        similarity_score: float = 0.0,
        current_time: Optional[float] = None,
    ) -> float:
        """
        Oblicza łączną wagę ważności faktu w danym momencie czasu.
        """
        now = current_time if current_time is not None else time.time()
        time_delta_seconds = max(0.0, now - record.timestamp)

        # 1. Składnik semantyczny Sim(q, f)
        sim_part = self.w_sim * max(0.0, min(1.0, similarity_score))

        # 2. Składnik wewnętrznej ważności I(f)
        imp_part = self.w_imp * max(0.0, min(1.0, record.importance_score))

        # 3. Składnik świeżości (Recency Decay) e^(-lambda * dt)
        rec_decay = math.exp(-self.decay_lambda * time_delta_seconds)
        rec_part = self.w_rec * rec_decay

        # 4. Składnik częstotliwości log(1 + N_access) - znormalizowany
        freq_normalized = min(1.0, math.log1p(record.access_count) / self._log_norm)
        freq_part = self.w_freq * freq_normalized

        score = sim_part + imp_part + rec_part + freq_part
        return float(score)


    def record_access(self, record: MemoryRecord, access_time: Optional[float] = None) -> None:
        """Rejestruje odpytanie faktu, zwiększając N_access i odświeżając last_accessed."""
        record.access_count += 1
        record.last_accessed = access_time if access_time is not None else time.time()

    def should_prune(
        self,
        record: MemoryRecord,
        current_time: Optional[float] = None,
    ) -> bool:
        """
        Decyduje czy fakt uległ zapomnieniu i powinien zostać usunięty w fazie Sleep Cycle.
        Zmienne stanu (is_state_variable) oraz fakty o maksymalnej ważności (I=1.0) nigdy nie są usuwane.
        """
        if record.is_state_variable or record.importance_score >= 0.95:
            return False

        # Wycena bazowa bez dopasowania zapytania (Sim=0)
        base_salience = self.calculate_salience(record, similarity_score=0.0, current_time=current_time)
        return base_salience < self.prune_threshold

