from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)



class PredictionCheck(BaseModel):
    """Definicja oczekiwanego stanu świata (Prior Expectation)."""
    check_id: str
    target_entity: str
    expected_predicate: str
    expected_value: str
    tolerance: Optional[float] = None  # Tolerancja dla wartości numerycznych
    check_type: str = Field(default="exact", description="'exact', 'numeric_range', 'regex'")
    last_checked: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PredictionError(BaseModel):
    """Zdarzenie błędu predykcji (Free Energy / Surprise Signal)."""
    check_id: str
    target_entity: str
    predicate: str
    expected_value: str
    observed_value: str
    discrepancy_score: float = Field(ge=0.0, le=1.0)
    severity: str = Field(default="WARNING", description="'INFO', 'WARNING', 'CRITICAL'")
    timestamp: float = Field(default_factory=time.time)
    world_model_updated: bool = False


class ActiveSensingEngine:
    """
    Faza C: Externalized Active Sensing (Predictive Coding dla Modeli Black-Box).
    
    Generuje deterministyczne oczekiwania (Prior Expectations) i oblicza błąd predykcji (Prediction Error)
    względem fizycznych obserwacji środowiska. W przypadku wykrycia rozbieżności dokonuje
    natychmiastowej aktualizacji modelu świata (mnemosyne_triple_add z supersede=True) z 0 zapytaniami do LLM.
    """

    def __init__(self):
        self._expectations: Dict[str, PredictionCheck] = {}
        self.error_history: List[PredictionError] = []

    def register_expectation(self, check: PredictionCheck) -> None:
        """Rejestruje regułę oczekiwanego stanu."""
        key = f"{check.target_entity}::{check.expected_predicate}"
        self._expectations[key] = check

    def expectation_checks(self) -> List[PredictionCheck]:
        """Zwraca listę wszystkich zarejestrowanych oczekiwań."""
        return list(self._expectations.values())

    def detect_discrepancy(
        self,
        observed_entity: str,
        observed_predicate: str,
        observed_value: Any,
    ) -> Optional[PredictionError]:
        """
        Porównuje obserwowany stan z oczekiwanym:
        Zwraca PredictionError w przypadku anomalii lub None w przypadku zgodności.
        """
        key = f"{observed_entity}::{observed_predicate}"
        expectation = self._expectations.get(key)
        if expectation is None:
            return None

        obs_str = str(observed_value).strip()
        exp_str = expectation.expected_value.strip()

        # 1. Sprawdzenie numeryczne z tolerancją
        if expectation.tolerance is not None:
            try:
                obs_num = float(obs_str)
                exp_num = float(exp_str)
                diff = abs(obs_num - exp_num)
                if diff > expectation.tolerance:
                    denominator = max(1e-5, abs(exp_num) + expectation.tolerance)
                    disc_score = min(1.0, diff / denominator)
                    severity = "CRITICAL" if diff > (2.0 * expectation.tolerance) else "WARNING"
                    return PredictionError(
                        check_id=expectation.check_id,
                        target_entity=observed_entity,
                        predicate=observed_predicate,
                        expected_value=exp_str,
                        observed_value=obs_str,
                        discrepancy_score=round(disc_score, 3),
                        severity=severity,
                    )
                else:
                    return None  # W granicach tolerancji
            except ValueError:
                logger.debug("Non-numeric values in prediction check; using exact string match.")


        # 2. Sprawdzenie dokładne (Exact Match)
        if obs_str.lower() != exp_str.lower():
            return PredictionError(
                check_id=expectation.check_id,
                target_entity=observed_entity,
                predicate=observed_predicate,
                expected_value=exp_str,
                observed_value=obs_str,
                discrepancy_score=1.0,
                severity="CRITICAL",
            )

        return None

    async def process_observation(
        self,
        observed_entity: str,
        observed_predicate: str,
        observed_value: Any,
        mnemosyne_triple_add_fn: Optional[Callable] = None,
    ) -> Optional[PredictionError]:
        """
        Przetwarza obserwację telemetryczną ze środowiska:
        Jeśli wykryto rozbieżność, dokonuje automatycznej aktualizacji modelu świata (supersede=True).
        """
        error = self.detect_discrepancy(observed_entity, observed_predicate, observed_value)
        if error is not None:
            self.error_history.append(error)

            # Natychmiastowa aktualizacja modelu świata bez zapytania do LLM
            if mnemosyne_triple_add_fn is not None:
                try:
                    await mnemosyne_triple_add_fn(
                        subject=observed_entity,
                        predicate=observed_predicate,
                        object_=str(observed_value),
                        confidence=1.0,
                        source="active_sensing_tool",
                        supersede=True,
                    )
                    error.world_model_updated = True
                except Exception:
                    error.world_model_updated = False

            # Aktualizacja samego oczekiwania (adaptacja do nowej rzeczywistości)
            key = f"{observed_entity}::{observed_predicate}"
            if key in self._expectations:
                self._expectations[key].expected_value = str(observed_value)
                self._expectations[key].last_checked = time.time()

        return error
