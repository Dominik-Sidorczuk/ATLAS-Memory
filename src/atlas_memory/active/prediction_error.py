from __future__ import annotations

import logging
import time
from collections import deque
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
        self.error_history: deque[PredictionError] = deque(maxlen=1000)
        self._entity_successes: Dict[str, int] = {}
        self._entity_coolings: Dict[str, float] = {}
        self._last_error_time: Dict[str, float] = {}

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
            obs_str = str(observed_value).strip()
            obs_upper = obs_str.upper()
            if any(marker in obs_upper for marker in ("FAIL", "ERROR", "CRITICAL", "DRIFT", "OOD", "NOISE", "CORRUPT")):
                err = PredictionError(
                    check_id=f"adhoc_{int(time.time() * 1000)}",
                    target_entity=observed_entity,
                    predicate=observed_predicate,
                    expected_value="SUCCESS",
                    observed_value=obs_str,
                    discrepancy_score=1.0,
                    severity="CRITICAL",
                )
                self.error_history.append(err)
                return err
            return None

        obs_str = str(observed_value).strip()
        exp_str = expectation.expected_value.strip()

        # 1. Obsługa typu 'regex'
        if expectation.check_type == "regex":
            import re
            try:
                if re.search(exp_str, obs_str):
                    return None
            except re.error:
                if obs_str.lower() == exp_str.lower():
                    return None
            err = PredictionError(
                check_id=expectation.check_id,
                target_entity=observed_entity,
                predicate=observed_predicate,
                expected_value=exp_str,
                observed_value=obs_str,
                discrepancy_score=1.0,
                severity="CRITICAL",
            )
            self.error_history.append(err)
            return err

        # 2. Obsługa typu 'numeric_range'
        if expectation.check_type == "numeric_range":
            low: Optional[float] = None
            high: Optional[float] = None
            if "min" in expectation.metadata and "max" in expectation.metadata:
                try:
                    low = float(expectation.metadata["min"])
                    high = float(expectation.metadata["max"])
                except (ValueError, TypeError):
                    pass
            if low is None or high is None:
                clean_exp = exp_str.strip("[]() ")
                parts = []
                if ".." in clean_exp:
                    parts = clean_exp.split("..", 1)
                elif "," in clean_exp:
                    parts = clean_exp.split(",", 1)
                elif "-" in clean_exp and not clean_exp.startswith("-"):
                    parts = clean_exp.split("-", 1)
                if len(parts) == 2:
                    try:
                        low = float(parts[0].strip())
                        high = float(parts[1].strip())
                    except ValueError:
                        pass
            if (low is None or high is None) and expectation.tolerance is not None:
                try:
                    center = float(exp_str)
                    low = center - expectation.tolerance
                    high = center + expectation.tolerance
                except ValueError:
                    pass

            if low is not None and high is not None:
                try:
                    obs_num = float(obs_str)
                    if low <= obs_num <= high:
                        return None
                    diff = (low - obs_num) if obs_num < low else (obs_num - high)
                    span = max(1.0, high - low)
                    disc_score = round(min(1.0, max(0.0001, diff / span)), 4)
                    severity = "CRITICAL" if disc_score >= 0.50 else ("MODERATE" if disc_score >= 0.10 else "LOW")
                    return PredictionError(
                        check_id=expectation.check_id,
                        target_entity=observed_entity,
                        predicate=observed_predicate,
                        expected_value=exp_str,
                        observed_value=obs_str,
                        discrepancy_score=disc_score,
                        severity=severity,
                    )
                except ValueError:
                    logger.debug("Non-numeric values in numeric_range check; falling back to exact match.")

        # 3. Sprawdzenie numeryczne z ciągłą skalą błędu (Continuous Severity)
        try:
            obs_num = float(obs_str)
            exp_num = float(exp_str)
            diff = abs(obs_num - exp_num)

            # 3a. Dokładne dopasowanie numeryczne - zerowy błąd
            if diff < 1e-9:
                return None

            # 3b. Gdy określono tolerancję i różnica się w niej mieści -> brak błędu
            if expectation.tolerance is not None and diff <= expectation.tolerance:
                return None

            # Ciągła miara rozbieżności relatywnej względem oczekiwanej wartości
            eff_diff = (diff - expectation.tolerance) if expectation.tolerance is not None else diff
            denominator = max(1.0, abs(exp_num) + (expectation.tolerance or 0.0))
            disc_score = round(min(1.0, max(0.0001, eff_diff / denominator)), 4)

            # Wielopoziomowa ocena powagi (Continuous Severity Thresholds)
            if expectation.tolerance is not None and diff > (2.0 * expectation.tolerance):
                severity = "CRITICAL"
            elif disc_score < 0.10:
                severity = "LOW"
            elif disc_score < 0.50:
                severity = "MODERATE"
            else:
                severity = "CRITICAL"

            return PredictionError(
                check_id=expectation.check_id,
                target_entity=observed_entity,
                predicate=observed_predicate,
                expected_value=exp_str,
                observed_value=obs_str,
                discrepancy_score=disc_score,
                severity=severity,
            )
        except ValueError:
            logger.debug("Non-numeric values in prediction check; using exact string match.")

        # 4. Sprawdzenie dokładne dla wartości kategorycznych/stringowych (Exact Match)
        if obs_str.lower() != exp_str.lower():
            err = PredictionError(
                check_id=expectation.check_id,
                target_entity=observed_entity,
                predicate=observed_predicate,
                expected_value=exp_str,
                observed_value=obs_str,
                discrepancy_score=1.0,
                severity="CRITICAL",
            )
            self.error_history.append(err)
            return err

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
            # Zabezpieczenie: jeśli zapis do modelu świata zawiódł, nie nadpisujemy bazy
            key = f"{observed_entity}::{observed_predicate}"
            if error.world_model_updated:
                if key in self._expectations:
                    self._expectations[key].expected_value = str(observed_value)
                    self._expectations[key].last_checked = time.time()
            elif mnemosyne_triple_add_fn is None:
                # Ochrona przed jednorazowym szumem przy braku potwierdzenia bazodanowego:
                # Wymagana ponowna spójna obserwacja
                pending = getattr(self, "_noise_filter", {})
                if pending.get(key) == str(observed_value):
                    if key in self._expectations:
                        self._expectations[key].expected_value = str(observed_value)
                        self._expectations[key].last_checked = time.time()
                else:
                    if not hasattr(self, "_noise_filter"):
                        self._noise_filter = {}
                    self._noise_filter[key] = str(observed_value)
        return error

    def record_success(self, entity: str, predicate: str = "") -> None:
        """Rejestruje bezbłędną obserwację sensoryczną (sukces predykcji), schładzając temperaturę poznawczą."""
        ent_lower = str(entity).strip().lower()
        if not ent_lower:
            return
        self._entity_successes[ent_lower] = self._entity_successes.get(ent_lower, 0) + 1
        curr_cooling = self._entity_coolings.get(ent_lower, 1.0)
        self._entity_coolings[ent_lower] = max(0.01, curr_cooling * 0.70)

        # Propagacja na stripped entity_id (np. rule:deploy_pipeline -> deploy_pipeline)
        clean_id = ent_lower.split(":")[-1]
        if clean_id != ent_lower:
            self._entity_successes[clean_id] = self._entity_successes.get(clean_id, 0) + 1
            self._entity_coolings[clean_id] = max(0.01, self._entity_coolings.get(clean_id, 1.0) * 0.70)

    def cool_entity(self, entity: str, factor: float = 0.5) -> float:
        """Wymusza schłodzenie temperatury dla encji o zadany czynnik relaksacji."""
        ent_lower = str(entity).strip().lower()
        if not ent_lower:
            return 0.0
        val = max(0.01, self._entity_coolings.get(ent_lower, 1.0) * float(factor))
        self._entity_coolings[ent_lower] = val
        clean_id = ent_lower.split(":")[-1]
        if clean_id != ent_lower:
            self._entity_coolings[clean_id] = val
        return val

    def is_entity_critical(self, entity: str) -> bool:
        """Sprawdza, czy encja notuje błędy CRITICAL, z uwzględnieniem histerezy chłodzenia i powrotu do krystalizacji."""
        ent_lower = str(entity).strip().lower()
        if not ent_lower:
            return False

        clean_id = ent_lower.split(":")[-1]

        # Jeśli encja odnotowała serię sukcesów po błędzie lub temperatura spadła poniżej progu krystalizacji
        succ = max(self._entity_successes.get(ent_lower, 0), self._entity_successes.get(clean_id, 0))
        if succ >= 3:
            return False

        temp = min(self.get_entity_temperature(ent_lower), self.get_entity_temperature(clean_id))
        if temp < 0.15:
            return False

        for err in reversed(self.error_history):
            target = str(err.target_entity).strip().lower()
            if (target == ent_lower or ent_lower in target or target in ent_lower or target == clean_id) and (
                err.severity.upper() == "CRITICAL" or err.discrepancy_score >= 0.70
            ):
                return True
        return False

    def get_entity_temperature(self, entity: str) -> float:
        """Oblicza lokalną temperaturę poznawczą T_cogn dla encji z uwzględnieniem czynnika schłodzenia."""
        ent_lower = str(entity).strip().lower()
        if not ent_lower:
            return 0.0
        clean_id = ent_lower.split(":")[-1]
        scores = []
        for err in self.error_history:
            target = str(err.target_entity).strip().lower()
            if target == ent_lower or ent_lower in target or target in ent_lower or target == clean_id:
                scores.append(float(err.discrepancy_score) ** 2)
        if not scores:
            return 0.0
        base_temp = float(sum(scores) / len(scores))
        cooling_mult = min(self._entity_coolings.get(ent_lower, 1.0), self._entity_coolings.get(clean_id, 1.0))
        succ = max(self._entity_successes.get(ent_lower, 0), self._entity_successes.get(clean_id, 0))
        if succ > 0:
            cooling_mult *= (0.75 ** succ)
        return float(max(0.0, base_temp * cooling_mult))


_GLOBAL_ACTIVE_SENSING: Optional[ActiveSensingEngine] = None


def get_active_sensing_engine() -> ActiveSensingEngine:
    """Zwraca globalny singleton instancji ActiveSensingEngine."""
    global _GLOBAL_ACTIVE_SENSING
    if _GLOBAL_ACTIVE_SENSING is None:
        _GLOBAL_ACTIVE_SENSING = ActiveSensingEngine()
    return _GLOBAL_ACTIVE_SENSING
