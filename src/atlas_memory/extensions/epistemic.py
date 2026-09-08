from __future__ import annotations

from typing import Tuple

from atlas_memory.models import EpistemicSource, MemoryRecord


class EpistemicCalibrator:
    """
    Moduł D: Pętla Oceny Wiarygodności (Epistemic Calibration).
    
    Sprawdza źródło wiedzy:
    - USER_EXPLICIT (1.0): bezpośrednia deklaracja użytkownika (najwyższy priorytet).
    - TOOL_OUTPUT (0.85): potwierdzony wynik narzędzia / API.
    - EXTERNAL_DOC (0.75): wpis z dokumentacji.
    - AGENT_INFERENCE (0.60): dedukcja / hipoteza agenta (najłatwiej nadpisywana).
    """

    SOURCE_PRIORITIES = {
        EpistemicSource.USER_EXPLICIT: 100,
        EpistemicSource.TOOL_OUTPUT: 70,
        EpistemicSource.EXTERNAL_DOC: 50,
        EpistemicSource.AGENT_INFERENCE: 30,
    }

    SOURCE_BASE_CONFIDENCE = {
        EpistemicSource.USER_EXPLICIT: 1.0,
        EpistemicSource.TOOL_OUTPUT: 0.85,
        EpistemicSource.EXTERNAL_DOC: 0.75,
        EpistemicSource.AGENT_INFERENCE: 0.60,
    }

    def calibrate(self, record: MemoryRecord) -> MemoryRecord:
        """
        Dostosowuje bazową pewność faktu na podstawie jego proweniencji epistemicznej.
        """
        base_factor = self.SOURCE_BASE_CONFIDENCE.get(record.source_type, 0.70)
        # Efektywna pewność: iloczyn deklarowanej pewności i wagi źródła
        record.confidence = round(float(record.confidence * base_factor), 3)
        return record

    def arbitrate_conflict(
        self,
        existing: MemoryRecord,
        incoming: MemoryRecord,
    ) -> Tuple[bool, str]:
        """
        Rozstrzyga spór pomiędzy dwoma sprzecznymi faktami.
        Zwraca: (czy_nowy_fakt_wygrywa, uzasadnienie)
        """
        p_exist = self.SOURCE_PRIORITIES.get(existing.source_type, 50)
        p_in = self.SOURCE_PRIORITIES.get(incoming.source_type, 50)

        # 1. Hierarchia źródeł: wyższy priorytet zawsze unieważnia niższy
        if p_in > p_exist:
            return True, f"epistemic_override: incoming source '{incoming.source_type.value}' outranks existing '{existing.source_type.value}'"
        elif p_in < p_exist:
            return False, f"epistemic_rejected: existing source '{existing.source_type.value}' outranks incoming '{incoming.source_type.value}'"

        # 2. Gdy priorytety są równe: porównujemy znaczniki czasu
        if incoming.timestamp > existing.timestamp:
            return True, "timestamp_override: newer record with equal epistemic rank"
        elif incoming.timestamp < existing.timestamp:
            return False, "stale_rejected: older record with equal epistemic rank"

        # 3. Równe znaczniki czasu: wyższa pewność
        if incoming.confidence >= existing.confidence:
            return True, "confidence_override: higher confidence at same timestamp"
        else:
            return False, "confidence_rejected: lower confidence at same timestamp"

