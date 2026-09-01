from __future__ import annotations

import enum
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EpistemicSource(str, enum.Enum):
    """
    Klasyfikacja źródeł wiedzy pod kątem wiarygodności (Epistemic Provenance).
    """
    USER_EXPLICIT = "user_explicit"      # Bezpośrednia deklaracja użytkownika (najwyższa wiarygodność, 1.0)
    TOOL_OUTPUT = "tool_output"          # Zweryfikowany wynik wywołania narzędzia/API (0.85)
    AGENT_INFERENCE = "agent_inference"  # Wewnętrzna dedukcja / hipoteza agenta (0.60)
    EXTERNAL_DOC = "external_doc"        # Informacja ze statycznej dokumentacji (0.75)


class MemoryRecord(BaseModel):
    """
    Rozszerzony rekord pamięci symboliczno-relacyjnej i wektorowej.
    """
    subject: str = Field(..., description="Podmiot relacji / identyfikator encji")
    predicate: str = Field(..., description="Predykat relacji (np. 'has_role', 'depends_on', 'state_is')")
    object: str = Field(..., description="Obiekt relacji lub wartość tekstowa/liczbowa")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Współczynnik pewności faktu")
    timestamp: float = Field(default_factory=time.time, description="Znacznik czasu utworzenia rekordu (UNIX epoch)")
    is_state_variable: bool = Field(default=False, description="Czy rekord jest zmienną stanu Source of Truth (L2 KV)")
    source_type: EpistemicSource = Field(default=EpistemicSource.USER_EXPLICIT, description="Proweniencja epistemiczna")
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Ważność faktu I(f) (0.1=pogoda, 1.0=hasła/architektura)")
    access_count: int = Field(default=0, description="Licznik odpytań faktu N_access")
    last_accessed: float = Field(default_factory=time.time, description="Znacznik czasu ostatniego odpytania")
    canonical_entity_id: Optional[str] = Field(default=None, description="Skanonizowany identyfikator encji po disambiguacji")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Dodatkowe metadane (np. ID sesji, tagi)")
    vector: Optional[List[float]] = Field(default=None, description="Opcjonalny wektor cech (embedding)")
    source_session_id: Optional[str] = Field(default=None, description="Identyfikator sesji / epizodu źródłowego")

    @property
    def effective_subject(self) -> str:
        """Zwraca skanonizowany identyfikator encji jeśli jest dostępny, w przeciwnym razie subject."""
        return self.canonical_entity_id or self.subject

    @property
    def triple_key(self) -> str:
        """Klucz trójki do identyfikacji relacji."""
        return f"{self.effective_subject}:{self.predicate}"

    @property
    def full_key(self) -> str:
        """Klucz pełnej trójki encji."""
        return f"{self.effective_subject}:{self.predicate}:{self.object}"


class LatentState(BaseModel):
    """
    Reprezentacja stanu ukrytego świata L1 (JEPA Latent World State s_t).
    """
    vector: List[float] = Field(..., description="Ciągły wektor stanu ukrytego")
    dimension: int = Field(..., description="Wymiarowość przestrzeni ukrytej")
    step_index: int = Field(default=0, description="Krok symulacji / sekwencji")
    timestamp: float = Field(default_factory=time.time, description="Znacznik czasu stanu")
    energy_score: float = Field(default=0.0, description="Wynik energetyczny JEPA / pewność predykcji")
    context_tags: List[str] = Field(default_factory=list, description="Tagi kontekstowe sesji")


class ActionPlan(BaseModel):
    """
    Akcja w przestrzeni L1 planowania Systemu 2 / JEPA Predictor.
    """
    name: str = Field(..., description="Nazwa akcji / narzędzia")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parametry akcji")
    latent_projection: Optional[List[float]] = Field(default=None, description="Rzutowanie akcji na wektor a_t")


class PredictedTransition(BaseModel):
    """
    Symulacja przejścia s_{t+1} = P(s_t, a_t) w przestrzeni ukrytej.
    """
    previous_state: LatentState
    action: ActionPlan
    predicted_state: LatentState
    simulated_reward: float = 0.0
    uncertainty: float = 0.0


class ConflictReport(BaseModel):
    """
    Raport z wykrycia i rozwiązania sprzeczności w wiedzy (L3 Sleep Phase).
    """
    conflict_type: str = Field(..., description="Typ konfliktu: 'state_value_override', 'epistemic_override', 'stale_update'")
    existing_record: MemoryRecord
    incoming_record: MemoryRecord
    resolution_strategy: str = Field(..., description="Zastosowana strategia (np. 'timestamp_override', 'epistemic_priority', 'reject_stale')")
    resolved_record: Optional[MemoryRecord] = None
    resolved_at: float = Field(default_factory=time.time)


class ConsolidationStats(BaseModel):
    """
    Statystyki nocnej fazy konsolidacji (Sleep-Cycle Consolidation).
    """
    records_analyzed: int = 0
    duplicates_merged: int = 0
    conflicts_resolved: int = 0
    orphaned_nodes_gc: int = 0
    decayed_pruned_records: int = 0
    compacted_episodes: int = 0
    baked_procedures: int = 0
    proposed_skills: List[str] = Field(default_factory=list, description="Lista proponowanych do kompilacji procedur SOP")
    duration_ms: float = 0.0


class RecallResult(BaseModel):
    """
    Wynik równoległego odpytania 4 warstw pamięci (L0-L2).
    """
    semantic_context: List[Dict[str, Any]] = Field(default_factory=list, description="Dopasowania z Episodic Vector Store")
    graph_topology: Dict[str, Any] = Field(default_factory=dict, description="Sąsiedztwo 1- i 2-go stopnia z Symbolic Graph")
    verified_state: Dict[str, Any] = Field(default_factory=dict, description="Niezmienne wartości ze stanu KV / SQLite")
    l0_ttt_state: Optional[Dict[str, Any]] = Field(default=None, description="Stan wewnętrzny plastyczności TTT")
    l1_latent_state: Optional[LatentState] = Field(default=None, description="Aktywny wektor stanu JEPA")
    retrieval_latency_ms: float = 0.0
