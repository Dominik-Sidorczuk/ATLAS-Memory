
"""Canonical Layer-0 Domain Models & Epistemic Primitives for ATLAS."""
from __future__ import annotations

import enum
import math
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator

HERMES_DEFAULT_SESSION: str = "hermes_default"
HERMES_GLOBAL_SESSION: str = "global"


# V152 (FILAR I): sanitizacja tekstu epistemicznego na granicy typu — chroni
# spaCy/OpenIE przed surogatami i znakami sterującymi C0/C1 (historia: token.c
# SIGSEGV przy surrogate + surowej prozie). Deterministyczna, "pure bytes-op"
# (regex) bez dotykania spaCy. Polskie litery (U+00A0+) i emoji-spoza-BMP
# (🚀=U+1F680) przechodzą; legalne pary surogatowe (single-codepoint emoji)
# nie są dotykane.
_TEXT_SANITIZE_RE = re.compile(
    "["
    "\ud800-\udfff"                            # lone surrogates
    "\x00-\x08\x0b\x0c\x0e-\x1f\x7f"          # C0 (zostawia \t\n\r)
    "\x80-\x9f"                                # C1
    "\u200b\u200c\u200d\ufeff"                 # Cf zero-width
    "]"
)


def sanitize_epistemic_text(value: Any) -> str:
    """V152 (FILAR I): usuwa niebezpieczne znaki z tekstu na granicy typu."""
    if not isinstance(value, str):
        return value
    return _TEXT_SANITIZE_RE.sub("", value)


# V151.1 (FILAR I): rozłączne typy aktów mowy. Kontrakt: shadow_reconcile NIGDY
# nie ekstrahuje ConversationalTurn (pytania/metadyskurs) jako encji; tylko
# EpistemicAssertion podlega canonicalizacji i konfliktu arbiterowi.
class SpeechAct(str, enum.Enum):
    CONVERSATIONAL_TURN = "conversational_turn"
    EPISTEMIC_ASSERTION = "epistemic_assertion"


# V151.1 (FILAR I): promujące akcje arbitrażowe. ConflictArbiter wg audytu
# ConflictArbiter.actions ma dziś {'accept','warn_conflict','supersede','reject'};
# żadna z tych akcji nie jest 'promote'/'accept_conflict', więc pojedyncze
# wywołanie shadow NIGDY nie ma uprawnień do wpisu state_variable w KV.
SHADOW_PROMOTION_ACTIONS: Tuple[str, ...] = ("promote", "accept_conflict")
SHADOW_PROMOTION_PENDING: str = "pending_arbitration"
SHADOW_PROMOTION_ARBITRATED: str = "arbitration_promoted"


def classify_speech_act(subject: Any, predicate: Any, object_value: Any) -> SpeechAct:
    """V151.1 (FILAR I): klasyfikacja aktu mowy trójki (surowa proza vs asercja EAV).

    Zasada typologiczna (NIE blacklistowa): meta-pytania / discourse-garbage
    subjects / interogatywy w obiekcie (trailing '?') to ConversationalTurn
    (proza — nie podlega ekstrakcji na encje). Wszystko inne to EpistemicAssertion
    (kanoniczny trójpodmiot EAV podlega normalnemu pipeline'owi shadow_workera).

    Bariera dla TEMATU encji jest DETERMINISTYCZNO-LEKSYKALNA (ten sam rdzeń
    co reguła 1 w `is_discourse_garbage_subject`: head-token ∈ QUERY_META_STOPWORDS
    ∪ stem(head) ∈ QUERY_META_STOPSTEMS ∪ _DISCOURSE_GARBAGE_EXTRA_STEMS), DOPEŁNIANA
    rozpoznaniem POSTACI gramatycznej head-tokenu (POS spaCy, zero-torch):
    head POS == ADP (przyimek, np. 'Z jakiej wersji' → 'z' ADP) → ConversationalTurn —
    temat gramatycznie nie może być podmiotem encji. Pomyłka POS dla krótkich nazw
    własnych demowałaby asercję epistemiczną (utrata dowodzenia użytkownika), więc
    POS rozstrzyga TYLKO w kierunku CONVERSATIONAL_TURN i tylko dla head-tokenu.

    V152 (FILAR I, ADR): fallback deterministyczny bez POS: stem(head) == 'z'
    (dopełnienie _DISCOURSE_GARBAGE_EXTRA_STEMS o 1 stem) — dodaje bariery
    dla przyimka nawet przy spaCy shutdown/slow. Żadnych nowych list słów —
    wyłącznie istniejące prymitywy core/linguistics.
    """
    raw = str(subject or "").strip()
    if raw:
        try:
            from atlas_memory.core.linguistics import (
                _DISCOURSE_GARBAGE_EXTRA_STEMS,
                QUERY_META_STOPSTEMS,
                QUERY_META_STOPWORDS,
                _discourse_stem,
                get_spacy_nlp,
            )

            head = raw.replace("-", "_").replace(" ", "_").split("_")[0].lower()
            stem = _discourse_stem(head)
            if head in QUERY_META_STOPWORDS or stem in QUERY_META_STOPSTEMS or (stem and stem in _DISCOURSE_GARBAGE_EXTRA_STEMS):
                return SpeechAct.CONVERSATIONAL_TURN
            # V152 (FILAR I): POS-typowe rozpoznanie head-tokenu tematu.
            # Adp-osadzony temat (przyimek otwierający frazę przysłówkową)
            # jest gramatycznie nie-podmiotem → ConversationalTurn. Rozstrzyga
            # wyłącznie w kierunku odrzucenia; asercji NIE demuje.
            try:
                nlp = get_spacy_nlp("pl", fast_lemma_only=False)
                if nlp is not None:
                    doc = nlp(head)
                    if doc and len(doc) and doc[0].pos_ == "ADP":
                        return SpeechAct.CONVERSATIONAL_TURN
            except Exception:
                pass
        except Exception:
            pass
    obj_text = str(object_value or "")
    pred_text = str(predicate or "")
    if "?" in obj_text or "?" in pred_text:
        return SpeechAct.CONVERSATIONAL_TURN
    return SpeechAct.EPISTEMIC_ASSERTION


def state_variable_authority(record: "MemoryRecord") -> bool:
    """V151.1 (FILAR I): jedyna procedura wyznaczania uprawnień state_variable.

    NIEZMIENNIK: rekord z proweniencji SHADOW_INFERENCE nigdy nie zapisuje się
    w KV jako state_variable z jednego wywołania shadow — arbitraż (ConflictArbiter)
    może PODNIEŚĆ tylko jawną akcją promującą (SHADOW_PROMOTION_ACTIONS) z
    uzasadnieniem; pojedynczy shadow commit nigdy nie ma uprawnień. Innymi
    słowy: zapis state_variable wymaga (a) rzetelnej proweniencji (niet-shadow)
    ALBO (b) shadow_promotion == arbitration_promoted.
    """
    try:
        if not bool(getattr(record, "is_state_variable", False)):
            return False
        st_val = getattr(record.source_type, "value", str(getattr(record, "source_type", "") or ""))
        if st_val == EpistemicSource.SHADOW_INFERENCE.value:
            promo = str((getattr(record, "metadata", None) or {}).get("shadow_promotion", ""))
            return promo == SHADOW_PROMOTION_ARBITRATED
        return True
    except Exception:
        return False


def clamp_conf(val: Any) -> float:
    """Klampuje wartość zaufania/pewności (confidence) do przedziału [0.0, 1.0]."""
    try:
        raw = float(val if val is not None else 1.0)
        if math.isnan(raw):
            raw = 1.0
    except (ValueError, TypeError):
        raw = 1.0
    return max(0.0, min(1.0, raw))


class EpistemicSource(str, enum.Enum):
    """Klasyfikacja źródeł wiedzy pod kątem wiarygodności (Epistemic Provenance)."""

    USER_EXPLICIT = "user_explicit"
    TOOL_OUTPUT = "tool_output"
    AGENT_INFERENCE = "agent_inference"
    # V151 (Wektor I, H1): fakty z cieniowej ekstrakcji LLM mają własną klasę
    # proweniencji. Shadow Worker NIGDY nie dowodzi woli użytkownika — dawniej
    # fallback w orchestrator.shadow_reconcile (linia 134) podawał się za
    # USER_EXPLICIT, fałszując dowodzenie epistemiczne najwyższej rangi.
    SHADOW_INFERENCE = "shadow_inference"
    EXTERNAL_DOC = "external_doc"
    SENSOR_OBSERVATION = "sensor_observation"
    CRDT_SYNC = "crdt_sync"
    HOT_BUFFER = "hot_buffer"


class MemoryRecord(BaseModel):
    """Rozszerzony rekord pamięci symboliczno-relacyjnej i wektorowej."""

    subject: str = Field(..., description="Podmiot relacji / identyfikator encji")
    predicate: str = Field(..., description="Predykat relacji (np. 'has_role', 'depends_on', 'state_is')")
    object: str = Field(..., description="Obiekt relacji lub wartość tekstowa/liczbowa")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Współczynnik pewności faktu")
    timestamp: float = Field(default_factory=time.time, description="Znacznik czasu utworzenia rekordu (UNIX epoch)")
    is_state_variable: bool = Field(default=False, description="Czy rekord jest zmienną stanu Source of Truth (L2 KV)")
    # V151.1 (FILAR I): rozłączne typy aktu mowy (patrz SpeechAct). Domyślnie
    # EPISTEMIC_ASSERTION (wszystkie istniejące rekordy to asercje EAV);
    # ConversationTurn = surowa proza (pytania/metadyskurs), NIE podlega
    # ekstrakcji jako encja — shadow_reconcile taki rekord odrzuca przed commitem.
    speech_act: SpeechAct = Field(default=SpeechAct.EPISTEMIC_ASSERTION, description="Akt mowy: asercja EAV vs surowa proza konwersacyjna")
    source_type: Union[EpistemicSource, str] = Field(default=EpistemicSource.USER_EXPLICIT, description="Proweniencja epistemiczna")
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Ważność faktu I(f) (0.1=pogoda, 1.0=hasła/architektura)")
    access_count: int = Field(default=0, description="Licznik odpytań faktu N_access")
    last_accessed: float = Field(default_factory=time.time, description="Znacznik czasu ostatniego odpytania")
    canonical_entity_id: Optional[str] = Field(default=None, description="Skanonizowany identyfikator encji po disambiguacji")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Dodatkowe metadane (np. ID sesji, tagi)")
    vector: Optional[List[float]] = Field(default=None, description="Opcjonalny wektor cech (embedding)")
    source_session_id: Optional[str] = Field(default=None, description="Identyfikator sesji / epizodu źródłowego")

    # V152 (FILAR I): granica typu normalizuje surowe wejście — lone surrogates,
    # C0/C1 control (poza \t\n\r) i Cf zero-width są usuwane z podmiotu/
    # predykatu/obiektu ZANIM cokolwiek trafi do KV/wektora/spaCy (SIGSEGV
    # token.c). Legalne emoji spoza BMP (U+1F680 itd.) przechodzą nietknięte.
    @field_validator("subject", "predicate", "object", mode="before")
    @classmethod
    def _v152_strip_unsafe_text(cls, value: Any) -> Any:
        return sanitize_epistemic_text(value)

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
    """Reprezentacja stanu ukrytego świata L1 (JEPA Latent World State s_t)."""

    vector: List[float] = Field(..., description="Ciągły wektor stanu ukrytego")
    dimension: int = Field(..., description="Wymiarowość przestrzeni ukrytej")
    step_index: int = Field(default=0, description="Krok symulacji / sekwencji")
    timestamp: float = Field(default_factory=time.time, description="Znacznik czasu stanu")
    energy_score: float = Field(default=0.0, description="Wynik energetyczny JEPA / pewność predykcji")
    context_tags: List[str] = Field(default_factory=list, description="Tagi kontekstowe sesji")


class ActionPlan(BaseModel):
    """Akcja w przestrzeni L1 planowania Systemu 2 / JEPA Predictor."""

    name: str = Field(..., description="Nazwa akcji / narzędzia")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parametry akcji")
    latent_projection: Optional[List[float]] = Field(default=None, description="Rzutowanie akcji na wektor a_t")
    action_encoding: Optional[str] = Field(default=None, description="Proweniencja wektora akcji: onnx_embed | hash_projection")


class PredictedTransition(BaseModel):
    """Symulacja przejścia s_{t+1} = P(s_t, a_t) w przestrzeni ukrytej."""

    previous_state: LatentState
    action: ActionPlan
    predicted_state: LatentState
    simulated_reward: float = 0.0
    uncertainty: float = 0.0


class ConflictReport(BaseModel):
    """Raport z wykrycia i rozwiązania sprzeczności w wiedzy (L3 Sleep Phase)."""

    conflict_type: str = Field(..., description="Typ konfliktu: 'state_value_override', 'epistemic_override', 'stale_update'")
    existing_record: MemoryRecord
    incoming_record: MemoryRecord
    resolution_strategy: str = Field(..., description="Zastosowana strategia (np. 'timestamp_override', 'epistemic_priority', 'reject_stale')")
    resolved_record: Optional[MemoryRecord] = None
    resolved_at: float = Field(default_factory=time.time)


class ConsolidationStats(BaseModel):
    """Statystyki nocnej fazy konsolidacji (Sleep-Cycle Consolidation)."""

    records_analyzed: int = 0
    duplicates_merged: int = 0
    conflicts_resolved: int = 0
    orphaned_nodes_gc: int = 0
    decayed_pruned_records: int = 0
    pruned_stale_facts: int = 0
    compacted_episodes: int = 0
    baked_procedures: int = 0
    proposed_skills: List[str] = Field(default_factory=list, description="Lista proponowanych do kompilacji procedur SOP")
    duration_ms: float = 0.0
    status: str = "ok"
    consolidated_records: int = 0
    facts_superseded: int = 0
    declarative_pairs_found: int = 0
    declarative_clusters: int = 0
    declarative_roots_blocked: int = 0
    declarative_orthogonal_slots_blocked: int = 0
    declarative_edges_wired: int = 0
    declarative_gc_pruned: int = 0
    anneal_status: str = "skipped"
    anneal_target: str = "Root"
    anneal_iterations: int = 0
    anneal_updated_edges: int = 0
    anneal_converged: bool = False
    anneal_duration_ms: float = 0.0


class HotBufferEntry(BaseModel):
    """Item stored in L1 working memory hot-KV buffer for immediate continuation."""

    key: str = Field(..., description="Unique memory key")
    value: Any = Field(..., description="Stored value payload")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence rating")
    source_type: str = Field(default="user_explicit", description="Epistemic source provenance")
    timestamp: float = Field(default_factory=time.time, description="Epoch timestamp of insertion")
    is_superseded: bool = Field(default=False, description="True if invalidated or superseded")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Item metadata")

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            val = getattr(self, key)
            return val if val is not None else default
        return self.metadata.get(key, default)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.metadata[key]



CRITICAL_TARGET_KEYWORDS: tuple[str, ...] = (
    "database", "production", "auth", "security", "payment",
    "kernel", "root", "credentials", "token", "secret",
    "firewall", "migration", "schema", "cluster", "quorum",
)

WHITELISTED_DOMAIN_TERMS = frozenset({
    "token", "tokens", "token_budget", "tokenizer", "max_tokens", "shared_secret",
})

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
SENSITIVE_KEY_PATTERN = re.compile(
    r"(?:\b|_)(api[-_\s]?key|password|client[-_\s]?secret|app[-_\s]?secret|bearer[-_\s]?token|secret[-_\s]?token|secret[-_\s]?key)(?:\b|_)",
    re.IGNORECASE,
)
SECRET_ASSIGN_PATTERN = re.compile(
    r"\bsecret\s*(?:\b|_)?\s*[:=]\s*\S",
    re.IGNORECASE,
)


def is_pii_entity(text: str) -> bool:
    if not text:
        return False
    t_clean = text.strip().lower()
    if t_clean in WHITELISTED_DOMAIN_TERMS:
        return False
    if t_clean == "secret":
        return True
    return bool(
        EMAIL_PATTERN.search(text)
        or SENSITIVE_KEY_PATTERN.search(text)
        or SECRET_ASSIGN_PATTERN.search(text)
    )


class _CounterMetric:
    def __init__(self, name: str, on_inc: Any = None) -> None:
        self.name = name
        self.value: int = 0
        self._on_inc = on_inc
        self._lock = threading.Lock()

    def inc(self, amount: int = 1) -> int:
        with self._lock:
            self.value += int(amount)
            val = self.value
        if self._on_inc is not None:
            self._on_inc(self.name, int(amount))
        return val


class _MetricsRegistry:
    def __init__(self) -> None:
        self._counters: Dict[str, _CounterMetric] = {}
        self.on_inc_hook: Any = None
        self._lock = threading.Lock()

    def counter(self, name: str) -> _CounterMetric:
        with self._lock:
            c = self._counters.get(name)
            if c is None:
                c = _CounterMetric(name, on_inc=lambda n, a: self.on_inc_hook(n, a) if self.on_inc_hook else None)
                self._counters[name] = c
            return c

    def get(self, name: str) -> int:
        with self._lock:
            c = self._counters.get(name)
            return c.value if c is not None else 0

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()


metrics = _MetricsRegistry()


class RetrievalRecord(BaseModel):
    """High-veracity memory record surfaced in cognitive retrieval/prefetch."""
    subject: str = Field(..., description="Entity or key subject")
    predicate: str = Field(default="state", description="Relationship predicate or slot")
    object: str = Field(..., description="Content, value or entity target")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Epistemic confidence level")
    source_type: str = Field(default="user_explicit", description="Epistemic provenance")
    score: float = Field(default=1.0, description="Ranking veracity score")
    is_state_variable: bool = Field(default=False, description="True if record is verified KV state variable")
    is_superseded: bool = Field(default=False, description="True if record has been superseded")
    timestamp: float = Field(default_factory=time.time, description="Creation or update epoch timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata tags")


class RetrievalContext(BaseModel):
    """Prefetch or recall context payload supplied to the agent."""
    query: str = Field(..., description="User query or context prompt")
    session_id: Optional[str] = Field(default=None, description="Hermes session identifier")
    records: List[RetrievalRecord] = Field(default_factory=list, description="Retrieved records sorted by veracity")
    count: int = Field(default=0, description="Number of records returned")
    skipped: bool = Field(default=False, description="True if retrieval gate skipped execution")
    latency_ms: float = Field(default=0.0, description="Execution duration in milliseconds")
    cache_hit: bool = Field(default=False, description="True if served from cognitive prefetch cache")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Session and gate telemetry")

__all__ = [
    "RetrievalRecord",
    "RetrievalContext",
    "EpistemicSource",
    "MemoryRecord",
    "LatentState",
    "ActionPlan",
    "PredictedTransition",
    "ConflictReport",
    "ConsolidationStats",
    "HotBufferEntry",
    "HERMES_DEFAULT_SESSION",
    "HERMES_GLOBAL_SESSION",
    "CRITICAL_TARGET_KEYWORDS",
    "clamp_conf",
    "is_pii_entity",
    "metrics",
]


class StandardProcedure(BaseModel):
    """V12/V15 Standard Operating Procedure (SOP) distilled from multi-turn trajectories."""
    procedure_id: str
    name: str
    trigger_pattern: str
    steps: List[str] = Field(default_factory=list)
    tool_sequence: List[str] = Field(default_factory=list)
    confidence: float = 0.85
    frequency: int = 1
    success_rate: float = 1.0
    source_sessions: List[str] = Field(default_factory=list)
