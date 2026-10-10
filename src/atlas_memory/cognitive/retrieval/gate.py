"""V81 Gate: SKIP tylko gdy max(score) < prog kalibrowany bez bypassu."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from atlas_memory.cognitive.calibration_engine import is_symbolic_or_identifier
from atlas_memory.core.linguistics import LanguageAssetManager
from atlas_memory.l1_working.hot_buffer import PERSISTENT_KEY_PREFIXES

_phatic = LanguageAssetManager.get_instance().get_phatic_tokens("pl")
_TRIVIAL = tuple(_phatic.get("trivial", []))
_CONT = tuple(_phatic.get("continuation", []))


def is_continuation_query(query: str) -> bool:
    ql = query.lower().strip()
    return any(ql == x or ql.startswith(x + " ") for x in _CONT)


def is_trivial_prompt(query: str) -> bool:
    clean = re.sub(r"[^\w\s]", "", query.lower()).strip()
    if any(clean == x or clean.startswith(x + " ") for x in _TRIVIAL):
        return True
    words = clean.split()
    return bool(words and any(x in _TRIVIAL for x in words[:2]) and len(words) <= 4)


_INTENT_KWS = LanguageAssetManager.get_instance().get_intent_keywords("pl")


def should_retrieve_query(
    query: str,
    explicit_entities: Optional[List[str]] = None,
    canonicalizer: Optional[Any] = None,
) -> tuple[bool, list, str]:
    if is_trivial_prompt(query):
        return False, [], "conversational_turn_no_entity"

    detected: List[str] = []
    if explicit_entities:
        if canonicalizer is not None and hasattr(canonicalizer, "canonicalize_list"):
            detected.extend(canonicalizer.canonicalize_list(explicit_entities))
        else:
            detected.extend(explicit_entities)

    # Colon-separated entity keys
    for m in re.findall(r"\b[A-Za-z0-9_-]+:[A-Za-z0-9_:-]+\b", query):
        if m not in detected:
            detected.append(m)

    # Alphanumeric nodes (e.g. node1, port8080, p1, p2)
    for m in re.findall(r"\b[A-Za-z]+[0-9]+\b", query):
        if m not in detected:
            detected.append(m)

    msg_clean = query.lower().strip()
    for kw in _INTENT_KWS:
        if kw in msg_clean and kw not in detected:
            detected.append(kw)

    return (True, detected, f"entity_match: {detected}") if detected else (True, [], "semantic_turn")


@dataclass
class GateDecision:
    should_retrieve: bool
    reason: str
    threshold: float
    threshold_selection_reason: str


_SYSTEM_NOTIFICATION_PREFIXES = (
    "[important: background process",
    "[system:",
    "<system_message>",
    "<system-reminder>",
    "[message] timestamp=",
)


# V114 (F-114): mierzalny próg kontynuacji. Dotychczas ścieżka continuation
# w `evaluate_gate` zwracała `GateDecision(., threshold=0.0)` i prefetch
# serwował KAŻDEGO gorącego kandydata (1244/1500 tokenów szumu przy
# gorącej sesji — defekt v93). Raport kalibracyjny v93: mediana score
# odrzuceń `below_threshold` w gałęzi continuation leżała w paśmie
# [0.10, 0.18]; ustawienie progu 0.18 odcina ten szum przy zachowaniu
# gorących dowodów (score >= 0.18 = medianę prefetch-utility dla
# poprzednich cykli snu). Decyzja: modułowa stała z komentarzem
# uzasadniającym, żeby każdy kolejny przegląd threshold-ów widział
# genezę. ADR: docs/adr/ADR-114-continuation-threshold.md.
CONTINUATION_MIN_SCORE: float = 0.18


def evaluate_gate(query: str, scored_candidates: List[Dict[str, Any]], calibration_engine: Optional[Any] = None) -> GateDecision:
    q = query.strip()
    if q.lower().startswith(_SYSTEM_NOTIFICATION_PREFIXES):
        return GateDecision(False, "system_notification", 0.0, "policy_skip")
    if not q:
        hot = [
            c for c in scored_candidates
            if c.get("source") == "hot_buffer"
            and not any(str(c.get("subject", "")).lower().startswith(p) for p in PERSISTENT_KEY_PREFIXES)
        ]
        return GateDecision(bool(hot), "empty_query_hot" if hot else "empty_query", 0.0, "no_candidate_fallback")
    if is_trivial_prompt(q):
        return GateDecision(False, "casual_greeting", 0.0, "no_candidate_fallback")
    if is_continuation_query(q):
        # V114 (F-114): continuation ścieżka musi mierzyć, nie przepuszczać.
        # Filtrujemy gorących kandydatów po score (>= CONTINUATION_MIN_SCORE)
        # i zwracamy threshold=CONTINUATION_MIN_SCORE, żeby prefetch dalej
        # stosował ten sam próg bez rozjazdu z semantyką bramki (wpp. prefetch
        # ciął po starej zasadzie i serwował 1244/1500 tokenów szumu).
        # Rozróżniamy dwa stany "brak gorących":
        #  - nigdy nie było dowodu hot_buffer → 'continuation_prompt' (bez zmian,
        #    regresja V81/V111: czysty skip pompy bez kandydatów),
        #  - dowód był, ale cały poniżej progu → 'continuation_below_threshold'
        #    (mierzalne odrzucenie szumu, nie ciche przepuszczenie).
        hot_all = [
            c for c in scored_candidates
            if c.get("source") == "hot_buffer"
            and not any(str(c.get("subject", "")).lower().startswith(p) for p in PERSISTENT_KEY_PREFIXES)
        ]
        if not hot_all:
            return GateDecision(False, "continuation_prompt", 0.0, "no_candidate_fallback")
        hot = [c for c in hot_all if float(c.get("score", 0.0)) >= CONTINUATION_MIN_SCORE]
        if not hot:
            return GateDecision(
                False,
                "continuation_below_threshold",
                CONTINUATION_MIN_SCORE,
                "no_candidate_fallback",
            )
        return GateDecision(True, "continuation", CONTINUATION_MIN_SCORE, "no_candidate_fallback")

    is_sym = is_symbolic_or_identifier(q)
    thresh = float(calibration_engine.get_threshold(query_type="symbolic" if is_sym else "prose", default=0.50 if is_sym else 0.25)) if (calibration_engine and getattr(calibration_engine, "enabled", False) and getattr(calibration_engine, "calibrator", None)) else (0.50 if is_sym else 0.25)
    reason = "learned" if (calibration_engine and getattr(calibration_engine, "enabled", False) and getattr(calibration_engine, "calibrator", None)) else "no_candidate_fallback"
    has_strong_graph_hit = any(
        float(c.get("graph_score", 0.0) or (c.get("scores") or {}).get("graph", 0.0) or 0.0) >= 0.70
        for c in scored_candidates
    ) if scored_candidates else False
    effective_thresh = min(thresh, 0.20) if has_strong_graph_hit else thresh

    if scored_candidates:
        max_score = max((float(c.get("score", 0.0)) for c in scored_candidates), default=0.0)
        if max_score < effective_thresh:
            return GateDecision(False, "below_threshold", effective_thresh, reason)
    else:
        return GateDecision(False, "empty_pool", thresh, reason)
    decision_reason = "graph_hit" if (has_strong_graph_hit and max_score < thresh) else "accepted"
    return GateDecision(True, decision_reason, effective_thresh, reason)
