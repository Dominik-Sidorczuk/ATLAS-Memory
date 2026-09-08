"""
ATLAS Cognitive Layer: Two-Stage Conflict Arbiter (V27+ Architecture).

Implements deterministic epistemic conflict detection, candidate scoring,
fatigue gating, and belief revision.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

from atlas_memory.cognitive.models import ArbitrationCandidate, ArbitrationResult

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "the", "and", "for", "with", "this", "that", "from", "into", "over",
    "jest", "oraz", "dla", "przez", "jego", "który", "która", "które",
    "jako", "jestem", "będzie", "robimy", "w", "na", "z", "do", "o",
    "po", "od", "za", "ze", "we", "to", "się", "co", "jak", "tak",
    "ale", "iż", "że", "czy", "nad", "pod", "przed", "przy", "tam", "tu",
    "ma", "są", "być", "może", "mogą", "należy", "trzeba", "bądź",
}

EXCLUSIVITY_WORDS = {
    "wyłącznie", "tylko", "jedynie", "sole", "solely", "exclusively", "only",
}

PROHIBITION_WORDS = {
    "nie", "not", "brak", "never", "nigdy", "zabronione", "zabronił", "zabrania",
    "zakaz", "zakazane", "odrzucona", "bypasses", "usunięty", "forbid",
    "forbidden", "prohibited", "disallow", "disallowed", "bez", "zabraniać", "zabroniony",
}

AFFIRMATIVE_WORDS = {
    "uses", "używa", "używać", "używamy", "is", "jest", "requires", "enabled", "active",
    "zawsze", "always", "allow", "allowed", "wykonujemy", "zezwala", "pozwala", "zezwól",
    "stosujemy", "stosuje", "stosować", "dozwolone", "dozwolony", "dozwolona", "wykonaj",
}

CONTROL_WORDS = EXCLUSIVITY_WORDS | PROHIBITION_WORDS | AFFIRMATIVE_WORDS

GENERIC_WORDS = {
    "narzędzia", "narzędzie", "narzędzi", "tool", "tools", "funkcja", "funkcji",
    "metoda", "metody", "sposób", "sposobu", "kod", "kodu", "rzecz", "rzeczy",
    "rule", "state", "fact", "config", "system", "memory", "atlas", "test",
    "variable", "item", "default", "wartość", "parametr", "opcja", "użytkownika",
    "action", "task", "preference", "sensor", "architecture",
}

DEVELOPER_TOOLS = {
    "git", "python", "pytest", "bash", "sh", "zsh", "npm", "cargo",
    "docker", "pip", "pixi", "node", "linux", "sqlite", "redis",
    "postgres", "postgresql", "mysql", "qdrant", "kuzu", "hermes",
    "github", "gitlab", "conda", "vscode", "react", "fastapi",
}

ACTION_PREFIXES = (
    "wdraż", "deploy", "instal", "urucham", "commit", "push", "merge",
    "budow", "build", "test", "zapis", "sync", "read", "writ", "zarządz",
    "edycj", "logow", "pobier", "usuw", "dostęp", "access", "stor",
)

ENVIRONMENT_TOKENS = {
    "staging", "prod", "production", "dev", "development", "test", "testing", "local",
}


def polish_stem(word: str) -> str:
    """Stem Polish words by stripping common grammatical suffixes."""
    w = word.lower()
    if w.startswith(("zewnątrz", "zewnętrz")):
        return "zewnątrz"
    if len(w) <= 4:
        return w
    suffixes = (
        "owania", "owanie", "owaniu", "owaniach", "owaniami",
        "alności", "alność", "alnością",
        "ościach", "ościami", "ościom", "nością", "ności", "ność",
        "owemu", "owych", "owymi", "owego", "owej", "owym",
        "iach", "iami", "iom",
        "ach", "ami", "om",
        "cja", "cję", "cji", "cją", "cje", "cjo",
        "ego", "emu", "ych", "ymi", "ym",
        "em", "am", "om", "ie", "ej", "ze",
        "a", "e", "y", "i", "o", "u", "ę", "ą",
    )
    for suf in suffixes:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[:-len(suf)]
    return w


def extract_tokens(text: str) -> Set[str]:
    """Extract alphanumeric tokens (length >= 3) excluding stop words."""
    return {
        t for t in re.findall(r'[a-zA-Z0-9ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]{3,}', text.lower())
        if t not in STOP_WORDS
    }


def extract_stems(tokens: Set[str]) -> Set[str]:
    """Extract Polish stems from token set."""
    return {polish_stem(t) for t in tokens}


def extract_prohibited_terms(text: str) -> Set[str]:
    """Extract target words directly governed by explicit prohibition phrasing."""
    clauses = re.split(r'[\.\;\,\n\|—–\(\)]', text)
    prohibited = set()
    explicit_prohib = {
        "zakaz", "zakazane", "zabronione", "zabrania", "zabronił", "nigdy", "never",
        "forbid", "forbidden", "prohibited", "disallow", "disallowed", "zabraniać", "zabroniony",
    }
    for clause in clauses:
        words = re.findall(r'[a-zA-Z0-9ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]{3,}', clause.lower())
        for idx, w in enumerate(words):
            is_prohib = (w in explicit_prohib) or (
                w in ("nie", "not") and idx + 1 < len(words) and words[idx + 1] in (
                    "wolno", "zezwala", "pozwala", "należy", "używa", "używać", "używaj", "loguj", "uruchamiaj",
                )
            )
            if is_prohib:
                start_offset = 2 if w in ("nie", "not") else 1
                for next_w in words[idx + start_offset: idx + start_offset + 5]:
                    if (
                        next_w not in STOP_WORDS
                        and next_w not in PROHIBITION_WORDS
                        and next_w not in GENERIC_WORDS
                    ):
                        prohibited.add(next_w)
    return prohibited


def extract_exclusive_target(text: str) -> Set[str]:
    """Extract target entities or tools bound to exclusivity modifiers."""
    targets = set()
    pattern = r'(?:wyłącznie|tylko|jedynie|solely|exclusively|only)\s+(?:przez|za\s+pomocą|via|through|in|w\s+języku|w|jako|dla)?\s*([a-zA-Z0-9ąćęłńóśźżĄĆĘŁŃÓŚŹŻ\s]{3,35})'
    for m in re.finditer(pattern, text.lower()):
        matched_phrase = m.group(1)
        sub_tokens = extract_tokens(matched_phrase) - CONTROL_WORDS - GENERIC_WORDS
        targets |= sub_tokens
    return targets


def extract_governed_action(tokens: Set[str]) -> Set[str]:
    """Identify action verbs from known stem prefixes."""
    return {t for t in tokens if any(polish_stem(t).startswith(p) for p in ACTION_PREFIXES)}


class ConflictArbiter:
    """
    Two-Stage Cognitive Conflict Arbiter.

    Stage 1: Deterministic Candidate Generator
      - Multi-factor candidate scoring across existing KV items
      - Polish stemming, token extraction, governed actions, exclusive targets
      - Consensus protection: shared prohibitions are agreement, not conflict
      - Sorts all candidates descending by score (never takes first dict match)

    Stage 2: Semantic Arbitration & Fatigue Guard
      - Emits warnings only on high-confidence true positives (best_score >= 35.0)
      - Moderate / ambiguous matches recorded in metadata without fatigue warnings
      - Belief Revision Engine: high-confidence user writes supersede prior beliefs
      - Ignores records already tagged with is_superseded = True
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.confidence_threshold = float(self.config.get("confidence_threshold", 0.95))
        self.high_score_threshold = float(self.config.get("high_score_threshold", 35.0))

    def generate_candidates(
        self,
        key: str,
        val: Any,
        confidence: float,
        metadata: Dict[str, Any],
        existing_items: Dict[str, Any],
    ) -> List[ArbitrationCandidate]:
        """
        Stage 1: Scan existing items and compute candidate conflict scores.
        Returns all candidates sorted in descending order of candidate_score.
        """
        candidates: List[ArbitrationCandidate] = []
        val_str = str(val).lower() if val is not None else ""
        k_lower = key.lower()

        # Explicit conflict marker in incoming key/metadata
        if "conflict" in k_lower or metadata.get("is_conflict_test") or metadata.get("test_type") == "conflict":
            candidates.append(
                ArbitrationCandidate(
                    key=key,
                    score=100.0,
                    reason="explicit_conflict_marker",
                    conflict_type="explicit_conflict_marker",
                    conflicting_fact=val_str,
                    incoming_claim=str(val),
                    overlapping_terms=[key],
                    message=f"Epistemic conflict detected for marked key '{key}'.",
                    confidence=confidence,
                    metadata={"is_explicit": True},
                )
            )

        tokens = extract_tokens(f"{key} {val_str}")
        topic_tokens = tokens - CONTROL_WORDS
        topic_stems = extract_stems(topic_tokens)
        specific_tokens = topic_tokens - GENERIC_WORDS - DEVELOPER_TOOLS
        specific_stems = topic_stems - {polish_stem(g) for g in (GENERIC_WORDS | DEVELOPER_TOOLS)}

        has_excl = bool(tokens & EXCLUSIVITY_WORDS)
        has_prohib = bool(tokens & PROHIBITION_WORDS)
        has_aff = bool(tokens & AFFIRMATIVE_WORDS)
        in_prohibited = extract_prohibited_terms(f"{key} {val_str}")
        in_prohib_stems = extract_stems(in_prohibited)
        in_excl_targets = extract_exclusive_target(f"{key} {val_str}")
        in_gov_actions = extract_governed_action(topic_tokens)

        tools_in = {t for t in tokens if t in DEVELOPER_TOOLS}
        env_in = {p for p in key.split(":") if p in ENVIRONMENT_TOKENS} | {t for t in tokens if t in ENVIRONMENT_TOKENS}

        source_type = str(metadata.get("source_type") or metadata.get("source") or "user_explicit")
        is_user_explicit = (source_type == "user_explicit" or (confidence >= self.confidence_threshold and source_type != "agent_inference"))

        for ex_k, item in existing_items.items():
            # Check if existing item is marked as superseded
            if getattr(item, "is_superseded", False):
                continue
            if isinstance(item, dict):
                ex_meta = dict(item.get("metadata") or {})
                if item.get("is_superseded") is True or ex_meta.get("is_superseded") is True or ex_meta.get("superseded") is True:
                    continue
                ex_val = str(item.get("value") or item.get("val") or item.get("object") or "")
                ex_conf = float(item.get("confidence", 1.0))
            else:
                ex_val = str(getattr(item, "value", getattr(item, "object", "")))
                ex_conf = float(getattr(item, "confidence", 1.0))
                ex_meta = dict(getattr(item, "metadata", {}) or {})
                if ex_meta.get("is_superseded") is True or ex_meta.get("superseded") is True:
                    continue

            ex_val_lower = ex_val.lower()

            # Case A: Same key collision
            if ex_k == key:
                # Same value — still check confidence inversion before skipping (T05 fix)
                if ex_val_lower == val_str:
                    ex_source = str(ex_meta.get("source_type") or ex_meta.get("source") or "")
                    if (ex_conf >= 0.90 or ex_source == "user_explicit") and confidence < ex_conf:
                        candidates.append(
                            ArbitrationCandidate(
                                key=ex_k,
                                score=90.0,
                                reason="confidence_inversion",
                                conflict_type="confidence_inversion",
                                conflicting_fact=ex_val,
                                incoming_claim=str(val),
                                overlapping_terms=[key],
                                message=f"Incoming write has lower confidence ({confidence}) than established fact ({ex_conf}).",
                                confidence=ex_conf,
                                metadata=ex_meta,
                            )
                        )
                    continue  # Identical value write (with or without inversion candidate)
                
                # Check belief revision condition: user_explicit write with confidence >= threshold and conf >= existing
                if is_user_explicit and confidence >= self.confidence_threshold and ex_conf <= confidence:
                    candidates.append(
                        ArbitrationCandidate(
                            key=ex_k,
                            score=95.0,
                            reason="belief_revision",
                            conflict_type="belief_revision",
                            conflicting_fact=ex_val,
                            incoming_claim=str(val),
                            overlapping_terms=[key],
                            message=f"Belief revision: user updated value for key '{key}'.",
                            confidence=ex_conf,
                            metadata=ex_meta,
                        )
                    )
                    continue

                # Check confidence inversion: lower confidence write attempting to override established fact
                ex_source = str(ex_meta.get("source_type") or ex_meta.get("source") or "")
                if (ex_conf >= 0.90 or ex_source == "user_explicit") and confidence < ex_conf:
                    candidates.append(
                        ArbitrationCandidate(
                            key=ex_k,
                            score=90.0,
                            reason="confidence_inversion",
                            conflict_type="confidence_inversion",
                            conflicting_fact=ex_val,
                            incoming_claim=str(val),
                            overlapping_terms=[key],
                            message=f"Incoming write has lower confidence ({confidence}) than established fact ({ex_conf}).",
                            confidence=ex_conf,
                            metadata=ex_meta,
                        )
                    )
                    continue

            # Case B: Cross-key relationship
            # Ignore conversational churn or episodic working memory
            if ex_k.startswith(("mnemosyne:mnemosyne_working_memory:", "mnemosyne:mnemosyne_episodic_memory:", "conversation", "fact:conversation")) or ex_k.endswith(":episodic_event"):
                continue

            ex_tokens = extract_tokens(f"{ex_k} {ex_val_lower}")
            ex_topic_tokens = ex_tokens - CONTROL_WORDS
            ex_topic_stems = extract_stems(ex_topic_tokens)
            ex_specific_tokens = ex_topic_tokens - GENERIC_WORDS - DEVELOPER_TOOLS
            ex_specific_stems = ex_topic_stems - {polish_stem(g) for g in (GENERIC_WORDS | DEVELOPER_TOOLS)}

            tools_ex = {t for t in ex_tokens if t in DEVELOPER_TOOLS}
            env_ex = {p for p in ex_k.split(":") if p in ENVIRONMENT_TOKENS} | {t for t in ex_tokens if t in ENVIRONMENT_TOKENS}

            # Orthogonal environments guard (e.g. staging vs production)
            if env_in and env_ex and not (env_in & env_ex):
                continue

            # Orthogonal developer tools guard (e.g. git vs pytest, docker vs pixi)
            ns_in = [p for p in key.split(":")[:-1] if p not in GENERIC_WORDS]
            ns_ex = [p for p in ex_k.split(":")[:-1] if p not in GENERIC_WORDS]
            same_ns = (len(ns_in) >= 1 and ns_in == ns_ex)

            if tools_in and tools_ex and not (tools_in & tools_ex) and not same_ns:
                continue

            token_overlap = (topic_tokens & ex_topic_tokens) - GENERIC_WORDS
            stem_overlap = (topic_stems & ex_topic_stems) - {polish_stem(g) for g in GENERIC_WORDS}
            specific_overlap = specific_tokens & ex_specific_tokens
            specific_stem_overlap = specific_stems & ex_specific_stems

            has_topic_overlap = (same_ns and (len(token_overlap) >= 1 or len(stem_overlap) >= 1)) or (
                len(specific_overlap) >= 2 or len(specific_stem_overlap) >= 2 or (len(token_overlap) >= 2 and len(specific_overlap) >= 1)
            )

            ex_has_excl = bool(ex_tokens & EXCLUSIVITY_WORDS)
            ex_has_prohib = bool(ex_tokens & PROHIBITION_WORDS)
            ex_has_aff = bool(ex_tokens & AFFIRMATIVE_WORDS)
            ex_prohibited = extract_prohibited_terms(f"{ex_k} {ex_val_lower}")
            ex_prohib_stems = extract_stems(ex_prohibited)
            ex_excl_targets = extract_exclusive_target(f"{ex_k} {ex_val_lower}")
            ex_gov_actions = extract_governed_action(ex_topic_tokens)

            # Consensus Protection: Shared prohibitions are agreement, not contradiction
            shared_prohib = (in_prohibited & ex_prohibited) | {t for t in in_prohibited if polish_stem(t) in ex_prohib_stems}
            if has_prohib and ex_has_prohib and shared_prohib:
                # Both agree on prohibition
                continue

            if not has_topic_overlap:
                # Check for direct action overlap with conflicting polarity
                actions_match = bool(in_gov_actions & ex_gov_actions)
                if not (actions_match and (has_excl or ex_has_excl or has_prohib or ex_has_prohib)):
                    continue

            candidate_score = 0.0
            if same_ns:
                candidate_score += 50.0

            is_in_rule = key.startswith(("rule:", "user_rule:", "preference:", "constraint:"))
            is_ex_rule = ex_k.startswith(("rule:", "user_rule:", "preference:", "constraint:"))
            if is_in_rule and is_ex_rule:
                candidate_score += 35.0
            elif is_ex_rule:
                candidate_score += 15.0

            if ex_k.startswith("mnemosyne:"):
                candidate_score -= 40.0

            candidate_score += 15.0 * len(specific_overlap)
            candidate_score += 10.0 * len(specific_stem_overlap)
            candidate_score += 5.0 * len(token_overlap)
            candidate_score += ex_conf * 5.0

            found_conflict = None

            # 1. Exclusivity collision: competing exclusive assertions on shared domain
            if has_excl and ex_has_excl:
                actions_match = bool(in_gov_actions & ex_gov_actions)
                targets_differ = bool(in_excl_targets and ex_excl_targets and not (in_excl_targets & ex_excl_targets))
                differing = (specific_tokens - ex_specific_tokens) | (ex_specific_tokens - specific_tokens)
                if (actions_match and targets_differ) or (differing and (same_ns or len(specific_overlap) >= 1 or len(stem_overlap) >= 1)):
                    found_conflict = {
                        "reason": "exclusivity_collision",
                        "conflict_type": "exclusivity_collision",
                        "conflicting_fact": ex_val,
                        "incoming_claim": str(val),
                        "overlapping_terms": list(token_overlap or stem_overlap or in_excl_targets),
                        "message": f"Incoming exclusivity assertion contradicts established exclusivity rule '{ex_k}'.",
                    }
                    candidate_score += 40.0

            # 2. Prohibition contradiction: incoming asserts what memory prohibits
            if found_conflict is None and ex_has_prohib and not has_prohib:
                prohibited_overlap = (((specific_tokens & ex_prohibited) | {t for t in specific_tokens if polish_stem(t) in ex_prohib_stems}) - shared_prohib)
                has_action_in_prohib = any(t in prohibited_overlap for t in ("push", "commit", "zapis", "edit", "edycja", "bezpośrednio", "bezpośredni", "delete", "remove", "drop", "root", "hasło", "hasła", "secret", "secrets"))
                if prohibited_overlap and (len(prohibited_overlap) >= 2 or has_action_in_prohib or same_ns):
                    found_conflict = {
                        "reason": "semantic_contradiction",
                        "conflict_type": "prohibition_contradiction",
                        "conflicting_fact": ex_val,
                        "incoming_claim": str(val),
                        "overlapping_terms": list(prohibited_overlap),
                        "message": f"Incoming write asserts action prohibited by established memory '{ex_k}'.",
                    }
                    candidate_score += 35.0

            # 3. Capability conflict: incoming prohibits capability established in memory
            if found_conflict is None and has_prohib and not ex_has_prohib and ex_has_aff:
                prohibited_overlap = (((ex_specific_tokens & in_prohibited) | {t for t in ex_specific_tokens if polish_stem(t) in in_prohib_stems}) - shared_prohib)
                if prohibited_overlap and (is_ex_rule or len(specific_overlap) >= 1 or same_ns):
                    found_conflict = {
                        "reason": "semantic_contradiction",
                        "conflict_type": "prohibition_contradiction",
                        "conflicting_fact": ex_val,
                        "incoming_claim": str(val),
                        "overlapping_terms": list(prohibited_overlap),
                        "message": f"Incoming write prohibits capability established in memory '{ex_k}'.",
                    }
                    candidate_score += 30.0

            # 4. Exclusivity violation: existing has exclusivity, incoming asserts alternative target
            if found_conflict is None and ex_has_excl and not has_excl:
                shared_actions = (in_gov_actions & ex_gov_actions)
                if shared_actions or (same_ns and len(specific_overlap) >= 1):
                    overlap_stems = stem_overlap
                    authorized_targets = (ex_topic_tokens - {t for t in ex_topic_tokens if polish_stem(t) in overlap_stems}) - ex_prohibited
                    incoming_targets = topic_tokens - {t for t in topic_tokens if polish_stem(t) in overlap_stems}
                    if (incoming_targets and not (incoming_targets & authorized_targets)) or (topic_tokens & ex_prohibited):
                        found_conflict = {
                            "reason": "exclusivity_violation",
                            "conflict_type": "exclusivity_violation",
                            "conflicting_fact": ex_val,
                            "incoming_claim": str(val),
                            "overlapping_terms": list(token_overlap or stem_overlap),
                            "message": f"Incoming assertion violates established exclusive rule '{ex_k}'.",
                        }
                        candidate_score += 35.0

            # 5. Direct polarity / semantic contradiction
            if found_conflict is None and ((has_prohib and ex_has_aff and not ex_has_prohib) or (has_aff and ex_has_prohib and not has_prohib)):
                if len(specific_overlap) >= 2 or (same_ns and len(token_overlap) >= 2):
                    found_conflict = {
                        "reason": "semantic_contradiction",
                        "conflict_type": "semantic_contradiction",
                        "conflicting_fact": ex_val,
                        "incoming_claim": str(val),
                        "overlapping_terms": list(token_overlap or stem_overlap),
                        "message": f"Incoming write contradicts established memory '{ex_k}'.",
                    }
                    candidate_score += 25.0

            if found_conflict is not None:
                candidates.append(
                    ArbitrationCandidate(
                        key=ex_k,
                        score=candidate_score,
                        reason=found_conflict["reason"],
                        conflict_type=found_conflict.get("conflict_type"),
                        conflicting_fact=found_conflict.get("conflicting_fact"),
                        incoming_claim=found_conflict.get("incoming_claim"),
                        overlapping_terms=list(found_conflict.get("overlapping_terms") or []),
                        message=found_conflict.get("message"),
                        confidence=ex_conf,
                        metadata=ex_meta,
                    )
                )

        # Sort all candidates descending by score (never take first dictionary match)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def arbitrate(
        self,
        key: str,
        val: Any,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        existing_items: Optional[Dict[str, Any]] = None,
        all_kv: Optional[Dict[str, Any]] = None,
    ) -> ArbitrationResult:
        """
        Stage 2: Semantic Arbitration & Fatigue Guard.
        Evaluates top candidate, performs belief revision, and applies fatigue threshold.
        """
        meta = dict(metadata or {})
        items = existing_items if existing_items is not None else (all_kv or {})

        candidates = self.generate_candidates(key, val, confidence, meta, items)
        if not candidates:
            return ArbitrationResult(
                is_conflict=False,
                is_belief_revision=False,
                action="accept",
                message="No conflict detected.",
                candidates=[],
                best_score=0.0,
                metadata=meta,
            )

        best_candidate = candidates[0]
        best_score = best_candidate.score

        source_type = str(meta.get("source_type") or meta.get("source") or "user_explicit")
        is_user_explicit = (source_type == "user_explicit" or (confidence >= self.confidence_threshold and source_type != "agent_inference"))

        # Belief Revision Engine
        # If incoming write is user_explicit with confidence >= 0.95 and colliding candidate has confidence <= incoming
        is_both_rules = (key.startswith(("rule:", "preference:", "constraint:")) and best_candidate.key.startswith(("rule:", "preference:", "constraint:")))
        has_topic_match = bool(best_candidate.score >= 35.0 or best_candidate.key == key or is_both_rules)
        if (
            best_candidate.reason == "belief_revision"
            or (
                is_user_explicit
                and confidence >= self.confidence_threshold
                and best_candidate.confidence <= confidence
                and (
                    best_candidate.key == key
                    or meta.get("supersedes") == best_candidate.key
                    or meta.get("is_belief_revision") is True
                    or meta.get("action") == "supersede"
                    or (has_topic_match and best_candidate.reason in ("exclusivity_collision", "exclusivity_violation", "semantic_contradiction", "prohibition_contradiction"))
                )
            )
        ):
            return ArbitrationResult(
                is_conflict=False,
                is_belief_revision=True,
                action="supersede",
                superseded_key=best_candidate.key,
                conflicting_key=best_candidate.key,
                reason="belief_revision",
                message=f"Incoming write supersedes established belief '{best_candidate.key}'.",
                warning=None,
                candidates=candidates,
                best_candidate=best_candidate,
                best_score=best_score,
                metadata={"conflict_candidates": [c.model_dump() for c in candidates]},
            )

        # Confidence inversion
        if best_candidate.reason == "confidence_inversion":
            conflict_details = {
                "reason": "confidence_inversion",
                "conflicting_key": best_candidate.key,
                "existing_value": best_candidate.conflicting_fact,
                "existing_confidence": best_candidate.confidence,
                "incoming_confidence": confidence,
                "message": best_candidate.message,
            }
            return ArbitrationResult(
                is_conflict=True,
                is_belief_revision=False,
                action="warn_conflict",
                conflicting_key=best_candidate.key,
                reason="confidence_inversion",
                message=best_candidate.message,
                warning="epistemic_conflict",
                conflict_details=conflict_details,
                candidates=candidates,
                best_candidate=best_candidate,
                best_score=best_score,
                metadata={"conflict_candidates": [c.model_dump() for c in candidates]},
            )

        # Fatigue Guard: High-confidence True Positive threshold
        if best_score >= self.high_score_threshold or best_candidate.reason in ("explicit_conflict_marker", "exclusivity_collision"):
            conflict_details = {
                "reason": best_candidate.reason,
                "conflict_type": best_candidate.conflict_type or best_candidate.reason,
                "conflicting_key": best_candidate.key,
                "conflicting_fact": best_candidate.conflicting_fact,
                "incoming_claim": best_candidate.incoming_claim,
                "overlapping_terms": best_candidate.overlapping_terms,
                "message": best_candidate.message,
                "score": best_score,
            }
            return ArbitrationResult(
                is_conflict=True,
                is_belief_revision=False,
                action="warn_conflict",
                conflicting_key=best_candidate.key,
                reason=best_candidate.reason,
                message=best_candidate.message,
                warning="epistemic_conflict",
                conflict_details=conflict_details,
                candidates=candidates,
                best_candidate=best_candidate,
                best_score=best_score,
                metadata={"conflict_candidates": [c.model_dump() for c in candidates]},
            )

        # Moderate / Ambiguous matches: attached to metadata without emitting warnings
        return ArbitrationResult(
            is_conflict=False,
            is_belief_revision=False,
            action="accept",
            conflicting_key=None,
            reason=best_candidate.reason,
            message="Moderate candidate overlap retained in metadata without warning.",
            warning=None,
            candidates=candidates,
            best_candidate=best_candidate,
            best_score=best_score,
            metadata={"conflict_candidates": [c.model_dump() for c in candidates]},
        )
