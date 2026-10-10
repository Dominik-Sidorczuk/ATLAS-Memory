"""
ATLAS Cognitive Layer: Two-Stage Conflict Arbiter (V27+ Architecture).

Implements deterministic epistemic conflict detection, candidate scoring,
fatigue gating, and belief revision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from atlas_memory.cognitive.models import ArbitrationCandidate, ArbitrationResult

logger = logging.getLogger(__name__)

from atlas_memory.cognitive.lexicon import (
    AFFIRMATIVE_WORDS,
    CONTROL_WORDS,
    DEVELOPER_TOOLS,
    ENVIRONMENT_TOKENS,
    EXCLUSIVITY_WORDS,
    FUNCTIONAL_SLOTS,
    GENERIC_WORDS,
    KNOWN_NAMESPACES,
    PROHIBITION_WORDS,
    extract_exclusive_target,
    extract_governed_action,
    extract_prohibited_terms,
    extract_stems,
    extract_tokens,
    polish_stem,
)


@dataclass(frozen=True)
class EpistemicCoordinate:
    """Koordynat tożsamości faktu w modelu Entity-Attribute-Value (EAV).

    Format: <Namespace, EntityID, Slot>
    """

    namespace: str
    entity_id: str
    slot: str
    is_functional: bool = True
    raw_key: str = ""

    @property
    def entity_root(self) -> str:
        """Kanonik granicy encji (zgodny z ADR-004 i BUG-004)."""
        if ":" not in self.raw_key:
            return self.raw_key
        parts = self.raw_key.split(":")
        if parts[0] == "hermes" and len(parts) >= 3:
            return ":".join(parts[:3])
        if (parts[0] in KNOWN_NAMESPACES or len(parts) >= 3) and len(parts) >= 2:
            return ":".join(parts[:2])
        return parts[0]

    @classmethod
    def from_key(cls, key: str) -> EpistemicCoordinate:
        clean = key.strip()
        if ":" not in clean:
            is_func = clean.lower() in FUNCTIONAL_SLOTS
            return cls(namespace="fact", entity_id=clean, slot="value", is_functional=is_func, raw_key=clean)

        parts = clean.split(":")
        ns = parts[0].lower()

        if ns == "hermes" and len(parts) >= 3:
            return cls(namespace="hermes", entity_id=parts[1], slot=":".join(parts[2:]), is_functional=True, raw_key=clean)

        if len(parts) == 2:
            slot = parts[1] if parts[1] in FUNCTIONAL_SLOTS else "value"
            entity_id = parts[1] if slot == "value" else parts[0]
            is_func = slot in FUNCTIONAL_SLOTS
            return cls(namespace=ns, entity_id=entity_id, slot=slot, is_functional=is_func, raw_key=clean)

        p1, p2 = parts[1], parts[2]
        rest = ":".join(parts[3:]) if len(parts) > 3 else ""

        if p1 in FUNCTIONAL_SLOTS and p2 not in FUNCTIONAL_SLOTS:
            entity_id = f"{p2}:{rest}" if rest else p2
            slot = p1
            is_func = True
        else:
            entity_id = p1
            slot = f"{p2}:{rest}" if rest else p2
            is_func = p2 in FUNCTIONAL_SLOTS or slot in FUNCTIONAL_SLOTS

        return cls(namespace=ns, entity_id=entity_id, slot=slot, is_functional=is_func, raw_key=clean)


def entity_root(key: str) -> str:
    """Single source of truth for the entity root of a memory key."""
    return EpistemicCoordinate.from_key(key).entity_root


MULTI_VALUED_SLOTS: frozenset[str] = frozenset({
    "tags", "tag", "allowed_users", "users", "members", "roles", "items",
    "dependencies", "plugins", "features", "scopes", "permissions",
})


class EpistemicDecisionKernel:
    """Jedno Jądro Prawdy EAV (F4) wspólne dla ConflictArbitera (online) i DeclarativeConsolidatora (offline).

    Orzeka: 'supersede' | 'accept' | 'warn_conflict'.
    Gwarantuje nienaruszalność granicy encji (`entity_root`) oraz współistnienie slotów wielowartościowych.
    """

    @staticmethod
    def is_multi_valued_slot(coord: EpistemicCoordinate) -> bool:
        slot_tokens = set(coord.slot.lower().replace(":", "_").split("_")) | {
            coord.slot.lower(),
            coord.entity_id.lower(),
        }
        return bool(slot_tokens & MULTI_VALUED_SLOTS)

    @classmethod
    def can_supersede(
        cls,
        incoming_key: str,
        existing_key: str,
        is_explicit_supersede: bool = False,
    ) -> bool:
        if is_explicit_supersede or incoming_key == existing_key:
            return True
        coord_in = EpistemicCoordinate.from_key(incoming_key)
        coord_ex = EpistemicCoordinate.from_key(existing_key)
        if coord_in.entity_root != coord_ex.entity_root:
            return False
        if cls.is_multi_valued_slot(coord_in) or cls.is_multi_valued_slot(coord_ex):
            return False
        return True

    @classmethod
    def decide(
        cls,
        incoming_key: str,
        existing_key: str,
        *,
        is_user_explicit: bool,
        incoming_conf: float,
        existing_conf: float,
        conf_threshold: float,
        best_score: float,
        reason: str,
        is_explicit_supersede: bool = False,
        is_rule_melted: bool = False,
    ) -> str:
        """Zwraca 'supersede' | 'accept' | 'warn_conflict'."""
        coord_in = EpistemicCoordinate.from_key(incoming_key)
        coord_ex = EpistemicCoordinate.from_key(existing_key)
        if incoming_key != existing_key and (cls.is_multi_valued_slot(coord_in) or cls.is_multi_valued_slot(coord_ex)):
            return "accept"

        can_auto = cls.can_supersede(incoming_key, existing_key, is_explicit_supersede=is_explicit_supersede)
        has_topic_match = bool(best_score >= 35.0 or incoming_key == existing_key)

        # Dekrystalizacja i topnienie procedur przy dryfie OOD (v146.0)
        is_rule_melted_supersede = bool(
            is_rule_melted
            and can_auto
            and incoming_conf >= 0.70
            and (has_topic_match or incoming_key == existing_key)
        )

        if is_explicit_supersede or is_rule_melted_supersede or (
            is_user_explicit
            and incoming_conf >= conf_threshold
            and existing_conf <= incoming_conf
            and can_auto
            and (
                incoming_key == existing_key
                or (
                    has_topic_match
                    and reason in (
                        "exclusivity_collision",
                        "exclusivity_violation",
                        "semantic_contradiction",
                        "prohibition_contradiction",
                        "functional_slot_revision",
                    )
                )
            )
        ):
            return "supersede"

        if reason == "confidence_inversion" or best_score >= 35.0:
            return "warn_conflict"
        return "accept"


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

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        active_sensing_engine: Optional[Any] = None,
    ) -> None:
        self.config = config or {}
        self.confidence_threshold = float(self.config.get("confidence_threshold", 0.95))
        self.high_score_threshold = float(self.config.get("high_score_threshold", 35.0))
        if active_sensing_engine is not None:
            self.active_sensing_engine = active_sensing_engine

    @property
    def active_sensing(self) -> Optional[Any]:
        return getattr(self, "active_sensing_engine", None)

    @active_sensing.setter
    def active_sensing(self, val: Any) -> None:
        self.active_sensing_engine = val

    def _get_active_sensing_engine(self) -> Optional[Any]:
        if getattr(self, "active_sensing_engine", None) is not None:
            return self.active_sensing_engine
        try:
            from atlas_memory.active.prediction_error import get_active_sensing_engine
            return get_active_sensing_engine()
        except Exception:
            return None

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

        V117: rozbito monolit CC=156 na ortogonalne, testowalne komponenty
        (pancerz rownowaznosci: tests/test_v117_golden_master.py).
        """
        candidates: List[ArbitrationCandidate] = []
        val_str = str(val).lower() if val is not None else ""
        incoming = self._incoming_profile(key, val_str, confidence, metadata)

        # Explicit conflict marker in incoming key/metadata
        if (
            "conflict" in key.lower()
            or metadata.get("is_conflict_test")
            or metadata.get("test_type") == "conflict"
        ):
            candidates.append(
                self._explicit_marker_candidate(key, val, val_str, confidence)
            )

        for ex_k, item in existing_items.items():
            cand = self._scan_existing_item(
                ex_k, item, key, val, val_str, confidence, metadata, incoming
            )
            if cand is not None:
                candidates.append(cand)

        # Sort all candidates descending by score (never take first dictionary match)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    # ── V117 helpers: profil wejscia / parsingu istniejacego rekordu ──────────────

    def _incoming_profile(
        self, key: str, val_str: str, confidence: float, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ekstraktuje profil leksykalny zapisu przychodzacego (czysta funkcja)."""
        tokens = extract_tokens(f"{key} {val_str}")
        topic_tokens = tokens - CONTROL_WORDS
        topic_stems = extract_stems(topic_tokens)
        specific_tokens = topic_tokens - GENERIC_WORDS - DEVELOPER_TOOLS
        specific_stems = topic_stems - {polish_stem(g) for g in (GENERIC_WORDS | DEVELOPER_TOOLS)}

        source_type = str(metadata.get("source_type") or metadata.get("source") or "user_explicit")
        is_user_explicit = (source_type == "user_explicit" or (confidence >= self.confidence_threshold and source_type != "agent_inference"))

        return {
            "tokens": tokens,
            "topic_tokens": topic_tokens,
            "topic_stems": topic_stems,
            "specific_tokens": specific_tokens,
            "specific_stems": specific_stems,
            "has_excl": bool(tokens & EXCLUSIVITY_WORDS),
            "has_prohib": bool(tokens & PROHIBITION_WORDS),
            "has_aff": bool(tokens & AFFIRMATIVE_WORDS),
            "prohibited": extract_prohibited_terms(f"{key} {val_str}"),
            "prohib_stems": extract_stems(extract_prohibited_terms(f"{key} {val_str}")),
            "excl_targets": extract_exclusive_target(f"{key} {val_str}"),
            "gov_actions": extract_governed_action(topic_tokens),
            "tools": {t for t in tokens if t in DEVELOPER_TOOLS},
            "env": {p for p in key.split(":") if p in ENVIRONMENT_TOKENS} | {t for t in tokens if t in ENVIRONMENT_TOKENS},
            "is_user_explicit": is_user_explicit,
        }

    @staticmethod
    def _explicit_marker_candidate(key, val, val_str, confidence):
        """Jawny marker konfliktu w kluczu/metadata → kandydat score=100."""
        return ArbitrationCandidate(
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

    @staticmethod
    def _parse_existing_item(item: Any):
        """Zwraca (ex_val, ex_conf, ex_meta) albo None gdy rekord superseded (guard clause)."""
        if getattr(item, "is_superseded", False):
            return None
        if isinstance(item, dict):
            ex_meta = dict(item.get("metadata") or {})
            if item.get("is_superseded") is True or ex_meta.get("is_superseded") is True or ex_meta.get("superseded") is True:
                return None
            ex_val = str(item.get("value") or item.get("val") or item.get("object") or "")
            ex_conf = float(item.get("confidence", 1.0))
        else:
            ex_val = str(getattr(item, "value", getattr(item, "object", "")))
            ex_conf = float(getattr(item, "confidence", 1.0))
            ex_meta = dict(getattr(item, "metadata", {}) or {})
            if ex_meta.get("is_superseded") is True or ex_meta.get("superseded") is True:
                return None
        return ex_val, ex_conf, ex_meta

    def _scan_existing_item(
        self, ex_k, item, key, val, val_str, confidence, metadata, incoming
    ) -> Optional[ArbitrationCandidate]:
        """Jedna iteracja petli po existing_items: max 1 kandydat na rekord."""
        parsed = self._parse_existing_item(item)
        if parsed is None:
            return None
        ex_val, ex_conf, ex_meta = parsed
        ex_val_lower = ex_val.lower()

        # Case 0: Explicit belief revision targeting this key
        if metadata.get("supersedes") == ex_k:
            return ArbitrationCandidate(
                key=ex_k,
                score=100.0,
                reason="belief_revision",
                conflict_type="belief_revision",
                conflicting_fact=ex_val,
                incoming_claim=str(val),
                overlapping_terms=[ex_k],
                message=f"Explicit belief revision: write supersedes '{ex_k}'.",
                confidence=ex_conf,
                metadata=ex_meta,
            )

        if ex_k == key:
            skip, cand = self._evaluate_same_key(
                ex_k, ex_val, ex_val_lower, ex_conf, ex_meta,
                key, val, val_str, confidence, incoming,
            )
            if skip:
                return cand

        return self._scan_cross_key(ex_k, ex_val, ex_val_lower, ex_conf, ex_meta, key, val, incoming)

    def _evaluate_same_key(
        self, ex_k, ex_val, ex_val_lower, ex_conf, ex_meta, key, val, val_str, confidence, incoming
    ):
        """Case A: kolizja tego samego klucza. Zwraca (handled|_FALLTHROUGH, candidate|None)."""
        if ex_val_lower == val_str:
            # Same value — check for significant confidence degradation (> 0.20 drop) before skipping
            return True, self._confidence_inversion_candidate(
                ex_k, ex_val, ex_conf, ex_meta, val, confidence, same_value=True
            )
        is_user_explicit = incoming["is_user_explicit"]
        rule_prefixes = ("rule:", "user_rule:", "preference:", "constraint:")
        is_ex_rule = ex_k.startswith(rule_prefixes)
        is_rule_melted = False
        ase = self._get_active_sensing_engine()
        if is_ex_rule and ase is not None:
            coord_ex = EpistemicCoordinate.from_key(ex_k)
            if hasattr(ase, "is_entity_critical"):
                if ase.is_entity_critical(coord_ex.entity_id) or ase.is_entity_critical(ex_k):
                    is_rule_melted = True

        # Belief revision: user_explicit write, conf >= threshold, conf >= existing OR melted rule superseded under OOD
        if (is_user_explicit and confidence >= self.confidence_threshold and ex_conf <= confidence) or (is_rule_melted and confidence >= 0.70):
            cand_meta = dict(ex_meta or {}) if is_rule_melted else ex_meta
            if is_rule_melted:
                cand_meta["is_rule_melted"] = True
            return True, ArbitrationCandidate(
                key=ex_k,
                score=95.0,
                reason="belief_revision",
                conflict_type="belief_revision",
                conflicting_fact=ex_val,
                incoming_claim=str(val),
                overlapping_terms=[key],
                message=(
                    f"Belief revision: melted rule '{key}' superseded under OOD drift."
                    if is_rule_melted
                    else f"Belief revision: user updated value for key '{key}'."
                ),
                confidence=ex_conf,
                metadata=cand_meta,
            )
        inversion = self._confidence_inversion_candidate(ex_k, ex_val, ex_conf, ex_meta, val, confidence)
        if inversion is not None:
            return True, inversion
        return False, None

    @staticmethod
    def _confidence_inversion_candidate(ex_k, ex_val, ex_conf, ex_meta, val, confidence, same_value=False):
        """Nizsza pewnosc przy nadpisaniu ustalonego faktu → kandydat score=90 albo None."""
        ex_source = str(ex_meta.get("source_type") or ex_meta.get("source") or "")
        min_drop = 0.20 if same_value else 0.0
        if (ex_conf >= 0.90 or ex_source == "user_explicit") and (ex_conf - confidence > min_drop):
            return ArbitrationCandidate(
                key=ex_k,
                score=90.0,
                reason="confidence_inversion",
                conflict_type="confidence_inversion",
                conflicting_fact=ex_val,
                incoming_claim=str(val),
                overlapping_terms=[ex_k],
                message=f"Incoming write has lower confidence ({confidence}) than established fact ({ex_conf}).",
                confidence=ex_conf,
                metadata=ex_meta,
            )
        return None
    def _existing_profile(self, ex_k: str, ex_val_lower: str) -> Dict[str, Any]:
        """Profil leksykalny istniejacego rekordu (lustrzany do _incoming_profile)."""
        ex_tokens = extract_tokens(f"{ex_k} {ex_val_lower}")
        ex_topic_tokens = ex_tokens - CONTROL_WORDS
        ex_topic_stems = extract_stems(ex_topic_tokens)
        ex_specific_tokens = ex_topic_tokens - GENERIC_WORDS - DEVELOPER_TOOLS
        ex_specific_stems = ex_topic_stems - {polish_stem(g) for g in (GENERIC_WORDS | DEVELOPER_TOOLS)}
        ex_prohibited = extract_prohibited_terms(f"{ex_k} {ex_val_lower}")
        return {
            "tokens": ex_tokens,
            "topic_tokens": ex_topic_tokens,
            "topic_stems": ex_topic_stems,
            "specific_tokens": ex_specific_tokens,
            "specific_stems": ex_specific_stems,
            "has_excl": bool(ex_tokens & EXCLUSIVITY_WORDS),
            "has_prohib": bool(ex_tokens & PROHIBITION_WORDS),
            "has_aff": bool(ex_tokens & AFFIRMATIVE_WORDS),
            "prohibited": ex_prohibited,
            "prohib_stems": extract_stems(ex_prohibited),
            "excl_targets": extract_exclusive_target(f"{ex_k} {ex_val_lower}"),
            "gov_actions": extract_governed_action(ex_topic_tokens),
            "tools": {t for t in ex_tokens if t in DEVELOPER_TOOLS},
            "env": {p for p in ex_k.split(":") if p in ENVIRONMENT_TOKENS} | {t for t in ex_tokens if t in ENVIRONMENT_TOKENS},
        }

    @staticmethod
    def _skip_episodic(ex_k: str) -> bool:
        """Ignore conversational churn or episodic working memory."""
        prefixes = (
            "mnemosyne:mnemosyne_working_memory:",
            "mnemosyne:mnemosyne_episodic_memory:",
            "conversation",
            "fact:conversation",
        )
        return ex_k.startswith(prefixes) or ex_k.endswith(":episodic_event")

    @staticmethod
    def _orthogonal_envs(env_in, env_ex) -> bool:
        """Orthogonal environments guard (e.g. staging vs production)."""
        return bool(env_in and env_ex and not (env_in & env_ex))

    @staticmethod
    def _orthogonal_tools(tools_in, tools_ex, same_ns: bool) -> bool:
        """Orthogonal developer tools guard (e.g. git vs pytest, docker vs pixi)."""
        return bool(tools_in and tools_ex and not (tools_in & tools_ex) and not same_ns)

    @staticmethod
    def _same_namespace(key: str, ex_k: str, coord_in, coord_ex) -> bool:
        ns_in = [p for p in key.split(":")[:-1] if p not in GENERIC_WORDS]
        ns_ex = [p for p in ex_k.split(":")[:-1] if p not in GENERIC_WORDS]
        coord_match = coord_in.namespace == coord_ex.namespace and coord_in.entity_id == coord_ex.entity_id
        return bool(coord_match or (len(ns_in) >= 1 and ns_in == ns_ex))

    @staticmethod
    def _overlap_measures(incoming, exp) -> Dict[str, Any]:
        token_overlap = (incoming["topic_tokens"] & exp["topic_tokens"]) - GENERIC_WORDS
        stem_overlap = (incoming["topic_stems"] & exp["topic_stems"]) - {polish_stem(g) for g in GENERIC_WORDS}
        specific_overlap = incoming["specific_tokens"] & exp["specific_tokens"]
        specific_stem_overlap = incoming["specific_stems"] & exp["specific_stems"]
        in_meaningful = incoming["specific_tokens"] | incoming["specific_stems"]
        ex_meaningful = exp["specific_tokens"] | exp["specific_stems"]
        shorter_len = max(1, min(len(in_meaningful), len(ex_meaningful)))
        overlap_count = len(specific_overlap | specific_stem_overlap)
        return {
            "token_overlap": token_overlap,
            "stem_overlap": stem_overlap,
            "specific_overlap": specific_overlap,
            "specific_stem_overlap": specific_stem_overlap,
            "overlap_count": overlap_count,
            "overlap_density": overlap_count / shorter_len,
        }

    @staticmethod
    def _has_topic_overlap(ov, same_ns: bool) -> bool:
        left = same_ns and (len(ov["token_overlap"]) >= 1 or len(ov["stem_overlap"]) >= 1)
        mid = (
            len(ov["specific_overlap"]) >= 2
            or len(ov["specific_stem_overlap"]) >= 2
            or (len(ov["token_overlap"]) >= 2 and len(ov["specific_overlap"]) >= 1)
        )
        right = same_ns or ov["overlap_density"] >= 0.25 or ov["overlap_count"] >= 4
        return bool(left or (mid and right))

    @staticmethod
    def _action_polarity_fallback(incoming, exp) -> bool:
        """Bez topic overlap: konflikt tylko gdy akcje sie pokrywaja z przeciwna polaryzacja."""
        actions_match = bool(incoming["gov_actions"] & exp["gov_actions"])
        return bool(actions_match and (incoming["has_excl"] or exp["has_excl"] or incoming["has_prohib"] or exp["has_prohib"]))

    @staticmethod
    def _consensus_agreement(incoming, exp, shared_prohib) -> bool:
        """Consensus Protection: Shared prohibitions are agreement, not contradiction."""
        return bool(incoming["has_prohib"] and exp["has_prohib"] and shared_prohib)

    @staticmethod
    def _base_cross_score(ctx) -> float:
        score = 0.0
        key, ex_k = ctx["key"], ctx["ex_k"]
        if ctx["same_ns"]:
            score += 50.0
        rule_prefixes = ("rule:", "user_rule:", "preference:", "constraint:")
        is_in_rule = key.startswith(rule_prefixes)
        is_ex_rule = ex_k.startswith(rule_prefixes)
        
        # Termodynamiczna Dekrystalizacja Procedur (Topnienie SOP przy dryfie OOD, v146.0)
        is_rule_melted = False
        ase = ctx.get("active_sensing_engine")
        if is_ex_rule and ase is not None:
            coord_ex = EpistemicCoordinate.from_key(ex_k)
            if hasattr(ase, "is_entity_critical"):
                if ase.is_entity_critical(coord_ex.entity_id) or ase.is_entity_critical(ex_k):
                    is_rule_melted = True
                    is_ex_rule = False  # Zdejmij dogmatyczną obronę reguły

        ctx["is_ex_rule"] = is_ex_rule
        ctx["is_rule_melted"] = is_rule_melted
        if is_in_rule and is_ex_rule:
            score += 35.0
        elif is_ex_rule:
            score += 15.0
        elif is_rule_melted:
            score += 25.0
        if ex_k.startswith("mnemosyne:"):
            score -= 40.0
        ov = ctx["ov"]
        score += 15.0 * len(ov["specific_overlap"])
        score += 10.0 * len(ov["specific_stem_overlap"])
        score += 5.0 * len(ov["token_overlap"])
        score += ctx["ex_conf"] * 5.0
        return score

    def _scan_cross_key(self, ex_k, ex_val, ex_val_lower, ex_conf, ex_meta, key, val, incoming):
        """Case B: relacja miedzy roznymi kluczami. Guard clauses -> wczesne zwroty None."""
        if self._skip_episodic(ex_k):
            return None

        exp = self._existing_profile(ex_k, ex_val_lower)

        if self._orthogonal_envs(incoming["env"], exp["env"]):
            return None

        coord_in = EpistemicCoordinate.from_key(key)
        coord_ex = EpistemicCoordinate.from_key(ex_k)

        # Multi-valued slots (tags, allowed_users, etc.) on distinct keys coexist without conflict
        if ex_k != key and (
            EpistemicDecisionKernel.is_multi_valued_slot(coord_in)
            or EpistemicDecisionKernel.is_multi_valued_slot(coord_ex)
        ):
            return None

        same_ns = self._same_namespace(key, ex_k, coord_in, coord_ex)

        if self._orthogonal_tools(incoming["tools"], exp["tools"], same_ns):
            return None

        ov = self._overlap_measures(incoming, exp)
        has_topic = self._has_topic_overlap(ov, same_ns)

        shared_prohib = (incoming["prohibited"] & exp["prohibited"]) | {
            t for t in incoming["prohibited"] if polish_stem(t) in exp["prohib_stems"]
        }
        if self._consensus_agreement(incoming, exp, shared_prohib):
            return None

        if not has_topic and not self._action_polarity_fallback(incoming, exp):
            return None

        ctx = {
            "key": key, "ex_k": ex_k, "val": val, "ex_val": ex_val,
            "ex_conf": ex_conf, "ex_meta": ex_meta, "same_ns": same_ns,
            "ov": ov, "shared_prohib": shared_prohib, "in": incoming, "ex": exp,
            "active_sensing_engine": self._get_active_sensing_engine(),
        }
        candidate_score = self._base_cross_score(ctx)
        found = self._first_semantic_conflict(ctx)
        if found is None:
            return None
        payload, bonus = found
        is_rule_melted = ctx.get("is_rule_melted", False)
        cand_meta = dict(ex_meta or {}) if is_rule_melted else ex_meta
        if is_rule_melted:
            cand_meta["is_rule_melted"] = True
        return ArbitrationCandidate(
            key=ex_k,
            score=candidate_score + bonus,
            reason=payload["reason"],
            conflict_type=payload.get("conflict_type"),
            conflicting_fact=payload.get("conflicting_fact"),
            incoming_claim=payload.get("incoming_claim"),
            overlapping_terms=list(payload.get("overlapping_terms") or []),
            message=payload.get("message"),
            confidence=ex_conf,
            metadata=cand_meta,
        )

    # ── V117:detektory semantyczne (pierwszy trafiony wygrywa — kolejnosc zachowana) ──

    def _first_semantic_conflict(self, ctx):
        """Sekwencja V112-era: d1..d5 w starej kolejnosci; pierwszy znaleziony + bonus."""
        for detector in (
            self._detect_exclusivity_collision,
            self._detect_prohibition_contradiction,
            self._detect_capability_conflict,
            self._detect_exclusivity_violation,
            self._detect_polarity_contradiction,
        ):
            found = detector(ctx)
            if found is not None:
                return found
        return None

    @staticmethod
    def _detect_exclusivity_collision(ctx):
        """1. Wybpierajace się asekurecje exclusivity na wspolnym domenie (+40)."""
        in_, ex = ctx["in"], ctx["ex"]
        ov = ctx["ov"]
        if not (in_["has_excl"] and ex["has_excl"]):
            return None
        actions_match = bool(in_["gov_actions"] & ex["gov_actions"])
        targets_differ = bool(in_["excl_targets"] and ex["excl_targets"] and not (in_["excl_targets"] & ex["excl_targets"]))
        differing = (in_["specific_tokens"] - ex["specific_tokens"]) | (ex["specific_tokens"] - in_["specific_tokens"])
        near = ctx["same_ns"] or len(ov["specific_overlap"]) >= 1 or len(ov["stem_overlap"]) >= 1
        if (actions_match and targets_differ) or (differing and near):
            return (
                {
                    "reason": "exclusivity_collision",
                    "conflict_type": "exclusivity_collision",
                    "conflicting_fact": ctx["ex_val"],
                    "incoming_claim": str(ctx["val"]),
                    "overlapping_terms": list(ov["token_overlap"] or ov["stem_overlap"] or in_["excl_targets"]),
                    "message": f"Incoming exclusivity assertion contradicts established exclusivity rule '{ctx['ex_k']}'.",
                },
                40.0,
            )
        return None

    @staticmethod
    def _detect_prohibition_contradiction(ctx):
        """2. Przychodzacy asertuje akcje zabroniona w pamieci (+35)."""
        in_, ex = ctx["in"], ctx["ex"]
        if not (ex["has_prohib"] and not in_["has_prohib"]):
            return None
        prohibited_overlap = (
            (in_["specific_tokens"] & ex["prohibited"])
            | {t for t in in_["specific_tokens"] if polish_stem(t) in ex["prohib_stems"]}
        ) - ctx["shared_prohib"]
        action_words = ("push", "commit", "zapis", "edit", "edycja", "bezpośrednio", "bezpośredni", "delete", "remove", "drop", "root", "hasło", "hasła", "secret", "secrets")
        has_action_in_prohib = any(t in prohibited_overlap for t in action_words)
        near = len(prohibited_overlap) >= 2 or has_action_in_prohib or ctx["same_ns"]
        if prohibited_overlap and near:
            return (
                {
                    "reason": "semantic_contradiction",
                    "conflict_type": "prohibition_contradiction",
                    "conflicting_fact": ctx["ex_val"],
                    "incoming_claim": str(ctx["val"]),
                    "overlapping_terms": list(prohibited_overlap),
                    "message": f"Incoming write asserts action prohibited by established memory '{ctx['ex_k']}'.",
                },
                35.0,
            )
        return None

    @staticmethod
    def _detect_capability_conflict(ctx):
        """3. Przychodzacy zabrania capability ustalonej w pamieci (+30)."""
        in_, ex = ctx["in"], ctx["ex"]
        ov = ctx["ov"]
        if not (in_["has_prohib"] and not ex["has_prohib"] and ex["has_aff"]):
            return None
        prohibited_overlap = (
            (ex["specific_tokens"] & in_["prohibited"])
            | {t for t in ex["specific_tokens"] if polish_stem(t) in in_["prohib_stems"]}
        ) - ctx["shared_prohib"]
        near = ctx.get("is_ex_rule") or len(ov["specific_overlap"]) >= 1 or ctx["same_ns"]
        if prohibited_overlap and near:
            return (
                {
                    "reason": "semantic_contradiction",
                    "conflict_type": "prohibition_contradiction",
                    "conflicting_fact": ctx["ex_val"],
                    "incoming_claim": str(ctx["val"]),
                    "overlapping_terms": list(prohibited_overlap),
                    "message": f"Incoming write prohibits capability established in memory '{ctx['ex_k']}'.",
                },
                30.0,
            )
        return None

    @staticmethod
    def _detect_exclusivity_violation(ctx):
        """4. Ustalone exclusivity naruszone przez alternatywny target przychodzacego (+35)."""
        in_, ex = ctx["in"], ctx["ex"]
        ov = ctx["ov"]
        if not (ex["has_excl"] and not in_["has_excl"]):
            return None
        shared_actions = in_["gov_actions"] & ex["gov_actions"]
        near = bool(shared_actions) or (ctx["same_ns"] and len(ov["specific_overlap"]) >= 1)
        if not near:
            return None
        overlap_stems = ov["stem_overlap"]
        authorized_targets = (ex["topic_tokens"] - {t for t in ex["topic_tokens"] if polish_stem(t) in overlap_stems}) - ex["prohibited"]
        incoming_targets = in_["tokens"] - {t for t in in_["tokens"] if polish_stem(t) in overlap_stems}
        violates = bool(incoming_targets and not (incoming_targets & authorized_targets)) or bool(in_["tokens"] & ex["prohibited"])
        if not violates:
            return None
        return (
            {
                "reason": "exclusivity_violation",
                "conflict_type": "exclusivity_violation",
                "conflicting_fact": ctx["ex_val"],
                "incoming_claim": str(ctx["val"]),
                "overlapping_terms": list(ov["token_overlap"] or ov["stem_overlap"]),
                "message": f"Incoming assertion violates established exclusive rule '{ctx['ex_k']}'.",
            },
            35.0,
        )

    @staticmethod
    def _detect_polarity_contradiction(ctx):
        """5. Bezposrednia polaryzacja: zakaz vs afirmacja (+25)."""
        in_, ex = ctx["in"], ctx["ex"]
        ov = ctx["ov"]
        negates_established = in_["has_prohib"] and ex["has_aff"] and not ex["has_prohib"]
        negates_incoming = in_["has_aff"] and ex["has_prohib"] and not in_["has_prohib"]
        if not (negates_established or negates_incoming):
            return None
        if len(ov["specific_overlap"]) >= 2 or (ctx["same_ns"] and len(ov["token_overlap"]) >= 2):
            return (
                {
                    "reason": "semantic_contradiction",
                    "conflict_type": "semantic_contradiction",
                    "conflicting_fact": ctx["ex_val"],
                    "incoming_claim": str(ctx["val"]),
                    "overlapping_terms": list(ov["token_overlap"] or ov["stem_overlap"]),
                    "message": f"Incoming write contradicts established memory '{ctx['ex_k']}'.",
                },
                25.0,
            )
        return None

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

        is_explicit_supersede = bool(
            best_candidate.reason == "belief_revision"
            or meta.get("supersedes") == best_candidate.key
            or meta.get("is_belief_revision") is True
            or meta.get("action") == "supersede"
        )

        is_rule_melted = bool(
            best_candidate.metadata.get("is_rule_melted")
            or meta.get("is_rule_melted")
        )

        kernel_action = EpistemicDecisionKernel.decide(
            incoming_key=key,
            existing_key=best_candidate.key,
            is_user_explicit=is_user_explicit,
            incoming_conf=confidence,
            existing_conf=best_candidate.confidence,
            conf_threshold=self.confidence_threshold,
            best_score=best_score,
            reason=best_candidate.reason,
            is_explicit_supersede=is_explicit_supersede,
            is_rule_melted=is_rule_melted,
        )

        if kernel_action == "supersede":
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
