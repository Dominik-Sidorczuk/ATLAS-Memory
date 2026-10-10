"""
ATLAS Cognitive Layer: Epistemic Write Path (V27+ Architecture).

Acts as the single choke point for all memory state write operations, orchestrating:
1. Epistemic conflict arbitration via ConflictArbiter.
2. Atomic belief revision: marks superseded keys in KV, wires (old)-[:SUPERSEDED_BY]->(new) in graph, purges old key from hot buffer.
3. Same-key value revision: tracks prior_value, increments revision_count, wires version transitions in graph.
4. Persistence to VerifiedKVStore (set_sync) with SHA-256 audit log.
5. Hot working memory buffer updates via HotBufferPolicy.
6. Multi-hop knowledge graph auto-insertion (arrow chains, dict topologies, relation triples).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from atlas_memory.cognitive.conflict_arbiter import ConflictArbiter
from atlas_memory.cognitive.models import (
    ArbitrationResult,
    clamp_conf,
)
from atlas_memory.core import is_pii_entity
from atlas_memory.core.agi_logger import log_conflict, log_remember
from atlas_memory.core.models import SpeechAct, classify_speech_act
from atlas_memory.core.protocols import (
    DualNodeGraphStoreProtocol,
    GraphStoreProtocol,
    HotBufferProtocol,
    KVStoreProtocol,
    VectorStoreProtocol,
)
from atlas_memory.l1_working.hot_buffer import HotBufferPolicy
from atlas_memory.l2_semantic.qdrant_store import normalize_upsert_key

logger = logging.getLogger(__name__)

CONVERSATIONAL_NOISE_TOKENS = {
    # Polish conjunctions, particles, interrogatives, typos, fillers
    "cztli", "czyli", "albo", "poniewaz", "ponieważ", "jakby", "oraz", "lecz",
    "wiec", "więc", "zeby", "żeby", "jezeli", "jeżeli", "jesli", "jeśli", "gdy",
    "wtedy", "dlaczego", "kiedy", "gdzie", "skad", "skąd", "dokad", "dokąd",
    "jak", "co", "kto", "czy", "oto", "otoz", "otóż", "takze", "także",
    "rowniez", "również", "chociaz", "chociaż", "choc", "choć", "jednak",
    "natomiast", "zatem", "przeciez", "przecież", "chyba", "moze", "może",
    "raczej", "tam", "tu", "tutaj", "teraz", "bądź", "badz", "ani",
    "zarówno", "zarowno", "aczkolwiek", "wszakże", "wszakze",
    # English discourse particles, conjunctions, interrogatives
    "which", "what", "where", "when", "why", "how", "who", "whom", "whose",
    "that", "this", "these", "those", "then", "than", "thus", "also", "and",
    "or", "but", "so", "because", "although", "since", "unless", "either",
    "neither", "nor", "if", "whether", "well", "actually", "basically",
}


def _levenshtein_le_1(s1: str, s2: str) -> bool:
    """Fast check if Levenshtein distance between s1 and s2 is <= 1."""
    if s1 == s2:
        return True
    l1, l2 = len(s1), len(s2)
    if abs(l1 - l2) > 1:
        return False
    if l1 == l2:
        diffs = sum(1 for c1, c2 in zip(s1, s2, strict=True) if c1 != c2)
        return diffs <= 1
    if l1 > l2:
        s1, s2 = s2, s1
    i = j = 0
    diffs = 0
    while i < len(s1) and j < len(s2):
        if s1[i] == s2[j]:
            i += 1
            j += 1
        else:
            diffs += 1
            if diffs > 1:
                return False
            j += 1
    return True


def is_lexically_grounded_entity(
    key: Any,
    val: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """
    DIRECTIVE 2: Heuristic & Lexical Grounding Check.
    Blocks hallucinated or ungrounded conversational noise, typos (e.g. 'cztli'),
    and isolated particles lacking domain namespace or structural grounding.
    """
    key_str = str(key or "").strip()
    if not key_str:
        return False, "empty_key"

    # 1. Namespaced keys (e.g., hermes:*, user:*, fact:*, doc:*) have explicit epistemic grounding
    if any(sep in key_str for sep in (":", "/", ".")) and len(key_str.split(":")[0]) > 1:
        return True, "namespaced_entity"

    clean = key_str.lower()

    # 2. Direct match against conversational noise / discourse particles
    if clean in CONVERSATIONAL_NOISE_TOKENS:
        return False, f"conversational_noise_token: '{clean}'"

    # 3. Typo guard against common discourse particles (Levenshtein distance <= 1 for tokens >= 4 chars)
    if len(clean) >= 4 and not clean.isdigit():
        for noise in CONVERSATIONAL_NOISE_TOKENS:
            if len(noise) >= 4 and _levenshtein_le_1(clean, noise):
                return False, f"conversational_noise_typo: '{clean}' (similar to '{noise}')"

    # 4. Extremely short bare tokens (< 3 chars) without namespace or digits
    if len(clean) < 3 and not clean.isdigit():
        return False, f"isolated_short_token: '{clean}'"

    # 5. Bare token with trivial affirmation/negation value without structural metadata
    val_clean = str(val or "").strip().strip('"\'').lower()
    meta = metadata or {}
    has_predicate = bool(meta.get("predicate") or meta.get("relation"))
    if val_clean in ("nie", "tak", "no", "yes", "true", "false", "ok") and not has_predicate:
        return False, f"ungrounded_discourse_affirmation_or_negation: '{clean}'='{val_clean}'"

    return True, "grounded"


class EpistemicWritePath:
    """
    Single Choke Point for ATLAS State Write Operations.
    """

    def __init__(
        self,
        kv_store: Optional[KVStoreProtocol] = None,
        graph_store: Optional[GraphStoreProtocol] = None,
        hot_buffer: Optional[HotBufferProtocol] = None,
        conflict_arbiter: Optional[ConflictArbiter] = None,
        vector_store: Optional[VectorStoreProtocol] = None,
        ttt_layer: Optional[Any] = None,
        jepa_buffer: Optional[Any] = None,
    ) -> None:
        self.kv_store = kv_store
        self.graph_store = graph_store
        self.hot_buffer = hot_buffer if hot_buffer is not None else HotBufferPolicy()
        self.conflict_arbiter = conflict_arbiter if conflict_arbiter is not None else ConflictArbiter()
        self.vector_store = vector_store
        self.ttt_layer = ttt_layer
        self.jepa_buffer = jepa_buffer
        self.session_touched_keys: Dict[str, Set[str]] = {}
        self.audit_log: list[Dict[str, Any]] = []

    def commit_relation(
        self,
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 1.0,
        session_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        **kwargs: Any,
    ) -> bool:
        """Commit an epistemic relation edge into the knowledge graph via the single write path choke point."""
        conf = clamp_conf(confidence)

        # Odrzucaj/ignoruj krawędzie prozy/dokumentacji (filtr F6):
        if str(predicate).strip().lower() in ("stated_memory", "raw_content", "documentation", "source_doc"):
            logger.debug("[WritePath] Ignored prose predicate: %s", predicate)
            return False
        if str(object).strip().count(" ") > 6 and any(punc in str(object) for punc in (".", ",", ";", "!", "?")):
            logger.debug("[WritePath] Ignored prose sentence object: %s", object)
            return False

        # Filtr PII:
        if is_pii_entity(subject) or is_pii_entity(predicate) or is_pii_entity(object):
            logger.debug("[WritePath] Ignored relation with PII")
            return False

        # Zapobiegaj samopętlom:
        if subject.strip() == object.strip():
            return False

        # DIRECTIVE 2: Heuristic & Lexical Grounding Check for entities
        sub_grounded, sub_r = is_lexically_grounded_entity(subject)
        obj_grounded, obj_r = is_lexically_grounded_entity(object)
        if not sub_grounded or not obj_grounded:
            logger.debug(
                "[WritePath] Ignored relation with ungrounded entity: sub='%s' (%s), obj='%s' (%s)",
                subject, sub_r, object, obj_r,
            )
            return False

        if self.graph_store is not None and callable(getattr(self.graph_store, "add_relation", None)):
            sub_c = subject.strip()
            pred_c = predicate.strip()
            obj_c = object.strip()
            call_kwargs = dict(kwargs)
            if session_id is not None:
                call_kwargs["session_id"] = session_id
            if timestamp is not None:
                call_kwargs["timestamp"] = timestamp
            add_fn = self.graph_store.add_relation
            try:
                add_fn(
                    sub_c,
                    pred_c,
                    obj_c,
                    confidence=conf,
                    **call_kwargs,
                )
                return True
            except TypeError:
                if not callable(add_fn):
                    return False
                try:
                    add_fn(sub_c, pred_c, obj_c, confidence=conf)
                    return True
                except TypeError:
                    add_fn(sub_c, pred_c, obj_c)
                    return True
        return False

    def _auto_insert_graph_edges(
        self,
        key: str,
        val: Any,
        metadata: Dict[str, Any],
        confidence: float,
        inserted_edges: Optional[List[Tuple[str, str, str]]] = None,
    ) -> int:
        """Parses and inserts multi-hop relations into runtime knowledge graph."""
        if self.graph_store is None or not callable(getattr(self.graph_store, "add_relation", None)):
            return 0

        sub = metadata.get("subject") or key
        pred = metadata.get("predicate") or metadata.get("relation")
        obj = metadata.get("object") or metadata.get("target")
        edges_added = 0

        def _commit_and_track(s: str, p: str, o: str) -> bool:
            added = self.commit_relation(str(s), str(p), str(o), confidence=confidence)
            if added and inserted_edges is not None:
                inserted_edges.append((str(s), str(p), str(o)))
            return added

        # 1. Dict payload: add each key-value as edge
        if isinstance(val, dict):
            for p_k, p_v in val.items():
                if _commit_and_track(str(sub), str(p_k), str(p_v)):
                    edges_added += 1

        # 2. Triple payload: pred and obj explicit
        elif pred and obj:
            if _commit_and_track(str(sub), str(pred), str(obj)):
                edges_added += 1
        # 2b. V152 (FILAR I, ADR): EAV z jawnym predykatem w meta i wartością
        # jako obiekt (subject=key, predicate=meta.predicate, object=value).
        # Dotąd w takim układzie pred+value brakowało trójki → brak relacji
        # grafu; poprawka utrzymuje kanon (key:predicate:value) bez pisania
        # nowego magazynu. Walidacja: atomy bez spacji i strzałek; commit_relation
        # i tak odrzuca prozę, PII, pętle, samopętle i nieuziemione encje.
        elif pred and isinstance(val, str):
            v = val.strip()
            if (
                v
                and " " not in v
                and len(v) <= 64
                and not re.search(r"->|→|=>", v)
            ):
                if _commit_and_track(str(sub), str(pred), v):
                    edges_added += 1

        # 3. String arrow chain syntax: A -> B -> C or A → B → C or A => B
        # V109.5 (F6 Prose Guard): Only treat as symbolic chain if total string <= 160 chars
        # and every segment is a concise entity (2..64 chars, <= 6 words, not a bare number like "155.").
        elif isinstance(val, str):
            raw_parts = [p.strip() for p in re.split(r"->|→|=>", val) if p.strip()] if ("->" in val or "→" in val or "=>" in val) else []
            is_valid_symbolic_chain = (
                len(raw_parts) >= 2
                and len(val.strip()) <= 160
                and all(
                    2 <= len(p) <= 64
                    and len(p.split()) <= 6
                    and not re.fullmatch(r"[\d\W_]+", p)
                    for p in raw_parts
                )
            )
            if is_valid_symbolic_chain:
                parts = raw_parts
                if str(sub) != str(parts[0]):
                    if _commit_and_track(str(sub), "leads_to", str(parts[0])):
                        edges_added += 1
                for i in range(len(parts) - 1):
                    if _commit_and_track(str(parts[i]), "leads_to", str(parts[i + 1])):
                        edges_added += 1
            elif val.startswith("test:") or (len(val.strip()) < 64 and " " not in val.strip() and ":" in val):
                if _commit_and_track(str(sub), "relates_to", val.strip()):
                    edges_added += 1
            else:
                val_str = val.strip()
                if len(val_str) >= 25 or any(sep in val_str for sep in (":", "->", "→", "=>", "(", ")")):
                    try:
                        from atlas_memory.ingest.mnemosyne_ingest import MnemosyneIngestEngine
                        extracted = MnemosyneIngestEngine.extract_and_insert_multihop_relations(self.graph_store, str(sub), val)
                        edges_added += int(extracted or 0)
                    except Exception as ex:
                        logger.debug("[WritePath] Runtime text graph extraction error: %s", ex)

        # V104.5 (FIX-1): Dual-Node HippoRAG 2 write-path indexing via DualNodeGraphStoreProtocol
        if isinstance(self.graph_store, DualNodeGraphStoreProtocol) and callable(getattr(self.graph_store, "index_dual_node_passage", None)):
            skey = str(key).strip()
            txt = str(val).strip()
            if skey and txt:
                from datetime import datetime, timezone

                raw_ing = metadata.get("ingest_at") if isinstance(metadata, dict) else None
                if isinstance(raw_ing, datetime):
                    ing_str = raw_ing.isoformat()
                elif raw_ing is not None and str(raw_ing).strip():
                    ing_str = str(raw_ing).strip()
                else:
                    ing_str = datetime.now(timezone.utc).isoformat()

                gist_val = str(metadata.get("gist") or "") if isinstance(metadata, dict) else None
                explicit_ents = metadata.get("entities") if isinstance(metadata, dict) else None
                explicit_syns = metadata.get("synonyms") if isinstance(metadata, dict) else None
                edges_added += self.graph_store.index_dual_node_passage(
                    source_key=skey,
                    text=txt,
                    ingest_at=ing_str,
                    gist=gist_val,
                    explicit_entities=list(explicit_ents) if isinstance(explicit_ents, (list, tuple)) else None,
                    explicit_synonyms=list(explicit_syns) if isinstance(explicit_syns, list) else None,
                    # V152 (FILAR I, ADR): stamp aktu mowy propagowany do
                    # ekstrakcji dual-node (użycie opcjonalne w implementacji).
                    speech_act=str(metadata.get("speech_act") or "") if isinstance(metadata, dict) else None,
                )

        return edges_added

    @staticmethod
    def _normalize_write_input(
        key: Any,
        value: Any,
        confidence: float,
        metadata: Optional[Dict[str, Any]],
        replay_at: Optional[Any],
    ) -> tuple[str, Any, float, Dict[str, Any], Optional[Any]]:
        """Normalizes either a record dict or `(key, value, confidence, metadata)` into a flat tuple."""
        from datetime import datetime

        if isinstance(key, dict):
            record = dict(key)
            key_str = str(record.get("source_key") or record.get("key") or record.get("subject") or "").strip()
            val_out = record.get("text") if "text" in record else record.get("value", value)
            conf_out = float(record.get("confidence", confidence))
            meta_out = dict(record.get("metadata") or metadata or {})
            if "entities" in record:
                meta_out["entities"] = list(record["entities"])
            if "synonyms" in record:
                meta_out["synonyms"] = list(record["synonyms"])
            eff_replay_at = replay_at if replay_at is not None else record.get("replay_at")
        else:
            key_str = str(key or "").strip()
            val_out = value
            conf_out = confidence
            meta_out = dict(metadata or {})
            eff_replay_at = replay_at

        if not key_str:
            raise ValueError("Key is required for write operation.")
        if eff_replay_at is not None:
            meta_out["ingest_at"] = (
                eff_replay_at.isoformat() if isinstance(eff_replay_at, datetime) else str(eff_replay_at)
            )
        return key_str, val_out, conf_out, meta_out, eff_replay_at

    def write(
        self,
        key: Any,
        value: Any = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        source_type: Optional[str] = None,
        reason: str = "write_path",
        session_id: Optional[str] = None,
        vector_record: Optional[Any] = None,
        replay_at: Optional[Any] = None,
        replay: bool = False,
    ) -> Dict[str, Any]:
        """
        Orchestrates single-choke-point epistemic write:
        1. Conflict arbitration via ConflictArbiter.
        2. Belief revision supersession, graph wiring, and hot buffer purge.
        3. Same-key value revision tracking and version transition graph wiring.
        4. VerifiedKVStore persistence with SHA-256 audit log.
        5. Working memory hot buffer update.
        6. Multi-hop knowledge graph + Dual-Node HippoRAG 2 auto-insertion (with atomic rollback).
        """
        key, value, confidence, meta, replay_at = self._normalize_write_input(
            key, value, confidence, metadata, replay_at
        )

        # V152 (FILAR I, ADR): bariera aktu mowy — rozwijać WSZYSTKO. Po
        # normalizacji, PRZED filtrem leksemu i PRZED jakimkolwiek dotknięciem
        # KV/wektora/grafu (ekstrakcja OpenIE na surowej prozie = SIGSEGV,
        # token.c). Rozłączne typowanie: ConversationalTurn → reject i stop.
        speech_act = classify_speech_act(
            key,
            meta.get("predicate") if isinstance(meta, dict) else None,
            value,
        )
        if speech_act is SpeechAct.CONVERSATIONAL_TURN:
            logger.warning(
                "[WritePath SpeechAct Gate] BLOCKED conversational turn key='%s' (val='%s'): "
                "conversational_turn_barrier",
                key,
                str(value)[:30],
            )
            return {
                "status": "rejected",
                "key": key,
                "persisted": False,
                "action": "reject_conversational_turn",
                "warning": "speech_act_gate",
                "reason": "conversational_turn_barrier",
                "edges_added": 0,
            }
        # Stamp aktu mowy (EPISTEMIC_ASSERTION) w meta zapisywanej do KV —
        # spójne z MemoryRecord.speech_act / orchestrator.shadow_reconcile.
        if isinstance(meta, dict) and "speech_act" not in meta:
            meta["speech_act"] = speech_act.value

        # DIRECTIVE 2: Heuristic & Lexical Grounding Check
        is_grounded, ground_reason = is_lexically_grounded_entity(key, value, meta)
        if not is_grounded:
            logger.warning(
                "[WritePath Epistemic Guard] BLOCKED ungrounded lexical token/typo key='%s' (val='%s'): %s",
                key, str(value)[:30], ground_reason,
            )
            return {
                "status": "rejected",
                "key": key,
                "persisted": False,
                "action": "reject_ungrounded",
                "warning": "lexical_grounding_failed",
                "reason": ground_reason,
                "edges_added": 0,
            }

        if (replay or replay_at is not None) and isinstance(self.graph_store, DualNodeGraphStoreProtocol):
            if self.graph_store.has_passage(key):
                self.audit_log.append(
                    {
                        "key": key,
                        "action": "replay_no_op",
                        "reason": "replay_no_op",
                        "ingest_at": meta.get("ingest_at"),
                    }
                )
                if self.kv_store is not None and hasattr(self.kv_store, "set_sync"):
                    try:
                        self.kv_store.set_sync(
                            key,
                            value,
                            confidence=clamp_conf(confidence),
                            metadata=meta,
                            reason="replay_no_op",
                        )
                    except Exception as kv_exc:
                        logger.debug("[WritePath] replay_no_op KV touch error: %s", kv_exc)
                return {
                    "status": "ok",
                    "key": key,
                    "persisted": True,
                    "no_op": True,
                    "no_op_reason": "replay_no_op",
                    "audit_reason": "replay_no_op",
                    "edges_added": 0,
                }

        confidence = clamp_conf(confidence)
        if source_type:
            meta["source_type"] = str(source_type)
        elif "source_type" not in meta:
            meta["source_type"] = "user_explicit"
        meta["confidence"] = confidence
        if session_id:
            meta["session_id"] = str(session_id)
            sess_key = str(session_id)
        else:
            sess_key = "default"
        if sess_key not in self.session_touched_keys:
            self.session_touched_keys[sess_key] = set()
        self.session_touched_keys[sess_key].add(key)

        # Fetch existing items from KV store for arbitration
        existing_items: Dict[str, Any] = {}
        if self.kv_store is not None:
            if hasattr(self.kv_store, "get_arbitration_candidates_sync"):
                try:
                    existing_items = self.kv_store.get_arbitration_candidates_sync(key=key, value=value, limit=100)
                except Exception as exc:
                    logger.debug("[WritePath] Error fetching get_arbitration_candidates_sync: %s", exc)
            elif hasattr(self.kv_store, "get_all_sync"):
                try:
                    existing_items = self.kv_store.get_all_sync(paged=True, limit=100)
                except Exception as exc:
                    logger.debug("[WritePath] Error fetching get_all_sync: %s", exc)
            elif hasattr(self.kv_store, "get_all"):
                try:
                    existing_items = self.kv_store.get_all()
                except Exception as exc:
                    logger.debug("[WritePath] Error fetching get_all: %s", exc)

            # Ensure explicitly superseded key is present in existing_items if specified in metadata
            sup_k = meta.get("supersedes")
            if sup_k and str(sup_k) not in existing_items:
                try:
                    sup_fn = getattr(self.kv_store, "get_sync", getattr(self.kv_store, "get", None))
                    if sup_fn:
                        sup_row = sup_fn(str(sup_k))
                        if sup_row:
                            existing_items[str(sup_k)] = sup_row
                except Exception as sup_exc:
                    logger.debug("[WritePath] Error fetching supersedes key '%s': %s", sup_k, sup_exc)

        # Read-Through do Hot Buffera: domieszkowanie pamięci operacyjnej RAM do existing_items
        hot_lock = getattr(self.hot_buffer, "_lock", None) if self.hot_buffer is not None else None
        with hot_lock if hot_lock is not None else contextlib.nullcontext():
            if self.hot_buffer is not None:
                try:
                    active_hot = (
                        self.hot_buffer.get_active(session_id=session_id)
                        if hasattr(self.hot_buffer, "get_active")
                        else list(self.hot_buffer)
                    )
                    for h_entry in active_hot:
                        h_k = str(getattr(h_entry, "key", "") or (h_entry.get("key") if isinstance(h_entry, dict) else ""))
                        if not h_k:
                            continue
                        h_dict = h_entry if isinstance(h_entry, dict) else {
                            "key": h_k,
                            "value": getattr(h_entry, "value", ""),
                            "confidence": float(getattr(h_entry, "confidence", 1.0)),
                            "metadata": dict(getattr(h_entry, "metadata", {}) or {}),
                            "source_type": getattr(h_entry, "source_type", "user_explicit"),
                            "timestamp": float(getattr(h_entry, "timestamp", 0.0)),
                        }
                        if h_k not in existing_items or float(h_dict.get("timestamp", 0.0)) >= float(
                            existing_items[h_k].get("timestamp", 0.0) if isinstance(existing_items[h_k], dict) else 0.0
                        ):
                            existing_items[h_k] = h_dict
                except Exception as h_exc:
                    logger.debug("[WritePath] Error merging hot_buffer items: %s", h_exc)

            # 1. Epistemic Conflict Arbitration
            arb_res: ArbitrationResult = self.conflict_arbiter.arbitrate(
                key=key,
                val=value,
                confidence=confidence,
                metadata=meta,
                existing_items=existing_items,
            )

        if (
            arb_res.action in ("reject", "reject_conflict", "block")
            or (arb_res.is_conflict and getattr(arb_res, "should_reject", False))
            or (arb_res.is_conflict and arb_res.reason == "confidence_inversion")
        ):
            logger.warning("[WritePath Epistemic Guard] BLOCKED conflicting write for key='%s': %s", key, arb_res.message)
            return {
                "status": "rejected",
                "key": key,
                "persisted": False,
                "action": "reject_conflict",
                "warning": "epistemic_conflict",
                "conflict_details": arb_res.conflict_details,
                "arbitration": arb_res,
                "edges_added": 0,
            }

        if arb_res.is_conflict and not arb_res.is_belief_revision:
            meta["conflict_warning"] = True
            if arb_res.conflict_details:
                meta["conflict_details"] = arb_res.conflict_details
            logger.warning("[WritePath Epistemic Guard] Conflict detected for key='%s': %s", key, arb_res.message)

        # 1b. Backup state for Saga 2PC transaction rollback
        existing_target_record = existing_items.get(key)
        backup_existing_kv = dict(existing_target_record) if isinstance(existing_target_record, dict) else (existing_target_record.copy() if hasattr(existing_target_record, "copy") else existing_target_record)

        backup_hot_target = None
        backup_hot_superseded = None
        if hasattr(self.hot_buffer, "_buffer"):
            for ent in self.hot_buffer._buffer:
                if ent.key == key:
                    backup_hot_target = ent.model_copy() if hasattr(ent, "model_copy") else ent
                elif arb_res.is_belief_revision and ent.key == (arb_res.superseded_key or meta.get("supersedes") or key):
                    backup_hot_superseded = ent.model_copy() if hasattr(ent, "model_copy") else ent

        # Checkpoint stanu dynamicznego L0/L1 dla symplektycznej kompensacji (Phi_undo)
        ttt_cp = self.ttt_layer.capture_checkpoint() if self.ttt_layer is not None and hasattr(self.ttt_layer, "capture_checkpoint") else None
        jepa_cp = self.jepa_buffer.capture_checkpoint() if self.jepa_buffer is not None and hasattr(self.jepa_buffer, "capture_checkpoint") else None

        # 2. Belief Revision Handling
        backup_old_item = None
        old_key_for_restore = None
        if arb_res.is_belief_revision:
            old_key = arb_res.superseded_key or meta.get("supersedes") or key
            old_key_for_restore = old_key
            meta["is_belief_revision"] = True
            meta["superseded_key"] = old_key

            # Atomically mark old key as superseded in KV store
            if old_key in existing_items:
                old_item = existing_items[old_key]
                backup_old_item = dict(old_item) if isinstance(old_item, dict) else old_item
                old_val = old_item.get("value", "")
                old_meta = dict(old_item.get("metadata") or {})
                old_meta["is_superseded"] = True
                old_meta["superseded_by"] = key
                old_meta["superseded_at"] = time.time()
                if self.kv_store is not None and hasattr(self.kv_store, "set_sync"):
                    try:
                        self.kv_store.set_sync(
                            old_key,
                            old_val,
                            confidence=float(old_item.get("confidence", 1.0)),
                            metadata=old_meta,
                            reason="superseded_by_belief_revision",
                        )
                        logger.debug("[WritePath] Atomically marked old key '%s' as superseded in KV store", old_key)
                    except Exception as exc:
                        logger.warning("[WritePath] Error marking old key superseded in KV: %s", exc)

            # Wire (old)-[:SUPERSEDED_BY]->(new) in graph store
            if str(old_key) != str(key) and self.graph_store is not None and hasattr(self.graph_store, "add_relation"):
                try:
                    self.graph_store.add_relation(
                        str(old_key),
                        "SUPERSEDED_BY",
                        str(key),
                        confidence=confidence,
                    )
                    logger.debug("[WritePath] Graph edge added: (%s)-[:SUPERSEDED_BY]->(%s)", old_key, key)
                except Exception as ge:
                    logger.debug("[WritePath] Error wiring SUPERSEDED_BY edge in graph: %s", ge)

            # Purge old key from hot working buffer
            purged_count = self.hot_buffer.purge(old_key)
            if purged_count > 0:
                logger.debug("[WritePath] Purged %d old entry for key '%s' from hot buffer", purged_count, old_key)

            # Purge old key from vector store to prevent orphan vectors
            if str(old_key) != str(key) and self.vector_store is not None:
                keys_to_purge = [str(old_key)]
                if not str(old_key).startswith("kv:"):
                    keys_to_purge.append(f"kv:{old_key}")
                else:
                    keys_to_purge.append(str(old_key)[3:])
                try:
                    for k_purge in keys_to_purge:
                        if hasattr(self.vector_store, "delete_by_upsert_key_sync"):
                            self.vector_store.delete_by_upsert_key_sync(k_purge)
                        elif hasattr(self.vector_store, "delete_sync"):
                            self.vector_store.delete_sync(k_purge)
                    logger.debug("[WritePath] Purged superseded vector points for key='%s'", old_key)
                except Exception as ve:
                    logger.warning("[WritePath] Vector delete error for %s: %s (flagging vector_tombstone_pending)", old_key, ve)
                    meta["vector_tombstone_pending"] = True
                    if self.kv_store is not None and hasattr(self.kv_store, "set_sync"):
                        try:
                            old_rec = self.kv_store.get_sync(str(old_key)) if hasattr(self.kv_store, "get_sync") else None
                            if old_rec:
                                o_val = old_rec.get("value", "") if isinstance(old_rec, dict) else getattr(old_rec, "value", "")
                                o_meta = dict((old_rec.get("metadata") if isinstance(old_rec, dict) else getattr(old_rec, "metadata", {})) or {})
                                o_meta["vector_tombstone_pending"] = True
                                o_meta["is_superseded"] = True
                                o_conf = float(old_rec.get("confidence", 1.0) if isinstance(old_rec, dict) else getattr(old_rec, "confidence", 1.0))
                                self.kv_store.set_sync(str(old_key), o_val, confidence=o_conf, metadata=o_meta, reason="vector_tombstone_pending")
                        except Exception as tomb_exc:
                            logger.error("[WritePath] Failed to set vector_tombstone_pending in KV for %s: %s", old_key, tomb_exc)

            log_conflict(
                log=logger,
                old_key=str(old_key),
                new_key=str(key),
                resolution="superseded",
                details="purged hot_buffer & vector point",
            )

        # 3. Same-key Value Revision Tracking
        existing_record = existing_items.get(key)
        if existing_record is not None and str(existing_record.get("value", "")) != str(value):
            prior_val = existing_record.get("value", "")
            prior_meta = dict(existing_record.get("metadata") or {})
            prior_rev = int(prior_meta.get("revision_count", 0))
            revision_count = prior_rev + 1

            meta["prior_value"] = str(prior_val)
            meta["revision_count"] = revision_count

            # Wire version transition in knowledge graph
            if self.graph_store is not None and hasattr(self.graph_store, "add_relation"):
                try:
                    self.graph_store.add_relation(
                        str(key),
                        "REVISED_FROM",
                        str(prior_val),
                        confidence=confidence,
                    )
                    self.graph_store.add_relation(
                        str(prior_val),
                        "TRANSITIONED_TO",
                        str(value),
                        confidence=confidence,
                    )
                    logger.debug("[WritePath] Graph version transition wired for '%s' (rev %d -> %d)", key, prior_rev, revision_count)
                except Exception as ge:
                    logger.debug("[WritePath] Error wiring version transition in graph: %s", ge)

        # 4. Persist to VerifiedKVStore with SHA-256 audit log
        persisted = False
        kv_savepoint_active = False
        if self.kv_store is not None and hasattr(self.kv_store, "set_sync"):
            if hasattr(self.kv_store, "savepoint"):
                try:
                    self.kv_store.savepoint("saga_tx")
                    kv_savepoint_active = True
                except Exception as sp_exc:
                    logger.debug("[WritePath] Could not create KV savepoint: %s", sp_exc)
            self.kv_store.set_sync(
                key,
                value,
                confidence=confidence,
                metadata=meta,
                reason=reason,
            )
            persisted = True
            logger.debug("[WritePath] Persisted key='%s' (conf=%.2f) to VerifiedKVStore", key, confidence)

        # 4b. Persist to Vector Store (Qdrant) via Epistemic Choke Point (Strict transactional check)
        if persisted and self.vector_store is not None and vector_record is not None:
            try:
                upsert_key = normalize_upsert_key(str(key))
                raw_txt = str(value) if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
                if hasattr(self.vector_store, "insert_sync"):
                    self.vector_store.insert_sync(vector_record, raw_text=raw_txt, upsert_key=upsert_key)
                elif hasattr(self.vector_store, "insert"):
                    res_ins = self.vector_store.insert(vector_record, raw_text=raw_txt, upsert_key=upsert_key)
                    if inspect.isawaitable(res_ins):
                        pass
                logger.debug("[WritePath] Vector indexed for key='%s' (upsert_key='%s')", key, upsert_key)
            except Exception as ve:
                logger.error("[WritePath] Transactional abort: Vector store insert failed for key='%s': %s", key, ve)
                persisted = False
                # Rollback KV store
                if kv_savepoint_active and hasattr(self.kv_store, "rollback_savepoint"):
                    try:
                        self.kv_store.rollback_savepoint("saga_tx")
                        kv_savepoint_active = False
                    except Exception as r_exc:
                        logger.error("[WritePath] Rollback savepoint failed: %s", r_exc)
                elif backup_existing_kv is not None and hasattr(self.kv_store, "set_sync"):
                    try:
                        self.kv_store.set_sync(
                            str(key),
                            backup_existing_kv.get("value"),
                            confidence=float(backup_existing_kv.get("confidence", 1.0)),
                            metadata=dict(backup_existing_kv.get("metadata") or {}),
                            reason="rollback_vector_failure",
                        )
                    except Exception as r_exc:
                        logger.error("[WritePath] Rollback restore of key '%s' in KV failed: %s", key, r_exc)
                elif self.kv_store is not None and hasattr(self.kv_store, "delete_sync"):
                    try:
                        self.kv_store.delete_sync(str(key))
                    except Exception as r_exc:
                        logger.error("[WritePath] Rollback delete of key '%s' in KV failed: %s", key, r_exc)

                # Restore superseded key in KV if belief revision (only if savepoint wasn't used)
                if not kv_savepoint_active and arb_res.is_belief_revision and backup_old_item is not None and old_key_for_restore and self.kv_store is not None and hasattr(self.kv_store, "set_sync"):
                    try:
                        self.kv_store.set_sync(
                            str(old_key_for_restore),
                            backup_old_item.get("value"),
                            confidence=float(backup_old_item.get("confidence", 1.0)),
                            metadata=dict(backup_old_item.get("metadata") or {}),
                            reason="rollback_vector_failure",
                        )
                    except Exception as r_exc:
                        logger.error("[WritePath] Rollback restore of superseded key '%s' failed: %s", old_key_for_restore, r_exc)

                if backup_hot_superseded is not None:
                    self.hot_buffer.append(backup_hot_superseded)
                if backup_hot_target is not None:
                    self.hot_buffer.append(backup_hot_target)

                # Symplektyczna kompensacja unitarna dynamicznego stanu L0/L1 (Phi_undo)
                if ttt_cp is not None and hasattr(self.ttt_layer, "restore_checkpoint"):
                    self.ttt_layer.restore_checkpoint(ttt_cp)
                if jepa_cp is not None and hasattr(self.jepa_buffer, "restore_checkpoint"):
                    self.jepa_buffer.restore_checkpoint(jepa_cp)
                raise

        # 5. Update Hot Working Memory Buffer
        self.hot_buffer.update(
            key=key,
            value=value,
            confidence=confidence,
            metadata=meta,
            source_type=meta.get("source_type", "user_explicit"),
        )

        # 6. Auto-insert Multi-Hop Graph Edges (with atomic Saga compensation on error)
        inserted_edges: List[Tuple[str, str, str]] = []
        try:
            edges_added = self._auto_insert_graph_edges(key, value, meta, confidence, inserted_edges=inserted_edges)
        except Exception as ge:
            logger.error("[WritePath] Transactional abort: Graph edge insertion failed for key='%s': %s", key, ge)
            persisted = False
            # 1. Rollback all inserted graph edges
            if self.graph_store is not None:
                for sub_e, pred_e, obj_e in inserted_edges:
                    try:
                        if callable(getattr(self.graph_store, "remove_edge_sync", None)):
                            self.graph_store.remove_edge_sync(sub_e, pred_e, obj_e)
                        elif callable(getattr(self.graph_store, "remove_edge", None)):
                            res = self.graph_store.remove_edge(sub_e, pred_e, obj_e)
                            if inspect.iscoroutine(res):
                                asyncio.create_task(res)
                    except Exception as re_exc:
                        logger.warning("[WritePath] Rollback remove_edge failed for (%s, %s, %s): %s", sub_e, pred_e, obj_e, re_exc)
                if callable(getattr(self.graph_store, "rollback_dual_node_record", None)) and callable(getattr(self.graph_store, "has_passage", None)):
                    try:
                        if self.graph_store.has_passage(str(key)):
                            self.graph_store.rollback_dual_node_record(str(key))
                    except Exception as rb_dn:
                        logger.warning("[WritePath] Rollback dual-node passage failed for %s: %s", key, rb_dn)

            # 2. Rollback vector store (defensively removing canonical and prefixed forms)
            if self.vector_store is not None:
                try:
                    k_canon = normalize_upsert_key(str(key))
                    if hasattr(self.vector_store, "delete_by_upsert_key_sync"):
                        self.vector_store.delete_by_upsert_key_sync(k_canon)
                        self.vector_store.delete_by_upsert_key_sync(f"kv:{k_canon}")
                    elif hasattr(self.vector_store, "delete_sync"):
                        self.vector_store.delete_sync(k_canon)
                        self.vector_store.delete_sync(f"kv:{k_canon}")
                except Exception as ve_exc:
                    logger.warning("[WritePath] Rollback vector delete failed for %s: %s", key, ve_exc)

            # 3. Rollback KV store (via savepoint rollback if available, zero audit pollution)
            if kv_savepoint_active and hasattr(self.kv_store, "rollback_savepoint"):
                try:
                    self.kv_store.rollback_savepoint("saga_tx")
                    kv_savepoint_active = False
                except Exception as rb_sp_err:
                    logger.error("[WritePath] Rollback savepoint failed: %s", rb_sp_err)
            elif self.kv_store is not None:
                if backup_existing_kv is not None and hasattr(self.kv_store, "set_sync"):
                    try:
                        self.kv_store.set_sync(
                            str(key),
                            backup_existing_kv.get("value"),
                            confidence=float(backup_existing_kv.get("confidence", 1.0)),
                            metadata=dict(backup_existing_kv.get("metadata") or {}),
                            reason="rollback_graph_failure",
                        )
                    except Exception as del_exc:
                        logger.warning("[WritePath] Rollback KV restore failed for %s: %s", key, del_exc)
                elif hasattr(self.kv_store, "delete_sync"):
                    try:
                        self.kv_store.delete_sync(str(key))
                    except Exception as del_exc:
                        logger.warning("[WritePath] Rollback delete_sync failed for %s: %s", key, del_exc)

            # 4. Rollback superseded record in KV if belief revision
            if arb_res.is_belief_revision and backup_old_item is not None and old_key_for_restore and str(old_key_for_restore) != str(key) and self.kv_store is not None and hasattr(self.kv_store, "set_sync"):
                try:
                    self.kv_store.set_sync(
                        str(old_key_for_restore),
                        backup_old_item.get("value"),
                        confidence=float(backup_old_item.get("confidence", 1.0)),
                        metadata=dict(backup_old_item.get("metadata") or {}),
                        reason="rollback_restore",
                    )
                    logger.debug("[WritePath] Restored superseded item for '%s' during rollback", old_key_for_restore)
                except Exception as res_exc:
                    logger.warning("[WritePath] Rollback restore failed for %s: %s", old_key_for_restore, res_exc)

            # 5. Rollback HotBuffer (purge new key and restore previous entries)
            self.hot_buffer.purge(str(key))
            if backup_hot_target is not None:
                self.hot_buffer.append(backup_hot_target)
            if backup_hot_superseded is not None:
                self.hot_buffer.append(backup_hot_superseded)

            # Symplektyczna kompensacja unitarna dynamicznego stanu L0/L1 (Phi_undo)
            if ttt_cp is not None and hasattr(self.ttt_layer, "restore_checkpoint"):
                self.ttt_layer.restore_checkpoint(ttt_cp)
            if jepa_cp is not None and hasattr(self.jepa_buffer, "restore_checkpoint"):
                self.jepa_buffer.restore_checkpoint(jepa_cp)
            raise

        if kv_savepoint_active and hasattr(self.kv_store, "release_savepoint"):
            try:
                self.kv_store.release_savepoint("saga_tx")
            except Exception as rel_exc:
                logger.debug("[WritePath] Release savepoint failed: %s", rel_exc)

        resp: Dict[str, Any] = {
            "status": "ok" if persisted else "error",
            "key": key,
            "persisted": persisted,
            "is_belief_revision": bool(arb_res.is_belief_revision),
            "revision_count": meta.get("revision_count", 0),
            "arbitration": arb_res,
            "edges_added": edges_added,
        }
        if arb_res.is_belief_revision:
            resp["action"] = "belief_revision"
            resp["superseded_key"] = arb_res.superseded_key
        if arb_res.is_conflict and not arb_res.is_belief_revision:
            resp["warning"] = "epistemic_conflict"
            resp["conflict_details"] = arb_res.conflict_details

        action_name = "belief_revision" if arb_res.is_belief_revision else ("revised" if meta.get("revision_count", 0) > 0 else "persisted")
        log_remember(
            log=logger,
            key=key,
            confidence=confidence,
            subject=meta.get("subject"),
            predicate=meta.get("predicate"),
            object_=meta.get("object"),
            revision=meta.get("revision_count", 0),
            edges_added=edges_added,
            vector_indexed=bool(persisted and self.vector_store is not None and vector_record is not None),
            kv_persisted=persisted,
            action=action_name,
        )

        return resp

    async def write_async(
        self,
        key: Any,
        value: Any = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        source_type: Optional[str] = None,
        reason: str = "write_path",
        session_id: Optional[str] = None,
        vector_record: Optional[Any] = None,
        replay_at: Optional[Any] = None,
        replay: bool = False,
    ) -> Dict[str, Any]:
        """Asynchronous wrapper for write() offloading I/O to thread pool."""
        return await asyncio.to_thread(
            self.write,
            key=key,
            value=value,
            confidence=confidence,
            metadata=metadata,
            source_type=source_type,
            reason=reason,
            session_id=session_id,
            vector_record=vector_record,
            replay_at=replay_at,
            replay=replay,
        )

