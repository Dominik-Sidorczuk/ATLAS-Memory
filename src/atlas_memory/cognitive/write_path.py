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

import logging
import re
import time
from typing import Any, Dict, Optional

from atlas_memory.causal.retro_causal_edge import is_pii_entity
from atlas_memory.cognitive.conflict_arbiter import ConflictArbiter
from atlas_memory.cognitive.hot_buffer import HotBufferPolicy
from atlas_memory.cognitive.models import (
    ArbitrationResult,
    clamp_conf,
)

logger = logging.getLogger(__name__)


class EpistemicWritePath:
    """
    Single Choke Point for ATLAS State Write Operations.
    """

    def __init__(
        self,
        kv_store: Optional[Any] = None,
        graph_store: Optional[Any] = None,
        hot_buffer: Optional[HotBufferPolicy] = None,
        conflict_arbiter: Optional[ConflictArbiter] = None,
    ) -> None:
        self.kv_store = kv_store
        self.graph_store = graph_store
        self.hot_buffer = hot_buffer if hot_buffer is not None else HotBufferPolicy()
        self.conflict_arbiter = conflict_arbiter if conflict_arbiter is not None else ConflictArbiter()

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
        if is_pii_entity(subject) or is_pii_entity(object):
            logger.debug("[WritePath] Ignored relation with PII")
            return False

        # Zapobiegaj samopętlom:
        if subject.strip() == object.strip():
            return False

        if self.graph_store is not None and hasattr(self.graph_store, "add_relation"):
            sub_c = subject.strip()[:60]
            pred_c = predicate.strip()[:40]
            obj_c = object.strip()[:80]
            call_kwargs = dict(kwargs)
            if session_id is not None:
                call_kwargs["session_id"] = session_id
            if timestamp is not None:
                call_kwargs["timestamp"] = timestamp
            try:
                self.graph_store.add_relation(
                    sub_c,
                    pred_c,
                    obj_c,
                    confidence=conf,
                    **call_kwargs,
                )
                return True
            except TypeError:
                try:
                    self.graph_store.add_relation(sub_c, pred_c, obj_c, confidence=conf)
                    return True
                except TypeError:
                    self.graph_store.add_relation(sub_c, pred_c, obj_c)
                    return True
        return False

    record_relation = commit_relation

    def _auto_insert_graph_edges(
        self,
        key: str,
        val: Any,
        metadata: Dict[str, Any],
        confidence: float,
    ) -> None:
        """Parses and inserts multi-hop relations into runtime knowledge graph."""
        if self.graph_store is None or not hasattr(self.graph_store, "add_relation"):
            return

        sub = metadata.get("subject") or key
        pred = metadata.get("predicate") or metadata.get("relation")
        obj = metadata.get("object") or metadata.get("target")

        # 1. Dict payload: add each key-value as edge
        if isinstance(val, dict):
            for p_k, p_v in val.items():
                try:
                    self.graph_store.add_relation(str(sub), str(p_k), str(p_v), confidence=confidence)
                    logger.info("[WritePath] Graph edge added: (%s)-[%s]->(%s)", sub, p_k, p_v)
                except Exception as ge:
                    logger.debug("[WritePath] Graph edge auto-insert error: %s", ge)

        # 2. Triple payload: pred and obj explicit
        elif pred and obj:
            try:
                self.graph_store.add_relation(str(sub), str(pred), str(obj), confidence=confidence)
                logger.info("[WritePath] Graph edge added: (%s)-[%s]->(%s)", sub, pred, obj)
            except Exception as ge:
                logger.debug("[WritePath] Graph edge auto-insert error: %s", ge)

        # 3. String arrow chain syntax: A -> B -> C or A → B → C or A => B
        elif isinstance(val, str):
            if "->" in val or "→" in val or "=>" in val:
                parts = [p.strip() for p in re.split(r"->|→|=>", val) if p.strip()]
                if parts:
                    if str(sub) != str(parts[0]):
                        try:
                            self.graph_store.add_relation(str(sub), "leads_to", str(parts[0]), confidence=confidence)
                            logger.info("[WritePath] Chain edge added: (%s)-[leads_to]->(%s)", sub, parts[0])
                        except Exception as ge:
                            logger.debug("[WritePath] Chain edge error: %s", ge)
                    for i in range(len(parts) - 1):
                        try:
                            self.graph_store.add_relation(str(parts[i]), "leads_to", str(parts[i + 1]), confidence=confidence)
                            logger.info("[WritePath] Chain edge added: (%s)-[leads_to]->(%s)", parts[i], parts[i + 1])
                        except Exception as ge:
                            logger.debug("[WritePath] Chain edge error: %s", ge)
            elif val.startswith("test:") or (len(val.strip()) < 64 and " " not in val.strip() and ":" in val):
                target_node = val.strip()
                rel_name = "relates_to"
                try:
                    self.graph_store.add_relation(str(sub), rel_name, target_node, confidence=confidence)
                    logger.info("[WritePath] Graph edge added: (%s)-[%s]->(%s)", sub, rel_name, target_node)
                except Exception as ge:
                    logger.debug("[WritePath] Graph edge auto-insert error: %s", ge)
            else:
                val_str = val.strip()
                if len(val_str) >= 25 or any(sep in val_str for sep in (":", "->", "→", "=>", "(", ")")):
                    try:
                        from atlas_memory.ingest.mnemosyne_ingest import MnemosyneIngestEngine
                        MnemosyneIngestEngine.extract_and_insert_multihop_relations(self.graph_store, str(sub), val)
                    except Exception as ex:
                        logger.debug("[WritePath] Runtime text graph extraction error: %s", ex)

    def write(
        self,
        key: str,
        value: Any,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        source_type: Optional[str] = None,
        reason: str = "write_path",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Orchestrates single-choke-point epistemic write:
        1. Conflict arbitration via ConflictArbiter.
        2. Belief revision supersession, graph wiring, and hot buffer purge.
        3. Same-key value revision tracking and version transition graph wiring.
        4. VerifiedKVStore persistence with SHA-256 audit log.
        5. Working memory hot buffer update.
        6. Multi-hop knowledge graph auto-insertion.
        """
        if not key:
            raise ValueError("Key is required for write operation.")

        confidence = clamp_conf(confidence)
        meta = dict(metadata or {})
        if source_type:
            meta["source_type"] = str(source_type)
        elif "source_type" not in meta:
            meta["source_type"] = "user_explicit"
        meta["confidence"] = confidence
        if session_id:
            meta["session_id"] = str(session_id)

        # Fetch existing items from KV store for arbitration
        existing_items: Dict[str, Any] = {}
        if self.kv_store is not None:
            if hasattr(self.kv_store, "get_all_sync"):
                try:
                    existing_items = self.kv_store.get_all_sync()
                except Exception as exc:
                    logger.debug("[WritePath] Error fetching get_all_sync: %s", exc)
            elif hasattr(self.kv_store, "get_all"):
                try:
                    existing_items = self.kv_store.get_all()
                except Exception as exc:
                    logger.debug("[WritePath] Error fetching get_all: %s", exc)

        # 1. Epistemic Conflict Arbitration
        arb_res: ArbitrationResult = self.conflict_arbiter.arbitrate(
            key=key,
            val=value,
            confidence=confidence,
            metadata=meta,
            existing_items=existing_items,
        )

        if arb_res.is_conflict and not arb_res.is_belief_revision:
            meta["conflict_warning"] = True
            if arb_res.conflict_details:
                meta["conflict_details"] = arb_res.conflict_details
            logger.warning("[WritePath Epistemic Guard] Conflict detected for key='%s': %s", key, arb_res.message)

        # 2. Belief Revision Handling
        if arb_res.is_belief_revision:
            old_key = arb_res.superseded_key or meta.get("supersedes") or key
            meta["is_belief_revision"] = True
            meta["superseded_key"] = old_key

            # Atomically mark old key as superseded in KV store
            if old_key in existing_items:
                old_item = existing_items[old_key]
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
                        logger.info("[WritePath] Atomically marked old key '%s' as superseded in KV store", old_key)
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
                    logger.info("[WritePath] Graph edge added: (%s)-[:SUPERSEDED_BY]->(%s)", old_key, key)
                except Exception as ge:
                    logger.debug("[WritePath] Error wiring SUPERSEDED_BY edge in graph: %s", ge)

            # Purge old key from hot working buffer
            purged_count = self.hot_buffer.purge(old_key)
            if purged_count > 0:
                logger.info("[WritePath] Purged %d old entry for key '%s' from hot buffer", purged_count, old_key)

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
                    logger.info("[WritePath] Graph version transition wired for '%s' (rev %d -> %d)", key, prior_rev, revision_count)
                except Exception as ge:
                    logger.debug("[WritePath] Error wiring version transition in graph: %s", ge)

        # 4. Persist to VerifiedKVStore with SHA-256 audit log
        persisted = False
        if self.kv_store is not None and hasattr(self.kv_store, "set_sync"):
            self.kv_store.set_sync(
                key,
                value,
                confidence=confidence,
                metadata=meta,
                reason=reason,
            )
            persisted = True
            logger.info("[WritePath] Persisted key='%s' (conf=%.2f) to VerifiedKVStore", key, confidence)

        # 5. Update Hot Working Memory Buffer
        self.hot_buffer.update(
            key=key,
            value=value,
            confidence=confidence,
            metadata=meta,
            source_type=meta.get("source_type", "user_explicit"),
        )

        # 6. Auto-insert Multi-Hop Graph Edges
        self._auto_insert_graph_edges(key, value, meta, confidence)

        resp: Dict[str, Any] = {
            "status": "ok" if persisted else "error",
            "key": key,
            "persisted": persisted,
            "is_belief_revision": bool(arb_res.is_belief_revision),
            "revision_count": meta.get("revision_count", 0),
            "arbitration": arb_res,
        }
        if arb_res.is_belief_revision:
            resp["action"] = "belief_revision"
            resp["superseded_key"] = arb_res.superseded_key
        if arb_res.is_conflict and not arb_res.is_belief_revision:
            resp["warning"] = "epistemic_conflict"
            resp["conflict_details"] = arb_res.conflict_details

        return resp

    async def write_async(
        self,
        key: str,
        value: Any,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        source_type: Optional[str] = None,
        reason: str = "write_path",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Asynchronous wrapper for write()."""
        return self.write(
            key=key,
            value=value,
            confidence=confidence,
            metadata=metadata,
            source_type=source_type,
            reason=reason,
            session_id=session_id,
        )
