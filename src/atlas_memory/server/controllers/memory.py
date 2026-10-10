"""Memory, KV, Prefetch & Audit RPC Controller for AtlasDaemon.

Encapsulates CRUD, Prefetch pipeline execution, observation commits,
task feedback, Mnemosyne sync, and cryptographic audit log verification.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

from atlas_memory.cognitive.models import HERMES_DEFAULT_SESSION, clamp_conf
from atlas_memory.core.models import MemoryRecord
from atlas_memory.ipc.wire_models import DEFAULT_SOCKET_PATH
from atlas_memory.l2_semantic.qdrant_store import normalize_upsert_key
from atlas_memory.l2_semantic.textnorm import normalize_stored_text
from atlas_memory.security.delete_guard import (
    DELETE_PROTECTED_PREFIXES,
    is_delete_protected,
)
from atlas_memory.server.telemetry import (
    build_daemon_stats_payload,
    clamp_rpc_limit,
)

logger = logging.getLogger("atlas.server.controllers.memory")

MAX_KEY_LENGTH = 1024
MAX_VALUE_LENGTH = 524288  # 512 KB
MAX_SESSION_ID_LENGTH = 256


def _validate_and_sanitize_kv_params(
    key: Any,
    val: Any,
    session_id: Any = None,
) -> tuple[str, str, str]:
    """Validates input lengths, patterns, and encoding to prevent state pollution and crashes."""
    if not key or not str(key).strip():
        raise ValueError("Key or subject is required and cannot be empty")
    k_str = str(key).strip()
    if "\x00" in k_str:
        raise ValueError("Key contains forbidden null bytes")
    if "../" in k_str or "..\\" in k_str:
        raise ValueError("Key contains forbidden path traversal patterns")
    if len(k_str) > MAX_KEY_LENGTH:
        raise ValueError(f"Key length ({len(k_str)}) exceeds maximum allowed ({MAX_KEY_LENGTH})")

    v_str = str(val if val is not None else "")
    if len(v_str) > MAX_VALUE_LENGTH:
        raise ValueError(f"Value length ({len(v_str)}) exceeds maximum allowed ({MAX_VALUE_LENGTH})")

    k_clean = normalize_stored_text(k_str)
    v_clean = normalize_stored_text(v_str)

    try:
        k_clean.encode("utf-8")
        v_clean.encode("utf-8")
    except UnicodeEncodeError as uee:
        raise ValueError(f"Unencodable unicode surrogate in parameters: {uee}") from uee

    s_clean = str(session_id or HERMES_DEFAULT_SESSION).strip()
    if len(s_clean) > MAX_SESSION_ID_LENGTH:
        raise ValueError(f"Session ID length ({len(s_clean)}) exceeds maximum allowed ({MAX_SESSION_ID_LENGTH})")
    try:
        s_clean.encode("utf-8")
    except UnicodeEncodeError as uee:
        raise ValueError(f"Unencodable unicode surrogate in session_id: {uee}") from uee

    return k_clean, v_clean, s_clean


class MemoryRpcControllerMixin:
    """Mixin providing memory, KV store, prefetch, and audit RPC handlers."""

    _DELETE_PROTECTED_PREFIXES = DELETE_PROTECTED_PREFIXES

    def _check_and_sync_memory_md(self, min_interval: float = 5.0) -> None:
        """Auto-sync all markdown files in ~/.hermes/memories/ if modification timestamp has changed.

        Throttled to avoid filesystem stat() storm on prefetch hot paths.
        """
        now = time.monotonic()
        last_check = getattr(self, "_last_memory_md_sync_check", 0.0)
        if now - last_check < min_interval:
            return
        self._last_memory_md_sync_check = now

        sock_path = getattr(self, "socket_path", None)
        if sock_path and str(sock_path) != str(DEFAULT_SOCKET_PATH):
            return
        from atlas_memory.ingest.mnemosyne_ingest import DEFAULT_HERMES_MEMORY_MD_PATH, MnemosyneIngestEngine
        memories_dir = DEFAULT_HERMES_MEMORY_MD_PATH.parent
        if not memories_dir.exists():
            return
        try:
            md_files = [f for f in memories_dir.glob("*.md") if not f.name.endswith(".lock")]
            if not md_files:
                return
            latest_mtime = max(f.stat().st_mtime for f in md_files)
            last_mtime = getattr(self, "_last_memory_md_mtime", 0.0)
            if latest_mtime > last_mtime:
                self._last_memory_md_mtime = latest_mtime
                engine = getattr(self, "engine", None)
                if engine is not None:
                    ingester = MnemosyneIngestEngine()
                    total_records = 0
                    for md_path in md_files:
                        try:
                            sync_res = ingester.sync_memory_md_into_engine(engine, path=str(md_path))
                            total_records += sync_res.get("records_imported", 0)
                        except Exception as file_exc:
                            logger.debug("Failed syncing %s: %s", md_path.name, file_exc)
                    if total_records > 0:
                        logger.info("Auto-synced %d memories from ~/.hermes/memories/ (mtime: %s)", total_records, latest_mtime)
        except Exception as exc:
            logger.debug("Auto-sync memories.md check failed: %s", exc)

    async def _start_memory_md_watcher(self) -> None:
        """Asynchronous non-blocking watcher for external modifications in ~/.hermes/memories/*.md."""
        sock_path = getattr(self, "socket_path", None)
        if sock_path and str(sock_path) != str(DEFAULT_SOCKET_PATH):
            return
        if not hasattr(self, "_last_memory_md_mtime"):
            from atlas_memory.ingest.mnemosyne_ingest import DEFAULT_HERMES_MEMORY_MD_PATH
            mem_dir = DEFAULT_HERMES_MEMORY_MD_PATH.parent
            try:
                if mem_dir.exists():
                    md_files = [f for f in mem_dir.glob("*.md") if not f.name.endswith(".lock")]
                    if md_files:
                        self._last_memory_md_mtime = max(f.stat().st_mtime for f in md_files)
            except Exception as exc:
                logger.debug("Initial memory md mtime scan failed: %s", exc)
        while getattr(self, "_serving", False):
            try:
                await asyncio.sleep(10.0)
                await asyncio.to_thread(self._check_and_sync_memory_md, 8.0)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("Memory MD watcher iteration failed: %s", exc)
 
    _memory_md_watcher_loop = _start_memory_md_watcher

    async def _handle_ping(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simple ping/pong heartbeat method."""
        return {"status": "ok"}

    async def _handle_prefetch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Prefetch memory records for query/session with full epistemic formatting and content deduplication."""
        params_dict = dict(params)
        if "limit" in params_dict:
            params_dict["limit"] = clamp_rpc_limit(params_dict.get("limit", 15))
        pipeline = getattr(self, "retrieval_pipeline", None)
        if pipeline is None:
            return {"status": "error", "error": "no_retrieval_pipeline"}
        return await pipeline.execute_prefetch(
            params_dict,
            engine=getattr(self, "engine", None),
            orchestrator=getattr(self, "orchestrator", None),
            hot_buffer=getattr(self, "_hot_kv_buffer", getattr(self, "hot_buffer", None)),
        )

    async def _handle_atlas_recall(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """V152-CLOSURE (ZADANIE 1): kanoniczny alias recall pod nazwą narzędzia Hermesowego.

        Konsumenci (hermes/tools.py``handle_atlas_recall``, zakres testów
        ``test_v152_closure_*.py``) mówią ``atlas_recall``; pipeline retrievalu
        nazywa się ``prefetch`` i zwraca wiersze kluczem ``records``. Ten
        adapter deleguje do ``prefetch`` i rekonfiguruje odpowiedź do
        ravnoważnego, interpreterowalnego kształtu ``{"hits": [...]}``, w którym
        ``score`` = fuzja relevancji (metadata.score / vector_score), a
        ``source`` = ścieżka zasilania (vector/hybrid/vector_hybrid).
        """
        params_dict = dict(params or {})
        top_k = params_dict.get("top_k")
        if top_k is not None and "limit" not in params_dict:
            params_dict["limit"] = top_k
        res = await self._handle_prefetch(params_dict)
        if not isinstance(res, dict) or res.get("status") == "error":
            return res if isinstance(res, dict) else {"status": "error", "error": "no_prefetch_result"}
        hits: list[Dict[str, Any]] = []
        for rec in res.get("records") or []:
            if not isinstance(rec, dict):
                continue
            meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
            dense = float(meta.get("dense_score") or 0.0)
            sparse = float(meta.get("sparse_score") or 0.0)
            fusion = float(meta.get("score") or 0.0)
            score = max(fusion, float(rec.get("vector_score") or 0.0), dense)
            src = str(rec.get("source") or "")
            if dense > 0.0 and sparse > 0.0:
                src = "vector_hybrid"
            elif dense > 0.0:
                src = "vector"
            hits.append(
                {
                    "key": rec.get("key") or rec.get("subject") or "",
                    "subject": rec.get("subject") or "",
                    "predicate": meta.get("predicate") or rec.get("predicate") or "",
                    "object": rec.get("object") or rec.get("content") or "",
                    "source": src,
                    "match_type": src,
                    "score": score,
                    "dense_score": dense,
                    "sparse_score": sparse,
                    "graph_score": float(meta.get("graph_score") or 0.0),
                    "confidence": rec.get("confidence"),
                    "timestamp": rec.get("timestamp"),
                }
            )
        return {"status": "ok", "query": params_dict.get("query", ""), "hits": hits, "count": len(hits)}

    async def _persist_state_variable(
        self,
        key: str,
        value: Any,
        session_id: str,
        confidence: float,
        metadata: Dict[str, Any],
        source_type: Optional[str] = None,
        reason: str = "rpc_set",
        task_success: bool = False,
        action_name: str = "remember",
        action_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Unified choke point for persisting state variables, indexing vector points, updating hot buffer, and recording transitions."""
        engine = getattr(self, "engine", None)
        if task_success and engine is not None and hasattr(engine, "kv") and engine.kv is not None:
            existing = engine.kv.get_sync(key) if hasattr(engine.kv, "get_sync") else None
            if existing:
                existing_conf = float(existing.get("confidence", confidence))
                confidence = clamp_conf(existing_conf + 0.05)
                reason = "task_feedback:success"
                metadata["confidence"] = confidence
                metadata["task_feedback"] = "success"

        write_path = getattr(self, "write_path", None)
        if write_path is None:
            return {"status": "error", "error": "no_write_path"}

        vector_record = None
        vector_store = getattr(write_path, "vector_store", None)
        if vector_store is None and engine is not None:
            vector_store = getattr(engine, "vector_store", None)
            if vector_store is not None:
                write_path.vector_store = vector_store

        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        if vector_store is not None and text and text.strip():
            record = MemoryRecord(
                subject=f"kv:{key}",
                predicate="state_is",
                object=text,
                confidence=float(confidence if isinstance(confidence, (int, float)) else 1.0),
                timestamp=time.time(),
                is_state_variable=True,
                metadata={"kv_key": str(key), "origin": "daemon_set"},
            )
            clean_key = str(key).replace(":", " ").replace("_", " ")
            embed_text = f"{clean_key}: {text}" if clean_key not in text else text
            if hasattr(vector_store, "_encode_text") and hasattr(vector_store, "dimension"):
                try:
                    enc_vec = vector_store._encode_text(embed_text)
                    if enc_vec and len(enc_vec) == vector_store.dimension:
                        record.vector = enc_vec
                except Exception as enc_err:
                    logger.debug("vector pre-encoding failed: %s", enc_err)
            vector_record = record

        try:
            res = await write_path.write_async(
                key=key,
                value=value,
                confidence=confidence,
                metadata=metadata,
                source_type=source_type,
                reason=reason,
                session_id=session_id,
                vector_record=vector_record,
            )
        except Exception as exc:
            logger.error("[MemoryController] Persist state variable aborted with rollback: %s", exc)
            return {"status": "error", "error": str(exc), "persisted": False}

        if (
            isinstance(res, dict)
            and res.get("status") in ("ok", "success")
            and res.get("persisted", True) is not False
        ):
            if vector_record is None:
                await self._index_kv_vector(str(key), value, confidence)
            hot_buf = getattr(self, "_hot_kv_buffer", getattr(self, "hot_buffer", None))
            if hot_buf is not None:
                hb_meta = dict(metadata or {})
                if session_id and "session_id" not in hb_meta:
                    hb_meta["session_id"] = session_id
                hot_buf.append({
                    "key": str(key),
                    "value": str(value),
                    "timestamp": time.time(),
                    "confidence": confidence,
                    "metadata": hb_meta,
                    "source_type": hb_meta.get("source_type", "user_explicit"),
                })

        if engine is not None and hasattr(engine, "record_mental_transition"):
            try:
                from atlas_memory.core.models import ActionPlan
                act_params = action_params if action_params is not None else {"key": key, "value": str(value)[:100]}
                action = ActionPlan(name=action_name, parameters=act_params)
                engine.record_mental_transition(action, session_id=session_id)
            except Exception as exc:
                logger.debug("Daemon %s record_mental_transition error: %s", action_name, exc)

        return res

    async def _handle_set(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Set a key-value pair in ATLAS Verified KV Store via EpistemicWritePath."""
        raw_key = params.get("key") or params.get("subject")
        raw_val = params.get("value") or params.get("object") or params.get("val") or params.get("target") or ""
        raw_sess = params.get("session_id") or HERMES_DEFAULT_SESSION

        key, val, session_id = _validate_and_sanitize_kv_params(raw_key, raw_val, raw_sess)
        confidence = clamp_conf(params.get("confidence", 1.0))
        metadata = dict(params.get("metadata") or {})
        for field in ("predicate", "relation", "target", "object", "subject"):
            if field in params and field not in metadata:
                metadata[field] = params[field]
        source_type = params.get("source_type")
        if source_type:
            metadata["source_type"] = str(source_type)
        metadata["confidence"] = confidence
        metadata["session_id"] = str(session_id)

        reason = params.get("reason", "rpc_set")
        task_success = bool(params.get("task_success") or params.get("success") or (params.get("outcome") == "success"))

        return await self._persist_state_variable(
            key=key,
            value=val,
            session_id=session_id,
            confidence=confidence,
            metadata=metadata,
            source_type=source_type,
            reason=reason,
            task_success=task_success,
            action_name="remember",
            action_params={"key": key, "value": str(val)[:100]},
        )

    async def _handle_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get a key-value record from ATLAS Verified KV Store."""
        key = params.get("key")
        if not key:
            raise ValueError("Key is required for get")
        engine = getattr(self, "engine", None)
        if engine is not None and hasattr(engine, "kv") and engine.kv is not None:
            state = await engine.kv.get_state(key)
            logger.debug("[IPC:GET] key='%s' -> found=%s", key, state is not None)
            return {"status": "ok", "key": key, "found": state is not None, "state": state}
        return {"status": "error", "error": "no_engine_available"}

    async def _handle_delete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a key from ATLAS Verified KV Store with deletion guards and SHA-256 audit log."""
        key = params.get("key")
        if not key:
            raise ValueError("Key is required for delete")

        is_prot, prefix = is_delete_protected(str(key))
        if is_prot:
            logger.warning("[RPC] delete REJECTED for protected key='%s' (prefix '%s')", key, prefix)
            return {
                "status": "rejected",
                "key": key,
                "deleted": False,
                "found": False,
                "reason": "protected_key",
                "message": f"Key '{key}' is protected (prefix '{prefix}'). Use soft tombstone or explicit ADR override.",
            }

        engine = getattr(self, "engine", None)
        if engine is not None and hasattr(engine, "kv") and engine.kv is not None:
            kv = engine.kv
            value_before_delete = await self._kv_value_text(str(key))
            if hasattr(kv, "delete_state"):
                deleted = await kv.delete_state(key)
            else:
                with kv.ensure_connection() as conn:
                    cursor = conn.execute("DELETE FROM state_variables WHERE key = ?", (key,))
                    deleted = bool(cursor.rowcount > 0)
            vector_deleted = await self._delete_kv_vector(str(key)) if deleted else 0
            if deleted and vector_deleted == 0:
                vector_deleted = await self._delete_kv_vector_by_text(str(key), value_before_delete)
            logger.info("[COG:DELETE] key='%s' -> deleted=%s vector_points=%d", key, deleted, vector_deleted)
            return {
                "status": "ok",
                "key": key,
                "deleted": deleted,
                "found": deleted,
                "vector_points_deleted": vector_deleted,
            }
        return {"status": "error", "error": "no_engine_available"}

    async def _kv_value_text(self, key: str) -> Optional[str]:
        """W3 (V72.2): treść rekordu KV jako TEKST — czytać PRZED kasowaniem wiersza."""
        engine = getattr(self, "engine", None)
        kv = getattr(engine, "kv", None) if engine is not None else None
        if kv is None:
            return None
        try:
            row: Any = None
            if hasattr(kv, "get_sync"):
                row = kv.get_sync(key)
            elif hasattr(kv, "get"):
                row = await kv.get(key)
            value = row.get("value") if isinstance(row, dict) else row
            if value is None:
                return None
            return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        except Exception as exc:
            logger.debug("odczyt treści KV przed delete nie powiódł się dla '%s': %s", key, exc)
            return None

    async def _delete_kv_vector_by_text(self, key: str, value_text: Optional[str] = None) -> int:
        """W3 (V72.2): fallback kasowania punktu po treści rekordu KV."""
        engine = getattr(self, "engine", None)
        vector_store = getattr(engine, "vector_store", None) if engine is not None else None
        remover = getattr(vector_store, "delete_by_text_sync", None)
        if not callable(remover):
            return 0
        try:
            value: Any = value_text
            if value is None:
                kv = getattr(engine, "kv", None)
                if kv is not None and hasattr(kv, "get_sync"):
                    row = kv.get_sync(key)
                    if isinstance(row, dict):
                        value = row.get("value")
            if value is None:
                return 0
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
            return int(remover(text) or 0)
        except Exception as exc:
            logger.debug("delete wektora po treści nie powiódł się dla '%s': %s", key, exc)
            return 0

    async def _index_kv_vector(self, key: str, value: Any, confidence: float) -> Optional[str]:
        """V72.2 (W4): indeksuje rekord KV w magazynie wektorowym pod upsert_key='kv:<klucz>'."""
        engine = getattr(self, "engine", None)
        vector_store = getattr(engine, "vector_store", None) if engine is not None else None
        if vector_store is None:
            return None
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        if not text or not text.strip():
            return None
        try:
            record = MemoryRecord(
                subject=f"kv:{key}",
                predicate="state_is",
                object=text,
                confidence=float(confidence if isinstance(confidence, (int, float)) else 1.0),
                timestamp=time.time(),
                is_state_variable=True,
                metadata={"kv_key": key, "origin": "daemon_set"},
            )
            upsert_key = normalize_upsert_key(key)
            clean_key = key.replace(":", " ").replace("_", " ")
            embed_text = f"{clean_key}: {text}" if clean_key not in text else text
            if hasattr(vector_store, "_encode_text") and hasattr(vector_store, "dimension"):
                try:
                    enc_vec = vector_store._encode_text(embed_text)
                    if enc_vec and len(enc_vec) == vector_store.dimension:
                        record.vector = enc_vec
                except Exception as enc_err:
                    logger.debug("vector pre-encoding failed: %s", enc_err)
            if hasattr(vector_store, "insert_sync"):
                return str(vector_store.insert_sync(record, raw_text=text, upsert_key=upsert_key))
            if hasattr(vector_store, "insert"):
                return str(await vector_store.insert(record, raw_text=text, upsert_key=upsert_key))
        except Exception as exc:
            logger.warning("wektor KV nie zaindeksowany dla klucza '%s': %s", key, exc)
        return None

    async def _delete_kv_vector(self, key: str) -> int:
        """V72.2 (W3): kasuje punkt wektorowy rekordu KV (obsługuje kv:<klucz> oraz <klucz>)."""
        engine = getattr(self, "engine", None)
        vector_store = getattr(engine, "vector_store", None) if engine is not None else None
        if vector_store is None:
            return 0
        remover = getattr(vector_store, "delete_by_upsert_key", None)
        remover_sync = getattr(vector_store, "delete_by_upsert_key_sync", None)
        keys_to_del = [f"kv:{key}", str(key)] if not str(key).startswith("kv:") else [str(key), str(key)[3:]]
        total = 0
        for k in keys_to_del:
            try:
                if callable(remover):
                    res = await remover(k)
                    total += int(res or 0)
                elif callable(remover_sync):
                    total += int(remover_sync(k) or 0)
            except Exception as exc:
                logger.warning("nie udało się skasować wektora KV dla klucza '%s': %s", k, exc)
        return total

    async def _handle_sync_memory_file(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Syncs Hermes MEMORY.md file into ATLAS Vault."""
        path = params.get("path")
        from atlas_memory.ingest.mnemosyne_ingest import MnemosyneIngestEngine
        ingester = MnemosyneIngestEngine()
        engine = getattr(self, "engine", None)
        if engine is not None:
            res = ingester.sync_memory_md_into_engine(engine, path=path)
            logger.info("[RPC] sync_memory_file -> %s", res)
            return res
        return {"status": "error", "error": "no_engine_available"}

    async def _handle_commit_observation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Commit an observation / fact into the ATLAS Vault via EpistemicWritePath."""
        subject = str(params.get("subject") or "").strip()
        if not subject:
            return {"status": "error", "error": "empty_subject", "message": "Subject cannot be empty"}

        predicate = str(params.get("predicate") or "observation").strip() or "observation"
        raw_obj = str(params.get("object") or "")
        raw_sess = params.get("session_id", HERMES_DEFAULT_SESSION)

        var_key, obj, session_id = _validate_and_sanitize_kv_params(f"{subject}:{predicate}", raw_obj, raw_sess)
        confidence = clamp_conf(params.get("confidence", 1.0))
        source = params.get("source", "user_explicit")
        task_success = bool(params.get("task_success") or params.get("success") or (params.get("outcome") == "success"))

        metadata: Dict[str, Any] = {
            "confidence": confidence,
            "source": source,
            "source_type": source,
            "session_id": session_id,
            "predicate": predicate,
            "subject": subject,
        }
        if params.get("task_success") is not None:
            metadata["task_success"] = bool(params["task_success"])

        res = await self._persist_state_variable(
            key=var_key,
            value=obj,
            session_id=session_id,
            confidence=confidence,
            metadata=metadata,
            source_type=source,
            reason="commit_observation",
            task_success=task_success,
            action_name=f"observe:{predicate}",
            action_params={"subject": subject, "object": obj},
        )

        if isinstance(res, dict) and res.get("status") == "error":
            return res

        logger.debug("[IPC:COMMIT] commit_observation key='%s' -> persisted=%s", var_key, res.get("persisted", True))
        return {
            "status": "ok",
            "committed": True,
            "subject": subject,
            "key": var_key,
            "confidence": confidence,
            "is_belief_revision": res.get("is_belief_revision", False),
        }

    async def _handle_task_feedback(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Dedicated RPC handler for closed-loop task feedback (G8). Updates record confidence."""
        key = str(params.get("key") or params.get("subject") or "").strip()
        if not key:
            return {"status": "error", "error": "empty_key", "message": "Key cannot be empty"}

        task_success = bool(params.get("task_success") or params.get("success") or (params.get("outcome") == "success") or params.get("task_success", True))
        if "task_success" in params and not params["task_success"]:
            task_success = False
        elif "success" in params and not params["success"]:
            task_success = False
        elif params.get("outcome") and params.get("outcome") != "success":
            task_success = False

        try:
            delta = float(params.get("delta", 0.05))
        except (ValueError, TypeError):
            delta = 0.05

        engine = getattr(self, "engine", None)
        if engine is None or not hasattr(engine, "kv") or engine.kv is None:
            return {"status": "error", "error": "no_kv_available"}

        existing = engine.kv.get_sync(key) if hasattr(engine.kv, "get_sync") else None
        if not existing:
            return {"status": "error", "error": "key_not_found", "message": f"Key '{key}' not found in KV store"}

        existing_conf = float(existing.get("confidence", 1.0))
        new_conf = clamp_conf(existing_conf + delta if task_success else existing_conf - delta)
        val = existing.get("value", "")
        meta = existing.get("metadata", {})
        meta = dict(meta) if isinstance(meta, dict) else {}
        meta["confidence"] = new_conf
        meta["task_feedback"] = "success" if task_success else "failure"

        await self._persist_state_variable(
            key=key,
            value=val,
            session_id=str(params.get("session_id") or HERMES_DEFAULT_SESSION),
            confidence=new_conf,
            metadata=meta,
            source_type=meta.get("source_type", "user_explicit"),
            reason="task_feedback:success" if task_success else "task_feedback:failure",
            task_success=False,
            action_name="task_feedback",
            action_params={"key": key, "task_success": task_success},
        )

        logger.info("[COG:FEEDBACK] key='%s' success=%s conf=%.2f->%.2f", key, task_success, existing_conf, new_conf)
        return {
            "status": "ok",
            "key": key,
            "previous_confidence": existing_conf,
            "confidence": new_conf,
            "task_success": task_success,
        }

    async def _handle_sync_mnemosyne(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Syncs all external Mnemosyne SQLite databases into ATLAS Vault."""
        engine = getattr(self, "engine", None)
        if engine is not None and hasattr(engine, "sync_all_mnemosyne_vaults"):
            res = engine.sync_all_mnemosyne_vaults()
            logger.info("[RPC] sync_mnemosyne -> %s", res)
            return res
        return {"status": "error", "error": "no_engine_active"}

    async def _handle_get_stats(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return real-time counts and memory vault status."""
        return await build_daemon_stats_payload(self, params)

    async def _handle_sync_turn(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sync conversation turn with heuristic SPO fact extraction."""
        user_content = params.get("user_content") or params.get("user_message", "")
        assistant_content = params.get("assistant_content") or params.get("assistant_response", "")
        turn_context = params.get("turn_context", {})
        if not user_content and turn_context:
            user_content = turn_context.get("user_message", "")
            assistant_content = turn_context.get("assistant_response", "")
        session_id = params.get("session_id", HERMES_DEFAULT_SESSION)

        extracted_count = 0
        combined_text = f"{user_content}\n{assistant_content}".strip()
        engine = getattr(self, "engine", None)
        wp = getattr(self, "write_path", None) or (getattr(engine, "write_path", None) if engine else None)

        async def _record_turn_fact(text: str, conf: float, source: str) -> bool:
            words = text.split()
            clean_subj = re.sub(r"[^a-zA-Z0-9_:]", "_", " ".join(words[:4])).strip("_") if words else "turn_fact"
            turn_key = f"turn:{clean_subj}"
            if len(turn_key) > 120:
                turn_key = f"{turn_key[:64]}_{hashlib.sha256(turn_key.encode('utf-8')).hexdigest()[:16]}"
            meta = {"session_id": session_id, "origin": "shadow_turn"}
            if wp is not None:
                await wp.write_async(
                    key=turn_key,
                    value=text,
                    confidence=conf,
                    metadata=meta,
                    source_type=source,
                    reason="sync_turn",
                    session_id=session_id,
                )
                return True
            return False

        turn_segments = [
            (user_content, "user_explicit", 0.90),
            (assistant_content, "agent_inference", 0.60),
        ]
        for seg_text, seg_source, seg_conf in turn_segments:
            if not seg_text:
                continue
            for line in seg_text.splitlines():
                ln = line.strip()
                if len(ln.split()) >= 3 and not ln.startswith(("#", "//", "```", "<")):
                    if await _record_turn_fact(ln, seg_conf, seg_source):
                        extracted_count += 1

        if extracted_count == 0 and combined_text:
            lines = [ln.strip() for ln in combined_text.splitlines() if ln.strip()]
            first_line = lines[0] if lines else combined_text[:80]
            fb_source = "user_explicit" if str(user_content or "").strip() else "agent_inference"
            fb_conf = 0.85 if fb_source == "user_explicit" else 0.60
            if await _record_turn_fact(first_line, fb_conf, fb_source):
                extracted_count += 1

        if engine is not None and hasattr(engine, "record_mental_transition"):
            try:
                from atlas_memory.core.models import ActionPlan
                action = ActionPlan(name="sync_turn", parameters={"facts_count": extracted_count, "session_id": session_id})
                engine.record_mental_transition(action, session_id=session_id)
            except Exception as exc:
                logger.debug("Daemon sync_turn record_mental_transition error: %s", exc)

        logger.info("[RPC] sync_turn session=%s -> extracted=%d facts", session_id, extracted_count)
        return {"status": "ok", "synced": True, "extracted_facts": extracted_count, "session_id": session_id}

    async def _handle_checkpoint_wal(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """FINDING-072: Wykonuje kontrolowany WAL checkpoint bazy głównej i audytu."""
        engine = getattr(self, "engine", None)
        if engine is None or not hasattr(engine, "kv") or engine.kv is None:
            return {"status": "error", "error": "no_engine_available"}
        truncate = bool(params.get("truncate", True))
        try:
            res = await engine.kv.checkpoint_wal(truncate=truncate)
            return {"status": "ok", "result": res}
        except Exception as exc:
            logger.warning("RPC checkpoint_wal failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    async def _handle_verify_audit_log(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Verify cryptographic integrity of SHA-256 audit log."""
        engine = getattr(self, "engine", None)
        if engine is None or not hasattr(engine, "kv") or engine.kv is None:
            return {"status": "error", "error": "no_engine_available", "valid": False, "checked_entries": 0}

        deep = bool(params.get("deep", True))
        kv = engine.kv
        if not hasattr(kv, "verify_audit_log"):
            return {"status": "error", "error": "verify_audit_log_not_supported", "valid": False, "checked_entries": 0}

        if params.get("heal") and hasattr(kv, "heal_audit_log_chain"):
            heal_res = kv.heal_audit_log_chain()
            if asyncio.iscoroutine(heal_res):
                await heal_res

        res = kv.verify_audit_log(deep=deep)
        if asyncio.iscoroutine(res):
            res = await res

        count = 0
        if hasattr(kv, "count_audit_entries_sync"):
            try:
                count = int(kv.count_audit_entries_sync())
            except Exception as c_err:
                logger.debug("Failed to count state_audit_log entries via count_audit_entries_sync: %s", c_err)

        if count == 0 and hasattr(kv, "_ensure_conn"):
            try:
                conn = kv._ensure_conn()
                with contextlib.closing(conn.cursor()) as cursor:
                    try:
                        cursor.execute("SELECT COUNT(*) FROM state_audit_log")
                    except Exception:
                        cursor.execute("SELECT COUNT(*) FROM audit.state_audit_log")
                    row = cursor.fetchone()
                    if row:
                        count = int(row[0])
            except Exception as c_err:
                logger.debug("Direct count of state_audit_log via connection failed: %s", c_err)

        if count == 0 and hasattr(kv, "audit_db_path") and kv.audit_db_path and kv.audit_db_path != ":memory:":
            try:
                audit_path = Path(kv.audit_db_path)
                if audit_path.exists():
                    with sqlite3.connect(str(audit_path)) as aconn:
                        row = aconn.execute("SELECT COUNT(*) FROM state_audit_log").fetchone()
                        if row:
                            count = int(row[0])
            except Exception as c_err:
                logger.debug("Audit DB direct file count failed: %s", c_err)

        broken_seq = 0
        if isinstance(res, tuple):
            is_valid = bool(res[0])
            if is_valid:
                if count == 0 and len(res) > 1 and res[1] is not None and int(res[1]) > 0:
                    count = int(res[1])
            else:
                if len(res) > 1 and res[1] is not None:
                    broken_seq = int(res[1])
        elif isinstance(res, bool):
            is_valid = res
        elif isinstance(res, dict):
            is_valid = bool(res.get("valid", True))
            if "checked_entries" in res and count == 0:
                count = int(res["checked_entries"])
            if "broken_seq" in res:
                broken_seq = int(res["broken_seq"])
            elif "tampered_seq" in res:
                broken_seq = int(res["tampered_seq"])
        elif isinstance(res, int):
            is_valid = True
            if count == 0:
                count = res
        else:
            is_valid = bool(res)

        if count == 0:
            logger.info("[RPC] verify_audit_log (deep=%s) -> empty_chain (checked_entries=0)", deep)
            return {
                "status": "empty_chain",
                "valid": False,
                "checked_entries": 0,
                "tampered": False,
                "message": "Audit log chain is empty; no cryptographic entries verified.",
            }

        tampered = not is_valid
        logger.info(
            "[RPC] verify_audit_log (deep=%s) -> valid=%s, checked_entries=%d, tampered=%s, broken_seq=%d",
            deep,
            is_valid,
            count,
            tampered,
            broken_seq,
        )
        response: Dict[str, Any] = {
            "status": "ok",
            "valid": is_valid,
            "checked_entries": count,
            "tampered": tampered,
        }
        if tampered and broken_seq:
            response["tampered_seq"] = broken_seq
            response["broken_seq"] = broken_seq

        return response


__all__ = [
    "MemoryRpcControllerMixin",
    "MAX_KEY_LENGTH",
    "MAX_VALUE_LENGTH",
    "MAX_SESSION_ID_LENGTH",
    "_validate_and_sanitize_kv_params",
]
