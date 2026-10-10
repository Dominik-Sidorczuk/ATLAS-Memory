from __future__ import annotations

import inspect
import json
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    HybridMemoryEngine = Any
    from atlas_memory.l3_procedural.sleep_baker import StandardProcedure

try:
    from atlas_memory.l3_procedural.skill_compiler import SafetyViolationError, compile_sop_to_skill
except ImportError:
    SafetyViolationError = Exception  # type: ignore — optional-dep fallback: brak hermes-core w tym srodowisku
    compile_sop_to_skill = None  # type: ignore — optional-dep fallback; consumidor guarduje 'is None'

from atlas_memory.core.models import ConsolidationStats

logger = logging.getLogger(__name__)


class PrefixCacheGuard:
    """
    Krok B: Prefix Cache Guard.
    
    Generuje deterministyczny, niezmienny blok stanu wstrzykiwany do system_prompt
    na samym początku sesji. Zapobiega unieważnieniu bufora KV Cache (prefix cache)
    w lokalnych silnikach inferencji (llama-cpp-python, vLLM, Ollama).
    """

    @staticmethod
    def build_immutable_prefix(
        verified_state: Dict[str, Any],
        system_rules: Optional[List[str]] = None,
        base_entities: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Tworzy deterministyczny ciąg znaków promptu bazowego z posortowanymi kluczami.
        """
        lines = [
            "# === HERMES AGENT IMMUTABLE KERNEL STATE ===",
            "<!-- KV-CACHE-PREFIX-GUARD: DO NOT MODIFY DURING SESSION -->",
        ]

        if system_rules:
            lines.append("## System Constraints:")
            for rule in sorted(system_rules):
                lines.append(f"- {rule}")

        if verified_state:
            lines.append("## Verified Source-of-Truth State:")
            # Sortowanie kluczy dla deterministycznego cache hitu
            for k in sorted(verified_state.keys()):
                v = verified_state[k]
                val_repr = v.get("value") if isinstance(v, dict) else v
                lines.append(f"- {k}: {json.dumps(val_repr, ensure_ascii=False, default=str)}")

        if base_entities:
            lines.append("## Registered Canonical Entities:")
            for item in sorted(base_entities, key=lambda x: x.get("id", "")):
                lines.append(f"- {item.get('id')}: {item.get('name')} (aliases: {', '.join(sorted(item.get('aliases', [])))})")

        lines.append("# === END OF IMMUTABLE KERNEL STATE ===\n")
        return "\n".join(lines)


class HermesSessionHook:
    """
    Session Hook zarządzający cyklem życia sesji w Hermes Agent:
    - on_session_start: wstrzyknięcie niezmiennego prefixu
    - on_session_end: asynchroniczny Sleep Cycle (konsolidacja i GC)
    """

    def __init__(self, memory_engine: HybridMemoryEngine):
        self.engine = memory_engine
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self._proposed_sop_ids: Set[str] = set()
        self._consolidated_sessions: Set[str] = set()

    def _cleanup_stale_sessions(self, ttl_seconds: float = 7200.0) -> int:
        """Prunes orphaned sessions older than ttl_seconds (default 2 hours)."""
        now = time.time()
        stale_ids = [
            sid for sid, meta in self.active_sessions.items()
            if now - meta.get("start_time", now) > ttl_seconds
        ]
        for sid in stale_ids:
            del self.active_sessions[sid]
        if stale_ids:
            logger.info("Pruned %d stale sessions from HermesSessionHook", len(stale_ids))
        return len(stale_ids)

    async def on_session_start(
        self,
        session_id: str,
        active_entities: Optional[List[str]] = None,
        system_rules: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Huk wywoływany na początku sesji:
        Pobiera niezmienny stan L2 i generuje prefix cache guard.
        """
        self._cleanup_stale_sessions()
        self._consolidated_sessions.discard(session_id)

        # Izolacja sesji: reset stanu operacyjnego L0 TTT i L1 JEPA
        if hasattr(self.engine, "ttt") and self.engine.ttt is not None and hasattr(self.engine.ttt, "reset_session"):
            self.engine.ttt.reset_session()
        if hasattr(self.engine, "latent") and self.engine.latent is not None and hasattr(self.engine.latent, "reset"):
            self.engine.latent.reset()

        if active_entities is not None:
            entities = list(active_entities)
        else:
            try:
                from atlas_memory.domain.registry import DomainProfileRegistry
                entities = [e["canonical_id"] for e in DomainProfileRegistry.get_instance().get_all_entities() if "canonical_id" in e]
            except Exception:
                entities = []
        verified_states = await self.engine.kv.get_states(entities)

        prefix_prompt = PrefixCacheGuard.build_immutable_prefix(
            verified_state=verified_states,
            system_rules=system_rules or ["Deterministic execution required", "Use search_memory for dynamic relations"],
        )

        session_meta = {
            "session_id": session_id,
            "start_time": time.time(),
            "entities": entities,
            "prefix_prompt": prefix_prompt,
        }
        self.active_sessions[session_id] = session_meta

        return {
            "session_id": session_id,
            "prefix_system_prompt": prefix_prompt,
            "verified_states": verified_states,
        }

    def get_touched_keys(self, session_id: str) -> Set[str]:
        """Pobiera i czyści zbiór kluczy zmodyfikowanych w bieżącej sesji (wraz z buforem L1)."""
        keys: Set[str] = set()
        write_path = getattr(self.engine, "write_path", None)
        if write_path and hasattr(write_path, "session_touched_keys"):
            keys.update(write_path.session_touched_keys.pop(session_id, set()))
            if session_id != "idle_sess":
                keys.update(write_path.session_touched_keys.pop("default", set()))
        sess_meta = self.active_sessions.get(session_id, {})
        if "touched_keys" in sess_meta:
            keys.update(sess_meta.get("touched_keys", set()))
        hot_buf = getattr(self.engine, "hot_buffer", None)
        if hot_buf is not None and hasattr(hot_buf, "get_session_entries"):
            for entry in hot_buf.get_session_entries(session_id):
                if getattr(entry, "key", None):
                    keys.add(str(entry.key))
        return keys

    def has_session_activity(self, session_id: str) -> bool:
        """Sprawdza czy sesja wygenerowała jakiekolwiek mutacje lub trajektorie."""
        write_path = getattr(self.engine, "write_path", None)
        if write_path and hasattr(write_path, "session_touched_keys"):
            if session_id in write_path.session_touched_keys and write_path.session_touched_keys[session_id]:
                return True
            if session_id != "idle_sess" and write_path.session_touched_keys.get("default"):
                return True
        sess_meta = self.active_sessions.get(session_id, {})
        if sess_meta.get("touched_keys"):
            return True
        hot_buf = getattr(self.engine, "hot_buffer", None)
        if hot_buf is not None and hasattr(hot_buf, "get_session_entries"):
            if hot_buf.get_session_entries(session_id):
                return True
        if session_id in self._consolidated_sessions:
            return False
        traj_buf = getattr(self.engine, "trajectory_buffer", None)
        if traj_buf and hasattr(traj_buf, "_session_ids") and session_id in traj_buf._session_ids:
            return True
        sleep_baker = getattr(self.engine, "sleep_baker", None) or (
            getattr(self.engine, "auditor", None) and getattr(self.engine.auditor, "sleep_baker", None)
        )
        if sleep_baker:
            b_sops = getattr(sleep_baker, "baked_sops", None)
            if isinstance(b_sops, dict):
                for sop_id, sop in b_sops.items():
                    if (
                        sop_id not in self._proposed_sop_ids
                        and getattr(sop, "success_rate", 0) >= 0.9
                        and getattr(sop, "invocations_count", 0) >= 5
                    ):
                        return True
        return False

    async def on_session_end(
        self,
        session_id: str,
        trigger_sleep_cycle: bool = True,
    ) -> ConsolidationStats:
        """
        Huk wywoływany po zakończeniu sesji:
        1. Szybkie wyjście (< 5ms) jeśli brak aktywności i mutacji w sesji.
        2. Opróżnienie kolejki commitów.
        3. Opcjonalne uruchomienie nocnej fazy konsolidacji (Sleep Cycle) zawężonej do zmodyfikowanych encji.
        """
        t0 = time.perf_counter()

        # 0. Fast-Exit (< 5ms) jeśli brak jakiejkolwiek aktywności lub mutacji w sesji
        if not self.has_session_activity(session_id):
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            stats = ConsolidationStats()
            stats.status = "idempotent_noop"
            stats.duration_ms = max(0.01, (time.perf_counter() - t0) * 1000.0)
            return stats

        # 1. Przetworzenie wszystkich zaległych rekordów
        if hasattr(self.engine, "process_all_pending"):
            proc = self.engine.process_all_pending()
            if inspect.isawaitable(proc):
                await proc

        # 2. Uruchomienie fazy snu (zawężonej do zmodyfikowanych kluczy)
        stats = ConsolidationStats()
        if trigger_sleep_cycle:
            touched_keys = self.get_touched_keys(session_id)
            if touched_keys:
                # Sciezka zaweżona: koszt ~ms (pomiary V88b/T4: 3-30 ms na 150 rekordach).
                stats = await self.engine.auditor.run_sleep_cycle_consolidation(
                    touched_keys=touched_keys
                )
                await self._reconcile_l1_to_l2(session_id, touched_keys, stats)
            else:
                # V88b/T4: sesja z aktywnoscia ale BEZ zmodyfikowanych kluczy nie moze
                # placic pelnego, nescope'owanego cyklu snu na sciezce session/end.
                # Pomiar na zywym daemonie: 5084.1 ms przy consolidated_records=0
                # (pelny cykl = unscoped przeglad calego store + kaskada). Pelna
                # konsolidacja zostaje dostepna jako jawny RPC trigger_sleep_consolidation.
                stats.status = "consolidation_deferred"

            # Sprawdź SOP z success_rate >= 0.9 i invocations_count >= 5 do propozycji kompilacji
            proposed: List[str] = []
            sleep_baker = getattr(self.engine, "sleep_baker", None) or getattr(self.engine.auditor, "sleep_baker", None)
            baked_sops: Dict[str, StandardProcedure] = getattr(sleep_baker, "baked_sops", {}) if sleep_baker else {}

            for _sop_id, sop in baked_sops.items():
                if (
                    sop.procedure_id not in self._proposed_sop_ids
                    and sop.success_rate >= 0.9
                    and sop.invocations_count >= 5
                ):
                    logger.info("Auto-propose skill compilation for %s", sop.name)
                    proposed.append(sop.procedure_id)
                    self._proposed_sop_ids.add(sop.procedure_id)
                    try:
                        if compile_sop_to_skill is not None:
                            compile_sop_to_skill(sop)
                            logger.info("Compiled skill: %s", sop.procedure_id)
                    except SafetyViolationError as e:
                        logger.warning("Unsafe SOP %s rejected during auto-propose: %s", sop.procedure_id, e)
                    except Exception as e:
                        logger.warning("Failed to compile skill for SOP %s: %s", sop.procedure_id, e)

            stats.proposed_skills = proposed

        # 3. Zakończenie sesji
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        self._consolidated_sessions.add(session_id)

        # Izolacja sesji: reset stanu operacyjnego L0 TTT i L1 JEPA
        if hasattr(self.engine, "ttt") and self.engine.ttt is not None and hasattr(self.engine.ttt, "reset_session"):
            self.engine.ttt.reset_session()
        if hasattr(self.engine, "latent") and self.engine.latent is not None and hasattr(self.engine.latent, "reset"):
            self.engine.latent.reset()

        stats.duration_ms = max(getattr(stats, "duration_ms", 0.0) or 0.0, (time.perf_counter() - t0) * 1000.0)
        return stats


    async def _reconcile_l1_to_l2(
        self,
        session_id: str,
        touched_keys: Set[str],
        stats: ConsolidationStats,
    ) -> None:
        """V109 Pillar 1: Dynamiczna Rekoncyliacja i Wygaszanie Pamięci Roboczej (L1 -> L2)."""
        hot_buf = getattr(self.engine, "hot_buffer", None)
        kv = getattr(self.engine, "kv", None)
        vec = getattr(self.engine, "vector_store", None)
        graph = getattr(self.engine, "graph_store", None) or getattr(self.engine, "graph", None)

        superseded_keys: Set[str] = set()
        l1_promoted = 0

        if kv is not None and touched_keys:
            try:
                from atlas_memory.cognitive.conflict_arbiter import entity_root
                roots = {entity_root(k) for k in touched_keys if entity_root(k)}
                get_all_sync = getattr(kv, "get_all_sync", None)
                all_kv = get_all_sync(paged=True, limit=1000) if callable(get_all_sync) else {}
                for k, data in (all_kv or {}).items():
                    if not isinstance(data, dict):
                        continue
                    meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
                    if data.get("is_superseded") or (meta and meta.get("is_superseded")):
                        if k in touched_keys or entity_root(k) in roots:
                            superseded_keys.add(str(k))
            except Exception as exc:
                logger.debug("[L1->L2] KV superseded check error: %s", exc)

        if hot_buf is not None:
            if hasattr(hot_buf, "superseded_history"):
                superseded_keys.update(getattr(hot_buf, "superseded_history", set()))
            if hasattr(hot_buf, "_buffer"):
                for entry in list(hot_buf._buffer):
                    if entry.is_superseded or entry.metadata.get("is_superseded"):
                        superseded_keys.add(entry.key)
            if superseded_keys and hasattr(hot_buf, "reconcile_superseded"):
                hot_buf.reconcile_superseded(superseded_keys)
            if hasattr(hot_buf, "get_session_entries"):
                for entry in hot_buf.get_session_entries(session_id):
                    if entry.key not in superseded_keys:
                        sessions_set = entry.metadata.setdefault("l2_consolidated_sessions", set())
                        if isinstance(sessions_set, list):
                            sessions_set = set(sessions_set)
                            entry.metadata["l2_consolidated_sessions"] = sessions_set
                        if session_id:
                            sessions_set.add(session_id)
                        entry.metadata["l2_consolidated"] = True
                        l1_promoted += 1

        for sk in superseded_keys:
            if kv is not None:
                try:
                    record = None
                    if hasattr(kv, "get_state_sync"):
                        record = kv.get_state_sync(sk)
                    elif hasattr(kv, "get_state"):
                        res = kv.get_state(sk)
                        record = await res if inspect.isawaitable(res) else res
                    if record is not None and isinstance(record, dict):
                        meta = dict(record.get("metadata") or {})
                        if not meta.get("is_superseded"):
                            meta["is_superseded"] = True
                            meta["superseded_at"] = time.time()
                            set_fn = getattr(kv, "set_state_sync", None) or getattr(kv, "set_state", None)
                            if set_fn is not None:
                                res = set_fn(
                                    key=sk,
                                    value=record.get("value"),
                                    metadata=meta,
                                    source_type=str(meta.get("source_type") or "system"),
                                )
                                if inspect.isawaitable(res):
                                    await res
                except Exception as exc:
                    logger.debug("[L1->L2] KV mark superseded error for %s: %s", sk, exc)

            if vec is not None:
                try:
                    del_async = getattr(vec, "delete_by_upsert_key", None)
                    del_sync = getattr(vec, "delete_by_upsert_key_sync", None)
                    if callable(del_async):
                        await del_async(sk)
                    elif callable(del_sync):
                        del_sync(sk)
                except Exception as exc:
                    logger.debug("[L1->L2] Vector eviction error for %s: %s", sk, exc)
            if graph is not None:
                try:
                    glock = getattr(graph, "_graph_lock", None)
                    if hasattr(graph, "mark_superseded"):
                        if glock is not None:
                            with glock:
                                m_res = graph.mark_superseded(sk)
                        else:
                            m_res = graph.mark_superseded(sk)
                        if inspect.isawaitable(m_res):
                            await m_res
                        nx_g = getattr(graph, "nx_graph", None) or (graph if hasattr(graph, "bump_revision") else None)
                        if nx_g is not None and hasattr(nx_g, "bump_revision"):
                            nx_g.bump_revision()
                    elif hasattr(graph, "nx_graph") and sk in getattr(graph, "nx_graph", {}):
                        if glock is not None:
                            with glock:
                                graph.nx_graph.nodes[sk]["is_superseded"] = True
                        else:
                            graph.nx_graph.nodes[sk]["is_superseded"] = True
                        if hasattr(graph.nx_graph, "bump_revision"):
                            graph.nx_graph.bump_revision()
                    elif hasattr(graph, "nodes") and sk in getattr(graph, "nodes", {}):
                        dlock = getattr(graph, "lock", None) or glock
                        if dlock is not None:
                            with dlock:
                                graph.nodes[sk]["is_superseded"] = True
                        else:
                            graph.nodes[sk]["is_superseded"] = True
                        if hasattr(graph, "bump_revision"):
                            graph.bump_revision()
                except Exception as exc:
                    logger.debug("[L1->L2] Graph mark superseded error for %s: %s", sk, exc)

        if superseded_keys:
            stats.facts_superseded = max(int(getattr(stats, "facts_superseded", 0) or 0), len(superseded_keys))
        total_reconciled = max(
            int(getattr(stats, "consolidated_records", 0) or 0),
            int(getattr(stats, "facts_superseded", 0) or 0) + l1_promoted,
        )
        stats.consolidated_records = total_reconciled
