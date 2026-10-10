from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from atlas_memory.cognitive.models import clamp_conf
from atlas_memory.core.models import ConflictReport, ConsolidationStats, EpistemicSource, MemoryRecord
from atlas_memory.extensions.decay_scorer import SalienceDecayEngine
from atlas_memory.l2_semantic.kuzu_graph import KuzuGraphStore
from atlas_memory.l2_semantic.kv_store import VerifiedKVStore
from atlas_memory.l2_semantic.qdrant_store import QdrantVectorStore
from atlas_memory.l3_procedural.declarative_consolidator import DeclarativeConsolidator

logger = logging.getLogger(__name__)


from atlas_memory.cognitive.precision import BayesianPrecisionTensor


@dataclass
class TransactionalDeltaTensor:
    """Odwracalna Monada Stanu (Reversible State Monad) dla atomowości transakcyjnej L3.

    Przed fizycznym dotknięciem jakiejkolwiek bazy danych (KV, Kùzu, Qdrant),
    syntetyzuje strukturę operacji forward i backward na wszystkich warstwach.
    Eliminuje asymetrię poznawczą (Split-Brain) gwarantując pełny symetryczny ROLLBACK-ALL
    oraz stan transakcji PENDING -> COMMITTED | FAILED bez maskującej flagi _write_path_done.
    """

    tx_id: str
    record: MemoryRecord
    status: str = "PENDING"
    error: Optional[str] = None
    applied_steps: List[str] = field(default_factory=list)
    backup_kv: Optional[Dict[str, Any]] = None

    async def execute(self, auditor: MemoryAuditor) -> bool:
        await auditor.kv.create_transaction_intent(self.tx_id, self.record)
        self.applied_steps.append("intent")

        try:
            # 1. Forward Kùzu Graph
            await auditor.graph.add_record(self.record)
            self.applied_steps.append("graph")

            # 2. Forward VerifiedKVStore (dla zmiennych stanu)
            if self.record.is_state_variable and auditor.kv is not None:
                subj = self.record.effective_subject
                self.backup_kv = await auditor.kv.get_state(subj)
                meta = dict(self.record.metadata or {})
                src = (
                    self.record.source_type.value
                    if hasattr(self.record.source_type, "value")
                    else str(self.record.source_type)
                )
                meta["source_type"] = src
                meta["subject"] = subj
                meta["predicate"] = self.record.predicate
                meta["object"] = str(self.record.object)
                meta["is_state_variable"] = True
                await auditor.kv.set_state(
                    key=subj,
                    value=self.record.object,
                    confidence=self.record.confidence,
                    metadata=meta,
                    reason=f"monad_commit_{src}",
                )
                self.applied_steps.append("kv")

            # 3. Forward Qdrant Vector Store
            if auditor.vector_store is not None:
                u_key = self.record.effective_subject if self.record.is_state_variable else self.record.triple_key
                try:
                    await auditor.vector_store.insert(self.record, upsert_key=u_key)
                except TypeError:
                    await auditor.vector_store.insert(self.record)
                self.applied_steps.append("vector")

            # 4. Commit
            await auditor.kv.mark_intent_committed(self.tx_id)
            self.status = "COMMITTED"
            from atlas_memory.core.agi_logger import log_saga_transaction
            log_saga_transaction(logger, self.tx_id, "atomic_write", self.applied_steps, status="committed")
            return True

        except Exception as e:
            self.status = "FAILED"
            self.error = str(e)
            logger.error(
                "[TransactionalDeltaTensor] Transaction %s failed: %s. Executing symplektic ROLLBACK-ALL.",
                self.tx_id,
                e,
            )

            # Symetryczny ROLLBACK-ALL w kolejności odwrotnej do wykonania
            if "vector" in self.applied_steps and auditor.vector_store is not None:
                u_key = self.record.effective_subject if self.record.is_state_variable else self.record.triple_key
                try:
                    del_async = getattr(auditor.vector_store, "delete_by_upsert_key", None)
                    del_sync = getattr(auditor.vector_store, "delete_by_upsert_key_sync", None)
                    if callable(del_async):
                        await del_async(u_key)
                    elif callable(del_sync):
                        del_sync(u_key)
                    elif hasattr(auditor.vector_store, "delete"):
                        await auditor.vector_store.delete(u_key)
                    elif hasattr(auditor.vector_store, "delete_sync"):
                        auditor.vector_store.delete_sync(u_key)
                except Exception as ve_err:
                    logger.warning("[TransactionalDeltaTensor] Rollback vector failed for %s: %s", u_key, ve_err)

            if "kv" in self.applied_steps and auditor.kv is not None:
                subj = self.record.effective_subject
                try:
                    if self.backup_kv is not None:
                        await auditor.kv.set_state(
                            key=subj,
                            value=self.backup_kv["value"],
                            confidence=self.backup_kv["confidence"],
                            metadata=self.backup_kv.get("metadata", {}),
                            reason="monad_rollback_restore",
                        )
                    else:
                        await auditor.kv.delete_state(subj)
                except Exception as kv_err:
                    logger.warning("[TransactionalDeltaTensor] Rollback KV failed for %s: %s", subj, kv_err)

            if "graph" in self.applied_steps and auditor.graph is not None:
                try:
                    await auditor.graph.remove_edge(
                        self.record.effective_subject, self.record.predicate, str(self.record.object)
                    )
                except Exception as g_err:
                    logger.warning("[TransactionalDeltaTensor] Rollback graph remove_edge failed: %s", g_err)

            if "intent" in self.applied_steps and auditor.kv is not None:
                try:
                    await auditor.kv.mark_intent_failed(self.tx_id, str(e))
                except Exception as in_err:
                    logger.warning("[TransactionalDeltaTensor] Mark intent failed error: %s", in_err)

            from atlas_memory.core.agi_logger import log_saga_transaction
            log_saga_transaction(logger, self.tx_id, "compensatory_rollback", self.applied_steps, status="rolled_back", reason=str(e))

            raise e


class MemoryAuditor:
    """
    L3: Asynchroniczny Audytor Pamięci i Faza Konsolidacji (Sleep Phase) z obsługą Odwracalnej Monady Stanu.
    
    Odpowiada za:
    1. Rozstrzyganie sporów epistemicznych i czasowych (online conflict resolution) przez Tensor Fristona.
    2. Atomowy zapis do wielu baz danych przez TransactionalDeltaTensor (ochrona przed split-brain).
    3. Nocną konsolidację: GC osieroconych węzłów, usuwanie zapomnianych faktów (Decay Pruning).
    4. Naprawę niespójności transakcyjnych po awarii (Crash Recovery).
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

    def __init__(
        self,
        graph_store: KuzuGraphStore,
        kv_store: VerifiedKVStore,
        vector_store: Optional[QdrantVectorStore] = None,
        calibrator: Optional[Any] = None,
        decay_engine: Optional[SalienceDecayEngine] = None,
        isotonic_calibrator: Optional[Any] = None,
        write_path: Optional[Any] = None,
    ):
        self.graph = graph_store
        self.kv = kv_store
        self.vector_store = vector_store
        self.decay_engine = decay_engine or SalienceDecayEngine()
        self.isotonic_calibrator = isotonic_calibrator
        self.conflict_history: deque[ConflictReport] = deque(maxlen=1000)
        self.precision_tensor = BayesianPrecisionTensor()
        if write_path is not None:
            self.write_path = write_path
        else:
            from atlas_memory.cognitive.write_path import EpistemicWritePath
            self.write_path = EpistemicWritePath(
                kv_store=self.kv,
                graph_store=self.graph,
                vector_store=self.vector_store,
            )

    def calibrate_record(self, record: MemoryRecord) -> MemoryRecord:
        """Dostosowuje bazową pewność faktu na podstawie tensora precyzji bayesowskiej."""
        prec = self.precision_tensor.get_precision(record.source_type)
        base_factor = min(1.0, max(0.1, prec / 2.5))
        record.confidence = clamp_conf(round(float(record.confidence * base_factor), 3))
        return record

    def arbitrate_record_conflict(
        self,
        existing: MemoryRecord,
        incoming: MemoryRecord,
        delta_margin: float = 0.05,
    ) -> Tuple[bool, str]:
        """Rozstrzyga spór pomiędzy dwoma sprzecznymi faktami za pomocą Tensora Precyzji Fristona."""
        lambda_exist = self.precision_tensor.compute_lambda(existing.source_type, existing.confidence)
        lambda_in = self.precision_tensor.compute_lambda(incoming.source_type, incoming.confidence)

        # 1. Bayesowski Tensor Precyzji: różnica przekracza delta_margin
        if (lambda_in - lambda_exist) > delta_margin:
            return (
                True,
                f"bayesian_precision_override: incoming lambda={lambda_in:.3f} outranks existing lambda={lambda_exist:.3f} (delta={lambda_in - lambda_exist:.3f} > {delta_margin})",
            )
        elif (lambda_exist - lambda_in) > delta_margin:
            return (
                False,
                f"bayesian_precision_rejected: existing lambda={lambda_exist:.3f} outranks incoming lambda={lambda_in:.3f} (delta={lambda_exist - lambda_in:.3f} > {delta_margin})",
            )

        # 2. Gdy wartości Lambda są w strefie remisowej (|delta| <= delta_margin): porównujemy znaczniki czasu
        if incoming.timestamp > existing.timestamp:
            return True, "timestamp_override: newer record with comparable epistemic lambda"
        elif incoming.timestamp < existing.timestamp:
            return False, "stale_rejected: older record with comparable epistemic lambda"

        # 3. Równe znaczniki czasu: wyższa deklarowana pewność
        if incoming.confidence >= existing.confidence:
            return True, "confidence_override: higher confidence at same timestamp"
        else:
            return False, "confidence_rejected: lower confidence at same timestamp"

    async def check_and_resolve_conflict(self, record: MemoryRecord) -> Tuple[Optional[ConflictReport], bool]:
        """Sprawdza i rozstrzyga spory w wiedzy przed zapisem."""
        self.calibrate_record(record)
        subj = record.effective_subject

        # 1. Sprawdzenie w KV Store dla zmiennych stanu
        if record.is_state_variable:
            existing_state = await self.kv.get_state(subj)
            if existing_state:
                old_val = existing_state["value"]
                old_ts = existing_state["timestamp"]
                old_conf = existing_state["confidence"]
                meta = existing_state.get("metadata", {})
                old_source_str = meta.get("source_type", "agent_inference")
                try:
                    old_source = EpistemicSource(old_source_str)
                except ValueError:
                    old_source = EpistemicSource.AGENT_INFERENCE

                if str(old_val) != str(record.object):
                    existing_rec = MemoryRecord(
                        subject=subj,
                        predicate=record.predicate,
                        object=str(old_val),
                        confidence=old_conf,
                        timestamp=old_ts,
                        is_state_variable=True,
                        source_type=old_source,
                    )

                    should_override, reason = self.arbitrate_record_conflict(existing_rec, record)
                    if should_override:
                        report = ConflictReport(
                            conflict_type="state_value_override",
                            existing_record=existing_rec,
                            incoming_record=record,
                            resolution_strategy=reason,
                            resolved_record=record,
                        )
                        self.conflict_history.append(report)
                        return report, True
                    else:
                        report = ConflictReport(
                            conflict_type="stale_or_low_rank_state_rejected",
                            existing_record=existing_rec,
                            incoming_record=record,
                            resolution_strategy=reason,
                            resolved_record=None,
                        )
                        self.conflict_history.append(report)
                        return report, False

        # 2. Sprawdzenie w Grafie Wiedzy
        existing_relations = await self.graph.get_node_relations(subj, record.predicate)
        for rel in existing_relations:
            if str(rel["object"]) != str(record.object):
                old_ts = rel["timestamp"]
                old_conf = rel["confidence"]

                existing_rec = MemoryRecord(
                    subject=subj,
                    predicate=record.predicate,
                    object=str(rel["object"]),
                    confidence=old_conf,
                    timestamp=old_ts,
                )

                if record.predicate.startswith("is_") or record.predicate.startswith("has_") or "status" in record.predicate:
                    should_override, reason = self.arbitrate_record_conflict(existing_rec, record)
                    if should_override:
                        await self.graph.remove_edge(subj, record.predicate, str(rel["object"]))
                        report = ConflictReport(
                            conflict_type="functional_relation_contradiction",
                            existing_record=existing_rec,
                            incoming_record=record,
                            resolution_strategy=reason,
                            resolved_record=record,
                        )
                        self.conflict_history.append(report)
                        return report, True
                    else:
                        report = ConflictReport(
                            conflict_type="stale_or_low_rank_relation_rejected",
                            existing_record=existing_rec,
                            incoming_record=record,
                            resolution_strategy=reason,
                            resolved_record=None,
                        )
                        self.conflict_history.append(report)
                        return report, False

        return None, True

    async def atomic_insert_with_saga(self, record: MemoryRecord) -> bool:
        """Atomowy zapis zabezpieczony przez Odwracalną Monadę Stanu (TransactionalDeltaTensor)."""
        tx_id = f"tx_{int(time.time() * 1000)}_{abs(hash(record.full_key)) % 100000}"
        monad = TransactionalDeltaTensor(tx_id=tx_id, record=record)
        return await monad.execute(self)

    async def recover_dangling_transactions(self) -> int:
        """
        Odzyskiwanie po awarii (Crash Recovery):
        Przeszukuje wiszące intencje w stanie PENDING i dokonuje ich kompensacji/naprawy.
        """
        dangling = await self.kv.get_dangling_intents()
        recovered_count = 0
        for tx in dangling:
            tx_id = tx["tx_id"]
            # Próba kompensacji w grafie
            try:
                await self.graph.remove_edge(tx["subject"], tx["predicate"], tx["object"])
            except Exception as exc:
                logger.debug("Crash recovery remove_edge ignored in graph for tx %s: %s", tx_id, exc)
            await self.kv.mark_intent_failed(tx_id, "recovered_after_crash")
            recovered_count += 1
        return recovered_count


    async def run_sleep_cycle_consolidation(
        self,
        touched_keys: Optional[Set[str]] = None,
    ) -> ConsolidationStats:
        """Nocna konsolidacja wiedzy (Sleep Phase)."""
        start_t = time.perf_counter()
        stats = ConsolidationStats()

        if touched_keys is not None and len(touched_keys) == 0:
            stats.status = "idempotent_noop"
            stats.duration_ms = (time.perf_counter() - start_t) * 1000.0
            return stats

        # 1. Odzyskanie ewentualnych wiszących transakcji
        await self.recover_dangling_transactions()

        # 2. Czyszczenie osieroconych węzłów w grafie (GC)
        pruned_orphans = await self.graph.garbage_collect_orphans()
        stats.orphaned_nodes_gc = pruned_orphans

        # 3. Synchronizacja spójności ze zmiennymi stanu KV i czyszczenie przestarzałych faktów (Salience GC)
        if touched_keys is not None:
            get_states: Any = getattr(self.kv, "get_states", None)
            if callable(get_states):
                all_states = await get_states(list(touched_keys))
            else:
                all_states = await self.kv.get_all_states()
        else:
            all_states = await self.kv.get_all_states()
        stats.records_analyzed = len(all_states)

        for key, data in all_states.items():
            try:
                conf = clamp_conf(data.get("confidence", 1.0))
                meta = data.get("metadata", {})
                if not isinstance(meta, dict):
                    meta = {}
                source_str = meta.get("source_type", meta.get("source", "agent_inference"))
                try:
                    source = EpistemicSource(source_str)
                except ValueError:
                    source = EpistemicSource.AGENT_INFERENCE

                # Sprawdzenie ochrony: flaga user_explicit z wysokim confidence >= 0.99
                is_user_explicit = (source == EpistemicSource.USER_EXPLICIT or source_str == "user_explicit")
                is_protected = is_user_explicit and (conf >= 0.99)

                rec_for_decay = MemoryRecord(
                    subject=key,
                    predicate="state_value",
                    object=str(data.get("value", "")),
                    confidence=conf,
                    timestamp=float(data.get("timestamp", 0.0)),
                    is_state_variable=False,  # Umożliwia ewaluację przez decay_engine.should_prune
                    source_type=source,
                    importance_score=float(meta.get("importance_score", 0.5)),
                )

                should_prune = False
                if not is_protected:
                    if conf < 0.3:
                        should_prune = True
                    elif self.decay_engine is not None and self.decay_engine.should_prune(rec_for_decay):
                        should_prune = True

                if should_prune:
                    if hasattr(self.kv, "delete_state"):
                        await self.kv.delete_state(key)
                    if hasattr(self.graph, "remove_edge"):
                        try:
                            await self.graph.remove_edge(key, "state_value", str(data.get("value", "")))
                        except Exception as exc:
                            logger.debug("Graph remove_edge ignored during Salience GC: %s", exc)
                    stats.pruned_stale_facts += 1
                    stats.decayed_pruned_records += 1
                else:
                    await self.graph.add_record(MemoryRecord(
                        subject=key,
                        predicate="state_value",
                        object=str(data.get("value", "")),
                        confidence=conf,
                        timestamp=float(data.get("timestamp", 0.0)),
                        is_state_variable=True,
                    ))
            except Exception as rec_err:
                logger.warning("Error processing state %s during sleep consolidation: %s", key, rec_err)

        if stats.pruned_stale_facts > 0:
            extra_orphans = await self.graph.garbage_collect_orphans()
            stats.orphaned_nodes_gc += extra_orphans

        # V66 (audyt §3B): faza 4 — deklaratywna kaskada snu L0→L1→L2 + Salience GC.
        # Nie-fatalna: awaria kaskady nie wywraca cyklu snu (proceduralna część i
        # stary decay-GC już zakończyły pracę); wynik meldowany w statystykach.
        try:
    
            consolidator = DeclarativeConsolidator(
                kv_store=self.kv,
                graph_store=self.graph,
                vector_store=self.vector_store,
            )
            dcs = await consolidator.consolidate(candidate_keys=touched_keys)
            stats.facts_superseded = dcs.facts_superseded
            stats.consolidated_records = dcs.facts_superseded + dcs.gc_pruned
            stats.declarative_pairs_found = dcs.pairs_found
            stats.declarative_clusters = dcs.clusters
            stats.declarative_roots_blocked = dcs.roots_blocked
            stats.declarative_orthogonal_slots_blocked = getattr(dcs, "orthogonal_slots_blocked", 0)
            stats.declarative_edges_wired = dcs.edges_wired
            stats.declarative_gc_pruned = dcs.gc_pruned
        except Exception as dc_err:
            logger.warning("V66 declarative cascade failed (non-fatal): %s", dc_err)

        # FINDING-072: Checkpoint WAL do plików bazodanowych po cyklu konsolidacji snu
        if hasattr(self.kv, "checkpoint_wal_sync"):
            try:
                await asyncio.to_thread(self.kv.checkpoint_wal_sync, truncate=True)
            except Exception as cp_err:
                logger.debug("WAL checkpoint during sleep consolidation error: %s", cp_err)

        stats.duration_ms = (time.perf_counter() - start_t) * 1000.0
        return stats
