from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

from atlas_memory.extensions.decay_scorer import SalienceDecayEngine
from atlas_memory.extensions.epistemic import EpistemicCalibrator
from atlas_memory.l2_semantic.kuzu_graph import KuzuGraphStore
from atlas_memory.l2_semantic.kv_store import VerifiedKVStore
from atlas_memory.l2_semantic.qdrant_store import QdrantVectorStore
from atlas_memory.models import ConflictReport, ConsolidationStats, EpistemicSource, MemoryRecord

logger = logging.getLogger(__name__)



class MemoryAuditor:
    """
    L3: Asynchroniczny Audytor Pamięci i Faza Konsolidacji (Sleep Phase) z obsługą Saga Pattern.
    
    Odpowiada za:
    1. Rozstrzyganie sporów epistemicznych i czasowych (online conflict resolution).
    2. Atomowy zapis do wielu baz danych w architekturze Saga / 2-Phase Commit (ochrona przed split-brain).
    3. Nocną konsolidację: GC osieroconych węzłów, usuwanie zapomnianych faktów (Decay Pruning).
    4. Naprawę niespójności transakcyjnych po awarii (Crash Recovery).
    """

    def __init__(
        self,
        graph_store: KuzuGraphStore,
        kv_store: VerifiedKVStore,
        vector_store: Optional[QdrantVectorStore] = None,
        calibrator: Optional[EpistemicCalibrator] = None,
        decay_engine: Optional[SalienceDecayEngine] = None,
    ):
        self.graph = graph_store
        self.kv = kv_store
        self.vector_store = vector_store
        self.calibrator = calibrator or EpistemicCalibrator()
        self.decay_engine = decay_engine or SalienceDecayEngine()
        self.conflict_history: List[ConflictReport] = []

    async def check_and_resolve_conflict(self, record: MemoryRecord) -> Tuple[Optional[ConflictReport], bool]:
        """Sprawdza i rozstrzyga spory w wiedzy przed zapisem."""
        self.calibrator.calibrate(record)
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

                    should_override, reason = self.calibrator.arbitrate_conflict(existing_rec, record)
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
                    should_override, reason = self.calibrator.arbitrate_conflict(existing_rec, record)
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
        """
        Atomowy zapis w architekturze Saga Pattern:
        1. Write-Ahead Intent do SQLite (status='PENDING').
        2. Zapis w grafie Kùzu i bazie wektorowej Qdrant.
        3. Zapis zmiennej stanu w SQLite i potwierdzenie (status='COMMITTED').
        4. W przypadku błędu: automatyczna kompensacja i status='FAILED'.
        """
        tx_id = f"tx_{int(time.time() * 1000)}_{abs(hash(record.full_key)) % 100000}"
        await self.kv.create_transaction_intent(tx_id, record)

        try:
            # Krok 1: Graf Kùzu
            await self.graph.add_record(record)

            # Krok 2: Baza wektorowa Qdrant
            if self.vector_store is not None:
                await self.vector_store.insert(record)

            # Krok 3: Stan w SQLite (jeśli state variable)
            if record.is_state_variable:
                meta = dict(record.metadata)
                meta["source_type"] = record.source_type.value
                await self.kv.set_state(
                    key=record.effective_subject,
                    value=record.object,
                    confidence=record.confidence,
                    metadata=meta,
                    reason=f"commit_{record.source_type.value}",
                )

            # Sukces: Oznaczenie intencji jako COMMITTED
            await self.kv.mark_intent_committed(tx_id)
            return True

        except Exception as e:
            # Kompensacja (Rollback): usunięcie dodanej krawędzi z grafu
            try:
                await self.graph.remove_edge(record.effective_subject, record.predicate, str(record.object))
            except Exception as exc:
                logger.debug("Rollback remove_edge ignored in graph: %s", exc)

            await self.kv.mark_intent_failed(tx_id, str(e))
            raise e

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


    async def run_sleep_cycle_consolidation(self) -> ConsolidationStats:
        """Nocna konsolidacja wiedzy (Sleep Phase)."""
        start_t = time.perf_counter()
        stats = ConsolidationStats()

        # 1. Odzyskanie ewentualnych wiszących transakcji
        await self.recover_dangling_transactions()

        # 2. Czyszczenie osieroconych węzłów w grafie (GC)
        pruned_orphans = await self.graph.garbage_collect_orphans()
        stats.orphaned_nodes_gc = pruned_orphans

        # 3. Synchronizacja spójności ze zmiennymi stanu KV
        all_states = await self.kv.get_all_states()
        stats.records_analyzed = len(all_states)

        for key, data in all_states.items():
            await self.graph.add_record(MemoryRecord(
                subject=key,
                predicate="state_value",
                object=str(data["value"]),
                confidence=data["confidence"],
                timestamp=data["timestamp"],
                is_state_variable=True,
            ))

        stats.duration_ms = (time.perf_counter() - start_t) * 1000.0
        return stats
