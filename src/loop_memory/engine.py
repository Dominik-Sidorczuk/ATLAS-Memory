from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from loop_memory.arrow_buffer.trajectory_buffer import ArrowTrajectoryBuffer
from loop_memory.extensions.canonicalizer import EntityCanonicalizer
from loop_memory.extensions.compactor import ContextCompactor
from loop_memory.extensions.decay_scorer import SalienceDecayEngine
from loop_memory.extensions.epistemic import EpistemicCalibrator
from loop_memory.l0_dynamic.ttt_layer import TTTLayer
from loop_memory.l1_working.jepa_latent import JEPALatentBuffer
from loop_memory.l2_semantic.kuzu_graph import KuzuGraphStore
from loop_memory.l2_semantic.kv_store import VerifiedKVStore
from loop_memory.l2_semantic.qdrant_store import QdrantVectorStore
from loop_memory.l3_procedural.auditor import MemoryAuditor
from loop_memory.l3_procedural.weight_baker import WeightBaker
from loop_memory.models import (
    ActionPlan,
    ConflictReport,
    MemoryRecord,
    PredictedTransition,
)

logger = logging.getLogger(__name__)



class HybridMemoryEngine:
    """
    Hybrydowy silnik pamięci produkcyjnej dla Hermes Agent.
    
    Zbudowany na dojrzałych komponentach:
    - L0: Numba JIT TTT Layer (kompresja online < 0.05 ms)
    - L1: Numba JIT JEPA Buffer + Apache Arrow (Zero-Copy RAM & Parquet)
    - L3: Sleep-Cycle Consolidation & PEFT LoRA Weight Baking + Saga Transaction Protection
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        graph_client: KuzuGraphStore,
        kv_store: VerifiedKVStore,
        ttt_layer: Optional[TTTLayer] = None,
        latent_buffer: Optional[JEPALatentBuffer] = None,
        trajectory_buffer: Optional[ArrowTrajectoryBuffer] = None,
        auditor: Optional[MemoryAuditor] = None,
        weight_baker: Optional[WeightBaker] = None,
        canonicalizer: Optional[EntityCanonicalizer] = None,
        decay_engine: Optional[SalienceDecayEngine] = None,
        compactor: Optional[ContextCompactor] = None,
        calibrator: Optional[EpistemicCalibrator] = None,
    ):
        self.vector_store = vector_store
        self.graph = graph_client
        self.kv = kv_store
        self.ttt = ttt_layer or TTTLayer()
        self.latent = latent_buffer or JEPALatentBuffer()
        self.trajectory_buffer = trajectory_buffer or ArrowTrajectoryBuffer(state_dim=self.latent.state_dim)
        self.canonicalizer = canonicalizer or EntityCanonicalizer()
        self.decay_engine = decay_engine or SalienceDecayEngine()
        self.compactor = compactor or ContextCompactor()
        self.calibrator = calibrator or EpistemicCalibrator()

        self.auditor = auditor or MemoryAuditor(
            graph_store=self.graph,
            kv_store=self.kv,
            vector_store=self.vector_store,
            calibrator=self.calibrator,
            decay_engine=self.decay_engine,
        )
        self.weight_baker = weight_baker or WeightBaker()

        self.audit_queue: asyncio.Queue[MemoryRecord] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._is_running: bool = False

    @classmethod
    def create_default(
        cls,
        db_path: str = ":memory:",
        qdrant_location: str = ":memory:",
        kuzu_path: Optional[str] = None,
        kuzu_buffer_pool_size: int = 128 * 1024 * 1024,
    ) -> HybridMemoryEngine:
        """Tworzy instancję pełnego klastra pamięci z buforem Arrow, Numba JIT i Saga Pattern."""
        v_store = QdrantVectorStore(location=qdrant_location)
        g_store = KuzuGraphStore(db_path=kuzu_path or ":memory:", buffer_pool_size_bytes=kuzu_buffer_pool_size)
        k_store = VerifiedKVStore(db_path=db_path)
        ttt = TTTLayer()
        latent = JEPALatentBuffer()
        traj_buf = ArrowTrajectoryBuffer(state_dim=latent.state_dim)
        canon = EntityCanonicalizer()
        decay = SalienceDecayEngine()
        comp = ContextCompactor()
        calib = EpistemicCalibrator()
        auditor = MemoryAuditor(g_store, k_store, v_store, calibrator=calib, decay_engine=decay)
        baker = WeightBaker()

        return cls(
            vector_store=v_store,
            graph_client=g_store,
            kv_store=k_store,
            ttt_layer=ttt,
            latent_buffer=latent,
            trajectory_buffer=traj_buf,
            auditor=auditor,
            weight_baker=baker,
            canonicalizer=canon,
            decay_engine=decay,
            compactor=comp,
            calibrator=calib,
        )

    async def recall(self, query: str, active_entities: List[str]) -> Dict[str, Any]:
        """
        Równoległe odpytanie pamięci:
        1. Kanonizacja aliasów encji (Entity Canonicalizer).
        2. Wektorowe pobranie kontekstu (Qdrant + FastEmbed).
        3. Grafowe pobranie relacji Cypher (Kùzu + NetworkX).
        4. Odpytanie bazy KV o twarde zmienne stanu (SQLite).
        """
        start_t = time.perf_counter()
        canon_entities = self.canonicalizer.canonicalize_list(active_entities)

        async with asyncio.TaskGroup() as tg:
            t_vector = tg.create_task(self._fetch_vectors(query))
            t_graph = tg.create_task(self._fetch_graph_relations(canon_entities))
            t_state = tg.create_task(self._fetch_state_vars(canon_entities))

        latency_ms = (time.perf_counter() - start_t) * 1000.0

        return {
            "semantic_context": t_vector.result(),
            "graph_topology": t_graph.result(),
            "verified_state": t_state.result(),
            "canonical_entities": canon_entities,
            "retrieval_latency_ms": latency_ms,
            "latent_state": self.latent.current_state.model_dump(),
        }

    async def commit_observation(self, record: MemoryRecord) -> None:
        """Asynchroniczny commit obserwacji do kolejki audytora.

        Fix atlas-v5: lazy start workera audytora — wcześniej start_worker()
        nigdy nie był wywoływany, więc kolejka commitów nie była konsumowana
        i graf/KV nie były aktualizowane (chyba że ktoś ręcznie wywołał
        start_worker()/process_all_pending()). commit_observation jest async,
        więc asyncio.create_task ma gwarantowaną działającą pętlę zdarzeń.
        """
        if not record.canonical_entity_id:
            record.canonical_entity_id = self.canonicalizer.canonicalize(record.subject)
        self.start_worker()
        await self.audit_queue.put(record)

    def record_mental_transition(self, action: ActionPlan, session_id: str = "default") -> PredictedTransition:
        """Wykonuje krok myślowy JEPA (Numba) i rejestruje go w buforze Arrow (Zero-Copy)."""
        transition = self.latent.predict_transition(self.latent.current_state, action)
        self.trajectory_buffer.append_transition(transition, session_id=session_id)
        return transition

    async def memory_auditor_worker(self) -> None:
        self._is_running = True
        try:
            while self._is_running:
                record = await self.audit_queue.get()
                try:
                    conflict = await self._check_conflict(record)
                    if conflict:
                        await self._resolve_conflict(record, conflict)
                    else:
                        await self._atomic_insert(record)
                except Exception as exc:
                    logger.warning("Audit worker failed to process record %s: %s", record.triple_key, exc)
                finally:
                    self.audit_queue.task_done()
        except asyncio.CancelledError:
            self._is_running = False

    def start_worker(self) -> asyncio.Task:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self.memory_auditor_worker())
        return self._worker_task

    async def stop_worker(self) -> None:
        self._is_running = False
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                logger.debug("Memory auditor worker stopped gracefully via cancellation.")
            self._worker_task = None


    async def process_all_pending(self) -> None:
        while not self.audit_queue.empty():
            record = await self.audit_queue.get()
            try:
                conflict = await self._check_conflict(record)
                if conflict:
                    await self._resolve_conflict(record, conflict)
                else:
                    await self._atomic_insert(record)
            finally:
                self.audit_queue.task_done()

    async def _fetch_vectors(self, query: str) -> List[Dict[str, Any]]:
        return await self.vector_store.search(query, top_k=5)

    async def _fetch_graph_relations(self, entities: List[str]) -> Dict[str, Any]:
        return await self.graph.get_subgraph_relations(entities, max_depth=2)

    async def _fetch_state_vars(self, entities: List[str]) -> Dict[str, Any]:
        return await self.kv.get_states(entities)

    async def _check_conflict(self, record: MemoryRecord) -> Optional[ConflictReport]:
        report, should_save = await self.auditor.check_and_resolve_conflict(record)
        return report

    async def _resolve_conflict(self, new_record: MemoryRecord, existing_conflict: ConflictReport) -> None:
        if existing_conflict.resolved_record is not None:
            await self._atomic_insert(existing_conflict.resolved_record)

    async def _atomic_insert(self, record: MemoryRecord) -> None:
        # Zapis chroniony wzorcem Saga Pattern (2PC z Write-Ahead Logiem intencji)
        await self.auditor.atomic_insert_with_saga(record)
