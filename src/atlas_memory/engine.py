from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import zlib
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from atlas_memory.l3_procedural.weight_baker import WeightBaker

import numpy as np

from atlas_memory.cognitive.write_path import EpistemicWritePath
from atlas_memory.core.models import (
    ActionPlan,
    MemoryRecord,
    PredictedTransition,
)
from atlas_memory.extensions.canonicalizer import EntityCanonicalizer
from atlas_memory.extensions.compactor import ContextCompactor
from atlas_memory.extensions.decay_scorer import SalienceDecayEngine
from atlas_memory.l0_dynamic.ttt_layer import TTTLayer
from atlas_memory.l1_working.jepa_latent import JEPALatentBuffer
from atlas_memory.l1_working.trajectory_buffer import ArrowTrajectoryBuffer
from atlas_memory.l2_semantic.kuzu_graph import KUZU_BUFFER_POOL_BYTES, KuzuGraphStore
from atlas_memory.l2_semantic.kv_store import VerifiedKVStore
from atlas_memory.l2_semantic.qdrant_store import QdrantVectorStore
from atlas_memory.l3_procedural.auditor import MemoryAuditor

logger = logging.getLogger(__name__)

# V68 (ARCH-001): źródło wektora karmiącego TTT (adapt_step).
#   "real"   (domyślne): wektor deterministycznie wyprowadzony z REALNIE
#            zaobserwowanego przejścia (zawartość ActionPlan z transcriptu
#            narzędzi / commitu obserwacji) — NIE z nieuczonych, losowych
#            macierzy JEPA (w_s/w_a). TTT przestaje optymalizować własny szum.
#   "mental" (legacy V55): predicted_state.vector po macierzach losowych JEPA.
# Odwracalne: zmiana tej stałej LUB `engine.ttt_vector_source` przywraca stare
# zachowanie bez żadnych zmian w API adapt_step / testach V55/V57.
TTT_VECTOR_SOURCE: str = "real"



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
    ):
        self.vector_store = vector_store
        self.graph = graph_client
        self.kv = kv_store
        self.latent = latent_buffer or JEPALatentBuffer()
        self.ttt = ttt_layer or TTTLayer(input_dim=self.latent.state_dim, hidden_dim=self.latent.action_dim)
        self.ttt_vector_source: str = TTT_VECTOR_SOURCE
        self.trajectory_buffer = trajectory_buffer or ArrowTrajectoryBuffer(state_dim=self.latent.state_dim)
        self.canonicalizer = canonicalizer or EntityCanonicalizer()
        self.decay_engine = decay_engine or SalienceDecayEngine()
        self.compactor = compactor or ContextCompactor()

        self.write_path = EpistemicWritePath(
            kv_store=self.kv,
            graph_store=self.graph,
            vector_store=self.vector_store,
        )
        self.hot_buffer = self.write_path.hot_buffer
        self.auditor = auditor or MemoryAuditor(
            graph_store=self.graph,
            kv_store=self.kv,
            vector_store=self.vector_store,
            decay_engine=self.decay_engine,
            write_path=self.write_path,
        )
        self._weight_baker = weight_baker
        self._hydrate_l0_step_counter()

    def _hydrate_l0_step_counter(self) -> None:
        """V87 (T3): wczytuje licznik krokow L0-TTT z SQLite do warstwy TTT."""
        kv = getattr(self, "kv", None)
        if kv is None or not hasattr(kv, "load_l0_step_counter"):
            return
        try:
            stored = kv.load_l0_step_counter()
        except Exception as hydrate_exc:
            logger.debug("L0-TTT counter hydration failed: %s", hydrate_exc)
            return
        if not isinstance(stored, dict):
            return
        steps = stored.get("step_count")
        if isinstance(steps, int) and steps > 0 and hasattr(self.ttt, "restore_counters"):
            energy = stored.get("total_energy")
            self.ttt.restore_counters(steps, float(energy) if isinstance(energy, (int, float)) else 0.0)
            logger.info("V87: odtworzono licznik krokow L0-TTT ze stanu trwalego: steps=%d", steps)

    def _persist_l0_step_counter(self) -> None:
        """V87 (T3): zapisuje biezacy licznik krokow L0-TTT do SQLite (monotonicznie)."""
        kv = getattr(self, "kv", None)
        if kv is None or not hasattr(kv, "save_l0_step_counter"):
            return
        try:
            kv.save_l0_step_counter(int(self.ttt.step_count), float(self.ttt.total_energy))
        except Exception as persist_exc:
            logger.debug("L0-TTT counter persist failed: %s", persist_exc)

    @property
    def weight_baker(self) -> Any:
        """Leniwie inicjalizowany moduł WeightBaker (ADR-001 §4 / brama I1)."""
        if getattr(self, "_weight_baker", None) is None:
            from atlas_memory.l3_procedural.weight_baker import WeightBaker
            self._weight_baker = WeightBaker()
        return self._weight_baker

    @weight_baker.setter
    def weight_baker(self, val: Any) -> None:
        self._weight_baker = val

    @classmethod
    def create_default(
        cls,
        db_path: str = ":memory:",
        qdrant_location: str = ":memory:",
        kuzu_path: Optional[str] = None,
        kuzu_buffer_pool_size: int = KUZU_BUFFER_POOL_BYTES,
        vector_backend=None,
    ) -> HybridMemoryEngine:
        """Tworzy instancję pełnego klastra pamięci z buforem Arrow, Numba JIT i Saga Pattern.

        `vector_backend`: opis z `l2_semantic.vector_backend.VectorBackend`. `None` =
        zachowanie domyślne (PolDense-68M, 512D, `hermes_episodic_memory_68m`).
        """
        if vector_backend is None:
            from atlas_memory.l2_semantic.vector_backend import build_encoder, load_vector_backend

            vb = load_vector_backend()
            try:
                enc = build_encoder(vb)
            except Exception:
                enc = None
            v_store = QdrantVectorStore(
                collection_name=vb.collection,
                location=qdrant_location,
                dimension=vb.dimension,
                encoder=enc,
            )
        else:
            from atlas_memory.l2_semantic.vector_backend import build_encoder

            v_store = QdrantVectorStore(
                collection_name=vector_backend.collection,
                location=qdrant_location,
                dimension=vector_backend.dimension,
                encoder=build_encoder(vector_backend),
            )
        g_store = KuzuGraphStore(db_path=kuzu_path or ":memory:", buffer_pool_size_bytes=kuzu_buffer_pool_size)
        k_store = VerifiedKVStore(db_path=db_path)
        latent = JEPALatentBuffer()
        ttt = TTTLayer(input_dim=latent.state_dim)
        traj_buf = ArrowTrajectoryBuffer(state_dim=latent.state_dim)
        canon = EntityCanonicalizer()
        decay = SalienceDecayEngine()
        comp = ContextCompactor()
        auditor = MemoryAuditor(g_store, k_store, v_store, decay_engine=decay)

        return cls(
            vector_store=v_store,
            graph_client=g_store,
            kv_store=k_store,
            ttt_layer=ttt,
            latent_buffer=latent,
            trajectory_buffer=traj_buf,
            auditor=auditor,
            weight_baker=None,
            canonicalizer=canon,
            decay_engine=decay,
            compactor=comp,
        )

    async def recall(self, query: str, active_entities: List[str]) -> Dict[str, Any]:
        """
        Równoległe odpytanie pamięci:
        1. Kanonizacja aliasów encji (Entity Canonicalizer).
        2. Wektorowe pobranie kontekstu (Qdrant + PolDense-68M ONNX).
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
        """Asynchroniczny commit obserwacji przez EpistemicWritePath oraz kolejkę audytora."""
        if not record.canonical_entity_id:
            record.canonical_entity_id = self.canonicalizer.canonicalize(record.subject)

        # V151.1 (FILAR I): kontraktowa bramka uprawnień state_variable.
        # Cień (SHADOW_INFERENCE) NIE ma prawa zapisać zmiennej stanu bez śladu
        # promocji arbitrażu — nawet gdy obiekt wjechał z gotową flagą (API
        # zewnętrzne, replay, migracja). Bez uprawnień: zwyczajny fakt + adnotacja
        # audytowa. Jedyna procedura wyznaczająca uprawnienie: state_variable_authority.
        from atlas_memory.core.models import SHADOW_PROMOTION_PENDING, state_variable_authority

        if record.is_state_variable and not state_variable_authority(record):
            logger.info(
                "[V151.1/FILAR-I] commit_observation: shadow bez promocji zdemotowany do zwykłego faktu: key=%s",
                record.canonical_entity_id or record.subject,
            )
            record.is_state_variable = False
            record.metadata = dict(record.metadata or {})
            record.metadata.setdefault("shadow_promotion", SHADOW_PROMOTION_PENDING)

        # V84: Bezpośredni zapis przez centralny EpistemicWritePath
        if hasattr(self, "write_path") and self.write_path is not None:
            var_key = record.effective_subject if record.is_state_variable else (
                record.triple_key if hasattr(record, "triple_key") else f"{record.effective_subject}:{record.predicate}"
            )
            metadata = dict(record.metadata or {})
            metadata.setdefault("confidence", record.confidence)
            metadata.setdefault("source_type", getattr(record.source_type, "value", str(record.source_type)))
            metadata.setdefault("predicate", record.predicate)
            metadata.setdefault("subject", record.effective_subject)
            metadata.setdefault("is_state_variable", record.is_state_variable)
            try:
                await self.write_path.write_async(
                    key=var_key,
                    value=record.object,
                    confidence=float(record.confidence if record.confidence is not None else 1.0),
                    metadata=metadata,
                    source_type=getattr(record.source_type, "value", str(record.source_type)),
                    reason="commit_observation",
                    vector_record=record,
                )
            except Exception as exc:
                logger.debug("WritePath commit_observation failed: %s", exc)

        # Domknięcie pętli uczenia modelu świata JEPA na zaobserwowanym stanie
        if self.latent is not None and hasattr(self.latent, "adapt_transition"):
            try:
                prev_st = self.latent.current_state
                obs_vec = self._real_observation_vector(
                    f"{record.subject} {record.predicate} {record.object}",
                    dim=self.latent.state_dim,
                )
                act_params = {"subject": str(record.subject), "object": str(record.object)}
                act = ActionPlan(name=str(record.predicate), parameters=act_params)
                self.latent.adapt_transition(prev_st, act, obs_vec)
            except Exception as jepa_err:
                logger.debug("Closed-loop JEPA adapt_transition on commit_observation error: %s", jepa_err)

    def process_active_sensing(
        self,
        action: ActionPlan,
        observed_vector: Any,
        lr: float = 0.01,
    ) -> float:
        """Domyka pętlę uczenia modelu świata JEPA na zaobserwowanym stanie środowiska."""
        if self.latent is None or not hasattr(self.latent, "adapt_transition"):
            return 0.0
        prev_st = self.latent.current_state
        return self.latent.adapt_transition(prev_st, action, observed_vector, lr=lr)

    def record_mental_transition(self, action: ActionPlan, session_id: str = "default") -> PredictedTransition:
        """Wykonuje krok myślowy JEPA (Numba) i rejestruje go w buforze Arrow (Zero-Copy).

        Fuzja L0 TTT -> L1 JEPA:
        1. Wyprowadza wektor cech akcji x.
        2. Adaptuje TTT online i rzutuje forward -> h_action.
        3. Przekazuje h_action do JEPA predict_transition/step.
        """
        h_action = None
        if self.ttt is not None:
            try:
                target_dim = int(getattr(self.ttt, "input_dim", getattr(self.latent, "state_dim", 32)))
                vec = self._real_observation_vector(action, dim=target_dim)
                if len(vec) != target_dim:
                    if len(vec) < target_dim:
                        vec = list(vec) + [0.0] * (target_dim - len(vec))
                    else:
                        vec = list(vec)[:target_dim]
                if hasattr(self.ttt, "adapt_step"):
                    self.ttt.adapt_step(vec)
                    # V87 (T3): licznik krokow L0-TTT jest trwaly — inaczej get_stats
                    # raportuje 0 po kazdym restarcie daemona mimo zapisow w bazie.
                    self._persist_l0_step_counter()
                if hasattr(self.ttt, "forward"):
                    h_action = self.ttt.forward(vec)
            except Exception as exc:
                logger.debug("TTT online adaptation error in record_mental_transition: %s", exc)

        transition = self.latent.step(action, action_latent=h_action)
        self.trajectory_buffer.append_transition(transition, session_id=session_id)
        return transition

    def _real_observation_vector(self, action: Any, dim: Optional[int] = None) -> List[float]:
        """V84: Wyprowadza semantyczny wektor cech akcji (enkoder ONNX lub leksykalny feature hashing z normą L2)."""
        out_dim = int(dim or getattr(self.latent, "state_dim", 32))
        if isinstance(action, str):
            action_text = action
            is_action_plan = False
        elif hasattr(action, "name"):
            action_text = f"action: {action.name} parameters: {json.dumps(getattr(action, 'parameters', {}) or {}, sort_keys=True, ensure_ascii=False, default=str)}"
            is_action_plan = True
        elif isinstance(action, dict):
            action_text = json.dumps(action, sort_keys=True, ensure_ascii=False, default=str)
            is_action_plan = False
        else:
            action_text = str(action)
            is_action_plan = False

        # 1. Próba użycia rzeczywistego enkodera gęstego (np. PolDense-68M)
        encoder = getattr(getattr(self, "vector_store", None), "encoder", None)
        if encoder is not None:
            try:
                if hasattr(encoder, "encode"):
                    vec = encoder.encode(action_text)
                    if len(vec) >= out_dim:
                        if is_action_plan and getattr(action, "action_encoding", None) is None:
                            action.action_encoding = "dense_encoder"
                        arr = np.array(vec[:out_dim], dtype=np.float64)
                        norm = np.linalg.norm(arr)
                        if norm > 1e-9:
                            arr /= norm
                        return [float(v) for v in arr]
            except Exception as enc_err:
                logger.debug("Encoder dense extraction failed, using deterministic term projection: %s", enc_err)

        # 2. Deterministyczna projekcja leksykalna (Feature Hashing Trick z normalizacją L2)
        if is_action_plan and getattr(action, "action_encoding", None) is None:
            action.action_encoding = "term_projection"
        vec = np.zeros(out_dim, dtype=np.float64)
        tokens = re.findall(r"\w+", action_text.lower())
        for token in tokens:
            h = zlib.crc32(token.encode("utf-8")) % out_dim
            vec[h] += 1.0

        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec /= norm
        else:
            vec.fill(1.0 / np.sqrt(out_dim))
        return [float(v) for v in vec]

    async def _fetch_vectors(self, query: str) -> List[Dict[str, Any]]:
        return await self.vector_store.search(query, top_k=5)

    async def _fetch_graph_relations(self, entities: List[str]) -> Dict[str, Any]:
        return await self.graph.get_subgraph_relations(entities, max_depth=2)

    async def _fetch_state_vars(self, entities: List[str]) -> Dict[str, Any]:
        return await self.kv.get_states(entities)

    def start_worker(self) -> asyncio.Task:
        """Kompatybilność wsteczna: w V153 zapis jest bezpośredni przez EpistemicWritePath."""
        async def _noop_worker():
            try:
                while True:
                    await asyncio.sleep(3600)
            except asyncio.CancelledError:
                pass

        if getattr(self, "_worker_task", None) is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(_noop_worker())
        return self._worker_task

    async def stop_worker(self) -> None:
        """Kompatybilność wsteczna: zatrzymanie workera."""
        if getattr(self, "_worker_task", None) is not None and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    async def process_all_pending(self) -> None:
        """Kompatybilność wsteczna: w V153 wszystkie obserwacje są natychmiast utrwalane przez EpistemicWritePath."""
        pass
