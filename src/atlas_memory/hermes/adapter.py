"""
Hermes Memory Adapter: Primary integration adapter connecting ATLAS cognitive layers with Hermes Agent.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any, Callable, Dict, List, Optional

from atlas_memory.active.prediction_error import ActiveSensingEngine, PredictionError
from atlas_memory.active.shadow_worker import OmniRouteShadowWorker
from atlas_memory.causal.retro_causal_edge import RetroCausalEngine
from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.orchestrator import MemoryOrchestrator
from atlas_memory.telemetry.cache_monitor import CacheHitMonitor

logger = logging.getLogger(__name__)


class HermesMemoryAdapter:
    """
    Główny Adapter Integracyjny Pamięci ATLAS dla Hermes Agent.
    
    Łączy wszystkie podsystemy w spójny interfejs dla agenta:
    1. orchestrated_search: Retrieval Policy Gate -> Dual-Engine -> Epistemic Re-Ranker -> Token Governor.
    2. orchestrated_ingest: asynchroniczne wrzucenie tury do kolejki Shadow Workera.
    3. get_cache_prefix: deterministyczny blok Prompt Caching z telemetrią CacheHitMonitor.
    4. what_if_analysis: symulacja przyczynowo-skutkowa Retro-Causal Edge (JEPA).
    5. process_telemetry_observation: Predictive Coding bez logitów (Active Sensing).
    """

    def __init__(
        self,
        orchestrator: MemoryOrchestrator,
        shadow_worker: Optional[OmniRouteShadowWorker] = None,
        cache_monitor: Optional[CacheHitMonitor] = None,
        causal_engine: Optional[RetroCausalEngine] = None,
        active_sensing: Optional[ActiveSensingEngine] = None,
    ):
        self.orchestrator = orchestrator
        self.shadow_worker = shadow_worker or OmniRouteShadowWorker(orchestrator=self.orchestrator)
        self.cache_monitor = cache_monitor or CacheHitMonitor()
        self.causal_engine = causal_engine or RetroCausalEngine(graph_client=self.orchestrator.engine.graph if hasattr(self.orchestrator, "engine") and self.orchestrator.engine else None)
        self.active_sensing = active_sensing or ActiveSensingEngine()
        self._last_prefix_hash: Optional[str] = None
        self._ensure_shadow_worker_started()

    def _ensure_shadow_worker_started(self) -> None:
        """Lazy start OmniRouteShadowWorker."""
        worker = getattr(self, "shadow_worker", None)
        if worker is None:
            return
        task = getattr(worker, "_worker_task", None)
        if task is not None and not task.done():
            return  # już działa
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("ATLAS: brak działającej pętli zdarzeń — shadow worker deferred do pierwszego ingest")
            return
        try:
            worker.start()
            logger.info("ATLAS shadow worker started (turn queue consumer)")
        except Exception as exc:
            logger.warning("ATLAS shadow worker start failed: %s", exc)

    @classmethod
    def create_default(
        cls,
        db_path: str = ":memory:",
        qdrant_location: str = ":memory:",
        omniroute_api_base: str = "http://localhost:20128/v1",
        omniroute_model: str = "qwen-mini",
        http_client_fn: Optional[Callable] = None,
    ) -> HermesMemoryAdapter:
        """Tworzy gotową, kompletną instancję adaptera ATLAS dla Hermes Agent."""
        engine = HybridMemoryEngine.create_default(db_path=db_path, qdrant_location=qdrant_location)
        orchestrator = MemoryOrchestrator(engine=engine)
        shadow = OmniRouteShadowWorker(
            orchestrator=orchestrator,
            api_base=omniroute_api_base,
            model=omniroute_model,
            http_client_fn=http_client_fn,
        )
        monitor = CacheHitMonitor()
        causal = RetroCausalEngine(graph_client=engine.graph, latent_buffer=engine.latent)
        sensing = ActiveSensingEngine()

        return cls(
            orchestrator=orchestrator,
            shadow_worker=shadow,
            cache_monitor=monitor,
            causal_engine=causal,
            active_sensing=sensing,
        )

    def get_cache_prefix(
        self,
        profile_state: Optional[Dict[str, Any]] = None,
        rules: Optional[List[str]] = None,
        model_name: str = "deepseek-chat",
    ) -> str:
        """
        Zwraca deterministyczny prefiks promptu i rejestruje telemetrię cache.
        """
        prefix_str = self.orchestrator.build_cache_contract_prefix(profile_state, rules)
        prefix_hash = hashlib.sha256(prefix_str.encode("utf-8")).hexdigest()[:16]

        is_hit = (self._last_prefix_hash == prefix_hash)
        self._last_prefix_hash = prefix_hash

        self.cache_monitor.record_turn(
            model=model_name,
            prefix_hash=prefix_hash,
            cached=is_hit,
            tokens_in_prefix=len(prefix_str) // 4,
        )

        return prefix_str

    async def orchestrated_search(
        self,
        query: str,
        explicit_entities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Główna metoda odpytania pamięci:
        - Decyzja Gate: czy szukać?
        - Jeśli TAK: pobranie podgrafu + wektorów -> Veracity Rank -> Budget Governor.
        """
        return await self.orchestrator.orchestrated_recall(
            user_message=query,
            explicit_entities=explicit_entities,
        )

    async def orchestrated_ingest(
        self,
        user_msg: str,
        agent_response: str,
        session_id: str = "default",
    ) -> None:
        """
        Asynchronicznie wrzuca turę do kolejki Shadow Workera (zero opóźnień w odpowiedzi).
        """
        self._ensure_shadow_worker_started()
        await self.shadow_worker.enqueue_turn(user_msg, agent_response, session_id=session_id)

    async def what_if_analysis(
        self,
        entity: str,
        action: str,
        depth: int = 2,
    ) -> Dict[str, Any]:
        """
        Analiza ekstrapolacji skutków akcji 'Co się stanie jeśli...' (Retro-Causal Edge + JEPA).
        """
        result = await self.causal_engine.evaluate_what_if(entity, action, depth=depth)
        return result.model_dump()

    async def process_telemetry_observation(
        self,
        entity: str,
        predicate: str,
        value: Any,
    ) -> Optional[PredictionError]:
        """
        Weryfikacja oczekiwań telemetrycznych środowiska (Active Sensing).
        """
        triple_add_fn = None
        if hasattr(self.orchestrator, "engine") and self.orchestrator.engine:
            async def _engine_triple_add(subject, predicate, object_, confidence, source, supersede):
                from atlas_memory.models import EpistemicSource, MemoryRecord
                await self.orchestrator.engine.commit_observation(MemoryRecord(
                    subject=subject,
                    predicate=predicate,
                    object=str(object_),
                    confidence=confidence,
                    source_type=EpistemicSource.TOOL_OUTPUT,
                    is_state_variable=True,
                ))
            triple_add_fn = _engine_triple_add

        return await self.active_sensing.process_observation(
            observed_entity=entity,
            observed_predicate=predicate,
            observed_value=value,
            mnemosyne_triple_add_fn=triple_add_fn,
        )

    def get_telemetry_report(self) -> Dict[str, Any]:
        """Zwraca raport telemetryczny działania pamięci ATLAS."""
        return {
            "cache_stats": self.cache_monitor.monthly_report(),
            "orchestrator_stats": self.orchestrator.stats,
            "active_sensing_errors_count": len(self.active_sensing.error_history),
        }

