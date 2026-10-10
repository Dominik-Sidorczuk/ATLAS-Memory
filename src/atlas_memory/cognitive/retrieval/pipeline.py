"""V81/V106 Unified RetrievalPipeline inside the cognitive/retrieval package."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from atlas_memory.cognitive.models import RetrievalContext, RetrievalRecord
from atlas_memory.cognitive.retrieval.gate import (
    is_continuation_query,
    is_trivial_prompt,
    should_retrieve_query,
)
from atlas_memory.cognitive.retrieval.prefetch import PrefetchEngine
from atlas_memory.cognitive.retrieval.render import (
    estimate_tokens,
    pack_knapsack_01,
    render_context_block,
)
from atlas_memory.cognitive.retrieval.score import (
    compute_content_fingerprint,
    compute_record_rank,
    get_authority_score,
    get_record_authority_score,
)
from atlas_memory.cognitive.retrieval.strategies.pipeline import ScoringPipeline
from atlas_memory.extensions.compactor import ContextCompactor

__all__ = [
    "RetrievalPipeline",
    "ScoringPipeline",
    "compute_content_fingerprint",
    "get_authority_score",
    "get_record_authority_score",
]


class RetrievalPipeline:
    """Unified RetrievalPipeline orchestrating gate, candidates, scoring, and 0/1 knapsack rendering."""

    generate_context_block = staticmethod(render_context_block)
    estimate_tokens = staticmethod(estimate_tokens)
    pack_knapsack_01 = staticmethod(pack_knapsack_01)
    compute_record_rank = staticmethod(compute_record_rank)
    is_continuation_query = staticmethod(is_continuation_query)
    is_trivial_prompt = staticmethod(is_trivial_prompt)
    get_authority_score = staticmethod(get_authority_score)

    @classmethod
    def format_inprocess_prefetch(
        cls,
        orchestrator: Any,
        records: List[Any],
        query: str,
        max_tokens: int = 1500,
    ) -> Dict[str, Any]:
        if orchestrator is None and not records:
            return {"formatted_context": "", "selected_facts": [], "estimated_tokens": 0}
        if (
            orchestrator is not None
            and hasattr(orchestrator, "epistemic_rank")
            and hasattr(orchestrator, "apply_token_budget")
        ):
            ranked = orchestrator.epistemic_rank(records, query=query)
            return orchestrator.apply_token_budget(ranked, max_tokens=max_tokens)

        from atlas_memory.core.models import MemoryRecord

        mem_records = [
            r
            if isinstance(r, MemoryRecord)
            else MemoryRecord(
                subject=str(
                    getattr(r, "subject", None)
                    or (r.get("subject") if isinstance(r, dict) else "rec")
                ),
                predicate=str(
                    getattr(r, "predicate", None)
                    or (r.get("predicate") if isinstance(r, dict) else "stated")
                ),
                object=str(
                    getattr(r, "object", None)
                    or (r.get("object") if isinstance(r, dict) else str(r))
                ),
            )
            for r in (records or [])
        ]
        packed = pack_knapsack_01(mem_records, budget=max_tokens, query=query)
        ctx = render_context_block(packed)
        return {
            "selected_facts": packed,
            "formatted_context": ctx,
            "estimated_tokens": sum(estimate_tokens(r) for r in packed),
        }

    def __init__(
        self,
        engine: Any = None,
        orchestrator: Any = None,
        vector_store: Any = None,
        kv_store: Any = None,
        hot_buffer: Any = None,
        graph_store: Any = None,
        calibration_engine: Any = None,
        enable_calibrated_thresholds: bool = True,
        default_token_budget: int = 1500,
        **kwargs: Any,
    ) -> None:
        self.engine, self.orchestrator, self.vector_store = (
            engine or self,
            orchestrator,
            vector_store,
        )
        self.kv_store, self.hot_buffer, self.graph_store = (
            kv_store,
            hot_buffer,
            graph_store,
        )
        self.calibration_engine, self.enable_calibrated_thresholds = (
            calibration_engine,
            enable_calibrated_thresholds,
        )
        self.default_token_budget = default_token_budget
        self.compactor = kwargs.get("compactor") or ContextCompactor()
        self._engine = PrefetchEngine(
            engine=self.engine,
            calibration_engine=calibration_engine,
            default_token_budget=default_token_budget,
        )

    def should_retrieve(
        self, query: str, explicit_entities: Optional[List[str]] = None
    ) -> tuple[bool, list, str]:
        return should_retrieve_query(query, explicit_entities)

    async def execute_prefetch(
        self,
        params: Dict[str, Any],
        engine: Optional[Any] = None,
        orchestrator: Optional[Any] = None,
        hot_buffer: Optional[Any] = None,
        check_sync_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        p = {"enable_calibrated_thresholds": self.enable_calibrated_thresholds, **params}
        return await self._engine.execute(
            params=p,
            engine=engine or self.engine,
            orchestrator=orchestrator or self.orchestrator,
            hot_buffer=hot_buffer or self.hot_buffer,
            graph_store=getattr(self, "graph_store", None),
            calibration_engine=self.calibration_engine,
            check_sync_callback=check_sync_callback,
        )

    async def retrieve(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 15,
        token_budget: Optional[int] = None,
        force: bool = False,
    ) -> RetrievalContext:
        res = await self.execute_prefetch(
            {
                "query": query,
                "session_id": session_id,
                "limit": limit,
                "token_budget": token_budget or self.default_token_budget,
                "force": force,
            }
        )
        records = [
            RetrievalRecord(
                subject=r.get("subject", ""),
                predicate=r.get("predicate", "stated_memory"),
                object=str(r.get("object", "")),
                confidence=float(r.get("confidence", 1.0)),
                source_type=str(r.get("source_type", "user_explicit")),
                score=float(r.get("score", 0.0)),
                metadata=dict(r.get("metadata") or {}),
            )
            for r in res.get("records", [])
        ]
        return RetrievalContext(
            query=query,
            session_id=session_id or "default",
            records=records,
            count=len(records),
            skipped=res.get("skipped", False),
            metadata={"reason": res.get("reason", "")},
        )

