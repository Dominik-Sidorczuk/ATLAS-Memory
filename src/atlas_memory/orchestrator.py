from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from atlas_memory.cognitive.calibration_engine import is_symbolic_or_identifier
from atlas_memory.cognitive.conflict_arbiter import ConflictArbiter
from atlas_memory.cognitive.retrieval.gate import should_retrieve_query
from atlas_memory.cognitive.retrieval.prefetch import PrefetchEngine
from atlas_memory.cognitive.retrieval.score import score_candidates
from atlas_memory.core.models import EpistemicSource, MemoryRecord
from atlas_memory.extensions.canonicalizer import EntityCanonicalizer
from atlas_memory.extensions.decay_scorer import SalienceDecayEngine

logger = logging.getLogger(__name__)


class MemoryOrchestrator:
    """Warstwa Decyzyjna i Fasada Polityki Pamięci dla Hermes Agent."""

    def __init__(self, engine: Optional[Any] = None, mnemosyne_client: Optional[Any] = None, canonicalizer: Optional[EntityCanonicalizer] = None, decay_engine: Optional[SalienceDecayEngine] = None, compactor: Optional[Any] = None, shadow_extractor: Optional[Callable[[str, str], List[Dict[str, Any]]]] = None, max_retrieval_tokens: int = 1500, calibration_engine: Optional[Any] = None, enable_calibrated_thresholds: Optional[bool] = None) -> None:
        self.engine, self.mnemosyne = (engine or mnemosyne_client), (mnemosyne_client or engine)
        self.canonicalizer = canonicalizer or getattr(engine, "canonicalizer", None) or EntityCanonicalizer()
        self.decay_engine = decay_engine or getattr(engine, "decay_engine", None) or SalienceDecayEngine()
        self.compactor, self.shadow_extractor, self.max_retrieval_tokens = compactor, shadow_extractor, max_retrieval_tokens
        from atlas_memory.core.config import get_config
        eff_calib_flag = get_config().enable_calibrated_thresholds if enable_calibrated_thresholds is None else bool(enable_calibrated_thresholds)
        self.enable_calibrated_thresholds, self.calibration_engine = eff_calib_flag, calibration_engine
        self._prefetch_engine = PrefetchEngine(engine=self.engine, calibration_engine=self.calibration_engine, default_token_budget=self.max_retrieval_tokens)
        self.stats = {"total_turns": 0, "retrieval_skipped_turns": 0, "retrieval_executed_turns": 0, "shadow_facts_extracted": 0, "tokens_saved_estimate": 0}

    def get_relevance_threshold(self, query: str = "") -> float:
        if getattr(self, "enable_calibrated_thresholds", False) and getattr(self, "calibration_engine", None) is not None:
            return self.calibration_engine.get_threshold(query_type="symbolic" if is_symbolic_or_identifier(query) else "prose", default=0.25)
        return 0.25

    def build_cache_contract_prefix(self, profile_state: Optional[Dict[str, Any]] = None, system_rules: Optional[List[str]] = None) -> str:
        from atlas_memory.hermes.prefix_guard import PrefixCacheGuard
        return PrefixCacheGuard.build_immutable_prefix(verified_state=profile_state or {}, system_rules=system_rules or ["You are an autonomous engineering agent (Hermes Agent).", "Memory Protocol: Dynamic facts are injected via Mnemosyne / search_memory.", "Deterministic Execution: Rely on Verified Source of Truth state variables."])

    def should_retrieve(self, message: str, explicit_entities: Optional[List[str]] = None) -> Tuple[bool, List[str], str]:
        self.stats["total_turns"] += 1
        can_ret, ents, reason = should_retrieve_query(message, explicit_entities=explicit_entities, canonicalizer=self.canonicalizer)
        self.stats["retrieval_executed_turns" if can_ret else "retrieval_skipped_turns"] += 1
        if not can_ret:
            self.stats["tokens_saved_estimate"] += self.max_retrieval_tokens
        return can_ret, ents, reason

    def epistemic_rank(self, records: List[MemoryRecord], query: str, current_time: Optional[float] = None) -> List[Tuple[MemoryRecord, float]]:
        thresh = self.get_relevance_threshold(query)
        cands = [{"record": r, "subject": r.effective_subject, "predicate": r.predicate, "object": r.object, "vector_score": float((r.metadata or {}).get("vector_score", (r.metadata or {}).get("score", 0.0))), "confidence": float(r.confidence if r.confidence is not None else 1.0), "source_type": getattr(r.source_type, "value", str(r.source_type)), "metadata": dict(r.metadata or {}), "timestamp": float(r.timestamp or 0.0)} for r in records]
        scored = score_candidates(query=query, candidates=cands)
        res = [(c["record"], float(c.get("score", 0.0)) * (0.40 if (float((c["record"].metadata or {}).get("vector_score", 0.0)) < thresh and float(c.get("_lex_score", c.get("lexical_score", 0.0))) <= 0.0) else 1.0)) for c in scored if "record" in c]
        seen = {id(r) for r, _ in res}
        return res + [(r, 0.0) for r in records if id(r) not in seen]

    def apply_token_budget(self, ranked_records: List[Tuple[MemoryRecord, float]], max_tokens: Optional[int] = None, strategy: str = "knapsack") -> Dict[str, Any]:
        from atlas_memory.cognitive.retrieval.render import estimate_tokens, pack_knapsack_01, render_context_block

        budget = max_tokens or self.max_retrieval_tokens
        if not ranked_records or budget <= 0:
            return {"selected_facts": [], "formatted_context": "", "estimated_tokens": 0, "budget_tokens": budget}
        candidates = [
            {
                "record": r,
                "subject": r.effective_subject,
                "predicate": r.predicate,
                "object": r.object,
                "confidence": r.confidence,
                "source_type": getattr(r.source_type, "value", str(r.source_type)),
                "score": float(s),
                "is_superseded": bool((r.metadata or {}).get("is_superseded")),
                "metadata": dict(r.metadata or {}),
            }
            for r, s in ranked_records
        ]
        packed = pack_knapsack_01(candidates, budget=budget)
        return {
            "selected_facts": [
                {
                    "subject": it["subject"],
                    "predicate": it["predicate"],
                    "object": it["object"],
                    "confidence": it["confidence"],
                    "source_type": it["source_type"],
                    "score": it["score"],
                }
                for it in packed
            ],
            "formatted_context": render_context_block(packed),
            "estimated_tokens": sum(estimate_tokens(it) for it in packed),
            "budget_tokens": budget,
        }

    async def orchestrated_recall(self, user_message: str = "", explicit_entities: Optional[List[str]] = None, **kwargs: Any) -> Dict[str, Any]:
        eff_query = kwargs.get("query") if kwargs.get("query") is not None else user_message
        should_run, entities, reason = self.should_retrieve(eff_query, explicit_entities=explicit_entities)
        if not should_run:
            return {"retrieval_skipped": True, "skipped": True, "reason": reason, "canonical_entities": entities, "context_block": "", "matched_facts_count": 0, "records": [], "selected_facts": []}
        fn = getattr(self.engine, "recall", None) or getattr(self.mnemosyne, "recall", None)
        if fn is not None and not (hasattr(self.engine, "kv") and self.engine.kv is not None):
            raw = await fn(eff_query, active_entities=entities)
            recs = [MemoryRecord(subject=r["subject"], predicate=r.get("predicate", ""), object=str(r.get("object", "")), confidence=float(r.get("confidence", 1.0)), source_type=EpistemicSource.TOOL_OUTPUT) for r in raw.get("graph_topology", {}).get("relations", [])] + [MemoryRecord(**v["record"]) for v in raw.get("semantic_context", []) if v.get("record")] + [MemoryRecord(subject=k, predicate="state_variable", object=str(v.get("value", "") if isinstance(v, dict) else v), confidence=float(v.get("confidence", 1.0) if isinstance(v, dict) else 1.0), source_type=EpistemicSource.USER_EXPLICIT) for k, v in raw.get("verified_state", {}).items()] if isinstance(raw, dict) else [r if isinstance(r, MemoryRecord) else MemoryRecord(**r) for r in raw]
            ranked = self.epistemic_rank(recs, query=eff_query)
            budgeted = self.apply_token_budget(ranked, max_tokens=kwargs.get("token_budget") or self.max_retrieval_tokens)
            return {"retrieval_skipped": False, "skipped": False, "reason": reason, "canonical_entities": entities, "context_block": budgeted["formatted_context"], "matched_facts_count": len(budgeted["selected_facts"]), "estimated_tokens": budgeted["estimated_tokens"], "records": [r for r, _ in ranked], "selected_facts": budgeted["selected_facts"]}
        engine_p = getattr(self, "_prefetch_engine", None) or PrefetchEngine(engine=self.engine, calibration_engine=self.calibration_engine, default_token_budget=self.max_retrieval_tokens)
        res = await engine_p.execute({"query": eff_query, "session_id": kwargs.get("session_id"), "token_budget": kwargs.get("token_budget") or self.max_retrieval_tokens, "limit": kwargs.get("limit", 15)})
        res["retrieval_skipped"] = res.get("skipped", False)
        raw_recs = res.get("records", [])
        res["selected_facts"] = res.get("selected_facts") or [{"subject": r.get("subject", ""), "predicate": r.get("predicate", ""), "object": r.get("object", ""), "confidence": r.get("confidence", 1.0), "score": r.get("score", 0.0)} for r in raw_recs if isinstance(r, dict)]
        res["matched_facts_count"] = len(res["selected_facts"])
        res["records"] = [r if isinstance(r, MemoryRecord) else MemoryRecord(subject=r.get("subject", ""), predicate=r.get("predicate", "stated_memory"), object=str(r.get("object", "")), confidence=float(r.get("confidence", 1.0)), source_type=EpistemicSource.USER_EXPLICIT if r.get("source_type") not in EpistemicSource._value2member_map_ else EpistemicSource(r.get("source_type")), metadata=r.get("metadata", {}), timestamp=float(r.get("timestamp", 0.0))) for r in raw_recs]
        return res

    async def shadow_reconcile(self, user_msg: str, agent_response: str, session_id: str = "default_session", triple_add_fn: Optional[Callable] = None) -> List[MemoryRecord]:
        raw = []
        if self.shadow_extractor:
            try:
                if inspect.iscoroutinefunction(self.shadow_extractor):
                    raw = await self.shadow_extractor(user_msg, agent_response)
                else:
                    raw = await asyncio.to_thread(self.shadow_extractor, user_msg, agent_response)
            except Exception:
                raw = []
        committed, add_fn = [], (triple_add_fn or (getattr(self.mnemosyne, "triple_add", None) if self.mnemosyne else None))
        for it in (raw or self._fallback_extract_facts(user_msg, agent_response)):
            raw_s, pred, obj = it.get("subject", "").strip(), it.get("predicate", "").strip(), str(it.get("object", "")).strip()
            if not (raw_s and pred and obj):
                continue
            # V151 (Wektor I, H1): detoksykacja proweniencji cieniowej.
            # Ekstrakcja LLM-a NIGDY nie dowodzi woli użytkownika — dawny
            # fallback 'user_explicit' (linia 134) fałszował dowodzenie
            # najwyższej rangi; od V151 fallback = SHADOW_INFERENCE.
            # Jawny claim 'user_explicit' ze shadow-a jest DEMOWANY do
            # SHADOW_INFERENCE; 'tool_output'/'agent_inference' pozostają.
            st_v = it.get("source_type", EpistemicSource.SHADOW_INFERENCE.value)
            st = EpistemicSource(st_v) if st_v in EpistemicSource._value2member_map_ else EpistemicSource.SHADOW_INFERENCE
            if st == EpistemicSource.USER_EXPLICIT:
                st = EpistemicSource.SHADOW_INFERENCE
            # V151.1 (FILAR I): typologiczna bariera aktu mowy. Trójka o temacie
            # będącym aktem konwersacyjnym (meta-pytanie, discourse-garbage) nie
            # podlega kanonizacji — wyrzucamy ją zanim trafi do arbitra/commita.
            from atlas_memory.core.models import (
                SHADOW_PROMOTION_ACTIONS,
                SpeechAct,
                classify_speech_act,
            )
            speech_act = classify_speech_act(raw_s, pred, obj)
            if speech_act == SpeechAct.CONVERSATIONAL_TURN:
                logger.info("[V151.1/FILAR-I] Shadow conversational turn dropped subject=%s", raw_s)
                continue
            rec = MemoryRecord(subject=raw_s, canonical_entity_id=self.canonicalizer.canonicalize(raw_s), predicate=pred, object=obj, confidence=float(it.get("confidence", 1.0)), source_type=st, timestamp=time.time(), is_state_variable=bool(it.get("is_state_variable", False)), source_session_id=session_id, speech_act=speech_act)
            # V151 (Wektor I, H1): zmienne stanu cieniowe MUSZĄ przejść ConflictArbiter
            # — od teraz brama epistemiczna jest obowiązkowa (default: ConflictArbiter()).
            if rec.is_state_variable:
                arb = getattr(self, "arbiter", None) or ConflictArbiter()
                try:
                    arb_res = arb.arbitrate(key=str(rec.canonical_entity_id or raw_s), val=rec.object, confidence=rec.confidence, metadata={"origin": "shadow_reconcile", "source_type": st.value}, existing_items={})
                except Exception as exc:
                    logger.warning("Shadow ConflictArbiter failed for %s — blokada zapisu: %s", rec.canonical_entity_id, exc)
                    arb_res = None
                if arb_res is None or str(arb_res.action) in ("reject", "reject_conflict", "block"):
                    logger.info("[V151] Shadow state write odrzucony przez arbiter: key=%s action=%s", rec.canonical_entity_id, getattr(arb_res, "action", "arbiter_error"))
                    continue
                action_str = str(arb_res.action)
                rec.metadata["shadow_arbitration"] = action_str
                # V151.1 (FILAR I): bramka uprawnień state_variable dla cienia.
                # Werdykt arbitra, który ROZSTRZYGA zapis (czysta zgoda lub
                # rewizja poglądu): accept / supersede / promote / accept_conflict.
                # Werdykt NIEROZSTRZYGNIĘTY (warn_conflict) → zostaje zwykłym
                # faktem (is_state_variable=False) z shadow_promotion='pending_arbitration'
                # dla audytu; engine.commit_observation powtórzy tę bramkę
                # kontraktowo, jeśli obiekt przejdzie z zewnątrz.
                is_conflict = bool(getattr(arb_res, "is_conflict", False))
                is_promoted = (not is_conflict) and (
                    action_str in SHADOW_PROMOTION_ACTIONS
                    or action_str in ("accept", "supersede")
                )
                if is_promoted:
                    rec.metadata["shadow_promotion"] = "arbitration_promoted"
                else:
                    rec.metadata["shadow_promotion"] = "pending_arbitration"
                    rec.is_state_variable = False
                    logger.info(
                        "[V151.1/FILAR-I] Shadow state demoted to plain fact: key=%s action=%s conflict=%s",
                        rec.canonical_entity_id, action_str, is_conflict,
                    )
            if add_fn:
                try:
                    await add_fn(subject=rec.canonical_entity_id, predicate=pred, object_=obj, confidence=rec.confidence, source=st.value, supersede=True)
                except Exception as exc:
                    logger.debug("Mnemosyne triple_add failed: %s", exc)
            if self.engine and hasattr(self.engine, "commit_observation"):
                try:
                    await self.engine.commit_observation(rec)
                except Exception as exc:
                    logger.debug("Engine commit_observation failed: %s", exc)
            committed.append(rec)
            self.stats["shadow_facts_extracted"] += 1
        return committed

    def _fallback_extract_facts(self, user_msg: str, agent_response: str = "") -> List[Dict[str, Any]]:
        text = (user_msg or "").strip()
        if not text:
            return []
        matches = list(re.findall(r"(?:mój |moje )?(\w[\w\s]{1,20}?)\s+(?:to|jest|wynosi)\s+([\w\.\-\/]+)", text, re.IGNORECASE))
        matches.extend(re.findall(r"(?:zmień|ustaw)\s+(\w[\w\s]{1,20}?)\s+na\s+([\w\.\-\/]+)", text, re.IGNORECASE))
        return [
            {
                "subject": subj.strip(),
                "predicate": "state_value",
                "object": obj.strip(),
                "source_type": "user_explicit",
                "confidence": 0.95,
                "is_state_variable": True,
            }
            for subj, obj in matches
        ]

    def prune_stale_facts(self, facts: List[MemoryRecord], threshold: float = 0.25, current_time: Optional[float] = None) -> Tuple[List[MemoryRecord], List[MemoryRecord]]:
        now = current_time or time.time()
        return [f for f in facts if not self.decay_engine.should_prune(f, current_time=now)], [f for f in facts if self.decay_engine.should_prune(f, current_time=now)]
