"""V81 Prefetch: orkiestracja pobierania, punktacji, bramkowania i renderu."""
from __future__ import annotations

import logging
import os
import time
import weakref
from typing import Any, Dict, Optional

import numpy as np

from atlas_memory.cognitive.calibration_engine import is_symbolic_or_identifier
from atlas_memory.cognitive.retrieval.candidates import gather_candidates
from atlas_memory.cognitive.retrieval.gate import evaluate_gate
from atlas_memory.cognitive.retrieval.render import estimate_tokens, render_context_block
from atlas_memory.cognitive.retrieval.score import score_candidates
from atlas_memory.core.agi_logger import log_recall, log_recall_skip
from atlas_memory.l1_working.hot_buffer import PERSISTENT_KEY_PREFIXES

logger = logging.getLogger(__name__)


_INSTANCES_DEDUP_CACHE: weakref.WeakKeyDictionary[PrefetchEngine, Dict[str, tuple[float, Dict[str, Any]]]] = weakref.WeakKeyDictionary()
_DEDUP_TTL_SECONDS: float = 1.5  # 1500ms Sliding Dedup Buffer (DIRECTIVE 3)


class PrefetchEngine:
    """Orkiestrator prefetchu pamięci zgodnie z architekturą V81."""

    def __init__(
        self,
        engine: Optional[Any] = None,
        calibration_engine: Optional[Any] = None,
        default_limit: int = 50,
        default_token_budget: int = 1500,
    ) -> None:
        self.engine = engine
        self.calibration_engine = calibration_engine
        self.default_limit = default_limit
        self.default_token_budget = default_token_budget

    @property
    def _dedup_cache(self) -> Dict[str, tuple[float, Dict[str, Any]]]:
        if self not in _INSTANCES_DEDUP_CACHE:
            _INSTANCES_DEDUP_CACHE[self] = {}
        return _INSTANCES_DEDUP_CACHE[self]

    @_dedup_cache.setter
    def _dedup_cache(self, val: Any) -> None:
        _INSTANCES_DEDUP_CACHE[self] = val

    @property
    def dedup_ttl_seconds(self) -> float:
        return _DEDUP_TTL_SECONDS

    # ─────────────────────────────────────────────────────────────────────────
    # v121 (T2) — execute decomposition: monolit CC=77 → helpery CC≤15.
    # V119 short_circuit logging (F-035) zachowany verbatim.
    # Kontrakt Golden Master (KNOWN_GOOD_HASH V116 d405668) zachowany bit-for-bit.
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_context(
        self,
        params: Dict[str, Any],
        engine: Optional[Any],
        calibration_engine: Optional[Any],
        orchestrator: Optional[Any],
    ) -> tuple[str, Any, int, int, bool, Any, Any, Any, bool]:
        """Parsuje params i rozwiązuje efektywne zależności (engine/cal/orchestrator)."""
        query = str(params.get("query") or "").strip()
        session_id = params.get("session_id")
        limit = int(params.get("limit") or self.default_limit)
        budget = int(params.get("token_budget") or params.get("budget") or params.get("max_tokens") or self.default_token_budget)
        force = bool(params.get("force", False))

        eff_engine = engine or self.engine
        eff_cal = calibration_engine or self.calibration_engine or getattr(eff_engine, "calibration_engine", None)
        eff_orch = orchestrator or getattr(eff_engine, "orchestrator", None)
        en_cal = bool(params.get("enable_calibrated_thresholds", getattr(self, "enable_calibrated_thresholds", getattr(eff_engine, "enable_calibrated_thresholds", True))))
        return query, session_id, limit, budget, force, eff_engine, eff_cal, eff_orch, en_cal

    @staticmethod
    def _check_sync(check_sync_callback: Optional[Any]) -> None:
        """Wywołuje check_sync_callback jeśli podpięty (błędy → debug log)."""
        if check_sync_callback is not None:
            try:
                check_sync_callback()
            except Exception as exc:
                logger.debug("check_sync_callback error: %s", exc)

    @staticmethod
    def _orchestrator_policy_skip(eff_orch: Optional[Any], query: str, force: bool) -> Optional[Dict[str, Any]]:
        """0. Orchestrator policy gate (jeśli podpięty). Zwraca skip dict lub None."""
        if eff_orch and hasattr(eff_orch, "should_retrieve") and not force:
            try:
                res_orch = eff_orch.should_retrieve(query)
                if isinstance(res_orch, tuple) and len(res_orch) >= 3:
                    can_ret, _, orch_reason = res_orch[:3]
                elif isinstance(res_orch, tuple) and len(res_orch) == 2:
                    can_ret, orch_reason = res_orch[0], str(res_orch[1])
                elif isinstance(res_orch, bool):
                    can_ret, orch_reason = res_orch, "orchestrator_policy"
                else:
                    can_ret, orch_reason = True, "ok"
                if not can_ret:
                    log_recall_skip(
                        log=logger,
                        query=query,
                        mode="POLICY",
                        pool_counts={},
                        gate_reason=orch_reason,
                        threshold=0.0,
                        max_score=0.0,
                        score_clause="(orchestrator_policy_gate)",
                    )
                    return {
                        "records": [], "context_block": "", "query": query, "count": 0,
                        "skipped": True, "reason": orch_reason, "threshold": 0.0,
                        "threshold_selection_reason": "orchestrator_policy_gate",
                    }
            except Exception as exc:
                logger.debug("orchestrator should_retrieve error: %s", exc)
        return None

    @staticmethod
    def _fast_policy_skip(query: str, force: bool) -> Optional[Dict[str, Any]]:
        """Fast exit dla system notification prefixes (bez zbierania puli)."""
        if not force:
            from atlas_memory.cognitive.retrieval.gate import _SYSTEM_NOTIFICATION_PREFIXES
            if query.lower().strip().startswith(_SYSTEM_NOTIFICATION_PREFIXES):
                log_recall_skip(
                    log=logger,
                    query=query,
                    mode="FAST_POLICY",
                    pool_counts={},
                    gate_reason="system_notification",
                    threshold=0.0,
                    max_score=0.0,
                    score_clause="(fast_exit)",
                )
                return {
                    "records": [], "context_block": "", "query": query, "count": 0,
                    "skipped": True, "reason": "system_notification", "threshold": 0.0,
                    "threshold_selection_reason": "policy_gate_fast_exit",
                }
        return None

    @staticmethod
    def _has_real_storage(eff_engine: Optional[Any]) -> bool:
        """True gdy silnik ma realny backend (KV/Qdrant) — warunek fallbacku orchestratora."""
        return bool(
            (getattr(eff_engine, "kv", None) or getattr(eff_engine, "kv_store", None))
            or (getattr(eff_engine, "vector_store", None) or getattr(eff_engine, "vector", None))
        )

    @staticmethod
    def _orchestrator_recall_record(d: Dict[str, Any]) -> Dict[str, Any]:
        """Normalizuje rekord orchestratora do kandydata (zero wymyślonych score'ów — V91 BUG-S5)."""
        k = str(d.get("subject") or d.get("id") or "orchestrator:rec")
        val = str(d.get("object") or d.get("content") or d.get("text") or "")
        return {
            "subject": k,
            "predicate": d.get("predicate", "stated_memory"),
            "object": val,
            "content": str(d.get("content") or val),
            "source": "orchestrator",
            # V91 (BUG-S5): wcześniej `d.get("vector_score", d.get("score", 0.95))`
            # WSTRZYKIWAŁO wymyślone 0.95, gdy rekord źródłowy nie miał scoringu —
            # fuzja DBSF dostawała konfabulację jako "podobieństwo" i wzmacniała ją
            # wagą alfa. Brak pomiaru = 0.0 (rekord nadal może wygrać leksykalnie).
            "vector_score": float(d.get("vector_score", 0.0)),
            "score": float(d.get("score", 0.0)),
            "confidence": float(d.get("confidence", 1.0) or d.get("veracity", 1.0)),
            "source_type": str(d.get("source_type", "user_explicit")),
            "metadata": dict(d.get("metadata") or {}),
        }

    @staticmethod
    async def _orchestrator_fallback_recall(
        eff_engine: Optional[Any],
        eff_orch: Optional[Any],
        query: str,
        session_id: Any,
        candidates: list,
    ) -> None:
        """Delegacja recall do zewnętrznego orchestratora przy braku bezpośredniego silnika bazodanowego."""
        has_real_storage = PrefetchEngine._has_real_storage(eff_engine)
        if not candidates and not has_real_storage and eff_orch and hasattr(eff_orch, "orchestrated_recall"):
            try:
                raw = await eff_orch.orchestrated_recall(query, session_id=session_id)
                recs = (
                    (raw.get("records") or raw.get("selected_facts") or [])
                    if isinstance(raw, dict)
                    else (raw if isinstance(raw, list) else [])
                )
                for r in recs:
                    d = r if isinstance(r, dict) else getattr(r, "__dict__", {})
                    candidates.append(PrefetchEngine._orchestrator_recall_record(d))
            except Exception as exc:
                logger.debug("orchestrator fallback recall error: %s", exc)

    def _score_pool(
        self,
        query: str,
        params: Dict[str, Any],
        candidates: list,
        eff_cal: Optional[Any],
    ) -> list:
        """2. Punktacja Continuous Fusion (Static-Anchor DBSF lub RRF) + 2b kalibrowane probs."""
        fusion_strat = str(params.get("fusion_strategy") or getattr(self, "fusion_strategy", "") or "").strip() or None
        scored = score_candidates(query=query, candidates=candidates, fusion_strategy=fusion_strat)
        self._apply_calibrated_probs(query, scored, eff_cal)
        return scored

    @staticmethod
    def _apply_calibrated_probs(query: str, scored: list, eff_cal: Optional[Any]) -> None:
        """2b. Obliczenie prawdopodobieństwa kalibrowanego (V93: wielowymiarowy prior DBSF)."""
        if eff_cal and getattr(eff_cal, "enabled", False) and getattr(eff_cal, "calibrator", None):
            is_sym = is_symbolic_or_identifier(query)
            q_type = "symbolic" if is_sym else "prose"
            scores_pool = [float(c.get("score", 0.0)) for c in scored]
            med_score = float(np.median(scores_pool)) if scores_pool else 0.0
            log_q_len = float(np.log(max(1, len(query.split()))))
            for c in scored:
                s_dense = float(c.get("dense_score", 0.0))
                s_sparse = float(c.get("sparse_score", 0.0))
                s_margin = float(c.get("score", 0.0)) - med_score
                x_multi = [s_dense, s_sparse, s_margin, log_q_len]
                prob = eff_cal.predict_proba(
                    score=float(c.get("score", 0.0)),
                    query_type=q_type,
                    x_multivariate=x_multi,
                )
                c["calibrated_prob"] = round(float(prob), 4)
                if "metadata" in c and isinstance(c["metadata"], dict):
                    c["metadata"]["calibrated_prob"] = c["calibrated_prob"]

    @staticmethod
    def _gate_diagnostics(query: str, candidates: list, scored: list, eff_cal: Optional[Any]):
        """3. Gate + diagnostyka: (gate, mode_diag, q_stems_diag, ch_counts)."""
        gate = evaluate_gate(query=query, scored_candidates=scored, calibration_engine=eff_cal)
        from atlas_memory.cognitive.retrieval.score import _query_stems
        q_stems_diag = _query_stems(query)
        mode_diag = "SYMBOLIC" if is_symbolic_or_identifier(query) else "PROSE"
        ch_counts: Dict[str, int] = {"kv": 0, "vec": 0, "graph": 0, "hot": 0}
        for c_item in candidates:
            src_c = str(c_item.get("source") or "kv")
            if "vec" in src_c:
                ch_counts["vec"] += 1
            elif "graph" in src_c or c_item.get("graph_hit"):
                ch_counts["graph"] += 1
            elif "hot" in src_c:
                ch_counts["hot"] += 1
            else:
                ch_counts["kv"] += 1
        return gate, mode_diag, q_stems_diag, ch_counts

    @staticmethod
    def _format_top_rejected(scored: list) -> str:
        """Podgląd top-1 odrzuconego rekordu (telemetria skip)."""
        top_rej = scored[0] if scored else {}
        return (
            f"{top_rej.get('subject', 'none')[:36]}(s={top_rej.get('score', 0.0):.3f},"
            f"d={top_rej.get('dense_score', 0.0):.2f},l={top_rej.get('sparse_score', 0.0):.2f},g={top_rej.get('graph_score', 0.0):.2f})"
        ) if top_rej else "none"

    @staticmethod
    def _gate_skip_log(
        query: str,
        gate: Any,
        force: bool,
        ch_counts: Dict[str, int],
        scored: list,
        mode_diag: str,
        q_stems_diag: list,
        candidates: list,
    ) -> Optional[Dict[str, Any]]:
        """3b. Skip path gdy gate odrzuca (bez force) — log + skip dict (V119 short_circuit)."""
        if gate.should_retrieve or force:
            return None
        max_cand_score = max([float(c.get("score", 0.0)) for c in scored], default=0.0)
        top_rej_info = PrefetchEngine._format_top_rejected(scored)
        score_clause = (
            "(short_circuit)"
            if gate.reason in ("continuation_prompt", "empty_pool", "system_notification", "empty_query", "casual_greeting")
            else f"max_score={max_cand_score:.3f} < thresh={gate.threshold:.3f}"
        )
        log_recall_skip(
            log=logger,
            query=query,
            mode=mode_diag,
            pool_counts=ch_counts,
            gate_reason=gate.reason,
            threshold=gate.threshold,
            max_score=max_cand_score,
            top_rejected=top_rej_info,
            score_clause=score_clause,
            stems=q_stems_diag,
        )
        return {
            "records": [], "context_block": "", "query": query, "count": 0,
            "skipped": True, "reason": gate.reason, "threshold": gate.threshold,
            "threshold_selection_reason": gate.threshold_selection_reason,
        }

    def _assemble_selection(
        self,
        query: str,
        scored: list,
        gate: Any,
        force: bool,
        limit: int,
        eff_cal: Optional[Any],
    ) -> tuple[list, Optional[Dict[str, Any]]]:
        """4. Wybór rekordów wg powodu gate'a (continuation / entropy filter / force)."""
        if gate.reason.startswith("continuation") or gate.reason.startswith("empty_query_hot"):
            min_thresh = float(getattr(gate, "threshold", 0.0) or 0.0) if gate.reason.startswith("continuation") else 0.0
            selected = [
                c for c in scored
                if c.get("source") == "hot_buffer"
                and float(c.get("score", 0.0)) >= min_thresh
                and not any(str(c.get("subject", "")).lower().startswith(p) for p in PERSISTENT_KEY_PREFIXES)
            ][:limit]
            if not selected and not force:
                log_recall_skip(
                    log=logger,
                    query=query,
                    mode="CONTINUATION",
                    pool_counts={},
                    gate_reason="continuation_prompt",
                    threshold=0.0,
                    max_score=0.0,
                    score_clause="(empty active session buffer)",
                )
                return selected, {
                    "records": [], "context_block": "", "query": query, "count": 0,
                    "skipped": True, "reason": "continuation_prompt", "threshold": 0.0,
                    "threshold_selection_reason": "no_active_session_working_memory",
                }
        elif not force:
            from atlas_memory.cognitive.retrieval.score import apply_entropy_salience_filter
            salient_pool, _eff_thresh = apply_entropy_salience_filter(scored, gate.threshold)
            if os.getenv("ATLAS_CALIB_V93") == "1" and eff_cal and getattr(eff_cal, "enabled", False):
                selected = [c for c in salient_pool if float(c.get("calibrated_prob", c.get("score", 0.0))) >= gate.threshold][:limit]
            else:
                selected = salient_pool[:limit]
        else:
            selected = scored[:limit]
        return selected, None

    @staticmethod
    def _trim_by_token_budget(selected: list, budget: int) -> tuple[list, int]:
        """Pakuje rekordy w budżet tokenów z użyciem kanonicznego Knapsack 0/1 (pack_knapsack_01); zwraca (budgeted, zużyte tokeny)."""
        if not selected or budget <= 0:
            return [], 0
        from atlas_memory.cognitive.retrieval.render import pack_knapsack_01
        budgeted = pack_knapsack_01(selected, budget=budget)
        tot = sum(estimate_tokens(r) for r in budgeted)
        return budgeted, tot

    @staticmethod
    def _emit_accepted(
        query: str,
        selected: list,
        scored: list,
        gate: Any,
        candidates: list,
        mode_diag: str,
        q_stems_diag: list,
        ch_counts: Dict[str, int],
        budget: int,
        tot: int,
    ) -> Dict[str, Any]:
        """Render context block + log accept + final result dict."""
        max_chars = max(4000, budget * 4)
        context_block = render_context_block(records=selected, max_object_chars=1200, max_total_chars=max_chars)
        log_recall(
            log=logger,
            query=query,
            mode=mode_diag,
            pool_counts=ch_counts,
            gate_reason=gate.reason,
            threshold=gate.threshold,
            tokens_used=tot,
            token_budget=budget,
            winners=selected,
            stems=q_stems_diag,
        )
        # V111 (F-028): bramka realnie przepuściła (skipped=False), ale entropy
        # filter/budget mógł wyciąć wszystko — telemetria musi to odróżniać od
        # normalnego accepted z wynikami.
        reason_out = "no_results" if (gate.reason == "accepted" and not selected) else gate.reason
        return {
            "records": selected, "context_block": context_block, "query": query, "count": len(selected),
            "skipped": False, "reason": reason_out, "threshold": gate.threshold,
            "threshold_selection_reason": gate.threshold_selection_reason,
        }

    def _check_dedup(
        self,
        query: str,
        session_id: Any,
        limit: int,
        budget: int,
        force: bool,
    ) -> Optional[Dict[str, Any]]:
        """DIRECTIVE 3: Sprawdza Sliding Dedup Buffer (TTL: 1500ms). Zwraca wynik lub None."""
        if force or not query:
            return None
        from atlas_memory.cognitive.retrieval.ppr import is_hipporag_enabled
        curr_hippo = is_hipporag_enabled()
        dedup_key = f"{query.strip()}||{session_id or ''}||{limit}||{budget}"
        now_ts = time.monotonic()
        entry = self._dedup_cache.get(dedup_key)
        if entry is not None:
            cached_time, cached_res = entry
            if "_hippo_enabled" in cached_res and cached_res.get("_hippo_enabled") != curr_hippo:
                return None
            age = now_ts - cached_time
            if age < self.dedup_ttl_seconds:
                logger.info(
                    "[RETRIEVAL:DEDUP_HIT] query='%s' session_id='%s' age=%.3fs (served from sliding buffer)",
                    query[:64],
                    session_id or "",
                    age,
                )
                hit_res = dict(cached_res)
                hit_res["dedup_hit"] = True
                hit_res.pop("_hippo_enabled", None)
                return hit_res
        return None

    def _store_dedup(
        self,
        query: str,
        session_id: Any,
        limit: int,
        budget: int,
        force: bool,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """DIRECTIVE 3: Zapisuje wynik do Sliding Dedup Buffer z ograniczeniem rozmiaru."""
        if not force and query:
            from atlas_memory.cognitive.retrieval.ppr import is_hipporag_enabled
            dedup_key = f"{query.strip()}||{session_id or ''}||{limit}||{budget}"
            now_ts = time.monotonic()
            stored = dict(result)
            stored["_hippo_enabled"] = is_hipporag_enabled()
            self._dedup_cache[dedup_key] = (now_ts, stored)
            if len(self._dedup_cache) > 200:
                cutoff = now_ts - 10.0
                self._dedup_cache = {k: v for k, v in self._dedup_cache.items() if v[0] >= cutoff}
        return result

    async def execute(
        self,
        params: Dict[str, Any],
        engine: Optional[Any] = None,
        orchestrator: Optional[Any] = None,
        hot_buffer: Optional[Any] = None,
        graph_store: Optional[Any] = None,
        calibration_engine: Optional[Any] = None,
        check_sync_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Główna metoda prefetchu wywoływana przez UDS i in-process.

        v121 (T2): cienka orkiestracja helperów CC≤15 (był monolit CC=77).
        """
        query, session_id, limit, budget, force, eff_engine, eff_cal, eff_orch, en_cal = self._resolve_context(
            params, engine, calibration_engine, orchestrator,
        )
        self._check_sync(check_sync_callback)

        dedup_hit = self._check_dedup(query, session_id, limit, budget, force)
        if dedup_hit is not None:
            return dedup_hit

        skip = self._orchestrator_policy_skip(eff_orch, query, force)
        if skip is not None:
            return self._store_dedup(query, session_id, limit, budget, force, skip)
        skip = self._fast_policy_skip(query, force)
        if skip is not None:
            return self._store_dedup(query, session_id, limit, budget, force, skip)

        # 1. Zbieranie kandydatów z warstw trwałych (KV, Qdrant, Kùzu)
        candidates = await gather_candidates(
            query=query, session_id=session_id, limit=limit,
            engine=eff_engine, orchestrator=eff_orch, hot_buffer=hot_buffer, graph_store=graph_store,
            calibration_engine=eff_cal, enable_calibrated_thresholds=en_cal,
            fallback_on_ppr_error=True,
        )
        await self._orchestrator_fallback_recall(eff_engine, eff_orch, query, session_id, candidates)

        # 2. Punktacja Continuous Fusion (Static-Anchor DBSF lub RRF)
        scored = self._score_pool(query, params, candidates, eff_cal)

        # 3. Bramka kalibrowana + diagnostyka kanałów
        gate, mode_diag, q_stems_diag, ch_counts = self._gate_diagnostics(query, candidates, scored, eff_cal)

        skip = self._gate_skip_log(query, gate, force, ch_counts, scored, mode_diag, q_stems_diag, candidates)
        if skip is not None:
            return self._store_dedup(query, session_id, limit, budget, force, skip)

        # 4. Wybór rekordów mieszczących się w budżecie
        selected, cont_skip = self._assemble_selection(query, scored, gate, force, limit, eff_cal)
        if cont_skip is not None:
            return self._store_dedup(query, session_id, limit, budget, force, cont_skip)
        selected, tot = self._trim_by_token_budget(selected, budget)

        res = self._emit_accepted(
            query, selected, scored, gate, candidates,
            mode_diag, q_stems_diag, ch_counts, budget, tot,
        )
        return self._store_dedup(query, session_id, limit, budget, force, res)
