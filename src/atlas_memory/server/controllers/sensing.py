"""Active Sensing, Attention Bus & Shadow Worker RPC Controller for AtlasDaemon.

Encapsulates background LLM turn extraction (OmniRoute shadow worker),
predictive coding active sensing probes, and the attention escalation bus.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict

logger = logging.getLogger("atlas.server.controllers.sensing")


class SensingRpcControllerMixin:
    """Mixin providing shadow worker, active sensing, and attention bus RPC handlers."""

    async def _handle_shadow_enqueue(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Kolejkowanie tury użytkownika do ekstrakcji SPO w tle (nieblokujące)."""
        worker = getattr(self, "shadow_worker", None)
        if worker is None:
            return {
                "status": "error",
                "error_code": "SHADOW_WORKER_DISABLED",
                "message": "Shadow worker is disabled (ATLAS_SHADOW_WORKER=0 or init failure)",
            }
        user_msg = str(params.get("user_msg") or "")
        agent_response = str(params.get("agent_response") or "")
        session_id = str(params.get("session_id") or "default")
        if not user_msg and not agent_response:
            return {
                "status": "error",
                "error_code": "EMPTY_TURN",
                "message": "shadow_enqueue requires non-empty 'user_msg' or 'agent_response'",
            }
        try:
            await worker.enqueue_turn(user_msg, agent_response, session_id)
        except Exception as exc:
            return {"status": "error", "error_code": "QUEUE_REJECTED", "message": str(exc)}
        return {"status": "ok", "queued": True, "queue_depth": worker.queue_depth}

    async def _handle_shadow_drain(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Przetwarza oczekujące tury i zwraca JAWNY status ekstrakcji.

        ZAKAZ cichego sukcesu: gdy klient LLM (OmniRoute) jest niedostępny,
        status jest 'error' z 'error_code'='OMNIROUTE_UNAVAILABLE' i oznaczonym
        fallbackiem regułowym ('fallback_used': true).
        """
        worker = getattr(self, "shadow_worker", None)
        if worker is None:
            return {
                "status": "error",
                "error_code": "SHADOW_WORKER_DISABLED",
                "message": "Shadow worker is disabled (ATLAS_SHADOW_WORKER=0 or init failure)",
            }
        processed = await worker.flush(wait_s=float(params.get("wait_s", 2.0)))
        outcomes = worker.pop_completed()
        processed = max(processed, len(outcomes))
        facts_committed = sum(o.get("facts_committed", 0) for o in outcomes)
        committed_keys = [k for o in outcomes for k in o.get("committed_keys", [])]
        sources = [o.get("extraction", {}).get("source") for o in outcomes]
        error_codes = [o.get("extraction", {}).get("error_code") for o in outcomes if o.get("extraction", {}).get("error_code")]
        any_fallback = any(o.get("extraction", {}).get("fallback_used") for o in outcomes)

        if any_fallback:
            return {
                "status": "error",
                "error_code": (error_codes[0] if error_codes else "OMNIROUTE_UNAVAILABLE"),
                "message": "LLM extraction failed; regex fallback was used (results may be degraded)",
                "processed_turns": processed,
                "facts_committed": facts_committed,
                "committed_keys": committed_keys,
                "extraction_source": "regex_fallback",
                "fallback_used": True,
                "outcomes": outcomes,
            }
        return {
            "status": "ok",
            "processed_turns": processed,
            "facts_committed": facts_committed,
            "committed_keys": committed_keys,
            "extraction_source": (sources[0] if sources else None),
            "fallback_used": False,
            "outcomes": outcomes,
        }

    async def _handle_shadow_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnostyka workera cieni: czy żyje, głębokość kolejki, ostatni status."""
        worker = getattr(self, "shadow_worker", None)
        if worker is None:
            return {"status": "error", "error_code": "SHADOW_WORKER_DISABLED"}
        extraction = dict(getattr(worker, "last_extraction_status", {}))
        return {
            "status": "ok",
            "running": bool(getattr(worker, "_is_running", False)),
            "queue_depth": worker.queue_depth,
            "in_flight": worker.in_flight,
            "last_extraction": extraction,
        }

    async def _handle_active_sensing(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Active sensing probe for predictive coding with closed-loop world model writeback."""
        action = str(params.get("action") or "").lower()
        if action == "register" or params.get("register_expectation"):
            active = getattr(self, "active_sensing", None)
            if active is not None:
                from atlas_memory.active.prediction_error import PredictionCheck
                target = str(params.get("target_entity") or params.get("target") or params.get("probe") or "")
                if not target:
                    return {"status": "error", "message": "target_entity is required to register an expectation"}
                check = PredictionCheck(
                    check_id=str(params.get("check_id") or f"chk_{target}"),
                    target_entity=target,
                    expected_predicate=str(params.get("expected_predicate") or params.get("predicate") or "state"),
                    expected_value=str(params.get("expected_value") or params.get("expected") or ""),
                    tolerance=float(params["tolerance"]) if "tolerance" in params and params["tolerance"] is not None else None,
                    check_type=str(params.get("check_type") or "exact"),
                )
                active.register_expectation(check)
                return {"status": "ok", "registered": True, "check_id": check.check_id, "target_entity": target}
            return {"status": "no_active_sensing", "registered": False}

        coder = getattr(self, "predictive_coder", None)
        if coder is not None:
            result = await coder.execute_sensing(params, graph_client=getattr(self, "graph_client", None))
            self._emit_attention_event(result)
            return result
        return {"probe": params.get("target_entity", ""), "sensed": True}

    def _emit_attention_event(self, result: Any) -> bool:
        """V68 (ARCH-007): producent zdarzeń uwagi (NIE może zablokować RPC).

        Event pojawia się TYLKO gdy execute_sensing zwrócił realny zapis
        korekty świata (metadata.writeback_key != None, tj. błąd predykcji
        o severity MODERATE/CRITICAL).
        """
        try:
            if not isinstance(result, dict):
                return False
            meta = result.get("metadata") or {}
            writeback_key = meta.get("writeback_key")
            if not writeback_key:
                return False
            pred_err = result.get("prediction_error") or {}
            event = {
                "type": "prediction_error_writeback",
                "writeback_key": str(writeback_key),
                "entity": str(pred_err.get("target_entity") or ""),
                "severity": str(pred_err.get("severity") or ""),
                "discrepancy_score": pred_err.get("discrepancy_score", 0.0),
                "observed": pred_err.get("observed"),
                "expected": pred_err.get("expected"),
                "timestamp": time.time(),
            }
            queue = getattr(self, "_attention_queue", None)
            if queue is not None:
                queue.put_nowait(event)
                logger.info(
                    "[AttentionBus] queued writeback=%s entity=%s sev=%s qsize=%d",
                    writeback_key, event["entity"], event["severity"],
                    queue.qsize(),
                )
                return True
            return False
        except asyncio.QueueFull:
            logger.debug("[AttentionBus] queue full — dropping event")
            return False
        except Exception as exc:
            logger.debug("[AttentionBus] emit failed: %s", exc)
            return False

    def _apply_attention_event(self, event: Dict[str, Any]) -> str:
        """V68 (ARCH-007): aplikacja zdarzenia uwagi do pamięci roboczej."""
        entity = str(event.get("entity") or "")
        if not entity:
            parts = str(event.get("writeback_key", "")).split(":")
            if len(parts) >= 2 and parts[0] == "probe":
                entity = parts[1]
        if not entity:
            entity = "unknown"
        key = f"attention:{entity}"
        hot_buf = getattr(self, "hot_buffer", getattr(self, "_hot_kv_buffer", None))
        if hot_buf is not None:
            hot_buf.update(
                key=key,
                value=f"prediction_error:{event.get('severity', '')}",
                confidence=1.0,
                source_type="attention_escalation",
                metadata={
                    "origin": "attention_bus_v68",
                    "writeback_key": event.get("writeback_key", ""),
                    "discrepancy_score": event.get("discrepancy_score", 0.0),
                },
            )
            logger.info(
                "[AttentionBus] attention escalated for '%s' (sev=%s, disc=%s)",
                entity, event.get("severity", ""), event.get("discrepancy_score", 0.0),
            )
        return key

    async def _attention_consumer(self) -> None:
        """V68 (ARCH-007): konsument busa uwagi — jedyny odbiorca probe:* ."""
        queue = getattr(self, "_attention_queue", None)
        if queue is None:
            return
        while True:
            event = await queue.get()
            try:
                self._apply_attention_event(event)
            except Exception as exc:
                logger.debug("[AttentionBus] apply failed: %s", exc)
            finally:
                queue.task_done()
