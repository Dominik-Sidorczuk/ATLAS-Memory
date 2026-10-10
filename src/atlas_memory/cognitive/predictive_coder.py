"""
ATLAS Cognitive Layer: Closed-Loop Predictive Coder (V27+ Architecture).

Implements active sensing discrepancy computation, severity gating,
controlled auto-annealing, and closed-loop world model writeback.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from typing import Any, Dict, Optional

from atlas_memory.cognitive.models import PredictionVerdict
from atlas_memory.cognitive.write_path import EpistemicWritePath
from atlas_memory.core.agi_logger import log_active_sensing, log_surprise

logger = logging.getLogger(__name__)


class PredictiveCoder:
    """
    Closed-Loop Active Sensing & Predictive Coding Engine.

    Pętla kognitywna predykcji:
    Predykcja (oczekiwanie) -> Obserwacja -> Błąd Predykcji (discrepancy) ->
    Bramkowanie tolerancji -> Anneal (tylko na MODERATE+) ->
    ZAMKNIĘCIE PĘTLI: Writeback faktu korekcyjnego do KV (probe:X:last_observed)
    i krawędzi do grafu wiedzy (world_model_updated: True).
    """

    def __init__(
        self,
        engine: Any = None,
        causal_engine: Any = None,
        active_sensing: Any = None,
    ) -> None:
        self.engine = engine
        self.causal_engine = causal_engine
        self.active_sensing = active_sensing

    def compute_discrepancy(
        self,
        observed: Any,
        expected: Any,
        tolerance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Computes numeric or categorical discrepancy between observed and expected sensory states.
        Respects tolerance thresholds to prevent low-level noise from triggering system alarms.
        """
        obs_str = str(observed).strip()
        exp_str = str(expected).strip()

        # Try numeric evaluation first
        try:
            obs_num = float(obs_str)
            exp_num = float(exp_str)

            diff = abs(obs_num - exp_num)
            tol = float(tolerance) if tolerance is not None else 0.0

            if diff <= tol:
                return {
                    "has_error": False,
                    "severity": "NONE",
                    "discrepancy_score": 0.0,
                    "relative_diff": 0.0,
                    "diff": diff,
                }

            if tolerance is None and diff < 1e-5:
                return {
                    "has_error": False,
                    "severity": "NONE",
                    "discrepancy_score": 0.0,
                    "relative_diff": 0.0,
                    "diff": diff,
                }

            scale_anchor = max(abs(exp_num), abs(obs_num), 1.0)
            rel_diff = diff / scale_anchor
            score = min(1.0, max(0.0, rel_diff))

            if score < 0.10:
                severity = "LOW"
            elif score < 0.50:
                severity = "MODERATE"
            else:
                severity = "CRITICAL"

            return {
                "has_error": True,
                "severity": severity,
                "discrepancy_score": score,
                "relative_diff": rel_diff,
                "diff": diff,
            }
        except (ValueError, TypeError):
            # Categorical string comparison
            if obs_str.lower() == exp_str.lower():
                return {
                    "has_error": False,
                    "severity": "NONE",
                    "discrepancy_score": 0.0,
                    "relative_diff": 0.0,
                    "diff": 0.0,
                }

            words_obs = set(re.findall(r"\w+", obs_str.lower()))
            words_exp = set(re.findall(r"\w+", exp_str.lower()))
            if not words_obs or not words_exp:
                score = 1.0
            else:
                jaccard = len(words_obs & words_exp) / len(words_obs | words_exp)
                score = 1.0 - jaccard

            if score < 0.20:
                severity = "LOW"
            elif score < 0.60:
                severity = "MODERATE"
            else:
                severity = "CRITICAL"

            return {
                "has_error": score > 0.0,
                "severity": severity,
                "discrepancy_score": score,
                "relative_diff": score,
                "diff": score,
            }

    async def execute_sensing(
        self,
        params: Dict[str, Any],
        graph_client: Any = None,
    ) -> Dict[str, Any]:
        """
        Executes an active sensing cycle with closed-loop world model writeback.
        """
        target_entity = str(params.get("target_entity") or params.get("probe") or params.get("target") or "")
        observed_value = params.get("observed_value") if "observed_value" in params else params.get("observed")
        expected_value = params.get("expected_value") if "expected_value" in params else params.get("expected")

        if not target_entity:
            raise ValueError("Target entity or probe name is required for active sensing")

        tolerance_val: Optional[float] = None
        if "tolerance" in params and params["tolerance"] is not None:
            try:
                tolerance_val = float(params["tolerance"])
            except (ValueError, TypeError):
                tolerance_val = None

        if expected_value is None and self.active_sensing is not None:
            checks = getattr(self.active_sensing, "expectation_checks", lambda: [])()
            for chk in checks:
                if getattr(chk, "target_entity", None) == target_entity:
                    expected_value = chk.expected_value
                    if tolerance_val is None and getattr(chk, "tolerance", None) is not None:
                        tolerance_val = chk.tolerance
                    break

        disc_report = self.compute_discrepancy(observed_value, expected_value, tolerance=tolerance_val)
        has_error = disc_report["has_error"]
        severity = disc_report["severity"]
        score = disc_report["discrepancy_score"]

        # 1. Gate Annealing: ONLY trigger for MODERATE or CRITICAL severity (score >= 0.10)
        should_anneal = has_error and (severity in ("MODERATE", "CRITICAL") or score >= 0.10)
        anneal_triggered = False
        anneal_result: Optional[Dict[str, Any]] = None

        if has_error:
            action_desc = "auto_anneal_triggered" if should_anneal else "error_detected"
            log_surprise(
                log=logger,
                target_entity=target_entity,
                severity=severity,
                discrepancy_score=score,
                expected_val=expected_value,
                observed_val=observed_value,
                action=action_desc,
            )

        causal = self.causal_engine or (self.engine.causal_engine if self.engine and hasattr(self.engine, "causal_engine") else None)
        if should_anneal and causal is not None and hasattr(causal, "recalibrate_graph_with_annealer"):
            anneal_triggered = True
            sync_anneal = bool(params.get("sync_anneal", False))
            if sync_anneal:
                try:
                    anneal_result = await causal.recalibrate_graph_with_annealer(target_entity=target_entity)
                    logger.debug(
                        "[PredictiveCoder] Active sensing sync anneal completed for '%s' (disc=%.4f, sev=%s) -> %s",
                        target_entity, score, severity, anneal_result,
                    )
                except Exception as a_exc:
                    logger.debug("Sync auto-anneal execution failed: %s", a_exc)
            else:
                async def _run_annealer_bg() -> None:
                    try:
                        res = await causal.recalibrate_graph_with_annealer(target_entity=target_entity)
                        logger.debug(
                            "[PredictiveCoder] Decoupled background auto-anneal completed for '%s' (disc=%.4f, sev=%s) -> %s",
                            target_entity, score, severity, res,
                        )
                    except Exception as bg_exc:
                        logger.debug("Background auto-anneal execution failed: %s", bg_exc)

                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(_run_annealer_bg())
                    await asyncio.sleep(0)
                    anneal_result = {"status": "scheduled_background", "target_entity": target_entity}
                except RuntimeError as loop_exc:
                    logger.debug("No active loop for background anneal: %s", loop_exc)

        # 2. CLOSED-LOOP WORLD MODEL WRITEBACK
        # When an error of MODERATE or CRITICAL severity is detected:
        # Write corrective state to KV (probe:X:last_observed) and graph
        world_model_updated = False
        writeback_key: Optional[str] = None

        if has_error and severity in ("MODERATE", "CRITICAL"):
            writeback_key = f"probe:{target_entity}:last_observed"
            meta = {
                "source_type": "sensor_observation",
                "target_entity": target_entity,
                "expected_value": expected_value,
                "discrepancy_score": score,
                "severity": severity,
                "timestamp": time.time(),
                "ttl": 3600,
            }
            write_path = getattr(self.engine, "write_path", None)
            if write_path is not None and hasattr(write_path, "write_async"):
                try:
                    await write_path.write_async(
                        key=writeback_key,
                        value=observed_value,
                        confidence=1.0,
                        metadata=meta,
                        reason="active_sensing_writeback",
                    )
                    world_model_updated = True
                    logger.debug(
                        "[PredictiveCoder] Closed-loop writeback committed via EpistemicWritePath: %s = %s",
                        writeback_key, observed_value,
                    )
                except Exception as wp_exc:
                    logger.debug("Active sensing write_path writeback failed: %s", wp_exc)

            if not world_model_updated:
                kv_store = self.engine.kv if self.engine and hasattr(self.engine, "kv") else None
                if kv_store is not None:
                    set_fn = getattr(kv_store, "set_sync", getattr(kv_store, "set", None))
                    if set_fn is not None:
                        try:
                            set_fn(
                                writeback_key,
                                observed_value,
                                confidence=1.0,
                                metadata=meta,
                                reason="active_sensing_writeback",
                                ttl=3600,
                            )
                            world_model_updated = True
                            logger.debug(
                                "[PredictiveCoder] Closed-loop writeback committed to KV (ttl=3600): %s = %s",
                                writeback_key, observed_value,
                            )
                        except TypeError:
                            set_fn(
                                writeback_key,
                                observed_value,
                                confidence=1.0,
                                metadata=meta,
                                reason="active_sensing_writeback",
                            )
                            world_model_updated = True
                            logger.debug(
                                "[PredictiveCoder] Closed-loop writeback committed to KV: %s = %s",
                                writeback_key, observed_value,
                            )
                        except Exception as kv_exc:
                            logger.debug("Active sensing KV writeback failed: %s", kv_exc)

            if world_model_updated and writeback_key:
                log_active_sensing(
                    log=logger,
                    writeback_key=writeback_key,
                    observed_value=observed_value,
                    status="committed",
                )

            # Knowledge graph relation update
            g_store = graph_client or (self.engine.graph if self.engine and hasattr(self.engine, "graph") else None)
            if g_store is not None:
                try:
                    wp = EpistemicWritePath(graph_store=graph_client) if graph_client is not None else (getattr(self.engine, "write_path", None) or EpistemicWritePath(graph_store=g_store))
                    if wp.commit_relation(
                        str(target_entity),
                        "observed_state",
                        str(observed_value),
                        confidence=1.0,
                        timestamp=time.time(),
                    ):
                        world_model_updated = True
                except Exception as g_exc:
                    logger.debug("Active sensing graph relation update failed: %s", g_exc)

        if has_error and self.active_sensing is not None:
            try:
                from atlas_memory.active.prediction_error import PredictionError
                p_err = PredictionError(
                    check_id=str(params.get("check_id") or f"chk_{target_entity}"),
                    target_entity=target_entity,
                    predicate=str(params.get("predicate") or "observed_state"),
                    expected_value=str(expected_value),
                    observed_value=str(observed_value),
                    discrepancy_score=score,
                    severity=severity,
                    timestamp=time.time(),
                    world_model_updated=world_model_updated,
                )
                if hasattr(self.active_sensing, "error_history"):
                    self.active_sensing.error_history.append(p_err)
            except Exception as pe_exc:
                logger.debug("Active sensing record error failed: %s", pe_exc)

        # 3. Assemble verdict response matching ATLAS UDS contract
        verdict = PredictionVerdict(
            verdict_id=f"verdict_{uuid.uuid4().hex[:8]}",
            status="ok",
            has_error=has_error,
            severity=severity,
            discrepancy_score=score,
            target_entity=target_entity,
            expected_value=expected_value,
            observed_value=observed_value,
            tolerance=tolerance_val,
            anneal_triggered=anneal_triggered,
            world_model_updated=world_model_updated,
            timestamp=time.time(),
            message=(
                f"Observation matches expectation within tolerance {tolerance_val}."
                if not has_error
                else f"Prediction error detected: observed={observed_value} vs expected={expected_value} (severity={severity}, disc={score:.4f})."
            ),
            metadata={
                "writeback_key": writeback_key,
                "relative_diff": disc_report.get("relative_diff", 0.0),
                "diff": disc_report.get("diff", 0.0),
                "anneal_result": anneal_result,
            },
        )

        resp = verdict.model_dump()
        resp["prediction_error"] = {
            "has_error": has_error,
            "severity": severity,
            "discrepancy_score": score,
            "target_entity": target_entity,
            "observed": observed_value,
            "expected": expected_value,
        }
        if anneal_result is not None:
            resp["anneal_details"] = anneal_result
        return resp
