"""
ATLAS Cognitive Layer: Closed-Loop Predictive Coder (V27+ Architecture).

Implements active sensing discrepancy computation, severity gating,
controlled auto-annealing, and closed-loop world model writeback.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any, Dict, Optional

from atlas_memory.cognitive.models import PredictionVerdict
from atlas_memory.cognitive.write_path import EpistemicWritePath

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

    def __init__(self, engine: Any = None, causal_engine: Any = None) -> None:
        self.engine = engine
        self.causal_engine = causal_engine

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

            denominator = max(abs(exp_num), 1e-6)
            rel_diff = diff / denominator
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

        disc_report = self.compute_discrepancy(observed_value, expected_value, tolerance=tolerance_val)
        has_error = disc_report["has_error"]
        severity = disc_report["severity"]
        score = disc_report["discrepancy_score"]

        # 1. Gate Annealing: ONLY trigger for MODERATE or CRITICAL severity (score >= 0.10)
        should_anneal = has_error and (severity in ("MODERATE", "CRITICAL") or score >= 0.10)
        anneal_triggered = False
        anneal_result: Optional[Dict[str, Any]] = None

        causal = self.causal_engine or (self.engine.causal_engine if self.engine and hasattr(self.engine, "causal_engine") else None)
        if should_anneal and causal is not None and hasattr(causal, "recalibrate_graph_with_annealer"):
            try:
                anneal_result = await causal.recalibrate_graph_with_annealer(
                    target_entity=target_entity,
                )
                anneal_triggered = True
                logger.info(
                    "[PredictiveCoder] Active sensing auto-anneal triggered for '%s' (disc=%.4f, sev=%s) -> %s",
                    target_entity, score, severity, anneal_result,
                )
            except Exception as a_exc:
                logger.debug("Auto-anneal execution failed: %s", a_exc)

        # 2. CLOSED-LOOP WORLD MODEL WRITEBACK
        # When an error of MODERATE or CRITICAL severity is detected:
        # Write corrective state to KV (probe:X:last_observed) and graph
        world_model_updated = False
        writeback_key: Optional[str] = None

        if has_error and severity in ("MODERATE", "CRITICAL"):
            writeback_key = f"probe:{target_entity}:last_observed"
            kv_store = self.engine.kv if self.engine and hasattr(self.engine, "kv") else None
            if kv_store is not None and hasattr(kv_store, "set_sync"):
                try:
                    kv_store.set_sync(
                        writeback_key,
                        observed_value,
                        confidence=1.0,
                        metadata={
                            "source_type": "sensor_observation",
                            "target_entity": target_entity,
                            "expected_value": expected_value,
                            "discrepancy_score": score,
                            "severity": severity,
                            "timestamp": time.time(),
                        },
                        reason="active_sensing_writeback",
                    )
                    world_model_updated = True
                    logger.info(
                        "[PredictiveCoder] Closed-loop writeback committed to KV: %s = %s",
                        writeback_key, observed_value,
                    )
                except Exception as kv_exc:
                    logger.debug("Active sensing KV writeback failed: %s", kv_exc)

            # Knowledge graph relation update
            g_store = graph_client or (self.engine.graph if self.engine and hasattr(self.engine, "graph") else None)
            if g_store is not None:
                try:
                    write_path = EpistemicWritePath(graph_store=graph_client) if graph_client is not None else (getattr(self.engine, "write_path", None) or EpistemicWritePath(graph_store=g_store))
                    if write_path.commit_relation(
                        str(target_entity),
                        "observed_state",
                        str(observed_value),
                        confidence=1.0,
                        timestamp=time.time(),
                    ):
                        world_model_updated = True
                except Exception as g_exc:
                    logger.debug("Active sensing graph relation update failed: %s", g_exc)

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
