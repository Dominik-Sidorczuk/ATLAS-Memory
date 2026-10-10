"""Sleep Cycle, Trajectory Ingest & Session Lifecycle RPC Controller for AtlasDaemon.

Encapsulates real agent trajectory collection, L3 sleep consolidation,
causal anneal phases, and session end notifications.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

from atlas_memory.cognitive.models import HERMES_DEFAULT_SESSION
from atlas_memory.core.agi_logger import log_sleep

logger = logging.getLogger("atlas.server.controllers.sleep")


def _resolve_sleep_watermark(buffer_max: int, watermark: int) -> int:
    """V65 (BUG-012): stale-epoch guard dla watermarku cyklu snu."""
    if watermark > buffer_max:
        logger.info(
            "Sleep watermark %d stale (buffer max %d) po restarcie daemona — reset do 0",
            watermark,
            buffer_max,
        )
        return 0
    return watermark


class SleepRpcControllerMixin:
    """Mixin providing sleep consolidation, trajectory ingestion, and session lifecycle RPC handlers."""

    async def _run_causal_anneal_phase(
        self,
        stats: Any,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """V68 (ARCH-003): faza wyżarzania grafu przyczynowego w cyklu snu."""
        info: Dict[str, Any] = {
            "status": "skipped",
            "target": str(params.get("anneal_target") or "Root"),
            "iterations": 0,
            "updated_edges": 0,
            "converged": False,
            "duration_ms": 0.0,
        }
        try:
            causal = getattr(self, "causal_engine", None)
            graph = getattr(causal, "graph", None) if causal is not None else None
            if causal is None or not hasattr(causal, "recalibrate_graph_with_annealer"):
                info["status"] = "no_causal_engine"
            elif graph is None:
                info["status"] = "no_graph"
            else:
                max_iter = int(params.get("anneal_max_iter", 100))
                res = await causal.recalibrate_graph_with_annealer(
                    target_entity=info["target"],
                    anneal_max_iter=max_iter,
                )
                if isinstance(res, dict):
                    for key in ("status", "iterations", "updated_edges", "converged", "duration_ms"):
                        if key in res:
                            info[key] = res[key]
        except Exception as exc:
            logger.warning("V68 sleep anneal phase failed (non-critical): %s", exc)
            info["status"] = "error"

        try:
            stats.anneal_status = str(info.get("status", "skipped"))
            stats.anneal_target = str(info.get("target", "Root"))
            stats.anneal_iterations = int(info.get("iterations", 0) or 0)
            stats.anneal_updated_edges = int(info.get("updated_edges", 0) or 0)
            stats.anneal_converged = bool(info.get("converged", False))
            stats.anneal_duration_ms = float(info.get("duration_ms", 0.0) or 0.0)
        except Exception as exc:
            logger.debug("V68 anneal stats merge failed: %s", exc)
        return info

    async def _handle_trajectory_ingest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """V64: przyjmuje REALNE trajektorie narzędzi z transcriptu agenta (objective success)."""
        session_id = str(params.get("session_id") or HERMES_DEFAULT_SESSION)
        steps = params.get("steps")
        if not isinstance(steps, list) or not steps:
            return {"status": "error", "error": "steps_required", "ingested": 0}
        engine = getattr(self, "engine", None)
        if engine is None or getattr(engine, "trajectory_buffer", None) is None:
            return {"status": "error", "error": "no_engine_buffer", "ingested": 0}
        buffer = engine.trajectory_buffer
        if not hasattr(buffer, "record_real_trajectory"):
            return {"status": "error", "error": "buffer_no_real_sink", "ingested": 0}
        clean: list[dict[str, Any]] = []
        for step in steps:
            if isinstance(step, dict) and (step.get("tool_name") or step.get("action") or step.get("tool")):
                clean.append(step)
        if not clean:
            return {"status": "error", "error": "no_valid_steps", "ingested": 0}
        try:
            count = buffer.record_real_trajectory(clean, session_id=session_id)
        except Exception as exc:
            logger.warning("trajectory/ingest record failed: %s", exc)
            return {"status": "error", "error": str(exc), "ingested": 0}
        logger.debug("[IPC:TRAJECTORY] ingest session=%s steps=%d", session_id, count)
        return {"status": "ok", "session_id": session_id, "ingested": count, "source": "real"}

    async def _handle_trigger_sleep_consolidation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Manually or autonomically triggers L3 sleep cycle consolidation and SOP skill compilation."""
        if getattr(self, "_is_sleeping", False):
            return {"status": "busy_sleep_in_progress", "baked_sops_count": 0}
        self._is_sleeping = True
        try:
            skills_dir = params.get("skills_dir") or (Path.home() / ".hermes" / "skills" / "baked-skills")
            if self.engine is None or getattr(self.engine, "auditor", None) is None:
                return {"status": "no_engine_auditor", "baked_sops_count": 0}

            stats = await self.engine.auditor.run_sleep_cycle_consolidation()

            from atlas_memory.l3_procedural.sleep_baker import SleepBaker

            min_freq = int(params.get("min_frequency", 1))
            baker = SleepBaker(min_frequency=min_freq)

            buffer = getattr(self.engine, "trajectory_buffer", None)
            all_trajs = getattr(buffer, "trajectories", []) if buffer is not None else []

            kv = getattr(self.engine, "kv", None)
            watermark = 0
            if kv is not None:
                try:
                    state = await kv.get_state("atlas:sleep_watermark")
                    if state and isinstance(state.get("value"), (int, float)):
                        watermark = int(state["value"])
                except Exception as exc:
                    logger.debug("sleep watermark read failed: %s", exc)

            buffer_max = max((int(t.get("max_step_id", 0)) for t in all_trajs), default=0)
            watermark = _resolve_sleep_watermark(buffer_max, watermark)

            new_trajs = [t for t in all_trajs if int(t.get("max_step_id", 0)) > watermark]
            if not new_trajs:
                logger.debug(
                    "[COG:SLEEP] cycle idempotent_noop: no new trajectories (watermark=%d, buffer=%d)",
                    watermark,
                    len(all_trajs),
                )
                return {
                    "status": "idempotent_noop",
                    "baked_sops_count": 0,
                    "baked_sops": [],
                    "watermark": watermark,
                    "consolidated_records": getattr(stats, "consolidated_records", 0),
                    "pruned_stale_facts": getattr(stats, "pruned_stale_facts", 0),
                    "facts_superseded": int(getattr(stats, "facts_superseded", 0)),
                    "declarative_gc_pruned": int(getattr(stats, "declarative_gc_pruned", 0)),
                }

            batch_cap = int(params.get("batch_cap", 200))
            if len(new_trajs) > batch_cap:
                new_trajs = new_trajs[-batch_cap:]

            baked_results = await baker.auto_consolidate_and_bake(
                trajectories=new_trajs, kv_store=kv, skills_dir=skills_dir
            )

            vector_store = getattr(self.engine, "vector_store", None)
            if vector_store is not None and baked_results:
                from atlas_memory.l3_procedural.sleep_baker import sop_vector_entry

                for res in baked_results:
                    try:
                        rec, embed_text = sop_vector_entry(res)
                        await vector_store.insert(
                            rec,
                            raw_text=embed_text,
                            upsert_key=f"sop::{res.get('sop_id', '')}",
                        )
                        res["vector_registered"] = True
                    except Exception as exc:
                        logger.debug("SOP vector registration failed: %s", exc)
                        res["vector_registered"] = False

            new_watermark = max((int(t.get("max_step_id", 0)) for t in new_trajs), default=watermark)
            if kv is not None and new_watermark > watermark:
                try:
                    await kv.set_state(
                        key="atlas:sleep_watermark",
                        value=new_watermark,
                        confidence=1.0,
                        metadata={"type": "sleep_watermark"},
                        reason="sleep_cycle_v64_idempotency",
                    )
                except Exception as exc:
                    logger.debug("sleep watermark advance failed: %s", exc)

            facts_superseded = int(getattr(stats, "facts_superseded", 0))
            if kv is not None and hasattr(kv, "record_sleep_checkpoint"):
                try:
                    sessions_consolidated = len(
                        {str(t.get("session_id") or t.get("session") or "") for t in new_trajs} - {""}
                    )
                    await kv.record_sleep_checkpoint(
                        checkpoint_id=f"sleep-{int(time.time() * 1000)}",
                        last_processed_step_id=new_watermark,
                        sessions_consolidated=sessions_consolidated,
                        sops_baked=len(baked_results),
                        facts_superseded=facts_superseded,
                        duration_ms=float(getattr(stats, "duration_ms", 0.0)),
                    )
                except Exception as exc:
                    logger.debug("sleep checkpoint record failed: %s", exc)

            anneal_info = await self._run_causal_anneal_phase(stats, params)
            from atlas_memory.core.agi_logger import (
                log_procedural_bake,
                log_skill_compilation,
            )
            for res in baked_results:
                sop_id = str(res.get("sop_id") or "")
                steps_cnt = int(res.get("steps_count") or len(res.get("steps") or []))
                trigger = str(res.get("signature") or res.get("name") or "")
                skill_path = str(res.get("skill_path") or "")
                log_procedural_bake(logger, sop_id, steps_cnt, trigger, skill_path=skill_path)
                if res.get("skill_registered"):
                    log_skill_compilation(logger, str(res.get("name") or sop_id), ast_status="passed", target_path=skill_path)

            log_sleep(
                logger,
                session_id="cycle",
                records_count=getattr(stats, "consolidated_records", 0),
                pruned_count=getattr(stats, "pruned_stale_facts", 0) + int(getattr(stats, "declarative_gc_pruned", 0)),
                sops_count=len(baked_results),
                duration_ms=float(getattr(stats, "duration_ms", 0.0) or 0.0),
            )

            return {
                "status": "ok",
                "consolidated_records": getattr(stats, "consolidated_records", 0),
                "pruned_stale_facts": getattr(stats, "pruned_stale_facts", 0),
                "facts_superseded": facts_superseded,
                "baked_sops_count": len(baked_results),
                "baked_sops": baked_results,
                "watermark": new_watermark,
                "trajectories_considered": len(new_trajs),
                "anneal": anneal_info,
            }
        finally:
            self._is_sleeping = False

    async def _handle_session_end(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handles session end notification from Hermes agent provider."""
        session_id = str(params.get("session_id") or "default")
        turn_count = int(params.get("turn_count") or 0)
        trigger_sleep = bool(params.get("trigger_sleep", True))

        consolidated = 0
        pruned = 0
        facts_sup = 0
        reason = "no_engine"
        if getattr(self, "engine", None) is not None:
            try:
                import importlib as _il
                _pg = _il.import_module("atlas_memory.hermes.prefix_guard")
                hook = getattr(self, "session_hook", None) or _pg.HermesSessionHook(self.engine)
                stats = await hook.on_session_end(session_id=session_id, trigger_sleep_cycle=trigger_sleep)
                consolidated = getattr(stats, "consolidated_records", 0) or getattr(stats, "facts_superseded", 0)
                pruned = getattr(stats, "pruned_stale_facts", 0) or getattr(stats, "declarative_gc_pruned", 0)
                facts_sup = getattr(stats, "facts_superseded", 0)
                if consolidated > 0:
                    reason = "consolidated"
                elif not trigger_sleep:
                    reason = "sleep_not_triggered"
                elif getattr(stats, "status", "") == "idempotent_noop":
                    reason = "idempotent_noop"
                elif getattr(stats, "status", "") == "consolidation_deferred":
                    reason = "consolidation_deferred"
                else:
                    reason = "sleep_idle"
            except Exception as exc:
                logger.warning("Error during session/end hook execution: %s", exc)
                reason = f"error: {exc}"
                facts_sup = 0

        status = "ok" if consolidated > 0 else "noop"
        if consolidated > 0 or facts_sup > 0:
            log_sleep(
                logger,
                session_id=session_id,
                records_count=consolidated,
                pruned_count=pruned,
                sops_count=0,
            )
        return {
            "status": status,
            "reason": reason,
            "session_id": session_id,
            "turn_count": turn_count,
            "consolidated_records": consolidated,
            "facts_superseded": facts_sup,
            "pruned_stale_facts": pruned,
        }

    async def _handle_telemetry_report(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return server telemetry report."""
        return {
            "pid": os.getpid(),
            "socket": str(getattr(self, "socket_path", "")),
            "serving": getattr(self, "_serving", False),
            "has_graph_client": getattr(self, "graph_client", None) is not None,
            "has_latent_buffer": getattr(self, "latent_buffer", None) is not None,
            "has_orchestrator": getattr(self, "orchestrator", None) is not None,
            "has_causal_engine": getattr(self, "causal_engine", None) is not None,
            "has_active_sensing": getattr(self, "active_sensing", None) is not None,
            "has_gossip": getattr(self, "gossip", None) is not None,
        }

    async def _handle_procedural_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Returns deep real-time status of L3 Procedural Engine and Reversible Monad."""
        engine = getattr(self, "engine", None)
        sops_cnt = 0
        sops_list = []
        if engine and hasattr(engine, "kv") and engine.kv is not None:
            try:
                conn_kv = engine.kv.ensure_connection()
                rows = conn_kv.execute(
                    "SELECT key FROM state_variables WHERE key LIKE 'sop:%'"
                ).fetchall()
                sops_cnt = len(rows)
                sops_list = [r[0] for r in rows[:50]]
            except Exception:
                pass

        traj_buf = getattr(engine, "trajectory_buffer", None)
        traj_count = len(getattr(traj_buf, "trajectories", [])) if traj_buf else 0

        auditor = getattr(engine, "auditor", None)
        decay = getattr(auditor, "decay_engine", None) if auditor else None

        from atlas_memory.l3_procedural.skill_compiler import ASTSafetyScanner
        ast_guard_active = hasattr(ASTSafetyScanner, "FORBIDDEN_CALLS")

        return {
            "status": "ok",
            "layer": "L3",
            "sops_count": sops_cnt,
            "sops": sops_list,
            "trajectories_buffered": traj_count,
            "saga_monad": "active" if auditor else "none",
            "decay_engine": "active" if decay else "none",
            "ast_safety_guard": "strict" if ast_guard_active else "none",
        }


__all__ = [
    "SleepRpcControllerMixin",
    "_resolve_sleep_watermark",
]
