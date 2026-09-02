from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from atlas_memory.l2_semantic.kv_store import VerifiedKVStore

logger = logging.getLogger("atlas.sleep_baker")


class Step(BaseModel):
    """Krok procedury w pamięci proceduralnej ATLAS."""
    tool_name: str = Field(description="Nazwa narzędzia lub akcji")
    params_pattern: Dict[str, Any] = Field(default_factory=dict, description="Wzorzec lub schemat parametrów narzędzia")
    expected_outcome: Optional[str] = Field(default=None, description="Oczekiwany rezultat lub kryterium sukcesu")


class StandardProcedure(BaseModel):
    """Skompilowana standardowa procedura operacyjna (SOP)."""
    procedure_id: str = Field(description="Unikalny identyfikator procedury")
    name: str = Field(default="", description="Czytelna nazwa procedury")
    description: str = Field(default="", description="Opis procedury")
    steps: List[Step] = Field(default_factory=list, description="Uporządkowana lista kroków")
    success_rate: float = Field(default=1.0, ge=0.0, le=1.0, description="Wskaźnik sukcesu procedury")
    invocations_count: int = Field(default=1, ge=1, description="Liczba zaobserwowanych wywołań sekwencji")
    signature: Optional[str] = Field(default=None, description="Podpis sekwencji akcji")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadane i kontekst procedury")
    created_at: float = Field(default_factory=time.time, description="Timestamp utworzenia SOP")


class SleepBaker:
    """
    L3: Procedural Memory Sleep Baker & Trajectory Consolidator.
    
    W fazie Offline / Sleep Cycle analizuje bufor trajektorii z warstwy L1,
    grupuje powtarzalne sekwencje akcji i wypieka z nich Standardowe Procedury (SOP).
    Zapisuje skompilowane procedury w L2 Verified KV Store jako bazę nawyków.
    """

    def __init__(
        self,
        min_frequency: int = 2,
        min_success_rate: float = 0.7,
    ):
        self.min_frequency = min_frequency
        self.min_success_rate = min_success_rate
        self.baked_sops: Dict[str, StandardProcedure] = {}
        self._lock = asyncio.Lock()

    def _extract_steps_and_sig(self, raw_steps: List[Any]) -> tuple[str, List[Step]]:
        """Ekstrahuje sygnaturę oraz znormalizowane obiekty Step z surowej listy kroków/akcji."""
        step_objs: List[Step] = []
        tool_names: List[str] = []

        for step in raw_steps:
            if isinstance(step, Step):
                step_objs.append(step)
                tool_names.append(step.tool_name)
            elif isinstance(step, dict):
                t_name = step.get("tool_name") or step.get("action") or step.get("name") or "unknown_tool"
                params = step.get("params_pattern") or step.get("params") or {}
                outcome = step.get("expected_outcome") or step.get("outcome")
                step_objs.append(Step(tool_name=t_name, params_pattern=params, expected_outcome=outcome))
                tool_names.append(t_name)
            elif isinstance(step, str):
                step_objs.append(Step(tool_name=step, params_pattern={}))
                tool_names.append(step)
            else:
                logger.warning("Skipping unsupported trajectory step format: %s", type(step))
                continue

        signature = "->".join(tool_names)
        return signature, step_objs

    def konsolidacja_sesji(self, trajektorie: List[Dict[str, Any]]) -> List[StandardProcedure]:
        """
        Konsoliduje surowe trajektorie sesji w zbiór powtarzalnych Standardowych Procedur (SOP).
        
        Grupuje trajektorie po sekwencji narzędzi, sprawdza próg częstości (> min_frequency)
        oraz wskaźnik sukcesu.
        """
        grouped_stats: Dict[str, Dict[str, Any]] = collections.defaultdict(
            lambda: {"count": 0, "successes": 0, "sample_steps": [], "contexts": []}
        )

        for item in trajektorie:
            raw_actions = item.get("steps") or item.get("actions") or []
            if not raw_actions:
                continue

            success = bool(item.get("success", True))
            context = item.get("context", "general")

            sig, steps = self._extract_steps_and_sig(raw_actions)
            if not sig:
                continue

            grouped_stats[sig]["count"] += 1
            if success:
                grouped_stats[sig]["successes"] += 1
            if not grouped_stats[sig]["sample_steps"]:
                grouped_stats[sig]["sample_steps"] = steps
            grouped_stats[sig]["contexts"].append(context)

        consolidated: List[StandardProcedure] = []

        for sig, stats in grouped_stats.items():
            count = stats["count"]
            successes = stats["successes"]
            rate = successes / max(1, count)

            if count >= self.min_frequency and rate >= self.min_success_rate:
                proc_hash = hashlib.sha256(sig.encode("utf-8")).hexdigest()[:8]
                proc_id = f"sop_{proc_hash}"
                name = f"SOP: {sig.replace('->', ' → ')}"

                sop = StandardProcedure(
                    procedure_id=proc_id,
                    name=name,
                    steps=stats["sample_steps"],
                    success_rate=rate,
                    invocations_count=count,
                    signature=sig,
                    metadata={"contexts": list(set(stats["contexts"]))},
                )
                self.baked_sops[proc_id] = sop
                consolidated.append(sop)

        return consolidated

    async def bake_into_sop(
        self,
        procedure: StandardProcedure,
        kv_store: Optional[VerifiedKVStore] = None,
    ) -> Dict[str, Any]:
        """
        Wypieka procedurę do formatu SOP z hashem SHA-256 i opcjonalnie zapisuje w VerifiedKVStore.
        """
        steps_dump = [s.model_dump() for s in procedure.steps]
        sop_repr = json.dumps({
            "procedure_id": procedure.procedure_id,
            "signature": procedure.signature,
            "steps": steps_dump,
        }, sort_keys=True, default=str)

        sop_hash = hashlib.sha256(sop_repr.encode("utf-8")).hexdigest()

        payload: Dict[str, Any] = {
            "sop_id": procedure.procedure_id,
            "name": procedure.name,
            "signature": procedure.signature or "",
            "steps": steps_dump,
            "success_rate": procedure.success_rate,
            "invocations_count": procedure.invocations_count,
            "sop_hash": sop_hash,
            "baked_at": time.time(),
            "metadata": procedure.metadata,
        }

        if kv_store is not None:
            await kv_store.set_state(
                key=f"sop::{procedure.procedure_id}",
                value=payload,
                confidence=procedure.success_rate,
                metadata={"type": "baked_sop", "sop_hash": sop_hash},
                reason="sleep_baking_sop_compilation",
            )

        async with self._lock:
            self.baked_sops[procedure.procedure_id] = procedure
        return payload

    async def auto_consolidate_and_bake(
        self,
        trajectories: List[Dict[str, Any]],
        kv_store: Optional[VerifiedKVStore] = None,
        skills_dir: Optional[Union[str, Path]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Wektor V37: End-to-end automatyczna konsolidacja trajektorii i wypiekanie skilli.
        
        Ekstrahuje kwalifikowane SOP, zapisuje je w KV Store i kompiluje do natywnego środowiska Hermesa.
        """
        sops = self.konsolidacja_sesji(trajectories)
        baked_results = []
        for sop in sops:
            baked = await self.bake_into_sop(sop, kv_store=kv_store)
            if skills_dir is not None:
                try:
                    from atlas_memory.l3_procedural.skill_compiler import compile_and_register_sop
                    compile_and_register_sop(sop, hermes_skills_root=Path(skills_dir))
                    baked["skill_registered"] = True
                except Exception as exc:
                    logger.warning("Could not auto-compile skill for SOP %s: %s", sop.procedure_id, exc)
                    baked["skill_registered"] = False
            baked_results.append(baked)
        return baked_results
