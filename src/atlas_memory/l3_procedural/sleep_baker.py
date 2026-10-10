from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from atlas_memory.core.models import MemoryRecord
from atlas_memory.l2_semantic.kv_store import VerifiedKVStore

logger = logging.getLogger("atlas.sleep_baker")

# V66 (audyt §3A): twarde dno progu częstości — pojedyncza sekwencja (1 wystąpienie)
# nie jest powtarzalną procedurą, więc dynamika progu nigdy nie schodzi poniżej 2.
MIN_FREQUENCY_FLOOR: int = 2


def sop_vector_entry(payload: Dict[str, Any]) -> Tuple[Any, str]:
    """V66 (LUKA 4, BUG-011): buduje (MemoryRecord, embed_text) dla wypieczonego SOP-a.

    Wspólne ciało logiki dla ``SleepBaker.bake_into_sop`` (ścieżka in-process) i
    daemona (rejestracja na głównej pętli zdarzeń po bake w wątku roboczym —
    locki ``QdrantVectorStore`` są związane z pętlą). Tekst embedowany zawiera
    nazwę + sekwencję narzędzi + sygnaturę, żeby szerokie ``atlas_recall``
    znajdowało procedurę w TEJ SAMEJ przestrzeni wektorowej co fakty.
    """

    sop_id = str(payload.get("sop_id", ""))
    steps = payload.get("steps") or []
    tool_sequence = " -> ".join(
        str(s.get("tool_name", "")) for s in steps if isinstance(s, dict)
    )
    signature = str(payload.get("signature") or tool_sequence)
    name = str(payload.get("name") or f"SOP {sop_id}")
    embed_text = (
        f"SOP {sop_id}: {name}. "
        f"Secwencja narzedzi: {tool_sequence}. "
        f"Sygnatura: {signature}. "
        f"Procedura wypieczona przez konsolidacje snu ATLAS."
    )
    record = MemoryRecord(
        subject=f"sop::{sop_id}",
        predicate="baked_sop",
        object=signature,
        confidence=float(payload.get("success_rate", 1.0) or 1.0),
        timestamp=float(payload.get("baked_at") or time.time()),
        is_state_variable=False,
        metadata={"type": "baked_sop", "sop_hash": str(payload.get("sop_hash", ""))},
    )
    return record, embed_text


def compute_effective_min_frequency(
    sessions_count: int,
    alpha: float = 1.0,
    override: Optional[int] = None,
    floor: int = MIN_FREQUENCY_FLOOR,
) -> int:
    """V66 (§3A): dynamiczny próg populacyjny dla wypiekania SOP.

    ``effective_min_freq = max(floor, ceil(alpha * log2(N_sesji)))``.
    Jawnie podany ``override`` (np. z RPC ``min_frequency``) NIGDY nie jest
    nadpisywany — zwracany jest jego exact value (nawet 1), bo to świadoma
    decyzja wołającego. Bez override próg rośnie logarytmicznie z liczbą
    unikalnych sesji, żeby SOP-y nie były wypiekane z szumu małej próby.

    Granice (alpha=1.0, floor=2): 1-3 sesje => 2, 4 => 2, 5-8 => 3, 9-16 => 4.
    """
    if override is not None:
        return int(override)
    n = max(1, int(sessions_count))
    if alpha <= 0.0:
        return int(floor)
    return int(max(floor, math.ceil(alpha * math.log2(n))))


class Step(BaseModel):
    """Krok procedury w pamięci proceduralnej ATLAS z modelem ciągłej przestrzeni dystrybucyjnej."""
    tool_name: str = Field(description="Nazwa narzędzia lub akcji")
    params_pattern: Dict[str, Any] = Field(default_factory=dict, description="Wzorzec lub schemat parametrów narzędzia")
    expected_outcome: Optional[str] = Field(default=None, description="Oczekiwany rezultat lub kryterium sukcesu")
    distribution_space: Dict[str, Any] = Field(default_factory=dict, description="Parametryczna przestrzeń rozkładu argumentów")
    thermal_damping: float = Field(default=0.85, description="Współczynnik tłumienia termicznego kroku")
    adaptation_vector: Optional[List[float]] = Field(default=None, description="Wektor adaptacji w ciągłej przestrzeni latentnej")


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


GENERIC_EXECUTION_TOOLS = {"bash", "terminal_run", "exec", "run_command", "terminal", "shell", "sh"}


def _resolve_tool_signature_token(t_name: str, params: Dict[str, Any]) -> str:
    """Disambiguates generic execution tools with their command/subcommand (e.g. bash[rm])."""
    base_name = str(t_name).strip()
    if base_name.lower() in GENERIC_EXECUTION_TOOLS and isinstance(params, dict):
        cmd = params.get("command") or params.get("cmd") or params.get("subcommand") or params.get("action")
        if isinstance(cmd, str) and cmd.strip():
            first_token = cmd.strip().split()[0].split("/")[-1]
            if first_token:
                return f"{base_name}[{first_token}]"
    return base_name


from atlas_memory.l3_procedural.skill_compiler import compile_and_register_sop


class SleepBaker:
    """
    L3: Procedural Memory Sleep Baker & Trajectory Consolidator.
    
    W fazie Offline / Sleep Cycle analizuje bufor trajektorii z warstwy L1,
    grupuje powtarzalne sekwencje akcji i wypieka z nich Standardowe Procedury (SOP).
    Zapisuje skompilowane procedury w L2 Verified KV Store jako bazę nawyków.
    """

    def __init__(
        self,
        min_frequency: Optional[int] = None,
        min_success_rate: float = 0.7,
        alpha: float = 1.0,
    ):
        # V66 (§3A): `min_frequency` to NADPISANY próg (override) — gdy podany, jest
        # użyty co do wartości (exact value) i NIGDY nie jest nadpisywany przez
        # dynamikę populacyjną. Gdy None — próg liczony dynamicznie z liczby
        # unikalnych sesji w analizowanych trajektoriach (alpha = konfig).
        self.min_frequency = min_frequency
        self.alpha = float(alpha)
        self.min_success_rate = min_success_rate
        # Wsteczna kompatybilność: `effective_min_frequency` to próg faktycznie
        # użyty w ostatnim uruchomieniu konsolidacji (deterministyczny default = dno).
        self.effective_min_frequency: int = (
            int(min_frequency) if min_frequency is not None else MIN_FREQUENCY_FLOOR
        )
        self.baked_sops: Dict[str, StandardProcedure] = {}
        self._lock = asyncio.Lock()

    def count_unique_sessions(self, trajectories: List[Dict[str, Any]]) -> int:
        """V66 (§3A): liczba unikalnych `session_id` w analizowanych trajektoriach."""
        sessions = set()
        for item in trajectories or []:
            if not isinstance(item, dict):
                continue
            sid = item.get("session_id") or item.get("session")
            if sid is not None and str(sid).strip():
                sessions.add(str(sid))
        return len(sessions)

    def resolve_min_frequency(self, trajectories: List[Dict[str, Any]]) -> int:
        """V66 (§3A): efektowny próg częstości dla partii trajektorii.

        Override (jawnie podany `min_frequency`) wygrywa zawsze; w przeciwnym razie
        `max(2, ceil(alpha * log2(N_sesji)))`.
        """
        effective = compute_effective_min_frequency(
            sessions_count=self.count_unique_sessions(trajectories),
            alpha=self.alpha,
            override=self.min_frequency,
        )
        self.effective_min_frequency = effective
        return effective

    def _extract_steps_and_sig(self, raw_steps: List[Any]) -> tuple[str, List[Step]]:
        """Ekstrahuje sygnaturę oraz znormalizowane obiekty Step z surowej listy kroków/akcji."""
        step_objs: List[Step] = []
        tool_names: List[str] = []

        for step in raw_steps:
            if isinstance(step, Step):
                sig_tok = _resolve_tool_signature_token(step.tool_name, step.params_pattern)
                step_objs.append(step)
                tool_names.append(sig_tok)
            elif isinstance(step, dict):
                t_name = step.get("tool_name") or step.get("tool") or step.get("action") or step.get("name") or "unknown_tool"
                params = step.get("params_pattern") or step.get("params") or {}
                outcome = step.get("expected_outcome") or step.get("outcome")
                sig_tok = _resolve_tool_signature_token(t_name, params)
                step_objs.append(Step(tool_name=sig_tok, params_pattern=params, expected_outcome=outcome))
                tool_names.append(sig_tok)
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

        Grupuje trajektorie po sekwencji narzędzi, sprawdza próg częstości
        (V66 §3A: dynamiczny próg populacyjny `max(2, ceil(alpha*log2(N_sesji)))`,
        chyba że `min_frequency` został jawnie nadpisany — wtedy exact value)
        oraz wskaźnik sukcesu.
        """
        effective_min_freq = self.resolve_min_frequency(trajektorie)

        grouped_stats: Dict[str, Dict[str, Any]] = collections.defaultdict(
            lambda: {"count": 0, "successes": 0, "sample_steps": [], "all_steps": [], "contexts": []}
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
            grouped_stats[sig]["all_steps"].append(steps)
            grouped_stats[sig]["contexts"].append(context)

        consolidated: List[StandardProcedure] = []

        for sig, stats in grouped_stats.items():
            count = stats["count"]
            successes = stats["successes"]
            rate = successes / max(1, count)

            if count >= effective_min_freq and rate >= self.min_success_rate:
                proc_hash = hashlib.sha256(sig.encode("utf-8")).hexdigest()[:8]
                proc_id = f"sop_{proc_hash}"
                name = f"SOP: {sig.replace('->', ' → ')}"

                # DYREKTYWA 1: Wyznacz ciągłą przestrzeń dystrybucyjną argumentów
                sample_steps = stats["sample_steps"]
                all_executions = stats["all_steps"]
                rich_steps: List[Step] = []
                for s_idx, base_step in enumerate(sample_steps):
                    collected_params: Dict[str, List[Any]] = collections.defaultdict(list)
                    for exec_seq in all_executions:
                        if s_idx < len(exec_seq):
                            for p_k, p_v in exec_seq[s_idx].params_pattern.items():
                                collected_params[p_k].append(p_v)

                    dist_space: Dict[str, Any] = {}
                    for p_k, val_list in collected_params.items():
                        if not val_list:
                            continue
                        numeric_vals = [float(v) for v in val_list if isinstance(v, (int, float)) and not isinstance(v, bool)]
                        if len(numeric_vals) == len(val_list) and len(numeric_vals) > 0:
                            mean_v = float(sum(numeric_vals) / len(numeric_vals))
                            var_v = float(sum((x - mean_v) ** 2 for x in numeric_vals) / max(1, len(numeric_vals)))
                            dist_space[p_k] = {
                                "type": "continuous_gaussian",
                                "mean": mean_v,
                                "variance": var_v,
                                "std": math.sqrt(var_v),
                                "bounds": [float(min(numeric_vals)), float(max(numeric_vals))],
                            }
                        else:
                            counts = collections.Counter(str(v) for v in val_list)
                            tot = len(val_list)
                            dist_space[p_k] = {
                                "type": "discrete_distribution",
                                "frequencies": {k: v / tot for k, v in counts.items()},
                                "cardinality": len(counts),
                            }

                    rich_steps.append(
                        Step(
                            tool_name=base_step.tool_name,
                            params_pattern=base_step.params_pattern,
                            expected_outcome=base_step.expected_outcome,
                            distribution_space=dist_space,
                            thermal_damping=0.85,
                        )
                    )

                sop = StandardProcedure(
                    procedure_id=proc_id,
                    name=name,
                    steps=rich_steps,
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
        vector_store: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Wypieka procedurę do formatu SOP z hashem SHA-256 i opcjonalnie zapisuje w VerifiedKVStore.

        V66 (LUKA 4, BUG-011): gdy podano ``vector_store``, SOP rejestrowany jest w TEJ
        SAMEJ przestrzeni wektorowej co fakty (encoder istniejącego indeksu), z
        ``upsert_key=sop::<id>`` — szeroki ``atlas_recall`` znajduje procedurę, a re-bake
        nie tworzy duplikatu wektora.
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

        if vector_store is not None:
            try:
                rec, embed_text = sop_vector_entry(payload)
                await vector_store.insert(
                    rec,
                    raw_text=embed_text,
                    upsert_key=f"sop::{procedure.procedure_id}",
                )
                payload["vector_registered"] = True
            except Exception as exc:  # wektor nie może wywrócić wypieku
                logger.warning("SOP vector registration failed for %s: %s", procedure.procedure_id, exc)
                payload["vector_registered"] = False

        async with self._lock:
            self.baked_sops[procedure.procedure_id] = procedure
        return payload

    async def auto_consolidate_and_bake(
        self,
        trajectories: List[Dict[str, Any]],
        kv_store: Optional[VerifiedKVStore] = None,
        skills_dir: Optional[Union[str, Path]] = None,
        vector_store: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Wektor V37: End-to-end automatyczna konsolidacja trajektorii i wypiekanie skilli.
        
        Ekstrahuje kwalifikowane SOP, zapisuje je w KV Store i kompiluje do natywnego środowiska Hermesa.
        V66: ``vector_store`` rejestruje SOP-y w przestrzeni wektorowej (BUG-011).
        V68 (ARCH-006): wynik = skille dla warstwy agenta Hermesa; silnik
        ATLAS nie importuje ProcedureHandler (uczciwa etykieta generatora,
        nie odruch O(1) — wykonawca poza zakresem, decyzja właściciela).
        """
        sops = self.konsolidacja_sesji(trajectories)
        baked_results = []
        for sop in sops:
            baked = await self.bake_into_sop(sop, kv_store=kv_store, vector_store=vector_store)
            if skills_dir is not None:
                try:
                    compile_and_register_sop(sop, hermes_skills_root=Path(skills_dir))
                    baked["skill_registered"] = True
                except Exception as exc:
                    logger.warning("Could not auto-compile skill for SOP %s: %s", sop.procedure_id, exc)
                    baked["skill_registered"] = False
            baked_results.append(baked)
        return baked_results
