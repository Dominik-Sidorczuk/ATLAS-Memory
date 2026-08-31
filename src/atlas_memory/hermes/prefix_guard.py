from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.l3_procedural.skill_compiler import SafetyViolationError, compile_sop_to_skill
from atlas_memory.l3_procedural.sleep_baker import StandardProcedure
from atlas_memory.models import ConsolidationStats

logger = logging.getLogger(__name__)


class PrefixCacheGuard:
    """
    Krok B: Prefix Cache Guard.
    
    Generuje deterministyczny, niezmienny blok stanu wstrzykiwany do system_prompt
    na samym początku sesji. Zapobiega unieważnieniu bufora KV Cache (prefix cache)
    w lokalnych silnikach inferencji (llama-cpp-python, vLLM, Ollama).
    """

    @staticmethod
    def build_immutable_prefix(
        verified_state: Dict[str, Any],
        system_rules: Optional[List[str]] = None,
        base_entities: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Tworzy deterministyczny ciąg znaków promptu bazowego z posortowanymi kluczami.
        """
        lines = [
            "# === HERMES AGENT IMMUTABLE KERNEL STATE ===",
            "<!-- KV-CACHE-PREFIX-GUARD: DO NOT MODIFY DURING SESSION -->",
        ]

        if system_rules:
            lines.append("## System Constraints:")
            for rule in sorted(system_rules):
                lines.append(f"- {rule}")

        if verified_state:
            lines.append("## Verified Source-of-Truth State:")
            # Sortowanie kluczy dla deterministycznego cache hitu
            for k in sorted(verified_state.keys()):
                v = verified_state[k]
                val_repr = v.get("value") if isinstance(v, dict) else v
                lines.append(f"- {k}: {json.dumps(val_repr, ensure_ascii=False)}")

        if base_entities:
            lines.append("## Registered Canonical Entities:")
            for item in sorted(base_entities, key=lambda x: x.get("id", "")):
                lines.append(f"- {item.get('id')}: {item.get('name')} (aliases: {', '.join(sorted(item.get('aliases', [])))})")

        lines.append("# === END OF IMMUTABLE KERNEL STATE ===\n")
        return "\n".join(lines)


class HermesSessionHook:
    """
    Session Hook zarządzający cyklem życia sesji w Hermes Agent:
    - on_session_start: wstrzyknięcie niezmiennego prefixu
    - on_session_end: asynchroniczny Sleep Cycle (konsolidacja i GC)
    """

    def __init__(self, memory_engine: HybridMemoryEngine):
        self.engine = memory_engine
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

    async def on_session_start(
        self,
        session_id: str,
        active_entities: Optional[List[str]] = None,
        system_rules: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Huk wywoływany na początku sesji:
        Pobiera niezmienny stan L2 i generuje prefix cache guard.
        """
        entities = active_entities or ["entity_nas_01", "entity_agent_core"]
        verified_states = await self.engine.kv.get_states(entities)

        prefix_prompt = PrefixCacheGuard.build_immutable_prefix(
            verified_state=verified_states,
            system_rules=system_rules or ["Deterministic execution required", "Use search_memory for dynamic relations"],
        )

        session_meta = {
            "session_id": session_id,
            "start_time": time.time(),
            "entities": entities,
            "prefix_prompt": prefix_prompt,
        }
        self.active_sessions[session_id] = session_meta

        return {
            "session_id": session_id,
            "prefix_system_prompt": prefix_prompt,
            "verified_states": verified_states,
        }

    async def on_session_end(
        self,
        session_id: str,
        trigger_sleep_cycle: bool = True,
    ) -> ConsolidationStats:
        """
        Huk wywoływany po zakończeniu sesji:
        1. Opróżnienie kolejki commitów.
        2. Opcjonalne uruchomienie nocnej fazy konsolidacji (Sleep Cycle).
        """
        # 1. Przetworzenie wszystkich zaległych rekordów
        await self.engine.process_all_pending()

        # 2. Uruchomienie fazy snu
        stats = ConsolidationStats()
        if trigger_sleep_cycle:
            stats = await self.engine.auditor.run_sleep_cycle_consolidation()

            # Sprawdź SOP z success_rate >= 0.9 i invocations_count >= 5 do propozycji kompilacji
            proposed: List[str] = []
            sleep_baker = getattr(self.engine, "sleep_baker", None) or getattr(self.engine.auditor, "sleep_baker", None)
            baked_sops: Dict[str, StandardProcedure] = getattr(sleep_baker, "baked_sops", {}) if sleep_baker else {}

            # Sprawdź również bezpośrednio w bazie KV (jeśli procedury zostały tam zapisane)
            for _sop_id, sop in baked_sops.items():
                if sop.success_rate >= 0.9 and sop.invocations_count >= 5:

                    logger.info("Auto-propose skill compilation for %s", sop.name)
                    proposed.append(sop.procedure_id)
                    try:
                        compile_sop_to_skill(sop)
                        logger.info("Compiled skill: %s", sop.procedure_id)
                    except SafetyViolationError as e:
                        logger.warning("Unsafe SOP %s rejected during auto-propose: %s", sop.procedure_id, e)
                    except Exception as e:
                        logger.error("Failed to compile skill for SOP %s: %s", sop.procedure_id, e)

            stats.proposed_skills = proposed

        # 3. Zakończenie sesji
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

        return stats

