from __future__ import annotations

from typing import Any, Callable, Dict

from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.models import EpistemicSource, MemoryRecord

# JSON Schema definicji narzędzi dla Hermes Agent / OpenAI-compatible Tools Runtime
SEARCH_MEMORY_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_memory",
        "description": "Odpytuje hybrydowy graf wiedzy (L2) i bazę wektorową o fakty, relacje oraz deterministyczny stan.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Semantyczne zapytanie do pamięci epizodycznej",
                },
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Kluczowe encje lub ich aliasy do sprawdzenia w grafie wiedzy",
                },
            },
            "required": ["query"],
        },
    },
}

COMMIT_OBSERVATION_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "commit_observation",
        "description": "Zgłasza nowy fakt, relację lub zmianę stanu do asynchronicznego audytora pamięci.",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "Podmiot relacji lub nazwa encji/zmiennej",
                },
                "predicate": {
                    "type": "string",
                    "description": "Predykat relacji (np. 'depends_on', 'has_status', 'state_value')",
                },
                "object": {
                    "type": "string",
                    "description": "Wartość docelowa lub obiekt relacji",
                },
                "confidence": {
                    "type": "number",
                    "description": "Współczynnik pewności faktu (0.0 do 1.0)",
                    "default": 1.0,
                },
                "is_state_variable": {
                    "type": "boolean",
                    "description": "Czy rekord stanowi twardą zmienną stanu Source of Truth",
                    "default": False,
                },
                "source_type": {
                    "type": "string",
                    "enum": ["user_explicit", "tool_output", "agent_inference", "external_doc"],
                    "description": "Proweniencja źródła informacji",
                    "default": "agent_inference",
                },
            },
            "required": ["subject", "predicate", "object"],
        },
    },
}


def create_hermes_tool_handlers(memory_engine: HybridMemoryEngine) -> Dict[str, Callable]:
    """
    Tworzy asynchroniczne handlery narzędzi powiązane z podaną instancją HybridMemoryEngine.
    """

    async def handle_search_memory(args: Dict[str, Any]) -> Dict[str, Any]:
        query = args.get("query", "")
        entities = args.get("entities", [])
        result = await memory_engine.recall(query=query, active_entities=entities)
        return {
            "semantic_matches": result.get("semantic_context", []),
            "graph_topology": result.get("graph_topology", {}),
            "verified_state": result.get("verified_state", {}),
            "latency_ms": result.get("retrieval_latency_ms", 0.0),
        }

    async def handle_commit_observation(args: Dict[str, Any]) -> Dict[str, Any]:
        src_str = args.get("source_type", "agent_inference")
        try:
            source_type = EpistemicSource(src_str)
        except ValueError:
            source_type = EpistemicSource.AGENT_INFERENCE

        record = MemoryRecord(
            subject=args.get("subject", ""),
            predicate=args.get("predicate", ""),
            object=str(args.get("object", "")),
            confidence=float(args.get("confidence", 1.0)),
            is_state_variable=bool(args.get("is_state_variable", False)),
            source_type=source_type,
        )

        await memory_engine.commit_observation(record)
        return {
            "status": "queued_for_validation",
            "subject": record.subject,
            "predicate": record.predicate,
            "object": record.object,
            "source_type": record.source_type.value,
        }

    return {
        "search_memory": handle_search_memory,
        "commit_observation": handle_commit_observation,
    }


def register_hermes_memory_tools(registry: Any, memory_engine: HybridMemoryEngine) -> None:
    """
    Rejestruje narzędzia pamięci bezpośrednio w rejestrze Tools Runtime Hermes Agent.
    """
    handlers = create_hermes_tool_handlers(memory_engine)

    if hasattr(registry, "register"):
        # Rejestracja ze schematem
        registry.register(
            name="search_memory",
            toolset="memory",
            schema=SEARCH_MEMORY_SCHEMA,
            is_async=True,
        )(handlers["search_memory"])

        registry.register(
            name="commit_observation",
            toolset="memory",
            schema=COMMIT_OBSERVATION_SCHEMA,
            is_async=True,
        )(handlers["commit_observation"])

