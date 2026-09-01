from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

if TYPE_CHECKING:
    pass

from atlas_memory.models import MemoryRecord
from atlas_memory.server.client import send_uds_request_sync
from atlas_memory.server.models import DEFAULT_SOCKET_PATH

ATLAS_RECALL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "atlas_recall",
        "description": "Odpytuje pamięć ATLAS (Kùzu Knowledge Graph, Qdrant wektory, Verified KV) o powiązane fakty, relacje i stan.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Semantyczne zapytanie do pamięci ATLAS",
                },
                "session_id": {
                    "type": "string",
                    "description": "Identyfikator sesji (opcjonalny)",
                    "default": "hermes_default",
                },
            },
            "required": ["query"],
        },
    },
}

ATLAS_REMEMBER_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "atlas_remember",
        "description": "Zapisuje fakt, regułę lub zmienną stanu do pamięci ATLAS z łańcuchem audytu SHA-256.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Klucz zmiennej stanu lub podmiot faktu (np. 'user:preference:language')",
                },
                "value": {
                    "type": "string",
                    "description": "Wartość lub treść faktu",
                },
                "confidence": {
                    "type": "number",
                    "description": "Współczynnik pewności faktu (0.0 do 1.0)",
                    "default": 1.0,
                },
                "reason": {
                    "type": "string",
                    "description": "Uzasadnienie zapisu lub źródło",
                    "default": "agent_explicit",
                },
            },
            "required": ["key", "value"],
        },
    },
}

ATLAS_WHAT_IF_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "atlas_what_if",
        "description": "Symuluje skutki przyczynowo-skutkowe planowanej akcji w grafie zależności Kùzu (What-If Reasoning & CPoF).",
        "parameters": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "Encja wyjściowa poddawana analizie (np. 'GHK-Cu', 'ProductionDatabase')",
                },
                "action": {
                    "type": "string",
                    "description": "Symulowana akcja (np. 'simulate_dose_increase', 'delete_cache', 'modify_schema')",
                },
                "depth": {
                    "type": "integer",
                    "description": "Maksymalna głębokość grafu (domyślnie 2 hopy)",
                    "default": 2,
                },
            },
            "required": ["entity", "action"],
        },
    },
}

ATLAS_ACTIVE_SENSING_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "atlas_active_sensing",
        "description": "Weryfikuje oczekiwany stan środowiska i wykrywa anomalie (Predictive Coding / Prediction Error).",
        "parameters": {
            "type": "object",
            "properties": {
                "probe": {
                    "type": "string",
                    "description": "Badany parametr lub encja (np. 'KlowStack_storage_temp', 'api_endpoint_status')",
                },
                "expected_value": {
                    "type": "string",
                    "description": "Oczekiwana wartość w normalnych warunkach (np. '4C', '200_OK')",
                },
                "observed_value": {
                    "type": "string",
                    "description": "Rzeczywista zaobserwowana wartość (np. '25C', '500_ERROR')",
                },
            },
            "required": ["probe", "observed_value"],
        },
    },
}

ATLAS_STATS_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "atlas_stats",
        "description": "Zwraca status daemona ATLAS, liczbę rekordów w KV, stan grafu Kùzu i telemetrię.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}


def create_uds_tool_handlers(socket_path: Path | str = DEFAULT_SOCKET_PATH) -> Dict[str, Callable]:
    """Tworzy handlery narzędzi komunikujące się bezpośrednio z daemonem ATLAS przez UDS."""
    sock = Path(socket_path)

    def handle_atlas_recall(args: Dict[str, Any]) -> Dict[str, Any]:
        query = args.get("query", "")
        session_id = args.get("session_id", "hermes_default")
        res = send_uds_request_sync(sock, "prefetch", {"query": query, "session_id": session_id})
        return res if isinstance(res, dict) else {"records": [], "count": 0, "status": "daemon_offline"}

    def handle_atlas_remember(args: Dict[str, Any]) -> Dict[str, Any]:
        key = args.get("key", "")
        val = args.get("value", "")
        conf = float(args.get("confidence", 1.0))
        reason = args.get("reason", "agent_explicit")
        res = send_uds_request_sync(sock, "set", {"key": key, "value": val, "confidence": conf, "reason": reason})
        return res if isinstance(res, dict) else {"status": "error", "persisted": False}

    def handle_atlas_what_if(args: Dict[str, Any]) -> Dict[str, Any]:
        entity = args.get("entity", "")
        action = args.get("action", "simulate")
        depth = int(args.get("depth", 2))
        res = send_uds_request_sync(sock, "what_if", {"entity": entity, "action": action, "depth": depth})
        return res if isinstance(res, dict) else {"status": "error", "causal_paths": []}

    def handle_atlas_active_sensing(args: Dict[str, Any]) -> Dict[str, Any]:
        probe = args.get("probe", "")
        exp = args.get("expected_value", "")
        obs = args.get("observed_value", "")
        res = send_uds_request_sync(sock, "active_sensing", {"target_entity": probe, "expected_value": exp, "observed_value": obs})
        return res if isinstance(res, dict) else {"status": "error", "has_error": False}

    def handle_atlas_stats(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        res = send_uds_request_sync(sock, "get_stats", {})
        return res if isinstance(res, dict) else {"status": "error", "serving": False}

    return {
        "atlas_recall": handle_atlas_recall,
        "atlas_remember": handle_atlas_remember,
        "atlas_what_if": handle_atlas_what_if,
        "atlas_active_sensing": handle_atlas_active_sensing,
        "atlas_stats": handle_atlas_stats,
    }


# Backwards-compatibility aliases
SEARCH_MEMORY_SCHEMA = ATLAS_RECALL_SCHEMA
COMMIT_OBSERVATION_SCHEMA = ATLAS_REMEMBER_SCHEMA


def create_hermes_tool_handlers(memory_engine: Any) -> Dict[str, Callable]:
    """Compatibility wrapper for engine-attached handlers."""
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
        record = MemoryRecord(
            subject=args.get("subject", args.get("key", "")),
            predicate=args.get("predicate", "state_value"),
            object=str(args.get("object", args.get("value", ""))),
            confidence=float(args.get("confidence", 1.0)),
            is_state_variable=bool(args.get("is_state_variable", False)),
        )
        await memory_engine.commit_observation(record)
        return {"status": "queued_for_validation", "subject": record.subject}

    return {
        "search_memory": handle_search_memory,
        "commit_observation": handle_commit_observation,
    }


def register_hermes_memory_tools(registry: Any, memory_engine: Any) -> None:
    """Compatibility registry function."""
    handlers = create_hermes_tool_handlers(memory_engine)
    if hasattr(registry, "register"):
        registry.register(name="search_memory", toolset="memory", schema=SEARCH_MEMORY_SCHEMA, is_async=True)(handlers["search_memory"])
        registry.register(name="commit_observation", toolset="memory", schema=COMMIT_OBSERVATION_SCHEMA, is_async=True)(handlers["commit_observation"])


