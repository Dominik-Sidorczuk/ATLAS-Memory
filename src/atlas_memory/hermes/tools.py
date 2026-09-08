from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from atlas_memory.cognitive.models import HERMES_DEFAULT_SESSION
from atlas_memory.models import MemoryRecord
from atlas_memory.server.client import send_uds_request_sync
from atlas_memory.server.models import DEFAULT_SOCKET_PATH

logger = logging.getLogger(__name__)

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
                    "default": HERMES_DEFAULT_SESSION,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maksymalna liczba rekordów do zwrócenia (domyślnie 15)",
                    "default": 15,
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
                    "description": "Encja wyjściowa poddawana analizie (np. 'DatabaseCluster', 'AuthService')",
                },
                "action": {
                    "type": "string",
                    "description": "Symulowana akcja (np. 'restart_service', 'delete_cache', 'modify_schema')",
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
                    "description": "Badany parametr lub encja (np. 'database_port', 'api_endpoint_status')",
                },
                "expected_value": {
                    "type": "string",
                    "description": "Oczekiwana wartość w normalnych warunkach (np. '5432', '200_OK')",
                },
                "observed_value": {
                    "type": "string",
                    "description": "Rzeczywista zaobserwowana wartość (np. '5433', '500_ERROR')",
                },
                "tolerance": {
                    "type": "number",
                    "description": "Dopuszczalna tolerancja numeryczna (brak błędu gdy |obs - exp| <= tolerance)",
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

ATLAS_GET_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "atlas_get",
        "description": "Pobiera stan lub wartość faktu bezpośrednio z Verified KV Store ATLAS po dokładnym kluczu (szybki dostęp O(1)).",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Dokładny klucz stanu lub faktu do pobrania z Verified KV Store",
                },
            },
            "required": ["key"],
        },
    },
}

ATLAS_DELETE_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "atlas_delete",
        "description": "Usuwa zmienną stanu lub fakt z pamięci ATLAS w Verified KV Store (atomowe usunięcie z łańcuchem audytu).",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Klucz zmiennej stanu lub faktu do usunięcia",
                },
                "reason": {
                    "type": "string",
                    "description": "Uzasadnienie usunięcia wpisu z pamięci",
                },
            },
            "required": ["key"],
        },
    },
}

ATLAS_TASK_FEEDBACK_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "atlas_task_feedback",
        "description": "Zgłasza informację zwrotną o wyniku zadania dla rekordu pamięci, podbijając confidence i adaptując online wagi L0-TTT.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Klucz rekordu pamięci, którego dotyczy wynik zadania",
                },
                "task_success": {
                    "type": "boolean",
                    "description": "Czy zadanie powiązane z tym faktem zakończyło się sukcesem (True/False)",
                },
                "delta": {
                    "type": "number",
                    "description": "Wartość zmiany współczynnika confidence (domyślnie 0.05)",
                    "default": 0.05,
                },
            },
            "required": ["key", "task_success"],
        },
    },
}

ATLAS_VERIFY_CHAIN_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "atlas_verify_chain",
        "description": "Weryfikuje nienaruszalność kryptograficznego łańcucha audytu SHA-256 w magazynie pamięci ATLAS.",
        "parameters": {
            "type": "object",
            "properties": {
                "heal": {
                    "type": "boolean",
                    "description": "Czy automatycznie naprawić przerwany łańcuch audytu (domyślnie false)",
                    "default": False,
                },
                "deep": {
                    "type": "boolean",
                    "description": "Czy przeprowadzić głęboką weryfikację hashy wartości (domyślnie false)",
                    "default": False,
                },
            },
        },
    },
}

ATLAS_SLEEP_SYNC_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "atlas_sleep_sync",
        "description": "Wymusza natychmiastową procedurę konsolidacji snu L3, Salience GC i destylację procedur SOP.",
        "parameters": {
            "type": "object",
            "properties": {
                "skills_dir": {
                    "type": "string",
                    "description": "Opcjonalna ścieżka do katalogu zapisu wygenerowanych procedur SOP",
                },
            },
        },
    },
}


ATLAS_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    ATLAS_RECALL_SCHEMA,
    ATLAS_REMEMBER_SCHEMA,
    ATLAS_WHAT_IF_SCHEMA,
    ATLAS_ACTIVE_SENSING_SCHEMA,
    ATLAS_STATS_SCHEMA,
    ATLAS_GET_SCHEMA,
    ATLAS_DELETE_SCHEMA,
    ATLAS_TASK_FEEDBACK_SCHEMA,
    ATLAS_VERIFY_CHAIN_SCHEMA,
    ATLAS_SLEEP_SYNC_SCHEMA,
]


def create_uds_tool_handlers(socket_path: Path | str = DEFAULT_SOCKET_PATH) -> Dict[str, Callable]:
    """Tworzy handlery narzędzi komunikujące się bezpośrednio z daemonem ATLAS przez UDS."""
    sock = Path(socket_path)

    def handle_atlas_recall(args: Dict[str, Any]) -> Dict[str, Any]:
        query = args.get("query", "")
        session_id = args.get("session_id", HERMES_DEFAULT_SESSION)
        try:
            raw_limit = int(args.get("limit", 15))
        except (ValueError, TypeError):
            raw_limit = 15
        limit = max(1, min(raw_limit, 100))
        res = send_uds_request_sync(
            sock,
            "prefetch",
            {"query": query, "session_id": session_id, "limit": limit, "force": args.get("force", False)},
        )
        return res if isinstance(res, dict) else {"records": [], "count": 0, "status": "daemon_offline"}

    def handle_atlas_remember(args: Dict[str, Any]) -> Dict[str, Any]:
        key = args.get("key") or args.get("subject") or args.get("name", "")
        val = args.get("value") or args.get("object") or args.get("content", "")
        conf = float(args.get("confidence", 1.0))
        reason = args.get("reason", "agent_explicit")
        session_id = args.get("session_id") or HERMES_DEFAULT_SESSION
        payload = {
            "key": key,
            "value": val,
            "confidence": conf,
            "reason": reason,
            "session_id": session_id,
        }
        if "outcome" in args:
            payload["outcome"] = args["outcome"]
        if "task_success" in args:
            payload["task_success"] = args["task_success"]
        res = send_uds_request_sync(
            sock,
            "set",
            payload,
        )
        return res if isinstance(res, dict) else {"status": "error", "persisted": False}

    def handle_commit_observation_uds(args: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "subject": args.get("subject", args.get("key", "")),
            "predicate": args.get("predicate", "observation"),
            "object": str(args.get("object", args.get("value", ""))),
            "confidence": float(args.get("confidence", 1.0)),
            "session_id": args.get("session_id", HERMES_DEFAULT_SESSION),
            "task_success": args.get("task_success", False),
        }
        if "outcome" in args:
            payload["outcome"] = args["outcome"]
        res = send_uds_request_sync(sock, "commit_observation", payload)
        return res if isinstance(res, dict) else {"status": "error", "committed": False}

    def handle_atlas_what_if(args: Dict[str, Any]) -> Dict[str, Any]:
        entity = args.get("entity") or args.get("target", "")
        action = args.get("action", "simulate")
        depth = int(args.get("depth", 2))
        res = send_uds_request_sync(sock, "what_if", {"entity": entity, "action": action, "depth": depth})
        return res if isinstance(res, dict) else {"status": "error", "causal_paths": []}

    def handle_atlas_active_sensing(args: Dict[str, Any]) -> Dict[str, Any]:
        probe = args.get("probe") or args.get("target_entity", "")
        exp = args.get("expected_value", "")
        obs = args.get("observed_value", "")
        tol = args.get("tolerance")
        payload: Dict[str, Any] = {
            "target_entity": probe,
            "expected_value": exp,
            "observed_value": obs,
        }
        if tol is not None:
            try:
                payload["tolerance"] = float(tol)
            except (ValueError, TypeError):
                pass
        res = send_uds_request_sync(
            sock,
            "active_sensing",
            payload,
        )
        return res if isinstance(res, dict) else {"status": "error", "has_error": False}

    def handle_atlas_stats(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        res = send_uds_request_sync(sock, "get_stats", {})
        return res if isinstance(res, dict) else {"status": "error", "serving": False}

    def handle_atlas_get(args: Dict[str, Any]) -> Dict[str, Any]:
        key = str(args.get("key", "")).strip()
        res = send_uds_request_sync(sock, "get", {"key": key})
        return res if isinstance(res, dict) else {"status": "error", "key": key, "found": False, "state": None}

    def handle_atlas_delete(args: Dict[str, Any]) -> Dict[str, Any]:
        key = str(args.get("key", "")).strip()
        reason = args.get("reason", "agent_delete")
        payload: Dict[str, Any] = {"key": key}
        if reason:
            payload["reason"] = reason
        res = send_uds_request_sync(sock, "delete", payload)
        return res if isinstance(res, dict) else {"status": "error", "key": key, "deleted": False, "found": False}

    def handle_atlas_task_feedback(args: Dict[str, Any]) -> Dict[str, Any]:
        key = str(args.get("key", "")).strip()
        task_success = bool(args.get("task_success", True))
        try:
            delta = float(args.get("delta", 0.05))
        except (ValueError, TypeError):
            delta = 0.05
        res = send_uds_request_sync(sock, "task_feedback", {"key": key, "task_success": task_success, "delta": delta})
        return res if isinstance(res, dict) else {"status": "error", "key": key, "confidence": 0.0, "task_success": task_success}

    def handle_atlas_verify_chain(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        args = args or {}
        heal = bool(args.get("heal", False))
        deep = bool(args.get("deep", False))
        res = send_uds_request_sync(sock, "verify_audit_log", {"heal": heal, "deep": deep}, timeout=20.0)
        return res if isinstance(res, dict) else {"status": "error", "valid": False, "checked_entries": 0}

    def handle_atlas_sleep_sync(args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        args = args or {}
        skills_dir = args.get("skills_dir")
        payload: Dict[str, Any] = {}
        if skills_dir:
            payload["skills_dir"] = str(skills_dir)
        res = send_uds_request_sync(sock, "trigger_sleep_consolidation", payload, timeout=30.0)
        return res if isinstance(res, dict) else {
            "status": "error",
            "consolidated_records": 0,
            "pruned_stale_facts": 0,
            "baked_sops_count": 0,
            "baked_sops": [],
        }

    return {
        "atlas_recall": handle_atlas_recall,
        "atlas_remember": handle_atlas_remember,
        "atlas_what_if": handle_atlas_what_if,
        "atlas_active_sensing": handle_atlas_active_sensing,
        "atlas_stats": handle_atlas_stats,
        "atlas_get": handle_atlas_get,
        "atlas_delete": handle_atlas_delete,
        "atlas_task_feedback": handle_atlas_task_feedback,
        "atlas_verify_chain": handle_atlas_verify_chain,
        "atlas_sleep_sync": handle_atlas_sleep_sync,
        "commit_observation": handle_commit_observation_uds,
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
        task_success = args.get("task_success", False)
        outcome = args.get("outcome")
        meta = args.get("metadata") or {}
        if isinstance(meta, dict):
            meta = dict(meta)
        else:
            meta = {}
        if task_success:
            meta["task_success"] = task_success
        if outcome:
            meta["outcome"] = outcome

        record = MemoryRecord(
            subject=args.get("subject", args.get("key", "")),
            predicate=args.get("predicate", "state_value"),
            object=str(args.get("object", args.get("value", ""))),
            confidence=float(args.get("confidence", 1.0)),
            is_state_variable=bool(args.get("is_state_variable", False)),
            metadata=meta,
        )
        await memory_engine.commit_observation(record)
        if hasattr(memory_engine, "record_mental_transition"):
            try:
                from atlas_memory.models import ActionPlan
                action = ActionPlan(name=f"observe:{record.predicate}", parameters={"subject": record.subject, "object": record.object})
                session_id = args.get("session_id", HERMES_DEFAULT_SESSION)
                memory_engine.record_mental_transition(action, session_id=session_id)
            except Exception as exc:
                logger.debug("Tools commit_observation record_mental_transition error: %s", exc)
        return {"status": "queued_for_validation", "subject": record.subject, "task_success": task_success, "outcome": outcome}

    return {
        "search_memory": handle_search_memory,
        "commit_observation": handle_commit_observation,
    }


def register_memory_tools(
    registry: Any,
    memory_engine: Any = None,
    socket_path: Optional[Path | str] = None,
) -> None:
    """Rejestruje pełny zestaw 10 narzędzi ATLAS w rejestrze narzędzi Hermesa przez UDS."""
    sock = socket_path or DEFAULT_SOCKET_PATH
    handlers = create_uds_tool_handlers(sock)
    is_async = False

    schema_map = {
        "atlas_recall": ATLAS_RECALL_SCHEMA,
        "atlas_remember": ATLAS_REMEMBER_SCHEMA,
        "atlas_what_if": ATLAS_WHAT_IF_SCHEMA,
        "atlas_active_sensing": ATLAS_ACTIVE_SENSING_SCHEMA,
        "atlas_stats": ATLAS_STATS_SCHEMA,
        "atlas_get": ATLAS_GET_SCHEMA,
        "atlas_delete": ATLAS_DELETE_SCHEMA,
        "atlas_task_feedback": ATLAS_TASK_FEEDBACK_SCHEMA,
        "atlas_verify_chain": ATLAS_VERIFY_CHAIN_SCHEMA,
        "atlas_sleep_sync": ATLAS_SLEEP_SYNC_SCHEMA,
    }

    if hasattr(registry, "register"):
        for tool_name, schema in schema_map.items():
            if tool_name in handlers:
                registry.register(
                    name=tool_name,
                    toolset="memory",
                    schema=schema,
                    is_async=is_async,
                )(handlers[tool_name])


def register_hermes_memory_tools(registry: Any, memory_engine: Any) -> None:
    """Compatibility registry function."""
    register_memory_tools(registry, memory_engine=memory_engine)
    handlers = create_hermes_tool_handlers(memory_engine)
    if hasattr(registry, "register"):
        registry.register(name="search_memory", toolset="memory", schema=SEARCH_MEMORY_SCHEMA, is_async=True)(handlers["search_memory"])
        registry.register(name="commit_observation", toolset="memory", schema=COMMIT_OBSERVATION_SCHEMA, is_async=True)(handlers["commit_observation"])


