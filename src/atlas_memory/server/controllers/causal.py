"""Causal Graph, Counterfactual Query & Annealing RPC Controller for AtlasDaemon.

Encapsulates Kùzu graph hydration, what_if counterfactual pathfinding against
RAM (NetworkX) and disk (Kùzu), relation ingestion, and causal graph self-annealing.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from atlas_memory.causal.models import CausalEdge, CausalPath
from atlas_memory.core.agi_logger import log_causal

logger = logging.getLogger("atlas.server.controllers.causal")


class CausalRpcControllerMixin:
    """Mixin providing causal graph queries, relation ingestion, and annealing RPC handlers."""

    def _what_if_paths_from_disk(
        self,
        graph_obj: Any,
        entity: str,
        action: Any,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """V72.2 (W2): buduje ścieżki ``what_if`` z RZECZYWISTYCH krawędzi Kùzu (dysk)."""
        getter = getattr(graph_obj, "get_relations_on_disk", None)
        if not callable(getter):
            return []
        try:
            relations = getter(entity, limit=limit) or []
        except Exception as exc:
            logger.debug("what_if: odczyt krawędzi z Kùzu nie powiódł się dla '%s': %s", entity, exc)
            return []
        if not relations:
            return []

        action_label = str(action) if action else "what_if"
        paths: List[Dict[str, Any]] = []
        for rel in relations:
            try:
                conf = float(rel.get("confidence", 1.0))
            except (TypeError, ValueError):
                conf = 1.0
            conf = max(0.0, min(1.0, conf))
            try:
                edge = CausalEdge(
                    source=str(rel["subject"]),
                    predicate=str(rel["predicate"]),
                    target=str(rel["object"]),
                    confidence=conf,
                    impact_type="disk_relation",
                )
                path = CausalPath(
                    source_entity=str(rel["subject"]),
                    simulated_action=action_label,
                    steps=[edge],
                    affected_target=str(rel["object"]),
                    cumulative_confidence=conf,
                    risk_level="MODERATE" if conf < 0.7 else "HIGH",
                    description=(
                        f"Ścieżka z Kùzu (dysk): {rel['subject']} -[{rel['predicate']}]-> {rel['object']} "
                        f"(ufność {conf:.2f})"
                    ),
                )
            except Exception as exc:
                logger.debug("what_if: krawędź z Kùzu odrzucona (%s): %s", rel, exc)
                continue
            paths.append(path.model_dump())
        return paths

    def _disk_what_if_response(
        self,
        graph_obj: Any,
        entity: str,
        action: Any,
        entity_in_ram: bool,
        entity_on_disk: bool,
    ) -> Optional[Dict[str, Any]]:
        """V72.2 (W2): odpowiedź ``what_if`` z prawdziwych krawędzi Kùzu — TYLKO gdy encji nie ma w RAM."""
        if not (entity_on_disk and not entity_in_ram):
            return None
        disk_paths = self._what_if_paths_from_disk(graph_obj, entity, action)
        if not disk_paths:
            return None
        top_out = disk_paths[0].get("affected_target") if disk_paths else None
        log_causal(logger, entity=entity, action=str(action), engine="KùzuDisk", paths_count=len(disk_paths), top_outcome=top_out)
        return {
            "status": "ok",
            "action": action,
            "entity": entity,
            "paths_count": len(disk_paths),
            "causal_paths": disk_paths,
            "causal_path": disk_paths[0],
            "message": (
                f"Encja '{entity}' istnieje w Kùzu (dysk), nie ma jej w RAM — "
                f"ścieżki pobrane bezpośrednio z magazynu dyskowego."
            ),
            "source": "kuzu_disk",
        }

    async def _handle_what_if(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """V124 (F-039): Counterfactual causal `what_if` query (RAM engine first, disk fallback)."""
        action = str(params.get("action") or "").strip()
        entity = str(params.get("entity") or params.get("target") or action).strip()
        if not action and not entity:
            return {
                "status": "error",
                "message": "Parameters 'entity' and/or 'action' are required for what_if",
                "causal_paths": [],
                "causal_path": None,
                "paths_count": 0,
            }
        if not action:
            action = "inspect"
        depth = int(params.get("depth", 2))

        # 1. RAM Causal Engine FIRST: consult live causal_engine if available
        engine = getattr(self, "causal_engine", None)
        if engine is not None and callable(getattr(engine, "causal_what_if", None)):
            try:
                paths = await engine.causal_what_if(entity, action, depth=depth)
                paths_data = [p.model_dump() if hasattr(p, "model_dump") else p for p in (paths or [])]
                if paths_data:
                    top_out = paths_data[0].get("affected_target") if paths_data else None
                    log_causal(logger, entity=entity, action=str(action), engine="RAM", paths_count=len(paths_data), top_outcome=top_out)
                    return {
                        "status": "ok",
                        "engine": "ram",
                        "action": action,
                        "entity": entity,
                        "paths_count": len(paths_data),
                        "causal_paths": paths_data,
                        "causal_path": paths_data[0],
                        "message": None,
                    }
            except Exception as exc:
                logger.warning("[RPC] what_if: silnik RAM zawiódł (%s) — fallback do ścieżki dyskowej", exc)

        # 2. Fallback to graph / disk (Kùzu)
        graph_obj = getattr(self, "graph_client", None) or (getattr(engine, "graph", None) if engine else None)
        entity_in_ram = False
        entity_on_disk = False
        if graph_obj is not None:
            if hasattr(graph_obj, "nx_graph") and graph_obj.nx_graph is not None:
                if entity in graph_obj.nx_graph:
                    entity_in_ram = True
                else:
                    ent_clean = str(entity).strip().lower().replace("_", " ")
                    for node in graph_obj.nx_graph.nodes:
                        node_str = str(node).strip().lower()
                        if node_str == ent_clean or (len(ent_clean) >= 5 and ent_clean in node_str):
                            entity_in_ram = True
                            break
            elif isinstance(graph_obj, dict):
                entity_in_ram = entity in graph_obj

            disk_check = getattr(graph_obj, "entity_exists", None)
            if callable(disk_check):
                try:
                    entity_on_disk = bool(disk_check(entity))
                except Exception as exc:
                    logger.debug("what_if: bramka Kùzu niedostępna dla '%s': %s", entity, exc)
                    entity_on_disk = False

        entity_exists = entity_in_ram or entity_on_disk

        if not entity_exists:
            log_causal(logger, entity=entity, action=str(action), engine="None", paths_count=0, top_outcome="not_found")
            return {
                "status": "entity_not_found",
                "action": action,
                "entity": entity,
                "paths_count": 0,
                "causal_paths": [],
                "causal_path": None,
                "message": f"Entity '{entity}' not found in causal knowledge graph. No causal paths can be derived.",
            }

        # Check disk paths via _disk_what_if_response (covers disk-only entities)
        disk_response = self._disk_what_if_response(graph_obj, entity, action, entity_in_ram, entity_on_disk)
        if disk_response is not None:
            return disk_response

        # If entity exists on disk/ram, attempt direct disk paths fetch if available
        disk_paths = self._what_if_paths_from_disk(graph_obj, entity, action)
        if disk_paths:
            return {
                "status": "ok",
                "action": action,
                "entity": entity,
                "paths_count": len(disk_paths),
                "causal_paths": disk_paths,
                "causal_path": disk_paths[0],
                "message": (
                    f"Encja '{entity}' istnieje w Kùzu (dysk) — "
                    f"ścieżki pobrane bezpośrednio z magazynu dyskowego."
                ),
                "source": "kuzu_disk",
            }

        summary_msg = f"Entity '{entity}' exists in graph, but no causal failure paths detected for action '{action}' at depth {depth}."
        return {
            "status": "ok",
            "action": action,
            "entity": entity,
            "paths_count": 0,
            "causal_paths": [],
            "causal_path": None,
            "message": summary_msg,
        }

    async def _hydrate_graph_from_disk(self) -> Dict[str, Any]:
        """V72.2 (W1): ładuje graf z pliku Kùzu do pamięci RAM daemona przed obsługą zapytań."""
        graph = getattr(self, "graph_client", None) or (
            getattr(getattr(self, "engine", None), "graph", None) if getattr(self, "engine", None) is not None else None
        )
        hydrate = getattr(graph, "hydrate_from_disk", None)
        if graph is None or not callable(hydrate):
            logger.debug("Hydracja RAM pominięta: graf bez metody hydrate_from_disk")
            return {"skipped": True}
        pre_rep = getattr(graph, "_pre_hydrated_report", None)
        if isinstance(pre_rep, dict) and pre_rep.get("loaded_nodes", 0) > 0:
            self._graph_hydration = pre_rep
            delattr(graph, "_pre_hydrated_report")
            logger.info("V72.2: hydracja grafu RAM z Kùzu zakończona (pre-ingest): %s", self._graph_hydration)
            return self._graph_hydration
        try:
            report = await hydrate()
        except Exception as exc:
            logger.warning("Hydracja RAM z Kùzu nie powiodła się (daemon startuje bez pełnego grafu): %s", exc)
            return {"error": f"{type(exc).__name__}: {exc}"[:160]}
        self._graph_hydration = report if isinstance(report, dict) else {"raw": report}
        logger.info("V72.2: hydracja grafu RAM z Kùzu zakończona: %s", self._graph_hydration)
        return self._graph_hydration

    async def _handle_anneal(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger autonomous causal graph annealer."""
        target_entity = params.get("target_entity") or params.get("entity", "Root")
        causal = getattr(self, "causal_engine", None)
        if causal is not None and hasattr(causal, "recalibrate_graph_with_annealer"):
            res = await causal.recalibrate_graph_with_annealer(target_entity=target_entity)
            logger.info("[COG:CAUSAL] anneal target='%s' -> %s", target_entity, res)
            return res
        return {"status": "error", "error": "no_causal_engine_or_annealer"}


__all__ = [
    "CausalRpcControllerMixin",
]
