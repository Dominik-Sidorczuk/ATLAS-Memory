from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import numpy as np

from atlas_memory.causal.models import (
    CausalEdge,
    CausalPath,
    CPoFNode,
    DiffusionNode,
    DiffusionResult,
    WhatIfResult,
)
from atlas_memory.models import ActionPlan


class RetroCausalEngine:
    """
    Faza B: Retro-Causal Edge (Ekstrapolacja Przyczynowo-Skutkowa & JEPA).
    
    Umożliwia symulację myślową 'Co się stanie jeśli...' (causal_what_if):
    1. Ekstrakcja podgrafu 1-2 hop z Kùzu / NetworkX.
    2. Propagacja pewności wzdłuż krawędzi przyczynowo-skutkowych.
    3. Fuzja z predykcją dynamiki świata P(s_{t+1}|s_t, a_t) z bufora L1 JEPA.
    4. Analiza dyfuzji przyczynowej i wykrywanie CPoF (Critical Points of Failure).
    """

    CRITICAL_PREDICATES = {
        "depends_on", "relies_on", "hosted_on", "hosts", "powers", "routes_to",
        "critical_for", "requires", "authenticates", "mounts", "databases",
    }

    def __init__(self, graph_client: Any, latent_buffer: Optional[Any] = None):
        self.graph = graph_client
        self.latent = latent_buffer

    async def causal_diffusion_analysis(
        self,
        start_node: str,
        max_depth: int = 3,
        decay_factor: float = 0.7,
    ) -> DiffusionResult:
        """
        Probabilistyczne rozchodzenie fali skutków przez graf zależności.
        Dla każdego węzła na ścieżce: impact = confidence * (decay_factor ** depth).
        Obsługuje cykle (visited set per path + max_depth).
        Gdy węzeł jest osiągany z wielu ścieżek, sumuje wpływ.
        """
        subgraph_data = await self._fetch_subgraph(start_node, max_depth=max_depth)
        relations = subgraph_data.get("relations", [])

        if not relations:
            return DiffusionResult(
                nodes=[],
                edges=[],
                total_impact=0.0,
                max_depth_reached=0,
            )

        adj: Dict[str, List[Dict[str, Any]]] = {}
        for r in relations:
            u = r.get("subject", "")
            v = str(r.get("object", ""))
            pred = r.get("predicate", "")
            conf = float(r.get("confidence", 1.0))
            if u not in adj:
                adj[u] = []
            adj[u].append({"target": v, "predicate": pred, "confidence": conf})

        # node_id -> {impact_score, min_path_length, max_confidence}
        node_stats: Dict[str, Dict[str, Any]] = {}
        collected_edges: List[CausalEdge] = []
        seen_edges: Set[tuple] = set()
        max_reached_depth = 0

        def traverse(current_node: str, current_conf: float, current_depth: int, path_visited: Set[str]):
            nonlocal max_reached_depth
            if current_depth > max_depth:
                return

            neighbors = adj.get(current_node, [])
            for edge in neighbors:
                tgt = edge["target"]
                pred = edge["predicate"]
                conf = edge["confidence"]

                if tgt in path_visited:
                    continue  # Unikaj cykli na bieżącej ścieżce

                edge_key = (current_node, pred, tgt)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    collected_edges.append(CausalEdge(
                        source=current_node,
                        predicate=pred,
                        target=tgt,
                        confidence=conf,
                        impact_type="critical_dependency" if pred in self.CRITICAL_PREDICATES else "general_dependency",
                    ))

                hop_depth = current_depth
                if hop_depth > max_reached_depth:
                    max_reached_depth = hop_depth

                cum_conf = current_conf * conf
                step_impact = cum_conf * (decay_factor ** hop_depth)

                if tgt not in node_stats:
                    node_stats[tgt] = {
                        "impact_score": 0.0,
                        "path_length": hop_depth,
                        "cumulative_confidence": cum_conf,
                    }
                else:
                    node_stats[tgt]["path_length"] = min(node_stats[tgt]["path_length"], hop_depth)
                    node_stats[tgt]["cumulative_confidence"] = max(node_stats[tgt]["cumulative_confidence"], cum_conf)

                node_stats[tgt]["impact_score"] += step_impact

                traverse(tgt, cum_conf, current_depth + 1, path_visited | {tgt})

        traverse(start_node, 1.0, 1, {start_node})

        diffusion_nodes = [
            DiffusionNode(
                node_id=nid,
                impact_score=round(stats["impact_score"], 6),
                path_length=stats["path_length"],
                cumulative_confidence=round(stats["cumulative_confidence"], 6),
            )
            for nid, stats in node_stats.items()
        ]

        total_impact = round(sum(n.impact_score for n in diffusion_nodes), 6)

        return DiffusionResult(
            nodes=diffusion_nodes,
            edges=collected_edges,
            total_impact=total_impact,
            max_depth_reached=max_reached_depth,
        )

    async def detect_cpof(self, node_id: str) -> List[CPoFNode]:
        """
        CPoF = Critical Point of Failure — węzeł, którego usunięcie rozspójnia >50% ścieżek/zależnych węzłów.
        Dla każdego węzła w podgrafie osiągalnym z node_id oblicza reachability impact.
        severity = len(affected_nodes) / total_reachable_nodes (0-1).
        """
        subgraph_data = await self._fetch_subgraph(node_id, max_depth=10)
        relations = subgraph_data.get("relations", [])

        if not relations:
            return []

        adj: Dict[str, List[str]] = {}
        for r in relations:
            u = r.get("subject", "")
            v = str(r.get("object", ""))
            if u not in adj:
                adj[u] = []
            adj[u].append(v)

        def get_reachable(root: str, excluded_node: Optional[str] = None) -> Set[str]:
            reachable = set()
            stack = [root]
            while stack:
                curr = stack.pop()
                for neighbor in adj.get(curr, []):
                    if neighbor == excluded_node or neighbor in reachable:
                        continue
                    reachable.add(neighbor)
                    stack.append(neighbor)
            return reachable

        all_reachable = get_reachable(node_id)
        total_reachable = len(all_reachable)
        if total_reachable == 0:
            return []

        cpof_nodes: List[CPoFNode] = []

        for candidate in all_reachable:
            reachable_without_c = get_reachable(node_id, excluded_node=candidate)
            # Węzły dotknięte to te z all_reachable (poza samym candidate), które tracą osiągalność
            affected = (all_reachable - {candidate}) - reachable_without_c
            severity = len(affected) / total_reachable

            if severity >= 0.5:
                cpof_nodes.append(CPoFNode(
                    node_id=candidate,
                    severity=round(severity, 4),
                    affected_nodes=sorted(list(affected)),
                    description=f"Węzeł '{candidate}' jest krytycznym punktem awarii: odcina {len(affected)}/{total_reachable} ({severity*100:.1f}%) zależnych węzłów.",
                ))

        cpof_nodes.sort(key=lambda x: x.severity, reverse=True)
        return cpof_nodes

    async def multi_hop_cypher_query(
        self,
        source: str,
        target: str,
        max_hops: int = 5,
        exclude_cycles: bool = True,
    ) -> List[CausalPath]:
        """
        Wykrywa wieloskokowe ścieżki przyczynowe między source a target przy użyciu zapytań grafowych.
        """
        raw_paths = []
        if hasattr(self.graph, "execute_multi_hop_cypher"):
            res = self.graph.execute_multi_hop_cypher(source, target, max_hops=max_hops)
            if hasattr(res, "__await__"):
                raw_paths = await res
            else:
                raw_paths = res
        elif isinstance(self.graph, dict):
            # Fallback dla mocków dict
            def find_paths(curr: str, tgt: str, current_path: List[Dict[str, Any]], visited: Set[str], hops: int):
                if hops > max_hops:
                    return
                for edge in self.graph.get(curr, []):
                    nxt = edge["target"] if isinstance(edge, dict) else edge
                    pred = edge.get("predicate", "leads_to") if isinstance(edge, dict) else "leads_to"
                    conf = edge.get("confidence", 1.0) if isinstance(edge, dict) else 1.0
                    edge_dict = {"source": curr, "predicate": pred, "target": nxt, "confidence": conf}
                    if nxt == tgt:
                        raw_paths.append(current_path + [edge_dict])
                    elif nxt not in visited or not exclude_cycles:
                        find_paths(nxt, tgt, current_path + [edge_dict], visited | {nxt}, hops + 1)
            find_paths(source, target, [], {source}, 1)

        causal_paths: List[CausalPath] = []
        for raw in raw_paths:
            if isinstance(raw, dict) and "edges" in raw:
                edges_data = raw["edges"]
            elif isinstance(raw, list):
                edges_data = raw
            else:
                continue

            steps: List[CausalEdge] = []
            nodes_in_path = [source]
            cum_conf = 1.0
            has_cycle = False

            for ed in edges_data:
                s = ed.get("source", ed.get("from", ""))
                t = ed.get("target", ed.get("to", ""))
                p = ed.get("predicate", "leads_to")
                c = float(ed.get("confidence", 1.0))

                if t in nodes_in_path:
                    has_cycle = True
                nodes_in_path.append(t)
                cum_conf *= c

                steps.append(CausalEdge(
                    source=s,
                    predicate=p,
                    target=t,
                    confidence=c,
                    impact_type="critical_dependency" if p in self.CRITICAL_PREDICATES else "general_dependency",
                ))

            if exclude_cycles and has_cycle:
                continue

            if not steps or steps[-1].target != target:
                continue

            cum_conf = round(cum_conf, 4)
            causal_paths.append(CausalPath(
                source_entity=source,
                simulated_action="multi_hop_query",
                steps=steps,
                affected_target=target,
                cumulative_confidence=cum_conf,
                risk_level="HIGH" if cum_conf >= 0.7 else "MODERATE",
                description=f"Multi-hop causal path from '{source}' to '{target}' ({len(steps)} hops)",
            ))

        causal_paths.sort(key=lambda p: p.cumulative_confidence, reverse=True)
        return causal_paths

    async def causal_what_if(
        self,
        entity: str,
        action: str,
        depth: int = 2,
    ) -> List[CausalPath]:
        """
        Symuluje skutki wykonania akcji na danej encji:
        Wyszukuje wszystkie ścieżki 1- i 2-hop w grafie zależności i propaguje confidence.
        """
        # 1. Pobranie podgrafu z Kùzu / grafu wiedzy
        subgraph_data = await self._fetch_subgraph(entity, max_depth=depth)
        relations = subgraph_data.get("relations", [])

        if not relations:
            return []

        # 2. Budowa lokalnego grafu skierowanego do wyszukiwania ścieżek
        adj: Dict[str, List[Dict[str, Any]]] = {}
        for r in relations:
            u = r.get("subject", "")
            v = str(r.get("object", ""))
            pred = r.get("predicate", "")
            conf = float(r.get("confidence", 1.0))
            if u not in adj:
                adj[u] = []
            adj[u].append({"target": v, "predicate": pred, "confidence": conf})

        # 3. DFS / BFS szukanie ścieżek o długości do `depth`
        paths: List[CausalPath] = []
        visited_nodes: Set[str] = {entity}

        # Opcjonalna ekstrapolacja z L1 JEPA
        jepa_divergence = None
        if self.latent is not None and hasattr(self.latent, "predict_transition"):
            try:
                curr_st = self.latent.current_state
                trans = self.latent.predict_transition(
                    curr_st,
                    ActionPlan(name=action, parameters={"target": entity}),
                )
                v_curr = np.array(curr_st.vector, dtype=np.float64)
                v_next = np.array(trans.predicted_state.vector, dtype=np.float64)
                jepa_divergence = float(np.linalg.norm(v_next - v_curr))
            except Exception:
                jepa_divergence = None

        def dfs_traverse(current_node: str, current_steps: List[CausalEdge], current_conf: float, current_depth: int):
            if current_depth > depth:
                return

            neighbors = adj.get(current_node, [])
            for edge_info in neighbors:
                tgt = edge_info["target"]
                pred = edge_info["predicate"]
                conf = edge_info["confidence"]

                if tgt in visited_nodes and tgt != current_node:
                    continue  # Unikanie cykli

                step = CausalEdge(
                    source=current_node,
                    predicate=pred,
                    target=tgt,
                    confidence=conf,
                    impact_type="critical_dependency" if pred in self.CRITICAL_PREDICATES else "general_dependency",
                )

                new_steps = current_steps + [step]
                new_conf = round(current_conf * conf, 4)

                # Ocena poziomu ryzyka ścieżki
                has_critical = any(s.predicate in self.CRITICAL_PREDICATES for s in new_steps)
                if has_critical and new_conf >= 0.7:
                    risk = "CRITICAL" if len(new_steps) == 1 else "HIGH"
                elif has_critical or new_conf >= 0.5:
                    risk = "MODERATE"
                else:
                    risk = "LOW"

                desc = f"Akcja '{action}' na '{entity}' wpływa na '{tgt}' przez relację '{pred}' (ufność: {new_conf:.2f})"

                paths.append(CausalPath(
                    source_entity=entity,
                    simulated_action=action,
                    steps=new_steps,
                    affected_target=tgt,
                    cumulative_confidence=new_conf,
                    risk_level=risk,
                    jepa_latent_divergence=jepa_divergence,
                    description=desc,
                ))

                visited_nodes.add(tgt)
                dfs_traverse(tgt, new_steps, new_conf, current_depth + 1)
                visited_nodes.remove(tgt)

        dfs_traverse(entity, [], 1.0, 1)

        # Sortowanie według ryzyka i pewności
        paths.sort(key=lambda p: (p.risk_level == "CRITICAL", p.risk_level == "HIGH", p.cumulative_confidence), reverse=True)
        return paths

    async def evaluate_what_if(
        self,
        entity: str,
        action: str,
        depth: int = 2,
    ) -> WhatIfResult:
        """Kompleksowa ocena bezpieczeństwa i skutków akcji."""
        paths = await self.causal_what_if(entity, action, depth=depth)

        highest_risk = None
        safety_score = 1.0

        if paths:
            highest_risk = paths[0].affected_target
            # Obliczenie współczynnika bezpieczeństwa (im więcej ścieżek krytycznych, tym niższe safety)
            risk_penalty = sum(
                0.3 if p.risk_level == "CRITICAL" else 0.15 if p.risk_level == "HIGH" else 0.05
                for p in paths
            )
            safety_score = max(0.0, round(1.0 - min(1.0, risk_penalty), 2))

        return WhatIfResult(
            source_entity=entity,
            simulated_action=action,
            paths=paths,
            highest_risk_target=highest_risk,
            overall_safety_score=safety_score,
            jepa_extrapolation_used=self.latent is not None,
        )

    async def _fetch_subgraph(self, entity: str, max_depth: int = 2) -> Dict[str, Any]:
        """Pobiera podgraf z obiektu grafu."""
        if hasattr(self.graph, "get_subgraph_relations"):
            return await self.graph.get_subgraph_relations([entity], max_depth=max_depth)
        elif hasattr(self.graph, "get_subgraph_for_entities"):
            return await self.graph.get_subgraph_for_entities([entity])
        return {"relations": []}
