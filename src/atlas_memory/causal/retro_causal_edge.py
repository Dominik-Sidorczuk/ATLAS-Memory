from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Optional, Set

import numpy as np

logger = logging.getLogger("atlas.causal.retro_causal_edge")

from atlas_memory.causal.models import (
    CausalEdge,
    CausalPath,
    CPoFNode,
    DiffusionNode,
    DiffusionResult,
)
from atlas_memory.core.models import EMAIL_PATTERN, ActionPlan, is_pii_entity


def _load_profile_critical_keywords() -> Set[str]:
    try:
        from atlas_memory.domain.registry import DomainProfileRegistry
        kw = DomainProfileRegistry.get_instance().get_all_critical_keywords()
        if kw:
            return kw
    except Exception as exc:
        logger.debug("Could not load critical keywords from DomainProfileRegistry: %s", exc)
    return {
        "crash", "failure", "fatal", "broken", "corrupt", "leak", "deadlock",
        "hazard", "toxic", "toxicity", "diminishing_returns", "danger",
        "dose", "mg", "per_dose", "overdose", "dosage", "tumor", "hepatotoxic",
        "lethal", "poison", "exceed", "limit", "truncate", "truncate_table",
        "drop_table", "delete_all", "database", "flush_all", "destroy",
    }


class RetroCausalEngine:
    """
    Faza B: Retro-Causal Edge (Ekstrapolacja Przyczynowo-Skutkowa & JEPA).
    
    Umożliwia symulację myślową 'Co się stanie jeśli...' (causal_what_if):
    1. Ekstrakcja podgrafu 1-2 hop z Kùzu / NetworkX.
    2. Propagacja pewności wzdłuż krawędzi przyczynowo-skutkowych.
    3. Fuzja z predykcją dynamiki świata P(s_{t+1}|s_t, a_t) z bufora L1 JEPA.
    4. Analiza dyfuzji przyczynowej i wykrywanie CPoF (Critical Points of Failure).
    """

    DEFAULT_CRITICAL_PREDICATES = {
        "depends_on", "relies_on", "hosted_on", "hosts", "powers", "routes_to",
        "critical_for", "requires", "authenticates", "mounts", "databases",
        "causes", "breaks", "blocks", "restricted_by", "requires_condition",
    }
    DEFAULT_CRITICAL_KEYWORDS = _load_profile_critical_keywords()

    # Canonical class-level attribute references
    CRITICAL_PREDICATES = DEFAULT_CRITICAL_PREDICATES
    CRITICAL_KEYWORDS = DEFAULT_CRITICAL_KEYWORDS

    def __init__(
        self,
        graph_client: Any,
        latent_buffer: Optional[Any] = None,
        critical_predicates: Optional[Set[str]] = None,
        critical_keywords: Optional[Set[str]] = None,
    ):
        self.graph = graph_client
        self.latent = latent_buffer
        self.critical_predicates = set(critical_predicates) if critical_predicates is not None else set(self.DEFAULT_CRITICAL_PREDICATES)
        self.critical_keywords = set(critical_keywords) if critical_keywords is not None else set(self.DEFAULT_CRITICAL_KEYWORDS)

    @staticmethod
    def _build_adjacency(relations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Buduje listę sąsiedztwa grafu skierowanego ze zbioru relacji."""
        adj: Dict[str, List[Dict[str, Any]]] = {}
        for r in relations:
            pred = r.get("predicate", "")
            if pred in ("stated_memory", "raw_content", "documentation", "source_doc"):
                continue
            u = r.get("subject", "")
            v = str(r.get("object", ""))
            if is_pii_entity(u) or is_pii_entity(v):
                continue
            conf = float(r.get("confidence", 1.0))
            if u not in adj:
                adj[u] = []
            adj[u].append({"target": v, "predicate": pred, "confidence": conf})
        return adj

    @staticmethod
    def _build_diffusion_nodes(node_stats: Dict[str, Dict[str, Any]]) -> List[DiffusionNode]:
        """Konwertuje zagregowane statystyki fali dyfuzji do listy obiektów DiffusionNode."""
        return [
            DiffusionNode(
                node_id=nid,
                impact_score=round(stats["impact_score"], 6),
                path_length=stats["path_length"],
                cumulative_confidence=min(1.0, max(0.0, round(stats["cumulative_confidence"], 6))),
            )
            for nid, stats in node_stats.items()
        ]

    def _compute_jepa_divergence(self, action: str, entity: str) -> Optional[float]:
        """Oblicza dywergencję przestrzeni ukrytej L1 JEPA po wykonaniu akcji."""
        if self.latent is None or not hasattr(self.latent, "predict_transition"):
            return None
        try:
            curr_st = self.latent.current_state
            trans = self.latent.predict_transition(
                curr_st,
                ActionPlan(name=action, parameters={"target": entity}),
            )
            v_curr = np.array(curr_st.vector, dtype=np.float64)
            v_next = np.array(trans.predicted_state.vector, dtype=np.float64)
            return float(np.linalg.norm(v_next - v_curr))
        except Exception:
            return None

    def _assess_path_risk(
        self,
        action: str,
        steps: List[CausalEdge],
        target: str,
        conf: float,
        jepa_divergence: Optional[float] = None,
        cpof_map: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Ocenia poziom ryzyka ścieżki przyczynowej na podstawie ciągłego tensora JEPA-CPoF (V154 Filar II).
        
        risk_potential = delta_jepa * sum(step.confidence * cpof_severity(step.target))
        Progi:
        >= 0.70 -> CRITICAL
        >= 0.40 -> HIGH
        >= 0.15 -> MODERATE
        inaczej -> LOW
        """
        has_critical_pred = any(s.predicate in self.critical_predicates for s in steps)
        cpof_dict = cpof_map or {}

        # 1. Obliczenie wag dotkniętych węzłów i relacji
        accum_step_risk = 0.0
        for s in steps:
            node_severity = cpof_dict.get(s.target, 0.5 if s.predicate in self.critical_predicates else 0.2)
            accum_step_risk += s.confidence * node_severity

        # Bazowa dywergencja JEPA (domyślnie 1.0 jeśli brak modułu L1 JEPA)
        jepa_factor = float(jepa_divergence) if jepa_divergence is not None else 1.0

        # Wycena semantyczna akcji/celu (Zero-Shot Semantic Risk Assessor)
        semantic_multiplier = 1.0
        try:
            from atlas_memory.causal.risk_assessor import SemanticRiskAssessor
            assessor = SemanticRiskAssessor.get_instance()
            lvl, a_conf = assessor.assess_action_risk(action)
            if assessor.is_critical(action) or assessor.is_critical(target):
                semantic_multiplier = 1.5
            elif assessor.is_volatile(action):
                semantic_multiplier = 1.2
            elif lvl == "SAFE" and a_conf >= 0.5:
                semantic_multiplier = 0.1
        except Exception as exc:
            logger.debug("SemanticRiskAssessor check failed: %s", exc)

        norm_step_risk = accum_step_risk / max(1.0, float(len(steps)))
        risk_potential = (jepa_factor * norm_step_risk * conf * semantic_multiplier)

        if has_critical_pred and risk_potential < 0.40 and conf >= 0.70 and semantic_multiplier > 0.5:
            risk_potential = max(risk_potential, 0.72)
        elif has_critical_pred and risk_potential < 0.20 and conf >= 0.40 and semantic_multiplier > 0.5:
            risk_potential = max(risk_potential, 0.45)

        if risk_potential >= 0.70:
            return "CRITICAL"
        if risk_potential >= 0.40:
            return "HIGH"
        if risk_potential >= 0.15:
            return "MODERATE"
        return "LOW"

    async def causal_diffusion_analysis(
        self,
        start_node: str,
        max_depth: int = 3,
        decay_factor: float = 0.7,
    ) -> DiffusionResult:
        """
        Probabilistyczne rozchodzenie fali skutków przez graf zależności.
        Dla każdego węzła na ścieżce: impact = confidence * (decay_factor ** depth).
        Zastąpiono wykładniczą rekurencję przeszukiwaniem warstwowym (BFS)
        z dynamiczną agregacją wpływu O(d * (|V| + |E|)).
        """
        safe_depth = max(1, int(max_depth))
        safe_decay = max(0.0, min(1.0, float(decay_factor)))  # clamp01 scope: nie-confidence (waga/penalty)

        subgraph_data = await self._fetch_subgraph(start_node, max_depth=safe_depth)
        relations = subgraph_data.get("relations", [])

        if not relations:
            return DiffusionResult(
                nodes=[],
                edges=[],
                total_impact=0.0,
                max_depth_reached=0,
            )

        adj = self._build_adjacency(relations)

        # node_id -> {impact_score, min_path_length, max_confidence}
        node_stats: Dict[str, Dict[str, Any]] = {}
        collected_edges: List[CausalEdge] = []
        seen_edges: Set[tuple] = set()
        max_reached_depth = 0

        # Przeszukiwanie warstwowe z dynamiczną agregacją wpływu
        current_layer: Dict[str, float] = {start_node: 1.0}

        for current_depth in range(1, safe_depth + 1):
            next_layer: Dict[str, float] = {}
            for u, u_conf in current_layer.items():
                for edge in adj.get(u, []):
                    tgt = edge["target"]
                    pred = edge["predicate"]
                    conf = edge["confidence"]

                    if tgt == start_node or tgt == u:
                        continue

                    edge_key = (u, pred, tgt)
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        collected_edges.append(CausalEdge(
                            source=u,
                            predicate=pred,
                            target=tgt,
                            confidence=conf,
                            impact_type="critical_dependency" if pred in self.critical_predicates else "general_dependency",
                        ))

                    max_reached_depth = max(max_reached_depth, current_depth)
                    trans_conf = u_conf * conf
                    step_impact = trans_conf * (safe_decay ** current_depth)

                    if tgt not in node_stats:
                        node_stats[tgt] = {
                            "impact_score": 0.0,
                            "path_length": current_depth,
                            "cumulative_confidence": min(1.0, trans_conf),
                        }
                    else:
                        node_stats[tgt]["path_length"] = min(node_stats[tgt]["path_length"], current_depth)
                        node_stats[tgt]["cumulative_confidence"] = min(1.0, max(node_stats[tgt]["cumulative_confidence"], trans_conf))

                    node_stats[tgt]["impact_score"] += step_impact
                    next_layer[tgt] = min(1.0, next_layer.get(tgt, 0.0) + trans_conf)

            if not next_layer:
                break
            current_layer = next_layer

        diffusion_nodes = self._build_diffusion_nodes(node_stats)
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
            # Węzły dotknięte to te z all_reachable, które tracą osiągalność przy awarii candidate (w tym sam candidate)
            affected = all_reachable - reachable_without_c
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
        adj = self._build_adjacency(relations)

        # 3. DFS / BFS szukanie ścieżek o długości do `depth` z korzeni dopasowanych
        paths: List[CausalPath] = []
        roots = subgraph_data.get("active_roots", [entity])
        if not roots:
            roots = [entity]

        # Opcjonalna ekstrapolacja z L1 JEPA
        jepa_divergence = self._compute_jepa_divergence(action, entity)
        cpof_list = await self.detect_cpof(entity)
        cpof_map = {n.node_id: n.severity for n in cpof_list}

        seen_path_signatures: Set[str] = set()

        def dfs_traverse(root_node: str, current_node: str, current_steps: List[CausalEdge], current_conf: float, current_depth: int, visited: Set[str]):
            if current_depth > depth:
                return

            neighbors = adj.get(current_node, [])
            for edge_info in neighbors:
                tgt = edge_info["target"]
                if is_pii_entity(tgt):
                    continue
                pred = edge_info["predicate"]
                conf = edge_info["confidence"]

                if tgt in visited:
                    continue  # Unikanie cykli

                step = CausalEdge(
                    source=current_node,
                    predicate=pred,
                    target=tgt,
                    confidence=conf,
                    impact_type="critical_dependency" if pred in self.critical_predicates else "general_dependency",
                )

                new_steps = current_steps + [step]
                new_conf = round(current_conf * conf, 4)

                # Ocena poziomu ryzyka ścieżki (V154 Filar II: JEPA-CPoF)
                risk = self._assess_path_risk(
                    action,
                    new_steps,
                    tgt,
                    new_conf,
                    jepa_divergence=jepa_divergence,
                    cpof_map=cpof_map,
                )

                desc = f"Akcja '{action}' na '{entity}' wpływa na '{tgt}' przez relację '{pred}' (ufność: {new_conf:.2f})"
                desc = EMAIL_PATTERN.sub("[REDACTED_PII]", desc)
                sig = "->".join(f"{s.source}-[{s.predicate}]->{s.target}" for s in new_steps)

                if sig not in seen_path_signatures:
                    seen_path_signatures.add(sig)
                    paths.append(CausalPath(
                        source_entity=root_node,
                        simulated_action=action,
                        steps=new_steps,
                        affected_target=tgt,
                        cumulative_confidence=new_conf,
                        risk_level=risk,
                        jepa_latent_divergence=jepa_divergence,
                        description=desc,
                    ))

                visited.add(tgt)
                dfs_traverse(root_node, tgt, new_steps, new_conf, current_depth + 1, visited)
                visited.remove(tgt)

        for r_node in roots:
            dfs_traverse(r_node, r_node, [], 1.0, 1, {r_node})

        # Sortowanie według ryzyka i pewności
        paths.sort(key=lambda p: (p.risk_level == "CRITICAL", p.risk_level == "HIGH", p.cumulative_confidence), reverse=True)
        return paths


    async def _fetch_subgraph(self, entity: str, max_depth: int = 2) -> Dict[str, Any]:
        """Pobiera podgraf z obiektu grafu."""
        if inspect.iscoroutinefunction(getattr(self.graph, "get_subgraph_relations", None)):
            res = await self.graph.get_subgraph_relations([entity], max_depth=max_depth)
            return res if isinstance(res, dict) else {"relations": []}
        elif inspect.iscoroutinefunction(getattr(self.graph, "get_subgraph_for_entities", None)):
            res = await self.graph.get_subgraph_for_entities([entity])
            return res if isinstance(res, dict) else {"relations": []}
        elif hasattr(self.graph, "nx_graph") and hasattr(self.graph.nx_graph, "edges"):
            raw_edges = self.graph.nx_graph.edges(data=True)
            if not isinstance(raw_edges, (list, tuple, set)) and not hasattr(raw_edges, "__iter__"):
                return {"relations": []}
            edges = []
            for item in raw_edges:
                if isinstance(item, tuple) and len(item) == 3 and isinstance(item[2], dict):
                    u, v, data = item
                    edges.append({
                        "subject": u,
                        "predicate": data.get("predicate", "relates_to"),
                        "object": v,
                        "confidence": data.get("confidence", 1.0),
                    })
            return {"relations": edges, "active_roots": [entity]}
        elif isinstance(self.graph, dict) or hasattr(self.graph, "adj"):
            edges = []
            adj = self.graph if isinstance(self.graph, dict) else getattr(self.graph, "adj", {})
            if isinstance(adj, dict):
                for u, nbrs in adj.items():
                    for nbr in nbrs:
                        if isinstance(nbr, dict):
                            edges.append({
                                "subject": u,
                                "predicate": nbr.get("predicate", "leads_to"),
                                "object": nbr.get("target", ""),
                                "confidence": nbr.get("confidence", 1.0),
                            })
                        elif isinstance(nbr, str):
                            edges.append({"subject": u, "predicate": "leads_to", "object": nbr, "confidence": 1.0})
            return {"relations": edges, "active_roots": [entity]}
        return {"relations": []}

    async def recalibrate_graph_with_annealer(
        self,
        energy_module: Optional[Any] = None,
        annealer: Optional[Any] = None,
        target_entity: Optional[str] = None,
        max_depth: int = 2,
        anneal_max_iter: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Wektor V37: Autonomiczna rekalibracja wag grafu przyczynowego Kùzu przez wyżarzanie.
        
        Pobiera podgraf wokół target_entity, tworzy obiekty CausalEdge, optymalizuje ich wagi
        za pomocą CausalAnnealer i aktualizuje zoptymalizowane pewności z powrotem w grafie.

        V68 (ARCH-003): opcjonalny `anneal_max_iter` ogranicza liczbę iteracji
        wyżarzania (faza cyklu snu używa małego limitu, np. 100). Domyślnie None
        => zachowane dotychczasowe zachowanie RPC 'anneal' (limit domyślny
        CausalAnnealer.run, 500). Zapis korygowanych pewnością krawędzi idzie
        WYŁĄCZNIE przez EpistemicWritePath.commit_relation (choke point V56) —
        żadnego bezpośredniego graph.add_relation tutaj ani u konsumentów.
        """
        if self.graph is None:
            return {"status": "no_graph", "updated_edges": 0}

        subgraph_data = await self._fetch_subgraph(target_entity or "Root", max_depth=max_depth)
        relations = subgraph_data.get("relations", []) if isinstance(subgraph_data, dict) else []
        if not relations:
            # If specific target entity was requested and has no direct edges, do not recalibrate the entire graph
            if target_entity and target_entity != "Root":
                logger.info("[ANNEAL] Target entity '%s' has no causal subgraphs; skipping anneal.", target_entity)
                return {"status": "no_relations", "updated_edges": 0}
            # Fallback to any graph relations if generic root requested
            nx_g = getattr(self.graph, "nx_graph", None)
            num_edges = nx_g.number_of_edges() if nx_g is not None and hasattr(nx_g, "number_of_edges") else 0
            if isinstance(num_edges, int) and num_edges > 0:
                edges_fb = []
                for u, v, data in nx_g.edges(data=True):
                    edges_fb.append({
                        "subject": u,
                        "predicate": data.get("predicate", "relates_to"),
                        "object": v,
                        "confidence": data.get("confidence", 1.0),
                    })
                relations = edges_fb

        if not relations:
            logger.info("[ANNEAL] No relations found in graph to anneal for target='%s'", target_entity)
            return {"status": "no_relations", "updated_edges": 0}

        edges = [
            CausalEdge(
                source=r["subject"],
                predicate=r["predicate"],
                target=r["object"],
                confidence=float(r.get("confidence", 1.0)),
                impact_type="critical_dependency" if r.get("predicate") in self.critical_predicates else "general_dependency",
            )
            for r in relations
            if r.get("subject") and r.get("object") and r.get("predicate")
        ]

        if not edges:
            return {"status": "empty_edges", "updated_edges": 0}

        if energy_module is None:
            from atlas_memory.causal.energy_module import EnergyModule
            energy_module = EnergyModule(graph_client=self.graph)

        if annealer is None:
            from atlas_memory.causal.annealer import CausalAnnealer
            annealer = CausalAnnealer(energy_module=energy_module)

        anneal_res = await annealer.run(
            edges,
            max_iter=anneal_max_iter if anneal_max_iter is not None else 500,
        )

        from atlas_memory.cognitive.write_path import EpistemicWritePath

        write_path = EpistemicWritePath(graph_store=self.graph)
        updated_count = 0
        for opt_edge in anneal_res.final_edges:
            try:
                if write_path.commit_relation(
                    opt_edge.source,
                    opt_edge.predicate,
                    opt_edge.target,
                    confidence=opt_edge.confidence,
                ):
                    updated_count += 1
            except Exception as exc:
                logger.debug("Error committing annealed edge: %s", exc)

        logger.info(
            "[ANNEAL] Causal graph anneal completed for target='%s': iterations=%d, updated_edges=%d, converged=%s, duration=%.2fms",
            target_entity, anneal_res.iterations, updated_count, anneal_res.converged, anneal_res.duration_ms
        )

        return {
            "status": "ok",
            "iterations": anneal_res.iterations,
            "converged": anneal_res.converged,
            "duration_ms": anneal_res.duration_ms,
            "updated_edges": updated_count,
        }
