from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from loop_memory.causal.models import CausalEdge


class EnergyModule:
    """
    Faza V19: Asynchronous Causal Graph Self-Annealing — Energy Module.
    
    Oblicza:
    1. intrinsic_cost(state, action): koszt błędu predykcji telemetrii / rozbieżności.
    2. trainable_critic(state): predykcja przyszłego ryzyka CPoF (Critical Point of Failure)
       na podstawie krawędzi grafu przyczynowego i ich pewności.
    3. compute_edge_energy_and_gradient(edge, edges_context): energia i gradient dE/dw dla krawędzi.
    
    Wszystkie wartości kosztu i krytyka są ściśle znormalizowane do zakresu [0.0, 1.0].
    """

    CRITICAL_PREDICATES: Set[str] = {
        "depends_on", "relies_on", "hosted_on", "hosts", "powers", "routes_to",
        "critical_for", "requires", "authenticates", "mounts", "databases",
    }

    VOLATILE_ACTIONS: Set[str] = {
        "restart", "reboot", "delete", "drop", "modify", "stop", "kill", "terminate",
        "alter", "truncate", "poweroff", "flush", "destroy",
    }

    def __init__(
        self,
        active_sensing: Optional[Any] = None,
        graph_client: Optional[Any] = None,
    ) -> None:
        self.active_sensing = active_sensing
        self.graph_client = graph_client
        self._error_penalties: Dict[str, float] = {}

    def register_prediction_error(self, state: str, action: str, discrepancy_score: float) -> None:
        """Rejestruje błąd predykcji telemetrii dla danej pary stan-akcja."""
        key = f"{state}::{action}"
        self._error_penalties[key] = max(0.0, min(1.0, float(discrepancy_score)))

    async def intrinsic_cost(
        self,
        state: str,
        action: str,
        current_confidence: Optional[float] = None,
    ) -> float:
        """
        Koszt błędu predykcji telemetrii dla danego stanu i akcji.
        Zwraca wartość w zakresie [0.0, 1.0].
        
        Uwzględnia:
        - Zarejestrowane błędy predykcji telemetrii / rozbieżności ze środowiskiem
        - Ryzyko akcji (akcje destrukcyjne / mutujące)
        - Aktualną pewność krawędzi (niska pewność zwiększa koszt niepewności)
        """
        key = f"{state}::{action}"

        # 1. Sprawdzenie zarejestrowanych błędów
        if key in self._error_penalties:
            base_cost = self._error_penalties[key]
        elif self.active_sensing is not None and hasattr(self.active_sensing, "error_history"):
            relevant_errors = [
                float(e.discrepancy_score)
                for e in self.active_sensing.error_history
                if getattr(e, "target_entity", "") == state
            ]
            base_cost = max(relevant_errors) if relevant_errors else 0.1
        else:
            # Heurystyka oparta na naturze akcji
            act_lower = str(action).lower()
            is_volatile = any(va in act_lower for va in self.VOLATILE_ACTIONS)
            base_cost = 0.5 if is_volatile else 0.1

        # 2. Modyfikacja przez confidence jeśli podano
        if current_confidence is not None:
            conf = max(0.0, min(1.0, float(current_confidence)))
            if base_cost > 0.4:
                cost = min(1.0, base_cost * (0.8 + 0.4 * conf))
            else:
                cost = max(0.0, base_cost * (1.2 - 0.4 * conf))
        else:
            cost = base_cost

        return round(float(max(0.0, min(1.0, cost))), 6)

    async def trainable_critic(
        self,
        state: str,
        edges: Optional[Any] = None,
    ) -> float:
        """
        Predykcja przyszłego ryzyka CPoF (Critical Point of Failure) dla danego węzła.
        Zwraca wartość w zakresie [0.0, 1.0].
        
        Heurystyka oparta na:
        - Liczbie krawędzi wychodzących z węzła (out-degree)
        - Obecności relacji krytycznych (CRITICAL_PREDICATES)
        - Pewności krawędzi zależnych
        """
        relevant_edges: List[CausalEdge] = []
        if edges is not None:
            if isinstance(edges, dict):
                relevant_edges = edges.get(state, [])
            else:
                relevant_edges = [e for e in edges if e.source == state]
        elif self.graph_client is not None:
            try:
                subgraph = await self._fetch_subgraph(state, max_depth=1)
                relations = subgraph.get("relations", [])
                for r in relations:
                    if r.get("subject") == state:
                        relevant_edges.append(
                            CausalEdge(
                                source=state,
                                predicate=str(r.get("predicate", "")),
                                target=str(r.get("object", "")),
                                confidence=float(r.get("confidence", 1.0)),
                            )
                        )
            except Exception:
                relevant_edges = []


        if not relevant_edges:
            return 0.0

        num_outgoing = len(relevant_edges)
        critical_edges = [
            e for e in relevant_edges
            if e.predicate.lower() in self.CRITICAL_PREDICATES or e.impact_type == "critical_dependency"
        ]

        critical_conf_sum = sum(e.confidence for e in critical_edges)
        total_conf_sum = sum(e.confidence for e in relevant_edges)

        # Współczynnik ryzyka:
        # - Stopień wyjścia (węzeł z wieloma zależnościami jest potencjalnym CPoF)
        # - Odsetek i waga krytycznych relacji
        branching_factor = min(1.0, num_outgoing / 5.0)
        critical_ratio = (critical_conf_sum / max(1e-5, total_conf_sum)) if total_conf_sum > 0 else 0.0

        raw_risk = (0.5 * critical_ratio) + (0.3 * branching_factor) + (0.2 * min(1.0, critical_conf_sum / 3.0))

        return round(float(max(0.0, min(1.0, raw_risk))), 6)

    async def compute_edge_energy_and_gradient(
        self,
        edge: CausalEdge,
        edges_context: Optional[List[CausalEdge]] = None,
        lambda_reg: float = 0.1,
    ) -> tuple[float, float]:
        """
        Oblicza energię E_e(w) oraz gradient dE/dw dla pojedynczej krawędzi.
        
        Formuła energii:
        E_e(w) = w * c_cost + (1 - w) * (1 - c_cost) * (1 - critic) + lambda_reg * (w - 0.5)^2
        gdzie:
        - c_cost = intrinsic_cost(edge.source, edge.predicate, edge.confidence)
        - critic = trainable_critic(edge.target, edges_context)
        
        Gradient dE/dw:
        dE/dw = c_cost - (1 - c_cost) * (1 - critic) + 2 * lambda_reg * (w - 0.5)
        """
        c_cost = await self.intrinsic_cost(edge.source, edge.predicate, current_confidence=edge.confidence)
        critic = await self.trainable_critic(edge.target, edges=edges_context)

        w = float(edge.confidence)

        # Obliczenie energii
        align_term = w * c_cost + (1.0 - w) * (1.0 - c_cost) * (1.0 - critic)
        reg_term = lambda_reg * ((w - 0.5) ** 2)
        energy = align_term + reg_term

        # Gradient
        grad = c_cost - ((1.0 - c_cost) * (1.0 - critic)) + 2.0 * lambda_reg * (w - 0.5)

        return float(energy), float(grad)

    async def _fetch_subgraph(self, entity: str, max_depth: int = 1) -> Dict[str, Any]:
        """Pobiera podgraf dla danego węzła z klienta grafowego."""
        if hasattr(self.graph_client, "get_subgraph_relations"):
            return await self.graph_client.get_subgraph_relations([entity], max_depth=max_depth)
        elif hasattr(self.graph_client, "get_subgraph_for_entities"):
            return await self.graph_client.get_subgraph_for_entities([entity])
        return {"relations": []}
