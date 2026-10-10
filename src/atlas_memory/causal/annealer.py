from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from atlas_memory.causal.energy_module import EnergyModule
from atlas_memory.causal.models import CausalEdge
from atlas_memory.cognitive.models import clamp_conf


class AnnealingResult(BaseModel):
    """Wynik procesu wyżarzania grafu przyczynowego (V19).

    Zawiera zoptymalizowane krawędzie, ślad energii per iteracja oraz
    metadane konwergencji procesu termodynamicznego.
    """
    model_config = ConfigDict(frozen=True, extra="ignore")

    final_edges: List[CausalEdge] = Field(default_factory=list)
    energy_trace: List[float] = Field(default_factory=list, description="Całkowita energia po każdej iteracji")
    iterations: int = Field(default=0, ge=0)
    converged: bool = Field(default=False)
    final_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    duration_ms: float = Field(default=0.0, ge=0.0)


class CausalAnnealer:
    """
    Faza V19: Asynchronous Causal Graph Self-Annealing.

    Termodynamiczna optymalizacja wag zaufania (confidence) krawędzi
    grafu przyczynowego wg formuły:

        w_e^(k+1) = w_e^(k) - eta * (dE/dw_e) + sigma * epsilon

    gdzie:
    - eta  = T(k)  (temperatura steruje krokiem gradientowym)
    - sigma = T(k) (temperatura steruje szumem eksploracyjnym)
    - epsilon ~ N(0, 1)

    Proces symuluje wyżarzanie (simulated annealing) — wysokie T na starcie
    eksploruje przestrzeń, malejące T stabilizuje rozwiązanie w minimum
    energii kognitywnej (koszt błędu predykcji + ryzyko CPoF + regularyzacja).
    """

    ENERGY_CONVERGENCE_EPS: float = 1e-6

    def __init__(
        self,
        energy_module: Optional[EnergyModule] = None,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.energy_module = energy_module or EnergyModule()
        self._rng = random.Random(rng_seed)

    async def run(
        self,
        graph: List[CausalEdge],
        T_init: Optional[float] = None,
        T_min: float = 0.01,
        decay: Optional[float] = None,
        max_iter: int = 500,
        lambda_reg: float = 0.1,
        active_sensing: Optional[Any] = None,
        entity: Optional[str] = None,
        precision_tensor: Optional[Any] = None,
    ) -> AnnealingResult:
        """Uruchamia wyżarzanie grafu sprzężone z temperaturą poznawczą i tensorem precyzji (V154)."""
        start = time.perf_counter()

        T_min = max(1e-8, float(T_min))
        max_iter = max(1, int(max_iter))

        if not graph:
            return AnnealingResult(
                final_edges=[],
                energy_trace=[],
                iterations=1,
                converged=True,
                final_temperature=T_min,
                duration_ms=0.0,
            )

        # Causal-Sensing Thermal Bridge (Filar I)
        if active_sensing is None:
            try:
                from atlas_memory.active.prediction_error import get_active_sensing_engine
                active_sensing = get_active_sensing_engine()
            except Exception:
                active_sensing = None

        target_ent = entity or (graph[0].source if graph else "")
        t_cogn = 0.0
        if active_sensing is not None and hasattr(active_sensing, "get_entity_temperature"):
            t_cogn = float(active_sensing.get_entity_temperature(target_ent))

        if t_cogn > 0.0:
            eff_T_init = max(T_min, t_cogn)
        elif T_init is not None:
            eff_T_init = max(T_min, float(T_init))
        else:
            eff_T_init = T_min

        eff_decay = float(decay) if decay is not None else (0.95 if eff_T_init > T_min else 1.0)
        eff_decay = max(0.01, min(1.0, eff_decay))

        # Precision Tensor resolution
        if precision_tensor is None:
            try:
                from atlas_memory.l3_procedural.auditor import BayesianPrecisionTensor
                precision_tensor = BayesianPrecisionTensor()
            except Exception:
                precision_tensor = None

        # Fast NumPy / O(1) incremental kernel when using standard EnergyModule (F2)
        if type(self.energy_module) is EnergyModule:
            return await self._run_vectorized_numpy(
                graph=graph,
                T_init=eff_T_init,
                T_min=T_min,
                decay=eff_decay,
                max_iter=max_iter,
                lambda_reg=lambda_reg,
                start=start,
                precision_tensor=precision_tensor,
            )

        return await self._run_reference_scalar(
            graph=graph,
            T_init=eff_T_init,
            T_min=T_min,
            decay=eff_decay,
            max_iter=max_iter,
            lambda_reg=lambda_reg,
            start=start,
            precision_tensor=precision_tensor,
        )

    async def _run_vectorized_numpy(
        self,
        graph: List[CausalEdge],
        T_init: float,
        T_min: float,
        decay: float,
        max_iter: int,
        lambda_reg: float,
        start: float,
        precision_tensor: Optional[Any] = None,
    ) -> AnnealingResult:
        """Jednowątkowe jądro NumPy + O(1) przyrostowy stan wyżarzania (F2: tolerancja 1e-9 względem EnergyModule)."""
        import numpy as np

        n_edges = len(graph)
        w_arr = np.array([clamp_conf(e.confidence) for e in graph], dtype=np.float64)
        base_costs = np.array(
            [
                await self.energy_module.intrinsic_cost(e.source, e.predicate, current_confidence=None)
                for e in graph
            ],
            dtype=np.float64,
        )
        is_crit = np.array(
            [
                (e.predicate.lower() in self.energy_module.CRITICAL_PREDICATES or e.impact_type == "critical_dependency")
                for e in graph
            ],
            dtype=np.float64,
        )

        # Precompute channel precisions for Langevin noise
        precisions = np.ones(n_edges, dtype=np.float64)
        if precision_tensor is not None and hasattr(precision_tensor, "get_precision"):
            for i, e in enumerate(graph):
                st = e.metadata.get("source_type", "agent_inference") if hasattr(e, "metadata") and e.metadata else "agent_inference"
                precisions[i] = max(0.05, float(precision_tensor.get_precision(st)))

        # Map node names to integer IDs for O(1) NumPy array indexing
        node_to_id: Dict[str, int] = {}
        for e in graph:
            if e.source not in node_to_id:
                node_to_id[e.source] = len(node_to_id)
            if e.target not in node_to_id:
                node_to_id[e.target] = len(node_to_id)

        n_nodes = len(node_to_id)
        src_ids = np.array([node_to_id[e.source] for e in graph], dtype=np.int64)
        tgt_ids = np.array([node_to_id[e.target] for e in graph], dtype=np.int64)

        out_degree = np.zeros(n_nodes, dtype=np.float64)
        total_conf_sum = np.zeros(n_nodes, dtype=np.float64)
        crit_conf_sum = np.zeros(n_nodes, dtype=np.float64)
        np.add.at(out_degree, src_ids, 1.0)
        np.add.at(total_conf_sum, src_ids, w_arr)
        np.add.at(crit_conf_sum, src_ids, w_arr * is_crit)

        def _critic_for_node(nid: int) -> float:
            deg = out_degree[nid]
            if deg <= 0.0:
                return 0.0
            reachability_impact = min(1.0, deg / max(1.0, float(n_nodes)))
            t_sum = total_conf_sum[nid]
            c_sum = crit_conf_sum[nid]
            c_ratio = min(1.0, max(0.0, c_sum / max(1e-5, t_sum))) if t_sum > 0.0 else 0.0
            raw_risk = max(reachability_impact, c_ratio * min(1.0, deg / 2.0))
            if not math.isfinite(raw_risk):
                raw_risk = 0.0
            return round(float(max(0.0, min(1.0, raw_risk))), 6)

        def _edge_energy_and_grad(idx: int, w_val: float) -> tuple[float, float]:
            bc = base_costs[idx]
            conf = clamp_conf(w_val)
            if bc > 0.4:
                raw_cost = bc * (0.8 + 0.4 * conf)
                cost = min(1.0, raw_cost)
                dc_dw = 0.4 * bc if raw_cost < 1.0 else 0.0
            else:
                raw_cost = bc * (1.2 - 0.4 * conf)
                cost = max(0.0, raw_cost)
                dc_dw = -0.4 * bc if raw_cost > 0.0 else 0.0
            if not math.isfinite(cost):
                cost = 0.1
                dc_dw = 0.0
            c_cost = float(max(0.0, min(1.0, cost)))
            critic = _critic_for_node(int(tgt_ids[idx]))
            k = 1.0 - critic
            align_term = conf * c_cost + (1.0 - conf) * (1.0 - c_cost) * k
            reg_term = lambda_reg * ((conf - 0.5) ** 2)
            energy = float(align_term + reg_term)
            grad = float(
                c_cost
                - (1.0 - c_cost) * k
                + 2.0 * lambda_reg * (conf - 0.5)
                + dc_dw * (conf - (1.0 - conf) * k)
            )
            if not math.isfinite(energy):
                energy = 1.0
            if not math.isfinite(grad):
                grad = 0.0
            return energy, grad

        temperature = max(T_min, T_init)
        energy_trace: List[float] = []
        prev_total_energy: float = float("inf")
        iterations: int = 0
        converged: bool = False
        reached_T_min: bool = False

        while iterations < max_iter:
            iterations += 1
            total_energy = 0.0

            for i in range(n_edges):
                old_w = float(w_arr[i])
                energy, grad = _edge_energy_and_grad(i, old_w)

                sigma = math.sqrt(2.0 * temperature / precisions[i])
                epsilon = self._rng.gauss(0.0, 1.0)
                proposed_w = clamp_conf(old_w - temperature * grad + sigma * epsilon)
                cand_energy, _ = _edge_energy_and_grad(i, proposed_w)

                dE = cand_energy - energy
                accept = False
                if not math.isfinite(dE):
                    accept = False
                elif dE <= 0.0:
                    accept = True
                elif temperature < 1e-8:
                    accept = False
                else:
                    exponent = -dE / temperature
                    if exponent < -700.0:
                        p_accept = 0.0
                    elif exponent > 700.0:
                        p_accept = 1.0
                    else:
                        p_accept = math.exp(exponent)
                    if self._rng.random() < p_accept:
                        accept = True

                if accept:
                    delta_w = proposed_w - old_w
                    w_arr[i] = proposed_w
                    s_id = int(src_ids[i])
                    total_conf_sum[s_id] += delta_w
                    if is_crit[i] > 0.0:
                        crit_conf_sum[s_id] += delta_w
                    total_energy += cand_energy
                else:
                    total_energy += energy

            if not math.isfinite(total_energy):
                total_energy = prev_total_energy if math.isfinite(prev_total_energy) else 0.0

            energy_trace.append(total_energy)

            if math.isfinite(prev_total_energy) and abs(total_energy - prev_total_energy) < self.ENERGY_CONVERGENCE_EPS:
                converged = True
                break
            prev_total_energy = total_energy

            if temperature <= T_min:
                reached_T_min = True
                break
            temperature *= decay

        if reached_T_min:
            temperature = T_min

        final_edges = [
            e.model_copy(update={"confidence": float(w_arr[idx])})
            for idx, e in enumerate(graph)
        ]
        duration_ms = (time.perf_counter() - start) * 1000.0
        return AnnealingResult(
            final_edges=final_edges,
            energy_trace=energy_trace,
            iterations=iterations,
            converged=converged,
            final_temperature=round(temperature, 6),
            duration_ms=round(duration_ms, 4),
        )

    async def _run_reference_scalar(
        self,
        graph: List[CausalEdge],
        T_init: float,
        T_min: float,
        decay: float,
        max_iter: int,
        lambda_reg: float,
        start: float,
        precision_tensor: Optional[Any] = None,
    ) -> AnnealingResult:
        """Referencyjna ścieżka skalarna (dla niestandardowych podklas/mocków EnergyModule oraz testów równoważności)."""
        working: List[CausalEdge] = [e.model_copy(deep=True) for e in graph]
        temperature = max(T_min, T_init)
        energy_trace: List[float] = []
        prev_total_energy: float = float("inf")
        iterations: int = 0
        converged: bool = False
        reached_T_min: bool = False

        edge_index: Dict[str, List[CausalEdge]] = {}
        for e in working:
            if e.source not in edge_index:
                edge_index[e.source] = []
            edge_index[e.source].append(e)

        while iterations < max_iter:
            iterations += 1
            total_energy = 0.0
            for i, edge in enumerate(working):
                energy, grad = await self.energy_module.compute_edge_energy_and_gradient(
                    edge,
                    edges_context=edge_index,
                    lambda_reg=lambda_reg,
                )
                eta = temperature
                prec = 1.0
                if precision_tensor is not None and hasattr(precision_tensor, "get_precision"):
                    st = edge.metadata.get("source_type", "agent_inference") if hasattr(edge, "metadata") and edge.metadata else "agent_inference"
                    prec = max(0.05, float(precision_tensor.get_precision(st)))
                sigma = math.sqrt(2.0 * temperature / prec)
                epsilon = self._rng.gauss(0.0, 1.0)
                proposed_w = clamp_conf(edge.confidence - eta * grad + sigma * epsilon)
                cand_edge = edge.model_copy(update={"confidence": proposed_w})
                cand_energy, _ = await self.energy_module.compute_edge_energy_and_gradient(
                    cand_edge,
                    edges_context=edge_index,
                    lambda_reg=lambda_reg,
                )
                dE = cand_energy - energy
                accept = False
                if not math.isfinite(dE):
                    accept = False
                elif dE <= 0:
                    accept = True
                elif temperature < 1e-8:
                    accept = False
                else:
                    exponent = -dE / temperature
                    p_accept = 0.0 if exponent < -700.0 else (1.0 if exponent > 700.0 else math.exp(exponent))
                    if self._rng.random() < p_accept:
                        accept = True
                if accept:
                    working[i] = cand_edge
                    total_energy += cand_energy
                    source_list = edge_index.get(cand_edge.source, [])
                    for idx, e_ref in enumerate(source_list):
                        if e_ref is edge:
                            source_list[idx] = cand_edge
                            break
                else:
                    total_energy += energy

            if not math.isfinite(total_energy):
                total_energy = prev_total_energy if math.isfinite(prev_total_energy) else 0.0
            energy_trace.append(total_energy)
            if math.isfinite(prev_total_energy) and abs(total_energy - prev_total_energy) < self.ENERGY_CONVERGENCE_EPS:
                converged = True
                break
            prev_total_energy = total_energy
            if temperature <= T_min:
                reached_T_min = True
                break
            temperature *= decay

        if reached_T_min:
            temperature = T_min
        duration_ms = (time.perf_counter() - start) * 1000.0
        return AnnealingResult(
            final_edges=working,
            energy_trace=energy_trace,
            iterations=iterations,
            converged=converged,
            final_temperature=round(temperature, 6),
            duration_ms=round(duration_ms, 4),
        )

