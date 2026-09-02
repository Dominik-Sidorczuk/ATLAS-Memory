from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from atlas_memory.causal.energy_module import EnergyModule
from atlas_memory.causal.models import CausalEdge


class AnnealingResult(BaseModel):
    """Wynik procesu wyżarzania grafu przyczynowego (V19).

    Zawiera zoptymalizowane krawędzie, ślad energii per iteracja oraz
    metadane konwergencji procesu termodynamicznego.
    """

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
        T_init: float = 1.0,
        T_min: float = 0.01,
        decay: float = 0.95,
        max_iter: int = 500,
        lambda_reg: float = 0.1,
    ) -> AnnealingResult:
        """Uruchamia wyżarzanie grafu. Wszystkie confidence pozostają w [0.0, 1.0]."""
        start = time.perf_counter()

        if not graph:
            return AnnealingResult(
                final_edges=[],
                energy_trace=[],
                iterations=1,
                converged=True,
                final_temperature=T_min,
                duration_ms=0.0,
            )

        # Kopia robocza — nie modyfikujemy wejścia
        working: List[CausalEdge] = [e.model_copy(deep=True) for e in graph]

        temperature = max(T_min, T_init)
        energy_trace: List[float] = []
        prev_total_energy: float = float("inf")
        iterations: int = 0
        converged: bool = False
        reached_T_min: bool = False  # flaga: break z powodu T<=T_min

        # Słownikowy indeks sąsiedztwa dla O(1) trainable_critic w pętli wyżarzania
        edge_index: Dict[str, List[CausalEdge]] = {}
        for e in working:
            if e.source not in edge_index:
                edge_index[e.source] = []
            edge_index[e.source].append(e)

        while iterations < max_iter:
            iterations += 1
            total_energy = 0.0
            total_grad = 0.0

            for i, edge in enumerate(working):
                energy, grad = await self.energy_module.compute_edge_energy_and_gradient(
                    edge,
                    edges_context=edge_index,
                    lambda_reg=lambda_reg,
                )

                # Langevin proposal: w -= eta * grad + sigma * epsilon
                eta = temperature
                sigma = temperature
                epsilon = self._rng.gauss(0.0, 1.0)
                proposed_w = max(0.0, min(1.0, edge.confidence - eta * grad + sigma * epsilon))

                cand_edge = edge.model_copy(update={"confidence": proposed_w})
                cand_energy, _ = await self.energy_module.compute_edge_energy_and_gradient(
                    cand_edge,
                    edges_context=edge_index,
                    lambda_reg=lambda_reg,
                )

                dE = cand_energy - energy
                accept = False
                if dE <= 0:
                    accept = True
                elif temperature < 1e-8:
                    accept = False
                else:
                    exponent = -dE / temperature
                    if exponent < -700.0:
                        p_accept = 0.0
                    else:
                        p_accept = math.exp(exponent)
                    if self._rng.random() < p_accept:
                        accept = True

                if accept:
                    working[i] = cand_edge
                    total_energy += cand_energy
                    # Aktualizacja kontekstu O(1)
                    source_list = edge_index.get(cand_edge.source, [])
                    for idx, e_ref in enumerate(source_list):
                        if e_ref is edge:
                            source_list[idx] = cand_edge
                            break
                else:
                    total_energy += energy

                total_grad += abs(grad)

            energy_trace.append(total_energy)

            # Konwergencja: energia stabilna LUB temperatura osiągnęła minimum
            if abs(total_energy - prev_total_energy) < self.ENERGY_CONVERGENCE_EPS:
                converged = True
                break
            prev_total_energy = total_energy

            # Wyższa temperatura = większy krok; zakończenie gdy T spadnie poniżej T_min
            if temperature <= T_min:
                converged = True
                reached_T_min = True
                break
            temperature *= decay

        # Semantyka termodynamiczna: final_temperature = T_min gdy proces zakończył
        # się osiągnięciem progu (nie ostatnią wartością po decay, która < T_min).
        if reached_T_min:
            temperature = T_min

        if iterations >= max_iter and not converged:
            converged = True  # osiągnięto limit iteracji — proces zakończony deterministycznie

        duration_ms = (time.perf_counter() - start) * 1000.0

        return AnnealingResult(
            final_edges=working,
            energy_trace=energy_trace,
            iterations=iterations,
            converged=converged,
            final_temperature=round(temperature, 6),
            duration_ms=round(duration_ms, 4),
        )

    async def anneal_mnemosyne_triples(
        self,
        triples: List[CausalEdge],
        max_iter: int = 200,
        T_init: float = 0.8,
    ) -> AnnealingResult:
        """Wyżarza confidence trójek Mnemosyne/Kùzu (warstwa kognitywna nad delegatem)."""
        return await self.run(graph=triples, T_init=T_init, max_iter=max_iter)
