from __future__ import annotations

import asyncio
import time
import zlib
from typing import List, Optional, Tuple

import numpy as np

from atlas_memory.l1_working.jepa_kernels import numba_jepa_rollout, numba_jepa_step
from atlas_memory.models import ActionPlan, LatentState, PredictedTransition


class JEPALatentBuffer:
    """
    L1: Pamięć Robocza & Bufor Świata (Numba JIT Accelerated JEPA Buffer).
    
    Utrzymuje stan ukryty s_t i realizuje predykcję dynamiki:
        s_{t+1} = tanh(s_t @ W_s + a_t @ W_a + bias)
    oraz wielokrokowe rollouty Systemu 2 skompilowane w Numbie z obsługą wywołań asynchronicznych (asyncio.to_thread).
    """

    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = 16,
        seed: Optional[int] = 42,
        max_history: int = 1000,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_history = max_history

        rng = np.random.default_rng(seed)
        self.w_s = (rng.standard_normal((state_dim, state_dim)) * (0.8 / np.sqrt(state_dim))).astype(np.float64)
        self.w_a = (rng.standard_normal((action_dim, state_dim)) * (0.8 / np.sqrt(action_dim))).astype(np.float64)
        self.bias = np.zeros((1, state_dim), dtype=np.float64)
        self.w_val = (rng.standard_normal((state_dim, 1)) * 0.1).astype(np.float64)

        # Rozgrzanie JIT
        dummy_s = np.zeros((1, state_dim), dtype=np.float64)
        dummy_a = np.zeros((1, action_dim), dtype=np.float64)
        numba_jepa_step(dummy_s, dummy_a, self.w_s, self.w_a, self.bias, self.w_val)

        self._current_state: LatentState = self._create_initial_state()
        self.history: List[LatentState] = [self._current_state]

    def _create_initial_state(self, vector: Optional[List[float]] = None) -> LatentState:
        if vector is None:
            vec = (np.random.randn(self.state_dim) * 0.1).tolist()
        else:
            vec = list(vector)
            if len(vec) < self.state_dim:
                vec.extend([0.0] * (self.state_dim - len(vec)))
            elif len(vec) > self.state_dim:
                vec = vec[:self.state_dim]

        return LatentState(
            vector=vec,
            dimension=self.state_dim,
            step_index=0,
            timestamp=time.time(),
            energy_score=0.0,
            context_tags=["init"],
        )

    @property
    def current_state(self) -> LatentState:
        return self._current_state

    def reset(self, initial_vector: Optional[List[float]] = None) -> LatentState:
        self._current_state = self._create_initial_state(initial_vector)
        self.history = [self._current_state]
        return self._current_state

    def encode_action(self, action: ActionPlan) -> np.ndarray:
        if action.latent_projection is not None and len(action.latent_projection) == self.action_dim:
            arr = np.array(action.latent_projection, dtype=np.float64)
        else:
            action_str = f"{action.name}:{sorted(action.parameters.items())}"
            h = zlib.crc32(action_str.encode("utf-8"))
            rng = np.random.default_rng(h)
            arr = (rng.standard_normal(self.action_dim) * 0.5).astype(np.float64)
        return arr.reshape(1, self.action_dim)


    def predict_transition(
        self,
        current_state: LatentState,
        action: ActionPlan,
    ) -> PredictedTransition:
        """Symulacja pojedynczego przejścia stanu w Numba JIT (synchroniczna)."""
        s_arr = np.ascontiguousarray(np.asarray(current_state.vector, dtype=np.float64)).reshape(1, self.state_dim)
        a_arr = self.encode_action(action)

        next_s, sim_reward, uncertainty = numba_jepa_step(
            s_arr,
            a_arr,
            self.w_s,
            self.w_a,
            self.bias,
            self.w_val,
        )

        predicted_latent = LatentState(
            vector=next_s.reshape(-1).tolist(),
            dimension=self.state_dim,
            step_index=current_state.step_index + 1,
            timestamp=time.time(),
            energy_score=uncertainty,
            context_tags=[action.name],
        )

        return PredictedTransition(
            previous_state=current_state,
            action=action,
            predicted_state=predicted_latent,
            simulated_reward=sim_reward,
            uncertainty=uncertainty,
        )

    async def predict_transition_async(
        self,
        current_state: LatentState,
        action: ActionPlan,
    ) -> PredictedTransition:
        """Asynchroniczne przejście stanu w puli wątków."""
        return await asyncio.to_thread(self.predict_transition, current_state, action)

    def simulate_rollout(
        self,
        candidate_actions: List[ActionPlan],
        initial_state: Optional[LatentState] = None,
    ) -> List[PredictedTransition]:
        """Wielokrokowa symulacja myślowa Systemu 2."""
        state = initial_state or self._current_state
        if not candidate_actions:
            return []

        actions_mat = np.vstack([self.encode_action(act) for act in candidate_actions])
        s_0 = np.ascontiguousarray(np.asarray(state.vector, dtype=np.float64)).reshape(1, self.state_dim)

        states_mat, rewards, uncertainties = numba_jepa_rollout(
            s_0,
            actions_mat,
            self.w_s,
            self.w_a,
            self.bias,
            self.w_val,
        )

        trajectory: List[PredictedTransition] = []
        curr_state = state

        for i, act in enumerate(candidate_actions):
            next_state = LatentState(
                vector=states_mat[i].tolist(),
                dimension=self.state_dim,
                step_index=curr_state.step_index + 1,
                timestamp=time.time(),
                energy_score=float(uncertainties[i]),
                context_tags=[act.name],
            )
            trans = PredictedTransition(
                previous_state=curr_state,
                action=act,
                predicted_state=next_state,
                simulated_reward=float(rewards[i]),
                uncertainty=float(uncertainties[i]),
            )
            trajectory.append(trans)
            curr_state = next_state

        return trajectory

    async def simulate_rollout_async(
        self,
        candidate_actions: List[ActionPlan],
        initial_state: Optional[LatentState] = None,
    ) -> List[PredictedTransition]:
        """Asynchroniczny rollout w puli wątków dla długich horyzontów planowania."""
        return await asyncio.to_thread(self.simulate_rollout, candidate_actions, initial_state)

    def select_best_action_trajectory(
        self,
        action_sequences: List[List[ActionPlan]],
    ) -> Tuple[List[ActionPlan], List[PredictedTransition], float]:
        best_score = -float("inf")
        best_seq: List[ActionPlan] = []
        best_trajectory: List[PredictedTransition] = []

        for seq in action_sequences:
            traj = self.simulate_rollout(seq)
            score = sum(t.simulated_reward - 0.2 * t.uncertainty for t in traj)
            if score > best_score:
                best_score = score
                best_seq = seq
                best_trajectory = traj

        return best_seq, best_trajectory, best_score

    async def select_best_action_trajectory_async(
        self,
        action_sequences: List[List[ActionPlan]],
    ) -> Tuple[List[ActionPlan], List[PredictedTransition], float]:
        """Asynchroniczna selekcja optymalnej ścieżki w puli wątków."""
        return await asyncio.to_thread(self.select_best_action_trajectory, action_sequences)

    def commit_state_transition(self, action: ActionPlan) -> LatentState:
        transition = self.predict_transition(self._current_state, action)
        self._current_state = transition.predicted_state
        self.history.append(self._current_state)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        return self._current_state

    def inject_l0_latent(self, compressed_l0_vec: List[float]) -> LatentState:
        """Fuzja cech ukrytych z warstwy L0 TTT do stanu L1."""
        l0_arr = np.array(compressed_l0_vec, dtype=np.float64)
        if len(l0_arr) < self.state_dim:
            padded = np.zeros(self.state_dim, dtype=np.float64)
            padded[:len(l0_arr)] = l0_arr
            l0_arr = padded
        elif len(l0_arr) > self.state_dim:
            l0_arr = l0_arr[:self.state_dim]

        current_arr = np.array(self._current_state.vector, dtype=np.float64)
        fused = np.tanh(0.7 * current_arr + 0.3 * l0_arr)

        self._current_state = LatentState(
            vector=fused.tolist(),
            dimension=self.state_dim,
            step_index=self._current_state.step_index + 1,
            timestamp=time.time(),
            energy_score=self._current_state.energy_score,
            context_tags=self._current_state.context_tags + ["l0_injected"],
        )
        self.history.append(self._current_state)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        return self._current_state

