from __future__ import annotations

import copy
import logging
import threading
import time
import zlib
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from atlas_memory.core.models import ActionPlan, LatentState, PredictedTransition
from atlas_memory.l1_working.jepa_kernels import numba_jepa_adapt_step, numba_jepa_step

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JEPACheckpoint:
    """Niezmienny zrzut stanu wag i pamięci ukrytej L1 JEPA dla odwracalności symplektycznej."""
    w_s: np.ndarray
    w_a: np.ndarray
    bias: np.ndarray
    w_val: np.ndarray
    current_state: LatentState
    history: List[LatentState]

JEPA_STATUS: str = "projection-linear-predictor"

_GLOBAL_ONNX_ENCODER: Any = None
_ENCODER_INITIALIZED: bool = False


def get_onnx_dense_encoder() -> Any:
    """Returns singleton instance of OnnxDenseEncoder or None if unavailable."""
    global _GLOBAL_ONNX_ENCODER, _ENCODER_INITIALIZED
    if not _ENCODER_INITIALIZED:
        try:
            from atlas_memory.l2_semantic.onnx_encoder import (
                OnnxDenseEncoder,
            )
            _GLOBAL_ONNX_ENCODER = OnnxDenseEncoder(lazy_init=False)
        except Exception as exc:
            logger.debug("OnnxDenseEncoder unavailable for JEPA action encoding: %s", exc)
            _GLOBAL_ONNX_ENCODER = None
        _ENCODER_INITIALIZED = True
    return _GLOBAL_ONNX_ENCODER


class JEPALatentBuffer:
    """
    L1: Pamięć Robocza & Bufor Świata (Numba JIT Accelerated JEPA Buffer).
    
    Status modułu: projection-linear-predictor (V80.3 T3).
    UCZCIWA ETYKIETA (V67, F11): to NIE jest pełny JEPA (Joint-Embedding
    Predictive Architecture) z EMA-teacherem i siecią celową — to model
    INSPIROWANY JEPA: liniowy/jednowarstwowy predyktor przejść stanu
        s_{t+1} = tanh(s_t @ W_s + a_t @ W_a + bias)
    uczony przez adaptację TTT (test-time training) na przejściach RAM-owych
    z realnych epizodów (engine.record_mental_transition). Brak momentum-encoder,
    brak EMA-teacher, brak pary augmentacji.
    Proweniencja wektorów akcji jest jawnie oznaczana w ActionPlan.action_encoding
    ('onnx_embed' | 'hash_projection' | 'ttt_latent').
    """

    def __init__(
        self,
        state_dim: int = 32,
        action_dim: int = 16,
        seed: int | None = 42,
        max_history: int = 1000,
        encoder: Any | None = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_history = max_history
        self.encoder = encoder
        self._lock = threading.Lock()

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
        self.history: list[LatentState] = [self._current_state]

    def _create_initial_state(self, vector: list[float] | None = None) -> LatentState:
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
        with self._lock:
            return self._current_state

    def reset(self, initial_vector: list[float] | None = None) -> LatentState:
        with self._lock:
            self._current_state = self._create_initial_state(initial_vector)
            self.history = [self._current_state]
            return self._current_state

    def _encode_action_vector(self, action: ActionPlan) -> np.ndarray:
        """Koduje akcję jako 1D wektor numpy (action_dim,) bez nadmiarowych wymiarów."""
        if action.latent_projection is not None and len(action.latent_projection) == self.action_dim:
            if action.action_encoding is None:
                action.action_encoding = "latent_projection"
            return np.asarray(action.latent_projection, dtype=np.float64)

        # Bezpieczna serializacja parametrów odporna na typy niesortowalne
        safe_params = sorted((str(k), str(v)) for k, v in action.parameters.items())
        action_str = f"{action.name}:{safe_params}"

        # Attempt ONNX dense encoding if available (T3 provenance)
        encoder = self.encoder if self.encoder is not None else get_onnx_dense_encoder()
        if encoder is not None:
            try:
                res = encoder.encode(action_str)
                vec = res[0] if (isinstance(res, list) and res and isinstance(res[0], list)) else res
                if vec is not None and len(vec) >= self.action_dim:
                    action.action_encoding = "onnx_embed"
                    return np.asarray(vec[:self.action_dim], dtype=np.float64)
            except Exception as exc:
                logger.debug("JEPA action ONNX encoding failed, using hash: %s", exc)

        # Fallback to deterministic hash projection
        h = zlib.crc32(action_str.encode("utf-8"))
        rng = np.random.default_rng(h)
        action.action_encoding = "hash_projection"
        return (rng.standard_normal(self.action_dim) * 0.5).astype(np.float64)

    def encode_action(self, action: ActionPlan, action_latent: Optional[np.ndarray] = None) -> np.ndarray:
        """Zwraca 2D rzutowanie akcji (1, action_dim). Obsługuje latent_projection z TTT."""
        if action_latent is not None:
            arr = np.asarray(action_latent, dtype=np.float64).reshape(-1)
            if len(arr) < self.action_dim:
                arr = np.pad(arr, (0, self.action_dim - len(arr)))
            elif len(arr) > self.action_dim:
                arr = arr[:self.action_dim]
            action.action_encoding = "ttt_latent"
            return arr.reshape(1, self.action_dim)
        return self._encode_action_vector(action).reshape(1, self.action_dim)

    def predict_transition(
        self,
        current_state: LatentState,
        action: ActionPlan,
        action_latent: Optional[np.ndarray] = None,
    ) -> PredictedTransition:
        """Symulacja pojedynczego przejścia stanu w Numba JIT (synchroniczna)."""
        s_arr = np.ascontiguousarray(np.asarray(current_state.vector, dtype=np.float64)).reshape(1, self.state_dim)
        a_arr = self.encode_action(action, action_latent=action_latent)

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

    def step(self, action: ActionPlan, action_latent: Optional[np.ndarray] = None) -> PredictedTransition:
        """
        Zatwierdza krok w pamięci roboczej: wykonuje predykcję z bieżącego stanu,
        aktualizuje self._current_state oraz dopisuje przejście do historii.
        """
        transition = self.predict_transition(self._current_state, action, action_latent=action_latent)
        self.commit_transition(transition)
        return transition

    def commit_transition(self, transition: PredictedTransition) -> None:
        """Zatwierdza zrealizowane przejście w pamięci roboczej."""
        with self._lock:
            self._current_state = transition.predicted_state
            self.history.append(transition.predicted_state)
            if len(self.history) > self.max_history:
                self.history.pop(0)

    def adapt_transition(
        self,
        previous_state: LatentState,
        action: ActionPlan,
        observed_vector: LatentState | list[float] | np.ndarray,
        lr: float = 0.01,
        weight_decay: float = 0.001,
        action_latent: Optional[np.ndarray] = None,
    ) -> float:
        """
        Krok adaptacji online TTT dla wag JEPA (w_s, w_a, bias) na zaobserwowanym
        przejściu stanu ze środowiska (real-world transition).
        Aktualizuje wagi, uaktualnia self._current_state oraz dopisuje stan do historii.
        Zwraca wartość straty MSE przed aktualizacją.
        """
        s_arr = np.ascontiguousarray(np.asarray(previous_state.vector, dtype=np.float64)).reshape(1, self.state_dim)
        a_arr = self.encode_action(action, action_latent=action_latent)
        if isinstance(observed_vector, LatentState):
            t_vec = observed_vector.vector
            next_state = observed_vector
        else:
            t_vec = list(observed_vector) if not isinstance(observed_vector, list) else observed_vector
            if len(t_vec) < self.state_dim:
                t_vec = t_vec + [0.0] * (self.state_dim - len(t_vec))
            elif len(t_vec) > self.state_dim:
                t_vec = t_vec[:self.state_dim]
            next_state = LatentState(
                vector=list(t_vec),
                dimension=self.state_dim,
                step_index=max(previous_state.step_index, self._current_state.step_index) + 1,
                timestamp=time.time(),
                energy_score=0.0,
                context_tags=[action.name, "observed"],
            )
        t_arr = np.ascontiguousarray(np.asarray(t_vec, dtype=np.float64)).reshape(1, self.state_dim)

        with self._lock:
            loss = numba_jepa_adapt_step(
                s_arr,
                a_arr,
                t_arr,
                self.w_s,
                self.w_a,
                self.bias,
                lr=float(lr),
                weight_decay=float(weight_decay),
            )
            next_state.energy_score = float(loss)
            self._current_state = next_state
            self.history.append(next_state)
            if len(self.history) > self.max_history:
                self.history.pop(0)

        return float(loss)

    def capture_checkpoint(self) -> JEPACheckpoint:
        """Rejestruje punkt fazowy wag i pamięci ukrytej JEPA dla kompensacji symplektycznej."""
        with self._lock:
            curr_copy = self._current_state.model_copy(deep=True) if hasattr(self._current_state, "model_copy") else copy.deepcopy(self._current_state)
            hist_copy = [s.model_copy(deep=True) if hasattr(s, "model_copy") else copy.deepcopy(s) for s in self.history]
            return JEPACheckpoint(
                w_s=self.w_s.copy(),
                w_a=self.w_a.copy(),
                bias=self.bias.copy(),
                w_val=self.w_val.copy(),
                current_state=curr_copy,
                history=hist_copy,
            )

    def restore_checkpoint(self, cp: JEPACheckpoint) -> None:
        """Cofa krok adaptacji i przywraca stan fazowy JEPA (Phi_undo)."""
        with self._lock:
            np.copyto(self.w_s, cp.w_s)
            np.copyto(self.w_a, cp.w_a)
            np.copyto(self.bias, cp.bias)
            np.copyto(self.w_val, cp.w_val)
            self._current_state = cp.current_state.model_copy(deep=True) if hasattr(cp.current_state, "model_copy") else copy.deepcopy(cp.current_state)
            self.history = [s.model_copy(deep=True) if hasattr(s, "model_copy") else copy.deepcopy(s) for s in cp.history]


