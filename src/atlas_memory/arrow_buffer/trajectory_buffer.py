from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

from atlas_memory.models import PredictedTransition


class ArrowTrajectoryBuffer:
    """
    Bufor pamięci roboczej i trajektorii oparty na Apache Arrow (Zero-Copy RAM & Parquet).
    
    Umożliwia:
    1. Przechowywanie milionów kroków i wektorów stanu w pamięci RAM w formacie kolumnowym.
    2. Bezkopiowe (Zero-Copy) przekazywanie macierzy tensorów do Numb JIT / NumPy.
    3. Natychmiastowy zrzut do formatu Apache Parquet dla fazy L3 Sleep Consolidation.
    """

    def __init__(self, state_dim: int = 32):
        self.state_dim = state_dim
        self._steps: List[int] = []
        self._timestamps: List[float] = []
        self._session_ids: List[str] = []
        self._latent_vectors: List[List[float]] = []
        self._action_names: List[str] = []
        self._rewards: List[float] = []
        self._uncertainties: List[float] = []
        self._metadata_jsons: List[str] = []

    def append_transition(
        self,
        transition: PredictedTransition,
        session_id: str = "default_session",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Dodaje pojedyncze przejście stanu do bufora."""
        self._steps.append(transition.predicted_state.step_index)
        self._timestamps.append(transition.predicted_state.timestamp)
        self._session_ids.append(session_id)
        self._latent_vectors.append(transition.predicted_state.vector)
        self._action_names.append(transition.action.name)
        self._rewards.append(transition.simulated_reward)
        self._uncertainties.append(transition.uncertainty)
        self._metadata_jsons.append(json.dumps(metadata or {}))

    def to_arrow_table(self) -> Any:
        """Kompiluje dane w pamięci RAM do natywnej tabeli Apache Arrow."""
        if not HAS_PYARROW:
            return None

        schema = pa.schema([
            ("step_id", pa.int64()),
            ("timestamp", pa.float64()),
            ("session_id", pa.string()),
            ("latent_vector", pa.list_(pa.float64())),
            ("action_name", pa.string()),
            ("reward", pa.float64()),
            ("uncertainty", pa.float64()),
            ("metadata", pa.string()),
        ])

        data = [
            pa.array(self._steps, type=pa.int64()),
            pa.array(self._timestamps, type=pa.float64()),
            pa.array(self._session_ids, type=pa.string()),
            pa.array(self._latent_vectors, type=pa.list_(pa.float64())),
            pa.array(self._action_names, type=pa.string()),
            pa.array(self._rewards, type=pa.float64()),
            pa.array(self._uncertainties, type=pa.float64()),
            pa.array(self._metadata_jsons, type=pa.string()),
        ]

        return pa.Table.from_arrays(data, schema=schema)

    def to_numpy_latent_matrix(self) -> np.ndarray:
        """
        Zwraca macierz stanów ukrytych (N x state_dim) bez narzutu pętli Pythona.
        """
        if not self._latent_vectors:
            return np.empty((0, self.state_dim), dtype=np.float64)
        return np.ascontiguousarray(np.array(self._latent_vectors, dtype=np.float64))

    def to_zero_copy_tensor(self) -> np.ndarray:
        """
        Zwraca bezkopiowy widok tensorowy NumPy na kolumny pamięci RAM PyArrow.
        """
        if not self._latent_vectors:
            return np.empty((0, self.state_dim), dtype=np.float64)
        if HAS_PYARROW and self._steps:
            table = self.to_arrow_table()
            if table is not None:
                flat = np.asarray(table["latent_vector"].combine_chunks().values)
                return flat.reshape(-1, self.state_dim)
        return self.to_numpy_latent_matrix()


    def dump_to_parquet(self, file_path: str) -> bool:
        """
        Zapisuje zawartość bufora do pliku Apache Parquet dla fazy L3 Sleep Consolidation.
        """
        if not HAS_PYARROW or not self._steps:
            return False

        table = self.to_arrow_table()
        pq.write_table(table, file_path, compression="snappy")
        return True

    @classmethod
    def load_from_parquet(cls, file_path: str, state_dim: int = 32) -> ArrowTrajectoryBuffer:
        """Ładuje historię epizodu z pliku Parquet."""
        buf = cls(state_dim=state_dim)
        if not HAS_PYARROW:
            return buf

        table = pq.read_table(file_path)
        buf._steps = table["step_id"].to_pylist()
        buf._timestamps = table["timestamp"].to_pylist()
        buf._session_ids = table["session_id"].to_pylist()
        buf._latent_vectors = table["latent_vector"].to_pylist()
        buf._action_names = table["action_name"].to_pylist()
        buf._rewards = table["reward"].to_pylist()
        buf._uncertainties = table["uncertainty"].to_pylist()
        buf._metadata_jsons = table["metadata"].to_pylist()
        return buf

    def __len__(self) -> int:
        return len(self._steps)

    def clear(self) -> None:
        self._steps.clear()
        self._timestamps.clear()
        self._session_ids.clear()
        self._latent_vectors.clear()
        self._action_names.clear()
        self._rewards.clear()
        self._uncertainties.clear()
        self._metadata_jsons.clear()

