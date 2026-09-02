from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

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
                if flat.size % self.state_dim == 0 and flat.size > 0:
                    return flat.reshape(-1, self.state_dim)
        return self.to_numpy_latent_matrix()


    def dump_to_parquet(
        self,
        file_path: Union[str, Path],
        compression: str = "zstd",
        compression_level: Optional[int] = None,
    ) -> bool:
        """
        Zapisuje zawartość bufora do pliku Apache Parquet dla fazy L3 Sleep Consolidation.
        Obsługuje algorytmy kompresji 'zstd', 'snappy', 'gzip', 'none'.
        """
        if not HAS_PYARROW or not self._steps:
            return False

        table = self.to_arrow_table()
        write_kwargs: Dict[str, Any] = {"compression": compression}
        if compression_level is not None:
            write_kwargs["compression_level"] = compression_level

        pq.write_table(table, str(file_path), **write_kwargs)
        return True

    def stream_batches(
        self,
        batch_size: int = 256,
        zero_copy: bool = True,
    ) -> Iterator[Dict[str, Any]]:
        """
        Strumieniuje dane partiami (RecordBatch) bez konieczności duplikowania całej macierzy w pamięci RAM.
        """
        total = len(self._steps)
        if total == 0:
            return

        if HAS_PYARROW and zero_copy:
            table = self.to_arrow_table()
            for batch in table.to_batches(max_chunksize=batch_size):
                b_steps = np.asarray(batch["step_id"])
                b_timestamps = np.asarray(batch["timestamp"])
                b_actions = batch["action_name"].to_pylist()
                b_rewards = np.asarray(batch["reward"])
                b_uncertainties = np.asarray(batch["uncertainty"])
                
                # Zero-copy extraction of nested list values
                flat_vals = np.asarray(batch["latent_vector"].flatten())
                if flat_vals.size % self.state_dim != 0:
                    raise ValueError(f"Malformed latent vectors in buffer: length mismatch. Expected multiple of {self.state_dim}.")
                latent_mat = flat_vals.reshape(-1, self.state_dim)

                yield {
                    "step_ids": b_steps,
                    "timestamps": b_timestamps,
                    "latent_matrix": latent_mat,
                    "action_names": b_actions,
                    "rewards": b_rewards,
                    "uncertainties": b_uncertainties,
                    "batch_size": len(b_steps),
                }
        else:
            # Fallback for non-pyarrow or pure numpy slicing
            for start_idx in range(0, total, batch_size):
                end_idx = min(start_idx + batch_size, total)
                yield {
                    "step_ids": np.array(self._steps[start_idx:end_idx], dtype=np.int64),
                    "timestamps": np.array(self._timestamps[start_idx:end_idx], dtype=np.float64),
                    "latent_matrix": np.array(self._latent_vectors[start_idx:end_idx], dtype=np.float64),
                    "action_names": self._action_names[start_idx:end_idx],
                    "rewards": np.array(self._rewards[start_idx:end_idx], dtype=np.float64),
                    "uncertainties": np.array(self._uncertainties[start_idx:end_idx], dtype=np.float64),
                    "batch_size": end_idx - start_idx,
                }

    @classmethod
    def stream_from_parquet(
        cls,
        file_path: Union[str, Path],
        batch_size: int = 256,
        state_dim: int = 32,
    ) -> Iterator[Dict[str, Any]]:
        """
        Bezpośredni streaming partii z pliku Parquet bez wczytywania całego pliku do RAM.
        """
        if not HAS_PYARROW:
            return

        with pq.ParquetFile(str(file_path)) as parquet_file:
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                b_steps = np.asarray(batch["step_id"])
                b_timestamps = np.asarray(batch["timestamp"])
                b_actions = batch["action_name"].to_pylist()
                b_rewards = np.asarray(batch["reward"])
                b_uncertainties = np.asarray(batch["uncertainty"])
                
                flat_vals = np.asarray(batch["latent_vector"].flatten())
                if flat_vals.size % state_dim != 0:
                    raise ValueError(f"Malformed latent vectors in parquet: length mismatch. Expected multiple of {state_dim}.")
                latent_mat = flat_vals.reshape(-1, state_dim)

                yield {
                    "step_ids": b_steps,
                    "timestamps": b_timestamps,
                    "latent_matrix": latent_mat,
                    "action_names": b_actions,
                    "rewards": b_rewards,
                    "uncertainties": b_uncertainties,
                    "batch_size": len(b_steps),
                }

    def get_compression_stats(self) -> Dict[str, Any]:
        """
        Zwraca statystyki zajętości pamięci RAM oraz kompresji kolumnowej.
        """
        total = len(self._steps)
        raw_latent_bytes = total * self.state_dim * 8  # float64
        raw_meta_bytes = sum(len(m.encode("utf-8")) for m in self._metadata_jsons)
        raw_total_bytes = raw_latent_bytes + raw_meta_bytes + total * (8 + 8 + 8 + 8 + 16)

        arrow_bytes = 0
        if HAS_PYARROW and total > 0:
            table = self.to_arrow_table()
            arrow_bytes = table.nbytes if table is not None else 0

        return {
            "total_records": total,
            "state_dim": self.state_dim,
            "raw_estimated_ram_bytes": raw_total_bytes,
            "arrow_columnar_bytes": arrow_bytes,
            "ram_efficiency_ratio": (raw_total_bytes / arrow_bytes) if arrow_bytes > 0 else 1.0,
        }

    @classmethod
    def load_from_parquet(cls, file_path: Union[str, Path], state_dim: int = 32) -> ArrowTrajectoryBuffer:
        """Ładuje historię epizodu z pliku Parquet."""
        buf = cls(state_dim=state_dim)
        if not HAS_PYARROW:
            return buf

        table = pq.read_table(str(file_path))
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

