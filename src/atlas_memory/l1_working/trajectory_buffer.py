from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator, TypedDict

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

from atlas_memory.core.models import PredictedTransition


class BufferCompressionStats(TypedDict):
    """Statystyki kompresji kolumnowej oraz pamięci RAM bufora."""
    total_records: int
    state_dim: int
    raw_estimated_ram_bytes: int
    arrow_columnar_bytes: int
    ram_efficiency_ratio: float


class TrajectoryBatch(TypedDict):
    """Pojedyncza partia danych strumieniowanych z bufora trajektorii."""
    step_ids: np.ndarray
    timestamps: np.ndarray
    latent_matrix: np.ndarray
    action_names: list[str]
    rewards: np.ndarray
    uncertainties: np.ndarray
    batch_size: int


class ArrowTrajectoryBuffer:
    """
    Bufor pamięci roboczej i trajektorii oparty na Apache Arrow (Zero-Copy RAM & Parquet).
    
    Umożliwia:
    1. Przechowywanie milionów kroków i wektorów stanu w pamięci RAM w formacie kolumnowym.
    2. Bezkopiowe (Zero-Copy) przekazywanie macierzy tensorów do Numba JIT / NumPy.
    3. Natychmiastowy zrzut do formatu Apache Parquet dla fazy L3 Sleep Consolidation.
    """

    def __init__(self, state_dim: int = 32, capacity: int = 10000) -> None:
        self.state_dim = state_dim
        self._capacity = max(capacity, 1000)
        self._latent_array: np.ndarray = np.zeros((self._capacity, self.state_dim), dtype=np.float32)
        self._steps: list[int] = []
        self._timestamps: list[float] = []
        self._session_ids: list[str] = []
        self._action_names: list[str] = []
        self._rewards: list[float] = []
        self._uncertainties: list[float] = []
        self._metadata_jsons: list[str] = []
        self._cached_table: Any = None

    def _ensure_capacity(self, needed: int) -> None:
        if needed > self._capacity:
            new_cap = max(self._capacity * 2, needed + 1024)
            new_arr = np.zeros((new_cap, self.state_dim), dtype=np.float32)
            if self._capacity > 0:
                new_arr[:self._capacity] = self._latent_array[:self._capacity]
            self._latent_array = new_arr
            self._capacity = new_cap

    @property
    def _latent_vectors(self) -> list[list[float]]:
        n = len(self._steps)
        if n == 0:
            return []
        return self._latent_array[:n].tolist()

    @_latent_vectors.setter
    def _latent_vectors(self, val: list[list[float]] | np.ndarray) -> None:
        if isinstance(val, np.ndarray):
            self._ensure_capacity(len(val))
            self._latent_array[:len(val)] = val
        elif isinstance(val, list):
            self._ensure_capacity(len(val))
            if val:
                self._latent_array[:len(val)] = np.asarray(val, dtype=np.float32)

    def record_real_trajectory(
        self,
        steps: list[dict[str, Any]],
        session_id: str = "default_session",
    ) -> int:
        """V64: rejestruje REALNĄ trajektorię narzędzi z transcriptu agenta (source of truth).

        W odróżnieniu od ``append_transition`` (mentalna, JEPA, simulated_reward),
        sukces kroku jest obiektywny (exit code / flaga błędu z wyniku narzędzia).
        Realny stan zapisany jest w kolumnie ``metadata`` pod kluczem ``origin="real"``
        (zero nowych kolumn — schemat Parquet i round-trip pozostają nietknięte).
        Nagroda w ``_rewards`` = +1.0 (sukces) / -1.0 (porażka) — spójna z flagą.
        Zwraca liczbę dodanych kroków.
        """
        base_step = len(self._steps)
        now_base = time.time()
        n = len(steps)
        self._ensure_capacity(base_step + n)
        for i, step in enumerate(steps):
            name = step.get("tool_name") or step.get("action") or step.get("tool") or step.get("name") or "unknown_tool"
            success = bool(step.get("success", True))
            self._steps.append(base_step + i + 1)
            self._timestamps.append(now_base + i * 0.001)
            self._session_ids.append(session_id)
            self._latent_array[base_step + i, :] = 0.0
            self._action_names.append(str(name))
            self._rewards.append(1.0 if success else -1.0)
            self._uncertainties.append(0.0)
            self._metadata_jsons.append(
                json.dumps({
                    "origin": "real",
                    "success": success,
                    "parameters": dict(step.get("parameters") or {}),
                    "error": step.get("error"),
                })
            )
        self._cached_table = None
        return len(steps)

    def append_transition(
        self,
        transition: PredictedTransition,
        session_id: str = "default_session",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Dodaje pojedyncze przejście stanu do bufora."""
        idx = len(self._steps)
        self._ensure_capacity(idx + 1)
        self._steps.append(transition.predicted_state.step_index)
        self._timestamps.append(transition.predicted_state.timestamp)
        self._session_ids.append(session_id)
        self._latent_array[idx, :] = transition.predicted_state.vector
        self._action_names.append(transition.action.name)
        self._rewards.append(transition.simulated_reward)
        self._uncertainties.append(transition.uncertainty)
        self._metadata_jsons.append(json.dumps(metadata or {}))
        self._cached_table = None

    def append_transitions(
        self,
        transitions: list[PredictedTransition],
        session_id: str = "default_session",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Dodaje partię przejść stanu (batch) do bufora z pojedynczą serializacją metadanych."""
        if not transitions:
            return
        base = len(self._steps)
        self._ensure_capacity(base + len(transitions))
        meta_json = json.dumps(metadata or {})
        for i, t in enumerate(transitions):
            self._steps.append(t.predicted_state.step_index)
            self._timestamps.append(t.predicted_state.timestamp)
            self._session_ids.append(session_id)
            self._latent_array[base + i, :] = t.predicted_state.vector
            self._action_names.append(t.action.name)
            self._rewards.append(t.simulated_reward)
            self._uncertainties.append(t.uncertainty)
            self._metadata_jsons.append(meta_json)
        self._cached_table = None

    def to_arrow_table(self) -> Any:
        """Kompiluje dane w pamięci RAM do natywnej tabeli Apache Arrow."""
        if not HAS_PYARROW:
            return None
        if self._cached_table is not None:
            return self._cached_table
        n = len(self._steps)
        if n == 0:
            return None

        schema = pa.schema([
            ("step_id", pa.int64()),
            ("timestamp", pa.float64()),
            ("session_id", pa.string()),
            ("latent_vector", pa.list_(pa.float32())),
            ("action_name", pa.string()),
            ("reward", pa.float64()),
            ("uncertainty", pa.float64()),
            ("metadata", pa.string()),
        ])

        flat_vals = self._latent_array[:n].ravel()
        offsets = np.arange(0, (n + 1) * self.state_dim, self.state_dim, dtype=np.int32)
        arrow_latent = pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), pa.array(flat_vals, type=pa.float32()))

        data = [
            pa.array(self._steps, type=pa.int64()),
            pa.array(self._timestamps, type=pa.float64()),
            pa.array(self._session_ids, type=pa.string()),
            arrow_latent,
            pa.array(self._action_names, type=pa.string()),
            pa.array(self._rewards, type=pa.float64()),
            pa.array(self._uncertainties, type=pa.float64()),
            pa.array(self._metadata_jsons, type=pa.string()),
        ]

        self._cached_table = pa.Table.from_arrays(data, schema=schema)
        return self._cached_table

    def to_numpy_latent_matrix(self) -> np.ndarray:
        """
        Zwraca macierz stanów ukrytych (N x state_dim) bez nadmiarowych kopii bufora.
        """
        n = len(self._steps)
        if n == 0:
            return np.empty((0, self.state_dim), dtype=self._latent_array.dtype)
        return self._latent_array[:n]

    def to_zero_copy_tensor(self) -> np.ndarray:
        """
        Zwraca bezkopiowy widok tensorowy NumPy na kolumny pamięci RAM (Zero-Copy).
        """
        n = len(self._steps)
        if n == 0:
            return np.empty((0, self.state_dim), dtype=self._latent_array.dtype)
        return self._latent_array[:n]

    def dump_to_parquet(
        self,
        file_path: str | Path,
        compression: str = "zstd",
        compression_level: int | None = None,
    ) -> bool:
        """
        Zapisuje zawartość bufora do pliku Apache Parquet dla fazy L3 Sleep Consolidation.
        Obsługuje algorytmy kompresji 'zstd', 'snappy', 'gzip', 'none'.
        """
        if not HAS_PYARROW or not self._steps:
            return False

        table = self.to_arrow_table()
        if table is None:
            return False
        write_kwargs: dict[str, Any] = {"compression": compression}
        if compression_level is not None:
            write_kwargs["compression_level"] = compression_level

        pq.write_table(table, str(file_path), **write_kwargs)
        return True

    def stream_batches(
        self,
        batch_size: int = 256,
        zero_copy: bool = True,
    ) -> Iterator[dict[str, Any]]:
        """
        Strumieniuje dane partiami bez konieczności generowania pełnej tabeli Arrow upfront.
        """
        total = len(self._steps)
        if total == 0:
            return

        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            b_steps = np.asarray(self._steps[start_idx:end_idx], dtype=np.int64)
            b_timestamps = np.asarray(self._timestamps[start_idx:end_idx], dtype=np.float64)
            latent_mat = self._latent_array[start_idx:end_idx] if zero_copy else self._latent_array[start_idx:end_idx].copy()
            b_actions = self._action_names[start_idx:end_idx]
            b_rewards = np.asarray(self._rewards[start_idx:end_idx], dtype=np.float64)
            b_uncertainties = np.asarray(self._uncertainties[start_idx:end_idx], dtype=np.float64)

            yield {
                "step_ids": b_steps,
                "timestamps": b_timestamps,
                "latent_matrix": latent_mat,
                "action_names": b_actions,
                "rewards": b_rewards,
                "uncertainties": b_uncertainties,
                "batch_size": end_idx - start_idx,
            }

    @classmethod
    def stream_from_parquet(
        cls,
        file_path: str | Path,
        batch_size: int = 256,
        state_dim: int = 32,
    ) -> Iterator[dict[str, Any]]:
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
                col = batch["latent_vector"]
                if hasattr(col, "values"):
                    flat_vals = col.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
                elif hasattr(col, "combine_chunks"):
                    flat_vals = col.combine_chunks().values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
                else:
                    flat_vals = np.asarray(col.to_pylist(), dtype=np.float32).ravel()
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

    def get_compression_stats(self) -> dict[str, Any]:
        """
        Zwraca statystyki zajętości pamięci RAM oraz kompresji kolumnowej.
        """
        total = len(self._steps)
        raw_latent_bytes = total * self.state_dim * 4  # float32
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
    def load_from_parquet(cls, file_path: str | Path, state_dim: int = 32) -> ArrowTrajectoryBuffer:
        """Ładuje historię epizodu z pliku Parquet z odpornością na ewolucję schematu."""
        buf = cls(state_dim=state_dim)
        if not HAS_PYARROW:
            return buf

        table = pq.read_table(str(file_path))
        col_names = set(table.column_names)
        buf._steps = table["step_id"].to_pylist() if "step_id" in col_names else []
        buf._timestamps = table["timestamp"].to_pylist() if "timestamp" in col_names else []
        buf._session_ids = table["session_id"].to_pylist() if "session_id" in col_names else []
        if "latent_vector" in col_names:
            col = table["latent_vector"]
            try:
                if hasattr(col, "combine_chunks"):
                    flat = col.combine_chunks().values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
                elif hasattr(col, "values"):
                    flat = col.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
                else:
                    flat = np.asarray(col.to_pylist(), dtype=np.float32).ravel()
            except Exception:
                flat = np.asarray(col.to_pylist(), dtype=np.float32).ravel()
            n = len(buf._steps)
            buf._ensure_capacity(n)
            if flat.size == n * state_dim and n > 0:
                buf._latent_array[:n] = flat.reshape(n, state_dim)
            else:
                raw_list = col.to_pylist()
                buf._latent_vectors = raw_list
        buf._action_names = table["action_name"].to_pylist() if "action_name" in col_names else []
        buf._rewards = table["reward"].to_pylist() if "reward" in col_names else []
        buf._uncertainties = table["uncertainty"].to_pylist() if "uncertainty" in col_names else []
        buf._metadata_jsons = table["metadata"].to_pylist() if "metadata" in col_names else []
        return buf

    def __len__(self) -> int:
        return len(self._steps)

    @property
    def count(self) -> int:
        """Liczba zarejestrowanych kroków w buforze."""
        return len(self._steps)

    @property
    def max_step_id(self) -> int:
        """V64: największy step_id w buforze (watermark cyklu snu)."""
        return max(self._steps) if self._steps else 0

    @property
    def trajectories(self) -> list[dict[str, Any]]:
        """
        Zwraca listę słowników trajektorii w formacie zgodnym z SleepBaker.konsolidacja_sesji.
        Każdy element to:
        {"session_id": sid, "steps": [{"tool": act, "action": act}], "actions": [act], "success": True, "context": "general"}
        """
        if not self._steps:
            return []

        session_groups: dict[str, list[tuple[int, str, float, str]]] = {}
        for sid, step_id, act, rew, meta in zip(
            self._session_ids, self._steps, self._action_names, self._rewards, self._metadata_jsons, strict=False
        ):
            if sid not in session_groups:
                session_groups[sid] = []
            session_groups[sid].append((step_id, act, rew, meta))

        result: list[dict[str, Any]] = []
        for sid, items in session_groups.items():
            step_ids = [it[0] for it in items]
            actions = [it[1] for it in items]
            steps: list[dict[str, Any]] = [
                {"step_id": s_id, "tool_name": act, "tool": act, "action": act}
                for s_id, act in zip(step_ids, actions, strict=True)
            ]
            real_flags: list[bool] = []
            for i, (_s_id, _act, _rew, meta_json) in enumerate(items):
                try:
                    meta = json.loads(meta_json) if meta_json else {}
                except (json.JSONDecodeError, TypeError):
                    meta = {}
                if isinstance(meta, dict) and meta.get("origin") == "real" and "success" in meta:
                    real_flags.append(bool(meta["success"]))
                    steps[i]["parameters"] = meta.get("parameters") or {}
                    steps[i]["error"] = meta.get("error")
            if len(real_flags) == len(items) and real_flags:
                is_success = all(real_flags) or (real_flags[-1] and (sum(real_flags) / len(real_flags) >= 0.5))
            else:
                rewards = [it[2] for it in items]
                if not rewards:
                    is_success = True
                else:
                    mean_rew = sum(rewards) / len(rewards)
                    total_rew = sum(rewards)
                    last_rew = rewards[-1]
                    is_success = (mean_rew >= 0.0 and last_rew >= 0.0) or total_rew > 0.0
            result.append({
                "session_id": sid,
                "steps": steps,
                "actions": actions,
                "success": is_success,
                "context": "general",
                "step_ids": step_ids,
                "max_step_id": max(step_ids),
            })
        return result

    def get_session_trajectories(self, session_id: str) -> list[dict[str, Any]]:
        """Filtruje trajektorie po identyfikatorze sesji."""
        return [t for t in self.trajectories if t["session_id"] == session_id]

    def clear(self) -> None:
        self._steps.clear()
        self._timestamps.clear()
        self._session_ids.clear()
        self._latent_array.fill(0.0)
        self._action_names.clear()
        self._rewards.clear()
        self._uncertainties.clear()
        self._metadata_jsons.clear()
        self._cached_table = None
