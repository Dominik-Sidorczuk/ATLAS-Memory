from __future__ import annotations

import json
import struct
from typing import Any, Dict, List, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class QuantizationConfig(BaseModel):
    """Konfiguracja kwantyzacji MIB."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    target_dim: int = Field(default=384, description="Docelowy wymiar wektora (384 | 1536)")
    compression_ratio: float = Field(default=32.0, description="Współczynnik kompresji (float32 do 1-bit)")


class QuantizedVector(BaseModel):
    """Zkwantyzowany binarnie wektor (upakowany w uint64)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: np.ndarray = Field(..., description="Tablica uint64 przechowująca binarne bity")
    original_dim: int = Field(..., description="Pierwotny wymiar wektora")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Opcjonalne metadane wektora")

    def to_bytes(self) -> bytes:
        """
        Wire format:
        magic(b"QV01", 4B) + original_dim(int32, 4B) + len(metadata, uint32, 4B) + metadata JSON utf-8 + data.tobytes()
        """
        meta_bytes = json.dumps(self.metadata).encode("utf-8")
        header = struct.pack("<4sII", b"QV01", int(self.original_dim), len(meta_bytes))
        return header + meta_bytes + self.data.tobytes()

    @classmethod
    def from_bytes(cls, blob: bytes) -> QuantizedVector:
        """
        Deserializuje zformatowany ciąg bajtów wire format z powrotem do QuantizedVector.
        """
        if len(blob) < 12:
            raise ValueError("Blob is too short to contain QuantizedVector header")
        magic, original_dim, meta_len = struct.unpack("<4sII", blob[:12])
        if magic != b"QV01":
            raise ValueError(f"Invalid magic bytes: {magic!r}, expected b'QV01'")

        meta_end = 12 + meta_len
        if len(blob) < meta_end:
            raise ValueError("Blob truncated in metadata section")
        meta_str = blob[12:meta_end].decode("utf-8")
        metadata = json.loads(meta_str) if meta_str else {}

        data_bytes = blob[meta_end:]
        data = np.frombuffer(data_bytes, dtype=np.uint64).copy()
        return cls(data=data, original_dim=int(original_dim), metadata=metadata)


class MIBQuantizer:
    """
    Minimum Information-Loss Binarization (MIB) Quantizer.
    
    Binarizuje wektory embeddingów używając mediany jako progu (float > median = 1, <= median = 0),
    a następnie upakowuje bity do uint64 (64 bity na uint64).
    384-dim -> 6 uint64 (48 bajtów, kompresja 32x względem float32 1536B)
    1536-dim -> 24 uint64 (192 bajty, kompresja 32x względem float32 6144B)
    """

    def __init__(self, config: QuantizationConfig | None = None) -> None:
        self.config = config or QuantizationConfig()

    def quantize(self, embedding: Union[np.ndarray, List[float]]) -> QuantizedVector:
        """Kwantyzuje pojedynczy wektor embeddingu float32 do QuantizedVector (uint64 array)."""
        if isinstance(embedding, list):
            arr = np.array(embedding, dtype=np.float32)
        else:
            arr = embedding.astype(np.float32)

        orig_dim = arr.shape[0]
        if orig_dim == 0:
            raise NotImplementedError("TODO: puste wektory nie sa obslugiwane przez MIBQuantizer")

        # Thresholding: MIB używa mediany wektora
        median_val = float(np.median(arr))
        bits = (arr > median_val).astype(np.uint8)

        # Packing do uint64
        num_u64 = (orig_dim + 63) // 64
        packed = np.zeros(num_u64, dtype=np.uint64)

        for i in range(num_u64):
            chunk = bits[i * 64 : min((i + 1) * 64, orig_dim)]
            val = np.uint64(0)
            for bit_idx, bit in enumerate(chunk):
                if bit:
                    val |= np.uint64(1) << np.uint64(bit_idx)
            packed[i] = val

        return QuantizedVector(
            data=packed,
            original_dim=orig_dim,
            metadata={"median_threshold": median_val},
        )

    def dequantize(self, qvec: QuantizedVector) -> np.ndarray:
        """Dekwantyzuje binarne wektory z powrotem do aproksymacji float32 (0.0 / 1.0)."""
        dim = qvec.original_dim
        out = np.zeros(dim, dtype=np.float32)

        for i in range(dim):
            u64_idx = i // 64
            bit_idx = i % 64
            val = qvec.data[u64_idx]
            if (val >> np.uint64(bit_idx)) & np.uint64(1):
                out[i] = 1.0
            else:
                out[i] = 0.0

        return out

    def compress(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Batch quantization zwracająca bezpośrednio surową tablicę 2D uint64.
        embeddings: np.ndarray shape (batch_size, dim)
        returns: np.ndarray shape (batch_size, num_uint64) dtype uint64
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        batch_size, dim = embeddings.shape
        num_u64 = (dim + 63) // 64
        out = np.zeros((batch_size, num_u64), dtype=np.uint64)

        for b in range(batch_size):
            qv = self.quantize(embeddings[b])
            out[b] = qv.data

        return out
