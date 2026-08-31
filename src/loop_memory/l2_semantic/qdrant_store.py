from __future__ import annotations

import asyncio
import logging
import time
import zlib
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

try:
    from fastembed import TextEmbedding
    HAS_FASTEMBED = True
except ImportError:
    HAS_FASTEMBED = False

from loop_memory.models import MemoryRecord
from loop_memory.quantization import MIBQuantizer, QuantizationConfig, SIMDHamming


class FastEmbedEncoder:
    """Lokalny enkoder FastEmbed (ONNX) z deterministycznym fallbackiem."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dimension: int = 384, lazy_init: bool = True):
        self.dimension = dimension
        self.model_name = model_name
        self._model = None
        self._initialized = False
        if not lazy_init:
            self._init_model()

    def _init_model(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        if HAS_FASTEMBED:
            try:
                self._model = TextEmbedding(model_name=self.model_name)
            except Exception:
                self._model = None

    def encode(self, text: str) -> List[float]:
        self._init_model()
        if self._model is not None:
            try:
                embeddings = list(self._model.embed([text]))
                if embeddings:
                    emb = embeddings[0].tolist()
                    if len(emb) == self.dimension:
                        return emb
                    elif len(emb) > self.dimension:
                        return emb[:self.dimension]
                    else:
                        return emb + [0.0] * (self.dimension - len(emb))
            except Exception as exc:
                logger.debug("Fastembed encode failed, falling back to ngram projection: %s", exc)

        # Fallback n-gram projection
        text_clean = text.lower().strip()
        if not text_clean:
            return [0.0] * self.dimension
        rng = np.random.default_rng(42)
        proj = rng.standard_normal((1024, self.dimension))
        ngrams = [text_clean[i : i + 3] for i in range(max(1, len(text_clean) - 2))]
        vec = np.zeros(self.dimension)
        for ng in ngrams:
            idx = zlib.crc32(ng.encode("utf-8")) % 1024
            vec += proj[idx]
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec /= norm
        return vec.tolist()



class QdrantVectorStore:
    """
    L2: Produkcyjny Magazyn Wektorowy Qdrant (Embedded / On-Disk) + FastEmbed.
    
    Obsługuje tryb embedded (:memory: lub katalog na dysku ./data/qdrant)
    bez konieczności uruchamiania zewnętrznego kontenera Dockera.
    """

    def __init__(
        self,
        collection_name: str = "hermes_episodic_memory",
        location: str = ":memory:",
        dimension: int = 384,
        encoder: Optional[FastEmbedEncoder] = None,
    ):
        self.collection_name = collection_name
        self.dimension = dimension
        self.encoder = encoder or FastEmbedEncoder(dimension=dimension)
        self._lock = asyncio.Lock()
        self.quantizer = MIBQuantizer(QuantizationConfig(target_dim=dimension))
        self._quantized_cache: List[np.ndarray] = []

        if HAS_QDRANT:
            if location == ":memory:":
                self.client = QdrantClient(location=":memory:")
            else:
                self.client = QdrantClient(path=location)

            self._ensure_collection()
        else:
            self.client = None
            self._records: List[Dict[str, Any]] = []
            self._vectors: List[np.ndarray] = []

    def _ensure_collection(self) -> None:
        if self.client is None:
            return
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.dimension,
                    distance=qmodels.Distance.COSINE,
                ),
            )

    async def insert(self, record: MemoryRecord, raw_text: Optional[str] = None) -> str:
        text = raw_text or f"{record.effective_subject} {record.predicate} {record.object}"
        vec = record.vector if (record.vector and len(record.vector) == self.dimension) else self.encoder.encode(text)

        async with self._lock:
            doc_id = f"vec_{int(time.time() * 1000)}_{abs(hash(text)) % 100000}"
            payload = {
                "text": text,
                "record": record.model_dump(),
                "timestamp": record.timestamp,
                "confidence": record.confidence,
                "metadata": record.metadata,
            }

            if self.client is not None:
                point_id = int(time.time() * 1000) % 0x7FFFFFFF
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        qmodels.PointStruct(
                            id=point_id,
                            vector=vec,
                            payload=payload,
                        )
                    ],
                )
            else:
                self._records.append({"id": doc_id, **payload})
                self._vectors.append(np.array(vec, dtype=np.float64))

            qvec = self.quantizer.quantize(vec)
            self._quantized_cache.append(qvec.data)

            return doc_id

    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.1,
        *,
        enable_in_memory_prefilter: bool = False,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Wyszukuje semantycznie pasujące rekordy pamięci.
        
        Parametr enable_in_memory_prefilter uruchamia lokalny pre-filtr w pamięci RAM
        (oparty o MIBQuantizer i SIMDHamming na buforze `self._quantized_cache`),
        co przyspiesza selekcję kandydatów przed dokładnym porównaniem.
        Uwaga: Jest to filtr in-memory po stronie Pythona, a nie natywny indeks wektorowy Qdranta.
        """
        # Backward compatibility alias for deprecated enable_hamming_prefilter
        if "enable_hamming_prefilter" in kwargs:
            logger.warning("enable_hamming_prefilter is deprecated; use enable_in_memory_prefilter instead.")
            enable_in_memory_prefilter = kwargs.pop("enable_hamming_prefilter")

        q_vec = self.encoder.encode(query)

        async with self._lock:
            # In-memory Hamming prefilter if enabled and cache available
            if enable_in_memory_prefilter and self._quantized_cache and len(self._quantized_cache) == len(self._records or getattr(self, "_vectors", [])):
                q_quant = self.quantizer.quantize(q_vec)
                candidates = np.stack(self._quantized_cache)
                hamming_dists = SIMDHamming.batch_hamming(q_quant.data, candidates)
                # Prefilter top-K * 4 kandydatów o najmniejszym dystansie Hamminga
                prefilter_k = min(len(hamming_dists), max(top_k * 4, 20))
                best_indices = np.argsort(hamming_dists)[:prefilter_k]
            else:
                best_indices = None

            if self.client is not None:
                search_res = []
                try:
                    if hasattr(self.client, "search"):
                        search_res = self.client.search(
                            collection_name=self.collection_name,
                            query_vector=q_vec,
                            limit=top_k,
                            score_threshold=min_similarity,
                        )
                    elif hasattr(self.client, "query_points"):
                        qp = self.client.query_points(
                            collection_name=self.collection_name,
                            query=q_vec,
                            limit=top_k,
                            score_threshold=min_similarity,
                        )
                        search_res = getattr(qp, "points", [])
                except Exception:
                    search_res = []

                results = []
                for hit in search_res:
                    payload = hit.payload or {}
                    results.append({
                        "id": str(hit.id),
                        "text": payload.get("text", ""),
                        "record": payload.get("record", {}),
                        "score": float(hit.score),
                        "confidence": payload.get("confidence", 1.0),
                        "timestamp": payload.get("timestamp", 0.0),
                    })
                return results
            else:
                if not self._vectors:
                    return []

                if best_indices is not None:
                    candidate_indices = best_indices
                    matrix = np.stack([self._vectors[i] for i in candidate_indices])
                else:
                    candidate_indices = np.arange(len(self._vectors))
                    matrix = np.stack(self._vectors)

                qv = np.array(q_vec, dtype=np.float64)
                qv_norm = np.linalg.norm(qv) + 1e-9
                m_norm = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
                sims = (matrix @ qv) / (m_norm.flatten() * qv_norm)

                results = []
                sub_indices = np.argsort(-sims)
                for s_idx in sub_indices:
                    score = float(sims[s_idx])
                    if score < min_similarity:
                        continue
                    real_idx = candidate_indices[s_idx]
                    entry = self._records[real_idx]
                    results.append({
                        "id": entry["id"],
                        "text": entry["text"],
                        "record": entry["record"],
                        "score": score,
                        "confidence": entry["confidence"],
                        "timestamp": entry["timestamp"],
                    })
                    if len(results) >= top_k:
                        break
                return results

    async def count(self) -> int:
        async with self._lock:
            if self.client is not None:
                info = self.client.get_collection(self.collection_name)
                return info.points_count or 0
            return len(self._records)

