"""F6: Kontrakty Protocol (typing.Protocol) dla warstw fizycznych ATLAS oraz adaptery DI na brzegu."""
from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class KVStoreProtocol(Protocol):
    """Kontrakt magazynu KV (zgodny z sygnaturami VerifiedKVStore w kv_store.py)."""

    def get_sync(self, key: str) -> Optional[Dict[str, Any]]: ...
    def set_sync(
        self,
        key: str,
        value: Any,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        reason: str = "sync_update",
    ) -> str: ...
    async def get_states(self, keys: List[str]) -> Dict[str, Dict[str, Any]]: ...
    async def filter_existing_keys(self, keys: List[str]) -> Set[str]: ...


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Kontrakt magazynu wektorowego (zgodny z QdrantVectorStore w qdrant_store.py: search bez session_id)."""

    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.1,
        query_filter: Optional[Any] = None,
    ) -> List[Dict[str, Any]]: ...
    def insert_sync(self, record: Any, upsert_key: Optional[str] = None) -> str: ...


@runtime_checkable
class GraphStoreProtocol(Protocol):
    """Kontrakt grafu przyczynowego (zgodny z KuzuGraphClient w kuzu_graph.py)."""

    async def get_node_relations(self, node_name: str, max_depth: int = 1) -> List[Dict[str, Any]]: ...


@runtime_checkable
class DualNodeGraphStoreProtocol(GraphStoreProtocol, Protocol):
    """Kontrakt grafu Dual-Node HippoRAG 2 (Passage + CONTAINS + SYNONYM + rollback)."""

    nx_graph: Any

    def add_passage(
        self,
        passage_id: str,
        source_key: str,
        text: str,
        ingest_at: Optional[str] = None,
    ) -> None: ...
    def create_entity(self, name: str, entity_type: str = "entity") -> None: ...
    def add_contains(
        self,
        passage_id: str,
        entity_name: str,
        weight: float = 1.0,
        persist_to_kuzu: bool = True,
    ) -> None: ...
    def add_synonym(self, entity_a: str, entity_b: str, sim: float) -> None: ...
    def rollback_dual_node_record(
        self,
        passage_id: str,
        created_entities: Optional[List[str]] = None,
    ) -> None: ...
    def has_passage(self, passage_id: str) -> bool: ...
    def index_dual_node_passage(
        self,
        source_key: str,
        text: str,
        ingest_at: Optional[str] = None,
        gist: Optional[str] = None,
        explicit_entities: Optional[List[str]] = None,
        explicit_synonyms: Optional[List[Tuple[str, str, float]]] = None,
        speech_act: Optional[str] = None,  # V152 (FILAR I): opcjonalny stamp aktu mowy
    ) -> int: ...
    def get_dual_node_coverage(self) -> float: ...


@runtime_checkable
class HotBufferProtocol(Protocol):
    """Kontrakt bufora pamięci roboczej L1 (zgodny z HotBufferPolicy w hot_buffer.py)."""

    def get_active(self, session_id: Optional[str] = None) -> List[Any]: ...
    def update(
        self,
        key: str,
        value: Any,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        source_type: str = "user_explicit",
    ) -> Any: ...


def resolve_retrieval_stores(
    engine: Optional[Any] = None,
    orchestrator: Optional[Any] = None,
    hot_buffer: Optional[Any] = None,
    kv_store: Optional[Any] = None,
    vector_store: Optional[Any] = None,
    graph_store: Optional[Any] = None,
) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
    """Rozwiązuje referencje magazynów na brzegu DI (poza gorącą pętlą candidates.py)."""
    hb = hot_buffer or getattr(engine, "hot_buffer", None)
    kv = kv_store or getattr(engine, "kv", None) or getattr(engine, "kv_store", None)
    if kv is None and orchestrator is not None:
        orch_eng = getattr(orchestrator, "engine", None)
        kv = getattr(orchestrator, "kv", None) or (
            orch_eng and (getattr(orch_eng, "kv", None) or getattr(orch_eng, "kv_store", None))
        )
    vec = vector_store or getattr(engine, "vector_store", None) or getattr(engine, "vector", None)
    gs = graph_store or getattr(engine, "graph", None)
    return hb, kv, vec, gs


def resolve_calibrated_threshold(
    calibration_engine: Optional[Any],
    enable_calibrated_thresholds: bool,
    is_symbolic: bool,
) -> float:
    """Pobiera próg z CalibrationEngine lub zwraca kanoniczny próg domyślny."""
    default_thr = 0.70 if is_symbolic else 0.35
    if enable_calibrated_thresholds and calibration_engine is not None and hasattr(calibration_engine, "get_threshold"):
        return float(
            calibration_engine.get_threshold(
                query_type="symbolic" if is_symbolic else "prose",
                default=default_thr,
            )
        )
    return default_thr


def fetch_hot_buffer_items(hb: Optional[Any], session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Zwraca znormalizowane słowniki elementów bufora roboczego."""
    if hb is None:
        return []
    raw_items = hb.get_active(session_id=session_id) if hasattr(hb, "get_active") else list(hb)
    normalized: List[Dict[str, Any]] = []
    for h in raw_items:
        if isinstance(h, dict):
            normalized.append({
                "key": str(h.get("key", "")),
                "value": str(h.get("value", "")),
                "metadata": dict(h.get("metadata") or {}),
                "confidence": float(h.get("confidence", 1.0)),
                "source_type": str(h.get("source_type", "user_explicit")),
                "timestamp": float(h.get("timestamp", 0.0)),
            })
        else:
            normalized.append({
                "key": str(getattr(h, "key", "")),
                "value": str(getattr(h, "value", "")),
                "metadata": dict(getattr(h, "metadata", {}) or {}),
                "confidence": float(getattr(h, "confidence", 1.0)),
                "source_type": str(getattr(h, "source_type", "user_explicit")),
                "timestamp": float(getattr(h, "timestamp", 0.0)),
            })
    return normalized


def search_kv_candidates(
    kv_store: Optional[Any],
    query: str,
    words: List[str],
    stems: List[str],
    limit: int = 100,
) -> Dict[str, Any]:
    """Jednokrotne pobranie kandydatów z warstwy KV (FTS5 / paged fallback)."""
    if kv_store is None or not query:
        return {}
    raw_kv: Optional[Dict[str, Any]] = None
    if hasattr(kv_store, "search_candidates_sync"):
        try:
            res_search = kv_store.search_candidates_sync(query=query, limit=limit, words=words, stems=stems)
            if isinstance(res_search, dict) and res_search:
                raw_kv = res_search
        except Exception as exc:
            logger.debug("search_candidates_sync error: %s", exc)
    if not raw_kv and hasattr(kv_store, "get_all_sync"):
        try:
            res_all = kv_store.get_all_sync(paged=True, limit=limit)
        except TypeError:
            res_all = kv_store.get_all_sync()
        if isinstance(res_all, dict):
            raw_kv = res_all
    elif not raw_kv and hasattr(kv_store, "get_all"):
        res_all = kv_store.get_all()
        if isinstance(res_all, dict):
            raw_kv = res_all
    return raw_kv if isinstance(raw_kv, dict) else {}


async def search_vector_candidates(
    vec_store: Optional[Any],
    query: str,
    limit: int,
    session_id: Optional[str] = None,
) -> List[Any]:
    """Jednokrotne zapytanie do magazynu wektorowego (hybrydowe lub gęste)."""
    if vec_store is None or not query:
        return []
    if hasattr(vec_store, "search_hybrid_async") and inspect.iscoroutinefunction(vec_store.search_hybrid_async):
        return (await vec_store.search_hybrid_async(query=query, limit=limit, session_id=session_id)) or []
    if hasattr(vec_store, "search"):
        res = vec_store.search(query, top_k=limit)
        if inspect.isawaitable(res):
            res = await res
        return res or []
    return []


async def fetch_graph_relations_batch(
    gs: Optional[Any],
    words: List[str],
    query: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Pobiera relacje grafowe dla słów kluczowych lub encji zapytania (V122: canonical + lematy)."""
    if gs is None or not hasattr(gs, "get_node_relations"):
        return []
    import asyncio

    nx_graph = getattr(gs, "nx_graph", gs)
    target_nodes: List[str] = []
    if nx_graph is not None and hasattr(nx_graph, "resolve_query_entities"):
        target_nodes = nx_graph.resolve_query_entities(words or [], query=query or "")

    if not target_nodes and words:
        target_nodes = words[:3]

    if not target_nodes:
        return []

    tasks = [asyncio.create_task(gs.get_node_relations(w)) for w in target_nodes[:5]]
    if not tasks:
        return []
    edge_lists = await asyncio.gather(*tasks, return_exceptions=True)
    edges: List[Dict[str, Any]] = []
    for el in edge_lists:
        if isinstance(el, list):
            edges.extend(el)
    return edges


def fetch_graph_ppr_candidates(
    gs: Optional[Any],
    query: str,
    limit: int = 15,
    budget_ms: float = 35.0,
) -> List[Dict[str, Any]]:
    """Lazy adapter delegating to cognitive.retrieval.ppr to eliminate Layer 0 coupling."""
    from atlas_memory.cognitive.retrieval.ppr import fetch_graph_ppr_candidates as _impl

    return _impl(gs, query, limit=limit, budget_ms=budget_ms)


def normalize_vector_hit(hit: Any) -> Dict[str, Any]:
    """Bezpieczna normalizacja trafienia wektorowego odporna na None i brak atrybutów."""
    if isinstance(hit, dict):
        rec = hit.get("record") or {}
        meta = dict(rec.get("metadata") or hit.get("metadata") or {})
        score = float(hit.get("score", 0.0) or 0.0)
        key = str(
            meta.get("kv_key")
            or hit.get("kv_key")
            or hit.get("upsert_key")
            or hit.get("key")
            or rec.get("subject")
            or hit.get("id")
            or ""
        )
        val = str(rec.get("object") or hit.get("text") or "")
        via_sparse = bool(hit.get("via_sparse", False))
        via_dense = bool(hit.get("via_dense", False))
        raw_sparse = hit.get("sparse_score")
        sparse_score = float(raw_sparse) if raw_sparse is not None else None
        vector_emb = hit.get("vector") or (rec.get("vector") if isinstance(rec, dict) else None)
    else:
        rec = getattr(hit, "payload", {}) or {}
        meta = dict((rec.get("metadata") if isinstance(rec, dict) else None) or getattr(hit, "metadata", {}) or {})
        score = float(getattr(hit, "score", 0.0) or 0.0)
        key = str(
            meta.get("kv_key")
            or getattr(hit, "kv_key", "")
            or getattr(hit, "upsert_key", "")
            or getattr(hit, "key", "")
            or (rec.get("subject") if isinstance(rec, dict) else "")
            or getattr(hit, "id", "")
        )
        val = str((rec.get("object") if isinstance(rec, dict) else "") or getattr(hit, "text", ""))
        via_sparse = bool(getattr(hit, "via_sparse", False))
        via_dense = bool(getattr(hit, "via_dense", False))
        raw_sparse = getattr(hit, "sparse_score", None) or (rec.get("sparse_score") if isinstance(rec, dict) else None)
        sparse_score = float(raw_sparse) if raw_sparse is not None else None
        vector_emb = getattr(hit, "vector", None) or (rec.get("vector") if isinstance(rec, dict) else None)

    is_sup = bool(
        meta.get("is_superseded")
        or meta.get("superseded")
        or (isinstance(hit, dict) and (hit.get("is_superseded") or hit.get("superseded")))
        or getattr(hit, "is_superseded", False)
    )
    tombstone_pending = bool(
        meta.get("vector_tombstone_pending")
        or (isinstance(hit, dict) and hit.get("vector_tombstone_pending"))
        or getattr(hit, "vector_tombstone_pending", False)
    )

    return {
        "key": key[3:] if key.startswith("kv:") else key,
        "value": val,
        "metadata": meta,
        "score": score,
        "via_sparse": via_sparse,
        "via_dense": via_dense,
        "sparse_score": sparse_score,
        "is_superseded": is_sup,
        "vector_tombstone_pending": tombstone_pending,
        "vector": vector_emb,
    }


async def verify_kv_keys_batch(eff_kv: Optional[Any], missing_keys: List[str]) -> Set[str]:
    """Weryfikacja wsadowa z priorytetem metod jednostkowych przed pełnym skanem bazy.
    Odrzuca klucze oznaczone is_superseded lub vector_tombstone_pending.
    """
    if not missing_keys:
        return set()
    if eff_kv is None:
        return set(missing_keys)

    def _is_valid_kv_record(rec: Any) -> bool:
        if rec is None:
            return False
        if isinstance(rec, dict):
            meta = rec.get("metadata")
            if isinstance(meta, dict):
                if meta.get("is_superseded") or meta.get("superseded") or meta.get("vector_tombstone_pending"):
                    return False
            if rec.get("is_superseded") or rec.get("superseded") or rec.get("vector_tombstone_pending"):
                return False
        else:
            meta = getattr(rec, "metadata", None)
            if isinstance(meta, dict):
                if meta.get("is_superseded") or meta.get("superseded") or meta.get("vector_tombstone_pending"):
                    return False
            if getattr(rec, "is_superseded", None) is True or getattr(rec, "vector_tombstone_pending", None) is True:
                return False
        return True

    if hasattr(eff_kv, "filter_existing_keys"):
        res = eff_kv.filter_existing_keys(missing_keys)
        existing = (await res) if inspect.isawaitable(res) else res
        if not isinstance(existing, (set, list, tuple)):
            raise TypeError(f"filter_existing_keys returned unexpected type: {type(existing)}")
        if hasattr(eff_kv, "get_states"):
            st_res = eff_kv.get_states(list(existing))
            states = (await st_res) if inspect.isawaitable(st_res) else st_res
            if isinstance(states, dict):
                return {k for k, v in states.items() if _is_valid_kv_record(v)}
        return set(existing)
    if hasattr(eff_kv, "get_states"):
        st_res = eff_kv.get_states(missing_keys)
        states = (await st_res) if inspect.isawaitable(st_res) else st_res
        if isinstance(states, dict):
            return {k for k, v in states.items() if _is_valid_kv_record(v)}
        if not isinstance(states, (set, list, tuple)):
            raise TypeError(f"get_states returned unexpected type: {type(states)}")
        return set(states.keys())
    if hasattr(eff_kv, "get_sync"):
        verified: Set[str] = set()
        for k in missing_keys:
            rec = eff_kv.get_sync(k)
            if _is_valid_kv_record(rec):
                verified.add(k)
        return verified
    if hasattr(eff_kv, "get_state"):
        verified = set()
        for k in missing_keys:
            res = eff_kv.get_state(k)
            if inspect.isawaitable(res):
                res = await res
            if _is_valid_kv_record(res):
                verified.add(k)
        return verified
    if hasattr(eff_kv, "get_all_sync"):
        all_kv = eff_kv.get_all_sync()
        if isinstance(all_kv, dict):
            return {k for k in missing_keys if k in all_kv and _is_valid_kv_record(all_kv[k])}
    return set(missing_keys)


@runtime_checkable
class UDSClientProtocol(Protocol):
    """Protocol for Unix Domain Socket JSON-RPC 2.0 client transport."""

    async def call(self, method: str, params: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None) -> Any: ...
    async def close(self) -> None: ...
