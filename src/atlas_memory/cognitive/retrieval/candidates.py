"""V82/F3/F6 Candidates: zoptymalizowany zbieracz kandydatów z KV, Qdrant i Kùzu (zero redundant queries, zero reflection)."""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Set

from atlas_memory.cognitive.calibration_engine import is_symbolic_or_identifier
from atlas_memory.cognitive.retrieval.ppr import (
    fetch_graph_ppr_candidates,
    is_hipporag_enabled,
    record_channel_query,
)
from atlas_memory.core.linguistics import (
    QUERY_META_STOPSTEMS,
    QUERY_META_STOPWORDS,
    _bilingual_stem,
)
from atlas_memory.core.protocols import (
    GraphStoreProtocol,
    HotBufferProtocol,
    KVStoreProtocol,
    VectorStoreProtocol,
    fetch_graph_relations_batch,
    fetch_hot_buffer_items,
    normalize_vector_hit,
    resolve_calibrated_threshold,
    resolve_retrieval_stores,
    search_kv_candidates,
    search_vector_candidates,
    verify_kv_keys_batch,
)
from atlas_memory.l1_working.hot_buffer import extract_effective_session, is_conversational_churn, is_session_accessible

logger = logging.getLogger(__name__)
PROBE_PREFIXES = ("agi:test:", "test:agi:")


async def gather_candidates(
    query: str,
    session_id: Optional[str] = None,
    limit: int = 50,
    engine: Optional[Any] = None,
    orchestrator: Optional[Any] = None,
    hot_buffer: Optional[HotBufferProtocol] = None,
    graph_store: Optional[GraphStoreProtocol] = None,
    calibration_engine: Optional[Any] = None,
    enable_calibrated_thresholds: bool = True,
    kv_store: Optional[KVStoreProtocol] = None,
    vector_store: Optional[VectorStoreProtocol] = None,
    fallback_on_ppr_error: bool = False,
) -> List[Dict[str, Any]]:
    """Zbiera kandydatów wyłącznie z warstw trwałych (KV, Qdrant, Kùzu) bez powtórzeń I/O i bez refleksji."""
    q_clean = query.strip()
    raw_words = [w.lower().strip(".,!?:;\"'()[]{}") for w in q_clean.split() if len(w) >= 3]
    words = [
        w for w in raw_words
        if w not in QUERY_META_STOPWORDS and _bilingual_stem(w) not in QUERY_META_STOPSTEMS
    ] or raw_words
    stems = [w[:6] for w in words if len(w) >= 4]
    bilingual_stems = [_bilingual_stem(w) for w in words if len(w) >= 3]
    candidates: Dict[str, Dict[str, Any]] = {}
    seen_keys: Set[str] = set()

    hb, eff_kv, vec_store, gs = resolve_retrieval_stores(
        engine=engine,
        orchestrator=orchestrator,
        hot_buffer=hot_buffer,
        kv_store=kv_store,
        vector_store=vector_store,
        graph_store=graph_store,
    )

    # 1. Hot buffer: robocza pamięć bieżącej sesji (newest-first, RAM)
    for h in reversed(fetch_hot_buffer_items(hb, session_id=session_id)):
        k_h, v_h = h["key"], h["value"]
        m_h, c_h = h["metadata"], h["confidence"]
        st_h, ts_h = h["source_type"], h["timestamp"]
        if k_h and k_h not in candidates and not _is_filtered(k_h, v_h, m_h, session_id):
            candidates[k_h] = _build_record(
                k_h,
                v_h,
                "hot_buffer",
                {
                    "metadata": m_h,
                    "confidence": c_h,
                    "source_type": st_h,
                    "predicate": "active_working_memory",
                    "timestamp": ts_h,
                },
                0.0,
            )
            seen_keys.add(k_h)

    # 2. KV Store (FTS5 + exact)
    raw_kv: Dict[str, Any] = {}
    if eff_kv is not None and q_clean:
        try:
            raw_kv = search_kv_candidates(eff_kv, query=q_clean, words=words, stems=stems, limit=100)
            for k, it in raw_kv.items():
                val = str(it.get("value", "") if isinstance(it, dict) else it)
                if not _is_filtered(
                    k, val, it.get("metadata") if isinstance(it, dict) else None, session_id
                ):
                    is_exact = k.lower() == q_clean.lower()
                    if k in candidates:
                        if is_exact:
                            candidates[k]["importance_score"] = 1.0
                            candidates[k]["exact_kv_hit"] = True
                    else:
                        rec = _build_record(
                            k,
                            val,
                            "kv",
                            it if isinstance(it, dict) else {},
                            0.0,
                        )
                        if is_exact:
                            rec["importance_score"] = 1.0
                            rec["exact_kv_hit"] = True
                        candidates[k] = rec
                        seen_keys.add(k)
        except Exception as exc:
            logger.debug("KV candidate error: %s", exc)

    # 2b. Inicjalizacja kluczy weryfikacji KV
    unverified_keys: Set[str] = set()
    verified_keys: Set[str] = set(raw_kv.keys())
    verified_keys.update(seen_keys)

    # 3. Współbieżne zapytania: Qdrant Vector Store + Kùzu Graph Store (dokładnie 1 asyncio.gather)
    min_vec = resolve_calibrated_threshold(
        calibration_engine=calibration_engine,
        enable_calibrated_thresholds=enable_calibrated_thresholds,
        is_symbolic=is_symbolic_or_identifier(q_clean),
    )

    hits: List[Any] = []
    graph_edges: List[Dict[str, Any]] = []
    use_hipporag = bool(q_clean and is_hipporag_enabled())
    graph_fetched = False
    ppr_task_res: Optional[Any] = None
    if q_clean and (vec_store is not None or (gs is not None and (words or use_hipporag))):
        vec_coro = search_vector_candidates(vec_store, query=q_clean, limit=limit, session_id=session_id)
        if use_hipporag and gs is not None:
            loop = asyncio.get_running_loop()
            ppr_coro = loop.run_in_executor(None, fetch_graph_ppr_candidates, gs, q_clean, limit)
            gather_res = await asyncio.gather(vec_coro, ppr_coro, return_exceptions=True)
            ppr_task_res = gather_res[1]
        elif gs is not None and words and not use_hipporag:
            graph_coro = fetch_graph_relations_batch(gs, words=words, query=q_clean)
            gather_res = await asyncio.gather(vec_coro, graph_coro, return_exceptions=True)
            graph_fetched = True
        else:
            gather_res = await asyncio.gather(vec_coro, return_exceptions=True)
        if isinstance(gather_res[0], Exception):
            logger.debug("Vector candidate search error: %s", gather_res[0])
        elif isinstance(gather_res[0], list):
            hits = gather_res[0]
        if not use_hipporag and len(gather_res) > 1:
            if isinstance(gather_res[1], Exception):
                logger.debug("Graph gather error: %s", gather_res[1])
            elif isinstance(gather_res[1], list):
                graph_edges = gather_res[1]

    raw_vector_hits = []
    for raw_hit in hits:
        norm = normalize_vector_hit(raw_hit)
        k_clean = norm["key"]
        v = norm["value"]
        meta = norm["metadata"]
        score = norm["score"]

        cand_text = f"{k_clean} {v}".lower()
        has_lex_hint = any(w in cand_text for w in words if len(w) >= 3) or any(s in cand_text for s in stems if len(s) >= 4)
        if not has_lex_hint and bilingual_stems:
            cand_tokens = [w for w in re.findall(r"\w+", cand_text) if len(w) >= 3]
            cand_stems = [_bilingual_stem(tok) for tok in cand_tokens]
            has_lex_hint = bool(set(cand_stems) & set(bilingual_stems))

        is_sparse_hit = norm["via_sparse"]
        is_daemon_set = str(meta.get("origin") or "") == "daemon_set"
        if is_daemon_set and not has_lex_hint:
            meta["is_daemon_no_lex"] = True
        if not is_sparse_hit and score < min_vec:
            continue
        if norm.get("is_superseded") or norm.get("vector_tombstone_pending") or meta.get("is_superseded") or meta.get("vector_tombstone_pending"):
            continue
        if is_sparse_hit:
            meta["via_sparse"] = True
        if norm["via_dense"]:
            meta["via_dense"] = True
        if norm["sparse_score"] is not None:
            meta["sparse_score"] = norm["sparse_score"]
        if not _is_filtered(k_clean, v, meta, session_id) and v:
            raw_vector_hits.append((k_clean, v, meta, score, norm.get("vector")))
            if k_clean not in verified_keys:
                unverified_keys.add(k_clean)

    # 4. Jedno wsadowe sprawdzenie kluczy w KV (Single Batch Check)
    if unverified_keys:
        try:
            existing_keys = await verify_kv_keys_batch(eff_kv, list(unverified_keys))
            verified_keys.update(existing_keys)
        except Exception as kv_err:
            logger.warning("KV batch verification failed conservatively: %s", kv_err)
            # Graceful degradation: do not purge valid vector hits on KV store failure
            verified_keys.update(unverified_keys)

    for k_clean, v, meta, score, vec in raw_vector_hits:
        if k_clean in verified_keys:
            if k_clean in candidates:
                if vec is not None and "vector" not in candidates[k_clean]:
                    candidates[k_clean]["vector"] = vec
                if score > float(candidates[k_clean].get("vector_score", 0.0)):
                    candidates[k_clean]["vector_score"] = score
                    if candidates[k_clean].get("importance_score", 0.0) < 1.0:
                        candidates[k_clean]["importance_score"] = max(
                            candidates[k_clean]["importance_score"], score
                        )
            else:
                candidates[k_clean] = _build_record(k_clean, v, "vector", {"metadata": meta, "vector": vec}, score)
                seen_keys.add(k_clean)

    # 5. Dołączenie relacji grafowych z Kùzu + HippoRAG 2 PPR (V104.5 FIX-2, FIX-3, FIX-4)
    qid_tag = q_clean[:48] if q_clean else "empty"
    if any(c.get("source") == "kv" for c in candidates.values()):
        record_channel_query("kv", qid_tag)
    if any(c.get("source") == "vector" for c in candidates.values()):
        record_channel_query("vector_dense", qid_tag)
    if any(c.get("source") == "hot_buffer" for c in candidates.values()):
        record_channel_query("hot_buffer", qid_tag)

    ppr_failed = False
    if q_clean and is_hipporag_enabled():
        record_channel_query("graph_ppr", qid_tag)
        try:
            if ppr_task_res is not None:
                if isinstance(ppr_task_res, Exception):
                    raise ppr_task_res
                ppr_cands = ppr_task_res
            else:
                ppr_cands = fetch_graph_ppr_candidates(gs, q_clean, limit=limit)
            _merge_ppr_candidates(candidates, ppr_cands, seen_keys, session_id)
        except Exception as exc:
            if not fallback_on_ppr_error:
                raise
            logger.warning("HippoRAG 2 PPR failed, falling back to exact/stemmed graph: %s", exc)
            ppr_failed = True

    if not is_hipporag_enabled() or ppr_failed:
        if not graph_fetched and gs is not None and (words or q_clean):
            try:
                graph_edges = await fetch_graph_relations_batch(gs, words=words, query=q_clean)
                graph_fetched = True
            except Exception as exc:
                logger.debug("Graph fallback gather error: %s", exc)
        if graph_edges:
            record_channel_query("graph", qid_tag)
        _merge_graph_edges(candidates, graph_edges, seen_keys, session_id)

    # V151.1 (F-031, 1b): single-source-of-truth gate — rekordy kanału grafowego
    # (relacje LUB PPR) weryfikowane przeciw KV: baza klucza z is_superseded=True
    # → odrzuć, zanim trafią do poola. Qdrant/Kùzu to projekcje, nie prawda.
    candidates = await _graph_channel_kv_truth_gate(candidates, eff_kv)

    return list(candidates.values())


def _merge_ppr_candidates(
    candidates: Dict[str, Dict[str, Any]],
    ppr_cands: List[Dict[str, Any]],
    seen_keys: Set[str],
    session_id: Optional[str],
) -> None:
    for p_cand in ppr_cands:
        k_p = str(p_cand.get("key") or p_cand.get("subject") or "")
        val_p = str(p_cand.get("object") or p_cand.get("value") or "")
        gs_p = float(p_cand.get("graph_score", 0.0) or 0.0)
        if not k_p or not val_p or _is_filtered(k_p, val_p, p_cand.get("metadata"), session_id):
            continue
        if k_p in candidates:
            candidates[k_p]["graph_score"] = max(float(candidates[k_p].get("graph_score", 0.0) or 0.0), gs_p)
            candidates[k_p]["graph_hit"] = True
            p_sev = (p_cand.get("metadata") or {}).get("cpof_severity") or p_cand.get("cpof_severity")
            if p_sev is not None:
                candidates[k_p].setdefault("metadata", {})["cpof_severity"] = float(p_sev)
        else:
            rec_p = _build_record(k_p, val_p, "graph", p_cand, 0.0)
            rec_p["graph_score"] = gs_p
            rec_p["graph_hit"] = True
            candidates[k_p] = rec_p
            seen_keys.add(k_p)


def _merge_graph_edges(
    candidates: Dict[str, Dict[str, Any]],
    graph_edges: List[Dict[str, Any]],
    seen_keys: Set[str],
    session_id: Optional[str],
) -> None:
    for edge in graph_edges:
        k = str(edge.get("subject", ""))
        val = str(edge.get("object", ""))
        if not _is_filtered(k, val, edge, session_id) and k not in candidates and val:
            rec_g = _build_record(k, val, "graph", edge, 0.0)
            rec_g["graph_score"] = float(edge.get("confidence", 0.5) or 0.5)
            rec_g["graph_hit"] = True
            candidates[k] = rec_g
            seen_keys.add(k)


# V151.1 (F-031, 1b): bramka read-path — KV = źródło prawdy, widoki projekcyjne.
# Rekord kanału grafowego (source=graph: fallback relacji LUB hipporag PPR) nie
# dziedziczy flagi is_superseded z KV, więc TU, przed wpuszczeniem do poola,
# weryfikujemy każdy kandydat przeciw KV po kanonicznych kluczach bazowych.
# To NIE czarna lista: rekonstrukcja klucza przez EpistemicCoordinate + usuwanie
# sufiksów transportowych (':state_value' → baza + znormalizowany slot 'state').

_KV_NAMESPACE_SEGMENTS = frozenset({"fact", "kv", "rule", "pref", "sop"})
_STATE_VALUE_SLOT_SUFFIX = "_value"


def _derive_kv_probe_keys(raw_key: Any) -> List[str]:
    """Kanoniczne klucze zapytań do KV dla surowego klucza widoku (Qdrant/Kùzu).

    Zdejmuje prefiks transportowy 'kv:' (te same canonicalizacja co w
    `normalize_upsert_key`), dodaje warianty bez namespace'u ('fact:') oraz
    bazę klucza bez sufiksu slotu ':state_value' (z kanonicznym slotem 'state').
    Zwraca listę bez duplikatów, w kolejności od najbardziej kanonicznego.
    """
    from atlas_memory.l2_semantic.qdrant_store import normalize_upsert_key

    canon = normalize_upsert_key(str(raw_key or ""))
    if not canon:
        return []

    probes: List[str] = []
    seen: Set[str] = set()

    def add(key: str) -> None:
        k = str(key or "").strip()
        if k and k not in seen:
            seen.add(k)
            probes.append(k)

    parts = [p for p in canon.split(":") if p]
    add(canon)

    # wariant bez namespace'u / z namespace'em 'fact:' (kanon zapisu faktów L2)
    if parts and parts[0] in _KV_NAMESPACE_SEGMENTS:
        add(":".join(parts[1:]))
        add(f"fact:{':'.join(parts[1:])}")
    elif parts:
        add(f"fact:{canon}")

    # baza klucza: ogon ':X_value' → ':X' (slot stanów) ORAZ bez ogona slotu
    if len(parts) >= 2 and parts[-1].endswith(_STATE_VALUE_SLOT_SUFFIX) and len(parts[-1]) > len(_STATE_VALUE_SLOT_SUFFIX):
        base_parts = parts[:-1]
        slot_head = parts[-1][: -len(_STATE_VALUE_SLOT_SUFFIX)]
        aliased = ":".join(base_parts + [slot_head])
        add(aliased)
        add(":".join(base_parts))
        if base_parts[0] not in _KV_NAMESPACE_SEGMENTS:
            add(f"fact:{aliased}")
            add(f"fact:{':'.join(base_parts)}")

    # pełna baza subjectu (po odpreksowaniu) — dla kluczy typu 'X:Y:state_value'
    if len(parts) >= 3 and parts[-1].endswith(_STATE_VALUE_SLOT_SUFFIX):
        add(":".join(parts[1:-1]))

    return probes


def _kv_state_is_superseded(state: Any) -> bool:
    """Jedna procedura oceny: czy wpis KV (dowolna projekcja shape'ów) niesie tombstone.

    Lustrzana semantyka `_is_superseded` konsolidatora V151: flaga w danych lub
    w metadanych rekordu (is_superseded / superseded / vector_tombstone_pending).
    """
    if not isinstance(state, dict):
        return False
    meta = state.get("metadata")
    meta = meta if isinstance(meta, dict) else {}
    return bool(
        state.get("is_superseded")
        or meta.get("is_superseded")
        or state.get("superseded")
        or meta.get("superseded")
        or meta.get("vector_tombstone_pending")
    )


async def _graph_channel_kv_truth_gate(
    candidates: Any,
    kv: Optional[Any],
) -> Any:
    """V151.1 F-031(1b): odrzuć rekordy kanału grafowego, których baza klucza
    jest w KV oznaczona is_superseded. Akceptuje dict[str, rec] albo list[rec] —
    zwraca ten sam kształt. Fail-open przy błędzie KV (brak dowodu ≠ tombstone).
    """
    if kv is None or not candidates:
        return candidates

    graph_recs: List[Any] = []
    pool = list(candidates.values()) if isinstance(candidates, dict) else list(candidates)
    if isinstance(candidates, dict):
        # klucz poola to kanoniczny klucz widoku — musi trafić do probe obok subjectu
        for key, rec in zip(list(candidates.keys()), pool, strict=True):
            if isinstance(rec, dict):
                rec.setdefault("key", str(key))
    for rec in pool:
        if isinstance(rec, dict) and (rec.get("source") == "graph" or rec.get("graph_hit")):
            graph_recs.append(rec)
    if not graph_recs:
        return candidates

    probe_map: Dict[str, List[Any]] = {}
    for rec in graph_recs:
        raw_keys = [rec.get("key"), rec.get("subject"), rec.get("canonical_entity_id")]
        derived: Set[str] = set()
        for raw in raw_keys:
            if raw:
                derived.update(_derive_kv_probe_keys(raw))
        if derived:
            probe_map[id(rec)] = list(derived)

    all_probes = {k for keys in probe_map.values() for k in keys}
    # V151.1 (F-031): bramka bez refleksji — wąski strukturalny protokół
    # (runtime-checkable, F3/F6 invariant: zero reflection w candidates.py).
    # Dowolny magazyn z batchowym get_states przechodzi; brak → fail-open
    # (brak dowodu ≠ tombstone), bez refleksyjnych atrap.
    from typing import Protocol, runtime_checkable

    @runtime_checkable
    class KVTruthStateReaderProtocol(Protocol):
        async def get_states(self, keys: List[str]) -> Dict[str, Any]: ...

    states: Dict[str, Any] = {}
    if isinstance(kv, KVTruthStateReaderProtocol):
        try:
            states = await kv.get_states(list(all_probes))
        except Exception as exc:
            logger.debug("[V151.1/F-031] KV truth gate probe failed (fail-open): %s", exc)
            return candidates

    tombstoned = {
        id(rec)
        for rec in graph_recs
        if any(_kv_state_is_superseded(states.get(pk)) for pk in probe_map.get(id(rec), []))
    }
    if not tombstoned:
        return candidates

    if isinstance(candidates, dict):
        for key, rec in list(candidates.items()):
            if id(rec) in tombstoned:
                candidates.pop(key, None)
    else:
        candidates = [rec for rec in candidates if id(rec) not in tombstoned]
    logger.info("[V151.1/F-031] read-path KV truth gate: odrzucono %d rekordów kanału graph", len(tombstoned))
    return candidates


def _is_filtered(key: str, val: str, meta: Optional[Dict[str, Any]], sess: Optional[str]) -> bool:
    kl, m = key.lower(), meta or {}
    if (
        any(kl.startswith(p) for p in PROBE_PREFIXES)
        or is_conversational_churn(key, val)
        or kl.startswith(("fact:conversation", "conversation:"))
    ):
        return True
    return bool(
        m.get("is_superseded")
        or m.get("superseded")
        or m.get("vector_tombstone_pending")
    ) or not is_session_accessible(
        extract_effective_session(key, m), sess
    )


def _build_record(
    k: str, v: Any, src: str, data: Dict[str, Any], vec_score: float
) -> Dict[str, Any]:
    m = dict(data.get("metadata") or {})
    if "cpof_severity" in data and "cpof_severity" not in m:
        m["cpof_severity"] = float(data["cpof_severity"])
    rec = {
        "key": k,
        "subject": k,
        "predicate": data.get("predicate", "stated_memory"),
        "object": str(v),
        "content": str(data.get("content") or v),
        "source": src,
        "vector_score": vec_score,
        "importance_score": 1.0 if vec_score >= 1.0 else vec_score,
        "confidence": float(data.get("confidence", 1.0) or data.get("veracity", 1.0)),
        "source_type": str(data.get("source_type") or m.get("source_type", "user_explicit")),
        "metadata": m,
        "timestamp": float(data.get("timestamp", 0.0) or m.get("timestamp", 0.0)),
    }
    if data.get("vector") is not None:
        rec["vector"] = data["vector"]
    return rec


gather = gather_candidates

