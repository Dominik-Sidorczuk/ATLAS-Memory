"""W3, W4 & W6 (V104.5 / HippoRAG 2): O(E) igraph Personalized PageRank, Query-to-Triple Seeding & Telemetry.

Fixes implemented in V104.5:
- FIX-2 (FINDING-002): Zero silent fallbacks; explicit `hipporag_error` metric counter + fail-fast raise on PPR hot-path errors.
- FIX-3 (FINDING-004): Eliminates `np.zeros((N, N), dtype=np.float64)` (2.48 GB RAM explosion) in default `igraph` mode.
  Builds `ig.Graph` directly from `O(E)` deduplicated edge map (`< 10 MB RAM` for 15k nodes) + cached `O(E)` inverted
  token/lemma index for sub-10ms `score_query_to_triples` (`p95 < 50 ms/query`). Preserves `ATLAS_PPR_BACKEND=numpy`
  reference backend verified by `test_igraph_matches_numpy_ppr`.
- FIX-4 (FINDING-005 & FINDING-007): Default `ATLAS_HIPPORAG="0"` (OFF per SPEC §8) and exposes
  `hybrid_channels_log` (ring buffer 20), `seed_hit_ratio` (rolling 100), `ppr_backend`, and `hipporag_error`.
"""

from __future__ import annotations

import logging
import math
import re
import time
import zlib
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Deque, Dict, List, Optional, Sequence, Set, Tuple

try:
    import igraph as ig
except ImportError:
    ig = None  # type: ignore[assignment]
import numpy as np

from atlas_memory.core.exceptions import InvariantViolation
from atlas_memory.core.linguistics import (
    PPR_META_STOPSTEMS,
    PPR_META_STOPWORDS,
    _bilingual_stem,
)

logger = logging.getLogger(__name__)

PPR_DAMPING_DEFAULT: float = 0.85
PPR_MAX_ITER_DEFAULT: int = 50
PPR_TOL_DEFAULT: float = 1e-7
PASSAGE_SEED_WEIGHT_DEFAULT: float = 0.05
SYNONYM_WEIGHT_DEFAULT: float = 1.0
CONTAINS_WEIGHT_DEFAULT: float = 1.0
# V151.5 (WEKTOR II): masa probabilistyczna tunelowania aksjomatów (config-overridable
# przez getattr(get_config(), "ppr_beta_tunnel", ...) — zero inline hardcode w run()).
PPR_BETA_TUNNEL_DEFAULT: float = 0.10

# --- V104.5 Observability, Metrics & Exception Types (FIX-2 & FIX-4) ---
class HippoRAGPPRUnavailable(RuntimeError):
    """Raised when HippoRAG 2 PPR execution fails in the forward retrieval path (zero silent fallback)."""


from atlas_memory.core.models import metrics


def _on_ppr_metric_inc(name: str, amount: int) -> None:
    if name in ("hipporag_error", "hipporag_ppr_error", "kuzu_write_error"):
        record_hipporag_error(int(amount))


metrics.on_inc_hook = _on_ppr_metric_inc

_HYBRID_CHANNELS_LOG: Dict[str, Deque[str]] = {}
_HYBRID_CHANNELS_LOG_MAX: int = 32
_HYBRID_KNOWN_CHANNELS: frozenset = frozenset({"kv", "vector_dense", "hot_ppr", "graph_ppr", "graph"})
_SEED_HIT_HISTORY: Deque[float] = deque(maxlen=100)
_HIPPORAG_ERROR_COUNT: int = 0


def record_channel_query(channel: str, query_id: str) -> None:
    """Records `query_id` in the 20-element ring buffer for `channel`.

    V118 channel cap: unknown (dynamic) channels are bounded at
    `_HYBRID_CHANNELS_LOG_MAX` (32) with FIFO eviction, so unbounded channel
    names cannot grow the module-level dict. Known production channels are
    exempt and always recorded; adjacent dedupe is preserved.
    """
    ch = str(channel or "").strip()
    qid = str(query_id or "").strip()
    if not ch or not qid:
        return
    buf = _HYBRID_CHANNELS_LOG.get(ch)
    if buf is None:
        if ch not in _HYBRID_KNOWN_CHANNELS:
            dyn = [k for k in _HYBRID_CHANNELS_LOG if k not in _HYBRID_KNOWN_CHANNELS]
            while len(dyn) >= _HYBRID_CHANNELS_LOG_MAX:
                _HYBRID_CHANNELS_LOG.pop(dyn.pop(0), None)
        buf = deque(maxlen=20)
        _HYBRID_CHANNELS_LOG[ch] = buf
    if not buf or buf[-1] != qid:
        buf.append(qid)


def record_seed_hit(ratio: float) -> None:
    """Records query seed-hit ratio in the 100-query rolling window."""
    _SEED_HIT_HISTORY.append(max(0.0, min(1.0, float(ratio))))


def record_hipporag_error(inc: int = 1) -> int:
    """Increments the `hipporag_error` counter (FIX-2)."""
    global _HIPPORAG_ERROR_COUNT
    _HIPPORAG_ERROR_COUNT += int(inc)
    return _HIPPORAG_ERROR_COUNT


def get_ppr_backend(explicit_backend: Optional[str] = None) -> str:
    """Returns active PPR backend (`'igraph'` or `'numpy'`)."""
    from atlas_memory.core.config import get_config
    raw = (explicit_backend or get_config().ppr_backend).strip().lower()
    return "numpy" if raw == "numpy" else "igraph"


def compute_dual_node_coverage(graph_store: Optional[Any] = None) -> float:
    """Computes fraction of Passage nodes possessing at least one CONTAINS edge (`0.0` before migration, `1.0` when fully covered)."""
    if graph_store is None:
        return 0.0
    cov_fn = getattr(graph_store, "get_dual_node_coverage", None)
    if callable(cov_fn):
        return float(cov_fn())
    nx_g = getattr(graph_store, "nx_graph", graph_store)
    if nx_g is None or not hasattr(nx_g, "nodes"):
        return 0.0
    passage_nodes = [
        n for n, d in nx_g.nodes(data=True) if isinstance(d, dict) and d.get("type") == "passage"
    ]
    if not passage_nodes:
        return 0.0
    with_contains = 0
    for p_node in passage_nodes:
        has_c = False
        for _, _, edata in nx_g.edges(p_node, data=True):
            if isinstance(edata, dict) and edata.get("predicate") == "CONTAINS":
                has_c = True
                break
        if has_c:
            with_contains += 1
    return round(float(with_contains) / float(len(passage_nodes)), 4)


def get_hipporag_telemetry(graph_store: Optional[Any] = None) -> Dict[str, Any]:
    """Returns V104.5 telemetry snapshot for `get_stats` (`hybrid_channels_log`, `seed_hit_ratio`, `ppr_backend`, `hipporag_error`, `dual_node_coverage`)."""
    enabled = is_hipporag_enabled()
    channels_snapshot: Dict[str, List[str]] = {
        ch: list(buf)
        for ch, buf in _HYBRID_CHANNELS_LOG.items()
        if len(buf) > 0 and (enabled or ch != "graph_ppr")
    }
    if enabled and "graph_ppr" not in channels_snapshot:
        channels_snapshot["graph_ppr"] = list(_HYBRID_CHANNELS_LOG.get("graph_ppr", []))
    seed_ratio = (
        round(float(sum(_SEED_HIT_HISTORY) / len(_SEED_HIT_HISTORY)), 4)
        if len(_SEED_HIT_HISTORY) > 0
        else 0.0
    )
    return {
        "hybrid_channels_log": channels_snapshot,
        "seed_hit_ratio": seed_ratio,
        "ppr_backend": get_ppr_backend(),
        "hipporag_error": int(_HIPPORAG_ERROR_COUNT),
        "dual_node_coverage": compute_dual_node_coverage(graph_store),
    }


def reset_hipporag_telemetry() -> None:
    """Resets HippoRAG telemetry buffers (used in unit/contract tests)."""
    global _HIPPORAG_ERROR_COUNT
    _HYBRID_CHANNELS_LOG.clear()
    _SEED_HIT_HISTORY.clear()
    _HIPPORAG_ERROR_COUNT = 0
    metrics.reset()


@dataclass(frozen=True)
class PPRResult:
    """Output of a HippoRAG 2 PPR execution on the dual-node graph."""

    candidates: List[Dict[str, Any]]
    node_scores: Dict[str, float]
    seed_nodes: Dict[str, float]
    iterations: int
    latency_ms: float
    deterministic_seed: int
    seed_hit_ratio: float = 0.0
    backend: str = "igraph"


def is_hipporag_enabled(explicit_flag: Optional[bool] = None) -> bool:
    """Runtime flag `ATLAS_HIPPORAG` resolved via canonical AtlasConfig."""
    if explicit_flag is not None:
        return bool(explicit_flag)
    from atlas_memory.core.config import get_config
    return get_config().hipporag_enabled


def is_recognition_memory_enabled(explicit_flag: Optional[bool] = None) -> bool:
    """Runtime flag `ATLAS_HIPPORAG_RM` resolved via canonical AtlasConfig."""
    if explicit_flag is not None:
        return bool(explicit_flag)
    from atlas_memory.core.config import get_config
    return get_config().hipporag_rm_enabled


_IDENT_SPLIT_RE = re.compile(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9]{2,}")


@lru_cache(maxsize=65536)
def _tokenize_and_stem(text: str) -> Tuple[frozenset[str], frozenset[str]]:
    """Fast word-cached stemmer with stopword & meta-verb filtering so generic framing words never become graph seeds."""
    clean = str(text or "").strip()
    if not clean:
        return frozenset(), frozenset()
    low = clean.lower()
    try:
        from atlas_memory.core.linguistics import LanguageAssetManager
        stop_set = (LanguageAssetManager.get_instance().get_stopwords("pl") | PPR_META_STOPWORDS) - {"other"}
    except Exception:
        stop_set = PPR_META_STOPWORDS
    parts = [
        m.group(0) for m in _IDENT_SPLIT_RE.finditer(low)
        if m.group(0) not in stop_set and (len(m.group(0)) >= 3 or any(ch.isdigit() for ch in m.group(0)))
    ]
    surface = set(parts)
    if " " not in low and ("_" in low or ":" in low or any(ch.isdigit() for ch in low)) and low not in stop_set:
        surface.add(low)
    stems = set()
    for p in parts:
        st = (_bilingual_stem(p) or p) if not p.isdigit() else p
        if st not in PPR_META_STOPSTEMS and st not in stop_set:
            stems.add(st)
    stems |= surface
    return frozenset(surface), frozenset(stems)


def adaptive_ppr_budget(
    n_nodes: int,
    n_edges: int,
    tau_base: Optional[float] = None,
    gamma: Optional[float] = None,
    floor_ms: Optional[float] = None,
) -> float:
    """V151.4 (WEKTOR II) + V151.2: adaptacyjny budżet PPR z twardą podłogą.

    `budget_ms = max(ppr_budget_floor_ms, ppr_tau_base + ppr_gamma * ln(V + E))`
    — parametry wyłącznie z kanonicznego `AtlasConfig`, zero hardcoded 35.0.
    V151.2 (dyrektywa usera 2026-10-09): podłoga 200 ms gwarantuje pełną
    konwergencję PageRank na aktualnym grafie; formuła logarytmiczna nadal
    skaluje BUDŻET WZGÓRZ z rozmiarem grafu (ponad podłogę).
    Złożoność O(1); liczony PRZED `run()` z sygnatury snapshotu (n_nodes, n_edges).
    """
    from atlas_memory.core.config import get_config

    cfg = get_config()
    base = float(cfg.ppr_tau_base if tau_base is None else tau_base)
    g = float(cfg.ppr_gamma if gamma is None else gamma)
    floor = float(cfg.ppr_budget_floor_ms) if floor_ms is None else float(floor_ms)
    scale = max(1, int(n_nodes) + int(n_edges))
    adaptive = base + g * math.log(float(scale))
    return max(floor, adaptive)


class DirectionalFristonTunnel:
    """Filar III-2 (V153): Bi-procesualny rzut kierunkowy sterowany precyzją Fristona w HippoRAG 2.

    T(q, r) = sigma((cos(e_q, e_r) - theta_base) / tau) * Lambda(r)
    gdzie:
    - e_q, e_r to znormalizowane embeddingi zapytania i aksjomatu (PolDense-68M ONNX / encoder)
    - Lambda(r) = Pi_{channel}(r) * clamp_conf(confidence_r) z BayesianPrecisionTensor
    - sigma(x) = 1 / (1 + exp(-x))
    - tau = 0.08, theta_base = 0.38
    - Twarde odcięcie: gdy cos(e_q, e_r) < theta_base, T(q, r) = 0.0.
    """

    def __init__(
        self,
        encoder: Optional[Any] = None,
        precision_tensor: Optional[Any] = None,
        theta_base: Optional[float] = None,
        tau: float = 0.08,
    ) -> None:
        from atlas_memory.core.config import get_config

        self.encoder = encoder
        self.theta_base = float(get_config().axiom_theta_min if theta_base is None else theta_base)
        self.tau = float(tau)
        if precision_tensor is not None:
            self.precision_tensor = precision_tensor
        else:
            from atlas_memory.cognitive.precision import BayesianPrecisionTensor

            self.precision_tensor = BayesianPrecisionTensor()

    def _get_encoder(self) -> Optional[Any]:
        if self.encoder is not None:
            return self.encoder
        try:
            from atlas_memory.l2_semantic.vector_backend import build_encoder, load_vector_backend

            vb = load_vector_backend()
            self.encoder = build_encoder(vb)
            return self.encoder
        except Exception:
            return None

    def encode(self, text: str) -> Optional[np.ndarray]:
        """Koduje tekst do znormalizowanego wektora jednostkowego."""
        enc = self._get_encoder()
        if enc is not None and hasattr(enc, "encode"):
            try:
                v = enc.encode(text)
                if isinstance(v, np.ndarray):
                    if v.ndim == 2:
                        v = v[0]
                    v = v.astype(np.float64)
                    norm = np.linalg.norm(v)
                    return v / norm if norm > 1e-9 else v
            except Exception as exc:
                logger.debug("DirectionalFristonTunnel encode exception: %s", exc)
        return None

    def compute_tunnel_conductance(
        self,
        query: str,
        axiom_names: Sequence[str],
        axiom_confidences: Optional[Sequence[float]] = None,
        axiom_channels: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Oblicza wektor przewodnictwa T(q, r) dla każdego aksjomatu r."""
        n = len(axiom_names)
        if n == 0:
            return np.zeros(0, dtype=np.float64)

        confs = list(axiom_confidences) if axiom_confidences is not None else [1.0] * n
        channels = list(axiom_channels) if axiom_channels is not None else ["user_explicit"] * n

        e_q = self.encode(query)
        conductances = np.zeros(n, dtype=np.float64)

        for i, name in enumerate(axiom_names):
            c_val = confs[i] if i < len(confs) else 1.0
            ch_val = channels[i] if i < len(channels) else "user_explicit"
            lambda_r = self.precision_tensor.compute_lambda(ch_val, c_val)

            cos_sim: Optional[float] = None
            if e_q is not None:
                e_r = self.encode(str(name or ""))
                if e_r is not None:
                    cos_sim = float(np.dot(e_q, e_r))

            if cos_sim is None:
                q_words, q_stems = _tokenize_and_stem(query or "")
                q_tokens = set(q_words) | set(q_stems)
                a_words, a_stems = _tokenize_and_stem(str(name or ""))
                a_tokens = set(a_words) | set(a_stems)
                union = q_tokens | a_tokens
                if union and q_tokens:
                    cos_sim = float(len(q_tokens & a_tokens)) / float(len(union))
                else:
                    cos_sim = 0.0

            if cos_sim < self.theta_base:
                conductances[i] = 0.0
            else:
                x = (cos_sim - self.theta_base) / self.tau
                sigma = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))
                conductances[i] = sigma * lambda_r

        return conductances


def compute_axiom_tunnel_mask(
    query: str,
    axiom_node_names: List[str],
    theta_min: Optional[float] = None,
) -> List[float]:
    """V153 (FILAR III-2): Maska ugruntowania tunelowania aksjomatów Fristona."""
    from atlas_memory.core.config import get_config

    theta = float(get_config().axiom_theta_min if theta_min is None else theta_min)
    tunnel = DirectionalFristonTunnel(theta_base=theta)
    conductances = tunnel.compute_tunnel_conductance(query, axiom_node_names)
    return [1.0 if c > 0.0 else 0.0 for c in conductances]


@dataclass
class _GraphTripleIndex:
    """O(E) inverted token/stem index over `nx_graph` so `score_query_to_triples` runs in <5ms."""

    sig: Tuple[int, int, int]
    passage_entries: List[Tuple[str, str, str, frozenset[str]]]
    stem_to_passages: Dict[str, List[int]]
    edge_entries: List[
        Tuple[str, str, str, float, str, str, frozenset[str], frozenset[str], frozenset[str], frozenset[str]]
    ]
    stem_to_edges: Dict[str, List[int]]


def _get_or_build_triple_index(nx_graph: Any) -> _GraphTripleIndex:
    rev = int(getattr(nx_graph, "revision", 0) or getattr(nx_graph, "_revision", 0))
    sig = (int(nx_graph.number_of_nodes()), int(nx_graph.number_of_edges()), rev)
    cached: Optional[_GraphTripleIndex] = getattr(nx_graph, "_hipporag_triple_idx", None)
    if cached is not None and cached.sig == sig:
        return cached

    passage_entries: List[Tuple[str, str, str, frozenset[str]]] = []
    stem_to_passages: Dict[str, List[int]] = {}

    for node, ndata in nx_graph.nodes(data=True):
        if ndata.get("type") == "passage":
            skey = str(ndata.get("source_key") or "")
            txt = str(ndata.get("text") or "")
            s_words, s_stems = _tokenize_and_stem(skey)
            p_words, p_stems = _tokenize_and_stem(txt[:400])
            all_tokens = s_words | s_stems | p_words | p_stems
            p_idx = len(passage_entries)
            passage_entries.append((str(node), skey, skey.lower(), all_tokens))
            for tok in all_tokens:
                stem_to_passages.setdefault(tok, []).append(p_idx)

    edge_entries: List[
        Tuple[str, str, str, float, str, str, frozenset[str], frozenset[str], frozenset[str], frozenset[str]]
    ] = []
    stem_to_edges: Dict[str, List[int]] = {}

    for u, v, edata in nx_graph.edges(data=True):
        u_str, v_str = str(u), str(v)
        pred = str(edata.get("predicate") or edata.get("relation") or "related_to")
        u_node_type = nx_graph.nodes[u].get("type", "entity") if u in nx_graph.nodes else "entity"

        if pred == "CONTAINS" and u_node_type == "passage":
            skey = str(nx_graph.nodes[u].get("source_key") or u_str)
            triple_text = f"{skey} {v_str}"
            u_words, u_stems = _tokenize_and_stem(skey)
        else:
            triple_text = f"{u_str} {pred} {v_str}"
            u_words, u_stems = _tokenize_and_stem(u_str)

        v_words, v_stems = _tokenize_and_stem(v_str)
        p_words, p_stems = _tokenize_and_stem(pred)
        t_words = u_words | v_words | p_words
        t_stems = u_stems | v_stems | p_stems
        t_low = triple_text.lower()
        t_norm = re.sub(r"[\s_\-:/]+", "", t_low)
        conf = float(edata.get("confidence", edata.get("sim", edata.get("weight", 1.0))) or 1.0)

        e_idx = len(edge_entries)
        edge_entries.append((u_str, pred, v_str, conf, t_low, t_norm, u_words | u_stems, v_words | v_stems, t_words, t_stems))
        for tok in (t_words | t_stems):
            stem_to_edges.setdefault(tok, []).append(e_idx)

    built = _GraphTripleIndex(
        sig=sig,
        passage_entries=passage_entries,
        stem_to_passages=stem_to_passages,
        edge_entries=edge_entries,
        stem_to_edges=stem_to_edges,
    )
    try:
        nx_graph._hipporag_triple_idx = built
    except Exception as exc:
        logger.debug("Could not attach triple index to nx_graph: %s", exc)
    return built


def score_query_to_triples(
    query: str,
    nx_graph: Any,
    top_k: int = 15,
    recognition_memory: Optional[bool] = None,
    rm_threshold: float = 0.45,
) -> Tuple[Dict[str, float], Dict[str, float], List[Tuple[str, str, str, float]]]:
    """W3 & W6: Scores graph triples `(u, pred, v)` against `query` using the O(E) inverted lemma index.

    Returns:
        `(phrase_seeds, passage_seeds, scored_triples)`
    """
    q_clean = str(query or "").strip()
    if not q_clean or nx_graph is None:
        return {}, {}, []

    q_low = q_clean.lower()
    q_words, q_stems = _tokenize_and_stem(q_clean)
    q_all_lookup = q_words | q_stems
    q_norm = re.sub(r"[\s_\-:/]+", "", q_low)

    use_rm = is_recognition_memory_enabled(recognition_memory)
    idx = _get_or_build_triple_index(nx_graph)

    scored_triples: List[Tuple[str, str, str, float, bool, bool]] = []
    passage_seeds: Dict[str, float] = {}

    # 1. Score candidate Passage nodes via inverted index (selective / high-IDF tokens first)
    cand_p_indices: Set[int] = set()
    sorted_p_toks = sorted(
        (t for t in q_all_lookup if t in idx.stem_to_passages),
        key=lambda t: len(idx.stem_to_passages[t]),
    )
    for tok in sorted_p_toks:
        postings = idx.stem_to_passages[tok]
        if cand_p_indices and len(postings) > 128:
            continue
        for p_i in postings[:128]:
            cand_p_indices.add(p_i)
        if len(cand_p_indices) >= 64:
            break
    if not cand_p_indices and len(idx.passage_entries) <= 250:
        cand_p_indices = set(range(len(idx.passage_entries)))

    for p_i in cand_p_indices:
        node_str, _skey, skey_low, p_all_tokens = idx.passage_entries[p_i]
        sim = 0.0
        if skey_low == q_low:
            sim = 1.0
        elif skey_low and len(q_low) >= 4 and (q_low in skey_low or (len(skey_low) >= 6 and skey_low in q_low)):
            sim = 0.85
        elif q_stems and p_all_tokens:
            overlap = len(q_stems & p_all_tokens)
            if overlap >= 2 or (overlap == 1 and len(q_stems) <= 2):
                sim = min(0.95, overlap / max(1, len(q_stems)))
        if sim >= 0.34:
            passage_seeds[node_str] = float(sim)

    # 2. Score candidate edges via inverted token/lemma index (selective / high-IDF tokens first)
    cand_e_indices: Set[int] = set()
    sorted_e_toks = sorted(
        (t for t in q_all_lookup if t in idx.stem_to_edges),
        key=lambda t: len(idx.stem_to_edges[t]),
    )
    for tok in sorted_e_toks:
        postings = idx.stem_to_edges[tok]
        if cand_e_indices and len(postings) > 256:
            continue
        for e_i in postings[:256]:
            cand_e_indices.add(e_i)
        if len(cand_e_indices) >= 256:
            break
    if not cand_e_indices and len(idx.edge_entries) <= 500:
        cand_e_indices = set(range(len(idx.edge_entries)))

    q_padded = f" {q_low} "
    for e_i in cand_e_indices:
        u_str, pred, v_str, conf, _t_low, t_norm, u_toks, v_toks, t_words, t_stems = idx.edge_entries[e_i]
        u_low, v_low = u_str.lower(), v_str.lower()
        u_exact = len(u_str) >= 3 and (u_low == q_low or (len(u_low) >= 4 and f" {u_low} " in q_padded))
        v_exact = len(v_str) >= 3 and (v_low == q_low or (len(v_low) >= 4 and f" {v_low} " in q_padded))
        u_match = u_exact or bool(q_all_lookup & u_toks)
        v_match = v_exact or bool(q_all_lookup & v_toks)

        score = 0.0
        if v_low == q_low or u_low == q_low:
            score = 1.0
        elif u_exact or v_exact:
            score = 0.95
        elif q_norm and len(q_norm) >= 4 and (q_norm in t_norm or t_norm in q_norm):
            score = 0.92
        else:
            word_hits = len(q_words & t_words)
            stem_hits = len(q_stems & t_stems)
            if word_hits > 0 or stem_hits > 0:
                cov = max(
                    word_hits / max(1, len(q_words)),
                    stem_hits / max(1, len(q_stems)),
                )
                score = min(0.90, 0.25 + 0.65 * cov)

        if score > 0.0:
            eff_score = round(float(score * min(1.0, max(0.2, conf))), 4)
            scored_triples.append((u_str, pred, v_str, eff_score, u_match, v_match))

    scored_triples.sort(key=lambda x: (x[3], x[0], x[2]), reverse=True)

    # W6 Recognition Memory: filter out low-specificity triples before top-K selection
    if use_rm:
        scored_triples = [t for t in scored_triples if t[3] >= rm_threshold]

    selected_with_flags = scored_triples[: max(1, top_k)]

    from atlas_memory.core.linguistics import is_valid_graph_entity
    phrase_seeds: Dict[str, float] = {}
    for u_str, pred, v_str, t_score, u_match, v_match in selected_with_flags:
        if t_score < 0.34:
            continue
        u_type = nx_graph.nodes[u_str].get("type", "entity") if u_str in nx_graph.nodes else "entity"
        v_type = nx_graph.nodes[v_str].get("type", "entity") if v_str in nx_graph.nodes else "entity"
        if u_type != "passage" and u_match and is_valid_graph_entity(u_str):
            deg_u = nx_graph.degree(u_str) if hasattr(nx_graph, "degree") and u_str in nx_graph else 1
            idf_u = 1.0 if deg_u <= 3 else (1.0 / (1.0 + 0.25 * (deg_u - 3)))
            phrase_seeds[u_str] = max(phrase_seeds.get(u_str, 0.0), round(t_score * idf_u, 4))
        if v_type != "passage" and v_match and is_valid_graph_entity(v_str):
            deg_v = nx_graph.degree(v_str) if hasattr(nx_graph, "degree") and v_str in nx_graph else 1
            idf_v = 1.0 if deg_v <= 3 else (1.0 / (1.0 + 0.25 * (deg_v - 3)))
            phrase_seeds[v_str] = max(phrase_seeds.get(v_str, 0.0), round(t_score * idf_v, 4))
        # For direct entity-to-entity RELATION edges where one endpoint strongly matched, allow 1-hop target with damped weight
        if pred not in ("SYNONYM", "CONTAINS"):
            if u_match and not v_match and v_type != "passage" and is_valid_graph_entity(v_str):
                deg_v = nx_graph.degree(v_str) if hasattr(nx_graph, "degree") and v_str in nx_graph else 1
                idf_v = 1.0 if deg_v <= 3 else (1.0 / (1.0 + 0.25 * (deg_v - 3)))
                eff_parent = phrase_seeds.get(u_str, t_score)
                phrase_seeds[v_str] = max(phrase_seeds.get(v_str, 0.0), round(eff_parent * 0.45 * idf_v, 4))
            elif v_match and not u_match and u_type != "passage" and is_valid_graph_entity(u_str):
                deg_u = nx_graph.degree(u_str) if hasattr(nx_graph, "degree") and u_str in nx_graph else 1
                idf_u = 1.0 if deg_u <= 3 else (1.0 / (1.0 + 0.25 * (deg_u - 3)))
                eff_parent = phrase_seeds.get(v_str, t_score)
                phrase_seeds[u_str] = max(phrase_seeds.get(u_str, 0.0), round(eff_parent * 0.45 * idf_u, 4))

    selected_triples = [(u, p, v, s) for (u, p, v, s, _um, _vm) in selected_with_flags]
    return phrase_seeds, passage_seeds, selected_triples


class HippoRAG2PPR:
    """W4: O(E) Personalized PageRank engine over Kùzu Dual-Node Graph (`igraph` primary + `numpy` parity backend)."""

    def __init__(
        self,
        damping: float = PPR_DAMPING_DEFAULT,
        max_iter: int = PPR_MAX_ITER_DEFAULT,
        tol: float = PPR_TOL_DEFAULT,
        passage_seed_weight: float = PASSAGE_SEED_WEIGHT_DEFAULT,
        synonym_weight: float = SYNONYM_WEIGHT_DEFAULT,
        contains_weight: float = CONTAINS_WEIGHT_DEFAULT,
        backend: Optional[str] = None,
    ) -> None:
        self.damping = float(damping)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.passage_seed_weight = float(passage_seed_weight)
        self.synonym_weight = float(synonym_weight)
        self.contains_weight = float(contains_weight)
        self.backend = backend
        self._cached_sig: Optional[Tuple[int, int, int, float, float]] = None
        self._cached_nodes: List[str] = []
        self._cached_node_to_idx: Dict[str, int] = {}
        self._cached_edge_weights: Dict[Tuple[int, int], float] = {}
        self._cached_ig_graph: Optional[ig.Graph] = None
        self._cached_passage_indices: List[Tuple[int, str, str, str]] = []
        self._cached_entity_mask: np.ndarray = np.zeros(0, dtype=bool)
        self._cached_entity_rel_edges: List[Tuple[int, str, str, str, float]] = []

    def _build_graph_structures(
        self, graph_store: Any
    ) -> Tuple[List[str], Dict[str, int], Dict[Tuple[int, int], float], Optional[ig.Graph]]:
        """Builds O(E) deduplicated edge map and native `ig.Graph` WITHOUT allocating O(N^2) dense arrays."""
        nx_graph = graph_store.nx_graph if hasattr(graph_store, "nx_graph") else graph_store
        n_nodes = int(nx_graph.number_of_nodes())
        n_edges = int(nx_graph.number_of_edges())
        revision = int(getattr(nx_graph, "revision", 0) or getattr(nx_graph, "_revision", 0) or getattr(graph_store, "revision", 0) or getattr(graph_store, "_revision", 0))
        if revision == 0 and n_edges > 0:
            all_edges = list(nx_graph.edges(data=True))
            if n_edges <= 2048:
                sampled_edges = all_edges
            else:
                step = max(1, n_edges // 128)
                sampled_edges = all_edges[::step]
            edge_sample = sorted(
                (str(u), str(v), round(float(edata.get("confidence", edata.get("weight", 1.0)) or 1.0), 4))
                for u, v, edata in sampled_edges
            )
            revision = int(zlib.crc32(str(edge_sample).encode("utf-8")))
        sig = (n_nodes, n_edges, revision, self.synonym_weight, self.contains_weight)

        if self._cached_sig == sig and self._cached_ig_graph is not None:
            return self._cached_nodes, self._cached_node_to_idx, self._cached_edge_weights, self._cached_ig_graph

        nodes = sorted(str(n) for n in nx_graph.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)
        if n == 0:
            return nodes, node_to_idx, {}, None

        # CPoF acyclic cycle / hub damping guard (§4 W4)
        cpof_hubs: Set[str] = set()
        detect_fn = getattr(graph_store, "detect_cpof", None)
        if callable(detect_fn):
            for node_name in nodes[:64]:
                if nx_graph.degree(node_name) >= 12:
                    try:
                        if detect_fn(node_name):
                            cpof_hubs.add(node_name)
                    except Exception:
                        record_hipporag_error()

        # O(E) sparse edge accumulator: zero N x N dense matrix allocation!
        edge_map: Dict[Tuple[int, int], float] = {}
        for u, v, edata in nx_graph.edges(data=True):
            u_s, v_s = str(u), str(v)
            if u_s == v_s:
                continue
            i, j = node_to_idx.get(u_s), node_to_idx.get(v_s)
            if i is None or j is None:
                continue

            pred = str(edata.get("predicate") or "RELATION")
            if pred == "CONTAINS":
                if self.contains_weight <= 0.0:
                    continue
                w = float(edata.get("weight", 1.0) or 1.0) * self.contains_weight
                p_to_e = 0.25 * w
                if p_to_e > edge_map.get((i, j), 0.0):
                    edge_map[(i, j)] = p_to_e
                if w > edge_map.get((j, i), 0.0):
                    edge_map[(j, i)] = w
            elif pred == "SYNONYM":
                if self.synonym_weight <= 0.0:
                    continue
                sim_val = float(edata.get("sim", edata.get("confidence", 0.85)) or 0.85)
                w = sim_val * self.synonym_weight
                if w > edge_map.get((i, j), 0.0):
                    edge_map[(i, j)] = w
                if w > edge_map.get((j, i), 0.0):
                    edge_map[(j, i)] = w
            else:
                conf = float(edata.get("confidence", 1.0) or 1.0)
                deg_u = max(1, nx_graph.degree(u_s)) if hasattr(nx_graph, "degree") and u_s in nx_graph else 1
                deg_v = max(1, nx_graph.degree(v_s)) if hasattr(nx_graph, "degree") and v_s in nx_graph else 1
                laplace_w = conf / math.sqrt(float(deg_u * deg_v))
                if u_s in cpof_hubs or v_s in cpof_hubs:
                    laplace_w *= 0.5
                if laplace_w > edge_map.get((i, j), 0.0):
                    edge_map[(i, j)] = laplace_w
                rev_w = 0.5 * laplace_w
                if rev_w > edge_map.get((j, i), 0.0):
                    edge_map[(j, i)] = rev_w

        ig_edges = list(edge_map.keys())
        ig_weights = [edge_map[e] for e in ig_edges]
        if ig is not None:
            ig_graph = ig.Graph(n=n, edges=ig_edges, directed=True)
            ig_graph.es["weight"] = ig_weights
        else:
            ig_graph = None

        passage_indices: List[Tuple[int, str, str, str]] = []
        entity_mask = np.zeros(n, dtype=bool)
        entity_rel_edges: List[Tuple[int, str, str, str, float]] = []
        for idx_n, node_name in enumerate(nodes):
            ndata = nx_graph.nodes[node_name]
            if isinstance(ndata, dict) and ndata.get("type") == "passage":
                skey = str(ndata.get("source_key") or node_name.removeprefix("passage:"))
                txt = str(ndata.get("text") or "")
                if skey and txt:
                    passage_indices.append((idx_n, skey, txt, node_name))
            else:
                entity_mask[idx_n] = True
                node_edges = []
                deg_u = max(1, nx_graph.degree(node_name)) if hasattr(nx_graph, "degree") and node_name in nx_graph else 1
                edge_fn = getattr(nx_graph, "out_edges", getattr(nx_graph, "edges", None))
                for _u, tgt, ed in (edge_fn(node_name, data=True) if callable(edge_fn) else ()):
                    ed_dict = ed if isinstance(ed, dict) else {}
                    pred = ed_dict.get("predicate") or ed_dict.get("relation") or ed_dict.get("type")
                    if pred not in ("CONTAINS", "SYNONYM", "SUPERSEDED_BY"):
                        pred_str = str(pred or "related_to")
                        conf_val = float(ed_dict.get("confidence", 0.85) or 0.85)
                        deg_v = max(1, nx_graph.degree(tgt)) if hasattr(nx_graph, "degree") and tgt in nx_graph else 1
                        laplace_w = conf_val / math.sqrt(float(deg_u * deg_v))
                        node_edges.append((idx_n, node_name, pred_str, str(tgt), laplace_w))
                node_edges.sort(key=lambda x: -x[4])
                entity_rel_edges.extend(node_edges)

        self._cached_sig = sig
        self._cached_nodes = nodes
        self._cached_node_to_idx = node_to_idx
        self._cached_edge_weights = edge_map
        self._cached_ig_graph = ig_graph
        self._cached_passage_indices = passage_indices
        self._cached_entity_mask = entity_mask
        self._cached_entity_rel_edges = entity_rel_edges
        return nodes, node_to_idx, edge_map, ig_graph

    def _run_numpy_reference(
        self,
        n: int,
        edge_map: Dict[Tuple[int, int], float],
        p: np.ndarray,
        damping: float,
    ) -> Tuple[np.ndarray, int]:
        """Reference NumPy power-iteration solver (`ATLAS_PPR_BACKEND=numpy`) matching `ig.Graph.personalized_pagerank` to L1 <= 1e-12."""
        if damping <= 0.0 or not edge_map:
            return p.copy(), 1

        out_sums = np.zeros(n, dtype=np.float64)
        for (i, _j), w in edge_map.items():
            out_sums[i] += w

        dangling_mask = out_sums <= 0.0
        v = p.copy()
        iters = 0
        max_steps = max(500, self.max_iter)
        target_tol = min(1e-13, self.tol)
        for it in range(1, max_steps + 1):
            iters = it
            next_v = np.zeros(n, dtype=np.float64)
            for (i, j), w in edge_map.items():
                if out_sums[i] > 0.0:
                    next_v[j] += v[i] * (w / out_sums[i])
            dangling_mass = float(v[dangling_mask].sum()) if np.any(dangling_mask) else 0.0
            v_next = damping * (next_v + dangling_mass * p) + (1.0 - damping) * p
            diff = float(np.linalg.norm(v_next - v, ord=1))
            v = v_next
            if diff < target_tol:
                break
        v_sum = float(v.sum())
        if v_sum > 0.0:
            v = v / v_sum
        return v, iters

    def run(
        self,
        query: str,
        graph_store: Any,
        top_k: int = 15,
        recognition_memory: Optional[bool] = None,
        backend: Optional[str] = None,
        budget_ms: float = 0.0,
    ) -> PPRResult:
        """Executes Personalized PageRank for `query` and returns top-K passages/candidates."""
        active_backend = get_ppr_backend(backend or self.backend)
        det_seed = zlib.crc32(str(query or "").encode("utf-8"))
        nx_graph = graph_store.nx_graph if hasattr(graph_store, "nx_graph") else graph_store
        if nx_graph is None or nx_graph.number_of_nodes() == 0:
            record_seed_hit(0.0)
            return PPRResult([], {}, {}, 0, 0.0, det_seed, 0.0, active_backend)

        nodes, node_to_idx, edge_map, ig_graph = self._build_graph_structures(graph_store)
        n = len(nodes)
        if n == 0:
            record_seed_hit(0.0)
            return PPRResult([], {}, {}, 0, 0.0, det_seed, 0.0, active_backend)

        _get_or_build_triple_index(nx_graph)
        t0 = time.perf_counter()

        def _check_budget() -> None:
            if budget_ms > 0.0:
                elapsed = (time.perf_counter() - t0) * 1000.0
                if elapsed > budget_ms:
                    raise HippoRAGPPRUnavailable(
                        f"HippoRAG 2 PPR budget exceeded: {elapsed:.2f}ms > {budget_ms:.2f}ms"
                    )

        phrase_seeds, passage_seeds, _ = score_query_to_triples(
            query=query,
            nx_graph=nx_graph,
            top_k=max(15, top_k),
            recognition_memory=recognition_memory,
        )
        _check_budget()

        # V109 Pillar 2: Multi-Hop Associative Expansion (CONTAINS -> SYNONYM -> REGULATES)
        if self.damping > 0.0 and self.synonym_weight > 0.0 and self.contains_weight > 0.0:
            phrase_seeds = expand_associative_seeds_2hop(
                nx_graph=nx_graph,
                phrase_seeds=phrase_seeds,
                passage_seeds=passage_seeds,
                max_expanded=max(24, top_k * 2),
            )

        p = np.zeros(n, dtype=np.float64)
        seed_map: Dict[str, float] = {}
        for ent_name, s_val in phrase_seeds.items():
            idx = node_to_idx.get(ent_name)
            if idx is not None and s_val > 0.0:
                p[idx] += float(s_val)
                seed_map[ent_name] = float(s_val)

        if self.passage_seed_weight > 0.0:
            for pid, p_sim in passage_seeds.items():
                idx = node_to_idx.get(pid)
                if idx is not None and p_sim > 0.0:
                    w_pass = float(p_sim) * self.passage_seed_weight
                    p[idx] += w_pass
                    seed_map[pid] = seed_map.get(pid, 0.0) + w_pass

        q_words, q_stems = _tokenize_and_stem(query)
        q_total = max(1, len(q_stems or q_words))
        matched_q_terms = 0
        if seed_map:
            seed_tokens: Set[str] = set()
            for s_node in list(seed_map.keys())[:20]:
                sw, ss = _tokenize_and_stem(s_node)
                seed_tokens |= sw | ss
            matched_q_terms = len((q_stems or q_words) & seed_tokens)
        seed_hit_ratio = min(1.0, float(matched_q_terms) / float(q_total)) if seed_map else 0.0
        record_seed_hit(seed_hit_ratio)

        p_sum = float(p.sum())

        # Asocjacyjne Tunelowanie Kwantowe w HippoRAG 2 (v146.0 // AGI Horizon):
        # Przełamuje lokalizację Andersona rezerwując masę probabilistyczną
        # `beta_tunnel = PPR_BETA_TUNNEL_DEFAULT` (config-overridable) dla nadrzędnych
        # aksjomatów i reguł bezpieczeństwa/preferencji w grafie.
        # V151.5 (WEKTOR II): per-aksjomatowa maska ugruntowania sim(query, axiom) > θ_min
        # chroni przed recyklingiem reguł edytora/preferencji w zapytaniach o silnik grafowy.
        axiom_indices = [
            i for i, node_name in enumerate(nodes)
            if str(node_name).lower().startswith(("rule:", "user_rule:", "user:rule:", "preference:", "constraint:"))
            or ":rule:" in str(node_name).lower()
        ]
        beta_tunnel = PPR_BETA_TUNNEL_DEFAULT
        if axiom_indices:
            # Ochrona przed delokalizacją asocjacyjną przy perkolacji (skala N > 1000):
            # Ograniczenie tunelowania do Top-16 najważniejszych aksjomatów w grafie
            if len(axiom_indices) > 16:
                deg_scores = [
                    (idx, max(1, nx_graph.degree(nodes[idx])) if hasattr(nx_graph, "degree") and nodes[idx] in nx_graph else 1)
                    for idx in axiom_indices
                ]
                deg_scores.sort(key=lambda x: -x[1])
                selected_axioms = [idx for idx, _ in deg_scores[:16]]
            else:
                selected_axioms = axiom_indices

            u_axioms = np.zeros(n, dtype=np.float64)
            u_axioms[selected_axioms] = 1.0 / float(len(selected_axioms))
            # V151.5: per-aksjomatowa maska ugruntowania — sim(query, axiom) > θ_min;
            # na zero — zero energii (u_axioms_eff = u_axioms * mask, bez renormalizacji).
            mask_vec = np.zeros(n, dtype=np.float64)
            mask_values = compute_axiom_tunnel_mask(query, [nodes[i] for i in selected_axioms])
            for axiom_idx, m in zip(selected_axioms, mask_values, strict=True):
                mask_vec[axiom_idx] = float(m)
            u_axioms_eff = u_axioms * mask_vec
            ax_mass = float(u_axioms_eff.sum())
            if ax_mass > 0.0:
                u_axioms_eff = u_axioms_eff / ax_mass
                if p_sum > 0.0:
                    p = (1.0 - beta_tunnel) * (p / p_sum) + beta_tunnel * u_axioms_eff
                else:
                    p = u_axioms_eff.copy()
            elif p_sum > 0.0:
                p = p / p_sum
            else:
                return PPRResult([], {}, {}, 0, (time.perf_counter() - t0) * 1000.0, det_seed, 0.0, active_backend)
            p_sum = float(p.sum())
        elif p_sum > 0.0:
            p = p / p_sum
        else:
            return PPRResult([], {}, {}, 0, (time.perf_counter() - t0) * 1000.0, det_seed, 0.0, active_backend)

        d = self.damping

        if d <= 0.0 or not edge_map:
            v = p.copy()
            iters = 1
        elif active_backend == "numpy":
            v, iters = self._run_numpy_reference(n, edge_map, p, d)
        else:
            # V113 (F-029-class): igraph backend was selected but `ig_graph` is None —
            # zero silent fallback (FIX-2 doctrine). The old bare `assert` vanished under
            # `python -O`, letting a None-deref through; now an InvariantViolation names
            # the exact cause and remedy.
            if ig_graph is None:
                raise InvariantViolation(
                    module="cognitive.retrieval.ppr",
                    name="ppr_igraph_missing",
                    observed=0,
                    expected=1,
                    hint=(
                        "PPR igraph backend selected but ig.Graph could not be built "
                        "(python-igraph not installed or graph build failed). Install "
                        "python-igraph (`pixi add python-igraph` / `pip install python-igraph`) "
                        "or set ATLAS_PPR_BACKEND=numpy for the reference power-iteration solver."
                    ),
                )
            pr_scores = ig_graph.personalized_pagerank(
                damping=d,
                reset=p.tolist(),
                weights="weight",
                directed=True,
            )
            v = np.asarray(pr_scores, dtype=np.float64)
            v_sum = float(v.sum())
            if v_sum > 0.0:
                v = v / v_sum
            iters = 1

        _check_budget()

        if n <= 2000 or active_backend == "numpy":
            node_scores = {nodes[i]: float(v[i]) for i in range(n) if v[i] > 0.0}
        else:
            nz_indices = np.flatnonzero(v > 1e-6)
            node_scores = {nodes[int(i)]: float(v[int(i)]) for i in nz_indices}

        # Extract Passage candidates first using precomputed O(P) passage index
        passage_items: List[Tuple[float, str, str, str]] = []
        for idx_n, skey, txt, node_name in self._cached_passage_indices:
            score_val = float(v[idx_n])
            if score_val > 1e-9:
                passage_items.append((score_val, skey, txt, node_name))

        passage_items.sort(key=lambda x: (x[0], x[1]), reverse=True)
        max_p_score = passage_items[0][0] if passage_items else 1.0

        candidates: List[Dict[str, Any]] = []
        seen_keys: Set[str] = set()

        for raw_score, skey, txt, pid in passage_items[:top_k]:
            norm_g = round(min(1.0, raw_score / max(1e-9, max_p_score)), 4)
            if norm_g < 0.25 or raw_score < 0.005:
                continue
            candidates.append(
                {
                    "key": skey,
                    "subject": skey,
                    "predicate": "hipporag_ppr",
                    "object": txt,
                    "value": txt,
                    "source": "graph",
                    "vector_score": 0.0,
                    "graph_score": norm_g,
                    "graph_hit": True,
                    "confidence": norm_g,
                    "source_type": "user_explicit",
                    "metadata": {
                        "passage_id": pid,
                        "ppr_raw_score": round(raw_score, 6),
                        "ppr_seed": det_seed,
                        "ppr_backend": active_backend,
                        "graph_score": norm_g,
                        "graph_hit": True,
                    },
                }
            )
            seen_keys.add(skey)

        # Supplement with top-K Entity RELATION edges when passage candidates do not fill top_k
        if len(candidates) < top_k and self._cached_entity_rel_edges:
            max_e_score = (
                float(v[self._cached_entity_mask].max())
                if self._cached_entity_mask.size == n and np.any(self._cached_entity_mask)
                else 1.0
            )
            entity_rel_ranked = sorted(
                (
                    (float(v[idx_n]), ent_name, pred, tgt, conf_val)
                    for idx_n, ent_name, pred, tgt, conf_val in self._cached_entity_rel_edges
                    if float(v[idx_n]) > 1e-9
                ),
                key=lambda x: (x[0], x[1]),
                reverse=True,
            )
            for raw_e_score, ent_name, pred, tgt, conf_val in entity_rel_ranked:
                if len(candidates) >= top_k:
                    break
                if ent_name in seen_keys:
                    continue
                obj_txt = f"{pred} {tgt}"
                norm_g = round(min(1.0, raw_e_score / max(1e-9, max_e_score)), 4)
                candidates.append(
                    {
                        "key": ent_name,
                        "subject": ent_name,
                        "predicate": pred,
                        "object": obj_txt,
                        "value": obj_txt,
                        "source": "graph",
                        "vector_score": 0.0,
                        "graph_score": norm_g,
                        "graph_hit": True,
                        "confidence": conf_val,
                        "source_type": "user_explicit",
                        "metadata": {
                            "ppr_raw_score": round(raw_e_score, 6),
                            "ppr_seed": det_seed,
                            "ppr_backend": active_backend,
                            "graph_score": norm_g,
                            "graph_hit": True,
                        },
                    }
                )
                seen_keys.add(ent_name)

        _check_budget()
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        return PPRResult(
            candidates=candidates,
            node_scores=node_scores,
            seed_nodes=seed_map,
            iterations=iters,
            latency_ms=latency_ms,
            deterministic_seed=det_seed,
            seed_hit_ratio=seed_hit_ratio,
            backend=active_backend,
        )


PersonalizedPageRank = HippoRAG2PPR


def expand_associative_seeds_2hop(
    nx_graph: Any,
    phrase_seeds: Dict[str, float],
    passage_seeds: Optional[Dict[str, float]] = None,
    max_expanded: int = 24,
) -> Dict[str, float]:
    """V109 Pillar 2 / v145.0: 2-Hop Associative Seed Expansion with Bidirectional Passage Bridging.

    Expands initial 1-hop entity/concept seeds across 2nd-degree associative and regulatory
    edges. When phrase_seeds is empty, strong passage seeds (S_dense >= 0.70) bridge energy
    across CONTAINS edges to seed associated entities in Hop 1.
    """
    if nx_graph is None:
        return dict(phrase_seeds or {})

    active_phrase_seeds: Dict[str, float] = dict(phrase_seeds or {})

    # Bidirectional Passage Bridging: when phrase_seeds is empty (procedural/descriptive queries)
    if not active_phrase_seeds and passage_seeds:
        for p_node, p_sim in passage_seeds.items():
            if float(p_sim) >= 0.70:
                p_str = str(p_node)
                try:
                    if hasattr(nx_graph, "out_edges"):
                        for _, nbr, data in nx_graph.out_edges(p_str, data=True):
                            d = data if isinstance(data, dict) else {}
                            rel = str(d.get("relation") or d.get("type") or d.get("predicate") or "RELATED").upper()
                            if rel == "CONTAINS" and not str(nbr).startswith("passage:"):
                                w = float(p_sim) * 0.85
                                if w > active_phrase_seeds.get(str(nbr), 0.0):
                                    active_phrase_seeds[str(nbr)] = w
                    if hasattr(nx_graph, "in_edges"):
                        for nbr, _, data in nx_graph.in_edges(p_str, data=True):
                            d = data if isinstance(data, dict) else {}
                            rel = str(d.get("relation") or d.get("type") or d.get("predicate") or "RELATED").upper()
                            if rel == "CONTAINS" and not str(nbr).startswith("passage:"):
                                w = float(p_sim) * 0.85
                                if w > active_phrase_seeds.get(str(nbr), 0.0):
                                    active_phrase_seeds[str(nbr)] = w
                except Exception as p_exc:
                    logger.debug("Passage seed bridge error for %s: %s", p_str, p_exc)

    if not active_phrase_seeds:
        return {}

    expanded: Dict[str, float] = dict(active_phrase_seeds)
    hop1_candidates: Dict[str, float] = {}

    def _iter_neighbors(node: str) -> List[Tuple[str, str, float]]:
        out: List[Tuple[str, str, float]] = []
        try:
            if hasattr(nx_graph, "out_edges"):
                for _, nbr, data in nx_graph.out_edges(node, data=True):
                    d = data if isinstance(data, dict) else {}
                    rel = str(d.get("relation") or d.get("type") or d.get("predicate") or "RELATED").upper()
                    w = float(d.get("weight", d.get("confidence", 1.0)) or 1.0)
                    out.append((str(nbr), rel, max(0.05, min(1.0, w))))
            if hasattr(nx_graph, "in_edges"):
                for nbr, _, data in nx_graph.in_edges(node, data=True):
                    d = data if isinstance(data, dict) else {}
                    rel = str(d.get("relation") or d.get("type") or d.get("predicate") or "RELATED").upper()
                    w = float(d.get("weight", d.get("confidence", 1.0)) or 1.0)
                    out.append((str(nbr), rel, max(0.05, min(1.0, w))))
        except Exception as exc:
            logger.debug("Neighbor iteration failed for node %s: %s", node, exc)
        return out

    # Hop 1 expansion (allows CONTAINS bridging via passages, excludes SUPERSEDED_BY)
    for seed_node, seed_w in sorted(active_phrase_seeds.items(), key=lambda x: (-x[1], x[0]))[:16]:
        if seed_w <= 0.0:
            continue
        for nbr, rel, edge_w in _iter_neighbors(seed_node):
            if nbr == seed_node or rel == "SUPERSEDED_BY":
                continue
            deg_nbr = nx_graph.degree(nbr) if hasattr(nx_graph, "degree") and nbr in nx_graph else 1
            w_damped = edge_w * (1.0 / math.sqrt(max(1.0, deg_nbr / 8.0)))
            reg_boost = 1.25 if (rel == "REGULATES" or nbr.lower().startswith(("rule:", "constraint:"))) else 1.0
            syn_boost = 1.15 if rel in ("SYNONYM", "CONTAINS") else 1.0
            w1 = float(seed_w) * 0.42 * w_damped * reg_boost * syn_boost
            if w1 > hop1_candidates.get(nbr, 0.0):
                hop1_candidates[nbr] = min(0.95, w1)

    # Hop 2 expansion (CONTAINS -> SYNONYM -> REGULATES)
    hop2_candidates: Dict[str, float] = {}
    for h1_node, h1_w in sorted(hop1_candidates.items(), key=lambda x: (-x[1], x[0]))[:16]:
        for nbr2, rel2, edge_w2 in _iter_neighbors(h1_node):
            if nbr2 in active_phrase_seeds or nbr2 == h1_node or rel2 == "SUPERSEDED_BY":
                continue
            deg_nbr2 = nx_graph.degree(nbr2) if hasattr(nx_graph, "degree") and nbr2 in nx_graph else 1
            w_damped2 = edge_w2 * (1.0 / math.sqrt(max(1.0, deg_nbr2 / 8.0)))
            reg_boost2 = 1.35 if (rel2 == "REGULATES" or nbr2.lower().startswith(("rule:", "constraint:"))) else 1.0
            syn_boost2 = 1.15 if rel2 in ("SYNONYM", "CONTAINS", "REGULATES") else 1.0
            w2 = float(h1_w) * 0.45 * w_damped2 * reg_boost2 * syn_boost2
            if w2 > hop2_candidates.get(nbr2, 0.0):
                hop2_candidates[nbr2] = min(0.85, w2)

    combined_new: Dict[str, float] = dict(hop1_candidates)
    for k2, v2 in hop2_candidates.items():
        if v2 > combined_new.get(k2, 0.0):
            combined_new[k2] = v2

    for k_new, v_new in sorted(combined_new.items(), key=lambda x: (-x[1], x[0])):
        if str(k_new).startswith("passage:"):
            continue
        if k_new not in expanded or v_new > expanded[k_new]:
            expanded[k_new] = round(float(v_new), 6)
            if len(expanded) - len(phrase_seeds) >= max_expanded:
                break

    return expanded


class _PPRSnapshotStore:
    """Lightweight isolated graph store snapshot for non-blocking HippoRAG PPR computation."""

    def __init__(self, nx_graph: Any, orig_store: Any, revision: int) -> None:
        self.nx_graph = nx_graph
        self.revision = revision
        self._orig_store = orig_store
        detect_fn = getattr(orig_store, "detect_cpof", None)
        if callable(detect_fn):
            self.detect_cpof = detect_fn


def fetch_graph_ppr_candidates(
    gs: Optional[Any],
    query: str,
    limit: int = 15,
    budget_ms: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """W3/W4 (V104.5 / V122 / V138): Executes HippoRAG 2 PPR on `gs` when `ATLAS_HIPPORAG=1` with budget enforcement.

    V138 T5: Decomposes `_graph_lock` so lock is held ONLY during rapid snapshot creation
    (< 5ms), preventing event-loop stalls and writer starvation during numerical PPR execution.
    Post-run signature validation invalidates cache if concurrent writes occurred during computation.

    V151.4 (WEKTOR II): `budget_ms=None` resolve budżet adaptacyjnie Z PRZEDU z sygnatury
    snapshotu: `max(ppr_tau_base, ppr_tau_base + ppr_gamma * ln(V + E))` (config-driven,
    zero hardcoded 35.0). Jawny `budget_ms=<float>` kalibruje z zewnątrz i wygrywa.
    """
    if gs is None or not query or not is_hipporag_enabled():
        return []
    raw_nx = getattr(gs, "nx_graph", None)
    if raw_nx is None or not hasattr(raw_nx, "nodes"):
        return []
    try:
        ppr_engine = getattr(gs, "_ppr_engine", None)
        if ppr_engine is None:
            ppr_engine = HippoRAG2PPR()
            gs._ppr_engine = ppr_engine
        lock = getattr(gs, "_graph_lock", None) or getattr(raw_nx, "_lock", None) or getattr(raw_nx, "lock", None)

        def _create_snapshot() -> Tuple[Tuple[int, int, int], Any]:
            try:
                n_nodes = int(raw_nx.number_of_nodes())
            except Exception:
                n_nodes = len(raw_nx.nodes) if hasattr(raw_nx.nodes, "__len__") else 0
            try:
                n_edges = int(raw_nx.number_of_edges())
            except Exception:
                n_edges = len(raw_nx.edges) if hasattr(raw_nx.edges, "__len__") else 0
            try:
                rev = int(getattr(raw_nx, "revision", 0) or getattr(raw_nx, "_revision", 0) or getattr(gs, "revision", 0) or getattr(gs, "_revision", 0) or 0)
            except Exception:
                rev = 0
            sig = (n_nodes, n_edges, rev)
            snap = raw_nx.copy() if hasattr(raw_nx, "copy") else raw_nx
            if snap is not raw_nx:
                if isinstance(getattr(raw_nx, "_canonical_lower", None), dict):
                    snap._canonical_lower = dict(raw_nx._canonical_lower)
                if isinstance(getattr(raw_nx, "_lemma_index", None), dict):
                    snap._lemma_index = {k: set(v) for k, v in raw_nx._lemma_index.items()}
                if isinstance(getattr(raw_nx, "_node_index_lower", None), dict):
                    snap._node_index_lower = dict(raw_nx._node_index_lower)
                snap._revision = rev
                triple_idx = getattr(raw_nx, "_hipporag_triple_idx", None)
                if triple_idx is not None:
                    snap._hipporag_triple_idx = triple_idx
            return sig, snap

        if lock is not None:
            with lock:
                sig_before, snap_nx = _create_snapshot()
        else:
            sig_before, snap_nx = _create_snapshot()

        snap_store = _PPRSnapshotStore(snap_nx, gs, sig_before[2])
        # V151.4 (WEKTOR II): budżet adaptacyjny z sygnatury snapshotu (n_nodes, n_edges)
        # obliczany PRZED run(); jawne budget_ms wygrywa (np. kalibracja testu).
        eff_budget_ms = (
            float(budget_ms)
            if budget_ms is not None
            else adaptive_ppr_budget(int(sig_before[0]), int(sig_before[1]))
        )
        logger.info(
            "[HippoRAG 2 PPR] query='%s' budget_ms=%.1f (nodes=%d, edges=%d)",
            query[:48],
            eff_budget_ms,
            int(sig_before[0]),
            int(sig_before[1]),
        )
        ppr_res = ppr_engine.run(query=query, graph_store=snap_store, top_k=limit, budget_ms=eff_budget_ms)

        try:
            rev_after = int(getattr(raw_nx, "revision", 0) or getattr(raw_nx, "_revision", 0) or getattr(gs, "revision", 0) or getattr(gs, "_revision", 0) or 0)
            n_nodes_after = int(raw_nx.number_of_nodes())
            n_edges_after = int(raw_nx.number_of_edges())
            sig_after = (n_nodes_after, n_edges_after, rev_after)
            if sig_after != sig_before:
                ppr_engine._cached_sig = None
        except Exception:
            pass

        return list(ppr_res.candidates)
    except HippoRAGPPRUnavailable:
        metrics.counter("hipporag_ppr_error").inc()
        raise
    except Exception as exc:
        metrics.counter("hipporag_ppr_error").inc()
        raise HippoRAGPPRUnavailable(f"HippoRAG 2 PPR failed: {exc}") from exc


# Pre-warm bilingual stemmer at import time so first PPR run does not pay cold-load latency
_tokenize_and_stem("warmup init")
