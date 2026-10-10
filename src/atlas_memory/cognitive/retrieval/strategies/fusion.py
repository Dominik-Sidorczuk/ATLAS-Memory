"""Fusion strategies for multi-signal retrieval: Static-Anchor DBSF, RRF, and Entropy Filter."""
from __future__ import annotations

import datetime
import logging
import math
import re
import time
import zlib
from typing import Any, Dict, List, Optional, Set, Tuple

from atlas_memory.cognitive.calibration_engine import is_symbolic_or_identifier
from atlas_memory.cognitive.retrieval.strategies.base import BaseScoringStrategy
from atlas_memory.cognitive.retrieval.strategies.dense import DenseCosineStrategy
from atlas_memory.cognitive.retrieval.strategies.graph import (
    GENERIC_KEY_PARTS,
    GraphSynergyStrategy,
)
from atlas_memory.core.linguistics import _bilingual_stem, is_non_salient_key_token

logger = logging.getLogger("atlas.cognitive.retrieval.strategies.fusion")

RRF_K: int = 60

# Static-Anchor DBSF Constants
DBSF_ALPHA: float = 0.35
DBSF_ALPHA_SYMBOLIC: float = 0.30
DBSF_ALPHA_DENSE: float = 0.65
DBSF_ALPHA_GRAPH: float = 0.10
DBSF_ALPHA_BM25: float = 0.25
DBSF_DENSE_MU: float = 0.50
DBSF_DENSE_BETA: float = 12.0
DBSF_BM25_SATURATION_PER_TERM: float = 6.0


def compute_content_fingerprint(text: str) -> str:
    """Compute content fingerprint for candidate deduplication with anti-collision hash."""
    t = str(text).strip()
    if ":" in t and len(t.split(":", 1)[0].split()) <= 3:
        t = t.split(":", 1)[1].strip()
    t = re.sub(r":stated_memory\b", "", t, flags=re.I)
    t_clean = re.sub(r"\s+", " ", t.strip().lower())
    tokens = [w for w in re.findall(r"\w+", t_clean) if len(w) >= 3 or w.isdigit()]
    prefix = "_".join(tokens[:8])
    content_hash = format(zlib.crc32(t_clean.encode("utf-8")), "08x")
    return f"{prefix}::{content_hash}" if prefix else content_hash


def get_record_authority_score(subj: str) -> int:
    """Returns integer authority score in [0, 110] based on subject namespace."""
    s = str(subj).lower()
    if s.startswith(("rule:", "user_rule:", "user:rule:", "preference:", "constraint:")) or ":rule:" in s:
        return 110
    if s.startswith(("hermes:user_md:", "hermes:memory_md:")):
        return 100
    if s.startswith("mnemosyne:canonical_facts:"):
        return 70
    if s.startswith("mnemosyne:mnemosyne_working_memory:"):
        return 30
    return 80 if s.startswith("fact:") else (90 if (s.startswith("database_") or ":" not in s) else 50)


def get_authority_score(subj: str) -> float:
    """Returns float authority multiplier for record ranking."""
    s = str(subj).lower()
    if s.startswith(("rule:", "user_rule:", "user:rule:", "preference:", "constraint:")) or ":rule:" in s:
        return 1.10
    if s.startswith(("hermes:user_md:", "hermes:memory_md:")):
        return 1.00
    return 0.70 if s.startswith("fact:") else 0.50


def compute_record_rank(record: Any, query: str = "", now: Optional[float] = None) -> float:
    """Computes base epistemic priority score for a record."""
    meta = record.get("metadata", {}) if isinstance(record, dict) else getattr(record, "metadata", {})
    if bool(record.get("is_superseded") if isinstance(record, dict) else getattr(record, "is_superseded", False)) or meta.get("is_superseded"):
        return -5.0
    st = str(record.get("source_type", "user_explicit") if isinstance(record, dict) else getattr(record, "source_type", "user_explicit")).lower()
    vw = {"user_explicit": 1.0, "tool_output": 0.85, "external_doc": 0.65, "agent_inference": 0.50}.get(st, 0.5)
    c = float(record.get("confidence", 1.0) if isinstance(record, dict) else getattr(record, "confidence", 1.0))
    current_time = now if now is not None else time.time()
    ts = float(record.get("timestamp", 0.0) if isinstance(record, dict) else getattr(record, "timestamp", 0.0))
    age_seconds = max(0.0, current_time - ts) if ts > 0.0 else 0.0
    subj = record.get("subject", "") if isinstance(record, dict) else getattr(record, "subject", "")
    auth = get_authority_score(str(subj))
    s_low = str(subj).lower()
    rank = 0.40 * (vw * c) + 0.25 * auth
    if (s_low.startswith(("rule:", "user_rule:", "user:rule:", "preference:", "constraint:")) or ":rule:" in s_low) and (st == "user_explicit" or c >= 0.99):
        rank += 1.0 / (1.0 + 0.15 * math.log1p(age_seconds / 86400.0))
    else:
        rank += (0.20 * (vw * c)) / (1.0 + 0.05 * (age_seconds / 3600.0))
    return round(rank, 4)


def is_trusted_lexical_match(
    query: str,
    cand: Dict[str, Any],
    q_stems: List[str],
    cand_doc_tokens: Set[str],
) -> bool:
    """Determines whether a lexical match is sufficiently trusted."""
    s_cand = str(cand.get("subject", "")).lower()
    o_cand = str(cand.get("object", "")).lower()
    q_lower = query.lower().strip()

    if q_lower and (q_lower in s_cand or q_lower in o_cand):
        return True
    if cand.get("source") == "graph":
        return True
    if s_cand == q_lower or (q_lower.startswith(s_cand) and len(s_cand) >= 4):
        return True

    subject_raw_parts = re.findall(r"[a-z0-9]+", s_cand)
    subject_entities = [p for p in subject_raw_parts if not is_non_salient_key_token(p)]
    if any(e in q_lower for e in subject_entities):
        return True

    q_stems_set = set(q_stems)
    if any(_bilingual_stem(e) in q_stems_set for e in subject_entities):
        return True

    matched = len(q_stems_set & cand_doc_tokens)
    tot = len(q_stems_set)
    cov = (matched / tot) if tot > 0 else 0.0
    if cov >= 0.50 and tot >= 2:
        return True
    if tot == 1 and matched == 1:
        return True

    return False


def compute_tiebreaks(c: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Computes authority and confidence tiebreak bonuses."""
    st = str(c.get("source_type") or "user_explicit").lower()
    vw = {"user_explicit": 1.0, "tool_output": 0.85, "external_doc": 0.65, "agent_inference": 0.50}.get(st, 0.5)
    conf = float(c.get("confidence", 1.0))
    auth = get_record_authority_score(str(c.get("subject", ""))) / 100.0
    auth_tiebreak = 0.01 * (auth / 1.10)
    conf_tiebreak = 0.005 * (vw * conf)
    return auth, vw, auth_tiebreak, conf_tiebreak


def write_metadata_mirror(c: Dict[str, Any]) -> None:
    """Mirrors scores and relevance reason inside candidate metadata."""
    c["metadata"] = {
        **(c.get("metadata") or {}),
        "score": c["score"],
        "relevance_reason": c["relevance_reason"],
        "dense_score": c["dense_score"],
        "sparse_score": c["sparse_score"],
        "graph_score": c["graph_score"],
        "graph_hit": c["graph_hit"],
    }


def _assemble_rrf_score(
    c: Dict[str, Any], vs: float, ls: float, gs: float, is_trusted: bool,
    rank_vec_id: int, rank_lex_id: int, auth: float, auth_tiebreak: float, conf_tiebreak: float,
    is_graph_cand: Optional[bool] = None,
) -> None:
    rv_norm = ((RRF_K + 1) / (RRF_K + rank_vec_id)) if vs > 0.0 else 0.0
    rb_norm = ((RRF_K + 1) / (RRF_K + rank_lex_id)) if ls > 0.0 else 0.0
    if vs > 0.0 and ls > 0.0:
        rrf = 0.5 * (vs * rv_norm + min(1.0, ls) * rb_norm) + 0.02
    elif vs > 0.0:
        rrf = vs * rv_norm
    else:
        rrf = min(1.0, ls) * rb_norm if is_trusted else (0.15 * min(1.0, ls) * rb_norm)
    if gs > 0.0:
        rrf = min(1.0, rrf + DBSF_ALPHA_GRAPH * gs)
    rule_tiebreak = 0.035 if (auth >= 1.0 and (vs >= 0.65 or (ls >= 0.25 and (vs > 0.0 or is_trusted)))) else 0.0
    if is_graph_cand is None:
        is_graph_cand = bool(gs > 0.0 or c.get("graph_hit") or c.get("source") == "graph")
    c["score"] = round(min(1.0, rrf + auth_tiebreak + conf_tiebreak + rule_tiebreak), 4)
    c["rrf_score"] = round(rrf, 4)
    c["dense_score"] = round(float(vs), 4)
    c["sparse_score"] = round(float(ls), 4)
    c["graph_score"] = round(float(gs), 4)
    c["graph_hit"] = is_graph_cand
    c["relevance_reason"] = f"dense(rank={rank_vec_id},sim={vs:.2f})+bm25(rank={rank_lex_id},sim={ls:.2f})+graph({gs:.2f})"


def _write_dbsf_scores(
    c: Dict[str, Any], fused: float, n_dense: float, n_lex: float, gs: float,
    vs: float, is_graph_cand: bool, auth_tiebreak: float, conf_tiebreak: float, rule_tiebreak: float,
) -> None:
    c["score"] = round(min(1.0, fused + auth_tiebreak + conf_tiebreak + rule_tiebreak), 4)
    c["rrf_score"] = round(fused, 4)
    c["dense_score"] = round(float(n_dense), 4)
    c["sparse_score"] = round(float(n_lex), 4)
    c["graph_score"] = round(float(gs), 4)
    c["graph_hit"] = is_graph_cand
    c["relevance_reason"] = f"dbsf(dense={n_dense:.2f},lex={n_lex:.2f},graph={gs:.2f},sim={vs:.2f})"


class RrfStrategy(BaseScoringStrategy):
    """Reciprocal Rank Fusion (RRF) Strategy."""

    @staticmethod
    def _extract_features(c: Dict[str, Any]) -> Tuple[float, float, float, float, Set[str], bool]:
        if hasattr(c, "pop"):
            vs = c.pop("_vec_score", None)
            if vs is None:
                vs = c.get("vector_score", 0.0)
            ls = c.pop("_lex_score", None)
            if ls is None:
                ls = c.get("bm25_score", 0.0)
            gs = c.pop("_graph_score", None)
            if gs is None:
                gs = c.get("graph_score", 0.0)
            raw_bm25 = c.pop("_raw_bm25", None)
            if raw_bm25 is None:
                raw_bm25 = c.get("raw_bm25", 0.0)
            cand_tokens_set = c.pop("_doc_tokens_set", None)
            if cand_tokens_set is None:
                cand_tokens_set = c.get("doc_tokens", set())
        else:
            vs = getattr(c, "vector_score", 0.0)
            ls = getattr(c, "bm25_score", 0.0)
            gs = getattr(c, "graph_score", 0.0)
            raw_bm25 = getattr(c, "raw_bm25", 0.0)
            cand_tokens_set = getattr(c, "doc_tokens", set())
        is_graph_cand = bool(gs > 0.0 or (hasattr(c, "get") and c.get("graph_hit")) or (hasattr(c, "get") and c.get("source") == "graph"))
        if is_graph_cand and gs <= 0.0:
            gs = 0.50
        return float(vs or 0.0), float(ls or 0.0), float(gs or 0.0), float(raw_bm25 or 0.0), cand_tokens_set, is_graph_cand

    @staticmethod
    def _write_zero_signal(c: Dict[str, Any]) -> None:
        c["score"] = c["rrf_score"] = 0.0
        c["dense_score"] = 0.0
        c["sparse_score"] = 0.0
        c["graph_score"] = 0.0
        c["graph_hit"] = False
        c["relevance_reason"] = "no_match(sim=0.0)"

    def score(
        self,
        query: str,
        candidates: List[Any],
        rank_vec: Optional[Dict[int, int]] = None,
        rank_lex: Optional[Dict[int, int]] = None,
        **kwargs: Any,
    ) -> None:
        """Scores candidates using Reciprocal Rank Fusion."""
        q_lower = query.strip().lower()
        q_clean = q_lower.replace("_", " ").replace(":", " ").replace("-", " ")
        q_stems = [_bilingual_stem(w) for w in re.findall(r"[^\W_]+", q_clean) if len(w) >= 3]

        rv_map = rank_vec or {}
        rl_map = rank_lex or {}

        for c in candidates:
            vs, ls, gs, _raw_bm25, cand_tokens_set, is_graph_cand = self._extract_features(c)
            if vs <= 0.0 and ls <= 0.0 and gs <= 0.0:
                self._write_zero_signal(c)
                continue

            is_trusted = is_trusted_lexical_match(query, c, q_stems, cand_tokens_set)
            auth, _vw, auth_tiebreak, conf_tiebreak = compute_tiebreaks(c)

            rank_vec_id = rv_map.get(id(c), len(rv_map) + 1)
            rank_lex_id = rl_map.get(id(c), len(rl_map) + 1)

            _assemble_rrf_score(
                c, vs, ls, gs, is_trusted, rank_vec_id, rank_lex_id,
                auth, auth_tiebreak, conf_tiebreak, is_graph_cand=is_graph_cand,
            )
            write_metadata_mirror(c)

    def score_candidate(
        self,
        c: Dict[str, Any],
        query: str,
        q_stems: List[str],
        q_lower: str,
        rank_vec: Dict[int, int],
        rank_lex: Dict[int, int],
    ) -> None:
        """Scores an individual candidate dictionary in-place using RRF."""
        self.score(query, [c], rank_vec=rank_vec, rank_lex=rank_lex)


class StaticAnchorDbsfFusion(BaseScoringStrategy):
    """Static-Anchor Continuous 3-Signal DBSF Fusion Strategy (dense ⊕ lex ⊕ graph)."""

    def __init__(self) -> None:
        self.dense_strategy = DenseCosineStrategy()

    normalize_dense = staticmethod(DenseCosineStrategy.normalize_dense)

    def score(
        self,
        query: str,
        candidates: List[Any],
        rank_vec: Optional[Dict[int, int]] = None,
        rank_lex: Optional[Dict[int, int]] = None,
        **kwargs: Any,
    ) -> None:
        """Fuses dense, lexical, and graph signals into final DBSF continuous scores."""
        q_lower = query.strip().lower()
        q_clean = q_lower.replace("_", " ").replace(":", " ").replace("-", " ")
        q_stems = [_bilingual_stem(w) for w in re.findall(r"[^\W_]+", q_clean) if len(w) >= 3]

        for c in candidates:
            vs, ls, gs, raw_bm25, cand_tokens_set, is_graph_cand = RrfStrategy._extract_features(c)
            if vs <= 0.0 and ls <= 0.0 and gs <= 0.0:
                RrfStrategy._write_zero_signal(c)
                continue

            is_trusted = is_trusted_lexical_match(query, c, q_stems, cand_tokens_set)
            auth, _vw, auth_tiebreak, conf_tiebreak = compute_tiebreaks(c)

            is_sym, eff_alpha, matched_stems, _query_cov, n_dense, n_lex = self._lex_signals(
                query, q_stems, cand_tokens_set, vs, ls, raw_bm25
            )
            s_cand_low = str(c.get("subject", "") if hasattr(c, "get") else getattr(c, "subject", "")).lower()
            ungrounded_scoped_entity = GraphSynergyStrategy.is_ungrounded_scoped_entity(
                s_cand_low, q_stems, matched_stems, ls, is_trusted, gs
            )
            is_daemon_no_lex = bool(c.get("is_daemon_no_lex") or (c.get("metadata") or {}).get("is_daemon_no_lex"))
            fused = self._combine_base_fused(
                vs, ls, n_dense, n_lex, eff_alpha, is_trusted,
                ungrounded_scoped_entity, matched_stems, q_stems,
                is_daemon_no_lex=is_daemon_no_lex,
            )
            fused = self._apply_graph_synergy(
                c, fused, vs, ls, gs, n_dense, n_lex,
                matched_stems, q_stems, is_sym, is_trusted, eff_alpha,
            )
            rule_tiebreak = self._dbsf_rule_tiebreak(auth, vs, ls, gs, is_trusted, ungrounded_scoped_entity)
            ttl_factor = self._compute_ttl_factor(s_cand_low, c, matched_stems, q_stems, n_dense, gs)
            fused = fused * ttl_factor

            _write_dbsf_scores(
                c, fused, n_dense, n_lex, gs, vs, is_graph_cand,
                auth_tiebreak, conf_tiebreak, rule_tiebreak,
            )
            write_metadata_mirror(c)

    def _lex_signals(
        self,
        query: str,
        q_stems: List[str],
        cand_tokens_set: Set[str],
        vs: float,
        ls: float,
        raw_bm25: float,
    ) -> Tuple[bool, float, int, float, float, float]:
        is_sym = is_symbolic_or_identifier(query)
        eff_alpha = DBSF_ALPHA_SYMBOLIC if is_sym else DBSF_ALPHA
        matched_stems = sum(1 for qs in q_stems if qs in cand_tokens_set)
        query_cov = (matched_stems / max(1, len(q_stems))) if q_stems else 1.0
        n_dense = self.dense_strategy.normalize_dense(vs)
        raw_lex_norm = raw_bm25 / max(DBSF_BM25_SATURATION_PER_TERM, len(q_stems) * DBSF_BM25_SATURATION_PER_TERM)
        if len(q_stems) >= 3 and matched_stems <= 1 and n_dense < 0.28:
            n_lex = min(1.0, max(ls, raw_lex_norm) * query_cov)
        else:
            n_lex = min(1.0, max(ls * query_cov, raw_lex_norm))
        return is_sym, eff_alpha, matched_stems, query_cov, n_dense, n_lex

    @staticmethod
    def _is_ungrounded_scoped_entity(
        s_cand_low: str,
        q_stems: List[str],
        matched_stems: int,
        ls: float,
        is_trusted: bool,
        gs: float,
    ) -> bool:
        return GraphSynergyStrategy.is_ungrounded_scoped_entity(
            s_cand_low, q_stems, matched_stems, ls, is_trusted, gs
        )

    @staticmethod
    def _combine_base_fused(
        vs: float,
        ls: float,
        n_dense: float,
        n_lex: float,
        eff_alpha: float,
        is_trusted: bool,
        ungrounded_scoped_entity: bool,
        matched_stems: int,
        q_stems: List[str],
        is_daemon_no_lex: bool = False,
    ) -> float:
        if vs > 0.0 and ls > 0.0:
            base_fused = eff_alpha * n_dense + (1.0 - eff_alpha) * n_lex
            if is_trusted:
                if vs >= 0.78:
                    fused = max(base_fused, n_dense)
                elif vs <= 0.74:
                    fused = base_fused
                else:
                    t = (vs - 0.74) / 0.04
                    s_blend = 3.0 * (t ** 2) - 2.0 * (t ** 3)
                    fused = (1.0 - s_blend) * base_fused + s_blend * max(base_fused, n_dense)
            else:
                fused = base_fused
            if is_daemon_no_lex:
                sig_gamma = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, (vs - 0.75) / 0.04))))
                fused = fused * sig_gamma
            return fused
        if vs > 0.0:
            if ungrounded_scoped_entity:
                fused = 0.15 * n_dense
            elif matched_stems == 0 and not is_trusted and len(q_stems) >= 2:
                fused = eff_alpha * n_dense
            elif vs <= 0.60:
                fused = eff_alpha * n_dense
            elif vs >= 0.80:
                fused = 1.0 * n_dense
            else:
                t = (vs - 0.60) / 0.20
                s_curve = 3.0 * (t ** 2) - 2.0 * (t ** 3)
                w_dense = eff_alpha + (1.0 - eff_alpha) * s_curve
                fused = w_dense * n_dense
            if is_daemon_no_lex:
                sig_gamma = 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, (vs - 0.75) / 0.04))))
                fused = fused * sig_gamma
            return fused
        return n_lex if is_trusted else ((1.0 - eff_alpha) * n_lex)

    @staticmethod
    def _apply_graph_synergy(
        c: Dict[str, Any],
        fused: float,
        vs: float,
        ls: float,
        gs: float,
        n_dense: float,
        n_lex: float,
        matched_stems: int,
        q_stems: List[str],
        is_sym: bool,
        is_trusted: bool,
        eff_alpha: float,
    ) -> float:
        if gs <= 0.0:
            return fused
        weak_lex_grounding = (
            matched_stems == 0
            or (len(q_stems) >= 3 and matched_stems <= 1 and not is_sym)
            or (n_lex < 0.22 and not is_sym)
        )
        if weak_lex_grounding:
            damped_val = min(0.18, 0.20 * gs + 0.50 * n_lex)
            if vs <= 0.0 and ls <= 0.0:
                full_val = min(0.22, 0.30 * gs) if not is_sym else ((DBSF_ALPHA_DENSE + DBSF_ALPHA_GRAPH) * gs)
            elif vs <= 0.0:
                full_val = (1.0 - eff_alpha) * n_lex + 2.0 * DBSF_ALPHA_GRAPH * gs
            else:
                full_val = (1.0 - DBSF_ALPHA_GRAPH) * fused + DBSF_ALPHA_GRAPH * max(gs, fused) + 0.02 * gs
            full_val = min(1.0, full_val)

            if n_dense <= 0.10:
                c["_guard_damped"] = "ungrounded_graph"
                return damped_val
            elif n_dense >= 0.16:
                return full_val
            else:
                t = (n_dense - 0.10) / 0.06
                s_ramp = 3.0 * (t ** 2) - 2.0 * (t ** 3)
                c["_guard_damped"] = "ungrounded_graph_ramped"
                return (1.0 - s_ramp) * damped_val + s_ramp * full_val
        if vs <= 0.0 and ls <= 0.0:
            fused = min(0.22, 0.30 * gs) if not is_sym else ((DBSF_ALPHA_DENSE + DBSF_ALPHA_GRAPH) * gs)
        elif vs <= 0.0:
            if is_sym and is_trusted:
                fused = max(fused, DBSF_ALPHA_BM25 * n_lex + (DBSF_ALPHA_DENSE + DBSF_ALPHA_GRAPH) * gs)
            else:
                fused = max(fused, (1.0 - eff_alpha) * n_lex + 2.0 * DBSF_ALPHA_GRAPH * gs)
        else:
            fused = (1.0 - DBSF_ALPHA_GRAPH) * fused + DBSF_ALPHA_GRAPH * max(gs, fused) + 0.02 * gs
        return min(1.0, fused)

    @staticmethod
    def _dbsf_rule_tiebreak(
        auth: float,
        vs: float,
        ls: float,
        gs: float,
        is_trusted: bool,
        ungrounded_scoped_entity: bool,
    ) -> float:
        if ungrounded_scoped_entity and gs < 0.70:
            return 0.0
        return 0.035 if (auth >= 1.0 and (vs >= 0.65 or gs >= 0.70 or (ls > 0.0 and (vs > 0.0 or gs > 0.0 or is_trusted)))) else 0.0

    @staticmethod
    def _compute_ttl_factor(
        s_cand_low: str,
        c: Dict[str, Any],
        matched_stems: int,
        q_stems: List[str],
        n_dense: float,
        gs: float,
    ) -> float:
        ttl_factor = 1.0
        if s_cand_low.endswith((":episodic_event", ":working_fact", ":episodic")) or s_cand_low.startswith(("fact:task__", "fact:conversation")):
            age_days = 0.0
            ts_val = float(c.get("timestamp", 0.0) or (c.get("metadata") or {}).get("timestamp", 0.0) or 0.0)
            if ts_val > 1e8:
                age_days = max(0.0, (time.time() - ts_val) / 86400.0)
            date_m = re.search(r"(202[4-9])[-_](0[1-9]|1[0-2])[-_](0[1-9]|[12][0-9]|3[01])", s_cand_low + " " + str(c.get("object", ""))[:120])
            if date_m:
                try:
                    dt_obj = datetime.datetime(int(date_m.group(1)), int(date_m.group(2)), int(date_m.group(3)), tzinfo=datetime.timezone.utc)
                    parsed_days = max(0.0, (time.time() - dt_obj.timestamp()) / 86400.0)
                    age_days = max(age_days, parsed_days)
                except Exception:
                    logger.debug("Date parse fallback in TTL calculation")
            if age_days > 3.0:
                ttl_factor = max(0.45, math.exp(-0.035 * (age_days - 3.0)))
        if len(q_stems) >= 3 and matched_stems <= 1 and n_dense < 0.28 and gs <= 0.0:
            ttl_factor *= 0.55
        return ttl_factor


class EntropySalienceFilter:
    """Adaptive Shannon Entropy Salience Filter (V109 Pillar 3)."""

    def __init__(self, threshold: float = 0.0, default_threshold: float = 0.0) -> None:
        self.default_threshold = threshold if threshold > 0.0 else default_threshold

    @staticmethod
    def compute_normalized_entropy(scores: List[float]) -> float:
        """Computes normalized Shannon entropy H(p) / ln(K) in [0.0, 1.0] over positive candidate scores."""
        pos = [float(s) for s in scores if float(s) > 1e-6]
        k = len(pos)
        if k < 2:
            return 0.0
        total = sum(pos)
        if total <= 0.0:
            return 0.0
        h = 0.0
        for s in pos:
            p = s / total
            if p > 0.0:
                h -= p * math.log(p)
        max_h = math.log(float(k))
        if max_h <= 0.0:
            return 0.0
        return min(1.0, max(0.0, h / max_h))

    def filter(
        self,
        scored_candidates: List[Dict[str, Any]],
        base_threshold: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Filters candidates based on entropy-adjusted threshold and salience cliff."""
        eff_base = self.default_threshold if base_threshold is None else float(base_threshold)
        if not scored_candidates:
            return [], float(eff_base)

        scores = [float(c.get("score", 0.0)) for c in scored_candidates]
        max_score = max(scores, default=0.0)
        pos = [float(s) for s in scores if float(s) > 1e-6]
        k = len(pos)
        h_norm = self.compute_normalized_entropy(scores)

        # Small-sample regularization preserving entropy dynamic range (v145.0)
        if k in (2, 3):
            h_eff = h_norm
        elif k >= 4:
            h_eff = (1.0 - 3.0 / float(k + 3)) * h_norm + (3.0 / float(k + 3)) * 0.5
        else:
            h_eff = 0.0

        if k >= 4 and h_eff >= 0.85:
            eff_threshold = max(float(eff_base), 0.55 * max_score)
        elif k >= 4 and h_eff >= 0.70:
            eff_threshold = max(float(eff_base), 0.40 * max_score)
        else:
            eff_threshold = float(eff_base)

        salience_cliff = 0.50 * max_score if k >= 4 else 0.0
        filtered: List[Dict[str, Any]] = []

        for c in scored_candidates:
            s = float(c.get("score", 0.0))
            if s < eff_threshold:
                continue

            subj = str(c.get("subject", "")).lower()
            is_protected_rule = (
                subj.startswith(("rule:", "user_rule:", "user:rule:", "preference:", "constraint:"))
                or ":rule:" in subj
            )

            d_score = float(c.get("dense_score", 0.0))
            l_score = float(c.get("sparse_score", 0.0))
            g_score = float(c.get("graph_score", 0.0))

            if (
                not is_protected_rule
                and g_score > 0.0
                and d_score < 0.12
                and l_score < 0.08
                and (s < salience_cliff or s < max(eff_threshold, 0.28))
            ):
                continue

            filtered.append(c)

        return filtered, round(eff_threshold, 4)


__all__ = [
    "DBSF_ALPHA",
    "DBSF_ALPHA_BM25",
    "DBSF_ALPHA_DENSE",
    "DBSF_ALPHA_GRAPH",
    "DBSF_ALPHA_SYMBOLIC",
    "DBSF_BM25_SATURATION_PER_TERM",
    "DBSF_DENSE_BETA",
    "DBSF_DENSE_MU",
    "EntropySalienceFilter",
    "GENERIC_KEY_PARTS",
    "RRF_K",
    "RrfStrategy",
    "StaticAnchorDbsfFusion",
    "compute_content_fingerprint",
    "compute_record_rank",
    "compute_tiebreaks",
    "get_authority_score",
    "get_record_authority_score",
    "is_trusted_lexical_match",
    "write_metadata_mirror",
    "_assemble_rrf_score",
    "_write_dbsf_scores",
]
