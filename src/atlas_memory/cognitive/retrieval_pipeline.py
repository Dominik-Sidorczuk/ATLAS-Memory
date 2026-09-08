"""
ATLAS Cognitive Layer: High-Precision Retrieval & Gating Pipeline (V27+ Architecture).

Implements:
1. Intent and continuation query gating (should_retrieve, continuation pattern matching).
2. Active hot working memory injection from HotBufferPolicy on empty or continuation queries.
3. Veracity-First Ranking with Dynamic Recency Prior:
   - Veracity weights: user_explicit (1.0) > tool_output (0.85) > external_doc (0.65) > agent_inference (0.50).
   - Authority scoring: system/rules > user > doc > agent.
   - Logarithmic Recency Prior for intra-tier freshness:
     delta_days = max(0.0, (now - rec_ts) / 86400.0)
     rule_bonus = 1.0 / (1.0 + 0.15 * math.log1p(delta_days))
     rank += rule_bonus for explicit user rules.
   - Superseded belief suppression: if meta.get('is_superseded') or rec.is_superseded -> rank = -5.0.
4. 0/1 Knapsack token budget packing for context fitting.
5. Formatted context block markdown generator ("## ATLAS Cognitive Context").
"""

from __future__ import annotations

import logging
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from atlas_memory.cognitive.hot_buffer import (
    HotBufferPolicy,
    extract_effective_session,
    is_conversational_churn,
    is_session_accessible,
)
from atlas_memory.cognitive.models import (
    HERMES_DEFAULT_SESSION,
    RetrievalContext,
    RetrievalRecord,
)
from atlas_memory.extensions.compactor import ContextCompactor

logger = logging.getLogger(__name__)

CONTINUATION_PATTERN = re.compile(
    r"^(?:kontynuuj|kontynuuj działanie|kontynuacja|dalej|continue|go on|proceed|next step|kolejny krok)[.!?,\s]*(?:działanie|prace|zadanie|krok|dalej)?$",
    re.IGNORECASE,
)

TRIVIAL_PROMPT_PATTERN = re.compile(
    r"^(?:cześć|czesc|hej|hejka|witaj|witajcie|siema|siemanko|siemka|dzień dobry|dzien dobry|dobry wieczór|dobry wieczor|witam|witam serdecznie|elo|yo|hello|hi|hey|greetings|thanks|thank you|thx|dzięki|dzieki|dziękuję|dziekuje|dziękuje|dzięks|ok|okej|dobrze|super|jasne|rozumiem|yes|no|tak|nie|do widzenia|do usłyszenia|na razie|narazie|pa|dobranoc|trzymaj się|bye|goodbye|see you|/.*)[.!?,\s]*(?:wielkie|bardzo|za pomoc|za wszystko|do jutra|do usłyszenia|miłego dnia|miłego wieczoru|serdeczne|z góry)?[.!?,\s]*$",
    re.IGNORECASE,
)

CASUAL_GREETING_PATTERN = re.compile(
    r"^(?:cześć|czesc|hej|hejka|witaj|witajcie|siema|siemanko|dzień dobry|dzien dobry|hello|hi|hey|super|dzięki|dzieki|dziękuję|dziekuje|ok|okej)\b.*(?:jak leci|jak tam|jak się masz|co tam|how are you|how is it going|what's up|do usłyszenia|do widzenia|za pomoc|miłego dnia)",
    re.IGNORECASE,
)


def compute_content_fingerprint(text: str) -> str:
    """Computes a normalized lexical fingerprint for deduplicating identical or near-identical memory facts."""
    clean = re.sub(r'[^\w\s]', ' ', text.lower(), flags=re.UNICODE).strip()
    for suffix in ("stated_memory", "working_fact", "canonical_spec", "episodic_event"):
        if clean.endswith(suffix):
            clean = clean[:-len(suffix)].strip()
    for prefix in (
        "correction", "obsidian mcp", "zasada", "reguła", "peptide research",
        "wklejki", "prompt", "user prefers", "fakt", "note", "uwaga", "memory",
    ):
        if clean.startswith(prefix):
            clean = clean[len(prefix):].strip()
    stops = {"jest", "się", "przez", "oraz", "albo", "tym", "nie", "ale", "dla", "jak", "the", "and", "for", "with"}
    words = [w for w in clean.split() if len(w) > 2 and w not in stops]
    return " ".join(words[:6]) if words else clean[:30]


def get_record_authority_score(subject: str) -> int:
    """Returns authority level for record key: higher score wins during deduplication and ranking."""
    s_lower = subject.lower()
    if s_lower.startswith(("rule:", "user_rule:", "preference:", "constraint:")) or ":rule:" in s_lower:
        return 110
    if s_lower.startswith(("hermes:user_md", "hermes:memory_md", "user_md:", "memory_md:")):
        return 100
    if s_lower.startswith("fact:"):
        return 80
    if "canonical_facts" in s_lower:
        return 70
    if "working_memory" in s_lower:
        return 30
    if "episodic" in s_lower:
        return 20
    if not s_lower.startswith("mnemosyne:"):
        return 90
    return 50


class RetrievalPipeline:
    """
    Cognitive Retrieval Pipeline for ATLAS Memory.
    """

    VERACITY_WEIGHTS: Dict[str, float] = {
        "user_explicit": 1.0,
        "tool_output": 0.85,
        "tool_observation": 0.85,
        "external_doc": 0.65,
        "agent_inference": 0.50,
    }

    def __init__(
        self,
        hot_buffer: Optional[HotBufferPolicy] = None,
        kv_store: Optional[Any] = None,
        vector_store: Optional[Any] = None,
        graph_store: Optional[Any] = None,
        orchestrator: Optional[Any] = None,
        default_token_budget: int = 1500,
        engine: Optional[Any] = None,
        compactor: Optional[ContextCompactor] = None,
        mnemosyne: Optional[Any] = None,
    ) -> None:
        self.hot_buffer = hot_buffer if hot_buffer is not None else HotBufferPolicy()
        self.kv_store = kv_store
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.orchestrator = orchestrator
        self.default_token_budget = default_token_budget
        self.engine = engine
        self.compactor = compactor or ContextCompactor()
        self.mnemosyne = mnemosyne

    @staticmethod
    def is_continuation_query(query: str) -> bool:
        """Returns True if query matches continuation phrasing."""
        return bool(CONTINUATION_PATTERN.match(query.strip()))

    @staticmethod
    def is_trivial_prompt(query: str) -> bool:
        """Returns True if query is casual chit-chat or trivial greeting."""
        q = query.strip()
        return bool(TRIVIAL_PROMPT_PATTERN.match(q) or CASUAL_GREETING_PATTERN.match(q))

    def should_retrieve(
        self,
        query: str,
        explicit_entities: Optional[List[str]] = None,
    ) -> Tuple[bool, List[str], str]:
        """
        Policy Gating for retrieval:
        - Filters chit-chat / greetings unless explicit entities present.
        - Detects CamelCase, code identifiers, acronyms, and intent keywords.
        """
        msg_clean = query.lower().strip()
        if not msg_clean:
            return False, [], "empty_query"

        if self.is_continuation_query(query):
            return True, ["continuation"], "continuation_intent"

        if self.is_trivial_prompt(query) and not explicit_entities:
            return False, [], "trivial_prompt"

        if self.orchestrator is not None and hasattr(self.orchestrator, "should_retrieve"):
            try:
                return self.orchestrator.should_retrieve(query, explicit_entities=explicit_entities)
            except Exception as exc:
                logger.debug("Orchestrator should_retrieve fallback: %s", exc)

        detected_entities: List[str] = []
        if explicit_entities:
            detected_entities.extend(explicit_entities)

        # Code identifiers with hyphens or underscores
        for sym in re.findall(r"\b[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+\b", query):
            if sym not in detected_entities and len(sym) >= 3:
                detected_entities.append(sym)

        # CamelCase identifiers
        for camel in re.findall(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-zA-Z0-9]+)+\b", query):
            if camel not in detected_entities and len(camel) >= 3:
                detected_entities.append(camel)

        # Colon-delimited state keys (e.g. test:hop:p1)
        for colon_key in re.findall(r"\b[A-Za-z0-9_-]+:[A-Za-z0-9_:-]+\b", query):
            if colon_key not in detected_entities:
                detected_entities.append(colon_key)

        # Intent keywords
        intent_keywords = (
            "chain", "hop", "fakt", "dawka", "port", "token", "rule", "config",
            "szukaj", "znajdź", "pokaż", "sprawdź", "notat", "obsidian", "peptyd",
            "skrypt", "baza", "model", "protokół", "badani", "raport", "informac",
            "pamięć", "wiedza", "stan", "variable", "preference",
        )
        for kw in intent_keywords:
            if kw in msg_clean and kw not in detected_entities:
                detected_entities.append(kw)

        if detected_entities:
            return True, detected_entities, f"entity_match: {detected_entities}"

        return False, [], "conversational_turn_no_entity"

    @staticmethod
    def get_authority_score(subject: str) -> float:
        """
        Authority hierarchy scoring:
        system/rules (1.10) > user (1.00) > doc (0.70) > agent (0.50).
        """
        s_lower = subject.lower()
        if s_lower.startswith(("rule:", "user_rule:", "preference:", "constraint:")) or ":rule:" in s_lower:
            return 1.10
        if s_lower.startswith(("hermes:user_md", "hermes:memory_md", "user_md:", "memory_md:")) or "user" in s_lower:
            return 1.00
        if s_lower.startswith("fact:") or "doc" in s_lower or "external" in s_lower or "canonical_facts" in s_lower:
            return 0.70
        if "working_memory" in s_lower or "agent" in s_lower or "infer" in s_lower:
            return 0.50
        if not s_lower.startswith("mnemosyne:"):
            return 0.90
        return 0.60

    def compute_record_rank(
        self,
        record: RetrievalRecord | Dict[str, Any],
        query: str = "",
        now: Optional[float] = None,
    ) -> float:
        """
        Veracity-First Ranking with Dynamic Recency Prior:
        - Veracity weights: user_explicit (1.0) > tool_output (0.85) > external_doc (0.65) > agent_inference (0.50)
        - Authority scoring: system/rules > user > doc > agent
        - Logarithmic Recency Prior for intra-tier freshness:
          delta_days = max(0.0, (now - rec_ts) / 86400.0)
          rule_bonus = 1.0 / (1.0 + 0.15 * math.log1p(delta_days))
          rank += rule_bonus for explicit user rules
        - Superseded belief suppression: if meta.get('is_superseded') -> rank = -5.0
        """
        current_time = time.time() if now is None else now

        if isinstance(record, dict):
            subject = str(record.get("subject") or record.get("key") or "")
            predicate = str(record.get("predicate", "state"))
            source_type = str(record.get("source_type", "user_explicit")).lower()
            conf = float(record.get("confidence", 1.0))
            imp = float(record.get("importance_score", 0.5))
            meta = dict(record.get("metadata") or {})
            is_superseded = bool(record.get("is_superseded") or meta.get("is_superseded") or meta.get("superseded"))
            rec_ts = float(record.get("timestamp") or meta.get("timestamp") or current_time)
        else:
            subject = record.subject
            predicate = record.predicate
            source_type = record.source_type.lower()
            conf = record.confidence
            imp = getattr(record, "importance_score", 0.5)
            meta = dict(record.metadata or {})
            is_superseded = bool(record.is_superseded or meta.get("is_superseded") or meta.get("superseded"))
            rec_ts = float(record.timestamp or current_time)

        # Superseded belief suppression: assign rank = -5.0 (suppressed)
        if is_superseded:
            return -5.0

        vw = self.VERACITY_WEIGHTS.get(source_type, 0.50)
        auth = self.get_authority_score(subject)

        # Relevance scoring (lexical overlap + vector score if available)
        relevance = max(
            float(meta.get("vector_score", 0.0)),
            float(meta.get("lexical_score", 0.0)),
            float(meta.get("score", 0.0)),
            0.0,
        )
        if query:
            q_words = [w for w in query.lower().split() if len(w) > 2]
            if q_words:
                rec_text = f"{subject} {predicate} {meta.get('object', '')}".lower()
                matches = sum(1 for w in q_words if w in rec_text)
                relevance = max(relevance, matches / len(q_words))

        # Base rank
        rank = 0.40 * (vw * conf) + 0.25 * auth + 0.20 * relevance + 0.15 * imp

        # Active working memory priority
        if predicate == "active_working_memory":
            rank += 2.0

        # Logarithmic Recency Prior for explicit user rules
        subj_lower = subject.lower()
        is_rule = subj_lower.startswith(("rule:", "user_rule:", "preference:", "constraint:")) or ":rule:" in subj_lower
        is_explicit_user = (source_type == "user_explicit" or conf >= 0.99)
        if is_rule and is_explicit_user:
            delta_days = max(0.0, (current_time - rec_ts) / 86400.0)
            rule_bonus = 1.0 / (1.0 + 0.15 * math.log1p(delta_days))
            rank += rule_bonus

        return rank

    @staticmethod
    def estimate_tokens(record: RetrievalRecord | Dict[str, Any]) -> int:
        """Estimates token count for context packing."""
        if isinstance(record, dict):
            s = str(record.get("subject", ""))
            o = str(record.get("object", record.get("value", "")))
        else:
            s = record.subject
            o = record.object
        text = f"• [{s}] {o[:1200]}"
        # Conservative token estimation: ~3.5 chars per token + formatting overhead
        return max(1, math.ceil(len(text) / 3.5) + 1)

    def pack_knapsack_01(
        self,
        records: List[RetrievalRecord],
        budget: int,
        query: str = "",
        now: Optional[float] = None,
    ) -> List[RetrievalRecord]:
        """
        0/1 Knapsack token budget packing for context fitting.
        Suppresses superseded items (rank <= 0.0) and maximizes total information value.
        """
        scored_records: List[Tuple[RetrievalRecord, float, int]] = []
        for r in records:
            rank = self.compute_record_rank(r, query=query, now=now)
            r.score = rank
            # Suppress superseded beliefs and invalid ranks
            if rank > 0.0:
                cost = self.estimate_tokens(r)
                scored_records.append((r, rank, cost))

        if not scored_records:
            return []

        # FAZA 3: Integracja ContextCompactor z 0/1 Knapsack
        # Gdy suma szacowanych tokenów kandydatów przekracza budget * 1.5 i kandydatów jest co najmniej 3,
        # dla kandydatów z dolnej połowy rankingu (lub bardzo długich obiektów > 150 znaków) skróć/zagęść obiekt.
        total_tokens = sum(item[2] for item in scored_records)
        if total_tokens > budget * 1.5 and len(scored_records) >= 3:
            scored_records.sort(key=lambda item: item[1], reverse=True)
            n_items = len(scored_records)
            halfway = n_items // 2
            compacted_records = []
            for idx, (r, rank, cost) in enumerate(scored_records):
                is_lower_half = idx >= halfway
                obj_str = str(r.object)
                is_long = len(obj_str) > 150
                if (is_lower_half or is_long) and not obj_str.endswith("... [compacted]"):
                    r.object = obj_str[:80] + "... [compacted]"
                    new_cost = self.estimate_tokens(r)
                    compacted_records.append((r, rank, new_cost))
                else:
                    compacted_records.append((r, rank, cost))
            scored_records = compacted_records

        # Greedy fallback for large candidate sets (> 100 items)
        if len(scored_records) > 100:
            scored_records.sort(key=lambda item: (item[1] / max(1, item[2]), item[1]), reverse=True)
            packed: List[RetrievalRecord] = []
            curr_w = 0
            for r, _score, cost in scored_records:
                if curr_w + cost <= budget:
                    packed.append(r)
                    curr_w += cost
            return packed

        n = len(scored_records)
        scale = 1 if budget <= 200 else 5
        b_scaled = budget // scale

        dp = [0.0] * (b_scaled + 1)
        keep = [[False] * (b_scaled + 1) for _ in range(n)]

        for i, (_, val, cost) in enumerate(scored_records):
            w = max(1, (cost + scale - 1) // scale)
            for cap in range(b_scaled, w - 1, -1):
                if dp[cap - w] + val > dp[cap]:
                    dp[cap] = dp[cap - w] + val
                    keep[i][cap] = True

        selected_indices = []
        curr_cap = b_scaled
        for i in range(n - 1, -1, -1):
            if keep[i][curr_cap]:
                selected_indices.append(i)
                w = max(1, (scored_records[i][2] + scale - 1) // scale)
                curr_cap -= w

        selected_indices.reverse()
        result = [scored_records[i][0] for i in selected_indices]
        # Final sort by rank descending
        result.sort(key=lambda r: r.score, reverse=True)
        return result

    @staticmethod
    def generate_context_block(records: List[RetrievalRecord]) -> str:
        """Clean formatted context block markdown generator."""
        if not records:
            return ""
        lines = ["## ATLAS Cognitive Context (No-GIL Hardware Memory)"]
        for r in records:
            lines.append(f"• [{r.subject}] {r.object}")
        return "\n".join(lines)

    @classmethod
    def format_inprocess_prefetch(
        cls,
        orchestrator: Any,
        records: List[Any],
        query: str,
        max_tokens: int = 1500,
    ) -> Dict[str, Any]:
        """Delegated helper for in-process memory prefetch formatting and budgeting."""
        if orchestrator is None:
            return {"formatted_context": "", "selected_facts": [], "estimated_tokens": 0}
        ranked = orchestrator.epistemic_rank(records, query=query)
        budgeted = orchestrator.apply_token_budget(ranked, max_tokens=max_tokens)
        return budgeted

    async def retrieve(
        self,
        query: str,
        session_id: Optional[str] = HERMES_DEFAULT_SESSION,
        limit: int = 15,
        token_budget: Optional[int] = None,
        force: bool = False,
    ) -> RetrievalContext:
        """
        Full retrieval execution orchestrating gating, hot-KV injection, ranking, and knapsack packing.
        """
        budget = token_budget if token_budget is not None else self.default_token_budget
        now = time.time()
        q_clean = query.strip()

        # 1. Hot Working Memory Check
        active_hot = self.hot_buffer.get_active(session_id=session_id, now=now)

        # 2. Empty Query Handling
        if not q_clean:
            if active_hot:
                logger.info("[RetrievalPipeline] Empty query: injecting %d active hot buffer records", len(active_hot))
                records = [
                    RetrievalRecord(
                        subject=h.key,
                        predicate="active_working_memory",
                        object=str(h.value),
                        confidence=h.confidence,
                        source_type=h.source_type,
                        score=1.0,
                        is_state_variable=True,
                        timestamp=h.timestamp,
                        metadata=h.metadata,
                    )
                    for h in active_hot[:limit]
                ]
                packed = self.pack_knapsack_01(records, budget=budget, query="", now=now)
                return RetrievalContext(
                    query="",
                    session_id=session_id,
                    records=packed,
                    count=len(packed),
                    skipped=False,
                    metadata={"injected_from": "hot_buffer", "type": "empty_query"},
                )
            return RetrievalContext(
                query="",
                session_id=session_id,
                records=[],
                count=0,
                skipped=True,
                metadata={"reason": "empty_query_no_hot"},
            )

        # 3. Continuation Query Handling
        is_continuation = self.is_continuation_query(q_clean)
        if is_continuation:
            if active_hot:
                logger.info("[RetrievalPipeline] Continuation query ('%s'): injecting %d active hot records", q_clean[:40], len(active_hot))
                records = [
                    RetrievalRecord(
                        subject=h.key,
                        predicate="active_working_memory",
                        object=str(h.value),
                        confidence=h.confidence,
                        source_type=h.source_type,
                        score=1.0,
                        is_state_variable=True,
                        timestamp=h.timestamp,
                        metadata=h.metadata,
                    )
                    for h in active_hot[:limit]
                ]
                packed = self.pack_knapsack_01(records, budget=budget, query=q_clean, now=now)
                return RetrievalContext(
                    query=q_clean,
                    session_id=session_id,
                    records=packed,
                    count=len(packed),
                    skipped=False,
                    metadata={"injected_from": "hot_buffer", "type": "continuation_query"},
                )
            if not force:
                logger.info("[RetrievalPipeline] Continuation query with no active hot memory, skipped (force=False)")
                return RetrievalContext(
                    query=q_clean,
                    session_id=session_id,
                    records=[],
                    count=0,
                    skipped=True,
                    metadata={"reason": "continuation_prompt_no_hot"},
                )

        # 4. Retrieval Policy Gate (Chit-Chat / Greetings)
        if not force:
            should_run, entities, reason = self.should_retrieve(q_clean)
            if not should_run:
                if active_hot:
                    logger.info("[RetrievalPipeline] Policy Gate triggered (%s) but injecting %d hot items", reason, len(active_hot))
                    records = [
                        RetrievalRecord(
                            subject=h.key,
                            predicate="active_working_memory",
                            object=str(h.value),
                            confidence=h.confidence,
                            source_type=h.source_type,
                            score=1.0,
                            is_state_variable=True,
                            timestamp=h.timestamp,
                            metadata=h.metadata,
                        )
                        for h in active_hot[:limit]
                    ]
                    packed = self.pack_knapsack_01(records, budget=budget, query=q_clean, now=now)
                    return RetrievalContext(
                        query=q_clean,
                        session_id=session_id,
                        records=packed,
                        count=len(packed),
                        skipped=False,
                        metadata={"injected_from": "hot_buffer", "policy_gate_reason": reason},
                    )
                logger.info("[RetrievalPipeline] Retrieval skipped by policy gate: %s", reason)
                return RetrievalContext(
                    query=q_clean,
                    session_id=session_id,
                    records=[],
                    count=0,
                    skipped=True,
                    metadata={"reason": reason},
                )

        # 5. Gather candidates from KV Store & Vector Store if attached
        candidates: List[RetrievalRecord] = []

        # Include hot working memory entries as candidates
        for h in active_hot:
            candidates.append(
                RetrievalRecord(
                    subject=h.key,
                    predicate="active_working_memory",
                    object=str(h.value),
                    confidence=h.confidence,
                    source_type=h.source_type,
                    score=1.0,
                    is_state_variable=True,
                    is_superseded=h.is_superseded,
                    timestamp=h.timestamp,
                    metadata=h.metadata,
                )
            )

        # Scan KV Store
        if self.kv_store is not None:
            try:
                all_kv = {}
                if hasattr(self.kv_store, "get_all_sync"):
                    all_kv = self.kv_store.get_all_sync()
                elif hasattr(self.kv_store, "get_all"):
                    all_kv = self.kv_store.get_all()

                q_lower = q_clean.lower()
                q_words = [w for w in q_lower.split() if len(w) > 2]
                for k, it in all_kv.items():
                    val_str = str(it.get("value", ""))
                    meta = dict(it.get("metadata") or {})
                    is_sup = bool(it.get("is_superseded") or meta.get("is_superseded"))
                    # Quick filter
                    k_lower = k.lower()
                    if any(w in k_lower or w in val_str.lower() for w in q_words) or (k_lower in q_lower):
                        candidates.append(
                            RetrievalRecord(
                                subject=k,
                                predicate="state_variable",
                                object=val_str,
                                confidence=float(it.get("confidence", 1.0)),
                                source_type=str(meta.get("source_type", "user_explicit")),
                                is_state_variable=True,
                                is_superseded=is_sup,
                                timestamp=float(it.get("timestamp", now)),
                                metadata=meta,
                            )
                        )
            except Exception as kv_exc:
                logger.debug("[RetrievalPipeline] KV scan error: %s", kv_exc)

        # Scan Vector Store
        if self.vector_store is not None and hasattr(self.vector_store, "search"):
            try:
                vec_hits = await self.vector_store.search(q_clean, top_k=limit)
                for hit in vec_hits:
                    rec_data = hit.get("record") or {}
                    subj = rec_data.get("subject") or hit.get("id", "")
                    obj_str = str(rec_data.get("object") or hit.get("text", ""))
                    if not obj_str or is_conversational_churn(subj, obj_str):
                        continue
                    score = float(hit.get("score", 0.0))
                    min_threshold = 0.35
                    if any(ch in q_clean for ch in (":", "_", "-")) or (" " not in q_clean.strip() and len(q_clean.strip()) > 3):
                        min_threshold = 0.70
                    if score < min_threshold:
                        continue
                    meta = dict(rec_data.get("metadata") or {})
                    meta["vector_score"] = score
                    candidates.append(
                        RetrievalRecord(
                            subject=subj,
                            predicate=rec_data.get("predicate", "semantic_fact"),
                            object=obj_str,
                            confidence=float(rec_data.get("confidence", 1.0)),
                            source_type=str(rec_data.get("source_type", "external_doc")),
                            score=score,
                            is_state_variable=False,
                            is_superseded=bool(meta.get("is_superseded")),
                            timestamp=float(rec_data.get("timestamp", now)),
                            metadata=meta,
                        )
                    )
            except Exception as vec_exc:
                logger.debug("[RetrievalPipeline] Vector search error: %s", vec_exc)

        # Scan Orchestrated Recall
        if self.orchestrator is not None and hasattr(self.orchestrator, "orchestrated_recall"):
            try:
                recall_res = await self.orchestrator.orchestrated_recall(q_clean, session_id=session_id)
                recs = recall_res.get("records", []) if isinstance(recall_res, dict) else (recall_res if isinstance(recall_res, list) else [])
                for r in recs:
                    rec_dict = r.model_dump() if hasattr(r, "model_dump") else dict(r)
                    subj = rec_dict.get("subject", "") or rec_dict.get("id", "")
                    obj_str = str(rec_dict.get("object") or rec_dict.get("content", ""))
                    if not obj_str or is_conversational_churn(subj, obj_str):
                        continue
                    meta = dict(rec_dict.get("metadata") or {})
                    candidates.append(
                        RetrievalRecord(
                            subject=subj,
                            predicate=str(rec_dict.get("predicate", "fact")),
                            object=obj_str,
                            confidence=float(rec_dict.get("confidence") or rec_dict.get("veracity") or 1.0),
                            source_type=str(rec_dict.get("source_type", "external_doc")),
                            score=float(rec_dict.get("score", 0.85)),
                            is_state_variable=False,
                            is_superseded=bool(meta.get("is_superseded")),
                            timestamp=float(rec_dict.get("timestamp", now)),
                            metadata=meta,
                        )
                    )
            except Exception as orc_exc:
                logger.debug("[RetrievalPipeline] Orchestrated recall error: %s", orc_exc)

        # 6. Rank and pack candidates with 0/1 Knapsack
        packed_records = self.pack_knapsack_01(candidates, budget=budget, query=q_clean, now=now)
        packed_records = packed_records[:limit]

        return RetrievalContext(
            query=q_clean,
            session_id=session_id,
            records=packed_records,
            count=len(packed_records),
            skipped=False,
            metadata={"budget": budget, "initial_candidates": len(candidates)},
        )

    async def execute_prefetch(
        self,
        params: Dict[str, Any],
        *,
        engine: Optional[Any] = None,
        orchestrator: Optional[Any] = None,
        hot_buffer: Optional[Any] = None,
        check_sync_callback: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """
        Prefetch memory records for query/session with full epistemic formatting,
        hot working memory injection, content deduplication, and veracity ranking.
        """
        query = (params.get("query") or "").strip()
        session_id = params.get("session_id") or HERMES_DEFAULT_SESSION
        try:
            raw_limit = int(params.get("limit", 15))
        except (ValueError, TypeError):
            raw_limit = 15
        limit = max(1, min(raw_limit, 100))
        force = bool(params.get("force", False))
        now_ts = time.time()

        try:
            token_budget = int(
                params.get("token_budget")
                or params.get("max_tokens")
                or params.get("budget")
                or self.default_token_budget
            )
        except (ValueError, TypeError):
            token_budget = self.default_token_budget

        target_engine = engine or self.engine
        target_orchestrator = orchestrator or self.orchestrator
        target_hot = hot_buffer if hot_buffer is not None else getattr(self, "_hot_kv_buffer", self.hot_buffer)

        # Active Hot-KV Working Buffer items (last 15 minutes / 900s) (Finding F-4)
        active_hot: List[Tuple[str, str, float, float, str, Dict[str, Any]]] = []
        if target_hot is not None:
            for item in target_hot:
                if isinstance(item, dict):
                    k = str(item.get("key", ""))
                    v = item.get("value", "")
                    ts = float(item.get("timestamp", 0.0))
                    conf = float(item.get("confidence", 1.0))
                    st = str(item.get("source_type", "user_explicit"))
                    meta = dict(item.get("metadata") or {})
                    is_sup = bool(item.get("is_superseded") or meta.get("is_superseded"))
                else:
                    k = str(getattr(item, "key", ""))
                    v = getattr(item, "value", "")
                    ts = float(getattr(item, "timestamp", 0.0))
                    conf = float(getattr(item, "confidence", 1.0))
                    st = str(getattr(item, "source_type", "user_explicit"))
                    meta = dict(getattr(item, "metadata", {}) or {})
                    is_sup = bool(getattr(item, "is_superseded", False) or meta.get("is_superseded"))

                if (now_ts - ts) > 900.0:
                    continue
                if is_sup:
                    continue
                v_str = str(v)
                if is_conversational_churn(k, v_str):
                    continue
                if k.startswith(("fact:conversation___user", "fact:conversation", "conversation___user", "conversation:")):
                    continue
                eff_sess = extract_effective_session(k, meta)
                if not is_session_accessible(eff_sess, session_id):
                    continue
                active_hot.append((k, v_str, ts, conf, st, meta))

        seen_hot_keys: Set[str] = set()
        unique_session_hot: List[Tuple[str, str, float, float, str, Dict[str, Any]]] = []
        for h_tuple in reversed(active_hot):
            k_h = h_tuple[0]
            if k_h not in seen_hot_keys:
                seen_hot_keys.add(k_h)
                unique_session_hot.append(h_tuple)

        # 0. Fast-path empty query
        if not query:
            if unique_session_hot:
                logger.info("[RPC] prefetch: empty query injecting %d active hot-KV items", len(unique_session_hot))
                hot_records = []
                lines = ["## ATLAS Cognitive Context (No-GIL Hardware Memory)"]
                total_tokens = 0
                for k_h, v_h, _, conf_h, st_h, meta_h in unique_session_hot[:limit]:
                    rec = {
                        "subject": k_h,
                        "predicate": "active_working_memory",
                        "object": v_h,
                        "importance_score": 1.0,
                        "confidence": conf_h,
                        "source_type": st_h,
                        "metadata": meta_h,
                    }
                    cost = self.estimate_tokens(rec)
                    if total_tokens + cost > token_budget and hot_records:
                        break
                    total_tokens += cost
                    hot_records.append(rec)
                    v_h_raw = str(v_h)
                    v_h_disp = v_h_raw[:1200] + " ...[truncated]" if len(v_h_raw) > 1200 else v_h_raw
                    lines.append(f"• [{k_h}] {v_h_disp}")
                if hot_records:
                    ctx_block = "\n".join(lines)
                    max_chars = max(4000, token_budget * 4)
                    if len(ctx_block) > max_chars:
                        ctx_block = ctx_block[:max_chars]
                    return {
                        "records": hot_records,
                        "context_block": ctx_block,
                        "query": "",
                        "count": len(hot_records),
                    }
            return {"records": [], "context_block": "", "query": "", "count": 0, "skipped": True, "reason": "empty_query"}

        # 0b. Continuation Pattern Gate (decoupled from force flag)
        is_continuation = self.is_continuation_query(query)
        if is_continuation:
            if unique_session_hot:
                logger.info("[RPC] prefetch: Continuation query ('%s') injecting %d active hot-KV items", query[:40], len(unique_session_hot))
                hot_records = []
                lines = ["## ATLAS Cognitive Context (No-GIL Hardware Memory)"]
                total_tokens = 0
                for k_h, v_h, _, conf_h, st_h, meta_h in unique_session_hot[:limit]:
                    rec = {
                        "subject": k_h,
                        "predicate": "active_working_memory",
                        "object": v_h,
                        "importance_score": 1.0,
                        "confidence": conf_h,
                        "source_type": st_h,
                        "metadata": meta_h,
                    }
                    cost = self.estimate_tokens(rec)
                    if total_tokens + cost > token_budget and hot_records:
                        break
                    total_tokens += cost
                    hot_records.append(rec)
                    v_h_raw = str(v_h)
                    v_h_disp = v_h_raw[:1200] + " ...[truncated]" if len(v_h_raw) > 1200 else v_h_raw
                    lines.append(f"• [{k_h}] {v_h_disp}")
                if hot_records:
                    ctx_block = "\n".join(lines)
                    max_chars = max(4000, token_budget * 4)
                    if len(ctx_block) > max_chars:
                        ctx_block = ctx_block[:max_chars]
                    return {
                        "records": hot_records,
                        "context_block": ctx_block,
                        "query": query,
                        "count": len(hot_records),
                    }
            if not force:
                logger.info("[RPC] prefetch SKIP (Policy Gate): continuation_prompt with no hot-KV for query='%s'", query[:40])
                return {"records": [], "context_block": "", "query": query, "count": 0, "skipped": True, "reason": "continuation_prompt"}

        # 1. Retrieval Policy Gate (Chit-Chat / Greetings / Policy screening)
        if not force:
            from atlas_memory.orchestrator import MemoryOrchestrator
            is_trivial = bool(MemoryOrchestrator.TRIVIAL_PROMPT_PATTERN.match(query) or MemoryOrchestrator.CASUAL_GREETING_PATTERN.match(query))
            if is_trivial:
                logger.info("[RPC] prefetch SKIP (Chit-Chat Gate): trivial prompt query='%s'", query[:40])
                return {"records": [], "context_block": "", "query": query, "count": 0, "skipped": True, "reason": "trivial_prompt"}

            should_run = True
            reason = "ok"
            if target_orchestrator is not None and hasattr(target_orchestrator, "should_retrieve"):
                should_run, _, reason = target_orchestrator.should_retrieve(query)

            if not should_run:
                if unique_session_hot:
                    logger.info("[RPC] prefetch: Policy Gate triggered ('%s') but injecting %d active hot-KV items", query[:40], len(unique_session_hot))
                    hot_records = []
                    lines = ["## ATLAS Cognitive Context (No-GIL Hardware Memory)"]
                    total_tokens = 0
                    for k_h, v_h, _, conf_h, st_h, meta_h in unique_session_hot[:limit]:
                        rec = {
                            "subject": k_h,
                            "predicate": "active_working_memory",
                            "object": v_h,
                            "importance_score": 1.0,
                            "confidence": conf_h,
                            "source_type": st_h,
                            "metadata": meta_h,
                        }
                        cost = self.estimate_tokens(rec)
                        if total_tokens + cost > token_budget and hot_records:
                            break
                        total_tokens += cost
                        hot_records.append(rec)
                        v_h_raw = str(v_h)
                        v_h_disp = v_h_raw[:1200] + " ...[truncated]" if len(v_h_raw) > 1200 else v_h_raw
                        lines.append(f"• [{k_h}] {v_h_disp}")
                    if hot_records:
                        ctx_block = "\n".join(lines)
                        max_chars = max(4000, token_budget * 4)
                        if len(ctx_block) > max_chars:
                            ctx_block = ctx_block[:max_chars]
                        return {
                            "records": hot_records,
                            "context_block": ctx_block,
                            "query": query,
                            "count": len(hot_records),
                        }
                logger.info("[RPC] prefetch SKIP (Policy Gate): %s for query='%s'", reason, query[:40])
                return {"records": [], "context_block": "", "query": query, "count": 0, "skipped": True, "reason": reason}

        # 2. Auto-sync MEMORY.md if modified
        if check_sync_callback is not None:
            try:
                check_sync_callback()
            except Exception as exc:
                logger.debug("Auto-sync memories check error: %s", exc)

        dedup_candidates: Dict[str, Dict[str, Any]] = {}
        seen_keys: Set[str] = set()

        # 3. Search KV store for fresh state variables and keyword/stem matches
        q_lower = query.lower()
        generic_modifiers = {"kompletnie", "bardzo", "całkiem", "trochę", "zawsze", "nigdy", "oraz", "albo", "jako", "tylko", "może", "about", "really", "quite", "super", "some"}
        words = [w.strip(".,!?:;\"'()[]{}") for w in q_lower.split() if len(w) > 2 and w not in generic_modifiers]
        if not words:
            words = [w.strip(".,!?:;\"'()[]{}") for w in q_lower.split() if len(w) > 2]
        stems = [w[:6] if len(w) >= 7 else w for w in words if len(w) >= 4]

        kv_store = None
        if target_engine is not None and hasattr(target_engine, "kv") and target_engine.kv is not None:
            kv_store = target_engine.kv
        elif self.kv_store is not None:
            kv_store = self.kv_store

        if kv_store is not None:
            try:
                if hasattr(kv_store, "get_all_sync"):
                    all_kv = kv_store.get_all_sync()
                else:
                    all_kv = kv_store.get_all()
                for k, item in all_kv.items():
                    val_str = str(item.get("value", ""))

                    # Filter out conversational noise and chat history leakage
                    if is_conversational_churn(k, val_str):
                        continue
                    if k.startswith(("fact:conversation___user", "fact:conversation", "conversation___user", "conversation:")):
                        continue

                    k_lower = k.lower()
                    meta = item.get("metadata", {}) or {}
                    is_superseded = bool(item.get("is_superseded") or meta.get("is_superseded") or meta.get("superseded"))
                    if is_superseded:
                        continue
                    effective_session = extract_effective_session(k, meta)
                    if not is_session_accessible(effective_session, session_id):
                        continue

                    val_tokens = set(re.findall(r'[a-zA-Z0-9]+', val_str.lower())) | set(re.findall(r'[a-zA-Z0-9_]+', val_str.lower()))
                    key_tokens = set(re.findall(r'[a-zA-Z0-9]+', k_lower)) | set(re.findall(r'[a-zA-Z0-9_]+', k_lower))
                    combined_tokens = val_tokens | key_tokens

                    matched_words = sum(1 for w in words if w in combined_tokens or any(tok.startswith(w) for tok in combined_tokens))
                    lexical_sim = (matched_words / max(len(words), 1)) if words else 0.0

                    is_exact_key_match = (k_lower in q_lower) or (q_lower in k_lower)
                    is_phrase_match = (len(q_lower) >= 5 and q_lower in val_str.lower())
                    is_word_match = any(w in combined_tokens for w in words)
                    is_stem_match = any(any(tok.startswith(s) for tok in combined_tokens) for s in stems if len(s) >= 5)
                    is_match = is_exact_key_match or is_phrase_match or is_word_match or is_stem_match

                    if is_match:
                        is_native_state = not k.startswith("mnemosyne:")
                        if is_phrase_match or is_exact_key_match:
                            score = 1.0
                        elif is_native_state:
                            score = 0.95
                        else:
                            score = 0.75

                        conf = float(item.get("confidence") if item.get("confidence") is not None else 1.0)
                        src_type = meta.get("source_type") or meta.get("source")
                        if not src_type:
                            if "agent" in k_lower or "infer" in k_lower:
                                src_type = "agent_inference"
                            elif "tool" in k_lower or "obs" in k_lower:
                                src_type = "tool_output"
                            elif is_native_state or conf >= 0.99:
                                src_type = "user_explicit"
                            else:
                                src_type = "external_doc"

                        is_hot_item = k in seen_hot_keys
                        meta_cand = dict(meta)
                        meta_cand["lexical_score"] = lexical_sim
                        rec_cand = {
                            "subject": k,
                            "predicate": "active_working_memory" if is_hot_item else ("state_variable" if is_native_state else "fact"),
                            "object": val_str,
                            "importance_score": score,
                            "confidence": conf,
                            "source_type": src_type,
                            "metadata": meta_cand,
                        }

                        # Content-level fingerprint deduplication (highest authority wins)
                        fp = compute_content_fingerprint(val_str)
                        if fp in dedup_candidates:
                            cur_auth = get_record_authority_score(dedup_candidates[fp]["subject"])
                            new_auth = get_record_authority_score(k)
                            if new_auth > cur_auth:
                                seen_keys.discard(dedup_candidates[fp]["subject"])
                                dedup_candidates[fp] = rec_cand
                                seen_keys.add(k)
                        else:
                            if k not in seen_keys:
                                dedup_candidates[fp] = rec_cand
                                seen_keys.add(k)
            except Exception as kv_scan_exc:
                logger.debug("KV scan failed: %s", kv_scan_exc)

        # 3b. Dense Semantic Vector Search (Qdrant Vector Store + FastEmbed)
        vec_store = None
        if target_engine is not None and hasattr(target_engine, "vector_store") and target_engine.vector_store is not None:
            vec_store = target_engine.vector_store
        elif self.vector_store is not None:
            vec_store = self.vector_store

        if vec_store is not None:
            try:
                vec_hits = await vec_store.search(query, top_k=limit)
                for hit in vec_hits:
                    rec_data = hit.get("record") or {}
                    subj = rec_data.get("subject") or hit.get("id", "")
                    obj_str = str(rec_data.get("object") or hit.get("text", ""))
                    if not obj_str or is_conversational_churn(subj, obj_str):
                        continue

                    score = float(hit.get("score", 0.0))
                    min_threshold = 0.35
                    # Jeśli zapytanie zawiera znaki identyfikatora (np. ':', '_', '-') lub jest pojedynczym słowem bez spacji (>3 znaki):
                    if any(ch in query for ch in (":", "_", "-")) or (" " not in query.strip() and len(query.strip()) > 3):
                        min_threshold = 0.70
                    if score < min_threshold:
                        continue
                    meta = dict(rec_data.get("metadata") or {})
                    is_superseded = bool(rec_data.get("is_superseded") or meta.get("is_superseded") or meta.get("superseded"))
                    if is_superseded:
                        continue
                    eff_sess = extract_effective_session(subj, meta)
                    if not is_session_accessible(eff_sess, session_id):
                        continue
                    meta["vector_score"] = score

                    rec_cand = {
                        "subject": subj,
                        "predicate": rec_data.get("predicate", "semantic_fact"),
                        "object": obj_str,
                        "importance_score": float(rec_data.get("importance_score", 0.85)),
                        "confidence": float(rec_data.get("confidence", 1.0)),
                        "source_type": rec_data.get("source_type", "user_explicit" if "hermes" in subj else "external_doc"),
                        "metadata": meta,
                    }

                    fp = compute_content_fingerprint(obj_str)
                    if fp in dedup_candidates:
                        cur_auth = get_record_authority_score(dedup_candidates[fp]["subject"])
                        new_auth = get_record_authority_score(subj)
                        if new_auth > cur_auth or (new_auth == cur_auth and score > float(dedup_candidates[fp].get("metadata", {}).get("vector_score", 0.0))):
                            seen_keys.discard(dedup_candidates[fp]["subject"])
                            dedup_candidates[fp] = rec_cand
                            seen_keys.add(subj)
                    else:
                        if subj not in seen_keys:
                            dedup_candidates[fp] = rec_cand
                            seen_keys.add(subj)
            except Exception as vec_exc:
                logger.debug("Vector search failed in prefetch: %s", vec_exc)

        # 4. Orchestrated recall (vector + graph + mnemosyne)
        if target_orchestrator is not None:
            if hasattr(target_orchestrator, "orchestrated_recall"):
                try:
                    recall_res = await target_orchestrator.orchestrated_recall(query, session_id=session_id)
                    if isinstance(recall_res, dict):
                        records = recall_res.get("records", [])
                        if not records:
                            records = recall_res.get("selected_facts", [])
                    elif isinstance(recall_res, list):
                        records = recall_res
                    else:
                        records = []

                    for r in records:
                        rec_dict = r.model_dump() if hasattr(r, "model_dump") else dict(r)
                        subj = rec_dict.get("subject", "") or rec_dict.get("id", "")
                        rec_dict["subject"] = subj
                        if not rec_dict.get("object") and rec_dict.get("content"):
                            rec_dict["object"] = str(rec_dict.get("content"))
                        if not rec_dict.get("predicate"):
                            rec_dict["predicate"] = "fact"

                        obj_str = str(rec_dict.get("object", ""))
                        if is_conversational_churn(subj, obj_str):
                            continue

                        rec_meta = rec_dict.get("metadata", {}) or {}
                        rec_sess = extract_effective_session(subj, rec_meta) or session_id
                        if not is_session_accessible(rec_sess, session_id):
                            continue

                        # Content-level deduplication with authority hierarchy
                        fp = compute_content_fingerprint(obj_str)
                        if fp in dedup_candidates:
                            cur_auth = get_record_authority_score(dedup_candidates[fp]["subject"])
                            new_auth = get_record_authority_score(subj)
                            if new_auth > cur_auth:
                                seen_keys.discard(dedup_candidates[fp]["subject"])
                                dedup_candidates[fp] = rec_dict
                                seen_keys.add(subj)
                        elif subj not in seen_keys and not subj.startswith(("fact:conversation___user", "conversation___user", "conversation:")):
                            seen_keys.add(subj)
                            if "confidence" not in rec_dict or rec_dict["confidence"] is None:
                                rec_dict["confidence"] = float(rec_dict.get("veracity", 1.0))
                            dedup_candidates[fp] = rec_dict
                except Exception as rec_exc:
                    logger.debug("orchestrated_recall failed: %s", rec_exc)

        # 4a. Mnemosyne Direct Recall if provided
        target_mnemosyne = getattr(self, "mnemosyne", None) or params.get("mnemosyne")
        if target_mnemosyne is not None and hasattr(target_mnemosyne, "prefetch"):
            try:
                raw_mne = target_mnemosyne.prefetch(query, session_id=session_id)
                if raw_mne:
                    from atlas_memory.hermes.atlas_provider import _parse_mnemosyne_context
                    mne_records = _parse_mnemosyne_context(raw_mne, fallback_query=query)
                    for mr in mne_records:
                        rec_dict = mr.model_dump() if hasattr(mr, "model_dump") else dict(mr)
                        subj = rec_dict.get("subject", "") or "mnemosyne_fact"
                        rec_dict["subject"] = subj
                        if not rec_dict.get("object") and rec_dict.get("content"):
                            rec_dict["object"] = str(rec_dict.get("content"))
                        if not rec_dict.get("predicate"):
                            rec_dict["predicate"] = "context"
                        obj_str = str(rec_dict.get("object", ""))
                        if is_conversational_churn(subj, obj_str):
                            continue
                        fp = compute_content_fingerprint(obj_str)
                        if fp in dedup_candidates:
                            cur_auth = get_record_authority_score(dedup_candidates[fp]["subject"])
                            new_auth = get_record_authority_score(subj)
                            if new_auth > cur_auth:
                                seen_keys.discard(dedup_candidates[fp]["subject"])
                                dedup_candidates[fp] = rec_dict
                                seen_keys.add(subj)
                        elif subj not in seen_keys:
                            seen_keys.add(subj)
                            if "confidence" not in rec_dict or rec_dict["confidence"] is None:
                                rec_dict["confidence"] = float(rec_dict.get("veracity", 1.0))
                            dedup_candidates[fp] = rec_dict
            except Exception as mne_exc:
                logger.debug("Mnemosyne prefetch in pipeline failed: %s", mne_exc)

        # 4b. Multi-hop Graph Subgraph Expansion
        target_graph = (
            self.graph_store
            or (target_engine.graph if target_engine and hasattr(target_engine, "graph") else None)
        )
        if target_graph is not None and hasattr(target_graph, "get_node_relations"):
            try:
                for ent in words[:5]:
                    if len(ent) >= 3:
                        edges = await target_graph.get_node_relations(ent)
                        for edge in edges:
                            edge_session = edge.get("session_id")
                            if not is_session_accessible(edge_session, session_id):
                                continue
                            sub = edge.get("subject", ent)
                            pred = edge.get("predicate", "relates_to")
                            obj = edge.get("object", "")
                            if obj and not is_conversational_churn(sub, str(obj)):
                                fp = compute_content_fingerprint(str(obj))
                                if fp not in dedup_candidates and sub not in seen_keys:
                                    rec_cand = {
                                        "subject": sub,
                                        "predicate": pred,
                                        "object": str(obj),
                                        "importance_score": 0.70,
                                        "confidence": float(edge.get("confidence", 0.9)),
                                        "source_type": "external_doc",
                                        "metadata": {"graph_edge": True, "session_id": edge_session},
                                    }
                                    dedup_candidates[fp] = rec_cand
                                    seen_keys.add(sub)
            except Exception as ge_exc:
                logger.debug("Graph expansion error in prefetch: %s", ge_exc)

        records_list = list(dedup_candidates.values())

        # Sort records by Veracity-First Ranking (User explicit > Tool output > External doc > Agent inference)
        veracity_weights = {
            "user_explicit": 1.0,
            "tool_output": 0.85,
            "tool_observation": 0.85,
            "external_doc": 0.65,
            "agent_inference": 0.50,
        }

        def _get_record_rank(r: Dict[str, Any]) -> float:
            is_superseded = bool(r.get("is_superseded") or (r.get("metadata") or {}).get("is_superseded") or (r.get("metadata") or {}).get("superseded"))
            if is_superseded:
                return -5.0
            st = str(r.get("source_type", "user_explicit")).lower()
            vw = veracity_weights.get(st, 0.5)
            c = float(r.get("confidence") if r.get("confidence") is not None else 1.0)
            imp = float(r.get("importance_score") if r.get("importance_score") is not None else 0.5)
            meta = r.get("metadata") or {}
            relevance = max(
                float(meta.get("vector_score", 0.0)),
                float(meta.get("lexical_score", 0.0)),
                float(meta.get("score", 0.0)),
                0.0,
            )
            auth = get_record_authority_score(str(r.get("subject", ""))) / 100.0
            rank = 0.40 * (vw * c) + 0.25 * auth + 0.20 * relevance + 0.15 * imp
            subj_lower = str(r.get("subject", "")).lower()
            is_rule = subj_lower.startswith(("rule:", "user_rule:", "preference:", "constraint:")) or ":rule:" in subj_lower
            if is_rule and st == "user_explicit" and c >= 0.99:
                rank += 1.0
            if r.get("predicate") == "active_working_memory":
                rank += 2.0
            return rank

        records_list.sort(key=_get_record_rank, reverse=True)
        records_list = [r for r in records_list if _get_record_rank(r) > 0.0]

        # 0/1 Knapsack Packing if token budget explicitly requested
        if "token_budget" in params or "budget" in params or "max_tokens" in params:
            budget = int(params.get("token_budget") or params.get("max_tokens") or params.get("budget") or self.default_token_budget)
            rec_models = [
                RetrievalRecord(
                    subject=r["subject"],
                    predicate=r.get("predicate", "fact"),
                    object=str(r.get("object", "")),
                    confidence=float(r.get("confidence") if r.get("confidence") is not None else 1.0),
                    source_type=str(r.get("source_type", "user_explicit")),
                    score=_get_record_rank(r),
                    timestamp=float(r.get("timestamp", now_ts)),
                    metadata=dict(r.get("metadata") or {}),
                )
                for r in records_list
            ]
            packed = self.pack_knapsack_01(rec_models, budget=budget, query=query, now=now_ts)
            packed_subjects = {p.subject for p in packed}
            records_list = [r for r in records_list if r["subject"] in packed_subjects]

        records_list = records_list[:limit]

        # Build clean formatted context block
        formatted_context = ""
        if records_list:
            lines = ["## ATLAS Cognitive Context (No-GIL Hardware Memory)"]
            for rec in records_list:
                subj = rec.get("subject", "")
                raw_obj = str(rec.get("object", ""))
                obj = raw_obj[:1200] + " ...[truncated]" if len(raw_obj) > 1200 else raw_obj
                lines.append(f"• [{subj}] {obj}")
            formatted_context = "\n".join(lines)
            max_chars = max(4000, token_budget * 4)
            if len(formatted_context) > max_chars:
                formatted_context = formatted_context[:max_chars]

        logger.info("[RPC] prefetch query='%s' (session='%s') -> %d deduplicated records", query[:50], session_id, len(records_list))
        return {
            "records": records_list,
            "context_block": formatted_context,
            "query": query,
            "count": len(records_list),
        }

    async def prefetch_for_context(
        self,
        context_text: str,
        session_id: Optional[str] = None,
        max_results: int = 15,
    ) -> List[Dict[str, Any]]:
        """Convenience helper: prefetch memory records for a given context string and session."""
        res = await self.execute_prefetch({
            "query": context_text,
            "session_id": session_id,
            "limit": max_results,
        })
        return res.get("records", [])

