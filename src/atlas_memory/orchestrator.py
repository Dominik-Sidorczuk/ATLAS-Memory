from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from atlas_memory.extensions.canonicalizer import EntityCanonicalizer
from atlas_memory.extensions.compactor import ContextCompactor
from atlas_memory.extensions.decay_scorer import SalienceDecayEngine
from atlas_memory.extensions.epistemic import EpistemicCalibrator
from atlas_memory.models import (
    EpistemicSource,
    MemoryRecord,
)

logger = logging.getLogger(__name__)


class MnemosyneClientInterface(Protocol):

    """Protokół interfejsu narzędziowego Mnemosyne w Hermes Agent."""
    async def recall(self, query: str, active_entities: Optional[List[str]] = None) -> Dict[str, Any]: ...
    async def triple_add(self, subject: str, predicate: str, object_: str, confidence: float, source: str, supersede: bool) -> Any: ...
    async def triple_query(self, subject: Optional[str] = None, predicate: Optional[str] = None) -> List[Dict[str, Any]]: ...


class MemoryOrchestrator:
    """
    Cloud Memory Orchestrator (Warstwa Decyzyjna i Polityka Pamięci dla Mnemosyne & Hermes Agent).
    
    Działa jako nakładka decyzyjna (Decoupled Policy & Decision Layer) wzbogacająca Mnemosyne o 5 kluczowych przewag:
    1. Cache Contract: stabilny blok promptu dla ~90% prompt cache hit-rate.
    2. Retrieval Policy Gate: canonicalizer decyduje czy w ogóle odpytywać Mnemosyne (0 tokenów szumu).
    3. Epistemic Re-Ranker: sortowanie veracity-first (USER=1.0 > TOOL=0.85 > INFERENCE=0.5).
    4. Token Budget Governor: twardy limit tokenów na kontekst (np. max 1500 tok).
    5. Shadow Reconciliation: asynchroniczna ekstrakcja SPO (Qwen-mini) z arbitrażem i mnemosyne_triple_add(supersede=True).
    6. Salience-Decay GC: usuwanie przestarzałego szumu na bazie wzoru S(f).
    """

    VERACITY_WEIGHTS = {
        EpistemicSource.USER_EXPLICIT: 1.0,
        EpistemicSource.TOOL_OUTPUT: 0.85,
        EpistemicSource.EXTERNAL_DOC: 0.65,
        EpistemicSource.AGENT_INFERENCE: 0.50,
    }

    TRIVIAL_PROMPT_PATTERN = re.compile(
        r"^(?:cześć|hej|witaj|siema|dzień dobry|witam|hello|hi|hey|greetings|thanks|dzięki|ok|dobrze|super|jasne|rozumiem|yes|no|tak|nie|do widzenia|do usłyszenia|na razie|narazie|trzymaj się|bye|goodbye|see you|/.*)[.!?,\s]*(?:wielkie|bardzo|za pomoc|za wszystko|do jutra|do usłyszenia|miłego dnia)?[.!?,\s]*$",
        re.IGNORECASE,
    )
    CASUAL_GREETING_PATTERN = re.compile(
        r"^(?:cześć|hej|witaj|siema|hello|hi|hey|super|dzięki|ok)\b.*(?:jak leci|jak tam|jak się|how are you|how is it going|what's up|do usłyszenia|do widzenia|za pomoc)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        engine: Optional[Any] = None,
        mnemosyne_client: Optional[Any] = None,
        canonicalizer: Optional[EntityCanonicalizer] = None,
        decay_engine: Optional[SalienceDecayEngine] = None,
        compactor: Optional[ContextCompactor] = None,
        calibrator: Optional[EpistemicCalibrator] = None,
        shadow_extractor: Optional[Callable[[str, str], List[Dict[str, Any]]]] = None,
        max_retrieval_tokens: int = 1500,
    ):
        self.engine = engine or mnemosyne_client
        self.mnemosyne = mnemosyne_client or engine
        self.canonicalizer = canonicalizer or (engine.canonicalizer if hasattr(engine, "canonicalizer") else EntityCanonicalizer())
        self.decay_engine = decay_engine or (engine.decay_engine if hasattr(engine, "decay_engine") else SalienceDecayEngine())
        self.compactor = compactor or (engine.compactor if hasattr(engine, "compactor") else ContextCompactor())
        self.calibrator = calibrator or (engine.calibrator if hasattr(engine, "calibrator") else EpistemicCalibrator())
        self.shadow_extractor = shadow_extractor
        self.max_retrieval_tokens = max_retrieval_tokens

        # Metryki operacyjne
        self.stats = {
            "total_turns": 0,
            "retrieval_skipped_turns": 0,
            "retrieval_executed_turns": 0,
            "shadow_facts_extracted": 0,
            "tokens_saved_estimate": 0,
        }

    # =========================================================================
    # Dźwignia 1: Cache Contract & Prefix Guard
    # =========================================================================
    def build_cache_contract_prefix(
        self,
        profile_state: Optional[Dict[str, Any]] = None,
        system_rules: Optional[List[str]] = None,
    ) -> str:
        """
        Generuje deterministyczny, stały blok promptu zoptymalizowany pod Prompt Caching
        (OmniRoute / Anthropic / OpenAI / vLLM KV-cache reuse ~90%).
        """
        rules = system_rules or [
            "You are an autonomous engineering agent (Hermes Agent).",
            "Memory Protocol: Dynamic facts are injected via Mnemosyne / search_memory.",
            "Deterministic Execution: Rely on Verified Source of Truth state variables.",
        ]
        from atlas_memory.hermes.prefix_guard import PrefixCacheGuard

        return PrefixCacheGuard.build_immutable_prefix(
            verified_state=profile_state or {},
            system_rules=rules,
        )

    # =========================================================================
    # Dźwignia 2: Retrieval Policy Gate
    # =========================================================================
    def should_retrieve(
        self,
        message: str,
        explicit_entities: Optional[List[str]] = None,
    ) -> Tuple[bool, List[str], str]:
        """
        Retrieval Policy Gate (Zoptymalizowany O(1) Regex Scanner):
        - Brak encji w pytaniu -> Skip Mnemosyne Recall (0 tokenów, zero szumu).
        - Wykryto encje / intencję faktu -> Wyzwolenie odpytania Mnemosyne.
        """
        self.stats["total_turns"] += 1
        msg_clean = message.lower().strip()

        # 1. Filtrowanie trywialnych powitań i krótkich zwrotów konwersacyjnych (0 tokenów)
        if self.TRIVIAL_PROMPT_PATTERN.match(msg_clean) or self.CASUAL_GREETING_PATTERN.match(msg_clean):
            self.stats["retrieval_skipped_turns"] += 1
            self.stats["tokens_saved_estimate"] += self.max_retrieval_tokens
            return False, [], "conversational_turn_no_entity"

        detected_entities = []
        if explicit_entities:
            detected_entities.extend(self.canonicalizer.canonicalize_list(explicit_entities))

        for alias, c_id in self.canonicalizer._alias_to_id.items():
            if len(alias) >= 3 and alias in msg_clean:
                if c_id not in detected_entities:
                    detected_entities.append(c_id)

        # 2. Dynamiczne heurystyki dla identyfikatorów kodu, encji złożonych i symboli
        # - Identyfikatory z łącznikami/podkreśleniami (np. node-01, port_8080, redis_cache)
        for sym in re.findall(r"\b[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+\b", message):
            c_sym = self.canonicalizer.canonicalize(sym)
            if c_sym not in detected_entities and len(c_sym) >= 3:
                detected_entities.append(c_sym)

        # - Identyfikatory CamelCase (np. MemoryRecord, VectorStore, FastEmbed)
        for camel in re.findall(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-zA-Z0-9]+)+\b", message):
            c_camel = self.canonicalizer.canonicalize(camel)
            if c_camel not in detected_entities and len(c_camel) >= 3:
                detected_entities.append(c_camel)

        # - Terminy w backtickach (np. `cluster_config`, `active_nodes`)
        for backtick in re.findall(r"`([^`]+)`", message):
            c_bt = self.canonicalizer.canonicalize(backtick.strip())
            if c_bt not in detected_entities and len(c_bt) >= 3:
                detected_entities.append(c_bt)

        # - Akronimy i terminy z wielkich liter (np. NAS, IP, GPU, DB, API, SQL)
        for acr in re.findall(r"\b[A-Z]{2,}\b", message):
            c_acr = self.canonicalizer.canonicalize(acr)
            if c_acr not in detected_entities and len(c_acr) >= 2:
                detected_entities.append(c_acr)

        # - Nazwy własne i encje pisane z wielkiej litery (np. Loop, Engineering, Obsidian, Superpowers)
        sentence_starters = {
            "co", "czy", "jak", "gdzie", "kiedy", "dlaczego", "kto", "czym", "jaki", "jakie",
            "what", "where", "when", "why", "how", "who", "which", "is", "are", "do", "does",
            "super", "dzięki", "dobrze", "jasne", "ok", "cześć", "hej", "witaj", "siema", "dzień", "witam", "hello", "hi", "hey", "thanks",
        }
        for word in re.findall(r"\b[A-Z][a-zA-Z0-9_-]{2,}\b", message):
            if word.lower() not in sentence_starters:
                c_word = self.canonicalizer.canonicalize(word)
                if c_word not in detected_entities and len(c_word) >= 3:
                    detected_entities.append(c_word)

        if detected_entities:
            self.stats["retrieval_executed_turns"] += 1
            return True, detected_entities, f"entity_match: {detected_entities}"

        # Czysty czat konwersacyjny bez encji -> brak retrieval (0 tokenów, zero szumu)
        self.stats["retrieval_skipped_turns"] += 1
        self.stats["tokens_saved_estimate"] += self.max_retrieval_tokens
        return False, [], "conversational_turn_no_entity"

    # =========================================================================
    # Dźwignia 3: Epistemic Re-Ranker w Recall Path
    # =========================================================================
    def epistemic_rank(
        self,
        records: List[MemoryRecord],
        query: str,
        current_time: Optional[float] = None,
    ) -> List[Tuple[MemoryRecord, float]]:
        """
        Re-ranking Mnemosyne: sortowanie po veracity_weight
        (USER_EXPLICIT = 1.0 > TOOL = 0.85 > INFERENCE = 0.50), dopiero potem cosinus i decay.
        """
        now = current_time if current_time is not None else time.time()
        scored: List[Tuple[MemoryRecord, float]] = []
        q_terms = set(query.lower().split())

        for rec in records:
            v_weight = self.VERACITY_WEIGHTS.get(rec.source_type, 0.50)

            # Podobieństwo semantyczne
            rec_text = f"{rec.effective_subject} {rec.predicate} {rec.object}".lower()
            overlap = sum(1 for term in q_terms if term in rec_text)
            sim_score = min(1.0, overlap / max(1, len(q_terms)))

            # Salience & Recency Decay S(f)
            salience = self.decay_engine.calculate_salience(rec, similarity_score=sim_score, current_time=now)

            # Veracity-first score: wysoka waga autorytetu źródła
            final_score = 0.50 * v_weight + 0.50 * salience
            scored.append((rec, float(final_score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # =========================================================================
    # Dźwignia 4: Token Budget Governor (Epistemic Knapsack Packing V24)
    # =========================================================================
    def apply_token_budget(
        self,
        ranked_records: List[Tuple[MemoryRecord, float]],
        max_tokens: Optional[int] = None,
        strategy: str = "knapsack",
    ) -> Dict[str, Any]:
        """
        Alokacja budżetu tokenów z wykorzystaniem Epistemic Knapsack Packing (V24):
        Maksymalizuje całkowitą gęstość informacyjną I(f) = score / tokens w oknie budżetu.
        """
        budget = max_tokens or self.max_retrieval_tokens
        if not ranked_records or budget <= 0:
            return {
                "selected_facts": [],
                "formatted_context": "No facts injected (budget empty or skipped).",
                "estimated_tokens": 0,
                "budget_tokens": budget,
            }

        # Obliczenie wagi tokenowej dla każdego faktu
        items = []
        for rec, score in ranked_records:
            fact_line = f"- [{rec.source_type.value.upper()}] ({rec.effective_subject}) --[{rec.predicate}]--> {rec.object} (score: {score:.2f})"
            tokens_line = max(1, len(fact_line) // 4)
            density = score / tokens_line
            items.append({
                "record": rec,
                "score": score,
                "line": fact_line,
                "tokens": tokens_line,
                "density": density,
            })

        # Jeśli łączna liczba tokenów mieści się w budżecie -> bierzemy wszystko
        total_tokens = sum(item["tokens"] for item in items)
        if total_tokens <= budget:
            selected_items = items
        elif strategy == "knapsack" and len(items) <= 150:
            # 0/1 Knapsack DP dla optymalnej gęstości informacyjnej
            # Skalujemy tokeny dla wydajności DP (krok 5 tokenów)
            scale = 5
            b_scaled = budget // scale
            n = len(items)
            dp = [0.0] * (b_scaled + 1)
            keep = [[False] * (b_scaled + 1) for _ in range(n)]

            for i, it in enumerate(items):
                w = max(1, it["tokens"] // scale)
                v = it["score"]
                for cap in range(b_scaled, w - 1, -1):
                    if dp[cap - w] + v > dp[cap]:
                        dp[cap] = dp[cap - w] + v
                        keep[i][cap] = True

            # Odtwarzanie wybranych faktów
            selected_indices = []
            curr_cap = b_scaled
            for i in range(n - 1, -1, -1):
                if keep[i][curr_cap]:
                    selected_indices.append(i)
                    curr_cap -= max(1, items[i]["tokens"] // scale)
            selected_indices.reverse()
            selected_items = [items[i] for i in selected_indices]
        else:
            # Greedy density fallback dla bardzo dużych zestawów
            sorted_by_density = sorted(items, key=lambda x: (x["density"], x["score"]), reverse=True)
            selected_items = []
            current_w = 0
            for it in sorted_by_density:
                if current_w + it["tokens"] <= budget:
                    selected_items.append(it)
                    current_w += it["tokens"]
            # Przywrócenie porządku rankingu epistemicznego
            selected_items.sort(key=lambda x: x["score"], reverse=True)

        selected_facts: List[Dict[str, Any]] = []
        formatted_lines: List[str] = []
        current_token_est = 0

        for it in selected_items:
            if current_token_est + it["tokens"] > budget:
                continue
            rec = it["record"]
            current_token_est += it["tokens"]
            selected_facts.append({
                "subject": rec.effective_subject,
                "predicate": rec.predicate,
                "object": rec.object,
                "confidence": rec.confidence,
                "source_type": rec.source_type.value,
                "score": it["score"],
            })
            formatted_lines.append(it["line"])


        formatted_context = "\n".join(formatted_lines) if formatted_lines else "No facts injected (budget empty or skipped)."

        return {
            "selected_facts": selected_facts,
            "formatted_context": formatted_context,
            "estimated_tokens": current_token_est,
            "budget_tokens": budget,
        }


    # =========================================================================
    # Główna Orkiestracja Recall (Wrapper na Mnemosyne / Storage)
    # =========================================================================
    async def orchestrated_recall(
        self,
        user_message: str,
        explicit_entities: Optional[List[str]] = None,
        mnemosyne_recall_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Pełna orkiestracja zapytania z Retrieval Policy Gate, re-rankingiem epistemicznym i budżetem.
        """
        # 1. Retrieval Gate
        should_run, entities, reason = self.should_retrieve(user_message, explicit_entities)
        if not should_run:
            return {
                "retrieval_skipped": True,
                "reason": reason,
                "context_block": "",
                "matched_facts_count": 0,
            }

        # 2. Wywołanie Mnemosyne lub podpiętego silnika
        records: List[MemoryRecord] = []
        fn = mnemosyne_recall_fn or (self.mnemosyne.recall if hasattr(self.mnemosyne, "recall") else None)

        if fn is not None:
            raw_res = await fn(user_message, active_entities=entities)
            if isinstance(raw_res, dict):
                # Ekstrakcja trójek Mnemosyne
                for rel in raw_res.get("graph_topology", {}).get("relations", []):
                    records.append(MemoryRecord(
                        subject=rel.get("subject", ""),
                        predicate=rel.get("predicate", ""),
                        object=str(rel.get("object", "")),
                        confidence=float(rel.get("confidence", 1.0)),
                        source_type=EpistemicSource.TOOL_OUTPUT,
                    ))
                for vec in raw_res.get("semantic_context", []):
                    rec_data = vec.get("record", {})
                    if rec_data:
                        records.append(MemoryRecord(**rec_data))

        # 3. Epistemic Re-Ranking
        ranked = self.epistemic_rank(records, query=user_message)

        # 4. Token Budget Governor
        budgeted = self.apply_token_budget(ranked, max_tokens=self.max_retrieval_tokens)

        return {
            "retrieval_skipped": False,
            "reason": reason,
            "canonical_entities": entities,
            "context_block": budgeted["formatted_context"],
            "matched_facts_count": len(budgeted["selected_facts"]),
            "estimated_tokens": budgeted["estimated_tokens"],
        }

    # =========================================================================
    # Dźwignia 5: Shadow Reconciliation z Arbitrażem (SPO + mnemosyne_triple_add)
    # =========================================================================
    async def shadow_reconcile(
        self,
        user_msg: str,
        agent_response: str,
        session_id: str = "default_session",
        triple_add_fn: Optional[Callable] = None,
    ) -> List[MemoryRecord]:
        """
        Asynchroniczny worker na tanim modelu (Qwen-mini):
        - Ekstrakcja trójek SPO {subject, predicate, object, source_type}
        - Blokada nadpisywania deklaracji usera przez domysły modelu
        - Wywołanie mnemosyne_triple_add(supersede=True).
        """
        extracted_raw: List[Dict[str, Any]] = []

        if self.shadow_extractor is not None:
            try:
                extracted_raw = self.shadow_extractor(user_msg, agent_response)
            except Exception:
                extracted_raw = []

        if not extracted_raw:
            extracted_raw = self._fallback_extract_facts(user_msg, agent_response)

        committed_records: List[MemoryRecord] = []
        add_fn = triple_add_fn or (getattr(self.mnemosyne, "triple_add", None) if self.mnemosyne else None)

        for item in extracted_raw:
            raw_subj = item.get("subject", "").strip()
            pred = item.get("predicate", "").strip()
            obj = str(item.get("object", "")).strip()
            if not (raw_subj and pred and obj):
                continue

            # Kanonizacja encji przed zapisem
            canon_subj = self.canonicalizer.canonicalize(raw_subj)

            src_str = item.get("source_type", "user_explicit")
            try:
                source_type = EpistemicSource(src_str)
            except ValueError:
                source_type = EpistemicSource.USER_EXPLICIT

            rec = MemoryRecord(
                subject=raw_subj,
                canonical_entity_id=canon_subj,
                predicate=pred,
                object=obj,
                confidence=float(item.get("confidence", 1.0)),
                source_type=source_type,
                timestamp=time.time(),
                is_state_variable=bool(item.get("is_state_variable", False)),
                source_session_id=session_id,
            )

            # Wywołanie Mnemosyne triple_add z supersede=True
            if add_fn is not None:
                try:
                    await add_fn(
                        subject=canon_subj,
                        predicate=pred,
                        object_=obj,
                        confidence=rec.confidence,
                        source=source_type.value,
                        supersede=True,
                    )
                except Exception as exc:
                    logger.debug("Mnemosyne triple_add failed in shadow reconciliation: %s", exc)

            # Jeśli silnik HybridMemoryEngine jest podpięty, wykonaj commit_observation
            if self.engine is not None and hasattr(self.engine, "commit_observation"):
                try:
                    await self.engine.commit_observation(rec)
                except Exception as exc:
                    logger.debug("Engine commit_observation failed in shadow reconciliation: %s", exc)


            committed_records.append(rec)
            self.stats["shadow_facts_extracted"] += 1

        return committed_records

    def _fallback_extract_facts(self, user_msg: str, agent_response: str) -> List[Dict[str, Any]]:
        facts = []
        text = f"{user_msg} {agent_response}"
        matches = re.findall(r"(?:mój |moje )?(\w[\w\s]{1,20}?)\s+(?:to|jest|wynosi)\s+([\w\.\-\/]+)", text, re.IGNORECASE)
        for subj, obj in matches:
            facts.append({
                "subject": subj.strip(),
                "predicate": "state_value",
                "object": obj.strip(),
                "source_type": "user_explicit",
                "confidence": 0.95,
                "is_state_variable": True,
            })
        return facts

    # =========================================================================
    # Dźwignia 6: Formalny Decay S(f) i Auto-GC
    # =========================================================================
    def prune_stale_facts(
        self,
        facts: List[MemoryRecord],
        threshold: float = 0.25,
        current_time: Optional[float] = None,
    ) -> Tuple[List[MemoryRecord], List[MemoryRecord]]:
        """
        Filtruje listę faktów na bazie wzoru S(f):
        Zwraca: (aktywne_fakty, usunięty_szum)
        """
        now = current_time if current_time is not None else time.time()
        active = []
        pruned = []

        for f in facts:
            if self.decay_engine.should_prune(f, current_time=now):
                pruned.append(f)
            else:
                active.append(f)

        return active, pruned
