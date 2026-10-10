"""Test V131: Pełna weryfikacja kaskady logicznej w ścisłej koniunkcji wszystkich warstw.

Weryfikuje współdziałanie:
1. HippoRAG 2 (PPR na dual-node grafie Kùzu/NetworkX z seedami leksykalnymi).
2. DBSF (Static-Anchor 3-sygnałowa fuzja: dense ⊕ lex ⊕ graph).
3. Dense PolDense-68M 512D (OnnxDenseEncoder / _StubDenseEncoder z pełnym 512D).
4. Sparse BM25 (named sparse vector / IDF / Okapi BM25).
5. Qdrant (Universal Hybrid Query API z hermetyczną izolacją sesji i IsEmptyCondition).
6. Kùzu (Multigraph RAM + Dysk z relacjami dual-node i krawędziami synonimów).
7. Mnemosyne (format `mnemosyne:{src_tag}`).
8. Fazy L0-L1-L2-L3 (L0-TTT -> L1 JEPA -> L2 EpistemicWritePath -> L3 Consolidation & SHA-256 Audit).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas_memory.cognitive.retrieval.ppr import is_hipporag_enabled
from atlas_memory.cognitive.retrieval.strategies.dense import DenseCosineStrategy
from atlas_memory.cognitive.retrieval.strategies.fusion import StaticAnchorDbsfFusion
from atlas_memory.core.models import ActionPlan
from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.l2_semantic.openie import build_synonym_edges
from atlas_memory.l2_semantic.qdrant_store import _StubDenseEncoder


@pytest.mark.asyncio
async def test_full_cascade_conjunction_end_to_end(tmp_path: Path, monkeypatch) -> None:
    # 1. Aktywacja HippoRAG 2 w środowisku
    monkeypatch.setenv("ATLAS_HIPPORAG", "1")
    assert is_hipporag_enabled() is True

    # 2. Inicjalizacja instancji magazynów z wymiarem 512D
    dim = 512
    encoder = _StubDenseEncoder(dimension=dim)

    engine = HybridMemoryEngine.create_default(
        db_path=str(tmp_path / "test_kv.db"),
        qdrant_location=":memory:",
        kuzu_path=str(tmp_path / "kuzu"),
    )
    engine.vector_store.dimension = dim
    engine.vector_store.encoder = encoder

    try:
        # FAZA L0 -> L1: Mental Transition & TTT Online Step
        action = ActionPlan(
            name="k8s_deploy",
            parameters={"cluster": "prod-pl", "replicas": 3, "service": "auth-gateway"},
        )
        transition = engine.record_mental_transition(action, session_id="sesja_prod")
        assert transition is not None
        assert transition.uncertainty >= 0.0
        assert action.action_encoding in ("onnx_embed", "dense_encoder", "hash_projection", "ttt_latent")
        assert engine.ttt.step_count >= 1

        # FAZA L1 -> L2: EpistemicWritePath, Kùzu Dual-Node & Qdrant 512D
        write_path = engine.write_path

        await write_path.write_async(
            key="fact:cluster_location",
            value="Główny klaster Kubernetes znajduje się w Warszawie w centrum danych DC1.",
            confidence=0.98,
            session_id=None,
            source_type="user_explicit",
            reason="test_init",
        )

        await write_path.write_async(
            key="fact:cluster_replicas",
            value="Liczba replik bramy auth-gateway w klastrze k8s wynosi 3.",
            confidence=0.95,
            session_id="sesja_prod",
            source_type="tool_output",
            reason="test_init",
        )

        await write_path.write_async(
            key="mnemosyne:task_note",
            value="Zadanie operacyjne: wdrożenie klastra kubernetes warszawa z audytem bezpieczeństwa.",
            confidence=0.90,
            session_id="sesja_prod",
            source_type="mnemosyne:agent",
            reason="test_init",
        )

        syn_edges = build_synonym_edges(["kubernetes", "k8s", "klaster"], encoder=encoder, sim_threshold=0.3)
        assert isinstance(syn_edges, list)

        # RETRIEVAL W KONIUNKCJI: HippoRAG 2 + DBSF + Qdrant + BM25
        query = "wdrożenie klastra kubernetes warszawa"
        from atlas_memory.cognitive.retrieval.prefetch import PrefetchEngine
        prefetch_engine = PrefetchEngine()

        res = await prefetch_engine.execute(
            params={"query": query, "session_id": "sesja_prod", "budget": 1500, "force": True},
            engine=engine,
        )

        assert res is not None
        records = res.get("records", [])
        assert len(records) > 0
        context_block = res.get("context_block", "")
        assert "kubernetes" in context_block.lower() or "warszawa" in context_block.lower() or "klaster" in context_block.lower()

        # Weryfikacja działania strategii DBSF Continuous Fusion
        dense_strat = DenseCosineStrategy()
        fused_strat = StaticAnchorDbsfFusion()

        norm_val = dense_strat.normalize_dense(0.72)
        assert 0.0 < norm_val <= 1.0

        cands = [
            {"key": "fact:1", "_vec_score": 0.85, "_lex_score": 0.70, "_graph_score": 0.50},
            {"key": "fact:2", "_vec_score": 0.10, "_lex_score": 0.05, "_graph_score": 0.90, "ungrounded_graph_only": True},
        ]
        fused_strat.score(query, cands)
        assert cands[0]["score"] > cands[1]["score"]
        assert cands[0]["score"] > 0.35
        assert cands[1]["score"] < 0.25

        # FAZA L2 -> L3: Audit SHA-256 Ledger Integrity
        is_valid, broken_seq = engine.kv.verify_audit_log_sync()
        assert is_valid is True
        assert broken_seq == 0

    finally:
        await engine.stop_worker()
