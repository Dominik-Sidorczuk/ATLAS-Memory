"""
Benchmark ATLAS V15 — 15 metryk SOTA Beyond 2027.

Weryfikuje kluczowe wskaźniki wydajnościowe i poprawnościowe pełnej architektury pamięci ATLAS:
1. import_atlas
2. engine_creation (< 100ms)
3. orchestrator_creation (< 50ms)
4. epistemic_ranking (< 50ms)
5. token_budget (<= 1500 tokens)
6. shadow_reconciliation (< 100ms)
7. causal_what_if (< 200ms)
8. diffusion_analysis (< 300ms)
9. uds_ipc_roundtrip (< 1ms per RPC ping)
10. hash_chain_integrity (100 entries verified)
11. sleep_baking (20 trajectories < 500ms)
12. cpof_detection (50 node graph < 200ms)
13. crdt_delta_sync_convergence (100 entries < 5ms)
14. hamming_popcount_throughput (1000 x 384-dim > 500k ops/sec)
15. skill_compilation_ast_safety (5 steps < 100ms)
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from atlas_memory.causal.models import DiffusionResult
from atlas_memory.causal.retro_causal_edge import RetroCausalEngine
from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.l2_semantic.kv_store import VerifiedKVStore
from atlas_memory.l3_procedural.skill_compiler import compile_sop_to_skill
from atlas_memory.l3_procedural.sleep_baker import SleepBaker, StandardProcedure, Step
from atlas_memory.models import EpistemicSource, MemoryRecord
from atlas_memory.orchestrator import MemoryOrchestrator
from atlas_memory.quantization import AVX512Hamming, MIBQuantizer, QuantizationConfig
from atlas_memory.server.atlas_daemon import AtlasDaemon
from atlas_memory.server.client import AtlasDaemonClient
from atlas_memory.sync.crdt import DeltaCRDT, SyncDelta
from atlas_memory.sync.crypto import SyncCrypto


class MockBenchGraphClient:
    """Wydajny mock graph store dla deterministycznych testów benchmarkowych."""
    def __init__(self, relations=None, multi_hop_paths=None):
        self.relations = relations or []
        self.multi_hop_paths = multi_hop_paths or []

    async def get_subgraph_relations(self, entities, max_depth=2):
        return {"relations": self.relations}

    async def execute_multi_hop_cypher(self, source, target, max_hops=5):
        return self.multi_hop_paths


def test_bench_01_import_atlas():
    """1. Import atlas_memory i wszystkich podmodułów."""
    start = time.perf_counter()
    import atlas_memory
    from atlas_memory import (
        quantization,
        sync,
    )
    elapsed = time.perf_counter() - start
    assert atlas_memory is not None
    assert sync is not None
    assert quantization is not None
    assert elapsed < 1.0


def test_bench_02_memory_engine_creation():
    """2. Inicjalizacja HybridMemoryEngine() < 100ms."""
    # Warmup imports & fastembed once if needed
    _ = HybridMemoryEngine.create_default()

    start = time.perf_counter()
    engine = HybridMemoryEngine.create_default()
    elapsed = time.perf_counter() - start
    assert engine is not None
    assert elapsed < 0.100, f"Engine creation took {elapsed*1000:.2f}ms >= 100ms"


def test_bench_03_orchestrator_creation():
    """3. Inicjalizacja MemoryOrchestrator() < 50ms."""
    engine = HybridMemoryEngine.create_default()
    start = time.perf_counter()
    orchestrator = MemoryOrchestrator(engine=engine)
    elapsed = time.perf_counter() - start
    assert orchestrator is not None
    assert elapsed < 0.050, f"Orchestrator creation took {elapsed*1000:.2f}ms >= 50ms"


def test_bench_04_epistemic_ranking():
    """4. Epistemic ranking dla 100 rekordów < 50ms."""
    engine = HybridMemoryEngine.create_default()
    orchestrator = MemoryOrchestrator(engine=engine)
    now = time.time()
    
    records = [
        MemoryRecord(
            subject=f"entity_{i % 10}",
            predicate="has_param",
            object=f"value_{i}",
            source_type=EpistemicSource.USER_EXPLICIT if i % 2 == 0 else EpistemicSource.AGENT_INFERENCE,
            confidence=0.8 + (i % 20) * 0.01,
            timestamp=now - float(i * 100),
            importance_score=0.9,
        )
        for i in range(100)
    ]

    start = time.perf_counter()
    ranked = orchestrator.epistemic_rank(records, query="entity parameter", current_time=now)
    elapsed = time.perf_counter() - start

    assert len(ranked) == 100
    assert elapsed < 0.050, f"Epistemic ranking took {elapsed*1000:.2f}ms >= 50ms"


def test_bench_05_token_budget():
    """5. Token budget compression: 50 dużych faktów skompresowane do <= 1500 tokenów."""
    engine = HybridMemoryEngine.create_default()
    orchestrator = MemoryOrchestrator(engine=engine)

    big_records = [
        (MemoryRecord(subject=f"entity_{i}", predicate="config_param", object="X" * 120, confidence=1.0), 0.95 - i * 0.01)
        for i in range(50)
    ]

    budgeted = orchestrator.apply_token_budget(big_records, max_tokens=1500)
    tok_est = budgeted["estimated_tokens"]
    assert tok_est <= 1500, f"Token estimate {tok_est} exceeded budget 1500"
    assert len(budgeted["selected_facts"]) > 0


@pytest.mark.asyncio
async def test_bench_06_shadow_reconciliation():
    """6. Rozwiązanie 10 konfliktów w Shadow Auditor < 100ms."""
    engine = HybridMemoryEngine.create_default()
    
    conflicts = [
        MemoryRecord(
            subject=f"service_status_{i}",
            predicate="state",
            object="online",
            source_type=EpistemicSource.USER_EXPLICIT,
            confidence=1.0,
            is_state_variable=True,
        )
        for i in range(10)
    ]

    start = time.perf_counter()
    for rec in conflicts:
        _, override = await engine.auditor.check_and_resolve_conflict(rec)
        assert override is True
    elapsed = time.perf_counter() - start

    assert elapsed < 0.100, f"Shadow reconciliation of 10 conflicts took {elapsed*1000:.2f}ms >= 100ms"
    await engine.kv.close()


@pytest.mark.asyncio
async def test_bench_07_causal_what_if():
    """7. Analiza Causal What-If dla głębokości 2 < 200ms."""
    relations = [
        {"subject": f"Node_{i}", "predicate": "depends_on", "object": f"Node_{i+1}", "confidence": 0.95}
        for i in range(20)
    ]
    graph = MockBenchGraphClient(relations=relations)
    causal_engine = RetroCausalEngine(graph_client=graph)

    start = time.perf_counter()
    paths = await causal_engine.causal_what_if("Node_0", "reboot_service", depth=2)
    elapsed = time.perf_counter() - start

    assert len(paths) >= 1
    assert elapsed < 0.200, f"Causal what_if took {elapsed*1000:.2f}ms >= 200ms"


@pytest.mark.asyncio
async def test_bench_08_diffusion_analysis():
    """8. Causal diffusion analysis < 300ms."""
    relations = [
        {"subject": f"Node_{i}", "predicate": "impacts", "object": f"Node_{i+1}", "confidence": 0.9}
        for i in range(30)
    ]
    graph = MockBenchGraphClient(relations=relations)
    causal_engine = RetroCausalEngine(graph_client=graph)

    start = time.perf_counter()
    res = await causal_engine.causal_diffusion_analysis("Node_0", max_depth=3, decay_factor=0.7)
    elapsed = time.perf_counter() - start

    assert isinstance(res, DiffusionResult)
    assert len(res.nodes) >= 1
    assert elapsed < 0.300, f"Diffusion analysis took {elapsed*1000:.2f}ms >= 300ms"


@pytest.mark.asyncio
async def test_bench_09_uds_ipc_roundtrip(tmp_path: Path):
    """9. UDS IPC ping roundtrip < 1ms avg (po 50 wywołaniach)."""
    sock_path = tmp_path / "bench_ipc.sock"
    pid_path = tmp_path / "bench_ipc.pid"

    daemon = AtlasDaemon(socket_path=sock_path, pid_path=pid_path)
    await daemon.start()
    try:
        client = AtlasDaemonClient(socket_path=sock_path)
        
        # Warmup
        await client.ping()

        # Measure 50 pings
        start = time.perf_counter()
        n_iters = 50
        for _ in range(n_iters):
            ok = await client.ping()
            assert ok is True
        total_time = time.perf_counter() - start
        avg_ms = (total_time / n_iters) * 1000.0

        assert avg_ms < 1.0, f"UDS IPC ping average latency {avg_ms:.3f}ms >= 1.0ms"
        await client.close()
    finally:
        await daemon.stop()


@pytest.mark.asyncio
async def test_bench_10_hash_chain_integrity():
    """10. SHA-256 Hash-Chain: dołączenie 100 wpisów i weryfikacja integralności."""
    kv = VerifiedKVStore(db_path=":memory:")

    start = time.perf_counter()
    for i in range(100):
        await kv.append_audit_log({"key": f"var_{i}", "value": f"val_{i}", "timestamp": 1000.0 + i})
    
    is_valid, broken_seq = await kv.verify_chain_integrity()
    elapsed = time.perf_counter() - start

    assert is_valid is True
    assert broken_seq == 0
    assert elapsed < 0.200, f"100 audit entries append+verify took {elapsed*1000:.2f}ms"
    await kv.close()


def test_bench_11_sleep_baking():
    """11. Sleep Baking: konsolidacja 20 trajektorii < 500ms."""
    baker = SleepBaker(min_frequency=2, min_success_rate=0.7)

    trajectories = []
    for i in range(20):
        pattern_type = i % 3
        if pattern_type == 0:
            steps = ["read_file", "patch", "pytest"]
        elif pattern_type == 1:
            steps = ["search_files", "read_file", "edit"]
        else:
            steps = ["terminal", "check_status"]

        trajectories.append({
            "context": f"task_{pattern_type}",
            "success": True,
            "steps": steps,
        })

    start = time.perf_counter()
    sops = baker.konsolidacja_sesji(trajectories)
    elapsed = time.perf_counter() - start

    assert len(sops) == 3
    assert elapsed < 0.500, f"Sleep baking 20 trajectories took {elapsed*1000:.2f}ms >= 500ms"


@pytest.mark.asyncio
async def test_bench_12_cpof_detection():
    """12. Detekcja CPoF na grafie 50 węzłów < 200ms."""
    # Tworzymy topologię: Root -> Hub -> [L1..L48]
    relations = [{"subject": "Root", "predicate": "connects", "object": "Hub", "confidence": 1.0}]
    for i in range(48):
        relations.append({"subject": "Hub", "predicate": "connects", "object": f"Leaf_{i}", "confidence": 1.0})

    graph = MockBenchGraphClient(relations=relations)
    causal_engine = RetroCausalEngine(graph_client=graph)

    start = time.perf_counter()
    cpofs = await causal_engine.detect_cpof("Root")
    elapsed = time.perf_counter() - start

    assert len(cpofs) >= 1
    assert cpofs[0].node_id == "Hub"
    assert len(cpofs[0].affected_nodes) == 48
    assert elapsed < 0.200, f"CPoF detection on 50 nodes took {elapsed*1000:.2f}ms >= 200ms"


def test_bench_13_crdt_delta_sync_convergence():
    """13. Delta-CRDT + AES-256-GCM: 100 wpisów export_delta + encrypt + decrypt + apply_delta roundtrip < 5ms."""
    key = SyncCrypto.generate_key()
    crypto = SyncCrypto(key)
    node_a = DeltaCRDT(node_id="agent_a")
    node_b = DeltaCRDT(node_id="agent_b")

    set_a = node_a.get_set("shared_kb")
    for i in range(100):
        set_a.add(f"knowledge_fact_{i}", timestamp=1000.0 + float(i), node_id="agent_a")
    node_a.clock.increment("agent_a")

    start = time.perf_counter()
    delta_a = node_a.export_delta()
    raw_payload = delta_a.model_dump_json().encode("utf-8")
    enc = crypto.encrypt(raw_payload)
    dec = crypto.decrypt(enc)
    restored_delta = SyncDelta.model_validate_json(dec.decode("utf-8"))
    node_b.apply_delta(restored_delta)
    elapsed = time.perf_counter() - start

    set_b = node_b.get_set("shared_kb")
    assert set_b.lookup("knowledge_fact_0") is True
    assert set_b.lookup("knowledge_fact_99") is True
    assert elapsed < 0.005, f"CRDT delta sync roundtrip took {elapsed*1000:.2f}ms >= 5ms"


def test_bench_14_hamming_popcount_throughput():
    """14. AVX-512 / Numba Hamming Distance throughput: 1000 x 384-dim wektorów > 500,000 ops/sec."""
    import numpy as np
    quantizer = MIBQuantizer(QuantizationConfig(target_dim=384))
    np.random.seed(42)
    raw_candidates = np.random.randn(1000, 384).astype(np.float32)
    raw_query = np.random.randn(384).astype(np.float32)

    candidates = quantizer.compress(raw_candidates)
    query = quantizer.quantize(raw_query).data

    # JIT warmup
    _ = AVX512Hamming.batch_hamming(query, candidates)

    iterations = 500
    start = time.perf_counter()
    for _ in range(iterations):
        _ = AVX512Hamming.batch_hamming(query, candidates)
    elapsed = time.perf_counter() - start

    total_ops = iterations * 1000
    ops_per_sec = total_ops / elapsed

    assert ops_per_sec > 500_000, f"Hamming throughput {ops_per_sec:.0f} ops/s <= 500,000 ops/s"


def test_bench_15_skill_compilation_ast_safety(tmp_path: Path):
    """15. Autonomous Skill Compilation + AST Safety scan: 5-step SOP < 100ms."""
    steps = [
        Step(tool_name="read_file", params_pattern={"path": "/tmp/test.txt"}, expected_outcome="content"),
        Step(tool_name="search_files", params_pattern={"pattern": "TODO"}, expected_outcome="matches"),
        Step(tool_name="patch", params_pattern={"path": "/tmp/test.txt", "old": "a", "new": "b"}, expected_outcome="diff"),
        Step(tool_name="terminal", params_pattern={"command": "pytest -q"}, expected_outcome="passed"),
        Step(tool_name="write_file", params_pattern={"path": "/tmp/out.txt", "content": "done"}, expected_outcome="written"),
    ]
    sop = StandardProcedure(
        procedure_id="bench_proc_pipeline",
        name="Benchmark Pipeline Procedure",
        metadata={"context": "test_context"},
        steps=steps,
        success_rate=1.0,
        invocations_count=10,
    )

    out_dir = tmp_path / "compiled_skill"

    start = time.perf_counter()
    skill_path = compile_sop_to_skill(sop, target_dir=out_dir)
    elapsed = time.perf_counter() - start

    assert skill_path.exists()
    assert (skill_path / "SKILL.md").exists()
    assert (skill_path / "scripts" / "handler.py").exists()
    assert elapsed < 0.100, f"Skill compilation took {elapsed*1000:.2f}ms >= 100ms"
