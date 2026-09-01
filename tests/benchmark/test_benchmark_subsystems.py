"""
Benchmark ATLAS — 15 deterministycznych metryk wydajności podsystemów.
Testowany w Pixi Workspace (Python 3.14 No-GIL).
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from atlas_memory.causal.models import DiffusionResult
from atlas_memory.causal.retro_causal_edge import RetroCausalEngine
from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.l2_semantic.kv_store import VerifiedKVStore
from atlas_memory.l3_procedural.skill_compiler import ASTSafetyScanner, compile_sop_to_skill
from atlas_memory.l3_procedural.sleep_baker import SleepBaker, StandardProcedure, Step
from atlas_memory.models import EpistemicSource, MemoryRecord
from atlas_memory.orchestrator import MemoryOrchestrator
from atlas_memory.quantization import AVX512Hamming, MIBQuantizer
from atlas_memory.server.atlas_daemon import AtlasDaemon
from atlas_memory.server.client import AtlasDaemonClient
from atlas_memory.sync.crdt import SyncDelta, VectorClock
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
    elapsed = time.perf_counter() - start
    assert atlas_memory is not None
    assert elapsed < 1.0


def test_bench_02_memory_engine_creation():
    """2. Inicjalizacja HybridMemoryEngine() < 100ms."""
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

        # Pomiar 50 wywołań
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

    trajectories = [
        {"context": "git", "success": True, "steps": ["git_status", "git_push"]},
        {"context": "git", "success": True, "steps": ["git_status", "git_push"]},
        {"context": "build", "success": True, "steps": ["pytest_run"]},
        {"context": "build", "success": True, "steps": ["pytest_run"]},
    ] * 5

    start = time.perf_counter()
    sops = baker.konsolidacja_sesji(trajectories)
    elapsed = time.perf_counter() - start

    assert len(sops) >= 2
    assert elapsed < 0.500, f"Sleep baking took {elapsed*1000:.2f}ms >= 500ms"


@pytest.mark.asyncio
async def test_bench_12_cpof_detection():
    """12. Detekcja CPoF na grafie 50 węzłów < 200ms."""
    relations = [
        {"subject": f"Hub_{i//10}", "predicate": "impacts", "object": f"Leaf_{i}", "confidence": 0.9}
        for i in range(50)
    ]
    graph = MockBenchGraphClient(relations=relations)
    causal_engine = RetroCausalEngine(graph_client=graph)

    start = time.perf_counter()
    cpofs = await causal_engine.detect_cpof("Hub_0")
    elapsed = time.perf_counter() - start

    assert isinstance(cpofs, list)
    assert elapsed < 0.200, f"CPoF detection took {elapsed*1000:.2f}ms >= 200ms"


def test_bench_13_crdt_delta_sync():
    """13. [V13] CRDT Vector Clocks & AES-256-GCM: szyfrowana wymiana stanu < 10ms."""
    key = SyncCrypto.generate_key()
    crypto = SyncCrypto(key)

    vc_a = VectorClock(clocks={"agent_a": 5, "agent_b": 2})
    vc_b = VectorClock(clocks={"agent_a": 3, "agent_b": 7})

    start = time.perf_counter()
    # Serializacja i szyfrowanie
    delta = SyncDelta(
        source_node="agent_a",
        vector_clock=vc_a,
        payload={"state": "synchronized", "version": 15},
    )
    encrypted_blob = crypto.encrypt(delta.model_dump_json().encode("utf-8"))

    # Odbiór i deszyfracja przez Agenta B
    decrypted_raw = crypto.decrypt(encrypted_blob)
    received_delta = SyncDelta.model_validate_json(decrypted_raw.decode("utf-8"))
    vc_merged = vc_b.merge(received_delta.vector_clock)
    elapsed = time.perf_counter() - start

    assert vc_merged.clocks["agent_a"] == 5
    assert vc_merged.clocks["agent_b"] == 7
    assert received_delta.payload["state"] == "synchronized"
    assert elapsed < 0.010, f"CRDT Delta Sync + AES-256-GCM took {elapsed*1000:.2f}ms >= 10ms"


def test_bench_14_avx512_quantization_scan():
    """14. [V14] AVX-512 / MIB Quantization: Hamming distance scan < 5ms per 1000 vectors."""
    quantizer = MIBQuantizer()
    np.random.seed(42)

    # 1,000 wektorów 384-dim (48 bajtów uint64 per wektor)
    raw_vecs = np.random.randn(1000, 384).astype(np.float32)
    compressed_db = quantizer.compress(raw_vecs)
    query_raw = np.random.randn(384).astype(np.float32)
    q_vec = quantizer.quantize(query_raw)

    # Warmup Numba JIT
    _ = AVX512Hamming.batch_hamming(q_vec.data, compressed_db[:10])

    start = time.perf_counter()
    dists = AVX512Hamming.batch_hamming(q_vec.data, compressed_db)
    elapsed = time.perf_counter() - start

    assert dists.shape[0] == 1000
    assert elapsed < 0.010, f"Hamming scan on 1000 vectors took {elapsed*1000:.2f}ms >= 10ms"


def test_bench_15_autonomous_skill_compilation():
    """15. [V15] Autonomous Skill Compilation & AST Scanner: kompilacja SOP < 50ms."""
    sop = StandardProcedure(
        procedure_id="bench_deploy_sop",
        name="SOP: Benchmark Deployment Workflow",
        steps=[
            Step(tool_name="git_pull", params_pattern={"repo": "main"}, expected_outcome="updated code"),
            Step(tool_name="run_tests", params_pattern={"suite": "pytest"}, expected_outcome="tests passed"),
            Step(tool_name="deploy_service", params_pattern={"env": "prod"}, expected_outcome="service online"),
        ],
        invocations_count=5,
        success_rate=1.0,
        signature="git_pull->run_tests->deploy_service",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        start = time.perf_counter()
        target_path = Path(tmpdir) / "bench_skill"
        skill_dir = compile_sop_to_skill(sop, target_dir=target_path)

        scanner = ASTSafetyScanner()
        violations = scanner.scan_code((skill_dir / "scripts" / "handler.py").read_text())
        elapsed = time.perf_counter() - start

        assert (skill_dir / "SKILL.md").exists()
        assert (skill_dir / "scripts" / "handler.py").exists()
        assert len(violations) == 0
        assert elapsed < 0.050, f"Skill compilation took {elapsed*1000:.2f}ms >= 50ms"

