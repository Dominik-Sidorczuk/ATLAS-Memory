"""
Hot Path Performance & Memory Profiler (Harness-Foundry V24).

Measures:
1. Retrieval Policy Gate Latency (P50/P95/P99).
2. Epistemic Knapsack Packing Throughput & Information Density.
3. RaBitQ Fast JIT Asymmetric Scan (100k vectors throughput).
4. Arrow Trajectory Zero-Copy RAM Tensor allocation.
5. RAM Footprint & Tracemalloc delta across all hot paths.

Emits results to docs/baselines/profiling_v24.json.
"""

from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


from atlas_memory.arrow_buffer.trajectory_buffer import ArrowTrajectoryBuffer
from atlas_memory.models import ActionPlan, EpistemicSource, LatentState, MemoryRecord, PredictedTransition
from atlas_memory.orchestrator import MemoryOrchestrator
from atlas_memory.quantization.rabitq_engine import RaBitQEngine


def profile_hot_paths() -> Dict[str, Any]:
    tracemalloc.start()
    results: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # 1. Retrieval Policy Gate Latency
    # -------------------------------------------------------------------------
    orchestrator = MemoryOrchestrator()
    orchestrator.canonicalizer.register_entity("entity_db", "database_server", aliases=["db", "postgres", "sql"])

    queries = [
        "Cześć, jak się masz?",
        "Jaka jest pogoda w Warszawie?",
        "Sprawdź status bazy danych postgres",
        "Podaj konfigurację serwera db",
        "Dzień dobry!",
    ] * 200  # 1,000 queries

    latencies_gate_us: List[float] = []
    for q in queries:
        t0 = time.perf_counter()
        _ = orchestrator.should_retrieve(q)
        t1 = time.perf_counter()
        latencies_gate_us.append((t1 - t0) * 1_000_000.0)

    latencies_gate_us.sort()
    results["policy_gate"] = {
        "iterations": len(queries),
        "p50_latency_us": round(float(np.percentile(latencies_gate_us, 50)), 3),
        "p95_latency_us": round(float(np.percentile(latencies_gate_us, 95)), 3),
        "p99_latency_us": round(float(np.percentile(latencies_gate_us, 99)), 3),
    }

    # -------------------------------------------------------------------------
    # 2. Epistemic Knapsack Packing Throughput
    # -------------------------------------------------------------------------
    records: List[tuple[MemoryRecord, float]] = []
    for i in range(100):
        src = EpistemicSource.USER_EXPLICIT if i % 4 == 0 else EpistemicSource.TOOL_OUTPUT
        rec = MemoryRecord(
            subject=f"entity_{i % 10}",
            predicate=f"relation_{i}",
            object=f"value_configuration_parameter_{i}_{'long_' * (i % 5)}",
            confidence=0.5 + 0.5 * (i / 100.0),
            source_type=src,
        )
        score = 0.5 + 0.5 * (i / 100.0)
        records.append((rec, score))

    t0 = time.perf_counter()
    knapsack_res = orchestrator.apply_token_budget(records, max_tokens=1500, strategy="knapsack")
    t1 = time.perf_counter()

    results["knapsack_packing"] = {
        "input_facts_count": len(records),
        "selected_facts_count": len(knapsack_res["selected_facts"]),
        "budget_tokens": knapsack_res["budget_tokens"],
        "estimated_tokens": knapsack_res["estimated_tokens"],
        "execution_time_ms": round((t1 - t0) * 1000.0, 3),
    }

    # -------------------------------------------------------------------------
    # 3. RaBitQ Fast JIT Asymmetric Scan (100k vectors)
    # -------------------------------------------------------------------------
    dim = 384
    n_candidates = 50_000
    engine = RaBitQEngine(dim=dim, bits=4, seed=42)

    rng = np.random.default_rng(123)
    dataset = rng.standard_normal((n_candidates, dim)).astype(np.float32)
    query = rng.standard_normal(dim).astype(np.float32)

    quantized = engine.quantize(dataset)

    # Warm-up JIT
    _ = engine.fast_asymmetric_scan(query, quantized, top_k=10)

    # Benchmark scan
    t0 = time.perf_counter()
    top_indices = engine.fast_asymmetric_scan(query, quantized, top_k=10)
    t1 = time.perf_counter()

    scan_time_ms = (t1 - t0) * 1000.0
    throughput_vectors_per_sec = int(n_candidates / max(t1 - t0, 1e-9))

    results["rabitq_fast_scan"] = {
        "candidates_count": n_candidates,
        "dimension": dim,
        "bits_per_dim": 4,
        "scan_time_ms": round(scan_time_ms, 3),
        "throughput_vectors_per_sec": throughput_vectors_per_sec,
        "top_indices_sample": top_indices[:3].tolist(),
    }

    # -------------------------------------------------------------------------
    # 4. Arrow Trajectory Zero-Copy RAM Tensor
    # -------------------------------------------------------------------------
    buf = ArrowTrajectoryBuffer(state_dim=32)
    for step in range(500):
        vec = [float(step * 0.01 + j) for j in range(32)]
        trans = PredictedTransition(
            previous_state=LatentState(step_index=step, timestamp=float(step), vector=vec, dimension=32),
            action=ActionPlan(name=f"action_{step % 5}"),
            predicted_state=LatentState(step_index=step + 1, timestamp=float(step + 1), vector=vec, dimension=32),
            simulated_reward=1.0,
            uncertainty=0.05,
        )


        buf.append_transition(trans, session_id="session_bench")

    t0 = time.perf_counter()
    tensor_view = buf.to_zero_copy_tensor()
    t1 = time.perf_counter()

    results["arrow_trajectory_zero_copy"] = {
        "trajectory_steps": len(buf),
        "tensor_shape": list(tensor_view.shape),
        "conversion_time_us": round((t1 - t0) * 1_000_000.0, 3),
    }

    # -------------------------------------------------------------------------
    # 5. Memory Tracemalloc Footprint
    # -------------------------------------------------------------------------
    current_ram, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results["memory_profiling"] = {
        "current_allocated_mb": round(current_ram / (1024 * 1024), 3),
        "peak_allocated_mb": round(peak_ram / (1024 * 1024), 3),
    }

    return results


def main():
    print("=" * 80)
    print(f"{'⚡ ATLAS V24: HOT PATH PROFILER & BENCHMARK HARNESS':^80}")
    print("=" * 80)

    profile = profile_hot_paths()
    print(json.dumps(profile, indent=2))

    # Save to baselines
    out_path = REPO_ROOT / "docs" / "baselines" / "profiling_v24.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    print("-" * 80)
    print(f"✅ Profiling report successfully committed to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
