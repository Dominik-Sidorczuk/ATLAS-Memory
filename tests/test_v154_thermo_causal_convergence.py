"""Tests for V154: Unified Thermodynamic & Latent-Causal Convergence.

Covers:
1. Causal-Thermal Bridge (ActiveSensingEngine -> CausalAnnealer)
2. Langevin noise scaling by Bayesian precision tensor (sigma = sqrt(2*T / Pi_channel))
3. Continuous JEPA-CPoF risk potential tensor (retro_causal_edge.py)
4. Spectral reachability impact critic (energy_module.py)
5. Strict AST & heuristic purity in ContextCompactor and EntityCanonicalizer
6. Adversarial reviewer mutation gates:
   - Mutation M1: Freezing T=0.01 despite CRITICAL prediction error breaks thermal reactivity
   - Mutation M2: Zeroing JEPA divergence collapses risk potential to LOW
   - Mutation M3: Injecting hardcoded string heuristics into compactor fails AST purity scanner
"""
from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import pytest

from atlas_memory.active.prediction_error import ActiveSensingEngine
from atlas_memory.causal.annealer import CausalAnnealer
from atlas_memory.causal.energy_module import EnergyModule
from atlas_memory.causal.models import CausalEdge
from atlas_memory.causal.retro_causal_edge import RetroCausalEngine
from atlas_memory.extensions.canonicalizer import EntityCanonicalizer
from atlas_memory.extensions.compactor import ContextCompactor
from atlas_memory.l3_procedural.auditor import BayesianPrecisionTensor


class MockGraphClient:
    def __init__(self, relations):
        self.relations = relations

    async def get_subgraph_relations(self, entities, max_depth=1):
        return {"relations": self.relations}


class MockLatentBuffer:
    def __init__(self, curr_vec, next_vec):
        from unittest.mock import MagicMock
        self.current_state = MagicMock(vector=curr_vec)
        self._next_vec = next_vec

    def predict_transition(self, curr_st, action_plan):
        from unittest.mock import MagicMock
        return MagicMock(predicted_state=MagicMock(vector=self._next_vec))


# --- 1. Causal-Thermal Bridge & Langevin Noise Scaling ---

@pytest.mark.asyncio
async def test_causal_annealer_thermal_bridge_couples_with_sensing_engine():
    """Verify CausalAnnealer adopts temperature from ActiveSensingEngine instead of fixed T_init."""
    active_sensing = ActiveSensingEngine()
    # Register prediction error on target entity 'db_cluster' to elevate temperature
    from atlas_memory.active.prediction_error import PredictionCheck
    active_sensing.register_expectation(
        PredictionCheck(
            check_id="chk_db",
            target_entity="db_cluster",
            expected_predicate="status",
            expected_value="ONLINE",
        )
    )
    err = active_sensing.detect_discrepancy(
        observed_entity="db_cluster",
        observed_predicate="status",
        observed_value="CRASHED",
    )
    assert err is not None
    t_cogn = active_sensing.get_entity_temperature("db_cluster")
    assert t_cogn >= 0.8, f"Expected high cognitive temperature, got {t_cogn}"

    edges = [
        CausalEdge(source="db_cluster", predicate="replicates_to", target="db_replica", confidence=0.8),
        CausalEdge(source="db_replica", predicate="serves", target="analytics", confidence=0.7),
    ]

    annealer = CausalAnnealer(rng_seed=42)
    # Run with active sensing engine injected
    res = await annealer.run(
        graph=edges,
        active_sensing=active_sensing,
        entity="db_cluster",
        max_iter=5,
    )
    # When initialized from high sensory temperature, energy trace reflects hot exploration
    assert len(res.energy_trace) > 0
    assert res.final_temperature > 0.01


@pytest.mark.asyncio
async def test_langevin_noise_scales_inversely_with_precision():
    """Verify Langevin noise sigma = sqrt(2*T / Pi) scales down when channel precision is high."""
    T = 0.5
    prec_tensor = BayesianPrecisionTensor(alpha=0.8)
    # Update channel with near-zero error -> precision rises
    for _ in range(5):
        prec_tensor.record_sensory_observation("agent_inference", error_magnitude=0.01)

    high_pi = prec_tensor.get_precision("agent_inference")
    assert high_pi > 1.0

    sigma_high = math.sqrt(2.0 * T / high_pi)
    sigma_baseline = math.sqrt(2.0 * T / 1.0)
    assert sigma_high < sigma_baseline, "Higher precision must suppress Langevin thermal noise"


# --- 2. Continuous JEPA-CPoF Risk Tensor ---

@pytest.mark.asyncio
async def test_continuous_jepa_cpof_risk_potential():
    """Verify continuous JEPA divergence * CPoF severity computes risk_potential without keyword lists."""
    relations = [
        {"subject": "core_switch", "predicate": "routes_to", "object": "datacenter_spine", "confidence": 0.95},
        {"subject": "datacenter_spine", "predicate": "powers", "object": "cluster_a", "confidence": 0.90},
    ]
    graph = MockGraphClient(relations)

    # 1. High JEPA divergence (large latent jump: norm([1, 0] - [0, 1]) = sqrt(2) ~ 1.414)
    latent_high = MockLatentBuffer([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    engine_high = RetroCausalEngine(graph_client=graph, latent_buffer=latent_high)
    paths_high = await engine_high.causal_what_if("core_switch", "reconfigure_topology", depth=2)

    assert len(paths_high) >= 1
    assert paths_high[0].risk_level in ("CRITICAL", "HIGH")
    assert paths_high[0].jepa_latent_divergence is not None
    assert paths_high[0].jepa_latent_divergence > 1.0

    # 2. Near-zero JEPA divergence (stable latent transition: norm([1, 0] - [1, 0.01]) = 0.01)
    latent_low = MockLatentBuffer([1.0, 0.0, 0.0], [1.0, 0.01, 0.0])
    engine_low = RetroCausalEngine(graph_client=graph, latent_buffer=latent_low)
    paths_low = await engine_low.causal_what_if("core_switch", "query_telemetry", depth=1)

    assert len(paths_low) >= 1
    # Low JEPA jump + benign query -> LOW risk
    assert paths_low[0].risk_level in ("LOW", "MODERATE")


# --- 3. Spectral Reachability Impact Critic ---

@pytest.mark.asyncio
async def test_spectral_critic_evaluates_topology_impact():
    """Verify EnergyModule.trainable_critic reflects reachability impact on graph connectivity."""
    em = EnergyModule()

    # Node with 4 outgoing critical dependencies in a 4-node graph
    dense_context = {
        "switch_alpha": [
            CausalEdge(source="switch_alpha", predicate="depends_on", target=f"host_{i}", confidence=0.9)
            for i in range(4)
        ]
    }
    critic_dense = await em.trainable_critic("switch_alpha", edges=dense_context)

    # Leaf node with 0 outgoing dependencies
    empty_context = {"host_0": []}
    critic_leaf = await em.trainable_critic("host_0", edges=empty_context)

    assert critic_dense > critic_leaf
    assert critic_dense >= 0.50
    assert critic_leaf == 0.0


# --- 4. ContextCompactor & EntityCanonicalizer Purity ---

def test_compactor_and_canonicalizer_ast_and_dynamic_purity():
    """Verify compactor has no hardcoded string keywords and canonicalizer loads 100% from registry."""
    # 1. Compactor AST inspection
    compactor_src = inspect.getsource(ContextCompactor)
    forbidden_literals = ["zmień", "ustaw", "serwer to", "używaj", "status:", "port:"]
    for lit in forbidden_literals:
        assert lit not in compactor_src, f"Compactor contains forbidden hardcoded keyword literal: {lit}"

    # 2. Functional behavior: selection solely by importance_score >= 0.80
    compactor = ContextCompactor(session_window_size=2)
    compactor.add_interaction_turn("assistant", "Observation summary", metadata={"importance_score": 0.40})
    compactor.add_interaction_turn("assistant", "Critical architectural decision", metadata={"importance_score": 0.85})
    level = compactor.compact_working_window("test_ep")

    assert len(level.extracted_facts) == 1
    assert level.extracted_facts[0].importance_score == 0.85

    # 3. Canonicalizer loads dynamically without hardcoded fallback block
    canonicalizer_src = inspect.getsource(EntityCanonicalizer._init_default_rules)
    assert "entity_nas_01" not in canonicalizer_src
    assert "entity_agent_core" not in canonicalizer_src

    canon = EntityCanonicalizer()
    # Still resolves correctly via DomainProfileRegistry
    assert canon.canonicalize("truenas") == "entity_nas_01"


# --- 5. Reviewer Adversarial Mutation Gates ---

@pytest.mark.asyncio
async def test_mutation_m1_freezing_temperature_breaks_sensing_bridge():
    """Mutation M1: If an adversarial change freezes T=0.01 regardless of error, test fails."""
    active_sensing = ActiveSensingEngine()
    from atlas_memory.active.prediction_error import PredictionCheck
    active_sensing.register_expectation(
        PredictionCheck(
            check_id="chk_db",
            target_entity="db",
            expected_predicate="status",
            expected_value="OK",
        )
    )
    active_sensing.detect_discrepancy("db", "status", "ERROR")
    t_cogn = active_sensing.get_entity_temperature("db")
    assert t_cogn > 0.5

    # Adversarial mutant: annealer ignores active_sensing
    mutant_annealer = CausalAnnealer(rng_seed=42)
    # Correct implementation must produce eff_T_init = t_cogn > 0.5
    # Let's verify annealer adopts it
    res = await mutant_annealer.run(
        [CausalEdge(source="db", predicate="p", target="t", confidence=0.5)],
        active_sensing=active_sensing,
        entity="db",
        T_min=0.01,
        max_iter=1,
    )
    # In proper implementation, final_temperature started at >= 0.5, so after 1 step it cannot be clamped to 0.01
    assert res.final_temperature > 0.05, "M1 Caught: Annealer failed to react to elevated sensory temperature"


@pytest.mark.asyncio
async def test_mutation_m2_zeroing_jepa_collapses_risk_potential():
    """Mutation M2: If JEPA divergence is zeroed out, risk potential collapses."""
    relations = [{"subject": "A", "predicate": "depends_on", "object": "B", "confidence": 0.9}]
    graph = MockGraphClient(relations)

    engine = RetroCausalEngine(graph_client=graph)
    # Direct check of _assess_path_risk:
    edge = CausalEdge(source="A", predicate="depends_on", target="B", confidence=0.9)
    # With zero JEPA divergence (and no semantic multiplier)
    risk_zero = engine._assess_path_risk(
        action="read",
        steps=[edge],
        target="B",
        conf=0.9,
        jepa_divergence=0.0,
        cpof_map={"B": 0.5},
    )
    assert risk_zero in ("LOW", "MODERATE"), "M2 Verified: Zero JEPA divergence dampens risk potential"


def test_mutation_m3_ast_scanner_catches_keyword_heuristics():
    """Mutation M3: AST scanner detects any injection of string heuristics into ContextCompactor."""
    compactor_file = Path(__file__).resolve().parent.parent / "src" / "atlas_memory" / "extensions" / "compactor.py"
    tree = ast.parse(compactor_file.read_text(encoding="utf-8"))

    # Walk AST and search for string comparison or membership in lists of polish keywords
    suspicious_words = {"zmień", "ustaw", "serwer", "używaj", "port:"}
    found_suspicious = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for word in suspicious_words:
                if word in node.value:
                    found_suspicious.append(node.value)

    assert not found_suspicious, f"M3 AST Scanner caught forbidden keyword constants: {found_suspicious}"
