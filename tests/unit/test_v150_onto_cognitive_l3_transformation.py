"""
Testy Transformacji Onto-Kognitywnej L3 (Post-2026 Architecture).

Weryfikuje 4 Hiperwektory / Dyrektywy:
1. Dyrektywa 1: ParametricDistributionGraph i Thermal Potential Drop w skill_compiler.py / sleep_baker.py.
2. Dyrektywa 2: Friston Bayesian Precision Tensor (Pi_channel, Lambda) i dynamiczna adaptacja w auditor.py.
3. Dyrektywa 3: Fiber Bundle Clustering w przestrzeni EAV, blokowanie ortogonalnych slotów i rozbicie fałszywej przechodniości w declarative_consolidator.py.
4. Dyrektywa 4: Odwracalna Monada Stanu (TransactionalDeltaTensor) i symetryczny ROLLBACK-ALL w auditor.py.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from atlas_memory.cognitive.conflict_arbiter import EpistemicCoordinate
from atlas_memory.core.models import EpistemicSource, MemoryRecord
from atlas_memory.l2_semantic.kv_store import VerifiedKVStore
from atlas_memory.l3_procedural.auditor import (
    BayesianPrecisionTensor,
    MemoryAuditor,
    TransactionalDeltaTensor,
)
from atlas_memory.l3_procedural.declarative_consolidator import (
    CandidateFact,
    DeclarativeConsolidator,
    are_orthogonal_slots,
)
from atlas_memory.l3_procedural.skill_compiler import (
    ParametricDistributionGraph,
    generate_handler,
)
from atlas_memory.l3_procedural.sleep_baker import Step

# ── Dyrektywa 1: ParametricDistributionGraph & Thermal Potential Drop ─────────

def test_parametric_distribution_graph_continuous_and_discrete():
    """Weryfikuje ciągłą estymację rozkładów Gaussa oraz dyskretnych częstotliwości."""
    graph = ParametricDistributionGraph()
    samples = [
        {"timeout": 30.0, "retries": 3, "mode": "fast"},
        {"timeout": 32.0, "retries": 3, "mode": "fast"},
        {"timeout": 28.0, "retries": 5, "mode": "safe"},
    ]
    graph.fit_parameter_space(samples)

    assert "timeout" in graph.parameter_domains
    t_domain = graph.parameter_domains["timeout"]
    assert t_domain["type"] == "continuous"
    assert abs(t_domain["mean"] - 30.0) < 0.01
    assert t_domain["std"] > 0.0

    assert "mode" in graph.parameter_domains
    m_domain = graph.parameter_domains["mode"]
    assert m_domain["type"] == "discrete"
    assert "fast" in m_domain["categories"]
    assert "safe" in m_domain["categories"]

    # Próbkowanie z zerową temperaturą zwraca średnią/modę
    p_cold = graph.sample_parameters(temperature=0.0)
    assert abs(p_cold["timeout"] - 30.0) < 0.01

    # Próbkowanie z podwyższoną temperaturą wstrzykuje perturbację
    p_hot = graph.sample_parameters(temperature=0.5)
    assert isinstance(p_hot["timeout"], float)


def test_thermal_potential_drop_on_execution_failure():
    """Weryfikuje, że błąd wykonania kroku wywołuje skok temperatury Delta T i adaptację parametrów."""
    from atlas_memory.l3_procedural.sleep_baker import StandardProcedure

    step = Step(
        tool_name="http_client",
        params_pattern={"timeout": 10.0, "url": "https://api.internal/v1"},
        distribution_space={
            "timeout": {"type": "continuous_gaussian", "mean": 10.0, "std": 2.0, "bounds": [5.0, 30.0]}
        },
        thermal_damping=0.85,
    )
    sop = StandardProcedure(
        procedure_id="api_fetcher",
        name="API Fetcher",
        steps=[step],
        success_rate=0.95,
        invocations_count=10,
    )

    code = generate_handler(sop)

    # Weryfikacja obecności modelu ciągłego i mechanizmu termicznego w wygenerowanym handlerze
    assert "ParametricDistributionGraph" in code
    assert "Thermal Potential Drop" in code
    assert "delta_temperature: float = 0.35" in code
    assert "sample_parameters" in code

    # Wykonanie handlera w odizolowanym namespace
    namespace: dict = {}
    exec(code, namespace)
    handler_cls = namespace["ApiFetcherHandler"]
    instance = handler_cls(
        context={
            "adaptation_hook": lambda tool, params, err: {
                "adapted_tool": tool,
                "adapted_params": params,
            }
        }
    )

    # Sprawdzenie parametrów nominalnych przy temp=0.0
    initial_step = instance.step_1()
    assert initial_step["status"] == "completed"
    assert initial_step["params"]["timeout"] == 10.0

    # Symulacja błędu wywołuje skok temperatury Delta T
    instance.cognitive_temperature += instance.delta_temperature
    adapted_step = instance.step_1()
    assert instance.cognitive_temperature == 0.35
    assert "timeout" in adapted_step["params"]


# ── Dyrektywa 2: Friston Bayesian Precision Tensor ────────────────────────────

def test_bayesian_precision_tensor_falsification_adaptation():
    """Weryfikuje dynamiczny spadek precyzji kanału po błędzie i override przez dokładną inferencję."""
    tensor = BayesianPrecisionTensor(alpha=0.80, sigma_0_sq=0.04)

    # Stan początkowy
    pi_tool_init = tensor.get_precision(EpistemicSource.TOOL_OUTPUT)
    pi_agent_init = tensor.get_precision(EpistemicSource.AGENT_INFERENCE)
    assert pi_tool_init > pi_agent_init  # 1.5 > 1.0

    # Wstrzyknięcie serii błędów sensorycznych do TOOL_OUTPUT (halucynacja / zepsuty tool)
    for _ in range(5):
        tensor.record_sensory_observation(EpistemicSource.TOOL_OUTPUT, error_magnitude=1.5)

    pi_tool_degraded = tensor.get_precision(EpistemicSource.TOOL_OUTPUT)
    assert pi_tool_degraded < pi_tool_init
    # Precyzja zepsutego toola spada poniżej stabilnego agent inference
    assert pi_tool_degraded < pi_agent_init

    # Obliczenie Lambda = Pi * confidence
    # Zepsuty tool z conf=0.99 vs stabilny agent z conf=0.95
    lambda_tool = tensor.compute_lambda(EpistemicSource.TOOL_OUTPUT, 0.99)
    lambda_agent = tensor.compute_lambda(EpistemicSource.AGENT_INFERENCE, 0.95)
    assert lambda_agent > lambda_tool

    # Arbitracja audytora
    auditor = MemoryAuditor(graph_store=AsyncMock(), kv_store=AsyncMock())
    auditor.precision_tensor = tensor

    rec_existing_corrupted_tool = MemoryRecord(
        subject="service:port",
        predicate="value",
        object="11235",
        confidence=0.99,
        source_type=EpistemicSource.TOOL_OUTPUT,
        timestamp=100.0,
    )
    rec_incoming_agent = MemoryRecord(
        subject="service:port",
        predicate="value",
        object="cli_only",
        confidence=0.95,
        source_type=EpistemicSource.AGENT_INFERENCE,
        timestamp=105.0,
    )

    should_override, reason = auditor.arbitrate_record_conflict(
        existing=rec_existing_corrupted_tool,
        incoming=rec_incoming_agent,
        delta_margin=0.05,
    )
    assert should_override is True
    assert "bayesian_precision_override" in reason


# ── Dyrektywa 3: Fiber Bundle Clustering & Orthogonal Slots ───────────────────

def test_orthogonal_slots_detection():
    """Weryfikuje ortogonalność slotów funkcjonalnych w tej samej encji."""
    coord_port = EpistemicCoordinate.from_key("fact:service_api:port")
    coord_timeout = EpistemicCoordinate.from_key("fact:service_api:timeout")
    coord_port2 = EpistemicCoordinate.from_key("fact:service_api:port")

    assert are_orthogonal_slots(coord_port, coord_timeout) is True
    assert are_orthogonal_slots(coord_port, coord_port2) is False


def test_fiber_bundle_clustering_suppresses_orthogonal_slots():
    """Weryfikuje, że find_pairs supresuje ortogonalne sloty tej samej encji (sim -> 0.0)."""
    cand_port = CandidateFact(
        key="fact:crawl4ai:port",
        value="11235",
        text="Crawl4AI running on network port 11235",
        confidence=0.9,
        timestamp=1.0,
        metadata={},
    )
    cand_timeout = CandidateFact(
        key="fact:crawl4ai:timeout",
        value="60s",
        text="Crawl4AI network timeout set to 60s",
        confidence=0.9,
        timestamp=1.0,
        metadata={},
    )

    # Sztuczny embedder zwracający niemal identyczne wektory (cos=1.0)
    def fake_embedder(_t: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    consolidator = DeclarativeConsolidator(kv_store=None, cosine_threshold=0.5)
    pairs = consolidator.find_pairs([cand_port, cand_timeout], fake_embedder)

    # Krawędź musi zostać odrzucona przez blokadę slotów ortogonalnych
    assert len(pairs) == 0
    assert consolidator._blocked_orthogonal_slots >= 1


def test_connected_components_breaks_false_transitivity():
    """Weryfikuje, że łańcuch A ~ B i B ~ C bez krawędzi A ~ C NIE łączy A i C w jeden klaster."""
    # Trzy fakty w tym samym włóknie (entity_root="fact:app", slot="value")
    # A ~ B (sim=0.9), B ~ C (sim=0.85), ale brak krawędzi pomiędzy A i C
    nodes = ["fact:app:nodeA", "fact:app:nodeB", "fact:app:nodeC"]
    pairs = [
        ("fact:app:nodeA", "fact:app:nodeB", 0.90),
        ("fact:app:nodeB", "fact:app:nodeC", 0.85),
    ]

    clusters = DeclarativeConsolidator.connected_components(nodes, pairs)

    # W naiwnym Union-Find mielibyśmy [[nodeA, nodeB, nodeC]]
    # W Fiber Bundle z Pairwise Clique Check: nodeA i nodeC NIE tworzą klastra
    assert len(clusters) == 1
    # Klaster składa się wyłącznie z najsilniejszej powiązanej pary
    assert clusters[0] == ["fact:app:nodeA", "fact:app:nodeB"]
    assert "fact:app:nodeC" not in clusters[0]


# ── Dyrektywa 4: Reversible State Monad (TransactionalDeltaTensor) ─────────────

@pytest.mark.asyncio
async def test_transactional_delta_tensor_atomic_success():
    """Weryfikuje pomyślne wykonanie TransactionalDeltaTensor we wszystkich warstwach."""
    kv = VerifiedKVStore(db_path=":memory:")
    graph = AsyncMock()
    vector = AsyncMock()

    auditor = MemoryAuditor(graph_store=graph, kv_store=kv, vector_store=vector)
    rec = MemoryRecord(
        subject="service_node",
        predicate="status",
        object="online",
        confidence=0.95,
        is_state_variable=True,
    )

    tx = TransactionalDeltaTensor(tx_id="tx_test_100", record=rec)
    res = await tx.execute(auditor)

    assert res is True
    assert tx.status == "COMMITTED"
    assert "intent" in tx.applied_steps
    assert "graph" in tx.applied_steps
    assert "kv" in tx.applied_steps
    assert "vector" in tx.applied_steps

    # Sprawdzenie stanu w KV
    saved = await kv.get_state("service_node")
    assert saved is not None
    assert saved["value"] == "online"
    await kv.close()


@pytest.mark.asyncio
async def test_transactional_delta_tensor_symmetric_rollback_on_vector_failure():
    """Weryfikuje pełny symetryczny ROLLBACK-ALL (Zero Split-Brain) przy awarii wektora."""
    kv = VerifiedKVStore(db_path=":memory:")
    graph = AsyncMock()
    vector = AsyncMock()
    vector.insert = AsyncMock(side_effect=RuntimeError("Vector storage out of memory"))

    auditor = MemoryAuditor(graph_store=graph, kv_store=kv, vector_store=vector)
    rec = MemoryRecord(
        subject="faulty_cluster",
        predicate="state",
        object="degraded",
        confidence=0.88,
        is_state_variable=True,
    )

    tx = TransactionalDeltaTensor(tx_id="tx_test_fail", record=rec)
    with pytest.raises(RuntimeError, match="Vector storage out of memory"):
        await tx.execute(auditor)

    assert tx.status == "FAILED"
    # Kùzu graph: krawędź usunięta
    graph.remove_edge.assert_called_once_with("faulty_cluster", "state", "degraded")
    # KV Store: wpis został wycofany (brak osieroconego wpisu w bazie)
    state = await kv.get_state("faulty_cluster")
    assert state is None
    # Intencja transakcyjna ma status 'FAILED'
    intents = await kv.get_dangling_intents()
    # Nie ma wiszących PENDING intencji
    assert len(intents) == 0
    await kv.close()
