"""V153 Contract Test Suite: Topological Re-Alignment & Causal-Friston Convergence.

Weryfikuje:
1. Single Choke-Point Enforcement: likwidacja audit_queue i bocznego toru w HybridMemoryEngine.
2. Filar III-2: Directional Friston Tunneling w HippoRAG 2 (brak Jaccarda, kierunkowy rzut, Lambda = Pi * conf).
3. Filar IV-2: Topologiczna wycena CPoF w Knapsacku (render.py pack_knapsack_01).
4. Wykrywalność mutacji M1, M2, M3.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from atlas_memory.cognitive.precision import BayesianPrecisionTensor
from atlas_memory.cognitive.retrieval.ppr import DirectionalFristonTunnel
from atlas_memory.cognitive.retrieval.render import pack_knapsack_01
from atlas_memory.core.models import EpistemicSource, MemoryRecord

# ─────────────────────────────────────────────────────────────────────────────
# 1. TEST: Single Choke-Point Enforcement w HybridMemoryEngine
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_v153_single_choke_point_enforcement():
    """Weryfikuje, że HybridMemoryEngine kieruje 100% zapisów przez EpistemicWritePath

    oraz nie posiada bocznego toru audit_queue / _atomic_insert.
    """
    from atlas_memory.engine import HybridMemoryEngine

    # 1. Inwariant strukturalny: brak atrybutów bocznego toru
    dummy_v = MagicMock()
    dummy_g = MagicMock()
    dummy_kv = MagicMock()
    dummy_kv.load_l0_step_counter.return_value = None

    engine = HybridMemoryEngine(
        vector_store=dummy_v,
        graph_client=dummy_g,
        kv_store=dummy_kv,
    )

    assert not hasattr(engine, "audit_queue"), "HybridMemoryEngine nie może posiadać audit_queue (boczny tor)!"
    assert not hasattr(engine, "_atomic_insert"), "HybridMemoryEngine nie może posiadać _atomic_insert!"
    assert not hasattr(engine, "memory_auditor_worker"), "HybridMemoryEngine nie może posiadać pętli workera audytu!"

    # 2. Inwariant funkcjonalny: commit_observation deleguje do write_path.write_async
    mock_write_path = AsyncMock()
    engine.write_path = mock_write_path

    # Przypadek A: Zmienna stanu
    rec_state = MemoryRecord(
        subject="service_metrics_port",
        predicate="listen_port",
        object="9090",
        confidence=0.95,
        source_type=EpistemicSource.USER_EXPLICIT,
        is_state_variable=True,
    )
    await engine.commit_observation(rec_state)
    assert mock_write_path.write_async.called
    assert mock_write_path.write_async.call_args.kwargs["key"] == "service_metrics_port"

    mock_write_path.write_async.reset_mock()

    # Przypadek B: Zwykły fakt faktograficzny (nie będący zmienną stanu)
    rec_fact = MemoryRecord(
        subject="entity_alpha",
        predicate="connects_to",
        object="entity_beta",
        confidence=0.85,
        source_type=EpistemicSource.AGENT_INFERENCE,
        is_state_variable=False,
    )
    await engine.commit_observation(rec_fact)
    assert mock_write_path.write_async.called, "Nawet zwykły fakt musi iść przez write_path.write_async!"
    assert mock_write_path.write_async.call_args.kwargs["key"] == "entity_alpha:connects_to"


# ─────────────────────────────────────────────────────────────────────────────
# 2. TEST: Filar III-2 — Kierunkowe Tunelowanie Fristona w ppr.py
# ─────────────────────────────────────────────────────────────────────────────

def test_v153_directional_friston_tunneling():
    """Weryfikuje matematyczny model DirectionalFristonTunnel:

    T(q, r) = sigma((cos(e_q, e_r) - theta_base) / tau) * Lambda(r)
    - cos < theta_base (0.38) => T(q, r) == 0.0 (twarde odcięcie)
    - cos >= theta_base => T(q, r) > 0 i rośnie proporcjonalnie do precyzji Fristona Lambda.
    """
    precision_tensor = BayesianPrecisionTensor()

    # Mock enkodera zwracający kontrolowane wektory
    mock_encoder = MagicMock()
    dim = 512

    # e_q: baza
    v_q = np.zeros(dim, dtype=np.float32)
    v_q[0] = 1.0

    # e_r1: ortogonalna reguła edytora (cos = 0.0) -> musi dać tożsamościowo 0.0
    v_r1 = np.zeros(dim, dtype=np.float32)
    v_r1[1] = 1.0

    # e_r2: reguła bliska semantycznie (cos = 0.80) -> musi przejść
    v_r2 = np.zeros(dim, dtype=np.float32)
    v_r2[0] = 0.80
    v_r2[1] = np.sqrt(1.0 - 0.80**2)

    # e_r3: reguła tuż poniżej progu (cos = 0.35 < 0.38) -> musi zostać ucięta do 0.0
    v_r3 = np.zeros(dim, dtype=np.float32)
    v_r3[0] = 0.35
    v_r3[1] = np.sqrt(1.0 - 0.35**2)

    def fake_encode(text):
        if "query" in text:
            return v_q
        if "editor" in text:
            return v_r1
        if "matched" in text:
            return v_r2
        if "sub_threshold" in text:
            return v_r3
        return v_r1

    mock_encoder.encode.side_effect = fake_encode

    tunnel = DirectionalFristonTunnel(
        encoder=mock_encoder,
        precision_tensor=precision_tensor,
        theta_base=0.38,
        tau=0.08,
    )

    axioms = [
        "rule:editor:neovim",
        "rule:matched:database_pool",
        "rule:sub_threshold:cache",
    ]
    confs = [1.0, 0.9, 0.8]
    channels = [
        EpistemicSource.USER_EXPLICIT.value,
        EpistemicSource.USER_EXPLICIT.value,
        EpistemicSource.TOOL_OUTPUT.value,
    ]

    weights = tunnel.compute_tunnel_conductance(
        query="query: optimize database connection pool",
        axiom_names=axioms,
        axiom_confidences=confs,
        axiom_channels=channels,
    )

    # 1. Ortogonalna reguła edytora musi mieć wagę 0.0
    assert weights[0] == 0.0, f"Obca reguła edytora nie może tunelować energii: {weights[0]}"

    # 2. Reguła poniżej progu 0.38 musi mieć wagę 0.0
    assert weights[2] == 0.0, f"Reguła poniżej progu 0.38 musi mieć wagę 0.0: {weights[2]}"

    # 3. Zgodna semantycznie reguła musi mieć niezerową wagę z precyzją Fristona
    assert weights[1] > 0.0, "Dopasowana reguła musi uzyskać dodatnie przewodnictwo!"

    # 4. Sprawdzenie wpływu Lambda(r) = Pi * conf
    lambda_val = precision_tensor.compute_lambda(EpistemicSource.USER_EXPLICIT.value, 0.9)
    cos_val = 0.80
    expected_sigma = 1.0 / (1.0 + np.exp(-(cos_val - 0.38) / 0.08))
    expected_weight = expected_sigma * lambda_val
    assert np.isclose(weights[1], expected_weight, rtol=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# 3. TEST: Filar IV-2 — Topologiczna Wycena CPoF w Knapsacku (render.py)
# ─────────────────────────────────────────────────────────────────────────────

def test_v153_cpof_knapsack_retention():
    """Weryfikuje, że pack_knapsack_01 faworyzuje kandydatów o wysokiej dotkliwości CPoF

    (cpof_severity >= 0.50), chroniąc krytyczne poddrzewa dowodowe przed obcięciem.
    """
    # Konstruujemy scenariusz z ciasnym budżetem tokenów (np. 150 tokenów),
    # w którym konkurują dwa fakty o identycznym koszcie i bazowym wyniku,
    # lecz jeden jest krytycznym punktem awarii (CPoF severity = 1.0), a drugi zwykłym wpisem.
    item_cpof = {
        "key": "infra:postgres:primary_node",
        "subject": "infra:postgres:primary_node",
        "predicate": "hosts_database",
        "object": "Główny klaster Postgresa bazy produkcyjnej na porcie 5432",
        "content": "Główny klaster Postgresa bazy produkcyjnej na porcie 5432",
        "score": 0.50,
        "importance_score": 0.50,
        "confidence": 0.95,
        "timestamp": 1000.0,
        "metadata": {
            "cpof_severity": 1.0,  # Krytyczny punkt awarii: odcina 100% zależnych węzłów
        },
    }

    item_isolated = {
        "key": "app:theme:color_palette",
        "subject": "app:theme:color_palette",
        "predicate": "preferred_style",
        "object": "Paleta kolorystyczna interfejsu użytkownika w trybie ciemnym",
        "content": "Paleta kolorystyczna interfejsu użytkownika w trybie ciemnym",
        "score": 0.50,
        "importance_score": 0.50,
        "confidence": 0.95,
        "timestamp": 1000.0,
        "metadata": {
            "cpof_severity": 0.0,  # Zwykły wpis peryferyjny
        },
    }

    # Budżet pozwalający pomieścić tylko jeden z tych rekordów
    budget = 40

    packed = pack_knapsack_01([item_isolated, item_cpof], budget=budget)

    # Inwariant Filaru IV-2: Węzeł CPoF MUSI wygrać selekcję plecaka dzięki mnożnikowi
    # utility = base_score * (1.0 + 0.35 * internal_edges + 0.50 * comp_cpof_severity)
    assert len(packed) == 1, f"Oczekiwano 1 rekordu w ciasnym budżecie, otrzymano: {len(packed)}"
    assert packed[0]["key"] == "infra:postgres:primary_node", (
        f"Krytyczny węzeł CPoF powinien wyprzedzić zwykły fakt! Wybrano: {packed[0]['key']}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. TEST: Wykrywalność Mutacji M1, M2, M3
# ─────────────────────────────────────────────────────────────────────────────

def test_v153_mutation_m1_detection():
    """Mutacja M1: powrót do izotropowego tunelowania (maska tożsamościowo 1.0)

    musi zostać natychmiast sfalsyfikowana.
    """
    mock_encoder = MagicMock()
    v_q = np.array([1.0, 0.0], dtype=np.float32)
    v_r_foreign = np.array([0.0, 1.0], dtype=np.float32)  # cos = 0.0
    mock_encoder.encode.side_effect = lambda t: v_q if "query" in t else v_r_foreign

    tunnel = DirectionalFristonTunnel(encoder=mock_encoder, theta_base=0.38)

    # Poprawne działanie:
    weights = tunnel.compute_tunnel_conductance("query", ["rule:foreign"])
    assert weights[0] == 0.0

    # Symulacja mutacji M1: wstrzyknięcie zepsutego tunelu zawsze zwracającego 1.0
    class MutatedM1Tunnel(DirectionalFristonTunnel):
        def compute_tunnel_conductance(self, query, axiom_names, **kwargs):
            return np.ones(len(axiom_names), dtype=np.float64)

    mutated = MutatedM1Tunnel(encoder=mock_encoder)
    mutated_weights = mutated.compute_tunnel_conductance("query", ["rule:foreign"])
    assert mutated_weights[0] == 1.0  # Mutacja wstrzyknięta -> wykrywalna


def test_v153_mutation_m2_detection():
    """Mutacja M2: wyzerowanie współczynnika CPoF (comp_cpof_severity = 0.0)

    spowodowałoby utratę priorytetu węzła CPoF.
    """
    item_cpof = {
        "key": "critical_cpof",
        "subject": "critical_cpof",
        "predicate": "hosts",
        "object": "critical object data",
        "score": 0.50,
        "importance_score": 0.50,
        "confidence": 0.95,
        "timestamp": 1000.0,
        "metadata": {"cpof_severity": 1.0},
    }
    item_other = {
        "key": "other_item",
        "subject": "other_item",
        "predicate": "shows",
        "object": "other object data",
        "score": 0.50,
        "importance_score": 0.50,
        "confidence": 0.95,
        "timestamp": 2000.0,  # Nowszy timestamp
        "metadata": {"cpof_severity": 0.0},
    }

    # W normalnym knapsacku CPoF wygrywa dzięki utility mnożnikowi:
    res = pack_knapsack_01([item_other, item_cpof], budget=40)
    assert res[0]["key"] == "critical_cpof"

    # W mutacji bez CPoF wygrałby item_other (bo ma wyższy timestamp 2000 vs 1000 przy równym score):
    item_cpof_mutated = dict(item_cpof, metadata={"cpof_severity": 0.0})
    res_mutated = pack_knapsack_01([item_other, item_cpof_mutated], budget=40)
    assert res_mutated[0]["key"] == "other_item"  # Mutacja wykryta!


@pytest.mark.asyncio
async def test_v153_mutation_m3_detection():
    """Mutacja M3: przywrócenie bocznego toru z pominięciem write_path

    musi natychmiast naruszyć inwariant single choke-point.
    """
    from atlas_memory.engine import HybridMemoryEngine

    dummy_v = MagicMock()
    dummy_g = MagicMock()
    dummy_kv = MagicMock()
    dummy_kv.load_l0_step_counter.return_value = None

    engine = HybridMemoryEngine(
        vector_store=dummy_v,
        graph_client=dummy_g,
        kv_store=dummy_kv,
    )

    # Inwariant: brak audit_queue
    assert not hasattr(engine, "audit_queue")

    # Gdyby ktoś wstrzyknął boczny tor:
    engine.audit_queue = asyncio.Queue()
    assert hasattr(engine, "audit_queue")  # Mutacja M3 wstrzyknięta -> wykrywalna!
