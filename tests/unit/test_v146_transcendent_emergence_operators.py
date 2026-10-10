"""
Unit tests for Protocol v146.0: Transcendent Cognitive Synthesis & Emergence Operators.
Weryfikacja 4 Filarów Transcendentnych:
1. Connected Steiner Knapsack (K_tree): Bonus za spójność podgrafu chroni łańcuchy geodezyjne.
2. Unitarna Anty-Adaptacja w Sadze (Phi_undo): Zerowa holonomia Berry'ego po rollbacku TTT i JEPA.
3. HippoRAG 2 Kwantowe Tunelowanie: beta_tunnel = 0.10 dla aksjomatów nadrzędnych.
4. Termodynamiczna Dekrystalizacja Procedur: Topnienie SOP w ConflictArbiter przy błędach OOD z ActiveSensingEngine.
"""

from unittest.mock import MagicMock

import networkx as nx
import numpy as np
import pytest

from atlas_memory.active.prediction_error import ActiveSensingEngine, PredictionError
from atlas_memory.cognitive.conflict_arbiter import ConflictArbiter
from atlas_memory.cognitive.retrieval.models import CandidateEntity
from atlas_memory.cognitive.retrieval.ppr import HippoRAG2PPR
from atlas_memory.cognitive.retrieval.render import pack_knapsack_01
from atlas_memory.cognitive.write_path import EpistemicWritePath
from atlas_memory.core.models import ActionPlan
from atlas_memory.l0_dynamic.ttt_layer import TTTCheckpoint, TTTLayer
from atlas_memory.l1_working.jepa_latent import JEPACheckpoint, JEPALatentBuffer

# ── Filar 1: K_tree Connected Steiner Knapsack ──────────────────────────────

def test_knapsack_tree_preserves_connected_subgraph():
    """Sprawdza, czy K_tree faworyzuje spójny łańcuch inferencji nad izolowane małe fakty."""
    # Tworzymy łańcuch A -> B -> C (każdy o koszcie 30 tokenów, score 10.0 każdy = suma 30.0)
    cand_a = CandidateEntity(
        key="node_a",
        metadata={"subject": "node_a", "predicate": "relates", "object": "node_b"},
    )
    cand_b = CandidateEntity(
        key="node_b",
        metadata={"subject": "node_b", "predicate": "relates", "object": "node_c"},
    )
    cand_c = CandidateEntity(
        key="node_c",
        metadata={"subject": "node_c", "predicate": "relates", "object": "node_d"},
    )
    cand_x = CandidateEntity(
        key="node_x",
        metadata={"subject": "node_x"},
    )

    # Izolowany fakt o wysokim jednostkowym score (score 12.0, koszt 15 tokenów -> gęstość 0.8)
    # W starym plecaku 0/1 pojedynczy fakt mógłby wypchnąć element łańcucha przy ciasnym budżecie.
    # W nowym K_tree: łańcuch [A, B, C] ma 2 krawędzie -> mnożnik (1 + 0.35 * 2) = 1.70 -> efektywny score 51.0!
    items = [
        ("• Chain Node A connects to B", 10.0, 30, cand_a),
        ("• Chain Node B connects to C", 10.0, 30, cand_b),
        ("• Chain Node C connects to D", 10.0, 30, cand_c),
        ("• Isolated trivial fact X", 12.0, 15, cand_x),
    ]

    # Budżet: 95 tokenów (mieści cały łańcuch 90 tokenów LUB fakt X + 2 elementy łańcucha)
    # Przy K_tree z bonusem poddrzewa, spójny komponent [A, B, C] ma wyższą gęstość i wchodzi w całości!
    packed = pack_knapsack_01(items, budget=95)
    packed_texts = [p[0] for p in packed]

    assert "• Chain Node A connects to B" in packed_texts
    assert "• Chain Node B connects to C" in packed_texts
    assert "• Chain Node C connects to D" in packed_texts


# ── Filar 2: Unitarna Anty-Adaptacja w Sadze (Phi_undo) ─────────────────────

def test_ttt_layer_checkpoint_and_restore():
    """Weryfikuje idempotencję capture_checkpoint i restore_checkpoint w TTTLayer."""
    ttt = TTTLayer(input_dim=16, hidden_dim=8, learning_rate=0.05)
    initial_w = ttt.w_ttt.copy()
    initial_step = ttt.step_count

    cp = ttt.capture_checkpoint()
    assert isinstance(cp, TTTCheckpoint)
    assert np.array_equal(cp.w_ttt, initial_w)

    # Adaptujemy stan (mutacja wag)
    obs = np.ones(16, dtype=np.float64)
    ttt.adapt_step(obs)
    assert not np.array_equal(ttt.w_ttt, initial_w)
    assert ttt.step_count == initial_step + 1

    # Przywracamy stan po abortowaniu Sagi
    ttt.restore_checkpoint(cp)
    assert np.array_equal(ttt.w_ttt, initial_w)
    assert ttt.step_count == initial_step


def test_jepa_latent_buffer_checkpoint_and_restore():
    """Weryfikuje idempotencję capture_checkpoint i restore_checkpoint w JEPALatentBuffer."""
    jepa = JEPALatentBuffer(state_dim=8, action_dim=4)
    initial_history_len = len(jepa.history)
    initial_w_s = jepa.w_s.copy()

    cp = jepa.capture_checkpoint()
    assert isinstance(cp, JEPACheckpoint)

    # Zapisujemy krok i mutujemy stan
    act = ActionPlan(name="action_probe", parameters={"target": "disk"})
    jepa.step(act)
    assert len(jepa.history) == initial_history_len + 1

    # Rollback checkpointa
    jepa.restore_checkpoint(cp)
    assert len(jepa.history) == initial_history_len
    assert np.array_equal(jepa.w_s, initial_w_s)


def test_saga_rollback_restores_ttt_and_jepa():
    """Weryfikuje, że w przypadku błędu transakcji Sagi w EpistemicWritePath stan TTT i JEPA jest cofany."""
    kv_store = MagicMock()
    vector_store = MagicMock()
    graph_store = MagicMock()
    ttt = TTTLayer(input_dim=8, hidden_dim=4)
    jepa = JEPALatentBuffer(state_dim=8, action_dim=4)

    # Ustawiamy zachowanie set_sync
    kv_store.get_all_sync.return_value = {}
    kv_store.get_arbitration_candidates_sync.return_value = {}

    write_path = EpistemicWritePath(
        kv_store=kv_store,
        vector_store=vector_store,
        graph_store=graph_store,
        ttt_layer=ttt,
        jepa_buffer=jepa,
    )

    # Pre-adaptujemy stan
    ttt.adapt_step(np.ones(8, dtype=np.float64))
    jepa.step(ActionPlan(name="action_init", parameters={}))
    pre_tx_ttt_w = ttt.w_ttt.copy()
    pre_tx_jepa_w = jepa.w_s.copy()
    pre_tx_history_len = len(jepa.history)

    # Symulujemy błąd w grafie (np. awaria indeksu krawędzi) podczas auto-insert
    graph_store.add_relation = MagicMock(side_effect=RuntimeError("Graph storage fault"))

    # Podczas próby zapisu faktu ze strukturą krawędzi: "step_one -> step_two"
    # W trakcie transakcji stan dynamiczny zostaje zmodyfikowany (symulacja kroków wewnątrz)
    def mutate_during_tx(*args, **kwargs):
        ttt.adapt_step(np.ones(8, dtype=np.float64) * 3.0)
        jepa.step(ActionPlan(name="action_in_tx", parameters={}))

    kv_store.set_sync.side_effect = mutate_during_tx

    with pytest.raises(RuntimeError, match="Graph storage fault"):
        write_path.write("chain:alpha", "step_one -> step_two")

    # Po rollbacku Sagi stan TTT i JEPA musi zostać cofnięty do stanu sprzed transakcji!
    assert np.array_equal(ttt.w_ttt, pre_tx_ttt_w)
    assert np.array_equal(jepa.w_s, pre_tx_jepa_w)
    assert len(jepa.history) == pre_tx_history_len


# ── Filar 3: HippoRAG 2 Kwantowe Tunelowanie Asocjacyjne ───────────────────

def test_hipporag2_ppr_quantum_tunneling():
    """Weryfikuje rezerwację energii beta_tunnel dla UGRUNTOWANYCH aksjomatów regułowych.

    V151.5 (WEKTOR II): tunelowanie wymaga sim(query_stems, axiom_stems) > θ_min;
    ślepe pompowanie masy do niepowiązanych reguł jest ZABRONIONE (zero energii).
    """
    g = nx.Graph()
    g.add_edge("fact_a", "fact_b", weight=1.0)
    g.add_edge("fact_b", "fact_c", weight=1.0)
    g.add_node("rule:security_policy")

    ppr = HippoRAG2PPR(damping=0.85, max_iter=20, backend="numpy")
    # Query ugruntowany semantycznie w aksjomacie (wspólne łodygi security/policy).
    res = ppr.run(query="security policy fact", graph_store=g)

    # Z tunelowaniem ugruntowanym aksjomat MUSI otrzymać masę probabilistyczną > 0
    assert "rule:security_policy" in res.node_scores
    assert res.node_scores["rule:security_policy"] > 0.0
    # Całkowity rozkład musi sumować się do 1.0
    total_mass = sum(res.node_scores.values())
    assert pytest.approx(total_mass, abs=1e-4) == 1.0

    # V151.5 kontrakt odwrotny: query NIEpowiązany z aksjomatem → zero energii tunelowania.
    res_ungrounded = ppr.run(query="fact_a", graph_store=g)
    assert res_ungrounded.node_scores.get("rule:security_policy", 0.0) == 0.0, (
        "reguła bez ugruntowania sim(query, rule) > θ_min nie może być pompowana"
    )


# ── Filar 4: Termodynamiczna Dekrystalizacja Procedur (Topnienie SOP) ──────

def test_procedure_decrystallization_under_ood_drift():
    """Weryfikuje, że reguła L3 ze statusem błędu CRITICAL z ActiveSensingEngine ulega stopnieniu."""
    ase = ActiveSensingEngine()
    # Rejestrujemy krytyczną rozbieżność dla encji reguły deploy_sop
    ase.error_history.append(
        PredictionError(
            check_id="check_deploy_01",
            target_entity="rule:deploy_sop",
            predicate="execution_status",
            expected_value="pass",
            observed_value="fatal_crash",
            discrepancy_score=1.0,
            severity="CRITICAL",
        )
    )

    assert ase.is_entity_critical("rule:deploy_sop") is True

    # Tworzymy arbiter z podpiętym silnikiem predykcji
    arbiter = ConflictArbiter(active_sensing_engine=ase)

    # Istniejąca reguła w pamięci (utwardzona z pewnością 0.98)
    existing_items = {
        "rule:deploy_sop": {
            "value": "Always deploy without staging check",
            "confidence": 0.98,
            "metadata": {"source": "manual_rule"},
        }
    }

    # Przychodząca świeża obserwacja ze środowiska o wysokiej pewności (0.85 >= 0.70)
    result = arbiter.arbitrate(
        key="rule:deploy_sop",
        val="Deploy requires canary testing and staging check",
        confidence=0.85,
        metadata={"source": "telemetry_runtime"},
        existing_items=existing_items,
    )

    # W normalnych warunkach (bez błędu krytycznego) reguła o pewności 0.98 nie zostałaby nadpisana
    # przez incoming_conf 0.85 (inwersja pewności / ochrona reguły).
    # W reżimie termodynamicznym reguła ulega stopnieniu (is_rule_melted=True) i zostaje superseded!
    assert result.action == "supersede"
    assert result.is_belief_revision is True
    assert result.superseded_key == "rule:deploy_sop"
