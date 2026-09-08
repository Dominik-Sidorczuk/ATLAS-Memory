from __future__ import annotations

from atlas_memory.l1_working.jepa_latent import JEPALatentBuffer
from atlas_memory.models import ActionPlan


def test_jepa_multi_horizon_counterfactual_rollout():
    buffer = JEPALatentBuffer(state_dim=32, action_dim=16)

    # 3 candidate plans of length 5
    plan_a = [ActionPlan(name=f"explore_node_{i}", parameters={"depth": i}) for i in range(5)]
    plan_b = [ActionPlan(name=f"query_db_{i}", parameters={"limit": 10}) for i in range(5)]
    plan_c = [ActionPlan(name=f"cache_write_{i}", parameters={"ttl": 60}) for i in range(5)]

    best_plan, trajectory, best_score = buffer.select_best_action_trajectory([plan_a, plan_b, plan_c])

    assert len(best_plan) == 5
    assert len(trajectory) == 5
    assert isinstance(best_score, float)

    # Check uncertainty bounds along rollout
    for step in trajectory:
        assert 0.0 <= step.uncertainty <= 1.0
        assert len(step.predicted_state.vector) == 32


def test_jepa_empty_and_single_action_edge_cases():
    buffer = JEPALatentBuffer(state_dim=16, action_dim=8)

    # Single action rollout
    single_plan = [ActionPlan(name="ping_health", parameters={})]
    best_plan, traj, score = buffer.select_best_action_trajectory([single_plan])

    assert len(best_plan) == 1
    assert len(traj) == 1
    assert traj[0].predicted_state.step_index == 1


def test_jepa_state_commitment_and_trajectory_history():
    buffer = JEPALatentBuffer(state_dim=16, action_dim=8)
    initial_step = buffer.current_state.step_index

    # Commit 3 transitions
    for i in range(3):
        act = ActionPlan(name=f"step_act_{i}", parameters={})
        next_s = buffer.commit_state_transition(act)
        assert next_s.step_index == initial_step + i + 1

    assert buffer.current_state.step_index == initial_step + 3
    assert len(buffer.history) >= 4
