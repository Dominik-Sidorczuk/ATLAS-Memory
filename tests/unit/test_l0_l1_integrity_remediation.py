"""Weryfikacja Integralności Warstw L0 Dynamic i L1 Working Memory (Zadania 1-5).

1. Zadanie 1: test_jepa_closed_loop_adaptation — monotoniczny spadek błędu w pętli uczenia świata.
2. Zadanie 2: test_cross_session_isolation — reset TTT i JEPA przy zmianie sesji.
3. Zadanie 3: test_trajectory_buffer_true_zero_copy — rzeczywisty zero-copy i float32 end-to-end.
4. Zadanie 4: test_shared_hot_buffer_concurrent_consolidation — wielosesyjne śledzenie l2_consolidated_sessions.
5. Zadanie 5: test_ttt_jepa_pipeline_integration — fuzja projekcji L0 TTT do wektora akcji L1 JEPA.
"""
from __future__ import annotations

import numpy as np
import pytest

from atlas_memory.core.models import ActionPlan, LatentState, PredictedTransition
from atlas_memory.engine import HybridMemoryEngine
from atlas_memory.l1_working.jepa_latent import JEPALatentBuffer
from atlas_memory.l1_working.session_tracker import HermesSessionHook
from atlas_memory.l1_working.trajectory_buffer import ArrowTrajectoryBuffer


def test_jepa_closed_loop_adaptation():
    """Zadanie 1: Weryfikacja domknięcia pętli uczenia modelu świata JEPA.

    Seria kroków z deterministycznym, powtarzalnym przejściem stanu s_t -> s_{t+1}
    musi skutkować spadkiem błędu rekonstrukcji MSE w czasie (prediction error).
    """
    buf = JEPALatentBuffer(state_dim=16, action_dim=8, seed=123)
    curr = buf.current_state
    action = ActionPlan(name="action_probe", parameters={"val": 1})
    # Środowisko z deterministyczną dynamiką: target = 0.5 * sin(curr) + 0.2
    target_vec = np.sin(np.asarray(curr.vector, dtype=np.float64)) * 0.5 + 0.2

    errors = []
    # Wykonaj 15 kroków adaptacji na tej samej dynamice
    for _ in range(15):
        loss = buf.adapt_transition(buf.current_state, action, target_vec, lr=0.08, weight_decay=0.0001)
        errors.append(loss)

    assert len(errors) == 15
    # Błąd w późniejszych krokach musi być niższy niż błąd początkowy
    assert errors[-1] < errors[0], f"Loss nie spadł: start={errors[0]}, end={errors[-1]}"
    # Stan bufora został zaktualizowany do zaobserwowanego
    assert buf.current_state.step_index == curr.step_index + 15
    assert len(buf.history) == 16


@pytest.mark.asyncio
async def test_cross_session_isolation():
    """Zadanie 2: Izolacja sesji i reset stanu epizodycznego (L0 / L1).

    Po zakończeniu sesji A i rozpoczęciu sesji B:
    - w_ttt w warstwie TTT jest wyzerowane (w_ttt.sum() == 0.0)
    - historia stanów JEPA jest zresetowana do 1 stanu początkowego (len(history) == 1)
    - stan ukryty nie zawiera danych ani kroków z sesji A.
    """
    engine = HybridMemoryEngine.create_default(db_path=":memory:", qdrant_location=":memory:")
    hook = HermesSessionHook(engine)

    # 1. Sesja A: wykonaj operacje i adaptację
    await hook.on_session_start("session_A")
    action_a = ActionPlan(name="write_doc", parameters={"doc": "secret_A"})
    engine.record_mental_transition(action_a, session_id="session_A")
    engine.record_mental_transition(action_a, session_id="session_A")

    # Weryfikacja, że sesja A zakumulowała stan
    assert engine.ttt.w_ttt.sum() != 0.0 or engine.ttt.step_count > 0
    assert len(engine.latent.history) > 1

    # 2. Zakończenie sesji A
    await hook.on_session_end("session_A", trigger_sleep_cycle=False)

    # 3. Rozpoczęcie sesji B
    await hook.on_session_start("session_B")

    # Asercje izolacji sesji:
    assert float(np.sum(np.abs(engine.ttt.w_ttt))) == 0.0, "Wagi w_ttt muszą być wyzerowane w nowej sesji"
    assert len(engine.latent.history) == 1, "Historia JEPA musi zawierać tylko stan początkowy"
    assert engine.latent.current_state.step_index == 0, "Indeks kroku JEPA musi być zresetowany do 0"


def test_trajectory_buffer_true_zero_copy(tmp_path):
    """Zadanie 3: Harmonizacja precyzji i rzeczywisty Zero-Copy (Float32 End-to-End).

    Weryfikuje, że:
    1. to_zero_copy_tensor() zwraca widok dzielący pamięć z _latent_array bez alokacji.
    2. to_arrow_table() buduje schemat float32 bez upcastingu do float64.
    3. Round-trip Parquet zachowuje format float32.
    """
    buf = ArrowTrajectoryBuffer(state_dim=16, capacity=100)
    for i in range(10):
        prev_st = LatentState(step_index=i, dimension=16, timestamp=float(i), vector=[float(j) for j in range(16)])
        pred_st = LatentState(step_index=i + 1, dimension=16, timestamp=float(i + 1), vector=[float(j + 1) for j in range(16)])
        buf.append_transition(
            PredictedTransition(
                previous_state=prev_st,
                predicted_state=pred_st,
                action=ActionPlan(name=f"act_{i}", parameters={}),
                simulated_reward=1.0,
                uncertainty=0.0,
            ),
            session_id="zero_copy_test",
        )

    # 1. Zero-Copy RAM check
    tensor = buf.to_zero_copy_tensor()
    assert tensor.dtype == np.float32
    assert np.shares_memory(tensor, buf._latent_array) is True, "Tensor musi dzielić pamięć z _latent_array (Zero-Copy)"

    # 2. Arrow schema check
    table = buf.to_arrow_table()
    assert table is not None
    field = table.schema.field("latent_vector")
    import pyarrow as pa
    assert field.type == pa.list_(pa.float32()), f"Oczekiwano list_(float32), otrzymano: {field.type}"

    # 3. Parquet round-trip check
    pq_path = tmp_path / "traj_f32.parquet"
    ok = buf.dump_to_parquet(pq_path)
    assert ok is True
    loaded = ArrowTrajectoryBuffer.load_from_parquet(pq_path, state_dim=16)
    assert loaded.to_zero_copy_tensor().dtype == np.float32


@pytest.mark.asyncio
async def test_shared_hot_buffer_concurrent_consolidation():
    """Zadanie 4: Bezpieczna konsolidacja rekordów współdzielonych w HotBuffer.

    Wpis globalny nie może zostać zablokowany dla sesji B po zakończeniu i konsolidacji sesji A.
    """
    engine = HybridMemoryEngine.create_default(db_path=":memory:", qdrant_location=":memory:")
    hook = HermesSessionHook(engine)

    # Dodaj współdzielony wpis globalny do HotBuffer
    entry = engine.hot_buffer.update(
        key="config:shared_llm_model",
        value="claude-3.7-sonnet",
        metadata={"session_id": "global"},
    )
    assert entry.key == "config:shared_llm_model"

    # Sesja A pobiera i konsoliduje
    entries_a = engine.hot_buffer.get_session_entries("session_A")
    assert any(e.key == "config:shared_llm_model" for e in entries_a)

    await hook.on_session_start("session_A")
    engine.write_path.session_touched_keys.setdefault("session_A", set()).add("config:shared_llm_model")
    await hook.on_session_end("session_A", trigger_sleep_cycle=True)

    # Weryfikacja: sesja A dodała się do l2_consolidated_sessions
    assert "session_A" in entry.metadata.get("l2_consolidated_sessions", set())

    # Sesja B startuje: wpis globalny nadal musi być widoczny dla sesji B!
    entries_b = engine.hot_buffer.get_session_entries("session_B")
    assert any(e.key == "config:shared_llm_model" for e in entries_b), (
        "Wpis współdzielony musi być dostępny dla sesji B mimo konsolidacji przez sesję A!"
    )


def test_ttt_jepa_pipeline_integration():
    """Zadanie 5: Funkcjonalna Fuzja Reprezentacji L0 TTT -> L1 JEPA.

    Weryfikuje, że w engine.record_mental_transition wyjście projekcji TTT (h_action)
    jest przekazywane do JEPA, a adaptacja wag TTT bezpośrednio wpływa na predykowany stan.
    """
    engine = HybridMemoryEngine.create_default(db_path=":memory:", qdrant_location=":memory:")
    action = ActionPlan(name="benchmark_action", parameters={"key": "test_pipeline"})

    # Krok 1: Wykonaj przejście bazowe
    t1 = engine.record_mental_transition(action, session_id="test_fuse")
    vec1 = list(t1.predicted_state.vector)
    assert t1.action.action_encoding == "ttt_latent"

    # Wprowadź sztuczną zmianę wag w TTT (symulacja silnej adaptacji w locie)
    engine.ttt.w_ttt += 0.5

    # Krok 2: Wykonaj to samo przejście — stan wygenerowany przez JEPA musi ulec zmianie
    t2 = engine.record_mental_transition(action, session_id="test_fuse")
    vec2 = list(t2.predicted_state.vector)

    assert vec1 != vec2, "Adaptacja wag w TTT musi bezpośrednio wpływać na predykowany stan w JEPA!"
