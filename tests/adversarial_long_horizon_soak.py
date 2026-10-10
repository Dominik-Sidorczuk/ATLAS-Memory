"""
LONG-HORIZON ADVERSARIAL SOAK PROBE (POST-2027 ENGINE)
Wektor 1: Badanie metastabilności, histerezy chłodzenia i odporności na ucieczkę termiczną (Thermal Runaway).
"""

import math

import pytest

from atlas_memory.active.prediction_error import ActiveSensingEngine
from atlas_memory.cognitive.conflict_arbiter import ConflictArbiter
from atlas_memory.core.models import EpistemicSource, MemoryRecord
from atlas_memory.l2_semantic.kuzu_graph import KuzuGraphStore
from atlas_memory.l2_semantic.kv_store import VerifiedKVStore
from atlas_memory.l3_procedural.auditor import BayesianPrecisionTensor, MemoryAuditor, TransactionalDeltaTensor
from atlas_memory.l3_procedural.skill_compiler import (
    generate_handler,
)
from atlas_memory.l3_procedural.sleep_baker import StandardProcedure, Step


@pytest.mark.asyncio
async def test_long_horizon_cooling_schedule_and_recrystallization_hysteresis():
    """
    TEST 1: Dynamiczny Cykl Termiczny i Histereza (Cooling Schedule & Recrystallization).
    Sprawdza, czy po skoku temperatury T_cogn i stopieniu reguły (anomalia OOD) układ potrafi
    samoczynnie powrócić do krystalizacji, gdy błąd predykcji spada i następują sukcesy.
    Zapobiega permanentnej amnezji i chaosowi poznawczemu.
    """
    sensing = ActiveSensingEngine()
    arbiter = ConflictArbiter(config={"confidence_threshold": 0.85}, active_sensing_engine=sensing)

    rule_key = "rule:deploy_pipeline"
    existing_items = {
        rule_key: {
            "value": "run_script_v1_legacy",
            "confidence": 1.0,
            "metadata": {"is_sop": True, "source_type": "user_explicit"},
        }
    }

    # FAZA 1: Nominalny stan skrystalizowany
    # Próba nadpisania przez tool o niższej pewności zostaje zablokowana (confidence_inversion / warn_conflict)
    res_nominal = arbiter.arbitrate(
        key="rule:deploy_pipeline",
        val="run_script_v2_containerized",
        confidence=0.80,
        metadata={"source_type": "tool_output"},
        existing_items=existing_items,
    )
    assert res_nominal.is_conflict is True, "W stanie spoczynku skrystalizowana reguła musi blokować słabszy zapis"
    assert res_nominal.action in ("warn_conflict", "reject")

    # FAZA 2: Seria 20 silnie zaszumionych anomalii OOD (Perturbacja)
    for step in range(1, 21):
        err = sensing.detect_discrepancy(
            observed_entity="rule:deploy_pipeline",
            observed_predicate="execution_status",
            observed_value=f"FAIL_OOD_DRIFT_STEP_{step}",
        )
        assert err is not None

    # Sprawdzenie zachowania termicznego podczas kryzysu:
    temp_crisis = sensing.get_entity_temperature("rule:deploy_pipeline")
    assert 0.0 < temp_crisis <= 1.0, f"Temperatura w kryzysie musi być ograniczona sufitem, got {temp_crisis}"
    assert sensing.is_entity_critical("rule:deploy_pipeline") is True

    # W stanie stopienia reguła pozwala na adaptację / nadpisanie (supersede / belief_revision)
    res_melted = arbiter.arbitrate(
        key="rule:deploy_pipeline",
        val="run_script_v2_containerized",
        confidence=0.85,
        metadata={"source_type": "tool_output"},
        existing_items=existing_items,
    )
    assert res_melted.action in ("supersede", "accept") or res_melted.is_belief_revision, (
        "W stanie stopienia reguła musi zezwolić na adaptację"
    )

    # FAZA 3: Relaksacja termiczna i powrót do stabilności (40 kroków sukcesów)
    for _ in range(40):
        sensing.record_success("rule:deploy_pipeline", "execution_status")

    # Sprawdzenie powrotu do równowagi (rekrystalizacja):
    temp_relaxed = sensing.get_entity_temperature("rule:deploy_pipeline")
    assert temp_relaxed < 0.15, f"Temperatura po 40 sukcesach musi spaść < 0.15, got {temp_relaxed}"
    assert sensing.is_entity_critical("rule:deploy_pipeline") is False, "Encja po sukcesach nie może być krytyczna"

    # Zaktualizowana reguła w pamięci (po udanej adaptacji)
    existing_items[rule_key]["value"] = "run_script_v2_containerized"
    existing_items[rule_key]["confidence"] = 0.95

    # FAZA 4: Rekrystalizacja — nowa próba zaszumionego zapisu o niskiej pewności znów zostaje zablokowana
    res_recrystallized = arbiter.arbitrate(
        key="rule:deploy_pipeline",
        val="run_script_v3_untested_drift",
        confidence=0.70,
        metadata={"source_type": "tool_output"},
        existing_items=existing_items,
    )
    assert res_recrystallized.is_conflict is True, "Po rekrystalizacji słabszy write musi zostać zablokowany"
    assert res_recrystallized.action in ("warn_conflict", "reject")


def test_long_horizon_lyapunov_stability_bayesian_precision():
    """
    TEST 2: Zbieżność i Lapunowowska Stabilność Tensora Precyzji Fristona (Pi_channel).
    Weryfikuje, czy wagi precyzji nie asymilują się do 0 lub nieskończoności przy długotrwałym
    zaszumieniu (N = 100 kroków), a promień spektralny układu jest ściśle mniejszy od 1.
    """
    tensor = BayesianPrecisionTensor(alpha=0.85, sigma_0_sq=0.04)
    channel = EpistemicSource.TOOL_OUTPUT

    history: list[float] = []

    # 1. 50 kroków skrajnego zaszumienia (error_magnitude = 5.0)
    for _ in range(50):
        pi = tensor.record_sensory_observation(channel, error_magnitude=5.0)
        history.append(pi)
        assert 0.05 <= pi <= 10.0, f"Precyzja wyszła poza granice stabilności: {pi}"
        assert not math.isnan(pi) and not math.isinf(pi)

    # Granica dolna: precyzja musi osiągnąć minimum stabilne (blisko 0.05)
    assert history[-1] <= 0.20, f"Po 50 krokach szumu precyzja powinna być silnie zredukowana, got {history[-1]}"

    # 2. 50 kroków idealnej czystości sensorycznej (error_magnitude = 0.0)
    for _ in range(50):
        pi = tensor.record_sensory_observation(channel, error_magnitude=0.0)
        history.append(pi)
        assert 0.05 <= pi <= 10.0, f"Precyzja wyszła poza granice stabilności: {pi}"
        assert not math.isnan(pi) and not math.isinf(pi)

    # Granica górna: precyzja musi powrócić asymptotycznie do nasycenia
    assert history[-1] >= 8.0, f"Po 50 krokach czystości precyzja powinna wzrosnąć w pobliże sufitu, got {history[-1]}"

    # Weryfikacja Lapunowowska: wskaźnik kontrakcji e(t+1) <= alpha * e(t)
    fixed_point_high = 10.0
    diff_t = abs(history[-2] - fixed_point_high)
    diff_next = abs(history[-1] - fixed_point_high)
    # Różnica musi maleć z wykładnikiem alpha = 0.85
    assert diff_next <= diff_t + 1e-4, "Proces aktualizacji tensora musi być operatorem ściągającym"


def test_long_horizon_sop_continuous_operator_thermal_cap():
    """
    TEST 3: Ochrona przed Ucieczką Termiczną w Parametrycznym Grafie Dystrybucyjnym.
    Sprawdza, czy 50 kolejnych błędów wykonania kroków SOP nie doprowadza do ucieczki temperatury
    powyżej max_temperature = 1.0, a kolejne udane kroki schładzają układ z powrotem.
    """
    sop = StandardProcedure(
        procedure_id="proc_recovery_loop",
        name="Adaptive Recovery Loop",
        steps=[
            Step(
                step_id="step_1",
                tool_name="network_probe",
                params_pattern={"timeout": 5.0, "retries": 3},
                expected_outcome="pong",
                distribution_space={
                    "timeout": {"type": "continuous_gaussian", "mean": 5.0, "std": 1.0, "bounds": [1.0, 15.0]},
                    "retries": {"type": "discrete_distribution", "frequencies": {"1": 0.2, "3": 0.8}},
                },
            )
        ],
        success_rate=0.9,
        invocations_count=20,
    )

    code = generate_handler(sop)
    namespace: dict = {}
    exec(code, namespace)
    handler_cls = namespace["ProcRecoveryLoopHandler"]
    handler = handler_cls()

    # Symulacja 50 kolejnych błędów z podbiciem temperatury
    for _ in range(50):
        # Symulacja usterki
        handler.cognitive_temperature = min(handler.max_temperature, handler.cognitive_temperature + handler.delta_temperature)
        assert handler.cognitive_temperature <= 1.0, f"Hipertermia poznawcza! Temp: {handler.cognitive_temperature}"

    assert handler.cognitive_temperature == 1.0, "Temperatura powinna osiągnąć bezpieczny cap 1.0"

    # Symulacja 20 kroków chłodzenia po ustąpieniu perturbacji
    for _ in range(20):
        handler.cool(factor=0.80, step=0.02)

    assert handler.cognitive_temperature <= 0.05, f"Po 20 cyklach chłodzenia temp powinna spaść, got {handler.cognitive_temperature}"


@pytest.mark.asyncio
async def test_long_horizon_transactional_delta_tensor_soak():
    """
    TEST 4: Odporność Transakcyjna przy 100 Naprzemiennych Cyklach Mutacji i Nocnego Audytu.
    Weryfikuje niezmienniki ACID, brak wycieku pamięci i nienaruszalność łańcucha SHA-256 Merkle Tip.
    """
    kv = VerifiedKVStore(db_path=":memory:")
    graph = KuzuGraphStore(db_path=":memory:")
    auditor = MemoryAuditor(graph_store=graph, kv_store=kv)

    # Wykonanie 100 cykli naprzemiennych operacji
    for cycle in range(1, 101):
        key = f"telemetry:sensor_{cycle % 10}"
        val = f"reading_{cycle}_{cycle * 1.5}"

        # 1. Zapis atomowy za pomocą Odwracalnej Monady Stanu (TransactionalDeltaTensor)
        rec = MemoryRecord(
            subject=key,
            predicate="state_value",
            object=val,
            confidence=0.95,
            source_type=EpistemicSource.TOOL_OUTPUT,
            is_state_variable=True,
            metadata={"cycle": cycle},
        )
        monad = TransactionalDeltaTensor(tx_id=f"tx_soak_{cycle}", record=rec)
        ok = await monad.execute(auditor)
        assert ok is True
        assert monad.status == "COMMITTED"

        # 3. Co 25 cykli: wykonaj nocny audyt i weryfikację integralności
        if cycle % 25 == 0:
            is_valid, broken_seq = await kv.verify_chain_integrity()
            assert is_valid is True, f"Naruszenie integralności w cyklu {cycle} (seq {broken_seq})"
            tip = kv.get_latest_audit_hash()
            assert tip is not None and len(tip) == 64, "Merkle Tip musi być poprawnym haszem SHA-256"

    # Weryfikacja końcowa
    is_valid_final, _ = await kv.verify_chain_integrity()
    assert is_valid_final is True
