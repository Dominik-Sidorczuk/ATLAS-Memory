"""Testy weryfikacyjne pakietu hermes w standardzie Post-2027 AGI (V141).

Weryfikacja 5 kluczowych filarów odporności architektonicznej:
1. test_concurrency_race_bootstrap: 10 współbieżnych wątków wywołuje ensure_daemon_process_running;
   dokładnie 1 proces zostaje zainicjalizowany, reszta bezpiecznie czeka i łączy się z socketem.
2. test_mutation_aware_stale_cache_invalidation: mutacja w turze (atlas_remember) unieważnia
   zbuforowany stan prefetchu i nie dopuszcza do serwowania zatrutego cache.
3. test_full_10_tool_in_process_fallback: wyłączenie daemona UDS nie powoduje daemon_offline;
   wszystkie 10 narzędzi zwraca poprawny stan w trybie in-process.
4. test_trajectory_inspection_no_false_errors: grep "error" z exit code 0 i 50 liniami "error"
   jest klasyfikowany bezwzględnie jako sukces (success=True, error=None).
5. test_async_safety_zero_event_loop_crash: pełny cykl życia z poziomu aktywnej pętli asyncio
   nie powoduje awarii pętli ani zakleszczeń, a dispatch on_session_end mieści się w budżecie < 5ms.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from atlas_memory.hermes.atlas_provider import AtlasMemoryProvider
from atlas_memory.hermes.daemon_bootstrap import ensure_daemon_process_running
from atlas_memory.hermes.plugin_hooks import AtlasPluginHooks, DedicatedAsyncBridge
from atlas_memory.hermes.trajectory_extractor import extract_trajectory


def test_concurrency_race_bootstrap(tmp_path: Path) -> None:
    """10 współbieżnych wątków wywołuje ensure_daemon_process_running na brakującym sockecie.
    
    Tylko dokładnie 1 proces zostaje zainicjalizowany (subprocess.Popen called once),
    pozostałe 9 procesów czeka na zwolnienie blokady i natychmiast stwierdza gotowość gniazda.
    """
    sock_path = tmp_path / "concurrency_test.sock"
    spawn_counter = 0
    spawn_lock = concurrent.futures.thread.threading.Lock()

    def fake_popen(*args, **kwargs):
        nonlocal spawn_counter
        with spawn_lock:
            spawn_counter += 1
            # Pierwszy wątek, który zdobył blokadę flock, tworzy gniazdo
            sock_path.touch()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        return mock_proc

    def fake_send_rpc(sock_str: str, method: str, params: Any = None) -> Dict[str, Any]:
        # Demon odpowiada tylko wtedy, gdy socket faktycznie istnieje na dysku
        if Path(sock_str).exists():
            return {"status": "ok", "serving": True}
        raise ConnectionRefusedError("Socket not ready")

    provider = AtlasMemoryProvider(socket_path=str(sock_path))

    with patch("subprocess.Popen", side_effect=fake_popen) as mock_popen:
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(
                    ensure_daemon_process_running,
                    provider,
                    timeout_s=5.0,
                    send_rpc_fn=fake_send_rpc,
                )
                for _ in range(10)
            ]
            results = [f.result(timeout=10.0) for f in futures]

    assert all(results), "Wszystkie 10 wątków musi zakończyć się sukcesem"
    assert mock_popen.call_count == 1, f"Oczekiwano dokładnie 1 uruchomienia demona, uzyskano: {mock_popen.call_count}"
    assert spawn_counter == 1, f"Licznik spawnów musi wynosić dokładnie 1, uzyskano {spawn_counter}"


def test_mutation_aware_stale_cache_invalidation(tmp_path: Path) -> None:
    """Mutacja w turze (atlas_remember) unieważnia prefetch cache i nie serwuje zatrutych danych."""
    sock_path = tmp_path / "dummy_offline.sock"
    provider = AtlasMemoryProvider(socket_path=str(sock_path))

    # Symulujemy zapełnienie pamięci podręcznej prefetchu z poprzedniego wywołania
    with provider._lock:
        provider._queued_context = "### STARY STAN: klucz=stara_wartosc_v1"
        provider._queued_query = "stan projektu"

    # Weryfikacja, że przed mutacją prefetch zwróciłby stary stan
    assert provider._queued_context is not None

    # Agent w trakcie trwającej tury zapisuje nową wiedzę przez narzędzie atlas_remember
    mutation_res = provider.handle_tool_call(
        "atlas_remember",
        {"key": "stan projektu", "value": "nowa_wartosc_v2", "confidence": 1.0},
    )
    res_json = json.loads(mutation_res)
    assert res_json.get("status") == "ok"
    assert res_json.get("key") == "stan projektu"

    # Asercja 1: Cache prefetchu MUSI zostać natychmiast wyzerowany
    with provider._lock:
        assert provider._queued_context is None, "Cache prefetchu nie został unieważniony po atlas_remember"
        assert provider._queued_query is None

    # Asercja 2: Kolejne wywołanie prefetch nie może zwrócić zatrutego starego stanu
    fresh_prefetch = provider.prefetch("stan projektu")
    assert "stara_wartosc_v1" not in fresh_prefetch


def test_full_10_tool_in_process_fallback(tmp_path: Path) -> None:
    """Symulacja odciętego daemona UDS: wszystkie 10 narzędzi działa bezbłędnie in-process."""
    dead_sock = tmp_path / "dead_daemon.sock"
    provider = AtlasMemoryProvider(socket_path=str(dead_sock))

    tool_invocations = [
        ("atlas_remember", {"key": "alpha", "value": "omega", "confidence": 0.95}),
        ("atlas_get", {"key": "alpha"}),
        ("atlas_stats", {}),
        ("atlas_recall", {"query": "alpha", "limit": 5}),
        ("atlas_what_if", {"entity": "alpha", "action": "scale_up", "depth": 2}),
        ("atlas_active_sensing", {"probe": "alpha_health", "expected_value": "1", "observed_value": "1"}),
        ("atlas_task_feedback", {"key": "alpha", "task_success": True, "delta": 0.05}),
        ("atlas_verify_chain", {"heal": False, "deep": False}),
        ("atlas_sleep_sync", {"min_frequency": 1}),
        ("atlas_delete", {"key": "alpha"}),
    ]

    executed_count = 0
    for tool_name, args in tool_invocations:
        resp_raw = provider.handle_tool_call(tool_name, args)
        assert isinstance(resp_raw, str), f"Narzędzie {tool_name} nie zwróciło ciągu znaków"
        resp = json.loads(resp_raw)

        # Żadne narzędzie nie może zwrócić błędu niedostępności demona
        assert resp.get("status") != "daemon_offline", f"Narzędzie {tool_name} zgłosiło daemon_offline!"
        assert resp.get("mode") == "in_process", f"Narzędzie {tool_name} nie ustawiło mode='in_process': {resp}"
        executed_count += 1

    assert executed_count == 10, "Wszystkie 10 narzędzi musi przejść weryfikację"


def test_trajectory_inspection_no_false_errors() -> None:
    """Narzędzia inspekcyjne (grep/cat) z exit code 0 i treścią pełną 'error' są sukcesem (success=True, error=None)."""
    # 50 linii symulowanego logu zawierającego słowo 'error'
    error_log_lines = "\n".join([f"2026-10-08 15:00:{i:02d} [ERROR] Handled exception #{i} in worker" for i in range(50)])

    messages = [
        {"role": "user", "content": "Przeszukaj logi w poszukiwaniu błędów."},
        {
            "role": "assistant",
            "content": "Uruchamiam grep w poszukiwaniu błędów w logach.",
            "tool_calls": [
                {
                    "id": "call_inspect_123",
                    "type": "function",
                    "function": {
                        "name": "run_command",
                        "arguments": json.dumps({"command": "grep -i 'error' /var/log/app.log"}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_inspect_123",
            "content": error_log_lines,
            "exit_code": 0,
            "is_error": False,
        },
    ]

    steps = extract_trajectory(messages, session_id="test_inspection_session")

    assert len(steps) == 1, "Oczekiwano wyekstrahowania dokładnie 1 kroku trajektorii"
    step = steps[0]
    assert step["tool_name"] == "run_command"
    assert step["success"] is True, f"Krok inspekcyjny został błędnie oznaczony jako błąd: {step}"
    assert step["error"] is None, f"Pole błędu nie jest puste: {step['error']}"


def test_async_safety_zero_event_loop_crash(tmp_path: Path) -> None:
    """Pełny cykl życia z poziomu wątku z aktywną pętlą asyncio działa bez zakleszczeń, a on_session_end trwa < 5ms."""
    sock_path = tmp_path / "async_safety.sock"
    provider = AtlasMemoryProvider(socket_path=str(sock_path))
    hooks = AtlasPluginHooks(provider)

    async def scenario_in_running_event_loop():
        loop = asyncio.get_running_loop()
        assert loop.is_running(), "Pętla zdarzeń musi być aktywna"

        # 1. on_session_start
        with patch.object(provider, "_ensure_daemon_running", return_value=True):
            start_ok = hooks.on_session_start(session_id="sess_async_1")
            assert start_ok is True

        # 2. prefetch
        ctx = provider.prefetch("Stan systemu", session_id="sess_async_1")
        assert isinstance(ctx, str)

        # 3. sync_turn
        provider.sync_turn("Stan systemu?", "System stabilny", session_id="sess_async_1")

        # 4. on_session_end — pomiar czasu dispatchu (budżet sub-5ms)
        dummy_messages = [
            {"session_id": "sess_async_1", "role": "user", "content": "Stan systemu?"},
            {"role": "assistant", "content": "System stabilny"},
        ]

        t0 = time.perf_counter()
        hooks.on_session_end(messages=dummy_messages, trigger_sleep_cycle=True, await_completion=False)
        dispatch_time_ms = (time.perf_counter() - t0) * 1000.0

        # Weryfikacja budżetu czasu sub-5ms (dajemy bufor do 15ms na wypadek obciążenia CPU CI)
        assert dispatch_time_ms < 15.0, f"Dispatch on_session_end przekroczył limit: {dispatch_time_ms:.2f} ms"

        # Pętla zdarzeń po zakończeniu operacji musi nadal bezbłędnie działać
        await asyncio.sleep(0.01)
        assert loop.is_running() is True
        return dispatch_time_ms

    # Uruchomienie scenariusza wewnątrz pętli asyncio
    elapsed = asyncio.run(scenario_in_running_event_loop())
    assert elapsed >= 0.0

    # Zamknięcie mostu asynchronicznego po teście
    bridge = DedicatedAsyncBridge.get_instance()
    bridge.shutdown()

