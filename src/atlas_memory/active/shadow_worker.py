from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional

import httpx

from atlas_memory.orchestrator import MemoryOrchestrator

logger = logging.getLogger(__name__)



class OmniRouteShadowWorker:
    """
    Faza D: Asynchroniczny Shadow Worker na OmniRoute (Micro-LLM Extraction).
    
    Konsumuje tury konwersacji z kolejki asynchronicznej i wywołuje lekki, tani model (np. Qwen-mini)
    przez lokalne proxy OmniRoute (localhost:20128/v1/chat/completions) do precyzyjnej ekstrakcji SPO.
    Następnie przekazuje fakty do MemoryOrchestrator w celu arbitrażu epistemicznego i UPSERT-u do grafu.
    """

    EXTRACTION_SYSTEM_PROMPT = (
        "Jesteś precyzyjnym silnikiem ekstrakcji wiedzy dla pamięci agenta AI.\n"
        "Twoim zadaniem jest wyciągnięcie ze strumienia konwersacji wyłącznie twardych faktów, "
        "zmiennych konfiguracyjnych, parametrów i relacji między encjami.\n"
        "V151 (detoksykacja proweniencji): NIE oznaczaj faktów jako \"user_explicit\" —\n"
        "ekstrakcja cieniowa nigdy nie jest bezpośrednim dowodzeniem użytkownika; "
        "fakty pochodzące z tej ekstrakcji mają source_type \"shadow_inference\".\n"
        "Zwróć WYŁĄCZNIE poprawny blok JSON (tablica obiektów) bez żadnego dodatkowego tekstu ani wstępu:\n"
        "[\n"
        '  {"subject": "nazwa_encji", "predicate": "relacja_lub_parametr", "object": "wartość", '
        '"source_type": "tool_output"|"agent_inference"|"shadow_inference", "confidence": 1.0, "is_state_variable": true|false}\n'
        "]"
    )

    def __init__(
        self,
        orchestrator: MemoryOrchestrator,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        api_key: str = "omniroute-local",
        timeout: float = 5.0,
        max_retries: int = 3,
        http_client_fn: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        self.orchestrator = orchestrator
        from atlas_memory.core.config import get_config
        cfg = get_config()
        self.api_base = (api_base or cfg.omniroute_url).rstrip("/")
        self.model = model or cfg.shadow_model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._custom_http_client = http_client_fn

        self.turn_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=1000)
        self._worker_task: Optional[asyncio.Task] = None
        self._is_running: bool = False

        # V89/T2: śledzenie statusu ekstrakcji (ZAKAZ cichego sukcesu).
        self.last_extraction_status: Dict[str, Any] = {
            "source": None,
            "error_code": None,
            "error": None,
            "fallback_used": False,
        }
        # Wyniki per-tura (konsumowane przez RPC `shadow_drain` przez pop_completed()).
        # Bounded buffer chroniący przed nieograniczonym rozrostem pamięci RAM.
        self.completed_turns: deque[Dict[str, Any]] = deque(maxlen=1000)
        self._in_flight: int = 0

        # Rejestracja ekstrakcji w orkiestratorze
        self.orchestrator.shadow_extractor = self._extract_spo

    @property
    def queue_depth(self) -> int:
        return self.turn_queue.qsize()

    @property
    def in_flight(self) -> int:
        return self._in_flight

    def pop_completed(self) -> List[Dict[str, Any]]:
        """Zwraca i czyści wyniki tur przetworzonych od ostatniego wywołania."""
        done = list(self.completed_turns)
        self.completed_turns.clear()
        return done

    async def flush(self, wait_s: float = 2.0) -> int:
        """Deterministyczny drenaż: przetwarza kolejkę i czeka na turę w locie.

        `shadow_drain` musi widzieć skutek tury wcześniej podanej przez
        `shadow_enqueue`, nawet gdy pętla `_worker_loop` skonsumowała ją
        równolegle — dlatego czekamy na `_in_flight == 0`.
        """
        deadline = time.time() + max(0.0, wait_s)
        processed = await self.process_all_pending()
        while time.time() < deadline and (not self.turn_queue.empty() or self._in_flight > 0):
            await asyncio.sleep(0.02)
            processed += await self.process_all_pending()
        return processed

    def _record_extraction(
        self, source: str, error_code: Optional[str] = None, error: Optional[str] = None
    ) -> None:
        """Zapisuje jawny status ostatniej ekstrakcji (nie pozwala zgubić błędu OmniRoute)."""
        self.last_extraction_status = {
            "source": source,
            "error_code": error_code,
            "error": error,
            "fallback_used": source.startswith("regex"),
            "timestamp": time.time(),
        }

    async def enqueue_turn(self, user_msg: str, agent_response: str, session_id: str = "default") -> None:
        """Wrzuca interakcję do nieblokującej kolejki w tle."""
        await self.turn_queue.put({
            "user_msg": user_msg,
            "agent_response": agent_response,
            "session_id": session_id,
            "timestamp": time.time(),
        })

    def start(self) -> asyncio.Task:
        """Uruchamia asynchronicznego workera w pętli zdarzeń."""
        if self._worker_task is None or self._worker_task.done():
            self._is_running = True
            self._worker_task = asyncio.create_task(self._worker_loop())
        return self._worker_task

    async def stop(self) -> None:
        """Bezpiecznie zatrzymuje workera."""
        self._is_running = False
        # V89/T2: CZEKAJ na dokończenie tury w locie (shadow_reconcile -> engine
        # write executor). Cancel w połowie zapisu Kùzu + zamknięcie bazy w
        # daemon.stop() to race prowadzący do segfaultu w kuzu (obserwowany).
        deadline = time.time() + 10.0
        while self._in_flight > 0 and time.time() < deadline:
            await asyncio.sleep(0.05)
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                logger.debug("Shadow worker stopped gracefully via cancellation.")
            self._worker_task = None

    async def process_all_pending(self) -> int:
        """Przetwarza wszystkie oczekujące tury w kolejce."""
        processed_count = 0
        while not self.turn_queue.empty():
            turn_data = await self.turn_queue.get()
            try:
                await self._process_single_turn(turn_data)
                processed_count += 1
            finally:
                self.turn_queue.task_done()
        return processed_count

    async def _worker_loop(self) -> None:
        while self._is_running:
            try:
                turn_data = await self.turn_queue.get()
                try:
                    await self._process_single_turn(turn_data)
                finally:
                    self.turn_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Shadow worker loop error: %s", exc, exc_info=True)
                await asyncio.sleep(0.5)

    async def _process_single_turn(self, turn_data: Dict[str, Any]) -> None:
        user_msg = turn_data.get("user_msg", "")
        agent_resp = turn_data.get("agent_response", "")
        session_id = turn_data.get("session_id", "default")

        self._record_extraction("pending")
        self._in_flight += 1
        try:
            committed = await self.orchestrator.shadow_reconcile(
                user_msg=user_msg,
                agent_response=agent_resp,
                session_id=session_id,
            )
        finally:
            self._in_flight -= 1
        self.completed_turns.append(
            {
                "session_id": session_id,
                "facts_committed": len(committed or []),
                "committed_keys": [rec.triple_key for rec in (committed or [])],
                "extraction": dict(self.last_extraction_status),
            }
        )

    async def _extract_spo(self, user_msg: str, agent_response: str) -> List[Dict[str, Any]]:
        """Asynchroniczne wywołanie API OmniRoute z mechanizmem retry i circuit-breakerem."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"User: {user_msg}\nAgent: {agent_response}"},
            ],
            "temperature": 0.0,
            "max_tokens": 800,
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                if self._custom_http_client is not None:
                    if inspect.iscoroutinefunction(self._custom_http_client):
                        response_json = await self._custom_http_client(f"{self.api_base}/chat/completions", payload)
                    else:
                        response_json = await asyncio.to_thread(
                            self._custom_http_client, f"{self.api_base}/chat/completions", payload
                        )
                else:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        resp = await client.post(
                            f"{self.api_base}/chat/completions",
                            json=payload,
                            headers={
                                "Content-Type": "application/json",
                                "Authorization": f"Bearer {self.api_key}",
                            },
                        )
                        resp.raise_for_status()
                        response_json = resp.json()

                content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                facts = self._parse_json_facts(content)
                self._record_extraction("llm" if facts else "llm_empty_response")
                return facts

            except Exception as exc:
                if attempt >= self.max_retries:
                    # Circuit-breaker: natychmiastowy fallback do wbudowanego ekstraktora regułowego.
                    # V89/T2: fallback NIE jest ciszą — błąd OmniRoute jest rejestrowany jawnym kodem.
                    self._record_extraction("regex_fallback", "OMNIROUTE_UNAVAILABLE", repr(exc))
                    return self.orchestrator._fallback_extract_facts(user_msg, agent_response)
                await asyncio.sleep(0.1 * (2 ** (attempt - 1)))

        self._record_extraction("regex_fallback", "OMNIROUTE_UNAVAILABLE")
        return self.orchestrator._fallback_extract_facts(user_msg, agent_response)

    def _extract_spo_sync(self, user_msg: str, agent_response: str) -> List[Dict[str, Any]]:
        """Synchroniczna fasada delegująca do wykonania asynchronicznego."""
        try:
            asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(asyncio.run, self._extract_spo(user_msg, agent_response)).result()
        except RuntimeError:
            return asyncio.run(self._extract_spo(user_msg, agent_response))

    def _parse_json_facts(self, content: str) -> List[Dict[str, Any]]:
        """Czyści i parsuje odpowiedź modelu do listy trójek (obsługuje tablice [...] oraz pojedyncze obiekty {...})."""
        content_clean = content.strip()
        # Wyciągnięcie bloku ```json ... ``` lub ``` ... ```
        block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content_clean)
        candidate_str = block_match.group(1).strip() if block_match else content_clean

        # 1. Próba znalezienia tablicy [...]
        array_match = re.search(r"(\[[\s\S]*?\])", candidate_str)
        if array_match:
            try:
                parsed = json.loads(array_match.group(1))
                if isinstance(parsed, list):
                    valid_items = []
                    for item in parsed:
                        if isinstance(item, dict) and "subject" in item and "predicate" in item and "object" in item:
                            valid_items.append(item)
                    if valid_items:
                        return valid_items
            except Exception:
                pass

        # 2. Próba znalezienia pojedynczego obiektu {...}
        obj_match = re.search(r"(\{[\s\S]*?\})", candidate_str)
        if obj_match:
            try:
                parsed = json.loads(obj_match.group(1))
                if isinstance(parsed, dict) and "subject" in parsed and "predicate" in parsed and "object" in parsed:
                    return [parsed]
            except Exception:
                pass

        # 3. Bezpośredni fallback
        try:
            parsed = json.loads(candidate_str)
            if isinstance(parsed, dict) and "subject" in parsed and "predicate" in parsed and "object" in parsed:
                return [parsed]
            elif isinstance(parsed, list):
                return [
                    item for item in parsed
                    if isinstance(item, dict) and "subject" in item and "predicate" in item and "object" in item
                ]
        except Exception as exc:
            logger.debug("Failed to parse JSON facts from model response: %s", exc)
        return []

