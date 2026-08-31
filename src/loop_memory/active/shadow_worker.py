from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from loop_memory.orchestrator import MemoryOrchestrator

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
        "Zwróć WYŁĄCZNIE poprawny blok JSON (tablica obiektów) bez żadnego dodatkowego tekstu ani wstępu:\n"
        "[\n"
        '  {"subject": "nazwa_encji", "predicate": "relacja_lub_parametr", "object": "wartość", '
        '"source_type": "user_explicit"|"tool_output"|"agent_inference", "confidence": 1.0, "is_state_variable": true|false}\n'
        "]"
    )

    def __init__(
        self,
        orchestrator: MemoryOrchestrator,
        api_base: str = "http://localhost:20128/v1",
        model: str = "qwen-mini",
        api_key: str = "omniroute-local",
        timeout: float = 5.0,
        max_retries: int = 3,
        http_client_fn: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        self.orchestrator = orchestrator
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._custom_http_client = http_client_fn

        self.turn_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._is_running: bool = False

        # Rejestracja ekstrakcji w orkiestratorze
        self.orchestrator.shadow_extractor = self._extract_spo_sync

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
            except Exception:
                await asyncio.sleep(0.5)

    async def _process_single_turn(self, turn_data: Dict[str, Any]) -> None:
        user_msg = turn_data.get("user_msg", "")
        agent_resp = turn_data.get("agent_response", "")
        session_id = turn_data.get("session_id", "default")

        await self.orchestrator.shadow_reconcile(
            user_msg=user_msg,
            agent_response=agent_resp,
            session_id=session_id,
        )

    def _extract_spo_sync(self, user_msg: str, agent_response: str) -> List[Dict[str, Any]]:
        """Synchroniczne wywołanie API OmniRoute z mechanizmem retry."""
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
                    response_json = self._custom_http_client(f"{self.api_base}/chat/completions", payload)
                else:
                    req_data = json.dumps(payload).encode("utf-8")
                    req = urllib.request.Request(
                        f"{self.api_base}/chat/completions",
                        data=req_data,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.api_key}",
                        },
                    )
                    with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                        response_json = json.loads(resp.read().decode("utf-8"))

                content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                return self._parse_json_facts(content)

            except Exception:
                if attempt == self.max_retries:
                    # Fallback do wbudowanego ekstraktora regułowego
                    return self.orchestrator._fallback_extract_facts(user_msg, agent_response)
                time.sleep(0.1 * (2 ** (attempt - 1)))

        return []

    def _parse_json_facts(self, content: str) -> List[Dict[str, Any]]:
        """Czyści i parsuje odpowiedź modelu do listy trójek."""
        content_clean = content.strip()
        # Wyciągnięcie bloku ```json ... ```
        json_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", content_clean)
        if json_match:
            content_clean = json_match.group(1)
        else:
            array_match = re.search(r"(\[[\s\S]*?\])", content_clean)
            if array_match:
                content_clean = array_match.group(1)

        try:
            parsed = json.loads(content_clean)
            if isinstance(parsed, list):
                valid_items = []
                for item in parsed:
                    if isinstance(item, dict) and "subject" in item and "predicate" in item and "object" in item:
                        valid_items.append(item)
                return valid_items
        except Exception as exc:
            logger.debug("Failed to parse JSON facts from model response: %s", exc)
        return []

