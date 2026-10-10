"""Wyciąganie REALNEJ trajektorii narzędzi z transcriptu Hermes (V64 Procedural Foundation).

Moduł deterministyczny: zamienia listę `messages` (kontrakt Hermes MemoryProvider:
asystent z `tool_calls` + wyniki `role=="tool"`) w sekwencję kroków
`{tool_name, action, tool, parameters, success, error}` z **obiektywnym**
kryterium sukcesu liczonym z treści odpowiedzi narzędzia (exit code / flaga
błędu / Traceback), a NIE z symulowanej nagrody modelu latentnego (BUG-010).

Zero I/O, zero torch, tylko stdlib.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional, Union

_EXIT_CODE_RE = re.compile(
    r"exit[\s_\-]?code[\s:=\-]+(-?\d+)", re.IGNORECASE
)
_TIMED_OUT_RE = re.compile(r"\btimed?\s*out\b", re.IGNORECASE)


@dataclass(slots=True)
class ToolExecutionResult:
    """Typowana koperta wyniku wykonania narzędzia (V155 Filar II)."""

    tool_name: str
    output: str
    exit_code: Optional[int] = None
    is_error: Optional[bool] = None
    timed_out: bool = False
    error_message: Optional[str] = None

    @classmethod
    def from_raw(
        cls,
        text: str,
        tool_name: str = "",
        parameters: Optional[dict[str, Any]] = None,
        raw_tool_msg: Optional[dict[str, Any]] = None,
    ) -> ToolExecutionResult:
        params = parameters or {}
        raw_msg = raw_tool_msg or {}
        output_str = text or ""

        # 1. Jawna flaga is_error w protokole narzędzi
        is_err: Optional[bool] = None
        if "is_error" in raw_msg:
            is_err = bool(raw_msg["is_error"])

        # 2. Kod wyjścia (exit code)
        exit_c: Optional[int] = None
        if "exit_code" in raw_msg:
            try:
                exit_c = int(raw_msg["exit_code"])
            except (ValueError, TypeError):
                pass
        elif "exit_code" in params:
            try:
                exit_c = int(params["exit_code"])
            except (ValueError, TypeError):
                pass
        else:
            match = _EXIT_CODE_RE.search(output_str)
            if match:
                exit_c = int(match.group(1))

        # 3. Timeout
        is_timeout = bool(_TIMED_OUT_RE.search(output_str))

        # 4. JSON payload inspection (np. narzędzia zwracające strukturalny błąd)
        err_msg: Optional[str] = None
        stripped = output_str.strip()
        if stripped.startswith("{"):
            try:
                obj = json.loads(stripped)
                if isinstance(obj, dict):
                    if obj.get("error"):
                        err_msg = str(obj["error"])
                        if is_err is None:
                            is_err = True
                    elif str(obj.get("status", "")).lower() in {"error", "failed", "failure"}:
                        err_msg = str(obj.get("status"))
                        if is_err is None:
                            is_err = True
                    elif str(obj.get("status", "")).lower() in {"ok", "success"}:
                        if is_err is None:
                            is_err = False
            except (json.JSONDecodeError, ValueError):
                pass

        return cls(
            tool_name=tool_name,
            output=output_str,
            exit_code=exit_c,
            is_error=is_err,
            timed_out=is_timeout,
            error_message=err_msg,
        )


def _content_to_text(content: Any) -> str:
    """Spłaszcza treść wyniku narzędzia do tekstu (str | list[{type:text,text}] | dict)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                txt = part.get("text") or part.get("content") or ""
                parts.append(str(txt))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, default=str)
    return str(content)


def _parse_arguments(raw: Any) -> dict[str, Any]:
    """Argumenty tool_call bywają JSON-stringiem albo dictem — zwracaj dict, {} przy błędzie."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"_raw": parsed}
        except (json.JSONDecodeError, ValueError):
            return {"_raw": raw}
    return {}


def _judge_result(
    result_or_text: Union[ToolExecutionResult, str],
    tool_name: str = "",
    parameters: Optional[dict[str, Any]] = None,
    raw_tool_msg: Optional[dict[str, Any]] = None,
) -> tuple[bool, str | None]:
    """Weryfikacja sukcesu narzędzia w standardzie typowanej koperty (V155 Filar II).

    1. Sukces weryfikowany jest wyłącznie na podstawie flag protokołu (is_error) oraz kodu wyjścia (exit_code == 0).
    2. Obecność tokenów 'error' lub 'traceback' w treści komend inspekcyjnych o kodzie wyjścia 0 traktowana jest jako SUKCES.
    """
    if isinstance(result_or_text, ToolExecutionResult):
        envelope = result_or_text
    else:
        envelope = ToolExecutionResult.from_raw(
            text=result_or_text,
            tool_name=tool_name,
            parameters=parameters,
            raw_tool_msg=raw_tool_msg,
        )

    # 1. Pusty wynik
    if not envelope.output or not envelope.output.strip():
        if envelope.is_error is False or (envelope.exit_code is not None and envelope.exit_code == 0):
            return True, None
        return False, "no_result"

    # 2. Jawna flaga błędu w protokole narzędzi
    if envelope.is_error is True:
        return False, envelope.error_message or envelope.output.strip()[:200] or "tool_flagged_error"

    # 3. Kod wyjścia (exit code)
    if envelope.exit_code is not None:
        if envelope.exit_code == 0:
            # Sukces bezwzględny — obecność słów "error" czy "traceback" w output traktowana jako dane
            return True, None
        return False, f"exit_code={envelope.exit_code}"

    # 4. Timeout
    if envelope.timed_out:
        return False, "timed_out"

    # 5. Jeśli flaga błędu explicite False lub brak sygnałów błędu
    return True, None


def extract_trajectory(
    messages: list[dict[str, Any]] | None, session_id: str
) -> list[dict[str, Any]]:
    """Wyciąga REALNĄ sekwencję wywołań narzędzi z transcriptu sesji.

    Zwraca [] gdy brak wywołań narzędzi. Każdy krok:
    {"tool_name", "action", "tool", "parameters", "success", "error", "session_id"}
    Sekwencja jest kolejnością appearances tool_calls w historii (chronologicznie).
    """
    if not messages or not isinstance(messages, list):
        return []

    # Indeks wyników narzędzi po tool_call_id (fallback: kolejność).
    results_by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    ordered_results: list[tuple[str, dict[str, Any]]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool":
            text = _content_to_text(msg.get("content"))
            call_id = msg.get("tool_call_id")
            if call_id:
                results_by_id[str(call_id)] = (text, msg)
            ordered_results.append((text, msg))

    steps: list[dict[str, Any]] = []
    pending_no_id = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") or {}
            if not isinstance(fn, dict):
                continue
            name = str(fn.get("name") or "").strip()
            if not name:
                continue
            call_id = str(call.get("id") or "").strip()
            if call_id:
                # Ścisłe dopasowanie po call_id — brak wyniku NIE może kraść wyniku innego wywołania (V83)
                result_text, raw_tool_msg = results_by_id.get(call_id, ("", {}))
            else:
                if pending_no_id < len(ordered_results):
                    result_text, raw_tool_msg = ordered_results[pending_no_id]
                else:
                    result_text, raw_tool_msg = "", {}
                pending_no_id += 1

            params = _parse_arguments(fn.get("arguments"))
            success, error = _judge_result(
                result_text,
                tool_name=name,
                parameters=params,
                raw_tool_msg=raw_tool_msg,
            )
            steps.append(
                {
                    "tool_name": name,
                    "action": name,
                    "tool": name,
                    "parameters": params,
                    "success": success,
                    "error": error,
                    "session_id": session_id,
                }
            )
    return steps
