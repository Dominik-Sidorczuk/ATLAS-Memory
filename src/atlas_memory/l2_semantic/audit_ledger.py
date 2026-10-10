"""V112 F-028+ (fail-loud-or-ledger): Evidence ledger for swallowed exceptions."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ── V112 F-028+ (fail-loud-or-ledger): dowodowy rejestr cichych polkniec ──────
# Kazde "except Exception" ktore NIE propaguje (swallow) musi zostawic slad:
# logger.warning + wpis do append-only JSONL (swallow ledger) albo do wstrzyk-
# nietego sinka (mock ledger / pipeline audytowy aplikacji). record_swallow
# NIGDY nie rzuca — awaria ledgera degraduje sie do samej log-warning.
SWALLOW_EVENT_TYPE = "swallow"
_SWALLOW_DEDUP_CAP = 1024
_swallow_lock = threading.Lock()
_swallow_sink: Optional[Callable[[Dict[str, Any]], None]] = None
_swallow_seen: Set[Tuple[str, str, str]] = set()


def set_swallow_sink(sink: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    """Instaluje (lub wyczyści, sink=None) alternatywny odbiornik zdarzen swallow."""
    global _swallow_sink
    with _swallow_lock:
        _swallow_sink = sink


def reset_swallow_state() -> None:
    """Test hook: czysci zbior dedupe i sink do stanu domyslnego."""
    global _swallow_sink
    with _swallow_lock:
        _swallow_seen.clear()
        _swallow_sink = None


def _default_swallow_ledger_path() -> Path:
    override = os.environ.get("ATLAS_SWALLOW_LEDGER", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".atlas" / "swallow_ledger.jsonl"


def record_swallow(module: str, key: str, exc: Optional[BaseException]) -> None:
    """Zarejestruj udowodnione polkniecie wyjatku (event typu 'swallow').

    Widocznosc dwojak: (1) logger.warning przy pierwszym sightingu,
    (2) append JSONL do pliku ledgera lub do wstrzymnietego sinka.
    Powtorzenia (module, key, exc_type) sa wyciszane do debug i pomijane
    w ledgerze (dedupe przeciw zalewowi na sciezce hot-path).
    Funkcja NIGDY nie propaguje wyjatku — nie wolno jej wywalic hot-path.
    """
    try:
        exc_type = type(exc).__name__ if exc is not None else "Unknown"
        sig = (str(module), str(key), exc_type)
        sink = None
        with _swallow_lock:
            already_seen = sig in _swallow_seen
            if not already_seen:
                if len(_swallow_seen) >= _SWALLOW_DEDUP_CAP:
                    _swallow_seen.clear()
                _swallow_seen.add(sig)
                sink = _swallow_sink
        if already_seen:
            logger.debug("swallow repeats suppressed (dedupe): %s key=%s %s", module, key, exc_type)
            return
        payload = {
            "event": SWALLOW_EVENT_TYPE,
            "module": str(module),
            "key": str(key),
            "exc_type": exc_type,
            "exc_msg": str(exc)[:500] if exc is not None else "",
            "ts": time.time(),
        }
        logger.warning(
            "Swallowed exception (evidenced) in %s (key=%s): %s: %s",
            module, key, exc_type, payload["exc_msg"],
        )
        if sink is not None:
            try:
                sink(dict(payload))
            except Exception as sink_exc:  # ledger-niedostepny -> tylko ostrzezenie
                logger.warning("swallow sink failed for %s key=%s: %s", module, key, sink_exc)
            return
        path = _default_swallow_ledger_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with _swallow_lock:
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        except OSError as file_exc:  # ledger-niedostepny -> tylko ostrzezenie
            logger.warning("swallow ledger file unavailable (%s) for %s key=%s: %s", path, module, key, file_exc)
    except Exception as outer_exc:  # absolutna gwarancja: record_swallow nie wywala hot-path
        logger.warning("record_swallow internal failure (non-propagating) in %s key=%s: %s", module, key, outer_exc)


__all__ = [
    "SWALLOW_EVENT_TYPE",
    "record_swallow",
    "reset_swallow_state",
    "set_swallow_sink",
]
