from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CacheEvent(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    model: str
    prefix_hash: str
    hit: bool
    tokens_saved: int = 0
    prompt_tokens: int = 0


class CacheHitMonitor:
    """
    Faza E: CacheHitMonitor & Telemetria Ekonomii Tokenów (Prompt Cache Metric Engine).
    
    Mierzy skuteczność kontraktu pamięci podręcznej (Cache Contract) dla modeli chmurowych (OmniRoute / Anthropic / vLLM):
    1. Śledzi wskaźnik Cache Hit-Rate (%) dla stabilnego prefiksu promptu.
    2. Oblicza oszczędności finansowe i liczbę zaoszczędzonych tokenów wejściowych (KV-cache reuse).
    3. Generuje raporty telemetryczne bez żadnego narzutu LLM.
    """

    MODEL_PRICING_PER_1M = {
        "deepseek-chat": {"uncached": 0.27, "cached": 0.07},
        "deepseek-reasoner": {"uncached": 0.55, "cached": 0.14},
        "qwen-2.5-72b": {"uncached": 0.35, "cached": 0.08},
        "claude-3-5-sonnet": {"uncached": 3.00, "cached": 0.30},
        "default": {"uncached": 0.50, "cached": 0.10},
    }

    def __init__(self):
        self._events: List[CacheEvent] = []
        self._model_stats: Dict[str, Dict[str, int]] = {}

    def report_cache_hit(
        self,
        model: str,
        prefix_hash: str,
        hit: bool,
        tokens_saved: int = 1500,
        prompt_tokens: int = 2000,
    ) -> None:
        """Rejestruje pojedyncze zdarzenie zapytania z informacją o trafieniu do cache."""
        event = CacheEvent(
            model=model,
            prefix_hash=prefix_hash,
            hit=hit,
            tokens_saved=tokens_saved if hit else 0,
            prompt_tokens=prompt_tokens,
        )
        self._events.append(event)

        if model not in self._model_stats:
            self._model_stats[model] = {"hits": 0, "misses": 0, "tokens_saved": 0, "total_tokens": 0}

        if hit:
            self._model_stats[model]["hits"] += 1
            self._model_stats[model]["tokens_saved"] += tokens_saved
        else:
            self._model_stats[model]["misses"] += 1
        self._model_stats[model]["total_tokens"] += prompt_tokens

    def record_turn(
        self,
        model: str,
        prefix_hash: str,
        cached: bool,
        tokens_in_prefix: int = 1500,
    ) -> None:
        """Wygodny alias do rejestracji tury."""
        self.report_cache_hit(
            model=model,
            prefix_hash=prefix_hash,
            hit=cached,
            tokens_saved=tokens_in_prefix,
            prompt_tokens=tokens_in_prefix + 500,
        )

    def get_hit_rate(self, model: Optional[str] = None) -> float:
        """Zwraca wskaźnik trafień cache (od 0.0 do 1.0)."""
        if model:
            stats = self._model_stats.get(model, {"hits": 0, "misses": 0})
            total = stats["hits"] + stats["misses"]
            return round(stats["hits"] / total, 4) if total > 0 else 0.0

        total_hits = sum(1 for e in self._events if e.hit)
        total_events = len(self._events)
        return round(total_hits / total_events, 4) if total_events > 0 else 0.0

    def monthly_report(self) -> Dict[str, Any]:
        """Generuje podsumowanie oszczędności tokenów i kosztów."""
        total_turns = len(self._events)
        total_hits = sum(1 for e in self._events if e.hit)
        total_saved_tokens = sum(e.tokens_saved for e in self._events)
        total_prompt_tokens = sum(e.prompt_tokens for e in self._events)
        overall_hit_rate = (total_hits / total_turns) if total_turns > 0 else 0.0

        # Estymacja oszczędności w USD
        total_usd_saved = 0.0
        by_model = {}

        for m_name, stats in self._model_stats.items():
            pricing = self.MODEL_PRICING_PER_1M.get(m_name, self.MODEL_PRICING_PER_1M["default"])
            saved_m_tokens = stats["tokens_saved"]
            # Oszczędność = (cena bez cache - cena z cache) * zaoszczędzone tokeny / 1M
            usd_diff_per_m = (pricing["uncached"] - pricing["cached"])
            model_saved_usd = (saved_m_tokens / 1_000_000.0) * usd_diff_per_m
            total_usd_saved += model_saved_usd

            m_total = stats["hits"] + stats["misses"]
            by_model[m_name] = {
                "total_turns": m_total,
                "hits": stats["hits"],
                "misses": stats["misses"],
                "hit_rate_pct": round((stats["hits"] / m_total) * 100.0, 2) if m_total > 0 else 0.0,
                "tokens_saved": saved_m_tokens,
                "usd_saved_estimate": round(model_saved_usd, 4),
            }

        return {
            "total_monitored_turns": total_turns,
            "overall_cache_hit_rate_pct": round(overall_hit_rate * 100.0, 2),
            "total_tokens_saved": total_saved_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "total_usd_saved_estimate": round(total_usd_saved, 4),
            "by_model": by_model,
        }

    def reset(self) -> None:
        """Resetuje zebrane metryki."""
        self._events.clear()
        self._model_stats.clear()

