from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from atlas_memory.models import EpistemicSource, MemoryRecord


class CompactionLevel(BaseModel):
    level: str  # "session_l1", "episode_summary", "permanent_profile_l2"
    source_items_count: int
    compressed_text: str
    extracted_facts: List[MemoryRecord] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)


class ContextCompactor:
    """
    Moduł C: Kaskadowe Streszczanie Kontekstu (Context Compaction / Rolling Summarizer).
    
    Tworzy streszczenia hierarchiczne:
    Sesje (L1) -> Epizody tygodniowe -> Stałe wzorce profilu (L2).
    Pozwala skompresować setki tur rozmów do pojedynczych gęstych rekordów.
    """

    def __init__(self, session_window_size: int = 8):
        self.window_size = session_window_size
        self._turn_buffer: List[Dict[str, Any]] = []
        self.compacted_history: List[CompactionLevel] = []

    def add_interaction_turn(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Dodaje turę interakcji do bufora roboczego.
        Zwraca True, jeśli bufor osiągnął limit i wymaga kompakcji.
        """
        self._turn_buffer.append({
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": time.time(),
        })
        return len(self._turn_buffer) >= self.window_size

    def compact_working_window(self, episode_id: str = "default_episode") -> CompactionLevel:
        """
        Kompaktuje bieżący bufor tur rozmowy (L1) do streszczenia epizodu
        oraz wyodrębnia kluczowe fakty relacyjne.
        """
        if not self._turn_buffer:
            return CompactionLevel(
                level="session_l1",
                source_items_count=0,
                compressed_text="Empty session buffer.",
            )

        items_count = len(self._turn_buffer)

        # Ekstrakcja kluczowych zdań i decyzji
        lines = []
        extracted_facts: List[MemoryRecord] = []

        for turn in self._turn_buffer:
            role = turn["role"].upper()
            content = turn["content"].strip()
            lines.append(f"[{role}]: {content}")

            # Heurystyczna ekstrakcja faktów z wypowiedzi konfiguracyjnych/decyzyjnych
            if any(k in content.lower() for k in ["zmień", "ustaw", "serwer to", "używaj", "status:", "port:"]):
                extracted_facts.append(MemoryRecord(
                    subject=f"episode_{episode_id}",
                    predicate="contains_decision",
                    object=content[:120],
                    confidence=0.90,
                    source_type=EpistemicSource.USER_EXPLICIT if role == "USER" else EpistemicSource.AGENT_INFERENCE,
                    importance_score=0.8,
                ))

        compressed_summary = (
            f"--- Epizod {episode_id} ({items_count} tur) ---\n"
            + "\n".join(lines[:4])
            + (f"\n... [{items_count - 4} pominiętych tur] ..." if items_count > 4 else "")
        )

        compaction_record = CompactionLevel(
            level="episode_summary",
            source_items_count=items_count,
            compressed_text=compressed_summary,
            extracted_facts=extracted_facts,
        )

        self.compacted_history.append(compaction_record)
        self._turn_buffer.clear()
        return compaction_record

    def clear(self) -> None:
        """Czyści bufor roboczy."""
        self._turn_buffer.clear()

