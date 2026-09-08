from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field

_NORM_PUNCT_RE = re.compile(r"[^\w\s\-_]")
_NORM_SPACES_RE = re.compile(r"\s+")


class EntityEntry(BaseModel):
    canonical_id: str
    canonical_name: str
    aliases: Set[str] = Field(default_factory=set)
    entity_type: str = "general"
    description: Optional[str] = None


class EntityCanonicalizer:
    """
    Moduł B: Rozpoznawanie Encji i Aliasów (Entity Disambiguation / Coreference).
    
    Mapuje synonimy, nazwy potoczne i warianty pisowni na jeden globalny
    identyfikator encji (canonical_entity_id) przed zapisem do grafu i bazy stanu.
    """

    def __init__(self):
        self._entities: Dict[str, EntityEntry] = {}
        self._alias_to_id: Dict[str, str] = {}
        self._init_default_rules()

    def _init_default_rules(self) -> None:
        """Inicjalizacja podstawowych encji dla środowiska serwerowego i agentowego."""
        self.register_entity(
            canonical_id="entity_nas_01",
            canonical_name="TrueNAS Server",
            aliases=["mój nas", "nas", "truenas", "advantech", "serwer plików", "home nas"],
            entity_type="hardware_server",
            description="Lokalny serwer plików NAS TrueNAS na maszynie Advantech",
        )
        self.register_entity(
            canonical_id="entity_agent_core",
            canonical_name="Hermes Agent",
            aliases=["hermes", "hermes agent", "asystent", "mój agent", "agent core"],
            entity_type="software_agent",
            description="Rdzeń autonomicznego agenta Hermes",
        )

    def _normalize(self, text: str) -> str:
        """Normalizacja tekstu: małe litery, usunięcie zbędnych znaków i spacji."""
        text = text.lower().strip()
        text = _NORM_PUNCT_RE.sub("", text)
        return _NORM_SPACES_RE.sub(" ", text)


    def register_entity(
        self,
        canonical_id: str,
        canonical_name: str,
        aliases: Optional[List[str]] = None,
        entity_type: str = "general",
        description: Optional[str] = None,
    ) -> EntityEntry:
        """Rejestruje encję kanoniczną wraz z listą aliasów."""
        norm_id = self._normalize(canonical_id)
        entry = self._entities.get(norm_id)
        if entry is None:
            entry = EntityEntry(
                canonical_id=canonical_id,
                canonical_name=canonical_name,
                aliases={self._normalize(canonical_name)},
                entity_type=entity_type,
                description=description,
            )
            self._entities[norm_id] = entry

        # Rejestracja nazwy kanonicznej w indeksie aliasów
        self._alias_to_id[self._normalize(canonical_name)] = canonical_id
        self._alias_to_id[norm_id] = canonical_id

        if aliases:
            for alias in aliases:
                norm_alias = self._normalize(alias)
                if norm_alias:
                    entry.aliases.add(norm_alias)
                    self._alias_to_id[norm_alias] = canonical_id

        return entry

    def canonicalize(self, raw_name: str) -> str:
        """
        Zwraca kanoniczny entity_id dla podanej nazwy/aliasu.
        Jeśli encja nie jest znana w słowniku, generuje znormalizowany identyfikator.
        """
        if not raw_name:
            return "unknown_entity"

        norm = self._normalize(raw_name)
        if norm in self._alias_to_id:
            return self._alias_to_id[norm]

        # Sprawdzenie częściowego dopasowania prefiksu
        for alias, c_id in self._alias_to_id.items():
            if len(alias) >= 4 and (alias in norm or norm in alias):
                return c_id

        # Fallback: znormalizowana forma
        return norm.replace(" ", "_")

    def get_entity_info(self, canonical_id: str) -> Optional[EntityEntry]:
        """Zwraca metadane encji na podstawie jej identyfikatora."""
        return self._entities.get(self._normalize(canonical_id))

    def canonicalize_list(self, entities: List[str]) -> List[str]:
        """Kanonizuje listę encji z usunięciem duplikatów."""
        result = []
        seen = set()
        for e in entities:
            c_id = self.canonicalize(e)
            if c_id not in seen:
                seen.add(c_id)
                result.append(c_id)
        return result

