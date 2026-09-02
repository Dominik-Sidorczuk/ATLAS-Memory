"""Epistemic Byzantine Fault Tolerant Conflict-Free Replicated Data Type (BFT-CRDT)."""

from __future__ import annotations

import time
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from atlas_memory.sync.crdt import LWWElementSet
from atlas_memory.sync.epistemic_reputation import EpistemicReputation
from atlas_memory.sync.threshold_signatures import SignatureResult, ThresholdSigner


class BFTLWWSet(BaseModel):
    """Rozszerzenie LWWElementSet o weryfikację kworum podpisów progowych i reputacji epistemicznej."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    quorum: int = Field(default=3, ge=1, description="Minimalna wymagana liczba podpisów w kworum")
    reputation: Optional[EpistemicReputation] = Field(
        default=None,
        description="Silnik reputacji epistemicznej peerów",
    )
    signer: Optional[ThresholdSigner] = Field(
        default=None,
        description="Weryfikator / sygnatariusz progowy",
    )
    underlying_set: LWWElementSet = Field(
        default_factory=LWWElementSet,
        description="Podkładowy stan LWWElementSet",
    )
    rejected_operations: List[str] = Field(
        default_factory=list,
        description="Rejestr odrzuconych operacji (dla audytu BFT)",
    )

    def bft_add(
        self,
        element: str,
        operation: str,
        signatures: List[SignatureResult],
        timestamp: Optional[float] = None,
        node_id: str = "",
    ) -> bool:
        now = time.time()
        ts = now if timestamp is None else timestamp
        if ts > now + 300.0:
            self.rejected_operations.append(
                f"REJECT_ADD: clock drift detected for element '{element}'"
            )
            return False

        # 1. Walidacja reputacji peera (jeśli silnik reputacji jest aktywny i podano node_id)
        if self.reputation is not None and node_id:
            if not self.reputation.is_trusted(node_id):
                self.rejected_operations.append(
                    f"REJECT_ADD: untrusted node '{node_id}' for element '{element}'"
                )
                return False

        # 2. Walidacja kworum podpisów
        has_q = False
        if self.signer is not None:
            has_q = self.signer.has_quorum(signatures, operation=operation)
        else:
            # Fallback na unikalne node_id sygnatariuszy
            unique_nodes = {s.node_id for s in signatures if s.node_id}
            has_q = len(unique_nodes) >= self.quorum

        if not has_q:
            self.rejected_operations.append(
                f"REJECT_ADD: insufficient quorum ({len(signatures)}/{self.quorum}) for element '{element}'"
            )
            return False

        # 3. Zastosowanie do podkładowego CRDT
        self.underlying_set.add(element=element, timestamp=ts, node_id=node_id)
        return True

    def bft_remove(
        self,
        element: str,
        operation: str,
        signatures: List[SignatureResult],
        timestamp: Optional[float] = None,
        node_id: str = "",
    ) -> bool:
        now = time.time()
        ts = now if timestamp is None else timestamp
        if ts > now + 300.0:
            self.rejected_operations.append(
                f"REJECT_REMOVE: clock drift detected for element '{element}'"
            )
            return False

        # 1. Walidacja reputacji peera
        if self.reputation is not None and node_id:
            if not self.reputation.is_trusted(node_id):
                self.rejected_operations.append(
                    f"REJECT_REMOVE: untrusted node '{node_id}' for element '{element}'"
                )
                return False

        # 2. Walidacja kworum podpisów
        has_q = False
        if self.signer is not None:
            has_q = self.signer.has_quorum(signatures, operation=operation)
        else:
            unique_nodes = {s.node_id for s in signatures if s.node_id}
            has_q = len(unique_nodes) >= self.quorum

        if not has_q:
            self.rejected_operations.append(
                f"REJECT_REMOVE: insufficient quorum ({len(signatures)}/{self.quorum}) for element '{element}'"
            )
            return False

        # 3. Zastosowanie usunięcia do CRDT
        self.underlying_set.remove(element=element, timestamp=ts, node_id=node_id)
        return True

    def lookup(self, element: str) -> bool:
        """Deleguje sprawdzenie obecności elementu do LWWElementSet."""
        return self.underlying_set.lookup(element)

    def merge(self, other: BFTLWWSet) -> BFTLWWSet:
        """Łączy dwa zbiory BFTLWWSet w sposób deterministyczny (SEC - Strong Eventual Consistency)."""
        merged_crdt = self.underlying_set.merge(other.underlying_set)
        combined_rejected = list(dict.fromkeys(self.rejected_operations + other.rejected_operations))

        return BFTLWWSet(
            quorum=max(self.quorum, other.quorum),
            reputation=self.reputation or other.reputation,
            signer=self.signer or other.signer,
            underlying_set=merged_crdt,
            rejected_operations=combined_rejected,
        )
