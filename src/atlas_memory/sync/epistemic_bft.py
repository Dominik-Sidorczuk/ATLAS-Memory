"""Compatibility alias for EpistemicReputation and BFTLWWSet (ADR-002)."""
from __future__ import annotations

from atlas_memory.sync.bft_crdt import BFTLWWSet
from atlas_memory.sync.epistemic_reputation import EpistemicReputation

__all__ = ["BFTLWWSet", "EpistemicReputation"]
