"""Compatibility alias for ThresholdSigner (ADR-002)."""
from __future__ import annotations

from atlas_memory.sync.threshold_signatures import SignatureResult, ThresholdSigner

__all__ = ["ThresholdSigner", "SignatureResult"]
