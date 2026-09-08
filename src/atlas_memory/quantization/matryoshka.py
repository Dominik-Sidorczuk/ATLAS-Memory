"""
Matryoshka Representation Learning (MRL) canonical module alias (ADR-002).

Re-exports MatryoshkaEmbedding from matryoshka_wrapper for clean architectural parity.
"""
from __future__ import annotations

from atlas_memory.quantization.matryoshka_wrapper import MatryoshkaEmbedding

__all__ = ["MatryoshkaEmbedding"]
