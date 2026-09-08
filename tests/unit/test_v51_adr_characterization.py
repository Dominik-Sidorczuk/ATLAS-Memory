"""
Regression & Characterization tests for ADR-001: Dormant & Research Modules in ATLAS V51.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from atlas_memory.l0_dynamic.ttt_layer import TTTLayer
from atlas_memory.sync.bft_crdt import BFTLWWSet, ThresholdSigner

pytestmark = pytest.mark.dead_module


def test_ttt_layer_dormant_characterization():
    """Weryfikuje, że uśpiona warstwa L0 TTTLayer zachowuje 100% sprawności obliczeniowej."""
    ttt = TTTLayer(input_dim=64, hidden_dim=32, learning_rate=0.05, seed=42)
    assert ttt.step_count == 0

    x = np.random.randn(1, 64).astype(np.float64)
    out = ttt.forward(x)
    assert out.shape == (1, 32)

    loss, elapsed_ms = ttt.adapt_step(x)
    assert isinstance(loss, float)
    assert loss >= 0.0
    assert elapsed_ms >= 0.0
    assert ttt.step_count == 1
    assert ttt.total_energy > 0.0


@pytest.mark.asyncio
async def test_ttt_layer_async_adaptation():
    """Weryfikuje asynchroniczną adaptację TTT w wątku tła."""
    ttt = TTTLayer(input_dim=32, hidden_dim=16, learning_rate=0.01, seed=123)
    x = [0.1] * 32
    loss, ms = await ttt.adapt_step_async(x)
    assert loss >= 0.0
    assert ttt.step_count == 1


def test_bft_lww_set_research_characterization():
    """Weryfikuje, że moduł badawczy BFTLWWSet poprawnie weryfikuje kworum podpisów."""
    key1 = os.urandom(32)
    key2 = os.urandom(32)
    signer1 = ThresholdSigner(node_id="node_a", signing_key=key1, quorum=2)
    signer1.register_peer("node_b", key2)
    signer2 = ThresholdSigner(node_id="node_b", signing_key=key2, quorum=2)

    bft_set = BFTLWWSet(quorum=2, signer=signer1)
    op = "ADD:fact_quantum_encryption"

    sig1 = signer1.sign(op, algorithm="hmac_sha256")
    sig2 = signer2.sign(op, algorithm="hmac_sha256")

    # Dodanie z pojedynczym podpisem (brak kworum 2) -> odrzucone
    accepted_single = bft_set.bft_add(
        element="fact_quantum_encryption",
        operation=op,
        signatures=[sig1],
        timestamp=100.0,
        node_id="node_a",
    )
    assert accepted_single is False
    assert bft_set.lookup("fact_quantum_encryption") is False

    # Dodanie z kworum 2 -> zaakceptowane
    accepted_quorum = bft_set.bft_add(
        element="fact_quantum_encryption",
        operation=op,
        signatures=[sig1, sig2],
        timestamp=100.0,
        node_id="node_a",
    )
    assert accepted_quorum is True
    assert bft_set.lookup("fact_quantum_encryption") is True
