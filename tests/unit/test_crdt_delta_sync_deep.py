from __future__ import annotations

import os

from atlas_memory.sync import (
    BFTLWWSet,
    EpistemicReputation,
    SignatureResult,
    ThresholdSigner,
)


def test_bft_crdt_multi_node_quorum_convergence():
    key1 = os.urandom(32)
    key2 = os.urandom(32)
    key3 = os.urandom(32)

    coord = ThresholdSigner(node_id="coord", signing_key=key1, quorum=2)
    coord.register_peer("peer-2", key2)
    coord.register_peer("peer-3", key3)

    p2 = ThresholdSigner(node_id="peer-2", signing_key=key2, quorum=2)
    p3 = ThresholdSigner(node_id="peer-3", signing_key=key3, quorum=2)

    op = "ADD:cluster_master_node"
    sig2 = p2.sign(op)
    sig3 = p3.sign(op)

    assert coord.has_quorum([sig2], operation=op) is False
    assert coord.has_quorum([sig2, sig3], operation=op) is True

    rep = EpistemicReputation(threshold=0.5)
    rep.record("peer-2", validated=True)
    crdt = BFTLWWSet(quorum=2, signer=coord, reputation=rep)

    success = crdt.bft_add(
        element="cluster_master_node",
        operation=op,
        signatures=[sig2, sig3],
        timestamp=100.0,
        node_id="peer-2",
    )
    assert success is True
    assert crdt.lookup("cluster_master_node") is True


def test_epistemic_reputation_slashes_byzantine_nodes():
    rep = EpistemicReputation(threshold=0.5)

    # Node with 100% successes -> trusted
    for _ in range(5):
        rep.record("node-honest", validated=True)
    assert rep.is_trusted("node-honest") is True

    # Node with mostly failures -> untrusted
    rep.record("node-malicious", validated=True)
    for _ in range(4):
        rep.record("node-malicious", validated=False)
    assert rep.is_trusted("node-malicious") is False


def test_threshold_signer_rejects_corrupted_signature_payload():
    signer = ThresholdSigner(node_id="node-1", signing_key=os.urandom(32), quorum=2)
    op = "ADD:entity_critical_state"
    sig = signer.sign(op)

    # Forged op
    assert signer.verify(sig, "ADD:entity_modified_state") is False

    # Corrupted signature bytes
    corrupted_sig = SignatureResult(
        node_id=sig.node_id,
        signature=sig.signature[:-2] + b"\xff\xff",
        algorithm=sig.algorithm,
    )
    assert signer.verify(corrupted_sig, op) is False
