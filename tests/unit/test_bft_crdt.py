"""Unit tests for Phase V20: Epistemic Byzantine Fault Tolerant CRDT."""

from __future__ import annotations

import os

import pytest

from atlas_memory.sync import (
    BFTLWWSet,
    EpistemicReputation,
    SignatureResult,
    ThresholdSigner,
)


@pytest.fixture
def hmac_key() -> bytes:
    return os.urandom(32)


@pytest.fixture
def ed25519_seed() -> bytes:
    return os.urandom(32)


def test_signature_hmac_verify_ok(hmac_key: bytes):
    """Test poprawności tworzenia i weryfikacji podpisu HMAC-SHA256."""
    signer = ThresholdSigner(node_id="node-1", signing_key=hmac_key, quorum=2)
    op = "ADD:fact_123"
    sig = signer.sign(op, algorithm="hmac_sha256")

    assert sig.node_id == "node-1"
    assert sig.algorithm == "hmac_sha256"
    assert signer.verify(sig, op) is True


def test_signature_ed25519_verify_ok(ed25519_seed: bytes):
    """Test poprawności tworzenia i weryfikacji podpisu asymetrycznego Ed25519."""
    signer1 = ThresholdSigner(node_id="node-1", signing_key=ed25519_seed, quorum=2)
    pubkey1 = signer1.get_public_key_bytes()

    signer2 = ThresholdSigner(
        node_id="node-2",
        signing_key=os.urandom(32),
        quorum=2,
        peer_public_keys={"node-1": pubkey1},
    )

    op = "ADD:knowledge_graph_triple"
    sig = signer1.sign(op, algorithm="ed25519")

    assert signer2.verify(sig, op) is True


def test_signature_tamper_detected(hmac_key: bytes):
    """Zmiana treści operacji powoduje niepowodzenie weryfikacji podpisu."""
    signer = ThresholdSigner(node_id="node-1", signing_key=hmac_key, quorum=2)
    op = "ADD:safe_operation"
    sig = signer.sign(op)

    tampered_op = "ADD:malicious_injection"
    assert signer.verify(sig, tampered_op) is False


def test_quorum_required(hmac_key: bytes):
    """Test weryfikacji kworum (np. 2 z 3 < quorum=3 -> False; 3 z 3 -> True)."""
    signer = ThresholdSigner(node_id="coordinator", signing_key=hmac_key, quorum=3)
    k1, k2, k3 = os.urandom(32), os.urandom(32), os.urandom(32)
    signer.register_peer("peer-1", k1)
    signer.register_peer("peer-2", k2)
    signer.register_peer("peer-3", k3)

    p1 = ThresholdSigner(node_id="peer-1", signing_key=k1, quorum=3)
    p2 = ThresholdSigner(node_id="peer-2", signing_key=k2, quorum=3)
    p3 = ThresholdSigner(node_id="peer-3", signing_key=k3, quorum=3)

    op = "MUTATE:triple_1"
    sig1 = p1.sign(op)
    sig2 = p2.sign(op)
    sig3 = p3.sign(op)

    assert signer.has_quorum([sig1, sig2], operation=op) is False
    assert signer.has_quorum([sig1, sig2, sig3], operation=op) is True


def test_bft_add_rejects_without_quorum(hmac_key: bytes):
    """bft_add odrzuca operację z niewystarczającą liczbą podpisów (< quorum)."""
    signer = ThresholdSigner(node_id="node-1", signing_key=hmac_key, quorum=3)
    crdt = BFTLWWSet(quorum=3, signer=signer)

    op = "ADD:user_pref_dark_mode"
    sig1 = signer.sign(op)

    # Tylko 1 podpis z 3 wymaganych
    res = crdt.bft_add(
        element="dark_mode",
        operation=op,
        signatures=[sig1],
        timestamp=100.0,
        node_id="node-1",
    )

    assert res is False
    assert crdt.lookup("dark_mode") is False
    assert len(crdt.rejected_operations) == 1
    assert "insufficient quorum" in crdt.rejected_operations[0]


def test_bft_add_rejects_untrusted_peer(hmac_key: bytes):
    """bft_add odrzuca operację pochodzącą od peera o niskiej reputacji (R < 0.5)."""
    rep = EpistemicReputation(threshold=0.5)
    # peer z 1 sukcesem na 5 prób -> R = 0.2 < 0.5
    rep.record("bad_peer", validated=True)
    for _ in range(4):
        rep.record("bad_peer", validated=False)

    assert rep.is_trusted("bad_peer") is False

    signer = ThresholdSigner(node_id="bad_peer", signing_key=hmac_key, quorum=1)
    crdt = BFTLWWSet(quorum=1, reputation=rep, signer=signer)

    op = "ADD:hallucinated_fact"
    sig = signer.sign(op)

    res = crdt.bft_add(
        element="hallucination",
        operation=op,
        signatures=[sig],
        timestamp=100.0,
        node_id="bad_peer",
    )

    assert res is False
    assert crdt.lookup("hallucination") is False
    assert len(crdt.rejected_operations) == 1
    assert "untrusted node 'bad_peer'" in crdt.rejected_operations[0]


def test_bft_add_accepts_trusted_with_quorum(hmac_key: bytes):
    """bft_add akceptuje operację przy zachowaniu kworum i zaufanego peera."""
    rep = EpistemicReputation(threshold=0.5)
    rep.record("good_peer", validated=True)

    k1, k2, k3 = os.urandom(32), os.urandom(32), os.urandom(32)
    coordinator_signer = ThresholdSigner(node_id="good_peer", signing_key=k1, quorum=3)
    coordinator_signer.register_peer("peer-2", k2)
    coordinator_signer.register_peer("peer-3", k3)

    p2 = ThresholdSigner(node_id="peer-2", signing_key=k2, quorum=3)
    p3 = ThresholdSigner(node_id="peer-3", signing_key=k3, quorum=3)

    crdt = BFTLWWSet(quorum=3, reputation=rep, signer=coordinator_signer)

    op = "ADD:verified_triple"
    sig1 = coordinator_signer.sign(op)
    sig2 = p2.sign(op)
    sig3 = p3.sign(op)

    res = crdt.bft_add(
        element="verified_triple",
        operation=op,
        signatures=[sig1, sig2, sig3],
        timestamp=100.0,
        node_id="good_peer",
    )

    assert res is True
    assert crdt.lookup("verified_triple") is True
    assert len(crdt.rejected_operations) == 0


def test_bft_remove_requires_quorum(hmac_key: bytes):
    """bft_remove wymaga kworum podpisów, inaczej element nie zostaje usunięty."""
    signer = ThresholdSigner(node_id="node-1", signing_key=hmac_key, quorum=2)
    crdt = BFTLWWSet(quorum=2, signer=signer)

    # Dodanie z kworum 2 (samo podpisane przez 2 różne węzły)
    k2 = os.urandom(32)
    signer.register_peer("node-2", k2)
    p2 = ThresholdSigner(node_id="node-2", signing_key=k2, quorum=2)

    op_add = "ADD:key_item"
    sig_add_1 = signer.sign(op_add)
    sig_add_2 = p2.sign(op_add)

    crdt.bft_add("key_item", op_add, [sig_add_1, sig_add_2], timestamp=10.0, node_id="node-1")
    assert crdt.lookup("key_item") is True

    # Próba usunięcia z tylko 1 podpisem
    op_rem = "REMOVE:key_item"
    sig_rem_1 = signer.sign(op_rem)
    rem_res = crdt.bft_remove("key_item", op_rem, [sig_rem_1], timestamp=20.0, node_id="node-1")

    assert rem_res is False
    assert crdt.lookup("key_item") is True  # Nadal istnieje!

    # Usunięcie z 2 podpisami
    sig_rem_2 = p2.sign(op_rem)
    rem_res2 = crdt.bft_remove("key_item", op_rem, [sig_rem_1, sig_rem_2], timestamp=20.0, node_id="node-1")
    assert rem_res2 is True
    assert crdt.lookup("key_item") is False


def test_sec_convergence():
    """Strong Eventual Consistency (SEC): dwa węzły po wymianie poprawnych operacji zbiegają się do identycznego stanu."""
    crdt_a = BFTLWWSet(quorum=2)
    crdt_b = BFTLWWSet(quorum=2)

    sigs = [
        SignatureResult(node_id="n1", signature=b"1"),
        SignatureResult(node_id="n2", signature=b"2"),
    ]

    crdt_a.bft_add("item1", "ADD:item1", sigs, timestamp=1.0, node_id="n1")
    crdt_b.bft_add("item2", "ADD:item2", sigs, timestamp=2.0, node_id="n2")

    merged_a = crdt_a.merge(crdt_b)
    merged_b = crdt_b.merge(crdt_a)

    assert merged_a.lookup("item1") is True
    assert merged_a.lookup("item2") is True
    assert merged_b.lookup("item1") is True
    assert merged_b.lookup("item2") is True
    assert merged_a.underlying_set.model_dump() == merged_b.underlying_set.model_dump()


def test_byzantine_agent_blocked(hmac_key: bytes):
    """Byzantyjski agent próbuje wstrzyknąć sprzeczny fakt bez kworum -> odrzucony i odnotowany."""
    rep = EpistemicReputation(threshold=0.6)
    rep.record("byzantine_agent", validated=False)
    rep.record("byzantine_agent", validated=False)

    crdt = BFTLWWSet(quorum=3, reputation=rep)

    sig_fake = SignatureResult(node_id="byzantine_agent", signature=b"forged")
    res = crdt.bft_add(
        element="poisoned_knowledge",
        operation="ADD:poison",
        signatures=[sig_fake],
        timestamp=50.0,
        node_id="byzantine_agent",
    )

    assert res is False
    assert crdt.lookup("poisoned_knowledge") is False
    assert len(crdt.rejected_operations) > 0


def test_reputation_scoring_and_disconnect():
    """Test funkcji scoringu epistemicznego i odłączania niezaufanych peerów."""
    rep = EpistemicReputation(threshold=0.5)

    rep.record("peer_a", validated=True)
    rep.record("peer_a", validated=True)
    rep.record("peer_b", validated=False)
    rep.record("peer_b", validated=True)
    rep.record("peer_c", validated=False)

    assert rep.score("peer_a") == 1.0
    assert rep.score("peer_b") == 0.5
    assert rep.score("peer_c") == 0.0
    assert rep.score("peer_unknown") == 0.0

    assert rep.is_trusted("peer_a") is True
    assert rep.is_trusted("peer_b") is True
    assert rep.is_trusted("peer_c") is False

    untrusted = rep.disconnect_untrusted(["peer_a", "peer_b", "peer_c", "peer_unknown"])
    assert "peer_c" in untrusted
    assert "peer_unknown" in untrusted
    assert "peer_a" not in untrusted
    assert "peer_b" not in untrusted
