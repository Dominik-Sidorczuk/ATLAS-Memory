"""
Unit Tests for Heterogeneous Multi-Agent Memory Synchronization.
"""
from __future__ import annotations

"""Unit tests for Heterogeneous Multi-Agent Memory Sync (CRDT + E2E Crypto + Gossip)."""


import pytest
from cryptography.exceptions import InvalidTag

from atlas_memory.sync.crdt import DeltaCRDT, LWWElementSet, VectorClock
from atlas_memory.sync.crypto import SyncCrypto
from atlas_memory.sync.protocol import GossipProtocol


def test_vector_clock_increment_and_merge() -> None:
    vc1 = VectorClock()
    vc2 = VectorClock()

    vc1.increment("node_a")
    vc1.increment("node_a")
    vc1.increment("node_b")

    vc2.increment("node_b")
    vc2.increment("node_b")
    vc2.increment("node_c")

    merged = vc1.merge(vc2)
    assert merged.clocks == {"node_a": 2, "node_b": 2, "node_c": 1}
    assert vc1.clocks == {"node_a": 2, "node_b": 2, "node_c": 1}


def test_vector_clock_happens_before() -> None:
    a = VectorClock(clocks={"node_1": 1, "node_2": 0})
    b = VectorClock(clocks={"node_1": 1, "node_2": 1})
    c = VectorClock(clocks={"node_1": 2, "node_2": 1})

    assert a.happens_before(b) is True
    assert b.happens_before(c) is True
    assert a.happens_before(c) is True
    assert c.happens_before(a) is False
    assert a.happens_before(a) is False


def test_vector_clock_concurrent() -> None:
    a = VectorClock(clocks={"node_1": 2, "node_2": 0})
    b = VectorClock(clocks={"node_1": 1, "node_2": 1})

    assert a.concurrent(b) is True
    assert b.concurrent(a) is True
    assert a.happens_before(b) is False
    assert b.happens_before(a) is False


def test_lww_add_remove_resolution() -> None:
    lww = LWWElementSet()
    lww.add("fact_alpha", timestamp=1.0, node_id="node_a")
    assert lww.lookup("fact_alpha") is True

    lww.remove("fact_alpha", timestamp=2.0, node_id="node_b")
    assert lww.lookup("fact_alpha") is False


def test_lww_conflict_higher_timestamp_wins() -> None:
    lww = LWWElementSet()
    # Add earlier, remove later -> removed
    lww.add("fact_beta", timestamp=1.0, node_id="node_a")
    lww.remove("fact_beta", timestamp=2.0, node_id="node_b")
    assert lww.lookup("fact_beta") is False

    # Add with even higher timestamp -> added
    lww.add("fact_beta", timestamp=3.0, node_id="node_a")
    assert lww.lookup("fact_beta") is True

    # Same timestamp tie-break: node_b > node_a
    lww_tie = LWWElementSet()
    lww_tie.add("fact_gamma", timestamp=10.0, node_id="node_a")
    lww_tie.remove("fact_gamma", timestamp=10.0, node_id="node_b")
    # removes meta (10.0, "node_b") > adds meta (10.0, "node_a") -> lookup is False
    assert lww_tie.lookup("fact_gamma") is False


def test_crypto_encrypt_decrypt_roundtrip() -> None:
    key = SyncCrypto.generate_key()
    crypto = SyncCrypto(key)

    plaintext = b"Secret multi-agent memory payload"
    aad = b"AAD_VECTOR_CLOCK_123"

    encrypted = crypto.encrypt(plaintext, aad=aad)
    assert len(encrypted) >= 12 + len(plaintext)
    assert encrypted != plaintext

    decrypted = crypto.decrypt(encrypted, aad=aad)
    assert decrypted == plaintext


def test_crypto_tampering_detected() -> None:
    key = SyncCrypto.generate_key()
    crypto = SyncCrypto(key)

    plaintext = b"Tamper-sensitive data"
    aad = b"AAD"
    encrypted = bytearray(crypto.encrypt(plaintext, aad=aad))

    # Flip one bit in the ciphertext / tag region
    encrypted[-1] ^= 0xFF

    with pytest.raises(InvalidTag):
        crypto.decrypt(bytes(encrypted), aad=aad)


def test_crypto_aad_mismatch() -> None:
    key = SyncCrypto.generate_key()
    crypto = SyncCrypto(key)

    plaintext = b"Replay protection test"
    encrypted = crypto.encrypt(plaintext, aad=b"valid_clock_v1")

    with pytest.raises(InvalidTag):
        crypto.decrypt(encrypted, aad=b"altered_clock_v2")


def test_gossip_roundtrip() -> None:
    shared_key = SyncCrypto.generate_key()

    crypto_a = SyncCrypto(shared_key)
    crdt_a = DeltaCRDT("agent_alice")
    gossip_a = GossipProtocol("agent_alice", crdt_a, crypto_a)

    crypto_b = SyncCrypto(shared_key)
    crdt_b = DeltaCRDT("agent_bob")
    gossip_b = GossipProtocol("agent_bob", crdt_b, crypto_b)

    # Alice adds a memory triple
    set_a = crdt_a.get_set("facts")
    set_a.add("user_likes_rust", timestamp=1.0, node_id="agent_alice")
    crdt_a.clock.increment("agent_alice")

    # Bob registers Alice
    gossip_b.register_peer("agent_alice")

    # Alice creates sync request for Bob
    sync_req_bytes = gossip_a.create_sync_request("agent_bob")

    # Bob processes Alice's sync request
    delta_applied = gossip_b.process_sync_response(sync_req_bytes)

    assert delta_applied.source_node == "agent_alice"
    assert "agent_alice" in gossip_b.peers
    assert crdt_b.get_set("facts").lookup("user_likes_rust") is True
    assert crdt_b.clock.clocks.get("agent_alice") == 1
"""Unit tests for Gossip Transport and CRDT Tombstone GC."""

from atlas_memory.sync.transport import (
    InMemoryGossipTransport,
    UDPGossipTransport,
    create_transport,
)


@pytest.mark.asyncio
async def test_inmemory_transport_roundtrip() -> None:
    """Tests InMemoryGossipTransport send and receive across peer queues."""
    t1 = InMemoryGossipTransport()
    t2 = InMemoryGossipTransport()

    # Register t2 as peer "node-2" in t1
    t1.register_peer_transport("node-2", t2)

    payload = b"test-delta-payload-12345"
    await t1.send("node-2", payload)

    received = await t2.receive(timeout=1.0)
    assert received == payload

    # Test timeout on empty queue
    empty = await t1.receive(timeout=0.05)
    assert empty is None


@pytest.mark.asyncio
async def test_udp_transport_send_receive() -> None:
    """Tests UDPGossipTransport datagram transmission on localhost with ephemeral ports."""
    t1 = UDPGossipTransport(local_port=0, host="127.0.0.1")
    t2 = UDPGossipTransport(local_port=0, host="127.0.0.1")

    await t1.start()
    await t2.start()

    port1 = t1.get_local_port()
    port2 = t2.get_local_port()

    assert port1 > 0
    assert port2 > 0
    assert port1 != port2

    # Map node-2 in t1 and node-1 in t2
    t1.register_peer("node-2", "127.0.0.1", port2)
    t2.register_peer("node-1", "127.0.0.1", port1)

    payload = b"udp-gossip-delta-bytes"
    await t1.send("node-2", payload)

    received = await t2.receive(timeout=2.0)
    assert received == payload

    await t1.close()
    await t2.close()


@pytest.mark.asyncio
async def test_gossip_with_transport_send() -> None:
    """Tests GossipProtocol initialized with transport and invoking send_sync."""
    key = SyncCrypto.generate_key()
    crypto = SyncCrypto(key)
    crdt1 = DeltaCRDT("agent-1")
    crdt2 = DeltaCRDT("agent-2")

    t1 = create_transport(kind="inmemory")
    t2 = create_transport(kind="inmemory")

    if isinstance(t1, InMemoryGossipTransport) and isinstance(t2, InMemoryGossipTransport):
        t1.register_peer_transport("agent-2", t2)
        t2.register_peer_transport("agent-1", t1)

    proto1 = GossipProtocol("agent-1", crdt1, crypto, transport=t1)
    proto2 = GossipProtocol("agent-2", crdt2, crypto, transport=t2)

    proto1.register_peer("agent-2")
    crdt1.get_set("facts").add("fact-1", timestamp=100.0, node_id="agent-1")

    # Send sync via transport
    await proto1.send_sync("agent-2")

    received_bytes = await t2.receive(timeout=1.0)
    assert received_bytes is not None

    delta = proto2.process_sync_response(received_bytes)
    assert delta.source_node == "agent-1"
    assert proto2.crdt.get_set("facts").lookup("fact-1") is True


def test_tombstone_gc_removes_old() -> None:
    """Tests that LWWElementSet GC purges tombstones older than TTL."""
    s = LWWElementSet(tombstone_ttl_seconds=1.0)
    s.add("item-1", timestamp=0.0, node_id="n1")
    s.remove("item-1", timestamp=0.0, node_id="n1")

    assert "item-1" in s.removes
    assert s.lookup("item-1") is False

    # GC at now=10.0 (cutoff is 9.0)
    purged = s.gc(now=10.0)
    assert purged == 1
    assert len(s.removes) == 0
    assert "item-1" not in s.adds


def test_tombstone_gc_keeps_recent() -> None:
    """Tests that LWWElementSet GC retains tombstones younger than TTL."""
    s = LWWElementSet(tombstone_ttl_seconds=1.0)
    s.add("item-recent", timestamp=9.5, node_id="n1")
    s.remove("item-recent", timestamp=9.5, node_id="n1")

    # GC at now=10.0 (cutoff is 9.0 -> 9.5 > 9.0, should keep)
    purged = s.gc(now=10.0)
    assert purged == 0
    assert len(s.removes) == 1
    assert s.lookup("item-recent") is False


def test_tombstone_gc_disabled_by_default() -> None:
    """Tests that GC is a no-op when tombstone_ttl_seconds is None."""
    s = LWWElementSet()  # Default None
    s.add("item-default", timestamp=0.0, node_id="n1")
    s.remove("item-default", timestamp=0.0, node_id="n1")

    purged = s.gc(now=1000.0)
    assert purged == 0
    assert len(s.removes) == 1
