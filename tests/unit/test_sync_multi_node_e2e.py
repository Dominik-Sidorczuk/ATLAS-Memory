"""Multi-Node E2E & Byzantine Security Tests for ATLAS V83 Sync Hardening (F0).

Proves the 5 critical multi-node invariants:
1. Cryptographic Signature Gate: Unsigned or invalid signature from untrusted node is hard-rejected (PermissionError, 0 KV writes).
2. Pre-Shared Key (PSK) Decoupling: Separate crypto instances across processes with shared PSK succeed; wrong PSK fails.
3. Vector & Graph Re-indexing: Replicated peer facts hydrate Qdrant vector index and Kùzu graph store, enabling recall.
4. Sequence Cursor Progression: export_delta advances peer cursor so identical seq ranges are never re-exported.
5. Tombstone TTL Immunity: Default None TTL prevents premature tombstone expiration and resurrected deletes.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest
from cryptography.exceptions import InvalidTag

from atlas_memory.cognitive.write_path import EpistemicWritePath
from atlas_memory.l2_semantic.kuzu_graph import KuzuGraphStore
from atlas_memory.l2_semantic.kv_store import VerifiedKVStore
from atlas_memory.l2_semantic.qdrant_store import QdrantVectorStore
from atlas_memory.sync.bridge import StateSyncBridge
from atlas_memory.sync.crdt import CompactLWWSet, DeltaCRDT
from atlas_memory.sync.crypto import SyncCrypto
from atlas_memory.sync.epistemic_reputation import EpistemicReputation
from atlas_memory.sync.peer_identity import PeerIdentity
from atlas_memory.sync.protocol import GossipProtocol


@pytest.mark.asyncio
async def test_signature_gate_rejects_untrusted_or_invalid_signature():
    """Invariant 1: Mutations from untrusted or invalidly signed envelopes MUST be hard-rejected with zero KV writes."""
    with tempfile.TemporaryDirectory() as tmp_b:
        kv_b = VerifiedKVStore(db_path=str(Path(tmp_b) / "atlas_b.db"))
        wp_b = EpistemicWritePath(kv_store=kv_b)
        bridge_b = StateSyncBridge(kv_store=kv_b, write_path=wp_b, node_id="node_b")

        psk = "shared_cluster_key_alpha_beta_gamma_12345"
        crypto_a = SyncCrypto.from_psk(psk)
        crypto_b = SyncCrypto.from_psk(psk)

        id_a = PeerIdentity(node_id="node_a")
        id_b = PeerIdentity(node_id="node_b")
        id_c = PeerIdentity(node_id="node_c_malicious")

        # Node B only trusts Node A
        id_b.register_peer("node_a", id_a.get_public_key_bytes())

        reputation_b = EpistemicReputation()

        proto_b = GossipProtocol(
            local_node_id="node_b",
            crdt=DeltaCRDT("node_b"),
            crypto=crypto_b,
            bridge=bridge_b,
            identity=id_b,
            reputation=reputation_b,
        )

        # Create malicious payload from untrusted node_c
        proto_c = GossipProtocol(
            local_node_id="node_c_malicious",
            crdt=DeltaCRDT("node_c_malicious"),
            crypto=crypto_a,
            identity=id_c,
        )
        proto_c.crdt.get_set("secrets").add("exploit", timestamp=time.time(), node_id="node_c_malicious")
        bad_req = proto_c.create_sync_request("node_b")

        # Must raise PermissionError and record reputation penalty
        with pytest.raises(PermissionError):
            await proto_b.process_sync_response_async(bad_req)

        # Zero writes committed to Node B KV store
        assert kv_b.get_sync("secrets") is None
        assert reputation_b.score("node_c_malicious") == 0.0

        kv_b.close_sync()


@pytest.mark.asyncio
async def test_psk_decoupling_across_instances():
    """Invariant 2: Independent SyncCrypto instances derived from shared PSK succeed; mismatch fails with InvalidTag."""
    psk_valid = "super_secure_cluster_pre_shared_key_2026"
    psk_wrong = "totally_different_unauthorized_key_9999"

    crypto_sender = SyncCrypto.from_psk(psk_valid)
    crypto_receiver = SyncCrypto.from_psk(psk_valid)
    crypto_attacker = SyncCrypto.from_psk(psk_wrong)

    plaintext = b'{"command": "replicate_state", "value": 42}'
    aad = b"authenticated_header_data"

    # Sender encrypts
    encrypted = crypto_sender.encrypt(plaintext, aad=aad)

    # Valid receiver decrypts successfully
    decrypted = crypto_receiver.decrypt(encrypted, aad=aad)
    assert decrypted == plaintext

    # Attacker with wrong PSK fails with InvalidTag
    with pytest.raises(InvalidTag):
        crypto_attacker.decrypt(encrypted, aad=aad)


@pytest.mark.asyncio
async def test_vector_and_graph_reindexing_on_apply():
    """Invariant 3: apply_remote_delta must hydrate both Qdrant vector index and Kùzu graph store."""
    with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
        kv_a = VerifiedKVStore(db_path=str(Path(tmp_a) / "atlas_a.db"))
        wp_a = EpistemicWritePath(kv_store=kv_a)
        bridge_a = StateSyncBridge(kv_store=kv_a, write_path=wp_a, node_id="node_a")

        kv_b = VerifiedKVStore(db_path=str(Path(tmp_b) / "atlas_b.db"))
        vec_b = QdrantVectorStore(location=":memory:")
        graph_b = KuzuGraphStore(db_path=str(Path(tmp_b) / "kuzu"))

        wp_b = EpistemicWritePath(
            kv_store=kv_b,
            vector_store=vec_b,
            graph_store=graph_b,
        )
        bridge_b = StateSyncBridge(kv_store=kv_b, write_path=wp_b, node_id="node_b")

        psk = "test_mesh_cluster_psk_999"
        crypto_a = SyncCrypto.from_psk(psk)
        crypto_b = SyncCrypto.from_psk(psk)

        id_a = PeerIdentity(node_id="node_a")
        id_b = PeerIdentity(node_id="node_b")
        id_a.register_peer("node_b", id_b.get_public_key_bytes())
        id_b.register_peer("node_a", id_a.get_public_key_bytes())

        proto_a = GossipProtocol(
            local_node_id="node_a",
            crdt=DeltaCRDT("node_a"),
            crypto=crypto_a,
            bridge=bridge_a,
            identity=id_a,
        )
        proto_b = GossipProtocol(
            local_node_id="node_b",
            crdt=DeltaCRDT("node_b"),
            crypto=crypto_b,
            bridge=bridge_b,
            identity=id_b,
        )

        # 1. Node A writes structured fact
        kv_a.set_sync("fact:database:port", "5432", confidence=0.95, reason="cluster_init")

        # 2. Node A creates sync request
        req = proto_a.create_sync_request("node_b")

        # 3. Node B applies sync response
        await proto_b.process_sync_response_async(req)

        # Verify KV in B
        rec_kv = kv_b.get_sync("fact:database:port")
        assert rec_kv is not None
        assert rec_kv["value"] == "5432"

        # Verify Vector in B
        assert vec_b.count_by_upsert_key_sync("fact:database:port") >= 1

        # Verify Graph in B
        rels = await graph_b.get_node_relations("fact")
        assert len(rels) >= 1
        assert any(r.get("predicate") == "database" or r.get("object") == "5432" for r in rels)

        kv_a.close_sync()
        kv_b.close_sync()
        graph_b.close()


@pytest.mark.asyncio
async def test_sequence_cursor_progression():
    """Invariant 4: Subsequent export_delta calls must advance the cursor and avoid infinite re-export."""
    with tempfile.TemporaryDirectory() as tmp_a:
        kv_a = VerifiedKVStore(db_path=str(Path(tmp_a) / "atlas_a.db"))
        wp_a = EpistemicWritePath(kv_store=kv_a)
        bridge_a = StateSyncBridge(kv_store=kv_a, write_path=wp_a, node_id="node_a")

        crypto = SyncCrypto.from_psk("cursor_test_psk")
        proto_a = GossipProtocol(
            local_node_id="node_a",
            crdt=DeltaCRDT("node_a"),
            crypto=crypto,
            bridge=bridge_a,
        )

        # 1. First write
        kv_a.set_sync("config:timeout", "30", confidence=0.9)

        # 2. First export: cursor at 0 -> returns seq [1], advances peer cursor to 1
        req1 = proto_a.create_sync_request("peer_b")
        env1 = proto_a.crypto.decrypt(bytes.fromhex(__import__("json").loads(req1.decode())["ciphertext_hex"]), aad=__import__("json").loads(req1.decode())["aad"].encode())
        delta1 = __import__("json").loads(env1.decode())
        assert len(delta1["payload"]["mutations"]) == 1

        # 3. Second export with no new writes: cursor is now at 1 -> returns 0 mutations!
        req2 = proto_a.create_sync_request("peer_b")
        env2 = proto_a.crypto.decrypt(bytes.fromhex(__import__("json").loads(req2.decode())["ciphertext_hex"]), aad=__import__("json").loads(req2.decode())["aad"].encode())
        delta2 = __import__("json").loads(env2.decode())
        assert len(delta2["payload"]["mutations"]) == 0

        # 4. Third write: advances max_seq to 2
        kv_a.set_sync("config:retries", "5", confidence=0.9)
        req3 = proto_a.create_sync_request("peer_b")
        env3 = proto_a.crypto.decrypt(bytes.fromhex(__import__("json").loads(req3.decode())["ciphertext_hex"]), aad=__import__("json").loads(req3.decode())["aad"].encode())
        delta3 = __import__("json").loads(env3.decode())
        assert len(delta3["payload"]["mutations"]) == 1
        assert delta3["payload"]["mutations"][0]["key"] == "config:retries"

        kv_a.close_sync()


def test_tombstone_ttl_immunity():
    """Invariant 5: Default None tombstone TTL preserves deletions indefinitely across restarts and sweeps."""
    lww = CompactLWWSet()  # Default tombstone_ttl_seconds is None

    # Add and remove
    lww.add("transient_key", timestamp=100.0, node_id="node_1")
    lww.remove("transient_key", timestamp=105.0, node_id="node_1")
    assert lww.lookup("transient_key") is False

    # Simulate massive passage of time (30 days = 2592000s)
    purged = lww.auto_gc(now=105.0 + 2592000.0)
    assert purged == 0  # Zero tombstones purged
    assert lww.lookup("transient_key") is False  # Key does not resurrect
