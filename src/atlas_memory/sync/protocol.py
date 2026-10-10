"""Gossip Protocol and Peer Synchronization for Multi-Agent Memory Networks (V83 Refactor).

Enhanced with StateSyncBridge storage integration and PeerIdentity cryptographic attestation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from atlas_memory.sync.bridge import StateSyncBridge
from atlas_memory.sync.crdt import DeltaCRDT, SyncDelta, VectorClock
from atlas_memory.sync.crypto import SyncCrypto
from atlas_memory.sync.peer_identity import PeerIdentity
from atlas_memory.sync.transport import GossipTransport

logger = logging.getLogger(__name__)


class PeerStatus(BaseModel):
    """Status metadata for a known sync peer."""

    peer_id: str = Field(...)
    vector_clock: VectorClock = Field(default_factory=VectorClock)
    last_seen: float = Field(default_factory=time.time)
    rtt_ms: Optional[float] = Field(default=None)


class GossipProtocol:
    """Decentralized gossip protocol engine for agent-to-agent delta sync."""

    def __init__(
        self,
        local_node_id: str,
        crdt: DeltaCRDT,
        crypto: SyncCrypto,
        transport: Optional[GossipTransport] = None,
        bridge: Optional[StateSyncBridge] = None,
        identity: Optional[PeerIdentity] = None,
        reputation: Optional[Any] = None,
    ) -> None:
        self.local_node_id: str = str(local_node_id)
        self.crdt: DeltaCRDT = crdt
        self.crypto: SyncCrypto = crypto
        self.transport: Optional[GossipTransport] = transport
        self.bridge: Optional[StateSyncBridge] = bridge
        self.identity: Optional[PeerIdentity] = identity
        self.reputation: Optional[Any] = reputation
        self.peers: Dict[str, PeerStatus] = {}
        self.peer_cursors: Dict[str, int] = {}

    def register_peer(self, peer_id: str, vector_clock: Optional[VectorClock] = None) -> PeerStatus:
        """Registers or updates a peer in the known peer table."""
        vc = vector_clock or VectorClock()
        status = PeerStatus(peer_id=peer_id, vector_clock=vc, last_seen=time.time())
        self.peers[peer_id] = status
        return status

    def create_sync_request(self, target_peer_id: str, since_seq: Optional[int] = None) -> bytes:
        """Creates an encrypted JSON-RPC sync payload to send to target peer."""
        if self.bridge is not None:
            if since_seq is None:
                since_seq = self.peer_cursors.get(target_peer_id, 0)
            audit_delta = self.bridge.export_delta_from_audit(since_seq=since_seq)
            max_seq = audit_delta.get("max_seq", since_seq)
            self.peer_cursors[target_peer_id] = max_seq
            delta = SyncDelta(
                source_node=self.local_node_id,
                vector_clock=self.bridge.clock,
                payload=audit_delta,
            )
            local_clocks = dict(self.bridge.clock.clocks)
        else:
            peer = self.peers.get(target_peer_id)
            since_vc = peer.vector_clock if peer else None
            delta = self.crdt.export_delta(since_vc)
            local_clocks = dict(self.crdt.clock.clocks)

        raw_payload = delta.model_dump_json().encode("utf-8")

        # Optional Ed25519 digital signature of payload
        signature_hex = ""
        if self.identity is not None:
            signature_hex = self.identity.sign(raw_payload).hex()

        # Serialized local clock as AAD to prevent replay
        aad = json.dumps(local_clocks, sort_keys=True).encode("utf-8")
        encrypted_blob = self.crypto.encrypt(raw_payload, aad=aad)

        envelope: Dict[str, Any] = {
            "source_node": self.local_node_id,
            "target_node": target_peer_id,
            "aad": aad.decode("utf-8"),
            "ciphertext_hex": encrypted_blob.hex(),
        }
        if signature_hex:
            envelope["signature_hex"] = signature_hex

        return json.dumps(envelope).encode("utf-8")

    async def send_sync(self, target_peer_id: str) -> None:
        """Sends an encrypted sync request to the target peer via the configured transport."""
        if self.transport is None:
            raise RuntimeError("Cannot send sync: no GossipTransport configured")
        payload = self.create_sync_request(target_peer_id)
        await self.transport.send(target_peer_id, payload)

    def _verify_envelope_signature(self, envelope: Dict[str, Any], source: str, decrypted_bytes: bytes) -> None:
        """Enforces cryptographic signature check and records epistemic reputation."""
        if self.identity is None:
            return
        sig_hex = envelope.get("signature_hex")
        if not sig_hex:
            logger.warning("Sync request from node %s rejected: missing required signature", source)
            if self.reputation is not None and hasattr(self.reputation, "record"):
                self.reputation.record(source, validated=False)
            raise PermissionError(f"Missing cryptographic signature from node '{source}'")

        try:
            is_valid = self.identity.verify(source, decrypted_bytes, bytes.fromhex(sig_hex))
        except Exception as exc:
            logger.warning("Sync request from node %s rejected on signature error: %s", source, exc)
            if self.reputation is not None and hasattr(self.reputation, "record"):
                self.reputation.record(source, validated=False)
            raise PermissionError(f"Cryptographic signature verification failed for node '{source}': {exc}") from exc

        if not is_valid:
            logger.warning("Sync request from node %s rejected: invalid cryptographic signature", source)
            if self.reputation is not None and hasattr(self.reputation, "record"):
                self.reputation.record(source, validated=False)
            raise PermissionError(f"Invalid cryptographic signature from node '{source}'")

        if self.reputation is not None and hasattr(self.reputation, "record"):
            self.reputation.record(source, validated=True)

    async def process_sync_response_async(self, request_bytes: bytes) -> SyncDelta:
        """Decrypts, verifies, and applies incoming sync request/response to storage bridge or CRDT."""
        envelope = json.loads(request_bytes.decode("utf-8"))
        aad = envelope["aad"].encode("utf-8")
        blob = bytes.fromhex(envelope["ciphertext_hex"])

        decrypted_bytes = self.crypto.decrypt(blob, aad=aad)
        delta_dict = json.loads(decrypted_bytes.decode("utf-8"))
        delta = SyncDelta.model_validate(delta_dict)

        source = envelope.get("source_node") or delta.source_node

        # Cryptographic gate: reject mutations if signature verification fails
        self._verify_envelope_signature(envelope, source, decrypted_bytes)

        # Apply mutations directly to storage bridge (if available) or in-memory CRDT
        if self.bridge is not None and ("mutations" in delta.payload or "mutations" in delta.payload.get("payload", {})):
            mutations_payload = delta.payload if "mutations" in delta.payload else delta.payload.get("payload", {})
            await self.bridge.apply_remote_delta(mutations_payload)
        else:
            self.crdt.apply_delta(delta)

        # Update peer status if known
        if source:
            if source in self.peers:
                self.peers[source].vector_clock.merge(delta.vector_clock)
                self.peers[source].last_seen = time.time()
            else:
                self.register_peer(source, delta.vector_clock)

        return delta

    def process_sync_response(self, request_bytes: bytes) -> SyncDelta:
        """Synchronous wrapper for processing sync responses."""
        envelope = json.loads(request_bytes.decode("utf-8"))
        aad = envelope["aad"].encode("utf-8")
        blob = bytes.fromhex(envelope["ciphertext_hex"])

        decrypted_bytes = self.crypto.decrypt(blob, aad=aad)
        delta_dict = json.loads(decrypted_bytes.decode("utf-8"))
        delta = SyncDelta.model_validate(delta_dict)

        source = envelope.get("source_node") or delta.source_node

        # Cryptographic gate: reject mutations if signature verification fails
        self._verify_envelope_signature(envelope, source, decrypted_bytes)

        if self.bridge is not None and ("mutations" in delta.payload or "mutations" in delta.payload.get("payload", {})):
            mutations_payload = delta.payload if "mutations" in delta.payload else delta.payload.get("payload", {})
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.bridge.apply_remote_delta(mutations_payload))
            except RuntimeError:
                asyncio.run(self.bridge.apply_remote_delta(mutations_payload))
        else:
            self.crdt.apply_delta(delta)

        if source:
            if source in self.peers:
                self.peers[source].vector_clock.merge(delta.vector_clock)
                self.peers[source].last_seen = time.time()
            else:
                self.register_peer(source, delta.vector_clock)

        return delta

    def detect_conflicts(self) -> List[Dict[str, Any]]:
        """Detects keys or elements that have concurrent or divergent values across peers."""
        conflicts: List[Dict[str, Any]] = []
        clock_source = self.bridge.clock if self.bridge is not None else self.crdt.clock
        for peer_id, peer in self.peers.items():
            if clock_source.concurrent(peer.vector_clock):
                conflicts.append(
                    {
                        "peer_id": peer_id,
                        "type": "concurrent_clock",
                        "local_clock": clock_source.clocks,
                        "peer_clock": peer.vector_clock.clocks,
                    }
                )
        return conflicts
