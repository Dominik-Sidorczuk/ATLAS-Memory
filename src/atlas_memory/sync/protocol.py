"""Gossip Protocol and Peer Synchronization for Multi-Agent Memory Networks."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from atlas_memory.sync.crdt import DeltaCRDT, SyncDelta, VectorClock
from atlas_memory.sync.crypto import SyncCrypto
from atlas_memory.sync.transport import GossipTransport


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
    ) -> None:
        self.local_node_id: str = local_node_id
        self.crdt: DeltaCRDT = crdt
        self.crypto: SyncCrypto = crypto
        self.transport: Optional[GossipTransport] = transport
        self.peers: Dict[str, PeerStatus] = {}

    def register_peer(self, peer_id: str, vector_clock: Optional[VectorClock] = None) -> PeerStatus:
        """Registers or updates a peer in the known peer table."""
        vc = vector_clock or VectorClock()
        status = PeerStatus(peer_id=peer_id, vector_clock=vc, last_seen=time.time())
        self.peers[peer_id] = status
        return status

    def create_sync_request(self, target_peer_id: str) -> bytes:
        """Creates an encrypted JSON-RPC sync payload to send to target peer."""
        peer = self.peers.get(target_peer_id)
        since_vc = peer.vector_clock if peer else None

        delta = self.crdt.export_delta(since_vc)
        raw_payload = delta.model_dump_json().encode("utf-8")

        # Serialized local clock as AAD to prevent replay
        aad = json.dumps(self.crdt.clock.clocks, sort_keys=True).encode("utf-8")
        encrypted_blob = self.crypto.encrypt(raw_payload, aad=aad)

        envelope = {
            "source_node": self.local_node_id,
            "target_node": target_peer_id,
            "aad": aad.decode("utf-8"),
            "ciphertext_hex": encrypted_blob.hex(),
        }
        return json.dumps(envelope).encode("utf-8")

    async def send_sync(self, target_peer_id: str) -> None:
        """Sends an encrypted sync request to the target peer via the configured transport."""
        if self.transport is None:
            raise RuntimeError("Cannot send sync: no GossipTransport configured")
        payload = self.create_sync_request(target_peer_id)
        await self.transport.send(target_peer_id, payload)

    def process_sync_response(self, request_bytes: bytes) -> SyncDelta:
        """Decrypts and applies an incoming sync request or response."""
        envelope = json.loads(request_bytes.decode("utf-8"))
        aad = envelope["aad"].encode("utf-8")
        blob = bytes.fromhex(envelope["ciphertext_hex"])

        decrypted_bytes = self.crypto.decrypt(blob, aad=aad)
        delta_dict = json.loads(decrypted_bytes.decode("utf-8"))
        delta = SyncDelta.model_validate(delta_dict)

        self.crdt.apply_delta(delta)

        # Update peer status if known
        source = envelope.get("source_node")
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
        for peer_id, peer in self.peers.items():
            if self.crdt.clock.concurrent(peer.vector_clock):
                conflicts.append(
                    {
                        "peer_id": peer_id,
                        "type": "concurrent_clock",
                        "local_clock": self.crdt.clock.clocks,
                        "peer_clock": peer.vector_clock.clocks,
                    }
                )
        return conflicts
