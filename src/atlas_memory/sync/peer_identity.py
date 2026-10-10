"""PeerIdentity: Asymmetric Ed25519 node authentication and attestation for memory sync.

Replaces simulated threshold multi-sig with cryptographically verified node identity:
- Deterministic 32-byte Ed25519 signing key per node.
- Digital signature verification for sync deltas and peer attestation.
- Bounded registry of trusted peer public keys (max 100 peers).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Union

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

logger = logging.getLogger(__name__)


class PeerIdentity:
    """Węzłowa tożsamość kryptograficzna oparta o cyfrowe podpisy asymetryczne Ed25519."""

    MAX_TRUSTED_PEERS = 100

    def __init__(
        self,
        node_id: str,
        signing_key_bytes: Optional[bytes] = None,
    ) -> None:
        self.node_id = str(node_id)
        if signing_key_bytes is not None and len(signing_key_bytes) >= 32:
            self._private_key = ed25519.Ed25519PrivateKey.from_private_bytes(signing_key_bytes[:32])
        else:
            self._private_key = ed25519.Ed25519PrivateKey.generate()

        self._peer_public_keys: Dict[str, bytes] = {}

    def get_public_key_bytes(self) -> bytes:
        """Zwraca surowe bajty (32 B) publicznego klucza Ed25519 tego węzła."""
        return self._private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def register_peer(self, peer_id: str, public_key_bytes: bytes) -> None:
        """Rejestruje klucz publiczny peera w rejestrze zaufanych tożsamości (z limitem LRU)."""
        pid = str(peer_id)
        if len(self._peer_public_keys) >= self.MAX_TRUSTED_PEERS and pid not in self._peer_public_keys:
            # Usuń najstarszy wpis aby zapobiec wyciekowi pamięci
            oldest_key = next(iter(self._peer_public_keys))
            del self._peer_public_keys[oldest_key]
        self._peer_public_keys[pid] = bytes(public_key_bytes[:32])

    def is_peer_registered(self, peer_id: str) -> bool:
        """Zwraca True jeśli klucz publiczny peera znajduje się w rejestrze."""
        return str(peer_id) in self._peer_public_keys or str(peer_id) == self.node_id

    def sign(self, data: Union[str, bytes]) -> bytes:
        """Generuje cyfrowy podpis Ed25519 dla przekazanego ładunku."""
        data_bytes = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        return self._private_key.sign(data_bytes)

    def verify(self, peer_id: str, data: Union[str, bytes], signature: bytes) -> bool:
        """Weryfikuje podpis ładunku z użyciem zarejestrowanego klucza publicznego peera."""
        pid = str(peer_id)
        data_bytes = data.encode("utf-8") if isinstance(data, str) else bytes(data)

        if pid == self.node_id:
            pub = self._private_key.public_key()
        elif pid in self._peer_public_keys:
            pub_bytes = self._peer_public_keys[pid]
            pub = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
        else:
            logger.debug("Verification failed: peer '%s' is not registered in trusted registry", pid)
            return False

        try:
            pub.verify(signature, data_bytes)
            return True
        except (InvalidSignature, Exception):
            return False

