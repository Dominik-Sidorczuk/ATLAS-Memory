"""Threshold signatures and quorum verification for Byzantine Fault Tolerant sync."""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, hmac, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from pydantic import BaseModel, ConfigDict, Field


class SignatureResult(BaseModel):
    """Model wyniku podpisu kryptograficznego dla operacji synchronizacji."""

    node_id: str = Field(description="Identyfikator węzła sygnującego")
    signature: bytes = Field(description="Surowe bajty podpisu")
    algorithm: str = Field(
        default="hmac_sha256",
        description="Algorytm podpisu ('hmac_sha256' | 'ed25519')",
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Znacznik czasu wygenerowania podpisu",
    )


class ThresholdSigner(BaseModel):
    """Klasa do podpisywania i weryfikacji operacji z progiem kworum (Threshold Signatures)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    node_id: str = Field(description="Identyfikator bieżącego węzła")
    signing_key: bytes = Field(description="Klucz prywatny / symetryczny węzła (32 bajty)")
    quorum: int = Field(default=3, ge=1, description="Wymagana liczba unikalnych podpisów (2f + 1)")
    algorithm: str = Field(default="hmac_sha256", description="Domyślny algorytm podpisu")
    peer_keys: Dict[str, bytes] = Field(
        default_factory=dict,
        description="Klucze symetryczne lub ziarna kluczy peerów",
    )
    peer_public_keys: Dict[str, bytes] = Field(
        default_factory=dict,
        description="Klucze publiczne Ed25519 peerów",
    )

    def register_peer(
        self,
        node_id: str,
        key_or_pubkey: bytes,
        is_public_key: bool = False,
    ) -> None:
        """Rejestruje klucz peera do weryfikacji podpisów."""
        if is_public_key:
            self.peer_public_keys[node_id] = key_or_pubkey
        else:
            self.peer_keys[node_id] = key_or_pubkey

    def get_public_key_bytes(self) -> bytes:
        """Zwraca surowe bajty klucza publicznego Ed25519 dla tego węzła."""
        priv = ed25519.Ed25519PrivateKey.from_private_bytes(self.signing_key[:32])
        return priv.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def sign(self, operation: str, algorithm: Optional[str] = None) -> SignatureResult:
        """Podpisuje zadaną operację przy użyciu skonfigurowanego algorytmu."""
        algo = algorithm or self.algorithm
        op_bytes = operation.encode("utf-8")

        if algo == "hmac_sha256":
            h = hmac.HMAC(self.signing_key, hashes.SHA256())
            h.update(op_bytes)
            sig_bytes = h.finalize()
            return SignatureResult(
                node_id=self.node_id,
                signature=sig_bytes,
                algorithm="hmac_sha256",
            )
        elif algo == "ed25519":
            priv = ed25519.Ed25519PrivateKey.from_private_bytes(self.signing_key[:32])
            sig_bytes = priv.sign(op_bytes)
            return SignatureResult(
                node_id=self.node_id,
                signature=sig_bytes,
                algorithm="ed25519",
            )
        else:
            raise NotImplementedError(f"Nieobsługiwany algorytm podpisu: {algo}")

    def verify(self, signature: SignatureResult, operation: str) -> bool:
        """Weryfikuje podpis dla zadanej operacji."""
        op_bytes = operation.encode("utf-8")

        if signature.algorithm == "hmac_sha256":
            key = self.peer_keys.get(signature.node_id, self.signing_key)
            h = hmac.HMAC(key, hashes.SHA256())
            h.update(op_bytes)
            try:
                h.verify(signature.signature)
                return True
            except (InvalidSignature, Exception):
                return False

        elif signature.algorithm == "ed25519":
            try:
                if signature.node_id in self.peer_public_keys:
                    pub_bytes = self.peer_public_keys[signature.node_id]
                    pub = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
                elif signature.node_id == self.node_id:
                    priv = ed25519.Ed25519PrivateKey.from_private_bytes(self.signing_key[:32])
                    pub = priv.public_key()
                elif signature.node_id in self.peer_keys:
                    raw_k = self.peer_keys[signature.node_id]
                    try:
                        pub = ed25519.Ed25519PublicKey.from_public_bytes(raw_k)
                    except Exception:
                        priv = ed25519.Ed25519PrivateKey.from_private_bytes(raw_k[:32])
                        pub = priv.public_key()
                else:
                    priv = ed25519.Ed25519PrivateKey.from_private_bytes(self.signing_key[:32])
                    pub = priv.public_key()

                pub.verify(signature.signature, op_bytes)
                return True
            except (InvalidSignature, Exception):
                return False

        else:
            return False

    def has_quorum(
        self,
        signatures: List[SignatureResult],
        operation: Optional[str] = None,
    ) -> bool:
        """Sprawdza, czy lista unikalnych podpisujących węzłów spełnia kworum (>= quorum)."""
        if not signatures:
            return False

        valid_signers = set()
        for sig in signatures:
            if not sig.node_id:
                continue
            if operation is not None:
                if not self.verify(sig, operation):
                    continue
            valid_signers.add(sig.node_id)

        return len(valid_signers) >= self.quorum
