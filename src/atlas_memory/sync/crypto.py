"""End-to-End Cryptographic layer for secure agent memory synchronization."""

from __future__ import annotations

import secrets
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class SyncCrypto:
    """AES-256-GCM authenticated encryption for sync payloads."""

    def __init__(self, key: bytes) -> None:
        if len(key) != 32:
            raise ValueError(f"AES-256 key must be exactly 32 bytes, got {len(key)}")
        self._key = key
        self._aesgcm = AESGCM(key)

    @classmethod
    def generate_key(cls) -> bytes:
        """Generates a secure random 32-byte (256-bit) key."""
        return secrets.token_bytes(32)

    def encrypt(self, plaintext: bytes, aad: Optional[bytes] = None) -> bytes:
        """Encrypts plaintext with a fresh 12-byte nonce using AES-256-GCM.

        Returns:
            bytes: 12-byte nonce prepended to ciphertext + tag.
        """
        nonce = secrets.token_bytes(12)
        ciphertext = self._aesgcm.encrypt(nonce, plaintext, aad)
        return nonce + ciphertext

    def decrypt(self, blob: bytes, aad: Optional[bytes] = None) -> bytes:
        """Decrypts a blob containing 12-byte nonce + ciphertext + tag.

        Raises:
            cryptography.exceptions.InvalidTag: If decryption fails or data was tampered with.
            ValueError: If blob is shorter than 12 bytes.
        """
        if len(blob) < 28:
            raise ValueError(f"Ciphertext blob must be at least 28 bytes (12 nonce + 16 tag), got {len(blob)}")
        nonce = blob[:12]
        ciphertext = blob[12:]
        return self._aesgcm.decrypt(nonce, ciphertext, aad)
