"""Transport layer for Gossip Protocol synchronization across distributed agents."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)



class GossipTransport(ABC):
    """Abstract base class for gossip synchronization transport."""

    @abstractmethod
    async def send(self, peer_id: str, payload: bytes) -> None:
        """Sends payload bytes to the specified peer."""

    @abstractmethod
    async def receive(self, timeout: float = 1.0) -> Optional[bytes]:
        """Receives incoming payload bytes, returning None if timeout is reached."""

    async def close(self) -> None:
        """Closes transport resources if any."""


class _UDPProtocol(asyncio.DatagramProtocol):
    """Internal DatagramProtocol handler for UDP gossip."""

    def __init__(self, queue: asyncio.Queue[bytes]) -> None:
        self.queue = queue
        self.transport: Optional[asyncio.DatagramTransport] = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        if isinstance(transport, asyncio.DatagramTransport):
            self.transport = transport

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        self.queue.put_nowait(data)

    def error_received(self, exc: Exception) -> None:
        logger.warning("UDPGossipProtocol error received: %s", exc)

    def connection_lost(self, exc: Optional[Exception]) -> None:
        if exc is not None:
            logger.warning("UDPGossipProtocol connection lost with exception: %s", exc)



class UDPGossipTransport(GossipTransport):
    """UDP Datagram-based Gossip transport for low-latency peer synchronization."""

    def __init__(
        self,
        local_port: int = 0,
        peers: Optional[Dict[str, Tuple[str, int]]] = None,
        host: str = "127.0.0.1",
    ) -> None:
        self.local_port: int = local_port
        self.host: str = host
        self.peers: Dict[str, Tuple[str, int]] = peers or {}
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._protocol: Optional[_UDPProtocol] = None

    async def start(self) -> None:
        """Initializes the UDP datagram endpoint."""
        if self._transport is not None:
            return
        loop = asyncio.get_running_loop()
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: _UDPProtocol(self._queue),
            local_addr=(self.host, self.local_port),
        )
        self._transport = transport
        self._protocol = protocol
        sockname = transport.get_extra_info("sockname")
        if sockname and isinstance(sockname, tuple) and len(sockname) >= 2:
            self.local_port = sockname[1]

    def get_local_port(self) -> int:
        """Returns the bound local port."""
        if self._transport is not None:
            sockname = self._transport.get_extra_info("sockname")
            if sockname and isinstance(sockname, tuple) and len(sockname) >= 2:
                return sockname[1]
        return self.local_port

    def register_peer(self, peer_id: str, host: str, port: int) -> None:
        """Registers a remote peer host and port."""
        self.peers[peer_id] = (host, port)

    async def send(self, peer_id: str, payload: bytes) -> None:
        """Sends datagram to registered peer."""
        if self._transport is None:
            await self.start()
        if peer_id not in self.peers:
            raise ValueError(f"Unknown peer: {peer_id}")
        addr = self.peers[peer_id]
        if self._transport is not None:
            self._transport.sendto(payload, addr)

    async def receive(self, timeout: float = 1.0) -> Optional[bytes]:
        """Receives incoming datagram payload."""
        if self._transport is None:
            await self.start()
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def close(self) -> None:
        """Closes the UDP datagram transport."""
        if self._transport is not None:
            self._transport.close()
            self._transport = None
            self._protocol = None


class InMemoryGossipTransport(GossipTransport):
    """In-Memory queue-based Gossip transport for local testing and zero-overhead IPC."""

    def __init__(
        self,
        peers: Optional[Dict[str, InMemoryGossipTransport]] = None,
    ) -> None:
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.peers: Dict[str, InMemoryGossipTransport] = peers or {}

    def register_peer_transport(self, peer_id: str, transport: InMemoryGossipTransport) -> None:
        """Registers another in-memory peer transport."""
        self.peers[peer_id] = transport

    async def send(self, peer_id: str, payload: bytes) -> None:
        """Delivers payload to peer queue or self queue if peer not mapped."""
        if peer_id in self.peers:
            await self.peers[peer_id].queue.put(payload)
        else:
            await self.queue.put(payload)

    async def receive(self, timeout: float = 1.0) -> Optional[bytes]:
        """Receives next payload from internal queue."""
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def close(self) -> None:
        """No-op for in-memory transport."""
        logger.debug("InMemoryGossipTransport closed.")



def create_transport(kind: str = "udp", **kwargs: Any) -> GossipTransport:
    """Factory function for creating gossip transport instances."""
    kind_lower = kind.lower()
    if kind_lower in ("udp", "datagram"):
        return UDPGossipTransport(**kwargs)
    elif kind_lower in ("inmemory", "in_memory", "memory", "queue"):
        return InMemoryGossipTransport(**kwargs)
    else:
        raise ValueError(f"Unknown transport kind: '{kind}'")
