# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Transport abstraction with optional aioquic backend and in-memory fallback."""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from .message import Message

logger = logging.getLogger("core.quic_transport")

MessageHandler = Callable[[Message], Awaitable[None]]


class InMemoryQuicTransport:
    """Process-local transport used in tests and as runtime fallback."""

    _registry: dict[str, "InMemoryQuicTransport"] = {}

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self._handler: MessageHandler | None = None
        self._running = False
        self._pending: dict[str, asyncio.Future[Message]] = {}

    async def listen(self, host: str, port: int) -> None:
        _ = (host, port)
        self._running = True
        self._registry[self.node_id] = self

    async def connect(self, host: str, port: int) -> object:
        _ = (host, port)
        if not self._running:
            raise RuntimeError("Transport must listen before connect")
        return object()

    async def send(self, node_id: str, message: Message) -> None:
        peer = self._registry.get(node_id)
        trace_id = message.payload.get("trace_id")
        if peer is None or peer._handler is None:
            logger.warning("InMemoryQuicTransport.send: no peer/handler for node=%s trace_id=%s", node_id, trace_id)
            return
        logger.debug("InMemoryQuicTransport.send: from=%s to=%s type=%s request_id=%s trace_id=%s", self.node_id, node_id, getattr(message.type, 'value', message.type), message.payload.get("request_id"), trace_id)
        await peer._handler(message)

    async def broadcast(self, message: Message, exclude: list[str] | None = None) -> None:
        excluded = set(exclude or [])
        tasks = []
        targets = []
        for peer_id, peer in self._registry.items():
            if peer_id == self.node_id or peer_id in excluded or peer._handler is None:
                continue
            targets.append(peer_id)
            tasks.append(peer._handler(message))
        trace_id = message.payload.get("trace_id")
        logger.debug("InMemoryQuicTransport.broadcast: from=%s trace_id=%s targets=%s type=%s", self.node_id, trace_id, targets, getattr(message.type, 'value', message.type))
        if tasks:
            await asyncio.gather(*tasks)

    async def request(self, node_id: str, message: Message, timeout: float = 5.0) -> Message | None:
        """Send a message and wait for a correlated response keyed by request_id."""
        request_id = message.payload.get("request_id")
        trace_id = message.payload.get("trace_id")
        if request_id is None:
            logger.debug("InMemoryQuicTransport.request without request_id: delegating to send for node=%s trace_id=%s", node_id, trace_id)
            await self.send(node_id, message)
            return None
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Message] = loop.create_future()
        self._pending[str(request_id)] = future
        try:
            logger.debug("InMemoryQuicTransport.request: from=%s to=%s request_id=%s trace_id=%s timeout=%s", self.node_id, node_id, request_id, trace_id, timeout)
            await self.send(node_id, message)
            resp = await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
            logger.debug("InMemoryQuicTransport.request: received response for request_id=%s trace_id=%s", request_id, trace_id)
            return resp
        except asyncio.TimeoutError:
            logger.warning("InMemoryQuicTransport.request: timeout waiting for response node=%s request_id=%s trace_id=%s", node_id, request_id, trace_id)
            return None
        finally:
            self._pending.pop(str(request_id), None)

    def resolve_pending(self, request_id: str, response: Message) -> None:
        """Resolve a pending request future with the given response message."""
        trace_id = response.payload.get("trace_id")
        future = self._pending.get(request_id)
        logger.debug("InMemoryQuicTransport.resolve_pending: request_id=%s trace_id=%s found=%s", request_id, trace_id, future is not None)
        if future is not None and not future.done():
            future.set_result(response)

    def on_message(self, handler: MessageHandler) -> None:
        self._handler = handler


class QuicTransport:
    """Adapter exposing QUIC-like API with optional real aioquic dependency."""

    def __init__(self, node_id: str, tls_cert: str | None = None, tls_key: str | None = None) -> None:
        self.node_id = node_id
        self.tls_cert = tls_cert
        self.tls_key = tls_key
        self._fallback = InMemoryQuicTransport(node_id=node_id)
        self._handler: MessageHandler | None = None
        self._connected_nodes: set[str] = set()
        self._has_aioquic = False
        try:
            import aioquic  # noqa: F401

            self._has_aioquic = True
        except Exception:
            self._has_aioquic = False

    async def listen(self, host: str, port: int) -> None:
        # Option A: use fallback transport in-process unless full runtime wiring is enabled.
        await self._fallback.listen(host, port)
        if self._handler is not None:
            self._fallback.on_message(self._handler)

    async def connect(self, host: str, port: int) -> object:
        _ = (host, port)
        if self._has_aioquic:
            self._connected_nodes.add(f"{host}:{port}")
        return await self._fallback.connect(host, port)

    async def send(self, node_id: str, message: Message) -> None:
        logger.debug("QuicTransport.send: from=%s to=%s type=%s request_id=%s trace_id=%s", self.node_id, node_id, getattr(message.type, 'value', message.type), message.payload.get("request_id"), message.payload.get("trace_id"))
        await self._fallback.send(node_id, message)

    async def broadcast(self, message: Message, exclude: list[str] | None = None) -> None:
        logger.debug("QuicTransport.broadcast: from=%s type=%s trace_id=%s exclude=%s", self.node_id, getattr(message.type, 'value', message.type), message.payload.get("trace_id"), exclude)
        await self._fallback.broadcast(message, exclude=exclude)

    async def request(self, node_id: str, message: Message, timeout: float = 5.0) -> Message | None:
        """Delegate request-response to fallback transport."""
        logger.debug("QuicTransport.request: from=%s to=%s request_id=%s trace_id=%s timeout=%s", self.node_id, node_id, message.payload.get("request_id"), message.payload.get("trace_id"), timeout)
        resp = await self._fallback.request(node_id, message, timeout=timeout)
        if resp is None:
            logger.warning("QuicTransport.request: no response from node=%s request_id=%s trace_id=%s", node_id, message.payload.get("request_id"), message.payload.get("trace_id"))
        else:
            logger.debug("QuicTransport.request: got response from node=%s request_id=%s trace_id=%s", node_id, message.payload.get("request_id"), message.payload.get("trace_id"))
        return resp

    def resolve_pending(self, request_id: str, response: Message) -> None:
        logger.debug("QuicTransport.resolve_pending: request_id=%s trace_id=%s", request_id, response.payload.get("trace_id"))
        self._fallback.resolve_pending(request_id, response)

    def on_message(self, handler: MessageHandler) -> None:
        self._handler = handler
        self._fallback.on_message(handler)
        logger.debug("QuicTransport.on_message: handler registered for node=%s", self.node_id)
