# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""TCP transport implementations.

- `TCPTransport`: network-real asyncio TCP socket transport used by runtime.
- `InMemoryTCPTransport`: process-local compatibility shim for legacy sync callers.
"""

from __future__ import annotations

import asyncio
import logging
import random
from threading import Event, Thread
from typing import Awaitable, Callable

from .message import Message, MessageType


logger = logging.getLogger("core.tcp_transport")

MessageHandler = Callable[[Message], Awaitable[None]]
LegacyHandler = Callable[[dict], dict]

_SEND_MAX_RETRIES = 3
_SEND_BASE_BACKOFF_MS = 200
_SEND_MAX_BACKOFF_MS = 10_000
_SEND_JITTER = True


class InMemoryTCPTransport:
    """Process-local transport shim with legacy compatibility helpers."""

    _registry: dict[str, "InMemoryTCPTransport"] = {}
    _address_registry: dict[tuple[str, int], str] = {}

    @classmethod
    def reset_shared(cls) -> None:
        cls._registry.clear()
        cls._address_registry.clear()

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 19100,
        *,
        node_id: str | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.node_id = node_id or f"tcp://{host}:{port}"
        self._handler: MessageHandler | None = None
        self._legacy_handlers: dict[str, LegacyHandler] = {}
        self._pending: dict[str, asyncio.Future[Message]] = {}
        self._connected_nodes: set[str] = set()
        self._running = False

    def register_peer(self, node_id: str, host: str, port: int) -> None:
        self.__class__._address_registry[(host, port)] = node_id

    def _activate_listener(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._running = True
        self.__class__._registry[self.node_id] = self
        self.__class__._address_registry[(self.host, self.port)] = self.node_id
        logger.info("InMemoryTCPTransport listening on %s:%s as node_id=%s", self.host, self.port, self.node_id)

    async def listen(self, host: str, port: int) -> None:
        self._activate_listener(host, port)

    async def connect(self, host: str, port: int) -> object:
        if not self._running:
            raise RuntimeError("Transport must listen before connect")
        peer_id = self.__class__._address_registry.get((host, port))
        if peer_id is not None:
            self._connected_nodes.add(peer_id)
        return object()

    def start_server(self) -> None:
        """Legacy shim for previous sync startup code."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.listen(self.host, self.port))
            return
        self._activate_listener(self.host, self.port)

    def on_message(self, handler: MessageHandler) -> None:
        self._handler = handler

    def register_handler(self, msg_type: str, handler: LegacyHandler) -> None:
        self._legacy_handlers[msg_type] = handler

    def send(self, target: str, *args):
        """Dispatch either the async node_id API or the legacy host/port API.

        New API:
            await transport.send(node_id, message)

        Legacy API:
            response = transport.send(host, port, {"type": "..."})
        """
        if args and isinstance(args[0], int):
            port = args[0]
            message = args[1] if len(args) > 1 else {}
            return self._send_legacy(target, port, message)
        if not args:
            raise TypeError("send() missing required message argument")
        return self._send_async(target, args[0])

    async def _send_async(self, node_id: str, message: Message) -> None:
        peer = self.__class__._registry.get(node_id)
        if peer is None:
            return
        await peer._deliver_async(message)

    async def broadcast(self, message: Message, exclude: list[str] | None = None) -> None:
        excluded = set(exclude or [])
        tasks = []
        for peer_id, peer in self.__class__._registry.items():
            if peer_id == self.node_id or peer_id in excluded:
                continue
            tasks.append(peer._deliver_async(message))
        if tasks:
            await asyncio.gather(*tasks)

    async def request(self, node_id: str, message: Message, timeout: float = 5.0) -> Message | None:
        request_id = message.payload.get("request_id")
        if request_id is None:
            await self._send_async(node_id, message)
            return None
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Message] = loop.create_future()
        self._pending[str(request_id)] = future
        try:
            await self._send_async(node_id, message)
            return await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            key = str(request_id)
            if key in self._pending:
                del self._pending[key]

    def resolve_pending(self, request_id: str, response: Message) -> None:
        future = self._pending.get(request_id)
        if future is not None and not future.done():
            future.set_result(response)

    async def _deliver_async(self, message: Message) -> None:
        if self._handler is not None:
            await self._handler(message)
            return
        legacy_handler = self._legacy_handlers.get(message.type.value)
        if legacy_handler is not None:
            legacy_handler(self._message_to_legacy_dict(message))

    def _send_legacy(self, host: str, port: int, message: dict) -> dict:
        node_id = self.__class__._address_registry.get((host, port))
        if node_id is None:
            return {}
        peer = self.__class__._registry.get(node_id)
        if peer is None:
            return {}
        msg_type = str(message.get("type", ""))
        handler = peer._legacy_handlers.get(msg_type)
        if handler is not None:
            return handler(dict(message)) or {}
        if peer._handler is None:
            return {}
        try:
            message_obj = self._legacy_dict_to_message(message)
        except Exception:
            return {}
        self._run_coroutine_sync(peer._deliver_async(message_obj))
        return {}

    @staticmethod
    def _run_coroutine_sync(coro: Awaitable[None]) -> None:
        async def _runner() -> None:
            await coro

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(_runner())
            return

        done = Event()
        error: list[BaseException] = []

        def runner() -> None:
            try:
                asyncio.run(_runner())
            except BaseException as exc:  # pragma: no cover
                error.append(exc)
            finally:
                done.set()

        thread = Thread(target=runner, daemon=True)
        thread.start()
        done.wait()
        if error:
            raise error[0]

    @staticmethod
    def _message_to_legacy_dict(message: Message) -> dict:
        return {
            "type": message.type.value,
            "sender_id": message.sender_id,
            "sender_region": message.sender_region,
            "timestamp": message.timestamp,
            "payload": dict(message.payload),
        }

    def _legacy_dict_to_message(self, message: dict) -> Message:
        payload = dict(message.get("payload", {}))
        payload.update(
            {
                k: v
                for k, v in message.items()
                if k not in {"type", "sender_id", "sender_region", "timestamp", "payload"}
            }
        )
        return Message.build(
            message_type=MessageType(str(message.get("type", MessageType.HELLO.value))),
            sender_id=str(message.get("sender_id", self.node_id)),
            sender_region=str(message.get("sender_region", "local")),
            payload=payload,
            trace_id=payload.get("trace_id"),
        )


class TCPTransport:
    """Network-real async TCP transport for runtime communication."""

    _known_addresses: dict[str, tuple[str, int]] = {}

    @classmethod
    def reset_shared(cls) -> None:
        cls._known_addresses.clear()

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 19100,
        *,
        node_id: str | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.node_id = node_id or f"tcp://{host}:{port}"
        self._handler: MessageHandler | None = None
        self._pending: dict[str, asyncio.Future[Message]] = {}
        self._connected_nodes: set[str] = set()
        self._peers: dict[str, tuple[str, int]] = {}
        self._server: asyncio.AbstractServer | None = None
        self._running = False

    def register_peer(self, node_id: str, host: str, port: int) -> None:
        self._peers[node_id] = (host, port)
        self.__class__._known_addresses[node_id] = (host, port)

    async def listen(self, host: str, port: int) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        self.host = host
        self.port = port
        self._server = await asyncio.start_server(self._handle_client, host, port)
        self._running = True
        self.register_peer(self.node_id, host, port)
        logger.info("TCPTransport listening on %s:%s as node_id=%s", host, port, self.node_id)

    async def connect(self, host: str, port: int) -> object:
        if not self._running:
            raise RuntimeError("Transport must listen before connect")
        _reader, writer = await asyncio.open_connection(host, port)
        join = self._with_sender_hint(
            Message.build(
                message_type=MessageType.JOIN,
                sender_id=self.node_id,
                sender_region="local",
                payload={"host": self.host, "port": self.port},
            )
        )
        encoded = join.serialize()
        writer.write(len(encoded).to_bytes(4, "big") + encoded)
        await writer.drain()
        writer.close()
        await writer.wait_closed()

        for node_id, addr in self.__class__._known_addresses.items():
            if addr == (host, port):
                self._connected_nodes.add(node_id)
                self._peers[node_id] = addr
                break
        return object()

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self._running = False

    def on_message(self, handler: MessageHandler) -> None:
        self._handler = handler

    async def send(self, node_id: str, message: Message) -> None:
        target = self._resolve_target(node_id)
        if target is None:
            logger.warning("TCPTransport.send: peer not found for node=%s", node_id)
            return

        outbound = self._with_sender_hint(message)
        data = outbound.serialize()
        packet = len(data).to_bytes(4, "big") + data

        host, port = target
        await self._send_with_retry(
            node_id=node_id,
            host=host,
            port=port,
            packet=packet,
            max_retries=_SEND_MAX_RETRIES,
            base_backoff_ms=_SEND_BASE_BACKOFF_MS,
            max_backoff_ms=_SEND_MAX_BACKOFF_MS,
            jitter=_SEND_JITTER,
        )

    async def _send_with_retry(
        self,
        *,
        node_id: str,
        host: str,
        port: int,
        packet: bytes,
        max_retries: int,
        base_backoff_ms: int,
        max_backoff_ms: int,
        jitter: bool,
    ) -> None:
        delay = max(0.0, base_backoff_ms / 1000.0)
        for attempt in range(max_retries + 1):
            try:
                _reader, writer = await asyncio.open_connection(host, port)
                writer.write(packet)
                await writer.drain()
                writer.close()
                await writer.wait_closed()
                return
            except Exception:
                if attempt >= max_retries:
                    logger.exception(
                        "TCPTransport.send failed after retries: from=%s to=%s (%s:%s)",
                        self.node_id,
                        node_id,
                        host,
                        port,
                    )
                    return
                sleep_for = min(delay, max_backoff_ms / 1000.0)
                if jitter:
                    sleep_for = random.uniform(0.0, sleep_for)
                logger.warning(
                    "TCPTransport.send retry: from=%s to=%s (%s:%s) attempt=%d/%d backoff=%.3fs",
                    self.node_id,
                    node_id,
                    host,
                    port,
                    attempt + 1,
                    max_retries + 1,
                    sleep_for,
                )
                await asyncio.sleep(sleep_for)
                delay *= 2

    async def broadcast(self, message: Message, exclude: list[str] | None = None) -> None:
        excluded = set(exclude or [])
        targets = [
            node_id
            for node_id in self._all_known_peer_ids()
            if node_id != self.node_id and node_id not in excluded
        ]
        if not targets:
            return
        await asyncio.gather(*(self.send(node_id, message) for node_id in targets))

    async def request(self, node_id: str, message: Message, timeout: float = 5.0) -> Message | None:
        request_id = message.payload.get("request_id")
        if request_id is None:
            await self.send(node_id, message)
            return None

        future: asyncio.Future[Message] = asyncio.get_running_loop().create_future()
        self._pending[str(request_id)] = future
        try:
            await self.send(node_id, message)
            return await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("TCPTransport.request timeout: node=%s request_id=%s", node_id, request_id)
            return None
        finally:
            key = str(request_id)
            if key in self._pending:
                del self._pending[key]

    def resolve_pending(self, request_id: str, response: Message) -> None:
        future = self._pending.get(request_id)
        if future is not None and not future.done():
            future.set_result(response)

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                header = await reader.readexactly(4)
                size = int.from_bytes(header, "big")
                payload = await reader.readexactly(size)
                message = Message.deserialize(payload)
                self._track_sender(message)
                request_id = message.payload.get("request_id")
                if request_id is not None:
                    request_key = str(request_id)
                    if request_key in self._pending:
                        self.resolve_pending(request_key, message)
                if self._handler is not None:
                    await self._handler(message)
        except asyncio.IncompleteReadError:
            pass
        except Exception:
            logger.exception("TCPTransport._handle_client failed for node=%s", self.node_id)
        finally:
            writer.close()
            await writer.wait_closed()

    def _resolve_target(self, node_id: str) -> tuple[str, int] | None:
        if node_id in self._peers:
            return self._peers[node_id]
        if node_id in self.__class__._known_addresses:
            addr = self.__class__._known_addresses[node_id]
            self._peers[node_id] = addr
            return addr
        if node_id.startswith("tcp://"):
            host_port = node_id.removeprefix("tcp://")
            if ":" in host_port:
                host, port_text = host_port.rsplit(":", 1)
                if port_text.isdigit():
                    addr = (host, int(port_text))
                    self._peers[node_id] = addr
                    self.__class__._known_addresses[node_id] = addr
                    return addr
        return None

    def _all_known_peer_ids(self) -> set[str]:
        return set(self.__class__._known_addresses.keys()) | set(self._peers.keys())

    def _with_sender_hint(self, message: Message) -> Message:
        payload = dict(message.payload)
        payload.setdefault("__tcp_sender_host", self.host)
        payload.setdefault("__tcp_sender_port", self.port)
        return Message(
            type=message.type,
            sender_id=message.sender_id,
            sender_region=message.sender_region,
            timestamp=message.timestamp,
            payload=payload,
        )

    def _track_sender(self, message: Message) -> None:
        sender_host = message.payload.get("__tcp_sender_host")
        sender_port = message.payload.get("__tcp_sender_port")
        if isinstance(sender_host, str) and isinstance(sender_port, int):
            self.register_peer(message.sender_id, sender_host, sender_port)
            self._connected_nodes.add(message.sender_id)


