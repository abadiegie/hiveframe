# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Cross-node replication manager built on top of transport and registry."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Protocol

from .message import Message, MessageType
from .registry import ClusterRegistry
from .wal import WriteAheadLog


logger = logging.getLogger("core.replication")


MessageHandler = Callable[[Message], Awaitable[None]]


class TransportProtocol(Protocol):
    _handler: MessageHandler | None

    def on_message(self, handler: MessageHandler) -> None: ...
    async def broadcast(self, message: Message, exclude: list[str] | None = None) -> None: ...
    async def send(self, node_id: str, message: Message) -> None: ...
    def resolve_pending(self, request_id: str, response: Message) -> None: ...


class ReplicationManager:
    """Replicates committed WAL entries from writer to read replicas."""

    def __init__(
        self,
        node_id: str,
        node_region: str,
        role: str,
        transport: TransportProtocol,
        registry: ClusterRegistry,
        wal: WriteAheadLog,
        apply_entry: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.node_id = node_id
        self.node_region = node_region
        self.role = role
        self.transport = transport
        self.registry = registry
        self.wal = wal
        self._last_lsn = 0
        self._ack_waiters: dict[int, asyncio.Future[bool]] = {}
        self._apply_entry = apply_entry
        self._seed_chunk_handler: Callable[[str, dict[str, list[Any]], int, str, bool], None] | None = None
        self._seen_chunk_offsets: dict[str, set[int]] = {}
        self._join_handler: Callable[[Message], Awaitable[None]] | None = None
        self._membership_update_handler: Callable[[Message], Awaitable[None]] | None = None
        self._route_update_handler: Callable[[Message], Awaitable[None]] | None = None
        self._oplog_push_handler: Callable[[Message], Awaitable[dict[str, Any]]] | None = None
        self._oplog_pull_handler: Callable[[Message], Awaitable[dict[str, Any]]] | None = None
        self._oplog_full_resync_handler: Callable[[Message], Awaitable[dict[str, Any]]] | None = None

    async def start(self) -> None:
        existing_handler = self.transport._handler
        if existing_handler is None:
            self.transport.on_message(self._on_message)
            return

        async def chained_handler(message: Message) -> None:
            await existing_handler(message)
            await self._on_message(message)

        self.transport.on_message(chained_handler)

    async def replicate_tx(self, lsn: int, tx_data: dict[str, Any]) -> bool:
        if self.role != "write":
            return False

        read_nodes = await self.registry.get_read_nodes()
        if not read_nodes:
            return True

        message = Message.build(
            message_type=MessageType.DELTA,
            sender_id=self.node_id,
            sender_region=self.node_region,
            payload={"lsn": lsn, "tx": tx_data},
        )
        tx_id = tx_data.get("tx_id") if isinstance(tx_data, dict) else None
        logger.info("Broadcasting DELTA lsn=%s tx_id=%s to %d read nodes", lsn, tx_id, len(read_nodes))
        await self.transport.broadcast(message)

        waiter = asyncio.get_running_loop().create_future()
        self._ack_waiters[lsn] = waiter
        try:
            await asyncio.wait_for(waiter, timeout=5.0)
            return True
        except asyncio.TimeoutError:
            return False
        finally:
            if lsn in self._ack_waiters:
                del self._ack_waiters[lsn]

    async def sync_from_lsn(self, from_lsn: int) -> list[dict[str, Any]]:
        return self.wal.get_since(from_lsn)

    async def _on_message(self, message: Message) -> None:
        if message.type == MessageType.DELTA and self.role == "read":
            await self._handle_delta(message)
        elif message.type == MessageType.SEED_CHUNK and self.role == "write":
            await self._handle_seed_chunk(message)
        elif message.type == MessageType.DELTA_ACK and self.role == "write":
            await self._handle_delta_ack(message)
        elif message.type == MessageType.SYNC_REQUEST and self.role == "write":
            await self._handle_sync_request(message)
        elif message.type == MessageType.SYNC_RESPONSE and self.role == "read":
            await self._handle_sync_response(message)
        elif message.type == MessageType.READ_SNAPSHOT_REQUEST:
            await self._handle_snapshot_request(message)
        elif message.type == MessageType.READ_SNAPSHOT_RESPONSE:
            await self._handle_snapshot_response(message)
        elif message.type == MessageType.JOIN:
            await self._handle_join(message)
        elif message.type == MessageType.MEMBERSHIP_UPDATE:
            await self._handle_membership_update(message)
        elif message.type == MessageType.ROUTE_UPDATE:
            await self._handle_route_update(message)
        elif message.type == MessageType.OPLOG_PUSH:
            await self._handle_oplog_push(message)
        elif message.type == MessageType.OPLOG_PULL:
            await self._handle_oplog_pull(message)
        elif message.type == MessageType.OPLOG_FULL_RESYNC:
            await self._handle_oplog_full_resync(message)
        elif message.type in {
            MessageType.OPLOG_PUSH_RESPONSE,
            MessageType.OPLOG_PULL_RESPONSE,
            MessageType.OPLOG_FULL_RESYNC_RESPONSE,
        }:
            await self._handle_oplog_response(message)

    async def _handle_delta(self, message: Message) -> None:
        lsn = int(message.payload["lsn"])
        tx_payload = dict(message.payload["tx"])

        if lsn <= self._last_lsn:
            return

        if self._apply_entry is not None:
            self._apply_entry(tx_payload)

        self._last_lsn = lsn
        await self.registry.update_lsn(self.node_id, lsn)

        ack = Message.build(
            message_type=MessageType.DELTA_ACK,
            sender_id=self.node_id,
            sender_region=self.node_region,
            payload={"lsn": lsn, "ok": True, "ts": time.time()},
        )
        await self.transport.send(message.sender_id, ack)

    async def _handle_delta_ack(self, message: Message) -> None:
        lsn = int(message.payload.get("lsn", 0))
        waiter = self._ack_waiters.get(lsn)
        if waiter is not None and not waiter.done():
            waiter.set_result(bool(message.payload.get("ok", True)))

    async def _handle_seed_chunk(self, message: Message) -> None:
        frame_id = str(message.payload.get("frame_id", ""))
        if not frame_id:
            return
        try:
            row_offset = int(message.payload.get("row_offset", 0))
        except (TypeError, ValueError):
            return
        raw_data = message.payload.get("data", {})
        if not isinstance(raw_data, dict):
            return
        data: dict[str, list[Any]] = {
            str(col): list(values) if isinstance(values, list) else [values]
            for col, values in raw_data.items()
        }
        transactional = bool(message.payload.get("transactional", True))

        seen = self._seen_chunk_offsets.setdefault(frame_id, set())
        if row_offset in seen:
            logger.debug(
                "Skipping duplicate SEED_CHUNK frame=%s offset=%d sender=%s",
                frame_id,
                row_offset,
                message.sender_id,
            )
            return
        seen.add(row_offset)

        if self._seed_chunk_handler is not None:
            self._seed_chunk_handler(frame_id, data, row_offset, message.sender_id, transactional)

    async def _handle_sync_request(self, message: Message) -> None:
        from_lsn = int(message.payload.get("from_lsn", 0))
        entries = self.wal.get_since(from_lsn)
        response = Message.build(
            message_type=MessageType.SYNC_RESPONSE,
            sender_id=self.node_id,
            sender_region=self.node_region,
            payload={"entries": entries},
        )
        await self.transport.send(message.sender_id, response)

    async def _handle_sync_response(self, message: Message) -> None:
        entries = list(message.payload.get("entries", []))
        for entry in entries:
            lsn = int(entry.get("lsn", 0))
            if lsn <= self._last_lsn:
                continue
            if self._apply_entry is not None:
                self._apply_entry(entry)
            self._last_lsn = lsn
            await self.registry.update_lsn(self.node_id, self._last_lsn)

    async def _handle_snapshot_request(self, message: Message) -> None:
        """Respond to a remote node requesting this node's snapshot for a specific frame."""
        request_id = str(message.payload.get("request_id", ""))
        frame_id = message.payload.get("frame_id", None)
        snapshot_data: dict[str, Any] = {}
        if hasattr(self, "_snapshot_provider") and self._snapshot_provider is not None:
            try:
                snapshot_data = self._snapshot_provider(frame_id)
            except TypeError:
                snapshot_data = self._snapshot_provider()
        response = Message.build(
            message_type=MessageType.READ_SNAPSHOT_RESPONSE,
            sender_id=self.node_id,
            sender_region=self.node_region,
            payload={"request_id": request_id, "snapshot": snapshot_data},
        )
        await self.transport.send(message.sender_id, response)

    async def _handle_snapshot_response(self, message: Message) -> None:
        """Resolve pending request-response future for snapshot fan-out."""
        request_id = str(message.payload.get("request_id", ""))
        self.transport.resolve_pending(request_id, message)

    async def _handle_join(self, message: Message) -> None:
        if self._join_handler is None:
            return
        await self._join_handler(message)

    async def _handle_membership_update(self, message: Message) -> None:
        if self._membership_update_handler is None:
            return
        await self._membership_update_handler(message)

    async def _handle_route_update(self, message: Message) -> None:
        if self._route_update_handler is None:
            return
        await self._route_update_handler(message)

    async def _handle_oplog_push(self, message: Message) -> None:
        if self._oplog_push_handler is None:
            return
        payload = await self._oplog_push_handler(message)
        response = Message.build(
            message_type=MessageType.OPLOG_PUSH_RESPONSE,
            sender_id=self.node_id,
            sender_region=self.node_region,
            payload={
                "request_id": message.payload.get("request_id"),
                **payload,
            },
        )
        await self.transport.send(message.sender_id, response)

    async def _handle_oplog_pull(self, message: Message) -> None:
        if self._oplog_pull_handler is None:
            return
        payload = await self._oplog_pull_handler(message)
        response = Message.build(
            message_type=MessageType.OPLOG_PULL_RESPONSE,
            sender_id=self.node_id,
            sender_region=self.node_region,
            payload={
                "request_id": message.payload.get("request_id"),
                **payload,
            },
        )
        await self.transport.send(message.sender_id, response)

    async def _handle_oplog_full_resync(self, message: Message) -> None:
        if self._oplog_full_resync_handler is None:
            return
        payload = await self._oplog_full_resync_handler(message)
        response = Message.build(
            message_type=MessageType.OPLOG_FULL_RESYNC_RESPONSE,
            sender_id=self.node_id,
            sender_region=self.node_region,
            payload={
                "request_id": message.payload.get("request_id"),
                **payload,
            },
        )
        await self.transport.send(message.sender_id, response)

    async def _handle_oplog_response(self, message: Message) -> None:
        request_id = str(message.payload.get("request_id", ""))
        if request_id:
            self.transport.resolve_pending(request_id, message)

    def set_snapshot_provider(self, provider: Callable[..., dict[str, Any]]) -> None:
        """Register a callback that returns this node's snapshot as a serializable dict.
        The callback may optionally accept a frame_id parameter.
        """
        self._snapshot_provider = provider

    def set_seed_chunk_handler(
        self,
        handler: Callable[[str, dict[str, list[Any]], int, str, bool], None],
    ) -> None:
        """Register handler for distributed seed chunks received over transport."""
        self._seed_chunk_handler = handler

    def set_cluster_control_handlers(
        self,
        *,
        join_handler: Callable[[Message], Awaitable[None]] | None = None,
        membership_update_handler: Callable[[Message], Awaitable[None]] | None = None,
        route_update_handler: Callable[[Message], Awaitable[None]] | None = None,
    ) -> None:
        """Register handlers for control-channel membership messages."""
        self._join_handler = join_handler
        self._membership_update_handler = membership_update_handler
        self._route_update_handler = route_update_handler

    def set_oplog_handlers(
        self,
        *,
        push_handler: Callable[[Message], Awaitable[dict[str, Any]]] | None = None,
        pull_handler: Callable[[Message], Awaitable[dict[str, Any]]] | None = None,
        full_resync_handler: Callable[[Message], Awaitable[dict[str, Any]]] | None = None,
    ) -> None:
        """Register handlers for op-log push/pull/full-resync messages."""
        self._oplog_push_handler = push_handler
        self._oplog_pull_handler = pull_handler
        self._oplog_full_resync_handler = full_resync_handler

