# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Heartbeat management for cross-region node liveness checks."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
import logging
from typing import Any, Awaitable, Callable, Protocol

from .message import Message, MessageType


class TransportProtocol(Protocol):
    async def broadcast(self, message: Message, exclude: list[str] | None = None) -> None: ...


class RegistryProtocol(Protocol):
    async def get_read_nodes(self) -> list[Any]: ...
    async def get_write_nodes(self) -> list[Any]: ...
    async def mark_failed(self, node_id: str) -> None: ...


logger = logging.getLogger("core.heartbeat")


class HeartbeatManager:
    """Sends periodic heartbeats and transitions nodes through suspect/failed."""

    def __init__(
        self,
        node_id: str,
        node_region: str,
        registry: RegistryProtocol,
        transport: TransportProtocol,
        interval_s: float = 10.0,
        timeout_s: float = 30.0,
        suspect_multiplier: float = 2.0,
    ) -> None:
        self.node_id = node_id
        self.node_region = node_region
        self.registry = registry
        self.transport = transport
        self.interval_s = interval_s
        self.timeout_s = timeout_s
        self.suspect_multiplier = max(1.0, suspect_multiplier)
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._node_states: dict[str, str] = {}
        self._status_callback: Callable[[str, str, str], Awaitable[None] | None] | None = None

        logger.info("HeartbeatManager initialized: node_id=%s region=%s interval_s=%s timeout_s=%s nats=%s",
                    node_id, node_region, interval_s, timeout_s, getattr(registry, 'nats_url', None))

    def on_status_change(
        self,
        callback: Callable[[str, str, str], Awaitable[None] | None],
    ) -> None:
        """Register optional callback(node_id, old_status, new_status)."""
        self._status_callback = callback

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("HeartbeatManager started for node %s", self.node_id)

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        logger.info("HeartbeatManager stopped for node %s", self.node_id)

    async def _run(self) -> None:
        while self._running:
            await self._send_heartbeat()
            await self._check_timeouts()
            await asyncio.sleep(self.interval_s)

    async def _send_heartbeat(self) -> None:
        message = Message.build(
            message_type=MessageType.HEARTBEAT,
            sender_id=self.node_id,
            sender_region=self.node_region,
            payload={"ts": time.time()},
        )
        logger.debug("Sending heartbeat from node %s payload=%s", self.node_id, message.payload)
        await self.transport.broadcast(message)

    async def _check_timeouts(self) -> None:
        now = time.time()
        read_nodes = await self.registry.get_read_nodes()
        write_nodes = await self.registry.get_write_nodes()
        candidates: dict[str, Any] = {n.node_id: n for n in [*read_nodes, *write_nodes]}
        for node in candidates.values():
            if node.node_id == self.node_id:
                continue
            age = now - node.last_seen
            if age > self.timeout_s * self.suspect_multiplier:
                logger.warning("Node timed out based on last_seen: node=%s last_seen=%s now=%s timeout_s=%s",
                               node.node_id, node.last_seen, now, self.timeout_s)
                await self._transition_status(node.node_id, "failed")
                continue
            if age > self.timeout_s:
                await self._transition_status(node.node_id, "suspect")
                continue
            await self._transition_status(node.node_id, "healthy")

    async def _transition_status(self, node_id: str, new_status: str) -> None:
        old_status = self._node_states.get(node_id, "healthy")
        if old_status == new_status:
            return

        if new_status == "failed":
            logger.info("Handling node failure for %s (marking failed in registry)", node_id)
            await self.registry.mark_failed(node_id)
        elif new_status == "suspect":
            mark_suspect = getattr(self.registry, "mark_suspect", None)
            if callable(mark_suspect):
                result = mark_suspect(node_id)
                if inspect.isawaitable(result):
                    await result

        self._node_states[node_id] = new_status

        if self._status_callback is not None:
            cb_result = self._status_callback(node_id, old_status, new_status)
            if inspect.isawaitable(cb_result):
                await cb_result
