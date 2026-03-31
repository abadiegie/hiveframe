# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Heartbeat management for cross-region node liveness checks."""

from __future__ import annotations

import asyncio
import contextlib
import time
import logging

from .message import Message, MessageType
from .quic_transport import QuicTransport
from .registry import ClusterRegistry


logger = logging.getLogger("core.heartbeat")


class HeartbeatManager:
    """Sends periodic heartbeats and marks timed-out nodes as failed."""

    def __init__(
        self,
        node_id: str,
        node_region: str,
        registry: ClusterRegistry,
        transport: QuicTransport,
        interval_s: float = 10.0,
        timeout_s: float = 30.0,
    ) -> None:
        self.node_id = node_id
        self.node_region = node_region
        self.registry = registry
        self.transport = transport
        self.interval_s = interval_s
        self.timeout_s = timeout_s
        self._running = False
        self._task: asyncio.Task[None] | None = None

        logger.info("HeartbeatManager initialized: node_id=%s region=%s interval_s=%s timeout_s=%s nats=%s",
                    node_id, node_region, interval_s, timeout_s, getattr(registry, 'nats_url', None))

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
        write_node = await self.registry.get_write_node()
        candidates = list(read_nodes)
        if write_node is not None:
            candidates.append(write_node)
        for node in candidates:
            if node.node_id == self.node_id:
                continue
            if now - node.last_seen > self.timeout_s:
                logger.warning("Node timed out based on last_seen: node=%s last_seen=%s now=%s timeout_s=%s",
                               node.node_id, node.last_seen, now, self.timeout_s)
                await self._handle_node_failure(node.node_id)

    async def _handle_node_failure(self, node_id: str) -> None:
        logger.info("Handling node failure for %s (marking failed in registry)", node_id)
        await self.registry.mark_failed(node_id)
