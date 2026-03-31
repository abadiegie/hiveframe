# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Cluster registry with optional NATS backend and in-memory fallback."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Awaitable, Callable
import logging


@dataclass(slots=True)
class NodeInfo:
    node_id: str
    host: str
    port: int
    role: str
    region: str
    partition_start: int
    partition_end: int
    last_seen: float
    lsn: int
    status: str


WatchCallback = Callable[[NodeInfo, str], Awaitable[None]]

_PARTITION_TOTAL = 1000


logger = logging.getLogger("core.registry")


class ClusterRegistry:
    """NATS-like async registry with in-memory behavior for local tests."""

    # Shared in-memory node store across all registry instances in the same process.
    # Mirrors InMemoryQuicTransport._registry so nodes can discover each other locally.
    _shared_nodes: dict[str, "NodeInfo"] = {}

    @classmethod
    def reset_shared(cls) -> None:
        """Clear shared state — call this in tests to avoid inter-test pollution."""
        cls._shared_nodes.clear()

    def __init__(self, nats_url: str) -> None:
        self.nats_url = nats_url
        self._connected = False
        self._watchers: list[WatchCallback] = []
        self._has_nats = False
        try:
            import nats  # noqa: F401

            self._has_nats = True
            logger.info("NATS library detected; ClusterRegistry can use NATS at %s", nats_url)
        except Exception:
            self._has_nats = False
            logger.info("NATS library not available; using in-memory registry")

    @property
    def _nodes(self) -> dict[str, "NodeInfo"]:
        return self.__class__._shared_nodes

    async def connect(self) -> None:
        self._connected = True
        logger.info("ClusterRegistry connected (nats_url=%s has_nats=%s)", self.nats_url, self._has_nats)

    async def register(self, node: NodeInfo) -> None:
        if not self._connected:
            raise RuntimeError("Registry must connect before register")
        event = "updated" if node.node_id in self._nodes else "joined"
        self._nodes[node.node_id] = node
        # Auto-assign and rebalance partitions whenever a writer node joins.
        if node.role == "write":
            self._rebalance_partitions()
        logger.info("Node %s %s (role=%s host=%s port=%s)", node.node_id, event, node.role, node.host, node.port)
        await self._notify(node, event)

    # -------------------------------------------------------------------------
    # Partition management
    # -------------------------------------------------------------------------

    def _write_nodes_sorted(self) -> list[NodeInfo]:
        """Return healthy write nodes sorted by node_id for deterministic ordering."""
        return sorted(
            [n for n in self._nodes.values() if n.role == "write" and n.status != "failed"],
            key=lambda n: n.node_id,
        )

    def _rebalance_partitions(self) -> None:
        """Divide [0, PARTITION_TOTAL) evenly across all healthy writer nodes."""
        writers = self._write_nodes_sorted()
        if not writers:
            logger.debug("No writers available for rebalance")
            return
        chunk = _PARTITION_TOTAL // len(writers)
        for idx, node in enumerate(writers):
            node.partition_start = idx * chunk
            node.partition_end = (
                (idx + 1) * chunk if idx < len(writers) - 1 else _PARTITION_TOTAL
            )
        logger.info("Rebalanced partitions across %d writers", len(writers))

    def get_owner_for_row(self, row_index: int) -> NodeInfo | None:
        """Return the writer node that owns the partition containing row_index.

        Maps row_index into [0, PARTITION_TOTAL) via modulo so datasets larger
        than PARTITION_TOTAL still route deterministically.
        """
        writers = self._write_nodes_sorted()
        if not writers:
            logger.debug("get_owner_for_row: no writers registered")
            return None
        slot = row_index % _PARTITION_TOTAL
        for node in writers:
            if node.partition_start <= slot < node.partition_end:
                logger.debug("get_owner_for_row: row_index=%s slot=%s owner=%s", row_index, slot, node.node_id)
                return node
        logger.debug("get_owner_for_row: defaulting to last writer %s for row_index=%s slot=%s", writers[-1].node_id, row_index, slot)
        return writers[-1]

    # -------------------------------------------------------------------------
    # Node querying
    # -------------------------------------------------------------------------

    async def get_write_node(self) -> NodeInfo | None:
        for node in self._nodes.values():
            if node.role == "write" and node.status != "failed":
                logger.debug("get_write_node -> %s", node.node_id)
                return node
        logger.debug("get_write_node -> None")
        return None

    async def get_write_nodes(self) -> list[NodeInfo]:
        """Return all healthy write nodes for fan-out style read operations."""
        nodes = [
            node
            for node in self._nodes.values()
            if node.role == "write" and node.status != "failed"
        ]
        logger.debug("get_write_nodes -> %d nodes", len(nodes))
        return nodes

    async def get_read_nodes(self) -> list[NodeInfo]:
        nodes = [
            node
            for node in self._nodes.values()
            if node.role == "read" and node.status != "failed"
        ]
        logger.debug("get_read_nodes -> %d nodes", len(nodes))
        return nodes

    async def get_node(self, node_id: str) -> NodeInfo | None:
        node = self._nodes.get(node_id)
        logger.debug("get_node %s -> %s", node_id, getattr(node, 'status', None))
        return node

    async def watch(self, callback: WatchCallback) -> None:
        self._watchers.append(callback)
        logger.debug("Watcher added; total watchers=%d", len(self._watchers))

    async def update_lsn(self, node_id: str, lsn: int) -> None:
        node = self._nodes.get(node_id)
        if node is None:
            logger.debug("update_lsn: node %s not found", node_id)
            return
        node.lsn = lsn
        node.last_seen = time.time()
        logger.debug("update_lsn: node=%s lsn=%d", node_id, lsn)
        await self._notify(node, "updated")

    async def mark_failed(self, node_id: str) -> None:
        node = self._nodes.get(node_id)
        if node is None:
            logger.debug("mark_failed: node %s not found", node_id)
            return
        node.status = "failed"
        node.last_seen = time.time()
        # Rebalance remaining writers after a node failure.
        self._rebalance_partitions()
        logger.info("Node marked failed: %s", node_id)
        await self._notify(node, "failed")

    async def _notify(self, node: NodeInfo, event_type: str) -> None:
        logger.debug("Notifying watchers: node=%s event=%s watchers=%d", node.node_id, event_type, len(self._watchers))
        for watcher in list(self._watchers):
            await watcher(node, event_type)
