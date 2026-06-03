# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Cluster registry with optional NATS backend and in-memory fallback."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Awaitable, Callable
import logging

from . import node_status


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
    leader_reachable: bool = True
    wal_reachable: bool = True
    leader_epoch: int = 0

    @property
    def capability(self) -> str:
        """Compute capability (rw|ro|drain) from current state."""
        return node_status.compute_capability(
            self.role, self.status, self.leader_reachable, self.wal_reachable
        )


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
            [n for n in self._nodes.values() if n.role == "write" and n.status == "healthy"],
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
            if node.role == "write" and node.status == "healthy":
                logger.debug("get_write_node -> %s", node.node_id)
                return node
        logger.debug("get_write_node -> None")
        return None

    async def get_write_nodes(self) -> list[NodeInfo]:
        """Return all healthy write nodes for fan-out style read operations."""
        nodes = [
            node
            for node in self._nodes.values()
            if node.role == "write" and node.status == "healthy"
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

    async def mark_healthy(self, node_id: str) -> None:
        """Mark a node healthy again after suspect/failed recovery."""
        node = self._nodes.get(node_id)
        if node is None:
            logger.debug("mark_healthy: node %s not found", node_id)
            return
        if node.status == "healthy":
            return
        node.status = "healthy"
        node.last_seen = time.time()
        if node.role == "write":
            self._rebalance_partitions()
        logger.info("Node marked healthy: %s", node_id)
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

    async def mark_suspect(self, node_id: str) -> None:
        """Mark a node as suspect (intermediate state before failure)."""
        node = self._nodes.get(node_id)
        if node is None:
            logger.debug("mark_suspect: node %s not found", node_id)
            return
        if node.status == "suspect":
            logger.debug("mark_suspect: node %s already suspect", node_id)
            return
        node.status = "suspect"
        node.last_seen = time.time()
        if node.role == "write":
            # Suspect writers are excluded from ownership; rebalance healthy writers.
            self._rebalance_partitions()
        logger.info("Node marked suspect: %s", node_id)
        await self._notify(node, "suspect")

    async def apply_partition_map(self, partition_map: list[dict[str, int | str]]) -> int:
        """Apply leader-provided partition ownership map to local registry state."""
        self._validate_partition_map(partition_map)
        applied = 0
        for entry in partition_map:
            node_id = str(entry.get("node_id", "") or "")
            if not node_id:
                continue
            node = self._nodes.get(node_id)
            if node is None:
                continue
            try:
                start = int(entry.get("partition_start", node.partition_start))
                end = int(entry.get("partition_end", node.partition_end))
            except (TypeError, ValueError):
                continue
            node.partition_start = start
            node.partition_end = end
            applied += 1
        if applied:
            logger.info("Applied partition map entries=%d", applied)
        return applied

    def _validate_partition_map(self, partition_map: list[dict[str, int | str]]) -> None:
        if not partition_map:
            raise ValueError("partition map is empty")
        prev_end: int | None = None
        seen_node_ids: set[str] = set()
        for idx, entry in enumerate(partition_map):
            node_id = str(entry.get("node_id", "") or "")
            if not node_id:
                raise ValueError(f"partition map entry {idx} missing node_id")
            if node_id in seen_node_ids:
                raise ValueError(f"partition map entry {idx} duplicate node_id '{node_id}'")
            seen_node_ids.add(node_id)
            node = self._nodes.get(node_id)
            if node is None or node.role != "write":
                raise ValueError(f"partition map entry {idx} references non-writer node '{node_id}'")
            start = int(entry.get("partition_start", -1))
            end = int(entry.get("partition_end", -1))
            if start < 0 or end > _PARTITION_TOTAL or start >= end:
                raise ValueError(f"partition map entry {idx} has invalid range [{start}, {end})")
            if prev_end is None:
                if start != 0:
                    raise ValueError("partition map must start at 0")
            else:
                if start != prev_end:
                    raise ValueError(f"partition map gap/overlap between {prev_end} and {start}")
            prev_end = end
        if prev_end != _PARTITION_TOTAL:
            raise ValueError(f"partition map must end at {_PARTITION_TOTAL}, got {prev_end}")

    async def update_capability_flags(self, node_id: str, leader_reachable: bool, wal_reachable: bool) -> None:
        """Update leader_reachable and wal_reachable flags for a node."""
        node = self._nodes.get(node_id)
        if node is None:
            logger.debug("update_capability_flags: node %s not found", node_id)
            return
        node.leader_reachable = leader_reachable
        node.wal_reachable = wal_reachable
        logger.debug("update_capability_flags: node=%s leader_reachable=%s wal_reachable=%s",
                     node_id, leader_reachable, wal_reachable)
        await self._notify(node, "updated")

    async def _notify(self, node: NodeInfo, event_type: str) -> None:
        logger.debug("Notifying watchers: node=%s event=%s watchers=%d", node.node_id, event_type, len(self._watchers))
        for watcher in list(self._watchers):
            await watcher(node, event_type)
