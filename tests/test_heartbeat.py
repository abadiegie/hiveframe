# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from dataclasses import dataclass

from core.heartbeat import HeartbeatManager


@dataclass
class _Node:
    node_id: str
    last_seen: float


class _FakeRegistry:
    def __init__(self, nodes: list[_Node]) -> None:
        self._nodes = {n.node_id: n for n in nodes}
        self.failed: list[str] = []
        self.suspect: list[str] = []

    async def get_read_nodes(self) -> list[_Node]:
        return []

    async def get_write_nodes(self) -> list[_Node]:
        return list(self._nodes.values())

    async def mark_failed(self, node_id: str) -> None:
        self.failed.append(node_id)

    async def mark_suspect(self, node_id: str) -> None:
        self.suspect.append(node_id)


class _FakeTransport:
    async def broadcast(self, message, exclude=None) -> None:  # pragma: no cover - not used
        _ = (message, exclude)


def test_heartbeat_transitions_suspect_then_failed() -> None:
    async def run() -> None:
        node = _Node(node_id="writer-1", last_seen=time.time() - 1.2)
        registry = _FakeRegistry([node])
        manager = HeartbeatManager(
            node_id="self",
            node_region="local",
            registry=registry,
            transport=_FakeTransport(),
            interval_s=0.1,
            timeout_s=1.0,
            suspect_multiplier=2.0,
        )

        seen: list[tuple[str, str, str]] = []
        manager.on_status_change(lambda node_id, old, new: seen.append((node_id, old, new)))

        await manager._check_timeouts()
        assert registry.suspect == ["writer-1"]
        assert registry.failed == []

        node.last_seen = time.time() - 2.5
        await manager._check_timeouts()

        assert registry.failed == ["writer-1"]
        assert seen == [
            ("writer-1", "healthy", "suspect"),
            ("writer-1", "suspect", "failed"),
        ]

    asyncio.run(run())


def test_heartbeat_transitions_back_to_healthy() -> None:
    async def run() -> None:
        node = _Node(node_id="writer-2", last_seen=time.time() - 1.3)
        registry = _FakeRegistry([node])
        manager = HeartbeatManager(
            node_id="self",
            node_region="local",
            registry=registry,
            transport=_FakeTransport(),
            interval_s=0.1,
            timeout_s=1.0,
            suspect_multiplier=2.0,
        )

        seen: list[tuple[str, str, str]] = []
        manager.on_status_change(lambda node_id, old, new: seen.append((node_id, old, new)))

        await manager._check_timeouts()
        node.last_seen = time.time()
        await manager._check_timeouts()

        assert ("writer-2", "suspect", "healthy") in seen
        assert registry.failed == []

    asyncio.run(run())

