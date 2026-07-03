# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time

import pytest

from core.sqlite_registry import NodeInfo, SQLiteRegistry, _PARTITION_TOTAL

_HALF = _PARTITION_TOTAL // 2
_QUARTER = _PARTITION_TOTAL // 4


def _make_node(node_id: str, port: int) -> NodeInfo:
    return NodeInfo(
        node_id=node_id,
        host="127.0.0.1",
        port=port,
        role="write",
        region="ap-southeast-1",
        partition_start=0,
        partition_end=_PARTITION_TOTAL,
        last_seen=time.time(),
        lsn=0,
        status="healthy",
    )


def test_sqlite_registry_rebalances_and_marks_failed(tmp_path) -> None:
    async def run() -> None:
        registry = SQLiteRegistry(str(tmp_path / "registry.db"))
        await registry.connect()
        assert registry._conn is not None
        journal_mode = registry._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert str(journal_mode).lower() == "wal"

        events: list[tuple[str, str]] = []

        async def watcher(node: NodeInfo, event: str) -> None:
            events.append((event, node.node_id))

        await registry.watch(watcher)

        await registry.register(_make_node("w1", 19300))
        await registry.register(_make_node("w2", 19301))

        first = await registry.get_node("w1")
        second = await registry.get_node("w2")
        assert first is not None and second is not None
        assert (first.partition_start, first.partition_end) == (0, _HALF)
        assert (second.partition_start, second.partition_end) == (_HALF, _PARTITION_TOTAL)

        write_node = await registry.get_write_node()
        assert write_node is not None
        assert write_node.node_id == "w1"
        assert registry.get_owner_for_row(_HALF + 1).node_id == "w2"

        await registry.update_lsn("w2", 9)
        updated = await registry.get_node("w2")
        assert updated is not None
        assert updated.lsn == 9
        assert ("updated", "w2") not in events

        await registry.mark_failed("w1")
        failed = await registry.get_node("w1")
        survivor = await registry.get_node("w2")
        assert failed is not None and failed.status == "failed"
        assert survivor is not None
        assert (survivor.partition_start, survivor.partition_end) == (0, _PARTITION_TOTAL)
        assert registry.get_owner_for_row(10).node_id == "w2"
        assert ("joined", "w1") in events
        assert ("joined", "w2") in events
        assert ("failed", "w1") in events
        registry.close()
        assert registry._conn is None

    asyncio.run(run())


def test_sqlite_registry_single_writer_guard_blocks_critical_metadata(tmp_path) -> None:
    async def run() -> None:
        registry = SQLiteRegistry(str(tmp_path / "registry_guard.db"))
        registry.set_write_guard_context(local_node_id="follower-1", is_leader=False)
        await registry.connect()

        node = _make_node("w1", 19310)
        await registry.register(node)

        # Liveness-only update is allowed.
        await registry.register(
            NodeInfo(
                node_id="w1",
                host="127.0.0.1",
                port=19310,
                role="write",
                region="ap-southeast-1",
                partition_start=0,
                partition_end=_PARTITION_TOTAL,
                last_seen=time.time(),
                lsn=1,
                status="healthy",
            )
        )

        # Critical metadata change (partition ownership) must be blocked.
        try:
            await registry.register(
                NodeInfo(
                    node_id="w1",
                    host="127.0.0.1",
                    port=19310,
                    role="write",
                    region="ap-southeast-1",
                    partition_start=10,
                    partition_end=_PARTITION_TOTAL - 10,
                    last_seen=time.time(),
                    lsn=2,
                    status="healthy",
                )
            )
            raise AssertionError("single-writer guard should block partition ownership updates")
        except RuntimeError as exc:
            assert "single-writer guard" in str(exc)

        registry.close()

    asyncio.run(run())


def test_sqlite_registry_mark_suspect_rebalances_writers(tmp_path) -> None:
    async def run() -> None:
        registry = SQLiteRegistry(str(tmp_path / "registry_suspect.db"))
        await registry.connect()

        await registry.register(_make_node("w1", 19320))
        await registry.register(_make_node("w2", 19321))

        await registry.mark_suspect("w1")
        suspect = await registry.get_node("w1")
        survivor = await registry.get_node("w2")
        assert suspect is not None and suspect.status == "suspect"
        assert survivor is not None
        assert (survivor.partition_start, survivor.partition_end) == (0, _PARTITION_TOTAL)
        assert registry.get_owner_for_row(_QUARTER).node_id == "w2"

        registry.close()

    asyncio.run(run())


def test_sqlite_registry_mark_healthy_restores_writer_partitioning(tmp_path) -> None:
    async def run() -> None:
        registry = SQLiteRegistry(str(tmp_path / "registry_healthy.db"))
        await registry.connect()

        await registry.register(_make_node("w1", 19350))
        await registry.register(_make_node("w2", 19351))

        await registry.mark_failed("w1")
        await registry.mark_healthy("w1")

        w1 = await registry.get_node("w1")
        w2 = await registry.get_node("w2")
        assert w1 is not None and w1.status == "healthy"
        assert w2 is not None
        assert (w1.partition_start, w1.partition_end) == (0, _HALF)
        assert (w2.partition_start, w2.partition_end) == (_HALF, _PARTITION_TOTAL)

        registry.close()

    asyncio.run(run())


def test_sqlite_registry_apply_partition_map(tmp_path) -> None:
    async def run() -> None:
        registry = SQLiteRegistry(str(tmp_path / "registry_route.db"))
        await registry.connect()

        await registry.register(_make_node("w1", 19330))
        await registry.register(_make_node("w2", 19331))

        applied = await registry.apply_partition_map(
            [
                {"node_id": "w1", "partition_start": 0, "partition_end": _QUARTER},
                {"node_id": "w2", "partition_start": _QUARTER, "partition_end": _PARTITION_TOTAL},
            ]
        )
        assert applied == 2

        w1 = await registry.get_node("w1")
        w2 = await registry.get_node("w2")
        assert w1 is not None and (w1.partition_start, w1.partition_end) == (0, _QUARTER)
        assert w2 is not None and (w2.partition_start, w2.partition_end) == (_QUARTER, _PARTITION_TOTAL)

        registry.close()

    asyncio.run(run())


def test_sqlite_registry_apply_partition_map_rejects_gap_and_keeps_previous(tmp_path) -> None:
    async def run() -> None:
        registry = SQLiteRegistry(str(tmp_path / "registry_route_invalid.db"))
        await registry.connect()

        await registry.register(_make_node("w1", 19340))
        await registry.register(_make_node("w2", 19341))

        await registry.apply_partition_map(
            [
                {"node_id": "w1", "partition_start": 0, "partition_end": _HALF},
                {"node_id": "w2", "partition_start": _HALF, "partition_end": _PARTITION_TOTAL},
            ]
        )
        with pytest.raises(ValueError, match="gap/overlap"):
            await registry.apply_partition_map(
                [
                    {"node_id": "w1", "partition_start": 0, "partition_end": _QUARTER},
                    {"node_id": "w2", "partition_start": _QUARTER + 50, "partition_end": _PARTITION_TOTAL},
                ]
            )

        w1 = await registry.get_node("w1")
        w2 = await registry.get_node("w2")
        assert w1 is not None and (w1.partition_start, w1.partition_end) == (0, _HALF)
        assert w2 is not None and (w2.partition_start, w2.partition_end) == (_HALF, _PARTITION_TOTAL)

        registry.close()

    asyncio.run(run())