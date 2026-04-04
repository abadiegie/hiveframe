# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time

from core.sqlite_registry import NodeInfo, SQLiteRegistry


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

        await registry.register(
            NodeInfo(
                node_id="w1",
                host="127.0.0.1",
                port=19300,
                role="write",
                region="ap-southeast-1",
                partition_start=0,
                partition_end=1000,
                last_seen=time.time(),
                lsn=0,
                status="healthy",
            )
        )
        await registry.register(
            NodeInfo(
                node_id="w2",
                host="127.0.0.1",
                port=19301,
                role="write",
                region="ap-southeast-1",
                partition_start=0,
                partition_end=1000,
                last_seen=time.time(),
                lsn=0,
                status="healthy",
            )
        )

        first = await registry.get_node("w1")
        second = await registry.get_node("w2")
        assert first is not None and second is not None
        assert (first.partition_start, first.partition_end) == (0, 500)
        assert (second.partition_start, second.partition_end) == (500, 1000)

        write_node = await registry.get_write_node()
        assert write_node is not None
        assert write_node.node_id == "w1"
        assert registry.get_owner_for_row(750).node_id == "w2"

        await registry.update_lsn("w2", 9)
        updated = await registry.get_node("w2")
        assert updated is not None
        assert updated.lsn == 9
        assert ("updated", "w2") in events

        await registry.mark_failed("w1")
        failed = await registry.get_node("w1")
        survivor = await registry.get_node("w2")
        assert failed is not None and failed.status == "failed"
        assert survivor is not None
        assert (survivor.partition_start, survivor.partition_end) == (0, 1000)
        assert registry.get_owner_for_row(10).node_id == "w2"
        assert ("joined", "w1") in events
        assert ("joined", "w2") in events
        assert ("failed", "w1") in events
        registry.close()
        assert registry._conn is None

    asyncio.run(run())

