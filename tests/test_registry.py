# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time

from core.registry import ClusterRegistry, NodeInfo


def test_registry_register_and_lookup() -> None:
    async def run() -> None:
        registry = ClusterRegistry("nats://127.0.0.1:4222")
        await registry.connect()
        node = NodeInfo(
            node_id="w1",
            host="127.0.0.1",
            port=19000,
            role="write",
            region="ap-southeast-1",
            partition_start=0,
            partition_end=100,
            last_seen=time.time(),
            lsn=0,
            status="healthy",
        )
        await registry.register(node)

        write_node = await registry.get_write_node()
        read_nodes = await registry.get_read_nodes()

        assert write_node is not None
        assert write_node.node_id == "w1"
        assert read_nodes == []

    asyncio.run(run())
