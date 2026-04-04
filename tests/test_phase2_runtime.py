# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os

import pandas as pd
import pytest

from core.cluster_runtime import ClusterRuntime, RuntimeConfig
from core.message import Message, MessageType
from core.transaction import Operation

# Integration tests only run when env var is set.
_RUN_INTEGRATION = os.getenv("RUN_PHASE2_INTEGRATION", "0") == "1"
integration = pytest.mark.skipif(not _RUN_INTEGRATION, reason="Set RUN_PHASE2_INTEGRATION=1 to run")


def test_runtime_start_registers_node_standalone_default() -> None:
    async def run() -> None:
        runtime = ClusterRuntime(
            RuntimeConfig(node_id="node-1", role="write", host="127.0.0.1", port=19100)
        )
        await runtime.start()
        node = await runtime.registry.get_node("node-1")
        assert node is not None
        assert node.role == "write"
        assert runtime.config.enable_cluster is False

    asyncio.run(run())


def test_transport_broadcast_inmemory_cluster_enabled() -> None:
    async def run() -> None:
        runtime_a = ClusterRuntime(
            RuntimeConfig(node_id="a", role="write", port=19110, enable_cluster=True)
        )
        runtime_b = ClusterRuntime(
            RuntimeConfig(node_id="b", role="read", port=19111, enable_cluster=True)
        )

        seen: list[str] = []

        async def handler(msg: Message) -> None:
            seen.append(msg.type.value)

        runtime_b.transport.on_message(handler)
        await runtime_a.start()
        await runtime_b.start()

        payload = Message.build(
            message_type=MessageType.HEARTBEAT,
            sender_id="a",
            sender_region="ap-southeast-1",
            payload={"n": 1},
        )
        await runtime_a.transport.broadcast(payload)
        await asyncio.sleep(0.01)

        assert "heartbeat" in seen

    asyncio.run(run())


def test_replication_wired_to_coordinator() -> None:
    """Write node coordinator has replication_manager wired after ClusterRuntime init."""
    runtime = ClusterRuntime(
        RuntimeConfig(node_id="writer-1", role="write", enable_cluster=True)
    )
    assert runtime.coordinator.replication_manager is runtime.replication


def test_runtime_sqlite_registry_uses_configured_db_path(tmp_path) -> None:
    async def run() -> None:
        db_path = tmp_path / "cluster-registry.db"
        runtime = ClusterRuntime(
            RuntimeConfig(
                node_id="sqlite-w1",
                role="write",
                host="127.0.0.1",
                port=19125,
                registry_backend="sqlite",
                db_path=str(db_path),
            )
        )

        await runtime.start()

        assert runtime.registry.db_path == str(db_path)
        assert db_path.exists()
        node = await runtime.registry.get_node("sqlite-w1")
        assert node is not None
        assert node.node_id == "sqlite-w1"

        runtime.registry.close()

    asyncio.run(run())


def test_runtime_tcp_transport_global_snapshot_roundtrip() -> None:
    async def run() -> None:
        runtime_a = ClusterRuntime(
            RuntimeConfig(
                node_id="w-a",
                role="write",
                host="127.0.0.1",
                port=19130,
                enable_cluster=True,
                transport_backend="tcp",
            )
        )
        runtime_b = ClusterRuntime(
            RuntimeConfig(
                node_id="w-b",
                role="write",
                host="127.0.0.1",
                port=19131,
                enable_cluster=True,
                transport_backend="tcp",
            )
        )

        await runtime_a.start()
        await runtime_b.start()

        runtime_a.coordinator.write_node._df = pd.DataFrame({"left": [1]})
        runtime_b.coordinator.write_node._df = pd.DataFrame({"right": [2]})

        merged = await runtime_a.read_global_snapshot()

        assert len(merged.index) == 2
        assert set(merged.columns) == {"left", "right"}
        assert 1 in merged["left"].dropna().tolist()
        assert 2 in merged["right"].dropna().tolist()

    asyncio.run(run())


@integration
def test_replication_delta_received_by_read_node() -> None:
    """Integration: write on writer-node delta is received by read-node in cluster mode."""
    received: list[dict] = []

    async def run() -> None:
        writer = ClusterRuntime(
            RuntimeConfig(node_id="w1", role="write", port=19120, enable_cluster=True)
        )
        reader = ClusterRuntime(
            RuntimeConfig(node_id="r1", role="read", port=19121, enable_cluster=True)
        )
        reader.replication._apply_entry = lambda entry: received.append(entry)

        await writer.start()
        await reader.start()

        ops = [Operation(cell_id="x_0", old_value=None, new_value="hello", author_type="human", author_id="u")]
        writer.coordinator.submit(ops)

        await asyncio.sleep(0.1)
        assert len(received) > 0

    asyncio.run(run())


@integration
def test_nats_registry_real_connect() -> None:
    """Integration: registry connects to real NATS at NATS_URL env (requires nats-py + server)."""
    nats_url = os.getenv("NATS_URL", "nats://127.0.0.1:4222")
    nats = pytest.importorskip("nats")
    _ = nats

    async def run() -> None:
        runtime = ClusterRuntime(
            RuntimeConfig(
                node_id="n1",
                role="write",
                nats_url=nats_url,
                registry_backend="nats",
                enable_cluster=True,
            )
        )
        await runtime.start()
        node = await runtime.registry.get_node("n1")
        assert node is not None

    asyncio.run(run())


