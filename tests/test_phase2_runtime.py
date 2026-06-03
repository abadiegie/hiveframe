# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
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


def test_non_transactional_submit_does_not_trigger_replication() -> None:
    runtime = ClusterRuntime(
        RuntimeConfig(node_id="writer-no-tx", role="write", enable_cluster=True)
    )

    calls: list[tuple[int, dict]] = []

    async def fake_replicate(lsn: int, tx_data: dict) -> bool:
        calls.append((lsn, tx_data))
        return True

    runtime.replication.replicate_tx = fake_replicate  # type: ignore[method-assign]

    ops = [Operation(cell_id="x_0", old_value=None, new_value="hello", author_type="human", author_id="u")]
    ok = runtime.coordinator.submit_non_transactional(ops)

    assert ok is True
    assert calls == []
    assert runtime.coordinator.wal._entries == []
    assert runtime.coordinator.write_node.get("x_0") == "hello"


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


def test_runtime_tcp_seed_hosts_bootstrap_connects_to_seed() -> None:
    async def run() -> None:
        seed = ClusterRuntime(
            RuntimeConfig(
                node_id="seed-node",
                role="write",
                host="127.0.0.1",
                port=19140,
                enable_cluster=True,
                transport_backend="tcp",
            )
        )
        joiner = ClusterRuntime(
            RuntimeConfig(
                node_id="joiner-node",
                role="read",
                host="127.0.0.1",
                port=19141,
                enable_cluster=True,
                transport_backend="tcp",
                seed_hosts=["127.0.0.1:19140"],
            )
        )

        await seed.start()
        await joiner.start()
        await asyncio.sleep(0.02)

        assert "seed-node" in joiner.transport._connected_nodes
        assert "joiner-node" in seed.transport._connected_nodes

        await joiner.heartbeat.stop()
        await seed.heartbeat.stop()
        await joiner.transport.close()
        await seed.transport.close()

    asyncio.run(run())


def test_runtime_get_cluster_stats_exposes_phase4_metrics() -> None:
    async def run() -> None:
        runtime = ClusterRuntime(
            RuntimeConfig(
                node_id="metrics-node",
                role="write",
                enable_cluster=True,
                is_leader=True,
                leader_node_id="metrics-node",
            )
        )
        await runtime.start()
        stats = runtime.get_cluster_stats()

        assert "coordinator" in stats
        assert "replication_lag_ops" in stats
        assert "outbox_depth" in stats
        assert "conflict_count" in stats
        assert "full_resync_count" in stats

        assert "replication_lag_ops" in stats["coordinator"]
        assert "outbox_depth" in stats["coordinator"]
        assert "conflict_count" in stats["coordinator"]
        assert "full_resync_count" in stats["coordinator"]

        await runtime.stop()

    asyncio.run(run())


def test_required_cluster_gate_times_out_without_leader() -> None:
    async def run() -> None:
        follower = ClusterRuntime(
            RuntimeConfig(
                node_id="follower-timeout",
                role="read",
                enable_cluster=True,
                required_cluster=True,
                leader_node_id="missing-leader",
                heartbeat_timeout_ms=250,
            )
        )
        with pytest.raises(RuntimeError, match="required_cluster startup gate failed"):
            await follower.start()

        # Best-effort cleanup if startup failed mid-sequence.
        try:
            await follower.stop()
        except Exception:
            pass

    asyncio.run(run())


def test_required_cluster_gate_succeeds_when_leader_reachable() -> None:
    async def run() -> None:
        leader = ClusterRuntime(
            RuntimeConfig(
                node_id="leader-ready",
                role="write",
                enable_cluster=True,
                is_leader=True,
                leader_node_id="leader-ready",
            )
        )
        follower = ClusterRuntime(
            RuntimeConfig(
                node_id="follower-ready",
                role="read",
                enable_cluster=True,
                required_cluster=True,
                leader_node_id="leader-ready",
            )
        )

        await leader.start()
        await follower.start()

        node = await follower.registry.get_node("follower-ready")
        assert node is not None
        assert node.leader_reachable is True

        await follower.stop()
        await leader.stop()

    asyncio.run(run())


def test_required_cluster_min_nodes_gate_blocks_without_quorum() -> None:
    async def run() -> None:
        leader = ClusterRuntime(
            RuntimeConfig(
                node_id="leader-quorum",
                role="write",
                enable_cluster=True,
                is_leader=True,
                leader_node_id="leader-quorum",
            )
        )
        follower = ClusterRuntime(
            RuntimeConfig(
                node_id="follower-quorum",
                role="read",
                enable_cluster=True,
                required_cluster=True,
                required_cluster_min_nodes=3,
                leader_node_id="leader-quorum",
                heartbeat_timeout_ms=250,
            )
        )

        await leader.start()
        with pytest.raises(RuntimeError, match="required_cluster startup gate failed"):
            await follower.start()

        await leader.stop()
        try:
            await follower.stop()
        except Exception:
            pass

    asyncio.run(run())


def test_runtime_writes_audit_events(tmp_path) -> None:
    async def run() -> None:
        audit_path = tmp_path / "audit.log"
        runtime = ClusterRuntime(
            RuntimeConfig(
                node_id="audit-node",
                role="write",
                enable_cluster=True,
                is_leader=True,
                leader_node_id="audit-node",
                audit_log_path=str(audit_path),
            )
        )

        await runtime.start()
        await runtime.promote_to_leader()
        await runtime.stop()

        assert audit_path.exists()
        lines = [line.strip() for line in audit_path.read_text().splitlines() if line.strip()]
        assert lines

        events = [json.loads(line).get("event") for line in lines]
        assert "JOIN" in events
        assert "LEADER_CHANGE" in events
        assert "LEAVE" in events

    asyncio.run(run())


def test_runtime_tcp_chunked_full_resync_via_join_membership_update() -> None:
    async def run() -> None:
        leader = ClusterRuntime(
            RuntimeConfig(
                node_id="leader-chunked",
                role="write",
                host="127.0.0.1",
                port=19160,
                enable_cluster=True,
                transport_backend="tcp",
                is_leader=True,
                leader_node_id="leader-chunked",
            )
        )

        follower = ClusterRuntime(
            RuntimeConfig(
                node_id="follower-chunked",
                role="read",
                host="127.0.0.1",
                port=19161,
                enable_cluster=True,
                transport_backend="tcp",
                leader_node_id="leader-chunked",
                seed_hosts=["127.0.0.1:19160"],
                full_resync_threshold_ops=1,
            )
        )

        await leader.start()

        assert leader.op_log is not None
        # Inflate leader snapshot so manifest/chunk path is exercised with many chunks.
        leader_ops = [
            {
                "op_id": f"1-{i:020d}",
                "entity": "registry",
                "key": f"route-{i}",
                "value": {"payload": "x" * 4096, "idx": i},
                "version": 1,
                "origin_node_id": "leader-chunked",
                "created_at": float(i),
            }
            for i in range(1, 180)
        ]
        leader.op_log.apply_acked(leader_ops)
        expected_last = leader.op_log.get_last_acked_op_id()
        assert expected_last is not None

        await follower.start()
        assert follower.op_log is not None

        fallback_used = {"value": False}
        original_fallback = follower.op_log._fallback_full_resync_ops

        async def _wrapped_fallback(leader_node_id: str, transport) -> None:
            fallback_used["value"] = True
            await original_fallback(leader_node_id, transport)

        follower.op_log._fallback_full_resync_ops = _wrapped_fallback  # type: ignore[method-assign]

        # Force a fresh JOIN after both replication handlers are active.
        join_msg = Message.build(
            message_type=MessageType.JOIN,
            sender_id="follower-chunked",
            sender_region="ap-southeast-1",
            payload={"host": "127.0.0.1", "port": 19161},
        )
        await follower.transport.send("leader-chunked", join_msg)
        await asyncio.sleep(0.25)

        got_last = follower.op_log.get_last_acked_op_id()
        assert got_last == expected_last
        assert fallback_used["value"] is False

        await follower.stop()
        await leader.stop()
        await follower.heartbeat.stop()
        await leader.heartbeat.stop()
        await follower.transport.close()
        await leader.transport.close()

    asyncio.run(run())


def test_required_cluster_role_quorum_gate() -> None:
    async def run() -> None:
        leader = ClusterRuntime(
            RuntimeConfig(
                node_id="leader-role-quorum",
                role="write",
                enable_cluster=True,
                is_leader=True,
                leader_node_id="leader-role-quorum",
            )
        )
        reader = ClusterRuntime(
            RuntimeConfig(
                node_id="reader-role-quorum",
                role="read",
                enable_cluster=True,
                required_cluster=True,
                required_cluster_min_nodes=2,
                required_cluster_min_write_nodes=1,
                required_cluster_min_read_nodes=1,
                leader_node_id="leader-role-quorum",
            )
        )

        await leader.start()
        await reader.start()
        await reader.stop()
        await leader.stop()

    asyncio.run(run())


def test_audit_log_rotation(tmp_path) -> None:
    async def run() -> None:
        audit_path = tmp_path / "audit-rotate.log"
        runtime = ClusterRuntime(
            RuntimeConfig(
                node_id="audit-rotate-node",
                role="write",
                enable_cluster=True,
                is_leader=True,
                leader_node_id="audit-rotate-node",
                audit_log_path=str(audit_path),
                audit_log_max_bytes=200,
                audit_log_backup_count=2,
            )
        )
        await runtime.start()

        for i in range(20):
            runtime._audit_event("JOIN", {"i": i, "padding": "x" * 40})

        await runtime.stop()

        backup_1 = tmp_path / "audit-rotate.log.1"
        backup_2 = tmp_path / "audit-rotate.log.2"
        assert audit_path.exists() or backup_1.exists()
        assert backup_1.exists() or backup_2.exists()

    asyncio.run(run())


def test_membership_update_triggers_full_resync_on_large_gap(tmp_path) -> None:
    async def run() -> None:
        follower = ClusterRuntime(
            RuntimeConfig(
                node_id="follower-gap",
                role="read",
                enable_cluster=True,
                leader_node_id="leader-gap",
                full_resync_threshold_ops=10,
                db_path=str(tmp_path / "follower_gap_registry.db"),
            )
        )
        assert follower.op_log is not None
        follower.op_log.apply_acked(
            [
                {
                    "op_id": "1-00000000000000000001",
                    "entity": "registry",
                    "key": "route",
                    "value": {"a": 1},
                    "version": 1,
                    "origin_node_id": "leader-gap",
                    "created_at": 1.0,
                }
            ]
        )

        called = {"n": 0}

        async def fake_resync(leader_node_id: str, _transport) -> None:
            called["n"] += 1
            assert leader_node_id == "leader-gap"

        follower.op_log.full_resync_from_leader = fake_resync  # type: ignore[method-assign]

        msg = Message.build(
            message_type=MessageType.MEMBERSHIP_UPDATE,
            sender_id="leader-gap",
            sender_region="ap-southeast-1",
            payload={
                "leader_node_id": "leader-gap",
                "leader_last_op_id": "1-00000000000000000100",
            },
        )
        await follower._handle_membership_update(msg)
        assert called["n"] == 1

        await follower.stop()

    asyncio.run(run())


def test_route_update_applies_partition_map_to_registry() -> None:
    async def run() -> None:
        runtime = ClusterRuntime(
            RuntimeConfig(
                node_id="route-follower",
                role="write",
                enable_cluster=True,
                leader_node_id="route-leader",
            )
        )
        await runtime.start()

        await runtime.registry.register(
            runtime._NodeInfo(
                node_id="route-w2",
                host="127.0.0.1",
                port=19210,
                role="write",
                region="ap-southeast-1",
                partition_start=0,
                partition_end=1000,
                last_seen=1.0,
                lsn=0,
                status="healthy",
            )
        )

        msg = Message.build(
            message_type=MessageType.ROUTE_UPDATE,
            sender_id="route-leader",
            sender_region="ap-southeast-1",
            payload={
                "partition_map": [
                    {"node_id": "route-follower", "partition_start": 0, "partition_end": 300},
                    {"node_id": "route-w2", "partition_start": 300, "partition_end": 1000},
                ]
            },
        )
        await runtime._handle_route_update(msg)

        local = await runtime.registry.get_node("route-follower")
        other = await runtime.registry.get_node("route-w2")
        assert local is not None
        assert other is not None
        assert (local.partition_start, local.partition_end) == (0, 300)
        assert (other.partition_start, other.partition_end) == (300, 1000)
        assert runtime.route_write(900).node_id == "route-w2"

        await runtime.stop()

    asyncio.run(run())


def test_route_update_rejects_invalid_partition_map_and_keeps_previous() -> None:
    async def run() -> None:
        runtime = ClusterRuntime(
            RuntimeConfig(
                node_id="route-follower-invalid",
                role="write",
                enable_cluster=True,
                leader_node_id="route-leader",
            )
        )
        await runtime.start()

        await runtime.registry.register(
            runtime._NodeInfo(
                node_id="route-w2-invalid",
                host="127.0.0.1",
                port=19211,
                role="write",
                region="ap-southeast-1",
                partition_start=0,
                partition_end=1000,
                last_seen=1.0,
                lsn=0,
                status="healthy",
            )
        )

        valid = Message.build(
            message_type=MessageType.ROUTE_UPDATE,
            sender_id="route-leader",
            sender_region="ap-southeast-1",
            payload={
                "partition_map": [
                    {"node_id": "route-follower-invalid", "partition_start": 0, "partition_end": 500},
                    {"node_id": "route-w2-invalid", "partition_start": 500, "partition_end": 1000},
                ]
            },
        )
        await runtime._handle_route_update(valid)

        invalid = Message.build(
            message_type=MessageType.ROUTE_UPDATE,
            sender_id="route-leader",
            sender_region="ap-southeast-1",
            payload={
                "partition_map": [
                    {"node_id": "route-follower-invalid", "partition_start": 0, "partition_end": 400},
                    {"node_id": "route-w2-invalid", "partition_start": 450, "partition_end": 1000},
                ]
            },
        )
        await runtime._handle_route_update(invalid)

        local = await runtime.registry.get_node("route-follower-invalid")
        other = await runtime.registry.get_node("route-w2-invalid")
        assert local is not None
        assert other is not None
        assert (local.partition_start, local.partition_end) == (0, 500)
        assert (other.partition_start, other.partition_end) == (500, 1000)

        await runtime.stop()

    asyncio.run(run())


def test_cluster_write_raises_when_remote_owner_not_resolvable() -> None:
    async def run() -> None:
        runtime = ClusterRuntime(
            RuntimeConfig(
                node_id="strict-local",
                role="write",
                enable_cluster=True,
                leader_node_id="strict-local",
                is_leader=True,
            )
        )
        await runtime.start()
        await runtime.registry.register(
            runtime._NodeInfo(
                node_id="strict-remote",
                host="127.0.0.1",
                port=19212,
                role="write",
                region="ap-southeast-1",
                partition_start=0,
                partition_end=1000,
                last_seen=1.0,
                lsn=0,
                status="healthy",
            )
        )

        route = Message.build(
            message_type=MessageType.ROUTE_UPDATE,
            sender_id="strict-local",
            sender_region="ap-southeast-1",
            payload={
                "partition_map": [
                    {"node_id": "strict-remote", "partition_start": 0, "partition_end": 500},
                    {"node_id": "strict-local", "partition_start": 500, "partition_end": 1000},
                ]
            },
        )
        await runtime._handle_route_update(route)

        from core.dataframe import DFrame

        frame = DFrame.from_runtime(runtime)
        with pytest.raises(RuntimeError, match="Remote write proxy/runtime is required"):
            frame["x"] = [42]

        await runtime.stop()

    asyncio.run(run())


def test_leader_metadata_ops_replicate_to_follower_via_oplog_pull() -> None:
    async def run() -> None:
        leader = ClusterRuntime(
            RuntimeConfig(
                node_id="meta-leader",
                role="write",
                enable_cluster=True,
                is_leader=True,
                leader_node_id="meta-leader",
                sync_interval_ms=50,
            )
        )
        follower = ClusterRuntime(
            RuntimeConfig(
                node_id="meta-follower",
                role="read",
                enable_cluster=True,
                leader_node_id="meta-leader",
                sync_interval_ms=50,
            )
        )

        await leader.start()
        await follower.start()
        await asyncio.sleep(0.3)

        assert leader.op_log is not None
        assert follower.op_log is not None
        leader_ops = leader.op_log.export_all_acked()
        follower_ops = follower.op_log.export_all_acked()

        assert any(op["entity"] == "cluster_routing" and op["key"] == "partition_map" for op in leader_ops)
        assert any(op["entity"] == "cluster_membership" and op["key"] == "members" for op in leader_ops)
        assert any(op["entity"] == "cluster_routing" and op["key"] == "partition_map" for op in follower_ops)

        stats = leader.get_cluster_stats()
        assert "oplog_pending_ops" in stats
        assert "oplog_last_acked_op_id" in stats

        await follower.stop()
        await leader.stop()

    asyncio.run(run())


