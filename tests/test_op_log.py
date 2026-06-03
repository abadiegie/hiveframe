# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

from core.message import Message, MessageType
from core.op_log import OperationLog


class _FakeTransport:
    def __init__(self, leader_log: OperationLog, *, corrupt_chunk: bool = False):
        self._leader_log = leader_log
        self._corrupt_chunk = corrupt_chunk

    async def request(self, _node_id: str, message: Message, timeout: float = 5.0) -> Message | None:
        _ = timeout
        request_id = str(message.payload.get("request_id", ""))
        if message.type == MessageType.OPLOG_PUSH:
            result = self._leader_log.accept_from_follower(list(message.payload.get("ops", [])))
            return Message.build(
                message_type=MessageType.OPLOG_PUSH_RESPONSE,
                sender_id="leader",
                sender_region="cluster",
                payload={"request_id": request_id, **result},
            )
        if message.type == MessageType.OPLOG_PULL:
            since = message.payload.get("since_op_id")
            ops = self._leader_log.export_since(str(since) if since is not None else None)
            return Message.build(
                message_type=MessageType.OPLOG_PULL_RESPONSE,
                sender_id="leader",
                sender_region="cluster",
                payload={"request_id": request_id, "ops": ops},
            )
        if message.type == MessageType.OPLOG_FULL_RESYNC:
            mode = str(message.payload.get("mode", "manifest"))
            snapshot_b64 = self._leader_log.export_sqlite_snapshot_b64()
            chunks, digest = self._leader_log.snapshot_chunks(snapshot_b64, 64)
            if self._corrupt_chunk and chunks:
                chunks[0] = chunks[0][::-1]

            if mode == "manifest":
                return Message.build(
                    message_type=MessageType.OPLOG_FULL_RESYNC_RESPONSE,
                    sender_id="leader",
                    sender_region="cluster",
                    payload={
                        "request_id": request_id,
                        "mode": "manifest",
                        "chunk_count": len(chunks),
                        "sha256": digest,
                        "leader_last_op_id": self._leader_log.get_last_acked_op_id(),
                    },
                )
            if mode == "chunk":
                idx = int(message.payload.get("chunk_index", 0))
                return Message.build(
                    message_type=MessageType.OPLOG_FULL_RESYNC_RESPONSE,
                    sender_id="leader",
                    sender_region="cluster",
                    payload={
                        "request_id": request_id,
                        "mode": "chunk",
                        "chunk_index": idx,
                        "chunk_b64": chunks[idx] if 0 <= idx < len(chunks) else "",
                    },
                )
            return Message.build(
                message_type=MessageType.OPLOG_FULL_RESYNC_RESPONSE,
                sender_id="leader",
                sender_region="cluster",
                payload={
                    "request_id": request_id,
                    "mode": "ops",
                    "ops": self._leader_log.export_all_acked(),
                    "leader_last_op_id": self._leader_log.get_last_acked_op_id(),
                },
            )
        return None


def test_operation_log_append_and_dedup(tmp_path) -> None:
    db_path = tmp_path / "oplog_append.db"
    log = OperationLog(str(db_path), node_id="n1", leader_epoch=3)

    op_id = log.append_local("registry", "leader_node_id", "node-a")
    assert op_id.startswith("3-")

    pending = log.get_pending_ops()
    assert len(pending) == 1
    assert pending[0]["key"] == "leader_node_id"

    assert log.dedup_check(op_id) is True
    assert log.dedup_check("999-00000000000000000001") is False

    log.close()


def test_operation_log_append_local_acked(tmp_path) -> None:
    db_path = tmp_path / "oplog_append_acked.db"
    log = OperationLog(str(db_path), node_id="leader", leader_epoch=4)

    op_id = log.append_local_acked("cluster_routing", "partition_map", {"writers": 2})
    assert op_id.startswith("4-")
    assert log.get_pending_ops() == []

    all_acked = log.export_all_acked()
    assert len(all_acked) == 1
    assert all_acked[0]["entity"] == "cluster_routing"
    assert all_acked[0]["key"] == "partition_map"

    log.close()


def test_operation_log_conflict_policy_leader_wins(tmp_path) -> None:
    db_path = tmp_path / "oplog_conflict.db"
    leader_log = OperationLog(str(db_path), node_id="leader", leader_epoch=1)

    # Existing leader value
    leader_log.apply_acked(
        [
            {
                "op_id": "1-00000000000000000001",
                "entity": "registry",
                "key": "leader_node_id",
                "value": "node-leader",
                "version": 1,
                "origin_node_id": "leader",
                "created_at": 1.0,
            }
        ]
    )

    # Follower attempts conflicting value for same key
    result = leader_log.accept_from_follower(
        [
            {
                "op_id": "1-00000000000000000002",
                "entity": "registry",
                "key": "leader_node_id",
                "value": "node-follower",
                "version": 1,
                "origin_node_id": "follower",
                "created_at": 2.0,
            }
        ]
    )

    assert len(result["acked_ops"]) == 0
    assert len(result["rejected_ops"]) == 1

    leader_log.close()


def test_operation_log_push_pull_and_full_resync(tmp_path) -> None:
    async def run() -> None:
        leader_db = tmp_path / "leader.db"
        follower_db = tmp_path / "follower.db"

        leader_log = OperationLog(str(leader_db), node_id="leader", leader_epoch=1)
        follower_log = OperationLog(str(follower_db), node_id="follower", leader_epoch=1)
        transport = _FakeTransport(leader_log)

        follower_log.append_local("registry", "route_map", {"a": [0, 500]})
        pushed = await follower_log.push_to_leader("leader", transport)
        assert pushed == 1

        # Pull back from leader and ensure follower sees acked state.
        pulled = await follower_log.pull_from_leader(
            "leader", transport, follower_log.get_last_acked_op_id()
        )
        assert pulled >= 0

        # Simulate follower local divergence then full-resync.
        follower_log.append_local("registry", "route_map", {"a": [0, 400]})
        await follower_log.full_resync_from_leader("leader", transport)

        pending_after_resync = follower_log.get_pending_ops()
        assert pending_after_resync == []
        assert follower_log.get_last_acked_op_id() is not None

        leader_log.close()
        follower_log.close()

    asyncio.run(run())


def test_operation_log_estimate_gap() -> None:
    assert OperationLog.estimate_gap("1-00000000000000000010", "1-00000000000000000025") == 15
    # Epoch jump is treated as a large divergence requiring conservative recovery.
    assert OperationLog.estimate_gap("1-00000000000000000010", "2-00000000000000000001") > 1_000_000


def test_operation_log_full_resync_falls_back_to_ops_on_checksum_mismatch(tmp_path) -> None:
    async def run() -> None:
        leader_db = tmp_path / "leader_checksum.db"
        follower_db = tmp_path / "follower_checksum.db"

        leader_log = OperationLog(str(leader_db), node_id="leader", leader_epoch=1)
        follower_log = OperationLog(str(follower_db), node_id="follower", leader_epoch=1)
        transport = _FakeTransport(leader_log, corrupt_chunk=True)

        leader_log.apply_acked(
            [
                {
                    "op_id": "1-00000000000000000003",
                    "entity": "registry",
                    "key": "leader_node_id",
                    "value": "leader",
                    "version": 1,
                    "origin_node_id": "leader",
                    "created_at": 1.0,
                }
            ]
        )

        await follower_log.full_resync_from_leader("leader", transport)
        assert follower_log.get_last_acked_op_id() == "1-00000000000000000003"

        leader_log.close()
        follower_log.close()

    asyncio.run(run())


