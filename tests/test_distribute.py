# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pandas as pd

from core.cluster_runtime import ClusterRuntime, RuntimeConfig
from core.dataframe import DFrame
from core.message import Message, MessageType


async def _shutdown_runtime(runtime: ClusterRuntime) -> None:
    await runtime.heartbeat.stop()
    if hasattr(runtime.transport, "close"):
        await runtime.transport.close()
    if hasattr(runtime.registry, "close"):
        runtime.registry.close()


def test_seed_distributed_single_node_fallback(tmp_path) -> None:
    async def run() -> None:
        db_path = tmp_path / "single.db"
        runtime = ClusterRuntime(
            RuntimeConfig(
                node_id="single-w1",
                role="write",
                host="127.0.0.1",
                port=19600,
                enable_cluster=True,
                registry_backend="sqlite",
                transport_backend="tcp",
                db_path=str(db_path),
            )
        )
        await runtime.start()
        try:
            df = DFrame.from_runtime(runtime)

            async def chunks():
                yield pd.DataFrame({"x": [1, 2]})
                yield pd.DataFrame({"x": [3, 4]})

            total = await df._seed_distributed(chunks())
            snapshot = runtime._build_frame_snapshot(df._frame_id)

            assert total == 4
            assert snapshot.shape == (4, 1)
            assert snapshot["x"].tolist() == [1, 2, 3, 4]
        finally:
            await _shutdown_runtime(runtime)

    asyncio.run(run())


def test_seed_distributed_two_nodes_splits_rows(tmp_path) -> None:
    async def run() -> None:
        db_path = tmp_path / "two.db"
        runtime_a = ClusterRuntime(
            RuntimeConfig(
                node_id="dist-a",
                role="write",
                host="127.0.0.1",
                port=19610,
                enable_cluster=True,
                registry_backend="sqlite",
                transport_backend="tcp",
                db_path=str(db_path),
            )
        )
        runtime_b = ClusterRuntime(
            RuntimeConfig(
                node_id="dist-b",
                role="write",
                host="127.0.0.1",
                port=19611,
                enable_cluster=True,
                registry_backend="sqlite",
                transport_backend="tcp",
                db_path=str(db_path),
            )
        )

        await runtime_a.start()
        await runtime_b.start()
        try:
            df = DFrame.from_runtime(runtime_a)

            async def chunks():
                yield pd.DataFrame({"x": [1, 2]})
                yield pd.DataFrame({"x": [3, 4]})
                yield pd.DataFrame({"x": [5, 6]})
                yield pd.DataFrame({"x": [7, 8]})

            total = await df._seed_distributed(chunks())
            await asyncio.sleep(0.05)

            a_frame = runtime_a._build_frame_snapshot(df._frame_id)
            b_frame = runtime_b._build_frame_snapshot(df._frame_id)

            assert total == 8
            assert len(a_frame.index) == 4
            assert len(b_frame.index) == 4
            assert sorted(a_frame["x"].tolist() + b_frame["x"].tolist()) == [1, 2, 3, 4, 5, 6, 7, 8]
        finally:
            await _shutdown_runtime(runtime_a)
            await _shutdown_runtime(runtime_b)

    asyncio.run(run())


def test_from_csv_lazy_distribute_false_ignores_runtime(tmp_path) -> None:
    async def run() -> None:
        db_path = tmp_path / "ignore.db"
        csv_path = tmp_path / "data.csv"
        pd.DataFrame({"x": list(range(10))}).to_csv(csv_path, index=False)

        runtime_a = ClusterRuntime(
            RuntimeConfig(
                node_id="lazy-a",
                role="write",
                host="127.0.0.1",
                port=19620,
                enable_cluster=True,
                registry_backend="sqlite",
                transport_backend="tcp",
                db_path=str(db_path),
            )
        )
        runtime_b = ClusterRuntime(
            RuntimeConfig(
                node_id="lazy-b",
                role="write",
                host="127.0.0.1",
                port=19621,
                enable_cluster=True,
                registry_backend="sqlite",
                transport_backend="tcp",
                db_path=str(db_path),
            )
        )

        await runtime_a.start()
        await runtime_b.start()
        try:
            df = await DFrame.from_csv_lazy(str(csv_path), chunk_size=3, runtime=runtime_a, distribute=False)
            await asyncio.sleep(0.05)

            local = runtime_a._build_frame_snapshot(df._frame_id)
            remote = runtime_b._build_frame_snapshot(df._frame_id)

            assert len(local.index) == 10
            assert remote.empty
        finally:
            await _shutdown_runtime(runtime_a)
            await _shutdown_runtime(runtime_b)

    asyncio.run(run())


def test_from_csv_lazy_distribute_true_two_nodes(tmp_path) -> None:
    async def run() -> None:
        db_path = tmp_path / "dist.csv.db"
        csv_path = tmp_path / "dist.csv"
        pd.DataFrame({"x": list(range(12))}).to_csv(csv_path, index=False)

        runtime_a = ClusterRuntime(
            RuntimeConfig(
                node_id="csvdist-a",
                role="write",
                host="127.0.0.1",
                port=19630,
                enable_cluster=True,
                registry_backend="sqlite",
                transport_backend="tcp",
                db_path=str(db_path),
            )
        )
        runtime_b = ClusterRuntime(
            RuntimeConfig(
                node_id="csvdist-b",
                role="write",
                host="127.0.0.1",
                port=19631,
                enable_cluster=True,
                registry_backend="sqlite",
                transport_backend="tcp",
                db_path=str(db_path),
            )
        )

        await runtime_a.start()
        await runtime_b.start()
        try:
            df = await DFrame.from_csv_lazy(
                str(csv_path),
                chunk_size=3,
                runtime=runtime_a,
                distribute=True,
            )
            await asyncio.sleep(0.05)

            a_frame = runtime_a._build_frame_snapshot(df._frame_id)
            b_frame = runtime_b._build_frame_snapshot(df._frame_id)
            assert len(a_frame.index) + len(b_frame.index) == 12
            assert len(a_frame.index) == 6
            assert len(b_frame.index) == 6
        finally:
            await _shutdown_runtime(runtime_a)
            await _shutdown_runtime(runtime_b)

    asyncio.run(run())


def test_remote_chunk_injection_via_seed_chunk_message(tmp_path) -> None:
    async def run() -> None:
        db_path = tmp_path / "seed-msg.db"
        runtime_a = ClusterRuntime(
            RuntimeConfig(
                node_id="msg-a",
                role="write",
                host="127.0.0.1",
                port=19640,
                enable_cluster=True,
                registry_backend="sqlite",
                transport_backend="tcp",
                db_path=str(db_path),
            )
        )
        runtime_b = ClusterRuntime(
            RuntimeConfig(
                node_id="msg-b",
                role="write",
                host="127.0.0.1",
                port=19641,
                enable_cluster=True,
                registry_backend="sqlite",
                transport_backend="tcp",
                db_path=str(db_path),
            )
        )

        await runtime_a.start()
        await runtime_b.start()
        try:
            payload = {
                "frame_id": "frame-msg",
                "row_offset": 0,
                "data": {"x": [11, 12]},
            }
            message = Message.build(
                message_type=MessageType.SEED_CHUNK,
                sender_id="msg-a",
                sender_region="ap-southeast-1",
                payload=payload,
            )

            await runtime_a.transport.send("msg-b", message)
            await asyncio.sleep(0.05)
            first = runtime_b._build_frame_snapshot("frame-msg")
            assert first["x"].tolist() == [11, 12]

            await runtime_a.transport.send("msg-b", message)
            await asyncio.sleep(0.05)
            second = runtime_b._build_frame_snapshot("frame-msg")
            assert second["x"].tolist() == [11, 12]
        finally:
            await _shutdown_runtime(runtime_a)
            await _shutdown_runtime(runtime_b)

    asyncio.run(run())


def test_distributed_read_global_snapshot_merges_nodes(tmp_path) -> None:
    async def run() -> None:
        db_path = tmp_path / "merge.db"
        csv_path = tmp_path / "merge.csv"
        pd.DataFrame({"x": list(range(10))}).to_csv(csv_path, index=False)

        runtime_a = ClusterRuntime(
            RuntimeConfig(
                node_id="merge-a",
                role="write",
                host="127.0.0.1",
                port=19650,
                enable_cluster=True,
                registry_backend="sqlite",
                transport_backend="tcp",
                db_path=str(db_path),
            )
        )
        runtime_b = ClusterRuntime(
            RuntimeConfig(
                node_id="merge-b",
                role="write",
                host="127.0.0.1",
                port=19651,
                enable_cluster=True,
                registry_backend="sqlite",
                transport_backend="tcp",
                db_path=str(db_path),
            )
        )

        await runtime_a.start()
        await runtime_b.start()
        try:
            df = await DFrame.from_csv_lazy(
                str(csv_path),
                chunk_size=2,
                runtime=runtime_a,
                distribute=True,
            )
            await asyncio.sleep(0.05)

            merged = await runtime_a.read_global_snapshot_for(df._frame_id)
            assert len(merged.index) == 10
            assert sorted(merged["x"].tolist()) == list(range(10))
        finally:
            await _shutdown_runtime(runtime_a)
            await _shutdown_runtime(runtime_b)

    asyncio.run(run())


