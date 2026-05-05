# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Runtime wiring for Phase 2 control/data plane over registry and transport."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
import uuid
from typing import Any
import logging

import pandas as pd

from .coordinator import TransactionCoordinator
from .heartbeat import HeartbeatManager
from .sqlite_registry import SQLiteRegistry
from .tcp_transport import TCPTransport
from .replication import ReplicationManager
from .registry import NodeInfo as MemNodeInfo
from .sqlite_registry import NodeInfo as SQLiteNodeInfo


logger = logging.getLogger("core.cluster_runtime")


@dataclass(slots=True)
class RuntimeConfig:
    node_id: str
    role: str
    region: str = "ap-southeast-1"
    host: str = "127.0.0.1"
    port: int = 19000
    partition_start: int = 0
    partition_end: int = 1000
    nats_url: str = "nats://127.0.0.1:4222"
    db_path: str = ".hiveframe/registry.db"
    enable_cluster: bool = False
    registry_backend: str = "memory"  # memory | nats | sqlite
    transport_backend: str = "memory"  # memory | quic | tcp
    required_cluster: bool = False
    wal_replay_cursor_path: str | None = None


class ClusterRuntime:
    """Owns coordinator + transport + registry + background managers."""

    def __init__(self, config: RuntimeConfig, coordinator: TransactionCoordinator | None = None) -> None:
        self.config = config
        self.coordinator = coordinator or TransactionCoordinator()
        if config.wal_replay_cursor_path:
            self.coordinator.set_wal_replay_cursor_path(config.wal_replay_cursor_path)
        elif (
            self.coordinator._wal_replay_enabled
            and self.coordinator._wal_replay_cursor_path is None
        ):
            # Node-scoped default avoids cursor collisions across instances on one host.
            self.coordinator.set_wal_replay_cursor_path(
                f".hiveframe/wal_replay_cursor_{config.node_id}.json"
            )

        # Registry selection
        if config.registry_backend == "sqlite":
            self.registry = SQLiteRegistry(config.db_path)
            self._NodeInfo = SQLiteNodeInfo
        elif config.registry_backend == "nats":
            from .registry import ClusterRegistry
            self.registry = ClusterRegistry(config.nats_url)
            self._NodeInfo = MemNodeInfo
        else:
            from .registry import ClusterRegistry
            self.registry = ClusterRegistry("")
            self._NodeInfo = MemNodeInfo

        # Transport selection
        if config.transport_backend == "tcp":
            self.transport = TCPTransport(host=config.host, port=config.port, node_id=config.node_id)
        else:
            from .quic_transport import QuicTransport

            # Both "memory" and "quic" share the same async transport contract.
            # QuicTransport already provides an in-process fallback when a full QUIC
            # runtime is not wired.
            self.transport = QuicTransport(node_id=config.node_id)

        self.replication = ReplicationManager(
            node_id=config.node_id,
            node_region=config.region,
            role=config.role,
            transport=self.transport,
            registry=self.registry,
            wal=self.coordinator.wal,
        )
        self.replication.set_seed_chunk_handler(self._inject_remote_chunk)
        # Wire replication back into coordinator so commits trigger delta broadcast.
        self.coordinator.replication_manager = self.replication
        self.heartbeat = HeartbeatManager(
            node_id=config.node_id,
            node_region=config.region,
            registry=self.registry,
            transport=self.transport,
        )

        # Log initialization info for observability
        logger.info(
            "ClusterRuntime initialized: id=%s role=%s host=%s port=%s region=%s enable_cluster=%s registry=%s transport=%s nats_url=%s db_path=%s",
            config.node_id,
            config.role,
            config.host,
            config.port,
            config.region,
            config.enable_cluster,
            config.registry_backend,
            config.transport_backend,
            config.nats_url,
            config.db_path,
        )

    async def start(self) -> None:
        # Wire snapshot provider (frame_id-aware) so this node can respond to READ_SNAPSHOT_REQUEST.
        self.replication.set_snapshot_provider(self._local_snapshot_as_dict)

        logger.info("Starting ClusterRuntime start sequence (enable_cluster=%s)", self.config.enable_cluster)

        if self.config.enable_cluster:
            logger.debug("Connecting to registry (nats_url=%s)", self.config.nats_url)
            await self.registry.connect()
            logger.info("Registry connected")

            logger.debug("Starting transport listener on %s:%s", self.config.host, self.config.port)
            await self.transport.listen(self.config.host, self.config.port)
            logger.info("Transport listening on %s:%s", self.config.host, self.config.port)

            logger.debug("Starting replication manager")
            await self.replication.start()
            logger.info("Replication manager started")

            logger.debug("Starting heartbeat manager")
            await self.heartbeat.start()
            logger.info("Heartbeat manager started")

            # Watch for node joins/failures to broadcast rebalance notifications.
            await self.registry.watch(self._on_registry_event)
            logger.debug("Registered registry watcher for node join/failure events")
        else:
            logger.debug("Cluster disabled; connecting registry in standalone mode")
            await self.registry.connect()
            logger.info("Registry connected (standalone)")

        # Register this node in the registry and log the registration
        node_info = self._NodeInfo(
            node_id=self.config.node_id,
            host=self.config.host,
            port=self.config.port,
            role=self.config.role,
            region=self.config.region,
            partition_start=self.config.partition_start,
            partition_end=self.config.partition_end,
            last_seen=time.time(),
            lsn=0,
            status="healthy",
        )

        await self.registry.register(node_info)
        logger.info("Node registered: %s (role=%s host=%s port=%s)", node_info.node_id, node_info.role, node_info.host, node_info.port)
        await self._refresh_transport_peers()
        await self.coordinator.start_wal_replay()

    async def _refresh_transport_peers(self) -> None:
        """Seed transport peer map from registry entries when transport supports it."""
        if not hasattr(self.transport, "register_peer"):
            return
        write_nodes = await self.registry.get_write_nodes()
        read_nodes = await self.registry.get_read_nodes()
        for node in [*write_nodes, *read_nodes]:
            if node.node_id == self.config.node_id:
                continue
            self.transport.register_peer(node.node_id, node.host, node.port)

    async def _on_registry_event(self, node, event: str) -> None:
        """Broadcast REBALANCE message to all peers when partition map changes."""
        logger.info("Registry event: %s for node %s role=%s", event, getattr(node, 'node_id', None), getattr(node, 'role', None))
        if event not in ("joined", "failed"):
            return
        from .message import Message, MessageType
        writers = await self.registry.get_write_nodes()
        await self._refresh_transport_peers()
        if not writers:
            logger.debug("No writer nodes found during registry event; skipping rebalance broadcast")
            return
        partition_map = [
            {
                "node_id": n.node_id,
                "partition_start": n.partition_start,
                "partition_end": n.partition_end,
            }
            for n in writers
        ]
        msg = Message.build(
            message_type=MessageType.REBALANCE,
            sender_id=self.config.node_id,
            sender_region=self.config.region,
            payload={
                "partition_map": partition_map,
            },
        )
        logger.info("Broadcasting REBALANCE to peers (excluding %s). Partition map entries=%d", self.config.node_id, len(partition_map))
        await self.transport.broadcast(msg, exclude=[self.config.node_id])

    def route_write(self, row_index: int):
        """Return the NodeInfo that owns the given row index based on current partition map.

        Returns None when cluster mode is off or no routing info is available
        (caller should fall back to local coordinator).
        """
        if not self.config.enable_cluster:
            return None
        return self.registry.get_owner_for_row(row_index) if hasattr(self.registry, 'get_owner_for_row') else None

    def _local_snapshot_as_dict(self, frame_id: str | None = None) -> dict[str, Any]:
        """Serialize write-node snapshot for a specific frame_id to a transport-safe dict."""
        frame = self._build_frame_snapshot(frame_id) if frame_id else self.coordinator.write_node.snapshot()
        return frame.to_dict(orient="list")

    def _inject_remote_chunk(
        self,
        frame_id: str,
        data: dict[str, list[Any]],
        row_offset: int,
        sender_id: str,
        transactional: bool = True,
    ) -> None:
        """Inject a remote SEED_CHUNK payload into the local write node."""
        if not data:
            return

        namespaced = {
            f"{frame_id}::{col}": values
            for col, values in data.items()
        }

        wn = self.coordinator.write_node
        with wn._lock:
            chunk_frame = pd.DataFrame(namespaced)
            chunk_frame.index = range(row_offset, row_offset + len(chunk_frame))
            if wn._df.empty:
                wn._df = chunk_frame
            else:
                wn._df = pd.concat([wn._df, chunk_frame])
            wn._version += 1

        if transactional:
            from .transaction import Transaction, Operation, TxState

            summary_tx = Transaction(operations=[
                Operation(
                    cell_id=f"{frame_id}::__remote_chunk__",
                    old_value=None,
                    new_value={
                        "row_offset": row_offset,
                        "rows": len(chunk_frame),
                        "sender": sender_id,
                    },
                    author_type="human",
                    author_id=f"remote:{sender_id}",
                )
            ])
            summary_tx.transition(TxState.VALIDATING)
            summary_tx.transition(TxState.LOCKED)
            summary_tx.transition(TxState.APPLYING)
            summary_tx.transition(TxState.COMMITTED)
            self.coordinator.wal.append(summary_tx)
        logger.info(
            "_inject_remote_chunk: frame=%s offset=%d rows=%d sender=%s transactional=%s",
            frame_id,
            row_offset,
            len(chunk_frame),
            sender_id,
            transactional,
        )

    async def read_global_snapshot_for(self, frame_id: str) -> pd.DataFrame:
        """Fan-out snapshot request for a specific frame_id to all healthy writer nodes and merge."""
        from .message import Message, MessageType

        local_frame = self._build_frame_snapshot(frame_id)
        write_nodes = await self.registry.get_write_nodes()
        await self._refresh_transport_peers()
        remote_nodes = [n for n in write_nodes if n.node_id != self.config.node_id]

        if not remote_nodes or not self.config.enable_cluster:
            logger.debug("No remote writer nodes available or cluster disabled; returning local frame")
            return local_frame

        async def fetch_one(node_id: str) -> pd.DataFrame:
            request_id = str(uuid.uuid4())
            msg = Message.build(
                message_type=MessageType.READ_SNAPSHOT_REQUEST,
                sender_id=self.config.node_id,
                sender_region=self.config.region,
                payload={"request_id": request_id, "frame_id": frame_id},
            )
            logger.debug("Requesting snapshot from node %s request_id=%s frame_id=%s", node_id, request_id, frame_id)
            response = await self.transport.request(node_id, msg, timeout=5.0)
            if response is None:
                logger.warning("Snapshot request to node %s timed out (request_id=%s)", node_id, request_id)
                return pd.DataFrame()
            logger.debug("Received snapshot response from node %s for request_id=%s", node_id, request_id)
            snapshot_dict = dict(response.payload.get("snapshot", {}))
            return self._frame_from_dict(snapshot_dict)

        tasks: list[tuple[str, asyncio.Task[pd.DataFrame]]] = [
            (node.node_id, asyncio.create_task(fetch_one(node.node_id))) for node in remote_nodes
        ]
        await asyncio.wait([task for _, task in tasks])
        remote_frames: list[pd.DataFrame] = []
        for node_id, task in tasks:
            try:
                frame = task.result()
                remote_frames.append(frame)
            except Exception as exc:
                logger.warning("Failed to fetch frame snapshot from remote writer %s: %s", node_id, exc)
        frames = [local_frame]
        for result in remote_frames:
            if not result.empty:
                frames.append(result)
        return self._merge_snapshots(frames)

    def _build_frame_snapshot(self, frame_id: str) -> pd.DataFrame:
        """Build a clean DataFrame from write_node columns belonging to frame_id.

        WriteNode stores columns as '{frame_id}::{col}' with pandas row index.
        Filter by prefix and strip it to get clean column names.
        """
        prefix = f"{frame_id}::"
        wn = self.coordinator.write_node
        with wn._lock:
            df_internal = wn._df
            matching = [c for c in df_internal.columns if c.startswith(prefix)]
            if not matching:
                return pd.DataFrame()
            subset = df_internal[matching].copy()
        subset.columns = [c[len(prefix):] for c in subset.columns]
        return subset.reset_index(drop=True)

    @staticmethod
    def _frame_from_dict(data: dict[str, Any]) -> pd.DataFrame:
        """Deserialize a snapshot dict received over transport back to DataFrame."""
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    @staticmethod
    def _merge_snapshots(frames: list[pd.DataFrame]) -> pd.DataFrame:
        """Merge node snapshots into one frame with schema union and stable row order."""
        if not frames:
            return pd.DataFrame()
        non_empty = [f.reset_index(drop=True) for f in frames if isinstance(f, pd.DataFrame) and not f.empty]
        if not non_empty:
            return pd.DataFrame()
        merged = pd.concat(non_empty, axis=0, ignore_index=True, sort=False)
        merged = merged.reindex(sorted(merged.columns), axis=1)
        return merged

    async def read_global_snapshot(self) -> pd.DataFrame:
        """Fan-out snapshot request to all healthy writer nodes and merge results.

        Falls back to local write-node snapshot when:
        - cluster mode is disabled, or
        - no remote writer nodes are registered, or
        - all remote requests time out.
        """
        from .message import Message, MessageType

        local_frame = self.coordinator.write_node.snapshot()

        # In standalone mode or with only this node in registry, skip fan-out.
        write_nodes = await self.registry.get_write_nodes()
        await self._refresh_transport_peers()
        remote_nodes = [n for n in write_nodes if n.node_id != self.config.node_id]

        if not remote_nodes or not self.config.enable_cluster:
            logger.debug("No remote writer nodes available or cluster disabled; returning local frame")
            return local_frame

        # Fan-out: request snapshot from each remote writer node concurrently.
        async def fetch_one(node_id: str) -> pd.DataFrame:
            request_id = str(uuid.uuid4())
            msg = Message.build(
                message_type=MessageType.READ_SNAPSHOT_REQUEST,
                sender_id=self.config.node_id,
                sender_region=self.config.region,
                payload={"request_id": request_id},
            )
            logger.debug("Requesting snapshot from node %s request_id=%s", node_id, request_id)
            response = await self.transport.request(node_id, msg, timeout=5.0)
            if response is None:
                logger.warning("Snapshot request to node %s timed out (request_id=%s)", node_id, request_id)
                return pd.DataFrame()
            logger.debug("Received snapshot response from node %s for request_id=%s", node_id, request_id)
            snapshot_dict = dict(response.payload.get("snapshot", {}))
            return self._frame_from_dict(snapshot_dict)

        tasks: list[tuple[str, asyncio.Task[pd.DataFrame]]] = [
            (node.node_id, asyncio.create_task(fetch_one(node.node_id))) for node in remote_nodes
        ]
        await asyncio.wait([task for _, task in tasks])
        remote_frames: list[pd.DataFrame] = []
        for node_id, task in tasks:
            try:
                frame = task.result()
                remote_frames.append(frame)
            except Exception as exc:
                logger.warning("Failed to fetch global snapshot from remote writer %s: %s", node_id, exc)

        frames = [local_frame]
        for result in remote_frames:
            if not result.empty:
                frames.append(result)

        return self._merge_snapshots(frames)
