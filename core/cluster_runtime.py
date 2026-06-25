# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Runtime wiring for Phase 2 control/data plane over registry and transport."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
from pathlib import Path
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
from .minio_storage import MinioDataDir, create_data_dir
from .op_log import OperationLog


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
    seed_hosts: list[str] = field(default_factory=list)
    required_cluster: bool = False
    required_cluster_min_nodes: int = 1
    required_cluster_min_write_nodes: int = 0
    required_cluster_min_read_nodes: int = 0
    wal_replay_cursor_path: str | None = None
    # Leader config (Phase 2)
    is_leader: bool = False
    leader_node_id: str | None = None
    # Heartbeat config
    heartbeat_interval_ms: int = 1000
    heartbeat_timeout_ms: int = 5000
    suspect_multiplier: int = 2
    suspect_rebalance_debounce_ms: int = 0
    # Transport retry config
    send_max_retries: int = 3
    send_base_backoff_ms: int = 200
    # Op-log sync
    sync_interval_ms: int = 1000
    push_threshold_ops: int = 100
    full_resync_threshold_ops: int = 20000
    # MinIO data directory
    data_dir_backend: str = "local"  # local | minio
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "hiveframe"
    minio_secure: bool = False
    minio_region: str = "us-east-1"
    minio_prefix: str = "hiveframe/"
    # Observability
    metrics_enabled: bool = True
    audit_log_path: str | None = None
    audit_log_max_bytes: int = 10_485_760
    audit_log_backup_count: int = 3


class ClusterRuntime:
    """Owns coordinator + transport + registry + background managers."""

    def __init__(self, config: RuntimeConfig, coordinator: TransactionCoordinator | None = None) -> None:
        self.config = config
        self._validate_leader_config(config)

        self.coordinator = coordinator or TransactionCoordinator()
        # Wire leader config into coordinator for write guard
        self.coordinator._node_id = config.node_id
        self.coordinator._is_leader = config.is_leader
        self.coordinator._leader_node_id = config.leader_node_id

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
            self.registry.set_write_guard_context(
                local_node_id=config.node_id,
                is_leader=config.is_leader,
            )
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
        self.replication.set_cluster_control_handlers(
            join_handler=self._handle_join_message,
            membership_update_handler=self._handle_membership_update,
            route_update_handler=self._handle_route_update,
        )
        self.replication.set_oplog_handlers(
            push_handler=self._handle_oplog_push,
            pull_handler=self._handle_oplog_pull,
            full_resync_handler=self._handle_oplog_full_resync,
        )
        # Wire replication back into coordinator so commits trigger delta broadcast.
        self.coordinator.replication_manager = self.replication
        self.heartbeat = HeartbeatManager(
            node_id=config.node_id,
            node_region=config.region,
            registry=self.registry,
            transport=self.transport,
            interval_s=max(0.05, float(config.heartbeat_interval_ms) / 1000.0),
            timeout_s=max(0.1, float(config.heartbeat_timeout_ms) / 1000.0),
            suspect_multiplier=max(1.0, float(config.suspect_multiplier)),
            suspect_rebalance_debounce_s=max(0.0, float(config.suspect_rebalance_debounce_ms) / 1000.0),
        )
        self.heartbeat.on_status_change(self._on_heartbeat_status_change)

        self._leader_broadcast_task: asyncio.Task[None] | None = None
        self._oplog_sync_task: asyncio.Task[None] | None = None
        self._wal_reachable = True
        self._replication_lag_ops = 0
        self._outbox_depth = 0
        self._conflict_count = 0
        self._full_resync_count = 0
        self._metrics_enabled = bool(config.metrics_enabled)
        self._audit_log_path = Path(config.audit_log_path) if config.audit_log_path else None
        self._audit_log_max_bytes = max(0, int(config.audit_log_max_bytes))
        self._audit_log_backup_count = max(0, int(config.audit_log_backup_count))
        self._minio: MinioDataDir | None = None
        if config.data_dir_backend == "minio":
            self._minio = MinioDataDir(
                endpoint=config.minio_endpoint,
                access_key=config.minio_access_key,
                secret_key=config.minio_secret_key,
                bucket=config.minio_bucket,
                secure=config.minio_secure,
                region=config.minio_region,
                prefix=config.minio_prefix,
            )
            logger.info("MinIO data directory enabled: bucket=%s prefix=%s", config.minio_bucket, config.minio_prefix)
            # Restore registry DB from MinIO if local doesn't exist yet
            db_path = Path(config.db_path)
            if not db_path.exists() and config.registry_backend == "sqlite":
                try:
                    db_object = f"registry/{config.node_id}.db"
                    self._minio.restore_db(db_path, db_object)
                except Exception as exc:
                    logger.info("No MinIO registry backup found for node %s: %s", config.node_id, exc)
        self._resync_snapshot_cache: dict[str, Any] | None = None
        self.op_log: OperationLog | None = None
        if config.enable_cluster:
            self.op_log = OperationLog(
                db_path=config.db_path,
                node_id=config.node_id,
                leader_epoch=1 if config.is_leader else 0,
            )

        # Log initialization info for observability
        logger.info(
            "ClusterRuntime initialized: id=%s role=%s host=%s port=%s region=%s enable_cluster=%s "
            "registry=%s transport=%s is_leader=%s leader_node_id=%s",
            config.node_id,
            config.role,
            config.host,
            config.port,
            config.region,
            config.enable_cluster,
            config.registry_backend,
            config.transport_backend,
            config.is_leader,
            config.leader_node_id,
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

            await self._bootstrap_seeds()

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
            leader_reachable=self.config.is_leader or self.config.leader_node_id is None,
            wal_reachable=True,
        )

        await self.registry.register(node_info)
        logger.info("Node registered: %s (role=%s host=%s port=%s)", node_info.node_id, node_info.role, node_info.host, node_info.port)
        self._audit_event("JOIN", {
            "node_id": node_info.node_id,
            "role": node_info.role,
            "status": node_info.status,
            "local": True,
        })
        await self._refresh_transport_peers()

        # Initialize capability flags after peer map is available.
        if self.config.enable_cluster and not self.config.is_leader and self.config.leader_node_id:
            leader_reachable = await self._check_leader_reachable_now()
            await self._set_local_capability_flags(leader_reachable=leader_reachable)

        if self.config.enable_cluster and self.config.required_cluster:
            await self._wait_required_cluster_ready()

        await self.coordinator.start_wal_replay()

        # Leader: start periodic MEMBERSHIP_UPDATE + ROUTE_UPDATE broadcast
        if self.config.enable_cluster and self.config.is_leader:
            self._leader_broadcast_task = asyncio.create_task(self._leader_broadcast_loop())
            logger.info("Leader broadcast loop started for node=%s", self.config.node_id)

        if self.config.enable_cluster and self.op_log is not None:
            self._oplog_sync_task = asyncio.create_task(self._oplog_sync_loop())
            logger.info("Op-log sync loop started for node=%s", self.config.node_id)

    async def _bootstrap_seeds(self) -> None:
        """Attempt initial transport connects to configured seed hosts."""
        seeds = self._parse_seed_hosts(self.config.seed_hosts)
        if not seeds:
            return

        for host, port in seeds:
            delay = 0.2
            connected = False
            for attempt in range(3):
                try:
                    await self.transport.connect(host, port)
                    connected = True
                    logger.info(
                        "Seed connect success: node=%s target=%s:%d attempt=%d",
                        self.config.node_id,
                        host,
                        port,
                        attempt + 1,
                    )
                    break
                except Exception as exc:
                    logger.warning(
                        "Seed connect failed: node=%s target=%s:%d attempt=%d error=%s",
                        self.config.node_id,
                        host,
                        port,
                        attempt + 1,
                        exc,
                    )
                    if attempt < 2:
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 2.0)

            if not connected:
                logger.warning(
                    "Seed unreachable after retries: node=%s target=%s:%d",
                    self.config.node_id,
                    host,
                    port,
                )

    def _parse_seed_hosts(self, seed_hosts: list[str]) -> list[tuple[str, int]]:
        targets: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()

        for raw in seed_hosts:
            value = str(raw).strip()
            if not value:
                continue
            if ":" not in value:
                logger.warning("Ignoring invalid seed host '%s' (expected host:port)", value)
                continue
            host, port_text = value.rsplit(":", 1)
            host = host.strip()
            if not host or not port_text.isdigit():
                logger.warning("Ignoring invalid seed host '%s' (expected host:port)", value)
                continue
            target = (host, int(port_text))
            if target == (self.config.host, self.config.port):
                continue
            if target in seen:
                continue
            seen.add(target)
            targets.append(target)

        return targets

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
        if event not in ("joined", "failed", "suspect"):
            return
        logger.debug("Registry event: %s for node %s role=%s", event, getattr(node, 'node_id', None), getattr(node, 'role', None))
        if event == "joined":
            self._audit_event("JOIN", {
                "node_id": getattr(node, "node_id", None),
                "role": getattr(node, "role", None),
                "status": getattr(node, "status", None),
            })
        elif event == "failed":
            self._audit_event("DEAD", {
                "node_id": getattr(node, "node_id", None),
                "role": getattr(node, "role", None),
                "status": getattr(node, "status", None),
            })
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
        read_nodes = await self.registry.get_read_nodes()
        all_nodes = list(writers) + list(read_nodes)
        members_payload = [
            {
                "node_id": n.node_id,
                "host": n.host,
                "port": n.port,
                "role": n.role,
                "status": n.status,
            }
            for n in all_nodes
        ]
        self._record_metadata_op("cluster_membership", "members", members_payload)
        self._record_metadata_op("cluster_routing", "partition_map", partition_map)
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

    async def _handle_join_message(self, message) -> None:
        """Leader-side JOIN handler: return membership with leader last_op_id for recovery gating."""
        if message.sender_id == self.config.node_id:
            return
        await self._refresh_transport_peers()
        if not self.config.is_leader:
            return

        from .message import Message, MessageType

        leader_last_op_id = self.op_log.get_last_acked_op_id() if self.op_log is not None else None
        writers = await self.registry.get_write_nodes()
        readers = await self.registry.get_read_nodes()
        members = [
            {
                "node_id": n.node_id,
                "host": n.host,
                "port": n.port,
                "role": n.role,
                "status": n.status,
            }
            for n in [*writers, *readers]
        ]
        membership = Message.build(
            message_type=MessageType.MEMBERSHIP_UPDATE,
            sender_id=self.config.node_id,
            sender_region=self.config.region,
            payload={
                "leader_node_id": self.config.node_id,
                "leader_last_op_id": leader_last_op_id,
                "members": members,
            },
        )
        await self.transport.send(message.sender_id, membership)

    async def _handle_membership_update(self, message) -> None:
        """Follower-side membership update handler with reconnect recovery sync gate."""
        leader_node_id = str(message.payload.get("leader_node_id", "") or "")
        if not leader_node_id:
            return

        self.coordinator._leader_node_id = leader_node_id
        if leader_node_id != self.config.node_id:
            fields = {f: getattr(self.config, f) for f in RuntimeConfig.__dataclass_fields__}
            fields["leader_node_id"] = leader_node_id
            fields["is_leader"] = False
            self.config = RuntimeConfig(**fields)
            # B4 fix: re-sync coordinator state after config rebuild so the write
            # guard sees the updated leader identity.
            self.coordinator._is_leader = False
            self.coordinator._leader_node_id = leader_node_id

        if self.config.is_leader or self.op_log is None:
            return

        leader_last_op_id = message.payload.get("leader_last_op_id")
        local_last_op_id = self.op_log.get_last_acked_op_id()
        gap = self.op_log.estimate_gap(local_last_op_id, str(leader_last_op_id) if leader_last_op_id else None)
        threshold = max(1, int(self.config.full_resync_threshold_ops))

        if leader_last_op_id and (local_last_op_id is None or gap > threshold):
            await self.op_log.full_resync_from_leader(leader_node_id, self.transport)
            self._full_resync_count += 1
            self._update_metrics_snapshot()
            await self._set_local_capability_flags(leader_reachable=True, wal_reachable=self._wal_reachable)

    async def _handle_route_update(self, message) -> None:
        """Apply leader partition map updates to local registry for deterministic routing."""
        partition_map = message.payload.get("partition_map", [])
        if not isinstance(partition_map, list) or not partition_map:
            return
        apply_partition_map = getattr(self.registry, "apply_partition_map", None)
        if not callable(apply_partition_map):
            return
        try:
            applied = await apply_partition_map(partition_map)
        except ValueError as exc:
            logger.warning("Rejected invalid ROUTE_UPDATE from leader=%s error=%s", message.sender_id, exc)
            return
        if applied:
            logger.info(
                "Applied ROUTE_UPDATE from leader=%s entries=%d",
                message.sender_id,
                applied,
            )

    # -------------------------------------------------------------------------
    # Leader config + capability helpers (Phase 2)
    # -------------------------------------------------------------------------

    @staticmethod
    def _validate_leader_config(config: RuntimeConfig) -> None:
        """Validate leader/follower config consistency at startup."""
        if not config.enable_cluster:
            return
        if config.is_leader:
            if config.leader_node_id is not None and config.leader_node_id != config.node_id:
                raise ValueError(
                    f"Config error: is_leader=True but leader_node_id={config.leader_node_id!r} "
                    f"does not match node_id={config.node_id!r}. "
                    "Set leader_node_id to this node's id or leave it None."
                )
        else:
            if config.leader_node_id is None:
                logger.warning(
                    "Node %s: is_leader=False but leader_node_id is not set. "
                    "Writes will NOT be blocked; set leader_node_id to enforce single-writer guard.",
                    config.node_id,
                )

    def get_local_capability(self) -> str:
        """Return this node's capability synchronously (best-effort, no network I/O).

        Used for write-path guards in sync contexts.
        Returns 'rw' | 'ro' | 'drain'.
        """
        if not self.config.enable_cluster:
            return "rw"

        # Try in-memory registry first (ClusterRegistry)
        if hasattr(self.registry, "_nodes"):
            node = self.registry._nodes.get(self.config.node_id)
            if node is not None:
                return node.capability

        # Try SQLite registry (sync)
        if hasattr(self.registry, "_fetchone_node"):
            try:
                node = self.registry._fetchone_node(
                    "SELECT * FROM nodes WHERE node_id=?", (self.config.node_id,)
                )
                if node is not None:
                    return node.capability
            except Exception:
                pass

        # Compute from config as fallback
        from .node_status import compute_capability
        leader_reachable = self.config.is_leader or self.config.leader_node_id is None
        return compute_capability(self.config.role, "healthy", leader_reachable, True)

    async def _set_local_capability_flags(
        self,
        *,
        leader_reachable: bool | None = None,
        wal_reachable: bool | None = None,
    ) -> None:
        """Persist local node capability flags into registry for capability computation."""
        if not hasattr(self.registry, "update_capability_flags"):
            return
        local = await self.registry.get_node(self.config.node_id)
        if local is None:
            return
        lr = bool(local.leader_reachable if leader_reachable is None else leader_reachable)
        wr = bool(local.wal_reachable if wal_reachable is None else wal_reachable)
        await self.registry.update_capability_flags(self.config.node_id, lr, wr)

    async def _check_leader_reachable_now(self) -> bool:
        """Best-effort leader reachability check used by required_cluster and recovery gate."""
        if not self.config.enable_cluster:
            return True
        if self.config.is_leader:
            return True
        leader_id = self.config.leader_node_id
        if not leader_id:
            return False
        leader = await self.registry.get_node(leader_id)
        if leader is None or leader.status != "healthy":
            return False

        # Condition 1: transport connect should work.
        if self.config.transport_backend == "tcp":
            try:
                await self.transport.connect(leader.host, int(leader.port))
            except Exception:
                return False

        # Condition 2: op gap should not exceed configured threshold.
        if self.op_log is not None:
            since = self.op_log.get_last_acked_op_id()
            pulled = await self.op_log.pull_from_leader(leader_id, self.transport, since)
            if pulled > max(1, int(self.config.full_resync_threshold_ops)):
                return False

        # Condition 3: WAL backend should be reachable (runtime-maintained flag).
        return bool(self._wal_reachable)

    async def _wait_required_cluster_ready(self) -> None:
        """Block startup until leader is reachable when required_cluster=True."""
        timeout_s = max(0.2, float(self.config.heartbeat_timeout_ms) / 1000.0)
        deadline = time.time() + timeout_s
        min_nodes = max(1, int(self.config.required_cluster_min_nodes))
        min_writers = max(0, int(self.config.required_cluster_min_write_nodes))
        min_readers = max(0, int(self.config.required_cluster_min_read_nodes))
        while time.time() < deadline:
            ready = await self._check_leader_reachable_now()
            healthy_nodes, healthy_writers, healthy_readers = await self._count_healthy_nodes_by_role()
            quorum_ok = (
                healthy_nodes >= min_nodes
                and healthy_writers >= min_writers
                and healthy_readers >= min_readers
            )
            await self._set_local_capability_flags(leader_reachable=ready, wal_reachable=self._wal_reachable)
            if ready and quorum_ok:
                return
            await asyncio.sleep(0.05)
        raise RuntimeError(
            f"required_cluster startup gate failed: leader node {self.config.leader_node_id!r} "
            f"is not reachable with healthy quorum total>={min_nodes}, write>={min_writers}, "
            f"read>={min_readers} within {timeout_s:.2f}s"
        )

    async def _count_healthy_nodes_by_role(self) -> tuple[int, int, int]:
        """Count healthy nodes currently visible in registry by role."""
        write_nodes = await self.registry.get_write_nodes()
        read_nodes = await self.registry.get_read_nodes()
        healthy_write_ids = {
            n.node_id for n in write_nodes if getattr(n, "status", "healthy") == "healthy"
        }
        healthy_read_ids = {
            n.node_id for n in read_nodes if getattr(n, "status", "healthy") == "healthy"
        }
        healthy_total = len(healthy_write_ids | healthy_read_ids)
        return healthy_total, len(healthy_write_ids), len(healthy_read_ids)

    async def _on_heartbeat_status_change(self, node_id: str, _old_status: str, new_status: str) -> None:
        """React to leader status transitions to keep follower capability safe."""
        if new_status == "suspect":
            self._audit_event("SUSPECT", {"node_id": node_id})
        elif new_status == "failed":
            self._audit_event("DEAD", {"node_id": node_id})

        if self.config.is_leader:
            return
        if not self.config.leader_node_id or node_id != self.config.leader_node_id:
            return
        if new_status in {"failed", "suspect"}:
            await self._set_local_capability_flags(leader_reachable=False, wal_reachable=self._wal_reachable)
            return
        if new_status == "healthy":
            recovered = await self._check_leader_reachable_now()
            await self._set_local_capability_flags(leader_reachable=recovered, wal_reachable=self._wal_reachable)

    def set_wal_reachable(self, reachable: bool) -> None:
        """Update local WAL reachability flag consumed by capability/recovery gates."""
        self._wal_reachable = bool(reachable)

    def _update_metrics_snapshot(self) -> None:
        if not self._metrics_enabled:
            return
        if self.op_log is None:
            self._outbox_depth = 0
            self._replication_lag_ops = 0
        else:
            pending = self.op_log.get_pending_ops()
            self._outbox_depth = len(pending)
            # follower lag proxy: unsynced pending depth
            self._replication_lag_ops = len(pending)
        self.coordinator.set_runtime_metrics(
            replication_lag_ops=self._replication_lag_ops,
            outbox_depth=self._outbox_depth,
            conflict_count=self._conflict_count,
            full_resync_count=self._full_resync_count,
        )

    def _record_metadata_op(self, entity: str, key: str, value: Any) -> None:
        """Persist leader metadata-plane changes to op-log for follower pull/resync paths."""
        if not self.config.enable_cluster:
            return
        if not self.config.is_leader:
            return
        if self.op_log is None:
            return
        try:
            self.op_log.append_local_acked(entity, key, value)
        except Exception as exc:
            logger.warning("Failed to append metadata op entity=%s key=%s error=%s", entity, key, exc)

    def get_cluster_stats(self) -> dict[str, Any]:
        """Return runtime + coordinator metrics snapshot for observability."""
        self._update_metrics_snapshot()
        return {
            "node_id": self.config.node_id,
            "is_leader": self.config.is_leader,
            "leader_node_id": self.config.leader_node_id,
            "required_cluster": self.config.required_cluster,
            "required_cluster_min_nodes": self.config.required_cluster_min_nodes,
            "required_cluster_min_write_nodes": self.config.required_cluster_min_write_nodes,
            "required_cluster_min_read_nodes": self.config.required_cluster_min_read_nodes,
            "metrics_enabled": self._metrics_enabled,
            "coordinator": self.coordinator.get_stats(),
            "replication_lag_ops": self._replication_lag_ops,
            "outbox_depth": self._outbox_depth,
            "conflict_count": self._conflict_count,
            "full_resync_count": self._full_resync_count,
            "oplog_pending_ops": len(self.op_log.get_pending_ops()) if self.op_log is not None else 0,
            "oplog_last_acked_op_id": self.op_log.get_last_acked_op_id() if self.op_log is not None else None,
        }

    def _audit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Append structured audit log event as JSON line when audit logging is enabled."""
        if self._audit_log_path is None:
            return
        event = {
            "ts": time.time(),
            "event": event_type,
            "node_id": self.config.node_id,
            "payload": payload,
        }
        try:
            self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_audit_log_if_needed()
            with self._audit_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
        except Exception as exc:
            logger.warning("Failed to write audit event %s: %s", event_type, exc)

    def _rotate_audit_log_if_needed(self) -> None:
        if self._audit_log_path is None:
            return
        if self._audit_log_max_bytes <= 0:
            return
        if not self._audit_log_path.exists():
            return
        try:
            size = self._audit_log_path.stat().st_size
        except Exception:
            return
        if size < self._audit_log_max_bytes:
            return

        if self._audit_log_backup_count <= 0:
            self._audit_log_path.unlink(missing_ok=True)
            return

        # Shift existing backups: .N -> .N+1, then current -> .1
        for idx in range(self._audit_log_backup_count, 0, -1):
            src = Path(f"{self._audit_log_path}.{idx}")
            if idx == self._audit_log_backup_count:
                if src.exists():
                    src.unlink()
                continue
            dst = Path(f"{self._audit_log_path}.{idx + 1}")
            if src.exists():
                src.replace(dst)

        first = Path(f"{self._audit_log_path}.1")
        self._audit_log_path.replace(first)

    async def _leader_broadcast_loop(self, interval_s: float = 5.0) -> None:
        """Periodically broadcast MEMBERSHIP_UPDATE + ROUTE_UPDATE to all peers (leader only)."""
        from .message import Message, MessageType
        while True:
            try:
                await asyncio.sleep(interval_s)
                all_nodes = await self.registry.get_write_nodes()
                all_nodes += await self.registry.get_read_nodes()
                membership_payload = [
                    {
                        "node_id": n.node_id,
                        "host": n.host,
                        "port": n.port,
                        "role": n.role,
                        "status": n.status,
                        "capability": n.capability if hasattr(n, "capability") else "rw",
                    }
                    for n in all_nodes
                ]
                writers = await self.registry.get_write_nodes()
                route_payload = [
                    {
                        "node_id": n.node_id,
                        "partition_start": n.partition_start,
                        "partition_end": n.partition_end,
                    }
                    for n in writers
                ]
                membership_msg = Message.build(
                    message_type=MessageType.MEMBERSHIP_UPDATE,
                    sender_id=self.config.node_id,
                    sender_region=self.config.region,
                    payload={
                        "leader_node_id": self.config.node_id,
                        "leader_last_op_id": self.op_log.get_last_acked_op_id() if self.op_log is not None else None,
                        "members": membership_payload,
                    },
                )
                route_msg = Message.build(
                    message_type=MessageType.ROUTE_UPDATE,
                    sender_id=self.config.node_id,
                    sender_region=self.config.region,
                    payload={"partition_map": route_payload},
                )
                await self.transport.broadcast(membership_msg, exclude=[self.config.node_id])
                await self.transport.broadcast(route_msg, exclude=[self.config.node_id])
                logger.debug(
                    "Leader broadcast: MEMBERSHIP_UPDATE (members=%d) + ROUTE_UPDATE (writers=%d)",
                    len(membership_payload),
                    len(route_payload),
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Leader broadcast loop error: %s", exc)

    async def _oplog_sync_loop(self) -> None:
        """Follower sync loop: push pending ops, pull acks, and trigger full-resync on large gap."""
        if self.op_log is None:
            return
        interval_s = max(0.05, float(self.config.sync_interval_ms) / 1000.0)
        while True:
            try:
                await asyncio.sleep(interval_s)
                if not self.config.enable_cluster:
                    continue
                leader_id = self.config.leader_node_id
                if not leader_id or leader_id == self.config.node_id:
                    continue
                assert leader_id is not None

                pending_ops = self.op_log.get_pending_ops()
                pending_count = len(pending_ops)
                self._outbox_depth = pending_count
                self._replication_lag_ops = pending_count
                if pending_count >= max(1, int(self.config.push_threshold_ops)):
                    pushed = await self.op_log.push_to_leader(leader_id, self.transport)
                    logger.debug("Op-log push: node=%s leader=%s pushed=%d", self.config.node_id, leader_id, pushed)

                if pending_count >= max(1, int(self.config.full_resync_threshold_ops)):
                    await self.op_log.full_resync_from_leader(leader_id, self.transport)
                    self._full_resync_count += 1
                    logger.warning(
                        "Op-log full-resync triggered: node=%s leader=%s pending=%d threshold=%d",
                        self.config.node_id,
                        leader_id,
                        pending_count,
                        self.config.full_resync_threshold_ops,
                    )
                    continue

                since_op_id = self.op_log.get_last_acked_op_id()
                pulled = await self.op_log.pull_from_leader(leader_id, self.transport, since_op_id)
                if pulled:
                    logger.debug("Op-log pull: node=%s leader=%s pulled=%d", self.config.node_id, leader_id, pulled)
                self._update_metrics_snapshot()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Op-log sync loop error (node=%s): %s", self.config.node_id, exc)

    async def _handle_oplog_push(self, message) -> dict[str, Any]:
        """Leader-side handler: accept follower pending operations."""
        if self.op_log is None:
            return {"acked_ops": [], "rejected_ops": []}
        if not self.config.is_leader:
            return {"acked_ops": [], "rejected_ops": list(message.payload.get("ops", []))}
        ops = list(message.payload.get("ops", []))
        result = self.op_log.accept_from_follower(ops)
        self._conflict_count += len(result.get("rejected_ops", []))
        self._update_metrics_snapshot()
        return result

    async def _handle_oplog_pull(self, message) -> dict[str, Any]:
        """Leader-side handler: provide acked operations since follower cursor."""
        if self.op_log is None or not self.config.is_leader:
            return {"ops": []}
        since_op_id = message.payload.get("since_op_id")
        ops = self.op_log.export_since(str(since_op_id) if since_op_id is not None else None)
        return {"ops": ops}

    async def _handle_oplog_full_resync(self, message) -> dict[str, Any]:
        """Leader-side handler: provide full-resync via manifest/chunk/ops modes."""
        if self.op_log is None or not self.config.is_leader:
            return {"ops": []}
        mode = str(message.payload.get("mode", "manifest"))

        if mode == "ops":
            return {
                "mode": "ops",
                "ops": self.op_log.export_all_acked(),
                "leader_last_op_id": self.op_log.get_last_acked_op_id(),
            }

        # Build snapshot chunks lazily and cache for follow-up chunk requests.
        cache = self._resync_snapshot_cache
        now = time.time()
        if cache is None or (now - float(cache.get("ts", 0.0))) > 5.0:
            snapshot_b64 = self.op_log.export_sqlite_snapshot_b64()
            chunks, digest = self.op_log.snapshot_chunks(snapshot_b64, 256 * 1024)
            cache = {
                "ts": now,
                "chunks": chunks,
                "sha256": digest,
                "leader_last_op_id": self.op_log.get_last_acked_op_id(),
            }
            self._resync_snapshot_cache = cache

        chunks = list(cache.get("chunks", []))
        if mode == "chunk":
            idx = int(message.payload.get("chunk_index", 0))
            if idx < 0 or idx >= len(chunks):
                return {
                    "mode": "chunk",
                    "chunk_index": idx,
                    "chunk_b64": "",
                }
            return {
                "mode": "chunk",
                "chunk_index": idx,
                "chunk_b64": chunks[idx],
                "sha256": cache.get("sha256"),
            }

        return {
            "mode": "manifest",
            "chunk_count": len(chunks),
            "sha256": cache.get("sha256"),
            "leader_last_op_id": cache.get("leader_last_op_id"),
        }

    async def promote_to_leader(self) -> None:
        """Promote this node to leader (manual failover).

        Sets is_leader=True, updates coordinator leader config,
        updates registry capability flags, and broadcasts MEMBERSHIP_UPDATE.

        Pre-epoch: authority is config-based + monotonic timestamp.
        Epoch enforcement will be added in Phase 3 via leader_epoch field.
        """
        from .message import Message, MessageType

        # Rebuild RuntimeConfig with updated leader fields
        fields = {f: getattr(self.config, f) for f in RuntimeConfig.__dataclass_fields__}
        fields["is_leader"] = True
        fields["leader_node_id"] = self.config.node_id
        self.config = RuntimeConfig(**fields)

        self.coordinator._is_leader = True
        self.coordinator._leader_node_id = self.config.node_id
        if self.op_log is not None:
            self.op_log.set_leader_epoch(max(1, self.op_log.get_leader_epoch() + 1))
            self._record_metadata_op("cluster_membership", "leader_node_id", self.config.node_id)
        if hasattr(self.registry, "set_write_guard_context"):
            self.registry.set_write_guard_context(local_node_id=self.config.node_id, is_leader=True)
        self._audit_event("LEADER_CHANGE", {
            "new_leader_node_id": self.config.node_id,
            "trigger": "manual_promote",
        })

        # Update capability flags in registry
        if hasattr(self.registry, "update_capability_flags"):
            await self.registry.update_capability_flags(self.config.node_id, True, True)

        logger.info("Node %s promoted to leader", self.config.node_id)

        # Broadcast MEMBERSHIP_UPDATE to notify followers
        all_nodes = await self.registry.get_write_nodes()
        all_nodes += await self.registry.get_read_nodes()
        msg = Message.build(
            message_type=MessageType.MEMBERSHIP_UPDATE,
            sender_id=self.config.node_id,
            sender_region=self.config.region,
            payload={
                "leader_node_id": self.config.node_id,
                "members": [
                    {
                        "node_id": n.node_id,
                        "host": n.host,
                        "port": n.port,
                        "role": n.role,
                        "status": n.status,
                    }
                    for n in all_nodes
                ],
            },
        )
        await self.transport.broadcast(msg, exclude=[self.config.node_id])
        logger.info("Broadcast MEMBERSHIP_UPDATE (leader promotion) from %s", self.config.node_id)

        # Start broadcast loop if not running
        if self._leader_broadcast_task is None or self._leader_broadcast_task.done():
            self._leader_broadcast_task = asyncio.create_task(self._leader_broadcast_loop())

    async def stop(self) -> None:
        """Stop all background tasks and clean up resources."""
        self._audit_event("LEAVE", {"node_id": self.config.node_id})
        if self._oplog_sync_task is not None and not self._oplog_sync_task.done():
            self._oplog_sync_task.cancel()
            try:
                await self._oplog_sync_task
            except asyncio.CancelledError:
                pass
        if self._leader_broadcast_task is not None and not self._leader_broadcast_task.done():
            self._leader_broadcast_task.cancel()
            try:
                await self._leader_broadcast_task
            except asyncio.CancelledError:
                pass
        await self.coordinator.stop_wal_replay()
        if self.op_log is not None:
            self.op_log.close()
        # Sync registry and op-log databases to MinIO on clean shutdown
        if self._minio is not None:
            await self._sync_databases_to_minio()

    async def _sync_databases_to_minio(self) -> None:
        """Upload local registry and op-log databases to MinIO object storage."""
        if self._minio is None:
            return
        try:
            node_id = self.config.node_id
            # Sync registry DB
            registry_object = f"registry/{node_id}.db"
            self._minio.sync_db(self.config.db_path, registry_object)
            logger.info("Synced registry DB to MinIO: %s", registry_object)
            # Sync op-log DB (same db_path for SQLite-backed op-log)
            if self.op_log is not None:
                oplog_object = f"oplog/{node_id}.db"
                self._minio.sync_db(self.config.db_path, oplog_object)
                logger.info("Synced op-log DB to MinIO: %s", oplog_object)
        except Exception as exc:
            logger.warning("Failed to sync databases to MinIO: %s", exc)

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
            raw_snapshot = dict(response.payload.get("snapshot", {}))
            snapshot_dict = {str(k): v for k, v in raw_snapshot.items()}
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
            raw_snapshot = dict(response.payload.get("snapshot", {}))
            snapshot_dict = {str(k): v for k, v in raw_snapshot.items()}
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
