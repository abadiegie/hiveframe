# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0
"""SQLite-based node registry for dev/homelab/single-machine deployments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import sqlite3
from threading import RLock
import time
from typing import Awaitable, Callable

from . import node_status

@dataclass(slots=True)
class NodeInfo:
    node_id: str
    host: str
    port: int
    role: str
    region: str
    partition_start: int
    partition_end: int
    last_seen: float
    lsn: int
    status: str
    leader_reachable: bool = True
    wal_reachable: bool = True
    leader_epoch: int = 0

    @property
    def capability(self) -> str:
        """Compute capability (rw|ro|drain) from current state."""
        return node_status.compute_capability(
            self.role, self.status, self.leader_reachable, self.wal_reachable
        )

WatchCallback = Callable[[NodeInfo, str], Awaitable[None]]
logger = logging.getLogger("core.sqlite_registry")
_PARTITION_TOTAL = 1000

class SQLiteRegistry:
    def __init__(self, db_path: str = ".hiveframe/registry.db"):
        self.db_path = db_path
        db_file = Path(db_path)
        if db_file.parent != Path(""):
            db_file.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._connected = False
        self._lock = RLock()
        self._watchers: list[WatchCallback] = []
        self._guard_local_node_id: str | None = None
        self._guard_is_leader: bool = True

    def set_write_guard_context(self, *, local_node_id: str, is_leader: bool) -> None:
        """Set local writer-guard context used for critical metadata protections."""
        self._guard_local_node_id = local_node_id
        self._guard_is_leader = bool(is_leader)

    def _enforce_single_writer_guard(self, existing: NodeInfo, incoming: NodeInfo) -> None:
        """Allow liveness updates, but block critical routing/epoch metadata from followers."""
        if self._guard_is_leader:
            return
        if self._guard_local_node_id is None:
            return

        critical_changed = (
            existing.partition_start != incoming.partition_start
            or existing.partition_end != incoming.partition_end
            or existing.leader_epoch != incoming.leader_epoch
        )
        if not critical_changed:
            return

        raise RuntimeError(
            "single-writer guard: non-leader registry instance cannot mutate critical metadata "
            f"for node '{incoming.node_id}'"
        )

    def _ensure_connection(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn is None:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                self._conn = conn
                self._init_schema()
            assert self._conn is not None
            return self._conn

    def _require_connection(self) -> sqlite3.Connection:
        conn = self._ensure_connection()
        if not self._connected:
            raise RuntimeError("Registry must connect before use")
        return conn

    def _init_schema(self) -> None:
        conn = self._conn
        if conn is None:
            return
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                host TEXT,
                port INTEGER,
                role TEXT,
                region TEXT,
                partition_start INTEGER,
                partition_end INTEGER,
                last_seen REAL,
                lsn INTEGER,
                status TEXT
            )
            """
        )
        # Migration: add new columns if they don't exist
        self._migrate_schema(conn)
        conn.commit()

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """Migration-safe schema evolution for Phase 1B/3 columns."""
        cursor = conn.execute("PRAGMA table_info(nodes)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        if "leader_reachable" not in existing_columns:
            logger.info("Migrating schema: adding leader_reachable column")
            conn.execute("ALTER TABLE nodes ADD COLUMN leader_reachable BOOLEAN DEFAULT 1")

        if "wal_reachable" not in existing_columns:
            logger.info("Migrating schema: adding wal_reachable column")
            conn.execute("ALTER TABLE nodes ADD COLUMN wal_reachable BOOLEAN DEFAULT 1")

        if "leader_epoch" not in existing_columns:
            logger.info("Migrating schema: adding leader_epoch column")
            conn.execute("ALTER TABLE nodes ADD COLUMN leader_epoch INTEGER DEFAULT 0")

    @staticmethod
    def _row_to_node(row: sqlite3.Row | tuple | None) -> NodeInfo | None:
        if row is None:
            return None
        # Handle old/new schema widths.
        # Old: (node_id, host, port, role, region, partition_start, partition_end, last_seen, lsn, status)
        # New: (..., status, leader_reachable, wal_reachable, leader_epoch)
        row_tuple = tuple(row)
        if len(row_tuple) == 10:
            # Old schema — add default values
            return NodeInfo(*row_tuple, leader_reachable=True, wal_reachable=True, leader_epoch=0)
        elif len(row_tuple) >= 12:
            if len(row_tuple) >= 13:
                return NodeInfo(*row_tuple[:13])
            return NodeInfo(*row_tuple[:12], leader_epoch=0)
        else:
            # Unexpected — fall back to tuple unpacking
            return NodeInfo(*row_tuple)

    def _fetchall_nodes(self, query: str, params: tuple = ()) -> list[NodeInfo]:
        conn = self._require_connection()
        with self._lock:
            cur = conn.execute(query, params)
            rows = cur.fetchall()
        parsed: list[NodeInfo] = []
        for row in rows:
            node = self._row_to_node(row)
            if node is not None:
                parsed.append(node)
        return parsed

    def _fetchone_node(self, query: str, params: tuple = ()) -> NodeInfo | None:
        conn = self._require_connection()
        with self._lock:
            cur = conn.execute(query, params)
            row = cur.fetchone()
        return self._row_to_node(row)

    async def connect(self) -> None:
        conn = self._ensure_connection()
        with self._lock:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
        self._connected = True
        logger.info("SQLiteRegistry connected (db_path=%s)", self.db_path)

    async def register(self, node: NodeInfo) -> None:
        conn = self._require_connection()
        existing = await self.get_node(node.node_id)
        if existing is not None:
            self._enforce_single_writer_guard(existing, node)
        event = "updated" if existing is not None else "joined"
        with self._lock:
            conn.execute(
                """
                INSERT INTO nodes (node_id, host, port, role, region, partition_start, partition_end, last_seen, lsn, status, leader_reachable, wal_reachable, leader_epoch)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    host=excluded.host, port=excluded.port, role=excluded.role, region=excluded.region,
                    partition_start=excluded.partition_start, partition_end=excluded.partition_end,
                    last_seen=excluded.last_seen, lsn=excluded.lsn, status=excluded.status,
                    leader_reachable=excluded.leader_reachable, wal_reachable=excluded.wal_reachable,
                    leader_epoch=excluded.leader_epoch
                """,
                (
                    node.node_id, node.host, node.port, node.role, node.region,
                    node.partition_start, node.partition_end, node.last_seen, node.lsn, node.status,
                    node.leader_reachable, node.wal_reachable, node.leader_epoch,
                ),
            )
            conn.commit()
        if node.role == "write" and node.status == "healthy":
            self._rebalance_partitions()
            node = await self.get_node(node.node_id) or node
        await self._notify(node, event)

    def _write_nodes_sorted(self) -> list[NodeInfo]:
        return self._fetchall_nodes(
            "SELECT * FROM nodes WHERE role='write' AND status='healthy' ORDER BY node_id"
        )

    def _rebalance_partitions(self) -> None:
        conn = self._require_connection()
        writers = self._write_nodes_sorted()
        if not writers:
            logger.debug("No writers available for rebalance")
            return
        chunk = _PARTITION_TOTAL // len(writers)
        with self._lock:
            for idx, node in enumerate(writers):
                start = idx * chunk
                end = (idx + 1) * chunk if idx < len(writers) - 1 else _PARTITION_TOTAL
                conn.execute(
                    "UPDATE nodes SET partition_start=?, partition_end=? WHERE node_id=?",
                    (start, end, node.node_id),
                )
            conn.commit()
        logger.info("Rebalanced partitions across %d SQLite writers", len(writers))

    def get_owner_for_row(self, row_index: int) -> NodeInfo | None:
        writers = self._write_nodes_sorted()
        if not writers:
            logger.debug("get_owner_for_row: no writers registered")
            return None
        slot = row_index % _PARTITION_TOTAL
        for node in writers:
            if node.partition_start <= slot < node.partition_end:
                logger.debug("get_owner_for_row: row_index=%s slot=%s owner=%s", row_index, slot, node.node_id)
                return node
        return writers[-1]

    async def get_node(self, node_id: str) -> NodeInfo | None:
        return self._fetchone_node("SELECT * FROM nodes WHERE node_id=?", (node_id,))

    async def get_write_node(self) -> NodeInfo | None:
        writers = self._write_nodes_sorted()
        return writers[0] if writers else None

    async def get_write_nodes(self) -> list[NodeInfo]:
        return self._fetchall_nodes(
            "SELECT * FROM nodes WHERE role='write' AND status='healthy' ORDER BY node_id"
        )

    async def get_read_nodes(self) -> list[NodeInfo]:
        return self._fetchall_nodes(
            "SELECT * FROM nodes WHERE role='read' AND status!='failed' ORDER BY node_id"
        )

    async def update_lsn(self, node_id: str, lsn: int) -> None:
        conn = self._require_connection()
        with self._lock:
            conn.execute(
                "UPDATE nodes SET lsn=?, last_seen=? WHERE node_id=?",
                (lsn, time.time(), node_id),
            )
            conn.commit()
        node = await self.get_node(node_id)
        if node is not None:
            await self._notify(node, "updated")

    async def mark_failed(self, node_id: str) -> None:
        conn = self._require_connection()
        node = await self.get_node(node_id)
        if node is None:
            logger.debug("mark_failed: node %s not found", node_id)
            return
        with self._lock:
            conn.execute(
                "UPDATE nodes SET status='failed', last_seen=? WHERE node_id=?",
                (time.time(), node_id),
            )
            conn.commit()
        self._rebalance_partitions()
        failed_node = await self.get_node(node_id) or NodeInfo(
            node_id=node.node_id,
            host=node.host,
            port=node.port,
            role=node.role,
            region=node.region,
            partition_start=node.partition_start,
            partition_end=node.partition_end,
            last_seen=time.time(),
            lsn=node.lsn,
            status="failed",
        )
        await self._notify(failed_node, "failed")

    async def mark_suspect(self, node_id: str) -> None:
        """Mark a node as suspect (intermediate state before failure)."""
        conn = self._require_connection()
        node = await self.get_node(node_id)
        if node is None:
            logger.debug("mark_suspect: node %s not found", node_id)
            return
        if node.status == "suspect":
            logger.debug("mark_suspect: node %s already suspect", node_id)
            return
        with self._lock:
            conn.execute(
                "UPDATE nodes SET status='suspect', last_seen=? WHERE node_id=?",
                (time.time(), node_id),
            )
            conn.commit()
        if node.role == "write":
            # Suspect writers should stop owning partitions immediately.
            self._rebalance_partitions()
        logger.info("Node marked suspect: %s", node_id)
        suspect_node = await self.get_node(node_id) or node
        await self._notify(suspect_node, "suspect")

    async def apply_partition_map(self, partition_map: list[dict[str, int | str]]) -> int:
        """Apply leader-provided partition ownership map without triggering writer guard."""
        conn = self._require_connection()
        applied = 0
        with self._lock:
            for entry in partition_map:
                node_id = str(entry.get("node_id", "") or "")
                if not node_id:
                    continue
                try:
                    start = int(entry.get("partition_start"))
                    end = int(entry.get("partition_end"))
                except (TypeError, ValueError):
                    continue
                cur = conn.execute(
                    "UPDATE nodes SET partition_start=?, partition_end=? WHERE node_id=?",
                    (start, end, node_id),
                )
                if cur.rowcount and int(cur.rowcount) > 0:
                    applied += 1
            conn.commit()
        if applied:
            logger.info("Applied partition map entries=%d", applied)
        return applied

    async def update_capability_flags(self, node_id: str, leader_reachable: bool, wal_reachable: bool) -> None:
        """Update leader_reachable and wal_reachable flags for a node."""
        conn = self._require_connection()
        node = await self.get_node(node_id)
        if node is None:
            logger.debug("update_capability_flags: node %s not found", node_id)
            return
        with self._lock:
            conn.execute(
                "UPDATE nodes SET leader_reachable=?, wal_reachable=? WHERE node_id=?",
                (leader_reachable, wal_reachable, node_id),
            )
            conn.commit()
        logger.debug("update_capability_flags: node=%s leader_reachable=%s wal_reachable=%s",
                     node_id, leader_reachable, wal_reachable)
        updated_node = await self.get_node(node_id) or node
        await self._notify(updated_node, "updated")

    async def watch(self, callback: WatchCallback) -> None:
        self._watchers.append(callback)

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
        self._connected = False

    async def _notify(self, node: NodeInfo, event_type: str) -> None:
        for watcher in list(self._watchers):
            await watcher(node, event_type)
