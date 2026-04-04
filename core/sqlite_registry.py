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

    def _ensure_connection(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn is None:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                self._conn = conn
                self._init_schema()
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
        conn.commit()

    @staticmethod
    def _row_to_node(row: sqlite3.Row | tuple | None) -> NodeInfo | None:
        if row is None:
            return None
        return NodeInfo(*tuple(row))

    def _fetchall_nodes(self, query: str, params: tuple = ()) -> list[NodeInfo]:
        conn = self._require_connection()
        with self._lock:
            cur = conn.execute(query, params)
            rows = cur.fetchall()
        return [NodeInfo(*tuple(row)) for row in rows]

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
        event = "updated" if existing is not None else "joined"
        with self._lock:
            conn.execute(
                """
                INSERT INTO nodes (node_id, host, port, role, region, partition_start, partition_end, last_seen, lsn, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    host=excluded.host, port=excluded.port, role=excluded.role, region=excluded.region,
                    partition_start=excluded.partition_start, partition_end=excluded.partition_end,
                    last_seen=excluded.last_seen, lsn=excluded.lsn, status=excluded.status
                """,
                (
                    node.node_id, node.host, node.port, node.role, node.region,
                    node.partition_start, node.partition_end, node.last_seen, node.lsn, node.status,
                ),
            )
            conn.commit()
        if node.role == "write" and node.status != "failed":
            self._rebalance_partitions()
            node = await self.get_node(node.node_id) or node
        await self._notify(node, event)

    def _write_nodes_sorted(self) -> list[NodeInfo]:
        return self._fetchall_nodes(
            "SELECT * FROM nodes WHERE role='write' AND status!='failed' ORDER BY node_id"
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
            "SELECT * FROM nodes WHERE role='write' AND status!='failed' ORDER BY node_id"
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
