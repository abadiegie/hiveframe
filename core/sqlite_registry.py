# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0
"""SQLite-based node registry for dev/homelab/single-machine deployments."""

import sqlite3
import time
from typing import Awaitable, Callable
from dataclasses import dataclass
import logging

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

class SQLiteRegistry:
    def __init__(self, db_path: str = ".hiveframe/registry.db"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()
        self._watchers: list[WatchCallback] = []

    def _init_schema(self):
        self._conn.execute("""
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
        """)
        self._conn.commit()

    async def connect(self) -> None:
        logger.info("SQLiteRegistry connected (db_path=%s)", self._conn)

    async def register(self, node: NodeInfo) -> None:
        self._conn.execute(
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
                node.partition_start, node.partition_end, node.last_seen, node.lsn, node.status
            )
        )
        self._conn.commit()
        await self._notify(node, "joined")

    async def get_node(self, node_id: str) -> NodeInfo | None:
        cur = self._conn.execute("SELECT * FROM nodes WHERE node_id=?", (node_id,))
        row = cur.fetchone()
        if not row:
            return None
        return NodeInfo(*row)

    async def get_write_nodes(self) -> list[NodeInfo]:
        cur = self._conn.execute("SELECT * FROM nodes WHERE role='write' AND status!='failed'")
        return [NodeInfo(*row) for row in cur.fetchall()]

    async def get_read_nodes(self) -> list[NodeInfo]:
        cur = self._conn.execute("SELECT * FROM nodes WHERE role='read' AND status!='failed'")
        return [NodeInfo(*row) for row in cur.fetchall()]

    async def update_lsn(self, node_id: str, lsn: int) -> None:
        self._conn.execute(
            "UPDATE nodes SET lsn=?, last_seen=? WHERE node_id=?",
            (lsn, time.time(), node_id)
        )
        self._conn.commit()

    async def mark_failed(self, node_id: str) -> None:
        self._conn.execute(
            "UPDATE nodes SET status='failed', last_seen=? WHERE node_id=?",
            (time.time(), node_id)
        )
        self._conn.commit()

    async def watch(self, callback: WatchCallback) -> None:
        self._watchers.append(callback)

    async def _notify(self, node: NodeInfo, event_type: str) -> None:
        for watcher in list(self._watchers):
            await watcher(node, event_type)
