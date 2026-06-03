# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Operation log for metadata sync across nodes (Phase 3 foundation)."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import sqlite3
import tempfile
import time
from pathlib import Path
from threading import RLock
from typing import Any

from .message import Message, MessageType


logger = logging.getLogger("core.op_log")


class OperationLog:
    """SQLite-backed operation log with push/pull/full-resync primitives."""

    def __init__(
        self,
        db_path: str,
        node_id: str,
        *,
        leader_epoch: int = 0,
    ) -> None:
        self.db_path = db_path
        self.node_id = node_id
        self._lock = RLock()
        db_file = Path(db_path)
        if db_file.parent != Path(""):
            db_file.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema(leader_epoch)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _resync_request_id(node_id: str, label: str) -> str:
        return f"oplog-resync-{label}-{node_id}-{int(time.time() * 1000)}"

    def _init_schema(self, leader_epoch: int) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS op_log (
                    op_id TEXT PRIMARY KEY,
                    entity TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    origin_node_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS op_meta (
                    meta_key TEXT PRIMARY KEY,
                    meta_value TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_op_status ON op_log(status)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_op_entity_key ON op_log(entity, key)"
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO op_meta(meta_key, meta_value) VALUES ('local_seq', '0')"
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO op_meta(meta_key, meta_value) VALUES ('leader_epoch', ?)",
                (str(int(leader_epoch)),),
            )
            self._conn.commit()

    def _next_local_seq(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT meta_value FROM op_meta WHERE meta_key='local_seq'"
            ).fetchone()
            seq = int(row[0]) if row is not None else 0
            next_seq = seq + 1
            self._conn.execute(
                "INSERT INTO op_meta(meta_key, meta_value) VALUES('local_seq', ?) "
                "ON CONFLICT(meta_key) DO UPDATE SET meta_value=excluded.meta_value",
                (str(next_seq),),
            )
            self._conn.commit()
            return next_seq

    def _leader_epoch(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT meta_value FROM op_meta WHERE meta_key='leader_epoch'"
            ).fetchone()
            return int(row[0]) if row is not None else 0

    def get_leader_epoch(self) -> int:
        return self._leader_epoch()

    @staticmethod
    def parse_op_id(op_id: str | None) -> tuple[int, int] | None:
        if not op_id:
            return None
        raw = str(op_id)
        if "-" not in raw:
            return None
        epoch_text, seq_text = raw.split("-", 1)
        if not epoch_text.isdigit() or not seq_text.isdigit():
            return None
        return int(epoch_text), int(seq_text)

    @classmethod
    def estimate_gap(cls, local_op_id: str | None, leader_op_id: str | None) -> int:
        local_parsed = cls.parse_op_id(local_op_id)
        leader_parsed = cls.parse_op_id(leader_op_id)
        if local_parsed is None or leader_parsed is None:
            return 0
        local_epoch, local_seq = local_parsed
        leader_epoch, leader_seq = leader_parsed
        if leader_epoch > local_epoch:
            # Conservative large gap for epoch mismatch.
            return 1_000_000_000
        return max(0, leader_seq - local_seq)

    def set_leader_epoch(self, epoch: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO op_meta(meta_key, meta_value) VALUES('leader_epoch', ?) "
                "ON CONFLICT(meta_key) DO UPDATE SET meta_value=excluded.meta_value",
                (str(int(epoch)),),
            )
            self._conn.commit()

    def append_local(self, entity: str, key: str, value: Any) -> str:
        seq = self._next_local_seq()
        epoch = self._leader_epoch()
        op_id = f"{epoch}-{seq:020d}"
        now = time.time()
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO op_log(op_id, entity, key, value_json, version, origin_node_id, status, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (op_id, entity, key, payload, 1, self.node_id, now, now),
            )
            self._conn.commit()
        return op_id

    def append_local_acked(self, entity: str, key: str, value: Any) -> str:
        """Append local operation and mark it acked immediately (leader-local metadata updates)."""
        op_id = self.append_local(entity, key, value)
        self.apply_acked(
            [
                {
                    "op_id": op_id,
                    "entity": entity,
                    "key": key,
                    "value": value,
                    "version": 1,
                    "origin_node_id": self.node_id,
                    "created_at": time.time(),
                }
            ]
        )
        return op_id

    def get_pending_ops(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM op_log WHERE status='pending' ORDER BY op_id"
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_last_acked_op_id(self) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT op_id FROM op_log WHERE status='acked' ORDER BY op_id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return str(row[0])

    def export_since(self, since_op_id: str | None) -> list[dict[str, Any]]:
        if since_op_id:
            query = (
                "SELECT * FROM op_log WHERE status='acked' AND op_id > ? ORDER BY op_id"
            )
            params: tuple[Any, ...] = (since_op_id,)
        else:
            query = "SELECT * FROM op_log WHERE status='acked' ORDER BY op_id"
            params = ()
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def export_all_acked(self) -> list[dict[str, Any]]:
        return self.export_since(None)

    def dedup_check(self, op_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM op_log WHERE op_id=? LIMIT 1", (op_id,)
            ).fetchone()
        return row is not None

    def apply_acked(self, ops: list[dict[str, Any]]) -> None:
        now = time.time()
        with self._lock:
            for op in ops:
                op_id = str(op["op_id"])
                payload = json.dumps(op.get("value", None), sort_keys=True, separators=(",", ":"))
                self._conn.execute(
                    """
                    INSERT INTO op_log(op_id, entity, key, value_json, version, origin_node_id, status, created_at, updated_at)
                    VALUES(?, ?, ?, ?, ?, ?, 'acked', ?, ?)
                    ON CONFLICT(op_id) DO UPDATE SET
                        entity=excluded.entity,
                        key=excluded.key,
                        value_json=excluded.value_json,
                        version=excluded.version,
                        origin_node_id=excluded.origin_node_id,
                        status='acked',
                        updated_at=excluded.updated_at
                    """,
                    (
                        op_id,
                        str(op.get("entity", "registry")),
                        str(op.get("key", "")),
                        payload,
                        int(op.get("version", 1)),
                        str(op.get("origin_node_id", self.node_id)),
                        float(op.get("created_at", now)),
                        now,
                    ),
                )
            self._conn.commit()

    def mark_rejected(self, op_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE op_log SET status='rejected_by_leader', updated_at=? WHERE op_id=?",
                (time.time(), op_id),
            )
            self._conn.commit()

    def accept_from_follower(
        self,
        ops: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Leader-side conflict resolution with default leader_wins policy."""
        acked: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        for op in ops:
            op_id = str(op.get("op_id", ""))
            if not op_id:
                continue
            if self.dedup_check(op_id):
                existing = self._load_op(op_id)
                if existing is not None:
                    acked.append(existing)
                continue

            entity = str(op.get("entity", "registry"))
            key = str(op.get("key", ""))
            latest = self._latest_acked_for_key(entity, key)
            if latest is not None:
                latest_value = latest.get("value")
                incoming_value = op.get("value")
                if latest_value != incoming_value:
                    # leader_wins: reject conflicting follower update
                    rejected.append(op)
                    continue

            normalized = {
                "op_id": op_id,
                "entity": entity,
                "key": key,
                "value": op.get("value"),
                "version": int(op.get("version", 1)),
                "origin_node_id": str(op.get("origin_node_id", self.node_id)),
                "created_at": float(op.get("created_at", time.time())),
            }
            self.apply_acked([normalized])
            acked.append(normalized)

        return {"acked_ops": acked, "rejected_ops": rejected}

    async def push_to_leader(self, leader_node_id: str, transport: Any) -> int:
        pending = self.get_pending_ops()
        if not pending:
            return 0
        request_id = f"oplog-push-{self.node_id}-{int(time.time() * 1000)}"
        payload = {
            "request_id": request_id,
            "source_node_id": self.node_id,
            "ops": pending,
        }
        msg = Message.build(
            message_type=MessageType.OPLOG_PUSH,
            sender_id=self.node_id,
            sender_region="cluster",
            payload=payload,
        )
        response = await transport.request(leader_node_id, msg, timeout=5.0)
        if response is None:
            return 0

        acked = list(response.payload.get("acked_ops", []))
        rejected = list(response.payload.get("rejected_ops", []))
        if acked:
            self.apply_acked(acked)
        for op in rejected:
            op_id = str(op.get("op_id", ""))
            if op_id:
                self.mark_rejected(op_id)
        return len(acked) + len(rejected)

    async def pull_from_leader(
        self,
        leader_node_id: str,
        transport: Any,
        since_op_id: str | None,
    ) -> int:
        request_id = f"oplog-pull-{self.node_id}-{int(time.time() * 1000)}"
        msg = Message.build(
            message_type=MessageType.OPLOG_PULL,
            sender_id=self.node_id,
            sender_region="cluster",
            payload={
                "request_id": request_id,
                "since_op_id": since_op_id,
            },
        )
        response = await transport.request(leader_node_id, msg, timeout=5.0)
        if response is None:
            return 0
        ops = list(response.payload.get("ops", []))
        if not ops:
            return 0
        self.apply_acked(ops)
        return len(ops)

    async def full_resync_from_leader(self, leader_node_id: str, transport: Any) -> None:
        manifest = await self._request_full_resync_manifest(leader_node_id, transport)
        if manifest is not None:
            applied = await self._download_and_apply_snapshot_chunks(
                leader_node_id,
                transport,
                chunk_count=int(manifest.get("chunk_count", 0)),
                expected_sha256=str(manifest.get("sha256", "")),
            )
            if applied:
                return

        await self._fallback_full_resync_ops(leader_node_id, transport)

    async def _request_full_resync_manifest(self, leader_node_id: str, transport: Any) -> dict[str, Any] | None:
        request_id = self._resync_request_id(self.node_id, "manifest")
        msg = Message.build(
            message_type=MessageType.OPLOG_FULL_RESYNC,
            sender_id=self.node_id,
            sender_region="cluster",
            payload={"request_id": request_id, "mode": "manifest"},
        )
        response = await transport.request(leader_node_id, msg, timeout=10.0)
        if response is None:
            return None
        if str(response.payload.get("mode", "")) != "manifest":
            # Backward compatibility with older response path.
            snapshot_b64 = response.payload.get("snapshot_b64")
            if isinstance(snapshot_b64, str) and snapshot_b64:
                try:
                    self.restore_sqlite_snapshot_b64(snapshot_b64)
                    return None
                except Exception:
                    return None
            return None
        return dict(response.payload)

    async def _download_and_apply_snapshot_chunks(
        self,
        leader_node_id: str,
        transport: Any,
        *,
        chunk_count: int,
        expected_sha256: str,
    ) -> bool:
        if chunk_count <= 0:
            return False
        buffer = bytearray()
        for chunk_index in range(chunk_count):
            request_id = self._resync_request_id(self.node_id, f"chunk-{chunk_index}")
            msg = Message.build(
                message_type=MessageType.OPLOG_FULL_RESYNC,
                sender_id=self.node_id,
                sender_region="cluster",
                payload={
                    "request_id": request_id,
                    "mode": "chunk",
                    "chunk_index": chunk_index,
                },
            )
            response = await transport.request(leader_node_id, msg, timeout=10.0)
            if response is None:
                return False
            if str(response.payload.get("mode", "")) != "chunk":
                return False
            chunk_b64 = response.payload.get("chunk_b64")
            if not isinstance(chunk_b64, str) or not chunk_b64:
                return False
            try:
                buffer.extend(base64.b64decode(chunk_b64.encode("ascii")))
            except Exception:
                return False

        digest = hashlib.sha256(bytes(buffer)).hexdigest()
        if expected_sha256 and digest != expected_sha256:
            logger.warning(
                "op_log full-resync snapshot checksum mismatch: expected=%s got=%s",
                expected_sha256,
                digest,
            )
            return False
        try:
            snapshot_b64 = base64.b64encode(bytes(buffer)).decode("ascii")
            self.restore_sqlite_snapshot_b64(snapshot_b64)
            return True
        except Exception as exc:
            logger.warning("op_log full-resync snapshot apply failed: %s", exc)
            return False

    async def _fallback_full_resync_ops(self, leader_node_id: str, transport: Any) -> None:
        request_id = self._resync_request_id(self.node_id, "ops")
        msg = Message.build(
            message_type=MessageType.OPLOG_FULL_RESYNC,
            sender_id=self.node_id,
            sender_region="cluster",
            payload={"request_id": request_id, "mode": "ops"},
        )
        response = await transport.request(leader_node_id, msg, timeout=10.0)
        if response is None:
            return
        ops = list(response.payload.get("ops", []))
        with self._lock:
            self._conn.execute("DELETE FROM op_log")
            self._conn.commit()
        self.apply_acked(ops)

    def export_sqlite_snapshot_b64(self) -> str:
        """Create a consistent SQLite snapshot via backup() and return base64 bytes."""
        with self._lock:
            with tempfile.NamedTemporaryFile(prefix="hiveframe_oplog_", suffix=".db", delete=False) as temp:
                temp_path = temp.name
            try:
                backup_conn = sqlite3.connect(temp_path)
                try:
                    self._conn.backup(backup_conn)
                    backup_conn.commit()
                finally:
                    backup_conn.close()
                payload = Path(temp_path).read_bytes()
            finally:
                Path(temp_path).unlink(missing_ok=True)
        return base64.b64encode(payload).decode("ascii")

    @staticmethod
    def snapshot_chunks(snapshot_b64: str, chunk_size_bytes: int) -> tuple[list[str], str]:
        raw = base64.b64decode(snapshot_b64.encode("ascii"))
        if chunk_size_bytes <= 0:
            chunk_size_bytes = 256 * 1024
        chunks: list[str] = []
        for start in range(0, len(raw), chunk_size_bytes):
            part = raw[start:start + chunk_size_bytes]
            chunks.append(base64.b64encode(part).decode("ascii"))
        digest = hashlib.sha256(raw).hexdigest()
        return chunks, digest

    def restore_sqlite_snapshot_b64(self, snapshot_b64: str) -> None:
        """Replace local DB with leader snapshot, then reopen connection safely."""
        payload = base64.b64decode(snapshot_b64.encode("ascii"))
        target = Path(self.db_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            self._conn.close()
            target.write_bytes(payload)
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._init_schema(self._leader_epoch())

    def _latest_acked_for_key(self, entity: str, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM op_log
                WHERE entity=? AND key=? AND status='acked'
                ORDER BY updated_at DESC, op_id DESC
                LIMIT 1
                """,
                (entity, key),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def _load_op(self, op_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM op_log WHERE op_id=?", (op_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        value_raw = row["value_json"]
        try:
            value = json.loads(value_raw)
        except Exception:
            value = value_raw
        return {
            "op_id": str(row["op_id"]),
            "entity": str(row["entity"]),
            "key": str(row["key"]),
            "value": value,
            "version": int(row["version"]),
            "origin_node_id": str(row["origin_node_id"]),
            "status": str(row["status"]),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }


