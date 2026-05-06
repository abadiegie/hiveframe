# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""In-memory write-ahead log with thread-safe append operations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
import os
import re
from urllib.parse import urlparse, urlunparse
from typing import Any
import json
import logging
from pathlib import Path

from .transaction import Transaction, TxState


logger = logging.getLogger("core.wal")


@dataclass(slots=True)
class WALEntry:
    """Serializable WAL entry."""

    lsn: int
    tx_id: str
    operations: list[dict[str, Any]]
    timestamp: str
    state: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize WAL entry as dict."""
        return {
            "lsn": self.lsn,
            "tx_id": self.tx_id,
            "operations": self.operations,
            "timestamp": self.timestamp,
            "state": self.state,
        }


class WriteAheadLog:
    """Append-only, in-memory WAL implementation for Phase 1. Now supports optional disk persistence."""

    def __init__(self, wal_path: str | None = None) -> None:
        self._lock = Lock()
        self._entries: list[WALEntry] = []
        self._next_lsn = 1
        self._wal_path = Path(wal_path) if wal_path else None
        logger.info("WriteAheadLog initialized%s", f" (persisted to {self._wal_path})" if self._wal_path else "")
        if self._wal_path:
            self._recover_from_disk()

    def append(self, transaction: Transaction) -> int:
        """Append a transaction snapshot and return its LSN. Also persist to disk if enabled."""
        with self._lock:
            lsn = self._next_lsn
            self._next_lsn += 1
            entry = WALEntry(
                lsn=lsn,
                tx_id=transaction.tx_id,
                operations=[op.to_dict() for op in transaction.operations],
                timestamp=datetime.now(timezone.utc).isoformat(),
                state=transaction.state.value,
            )
            self._entries.append(entry)
            if self._wal_path:
                with open(self._wal_path, "a") as f:
                    f.write(json.dumps(entry.to_dict()) + "\n")
            logger.info("WAL.append lsn=%s tx_id=%s state=%s ops=%d", lsn, transaction.tx_id, transaction.state.value, len(entry.operations))
            return lsn

    def _recover_from_disk(self) -> None:
        """Replay WAL dari disk saat startup."""
        if not self._wal_path.exists():
            return
        with open(self._wal_path) as f:
            for line in f:
                entry_dict = json.loads(line)
                self._entries.append(WALEntry(**entry_dict))
        self._next_lsn = len(self._entries) + 1
        logger.info("WAL recovered %d entries from %s", len(self._entries), self._wal_path)

    def compact(self, keep_last_n: int = 1000) -> int:
        """
        Buang entries lama, simpan hanya N terakhir.
        Sebelum compaction, pastikan read_node sudah sync sampai LSN yang akan di-compacted.
        Return: jumlah entries yang dibuang.
        """
        with self._lock:
            if len(self._entries) <= keep_last_n:
                return 0
            removed = len(self._entries) - keep_last_n
            self._entries = self._entries[-keep_last_n:]
            logger.info("WAL.compact: removed %d entries, kept %d", removed, keep_last_n)
            return removed

    def compact_before_lsn(self, lsn: int) -> int:
        """Buang semua entries dengan LSN < lsn."""
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if e.lsn >= lsn]
            removed = before - len(self._entries)
            logger.info("WAL.compact_before_lsn: removed %d entries before lsn %d", removed, lsn)
            return removed

    def get_since(self, lsn: int) -> list[dict[str, Any]]:
        """Return entries with LSN greater than the given value."""
        with self._lock:
            results = [entry.to_dict() for entry in self._entries if entry.lsn > lsn]
            logger.debug("WAL.get_since lsn=%s results=%d", lsn, len(results))
            return results

    def get_committed(self) -> list[dict[str, Any]]:
        """Return entries that represent committed transactions."""
        with self._lock:
            results = [
                entry.to_dict()
                for entry in self._entries
                if entry.state in {TxState.COMMITTED.value, TxState.SYNCED.value}
            ]
            logger.debug("WAL.get_committed count=%d", len(results))
            return results

    def get_cell_history(self, cell_id: str) -> list[dict]:
        """
        Return semua perubahan untuk satu cell, urut dari terlama.
        Ini adalah audit trail per-cell yang bisa di-query.
        """
        with self._lock:
            history = []
            for entry in self._entries:
                for op in entry.operations:
                    if op["cell_id"] == cell_id:
                        history.append({
                            "lsn": entry.lsn,
                            "timestamp": entry.timestamp,
                            "tx_id": entry.tx_id,
                            "old_value": op["old_value"],
                            "new_value": op["new_value"],
                            "author_type": op["author_type"],
                            "author_id": op["author_id"],
                            "confidence": op.get("confidence"),
                        })
            return history

    def get_metrics(self) -> dict[str, int]:
        with self._lock:
            return {
                "total_entries": len(self._entries),
                "last_lsn": self._next_lsn - 1,
            }


class RedisWriteAheadLog:
    """Redis-backed WAL with global INCR LSN ordering.

    This is a staged backend for multi-instance consistency: Redis stores the
    authoritative WAL sequence while local write/read nodes remain materialized
    state for fast access.
    """

    def __init__(
        self,
        redis_url: str,
        prefix: str = "hiveframe:wal",
        redis_client: Any | None = None,
    ) -> None:
        self._lock = Lock()
        self._prefix = prefix
        self._lsn_key = f"{prefix}:lsn"
        self._entries_key = f"{prefix}:entries"

        if redis_client is not None:
            self._redis = redis_client
        else:
            try:
                import redis
            except ImportError as exc:
                raise ImportError(
                    "redis package required for RedisWriteAheadLog: pip install hiveframe[redis]"
                ) from exc
            self._redis = redis.Redis.from_url(redis_url, decode_responses=True)

        logger.info("RedisWriteAheadLog initialized prefix=%s", self._prefix)

    def _entry_from_tx(self, lsn: int, transaction: Transaction) -> dict[str, Any]:
        return {
            "lsn": lsn,
            "tx_id": transaction.tx_id,
            "operations": [op.to_dict() for op in transaction.operations],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": transaction.state.value,
        }

    def _decode_entries(self, raw_entries: list[str]) -> list[dict[str, Any]]:
        decoded: list[dict[str, Any]] = []
        for raw in raw_entries:
            try:
                decoded.append(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                logger.warning("Redis WAL decode failure; skipping malformed entry")
        return decoded

    def append(self, transaction: Transaction) -> int:
        with self._lock:
            lsn = int(self._redis.incr(self._lsn_key))
            entry = self._entry_from_tx(lsn, transaction)
            payload = json.dumps(entry, separators=(",", ":"), sort_keys=True)
            self._redis.zadd(self._entries_key, {payload: lsn})
            logger.info(
                "RedisWAL.append lsn=%s tx_id=%s state=%s ops=%d",
                lsn,
                transaction.tx_id,
                transaction.state.value,
                len(entry["operations"]),
            )
            return lsn

    def get_since(self, lsn: int) -> list[dict[str, Any]]:
        with self._lock:
            raw = self._redis.zrangebyscore(self._entries_key, f"({lsn}", "+inf")
            results = self._decode_entries(raw)
            logger.debug("RedisWAL.get_since lsn=%s results=%d", lsn, len(results))
            return results

    def get_committed(self) -> list[dict[str, Any]]:
        with self._lock:
            raw = self._redis.zrange(self._entries_key, 0, -1)
            entries = self._decode_entries(raw)
            results = [
                entry
                for entry in entries
                if entry.get("state") in {TxState.COMMITTED.value, TxState.SYNCED.value}
            ]
            logger.debug("RedisWAL.get_committed count=%d", len(results))
            return results

    def compact(self, keep_last_n: int = 1000) -> int:
        with self._lock:
            size = int(self._redis.zcard(self._entries_key))
            if size <= keep_last_n:
                return 0
            remove_count = size - keep_last_n
            self._redis.zremrangebyrank(self._entries_key, 0, remove_count - 1)
            logger.info("RedisWAL.compact: removed %d entries, kept %d", remove_count, keep_last_n)
            return remove_count

    def compact_before_lsn(self, lsn: int) -> int:
        with self._lock:
            removed = int(self._redis.zremrangebyscore(self._entries_key, "-inf", lsn - 1))
            logger.info("RedisWAL.compact_before_lsn: removed %d entries before lsn %d", removed, lsn)
            return removed

    def get_cell_history(self, cell_id: str) -> list[dict]:
        with self._lock:
            raw = self._redis.zrange(self._entries_key, 0, -1)
            entries = self._decode_entries(raw)
            history = []
            for entry in entries:
                for op in entry.get("operations", []):
                    if op.get("cell_id") == cell_id:
                        history.append(
                            {
                                "lsn": entry.get("lsn"),
                                "timestamp": entry.get("timestamp"),
                                "tx_id": entry.get("tx_id"),
                                "old_value": op.get("old_value"),
                                "new_value": op.get("new_value"),
                                "author_type": op.get("author_type"),
                                "author_id": op.get("author_id"),
                                "confidence": op.get("confidence"),
                            }
                        )
            return history

    def get_metrics(self) -> dict[str, int]:
        with self._lock:
            total_entries = int(self._redis.zcard(self._entries_key))
            last_entries = self._decode_entries(self._redis.zrange(self._entries_key, -1, -1))
            last_lsn = int(last_entries[0].get("lsn", 0)) if last_entries else 0
            return {
                "total_entries": total_entries,
                "last_lsn": last_lsn,
            }


class MySQLWriteAheadLog:
    """MySQL-backed WAL using AUTO_INCREMENT LSN ordering.

    This backend is optional and intended for multi-instance deployments where
    a shared SQL store is preferred over Redis.
    """

    def __init__(
        self,
        mysql_dsn: str,
        table_name: str = "hiveframe_wal",
        mysql_conn: Any | None = None,
        auto_create_db: bool = False,
    ) -> None:
        self._lock = Lock()
        self._table = self._validate_table_name(table_name)
        self._tx_cells_table = self._validate_table_name(f"{self._table}_tx_cells")
        self._table_sql = self._quote_mysql_identifier(self._table)
        self._tx_cells_table_sql = self._quote_mysql_identifier(self._tx_cells_table)

        if mysql_conn is not None:
            self._conn = mysql_conn
        else:
            try:
                import pymysql
            except ImportError as exc:
                raise ImportError(
                    "pymysql package required for MySQLWriteAheadLog: pip install hiveframe[mysql]"
                ) from exc
            if auto_create_db:
                self._ensure_database(pymysql, mysql_dsn)
            params = self._parse_mysql_dsn(mysql_dsn)
            self._conn = pymysql.connect(autocommit=True, **params)

        self._ensure_table()
        logger.info("MySQLWriteAheadLog initialized table=%s", self._table)

    @staticmethod
    def _parse_mysql_dsn(dsn: str, include_database: bool = True) -> dict[str, Any]:
        parsed = urlparse(dsn)
        if parsed.scheme not in {"mysql", "mysql+pymysql"}:
            raise ValueError("mysql dsn must use mysql:// or mysql+pymysql://")
        database = parsed.path.lstrip("/")
        if not database:
            raise ValueError("mysql dsn must include database name")
        params = {
            "host": parsed.hostname or "127.0.0.1",
            "port": parsed.port or 3306,
            "user": parsed.username,
            "password": parsed.password,
            "charset": "utf8mb4",
        }
        if include_database:
            params["database"] = database
        return params

    @staticmethod
    def _quote_mysql_identifier(name: str) -> str:
        return f"`{name.replace('`', '``')}`"

    @staticmethod
    def _validate_table_name(name: str) -> str:
        table_name = str(name).strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
            raise ValueError(
                f"invalid mysql table name '{name}'. Use letters, numbers, and underscores only."
            )
        if len(table_name) > 64:
            raise ValueError(
                f"invalid mysql table name '{name}'. MySQL identifier length must be <= 64."
            )
        return table_name

    @staticmethod
    def override_dsn_database(dsn: str, database: str) -> str:
        parsed = urlparse(dsn)
        if parsed.scheme not in {"mysql", "mysql+pymysql"}:
            raise ValueError("mysql dsn must use mysql:// or mysql+pymysql://")
        db_name = database.strip()
        if not db_name:
            raise ValueError("mysql database override must not be empty")
        return urlunparse(parsed._replace(path=f"/{db_name}"))

    def _ensure_database(self, pymysql_module: Any, dsn: str) -> None:
        params = self._parse_mysql_dsn(dsn)
        database = str(params["database"])
        bootstrap_params = self._parse_mysql_dsn(dsn, include_database=False)
        conn = pymysql_module.connect(autocommit=True, **bootstrap_params)
        try:
            sql = f"CREATE DATABASE IF NOT EXISTS {self._quote_mysql_identifier(database)}"
            with conn.cursor() as cur:
                cur.execute(sql)
        finally:
            close = getattr(conn, "close", None)
            if callable(close):
                close()

    def _ensure_table(self) -> None:
        wal_sql = (
            f"CREATE TABLE IF NOT EXISTS {self._table_sql} ("
            "lsn BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,"
            "tx_id VARCHAR(128) NOT NULL,"
            "state VARCHAR(32) NOT NULL,"
            "ts VARCHAR(64) NOT NULL,"
            "operations_json LONGTEXT NOT NULL"
            ") ENGINE=InnoDB"
        )
        tx_cells_sql = (
            f"CREATE TABLE IF NOT EXISTS {self._tx_cells_table_sql} ("
            "id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,"
            "lsn BIGINT NOT NULL,"
            "tx_id VARCHAR(128) NOT NULL,"
            "frame_id VARCHAR(128) NULL,"
            "cell_id VARCHAR(255) NOT NULL,"
            "INDEX idx_tx_id (tx_id),"
            "INDEX idx_frame_id (frame_id),"
            "INDEX idx_cell_id (cell_id),"
            "INDEX idx_lsn (lsn)"
            ") ENGINE=InnoDB"
        )
        with self._conn.cursor() as cur:
            cur.execute(wal_sql)
            cur.execute(tx_cells_sql)

    @staticmethod
    def _extract_frame_id(cell_id: Any) -> str | None:
        if not isinstance(cell_id, str):
            return None
        if "::" not in cell_id:
            return None
        frame_id = cell_id.split("::", 1)[0].strip()
        return frame_id or None

    def append(self, transaction: Transaction) -> int:
        with self._lock:
            entry_ts = datetime.now(timezone.utc).isoformat()
            operations_json = json.dumps([op.to_dict() for op in transaction.operations], separators=(",", ":"), sort_keys=True)
            lsn = 0
            try:
                with self._conn.cursor() as cur:
                    cur.execute("START TRANSACTION")
                    cur.execute(
                        (
                            f"INSERT INTO {self._table_sql} (tx_id, state, ts, operations_json) "
                            "VALUES (%s, %s, %s, %s)"
                        ),
                        (
                            transaction.tx_id,
                            transaction.state.value,
                            entry_ts,
                            operations_json,
                        ),
                    )
                    lsn = int(getattr(cur, "lastrowid", 0) or 0)
                    if transaction.operations:
                        cur.executemany(
                            (
                                f"INSERT INTO {self._tx_cells_table_sql} (lsn, tx_id, frame_id, cell_id) "
                                "VALUES (%s, %s, %s, %s)"
                            ),
                            [
                                (
                                    lsn,
                                    transaction.tx_id,
                                    self._extract_frame_id(operation.cell_id),
                                    operation.cell_id,
                                )
                                for operation in transaction.operations
                            ],
                        )
                    cur.execute("COMMIT")
            except Exception:
                with self._conn.cursor() as rollback_cur:
                    rollback_cur.execute("ROLLBACK")
                raise
            logger.info(
                "MySQLWAL.append lsn=%s tx_id=%s state=%s ops=%d",
                lsn,
                transaction.tx_id,
                transaction.state.value,
                len(transaction.operations),
            )
            return lsn

    @staticmethod
    def _row_to_entry(row: tuple[Any, ...]) -> dict[str, Any]:
        lsn, tx_id, state, ts, operations_json = row
        try:
            operations = json.loads(operations_json)
        except (json.JSONDecodeError, TypeError):
            operations = []
        return {
            "lsn": int(lsn),
            "tx_id": str(tx_id),
            "operations": operations,
            "timestamp": str(ts),
            "state": str(state),
        }

    def get_since(self, lsn: int) -> list[dict[str, Any]]:
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(
                    (
                        f"SELECT lsn, tx_id, state, ts, operations_json FROM {self._table_sql} "
                        "WHERE lsn > %s ORDER BY lsn ASC"
                    ),
                    (int(lsn),),
                )
                rows = list(cur.fetchall())
            return [self._row_to_entry(row) for row in rows]

    def get_committed(self) -> list[dict[str, Any]]:
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(
                    (
                        f"SELECT lsn, tx_id, state, ts, operations_json FROM {self._table_sql} "
                        "WHERE state IN (%s, %s) ORDER BY lsn ASC"
                    ),
                    (TxState.COMMITTED.value, TxState.SYNCED.value),
                )
                rows = list(cur.fetchall())
            return [self._row_to_entry(row) for row in rows]

    def compact(self, keep_last_n: int = 1000) -> int:
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(f"SELECT lsn FROM {self._table_sql} ORDER BY lsn ASC")
                lsns = [int(row[0]) for row in cur.fetchall()]
                if len(lsns) <= keep_last_n:
                    return 0
                threshold = lsns[-keep_last_n]
                cur.execute(f"DELETE FROM {self._table_sql} WHERE lsn < %s", (threshold,))
                removed = int(getattr(cur, "rowcount", 0))
                cur.execute(f"DELETE FROM {self._tx_cells_table_sql} WHERE lsn < %s", (threshold,))
            logger.info("MySQLWAL.compact: removed %d entries, kept %d", removed, keep_last_n)
            return removed

    def compact_before_lsn(self, lsn: int) -> int:
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self._table_sql} WHERE lsn < %s", (int(lsn),))
                removed = int(getattr(cur, "rowcount", 0))
                cur.execute(f"DELETE FROM {self._tx_cells_table_sql} WHERE lsn < %s", (int(lsn),))
            logger.info("MySQLWAL.compact_before_lsn: removed %d entries before lsn %d", removed, lsn)
            return removed

    def get_cell_history(self, cell_id: str) -> list[dict]:
        with self._lock:
            try:
                with self._conn.cursor() as cur:
                    cur.execute(
                        (
                            "SELECT DISTINCT w.lsn, w.tx_id, w.state, w.ts, w.operations_json "
                            f"FROM {self._table_sql} AS w "
                            f"INNER JOIN {self._tx_cells_table_sql} AS t ON t.lsn = w.lsn "
                            "WHERE t.cell_id = %s "
                            "ORDER BY w.lsn ASC"
                        ),
                        (cell_id,),
                    )
                    rows = list(cur.fetchall())
                # Fallback for legacy rows that may exist in WAL but not in trace table.
                if not rows:
                    return self._get_cell_history_full_scan(cell_id)
                return self._build_cell_history_from_rows(rows, cell_id)
            except Exception as exc:
                logger.warning("MySQLWAL trace-table history lookup failed for cell_id=%s: %s", cell_id, exc)
                return self._get_cell_history_full_scan(cell_id)

    def _build_cell_history_from_rows(self, rows: list[tuple[Any, ...]], cell_id: str) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        for row in rows:
            entry = self._row_to_entry(row)
            for op in entry.get("operations", []):
                if op.get("cell_id") == cell_id:
                    history.append(
                        {
                            "lsn": entry.get("lsn"),
                            "timestamp": entry.get("timestamp"),
                            "tx_id": entry.get("tx_id"),
                            "old_value": op.get("old_value"),
                            "new_value": op.get("new_value"),
                            "author_type": op.get("author_type"),
                            "author_id": op.get("author_id"),
                            "confidence": op.get("confidence"),
                        }
                    )
        return history

    def _get_cell_history_full_scan(self, cell_id: str) -> list[dict[str, Any]]:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT lsn, tx_id, state, ts, operations_json FROM {self._table_sql} ORDER BY lsn ASC")
            rows = list(cur.fetchall())
        return self._build_cell_history_from_rows(rows, cell_id)

    def get_metrics(self) -> dict[str, int]:
        with self._lock:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"SELECT COUNT(*), COALESCE(MAX(lsn), 0) FROM {self._table_sql}"
                )
                row = cur.fetchall()[0]
            return {
                "total_entries": int(row[0]),
                "last_lsn": int(row[1]),
            }


def create_default_wal() -> WriteAheadLog | RedisWriteAheadLog | MySQLWriteAheadLog:
    """Create WAL backend based on environment configuration.

    - HIVEFRAME_WAL_BACKEND=memory (default) -> WriteAheadLog()
    - HIVEFRAME_WAL_BACKEND=file -> WriteAheadLog(wal_path=...)
    - HIVEFRAME_WAL_BACKEND=redis -> RedisWriteAheadLog(...)
    - HIVEFRAME_WAL_BACKEND=mysql -> MySQLWriteAheadLog(...)
    """

    backend = os.getenv("HIVEFRAME_WAL_BACKEND", "memory").strip().lower()
    if backend == "memory":
        return WriteAheadLog()
    if backend == "file":
        wal_path = os.getenv("HIVEFRAME_WAL_PATH")
        return WriteAheadLog(wal_path=wal_path)
    if backend == "redis":
        redis_url = os.getenv("HIVEFRAME_REDIS_URL", "redis://127.0.0.1:6379/0")
        redis_prefix = os.getenv("HIVEFRAME_WAL_REDIS_PREFIX", "hiveframe:wal")
        return RedisWriteAheadLog(redis_url=redis_url, prefix=redis_prefix)
    if backend == "mysql":
        mysql_dsn = os.getenv("HIVEFRAME_MYSQL_DSN", "mysql://root@127.0.0.1:3306/hiveframe")
        mysql_database = os.getenv("HIVEFRAME_WAL_MYSQL_DATABASE") or os.getenv("HIVEFRAME_MYSQL_DATABASE")
        if mysql_database:
            mysql_dsn = MySQLWriteAheadLog.override_dsn_database(mysql_dsn, mysql_database)
        mysql_table = os.getenv("HIVEFRAME_WAL_MYSQL_TABLE") or os.getenv("HIVEFRAME_MYSQL_TABLE") or "hiveframe_wal"
        auto_create_db = os.getenv("HIVEFRAME_WAL_MYSQL_AUTO_CREATE_DB", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        return MySQLWriteAheadLog(
            mysql_dsn=mysql_dsn,
            table_name=mysql_table,
            auto_create_db=auto_create_db,
        )
    raise ValueError(
        f"Unknown HIVEFRAME_WAL_BACKEND='{backend}'. Use memory|file|redis|mysql."
    )

