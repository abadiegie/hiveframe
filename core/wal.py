# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""In-memory write-ahead log with thread-safe append operations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
import os
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


def create_default_wal() -> WriteAheadLog | RedisWriteAheadLog:
    """Create WAL backend based on environment configuration.

    - HIVEFRAME_WAL_BACKEND=memory (default) -> WriteAheadLog()
    - HIVEFRAME_WAL_BACKEND=file -> WriteAheadLog(wal_path=...)
    - HIVEFRAME_WAL_BACKEND=redis -> RedisWriteAheadLog(...)
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
    raise ValueError(
        f"Unknown HIVEFRAME_WAL_BACKEND='{backend}'. Use memory|file|redis."
    )

