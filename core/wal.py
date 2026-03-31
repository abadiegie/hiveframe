# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""In-memory write-ahead log with thread-safe append operations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
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
