# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Transaction coordinator orchestrating lock, apply, WAL, and sync."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
import os
import time
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

from .lock_manager import CellLockManager
from .read_node import ReadNode
from .transaction import InvalidTransitionError, Operation, Transaction, TxState
from .wal import WriteAheadLog, create_default_wal
from .write_node import WriteNode

if TYPE_CHECKING:
    from .replication import ReplicationManager

# Use a module-specific logger for clearer observability
logger = logging.getLogger("core.coordinator")


class NotLeaderError(Exception):
    """Raised when a non-leader node attempts a write in leader-required mode.

    Route the write to the leader node instead of this follower.
    """


class CapabilityError(Exception):
    """Raised when node capability is insufficient for the requested operation.

    Node must have capability='rw' to accept writes. Current capability is
    'ro' (read-only) or 'drain'.
    """


@dataclass(slots=True)
class CoordinatorStats:
    """Basic coordinator counters and timing metrics."""

    tx_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    total_commit_ms: float = 0.0
    replication_lag_ops: int = 0
    outbox_depth: int = 0
    conflict_count: int = 0
    full_resync_count: int = 0

    @property
    def avg_commit_ms(self) -> float:
        if self.success_count == 0:
            return 0.0
        return self.total_commit_ms / self.success_count


class TransactionCoordinator:
    """Coordinates transactional writes and read APIs across nodes."""

    def __init__(
        self,
        write_node: WriteNode | None = None,
        read_node: ReadNode | None = None,
        lock_manager: CellLockManager | None = None,
        wal: WriteAheadLog | None = None,
        replication_manager: "ReplicationManager | None" = None,
        node_id: str | None = None,
        is_leader: bool = False,
        leader_node_id: str | None = None,
    ) -> None:
        self.write_node = write_node or WriteNode()
        self.read_node = read_node or ReadNode()
        self.lock_manager = lock_manager or CellLockManager()
        self.wal = wal or create_default_wal()
        self.replication_manager = replication_manager
        self.stats = CoordinatorStats()
        self._stats_lock = Lock()
        self._node_id = node_id
        self._is_leader = is_leader
        self._leader_node_id = leader_node_id
        self.write_node.set_delta_callback(self.read_node.receive_delta)
        self._wal_replay_task: asyncio.Task[None] | None = None
        self._wal_replay_stop: asyncio.Event | None = None
        self._wal_replay_last_lsn = 0
        self._wal_replay_interval_s = max(0.01, float(os.getenv("HIVEFRAME_WAL_REPLAY_INTERVAL_S", "0.1")))
        self._wal_replay_cursor_path: Path | None = None
        cursor_path_raw = os.getenv("HIVEFRAME_WAL_REPLAY_CURSOR_PATH", "").strip()
        if cursor_path_raw:
            self._wal_replay_cursor_path = Path(cursor_path_raw)
        self._wal_replay_save_every = max(1, int(os.getenv("HIVEFRAME_WAL_REPLAY_SAVE_EVERY", "1")))
        self._wal_replay_pending_since_save = 0
        replay_env = os.getenv("HIVEFRAME_WAL_REPLAY_ENABLED")
        if replay_env is None:
            self._wal_replay_enabled = self.wal.__class__.__name__ in {
                "RedisWriteAheadLog",
                "MySQLWriteAheadLog",
            }
        else:
            self._wal_replay_enabled = replay_env.strip().lower() in {"1", "true", "yes", "on"}

        logger.info("TransactionCoordinator initialized write_node=%s read_node=%s", getattr(self.write_node, 'node_id', 'local'), getattr(self.read_node, 'node_id', 'local'))

    def submit(self, operations: list[Operation]) -> Transaction:
        """Execute full transaction lifecycle for a list of operations."""
        # Leader guard: if leader_node_id is configured and this is not the leader, reject writes.
        if (
            self._leader_node_id is not None
            and self._node_id is not None
            and self._node_id != self._leader_node_id
        ):
            raise NotLeaderError(
                f"Write rejected: this node ({self._node_id!r}) is not the leader "
                f"({self._leader_node_id!r}). Route writes to the leader node."
            )

        tx = Transaction(operations=operations)
        start = time.perf_counter()

        with self._stats_lock:
            self.stats.tx_count += 1

        logger.debug("Submitting transaction tx_id=%s ops=%d", tx.tx_id, len(operations))

        try:
            tx.transition(TxState.VALIDATING)
            cell_ids = [op.cell_id for op in operations]

            locked = self.lock_manager.acquire(tx.tx_id, cell_ids)
            if not locked:
                logger.warning("Lock conflict tx_id=%s", tx.tx_id)
                raise RuntimeError("Lock conflict")
            tx.transition(TxState.LOCKED)
            logger.debug("Transaction locked tx_id=%s", tx.tx_id)

            tx.transition(TxState.APPLYING)
            applied = self.write_node.apply(tx)
            if not applied:
                logger.error("Write apply failed tx_id=%s", tx.tx_id)
                raise RuntimeError("Write apply failed")

            tx.transition(TxState.COMMITTED)
            lsn = self.wal.append(tx)
            self._advance_wal_replay_cursor(int(lsn), persist=True)
            logger.info("Transaction committed tx_id=%s lsn=%s", tx.tx_id, lsn)
            tx.transition(TxState.SYNCING)

            # Fire-and-forget delta replication; never blocks commit path.
            if self.replication_manager is not None:
                logger.debug("Scheduling replication for tx_id=%s lsn=%s", tx.tx_id, lsn)
                self._fire_replication(lsn, tx)

            tx.transition(TxState.SYNCED)

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            with self._stats_lock:
                self.stats.success_count += 1
                self.stats.total_commit_ms += elapsed_ms
            logger.debug("Transaction committed complete tx_id=%s elapsed_ms=%.2f", tx.tx_id, elapsed_ms)
            return tx
        except (InvalidTransitionError, RuntimeError, ValueError) as exc:
            logger.exception("Transaction failed tx_id=%s error=%s", tx.tx_id, exc)
            tx.error = str(exc)
            try:
                tx.fail()
            except InvalidTransitionError:
                pass
            failed_lsn = self.wal.append(tx)
            self._advance_wal_replay_cursor(int(failed_lsn), persist=True)
            with self._stats_lock:
                self.stats.failed_count += 1
            return tx
        finally:
            self.lock_manager.release(tx.tx_id)

    def submit_non_transactional(self, operations: list[Operation]) -> bool:
        """Apply operations directly without coordinator locks/WAL bookkeeping.

        This path intentionally skips WAL append and replication_manager dispatch.
        Use it for simple/single-writer distributed mode where readers fetch data
        from writer snapshots directly instead of relying on read-replica sync.
        """
        if not operations:
            return True
        tx = Transaction(operations=operations)
        applied = self.write_node.apply(tx)
        if not applied:
            logger.error("Non-transactional apply failed tx_id=%s", tx.tx_id)
            with self._stats_lock:
                self.stats.failed_count += 1
            return False
        logger.debug("Non-transactional apply success tx_id=%s ops=%d", tx.tx_id, len(operations))
        return True

    def read(self, cell_ids: list[str]) -> dict[str, Any]:
        """Read potentially lagging values from read node cache."""
        return self.read_node.query(cell_ids)

    def read_fresh(self, cell_ids: list[str]) -> dict[str, Any]:
        """Read latest values directly from write node."""
        return {cell_id: self.write_node.get(cell_id) for cell_id in cell_ids}

    def get_stats(self) -> dict[str, float | int]:
        """Return coordinator metrics snapshot."""
        with self._stats_lock:
            return {
                "tx_count": self.stats.tx_count,
                "success_count": self.stats.success_count,
                "failed_count": self.stats.failed_count,
                "avg_commit_ms": self.stats.avg_commit_ms,
                "replication_lag_ops": self.stats.replication_lag_ops,
                "outbox_depth": self.stats.outbox_depth,
                "conflict_count": self.stats.conflict_count,
                "full_resync_count": self.stats.full_resync_count,
            }

    def set_runtime_metrics(
        self,
        *,
        replication_lag_ops: int,
        outbox_depth: int,
        conflict_count: int,
        full_resync_count: int,
    ) -> None:
        """Update runtime-derived counters exposed via coordinator stats."""
        with self._stats_lock:
            self.stats.replication_lag_ops = max(0, int(replication_lag_ops))
            self.stats.outbox_depth = max(0, int(outbox_depth))
            self.stats.conflict_count = max(0, int(conflict_count))
            self.stats.full_resync_count = max(0, int(full_resync_count))

    def _fire_replication(self, lsn: int, tx: Transaction) -> None:
        """Schedule replication in background without blocking caller thread."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self.replication_manager.replicate_tx(lsn, tx.to_dict())
            )
            logger.debug("Replication task created for lsn=%s", lsn)
        except RuntimeError:
            # No running event loop (sync context); best-effort via new thread loop.
            logger.debug("No running loop to schedule replication for lsn=%s; skipping background schedule", lsn)
            pass

    async def start_wal_replay(self) -> None:
        """Start background replay from shared WAL into local materialized state."""
        if not self._wal_replay_enabled:
            return
        if self._wal_replay_task is not None and not self._wal_replay_task.done():
            return
        self._load_wal_replay_checkpoint()
        self._wal_replay_stop = asyncio.Event()
        self._wal_replay_task = asyncio.create_task(self._wal_replay_loop())
        logger.info("WAL replay loop started interval=%.3fs", self._wal_replay_interval_s)

    async def stop_wal_replay(self) -> None:
        """Stop background WAL replay task if running."""
        if self._wal_replay_stop is not None:
            self._wal_replay_stop.set()
        if self._wal_replay_task is not None:
            await self._wal_replay_task
        self._save_wal_replay_checkpoint(force=True)
        self._wal_replay_task = None
        self._wal_replay_stop = None

    async def _wal_replay_loop(self) -> None:
        while self._wal_replay_stop is not None and not self._wal_replay_stop.is_set():
            try:
                entries = list(self.wal.get_since(self._wal_replay_last_lsn))
                entries.sort(key=lambda item: int(item.get("lsn", 0)))
                for entry in entries:
                    lsn = int(entry.get("lsn", 0))
                    if lsn <= self._wal_replay_last_lsn:
                        continue
                    state = str(entry.get("state", ""))
                    if state in {TxState.COMMITTED.value, TxState.SYNCED.value}:
                        self._apply_wal_entry(entry)
                    self._advance_wal_replay_cursor(lsn, persist=False)
            except Exception as exc:
                logger.warning("WAL replay iteration failed: %s", exc)

            try:
                stop_event = self._wal_replay_stop
                if stop_event is None:
                    break
                await asyncio.wait_for(stop_event.wait(), timeout=self._wal_replay_interval_s)
            except asyncio.TimeoutError:
                continue

    def set_wal_replay_cursor_path(self, path: str | None) -> None:
        """Override replay cursor checkpoint path for this coordinator instance."""
        self._wal_replay_cursor_path = Path(path) if path else None

    def _advance_wal_replay_cursor(self, lsn: int, *, persist: bool) -> None:
        if lsn <= self._wal_replay_last_lsn:
            return
        self._wal_replay_last_lsn = lsn
        self._wal_replay_pending_since_save += 1
        self._save_wal_replay_checkpoint(force=persist)

    def _load_wal_replay_checkpoint(self) -> None:
        if self._wal_replay_cursor_path is None:
            return
        try:
            if not self._wal_replay_cursor_path.exists():
                return
            payload = json.loads(self._wal_replay_cursor_path.read_text())
            loaded_lsn = int(payload.get("last_lsn", 0))
            self._wal_replay_last_lsn = max(self._wal_replay_last_lsn, loaded_lsn)
            logger.info(
                "Loaded WAL replay cursor last_lsn=%d from %s",
                self._wal_replay_last_lsn,
                self._wal_replay_cursor_path,
            )
        except Exception as exc:
            logger.warning("Failed to load WAL replay cursor from %s: %s", self._wal_replay_cursor_path, exc)

    def _save_wal_replay_checkpoint(self, *, force: bool = False) -> None:
        if self._wal_replay_cursor_path is None:
            return
        if not force and self._wal_replay_pending_since_save < self._wal_replay_save_every:
            return
        try:
            self._wal_replay_cursor_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._wal_replay_cursor_path.with_suffix(self._wal_replay_cursor_path.suffix + ".tmp")
            tmp_path.write_text(json.dumps({"last_lsn": self._wal_replay_last_lsn}, sort_keys=True))
            tmp_path.replace(self._wal_replay_cursor_path)
            self._wal_replay_pending_since_save = 0
        except Exception as exc:
            logger.warning("Failed to save WAL replay cursor to %s: %s", self._wal_replay_cursor_path, exc)

    def _apply_wal_entry(self, entry: dict[str, Any]) -> None:
        raw_ops = list(entry.get("operations", []))
        if not raw_ops:
            return

        ops: list[Operation] = []
        for raw in raw_ops:
            cell_id = str(raw.get("cell_id", ""))
            if not cell_id:
                continue
            current = self.write_node.get(cell_id)
            next_value = raw.get("new_value")
            # Idempotence: skip no-op writes to avoid unnecessary version churn.
            if current == next_value:
                continue
            ops.append(
                Operation(
                    cell_id=cell_id,
                    old_value=current,
                    new_value=next_value,
                    author_type=str(raw.get("author_type", "replica")),
                    author_id=raw.get("author_id"),
                    confidence=raw.get("confidence"),
                )
            )

        if not ops:
            return

        tx = Transaction(operations=ops, tx_id=str(entry.get("tx_id", "replay")))
        applied = self.write_node.apply(tx)
        if not applied:
            logger.warning("WAL replay apply failed tx_id=%s ops=%d", tx.tx_id, len(ops))

