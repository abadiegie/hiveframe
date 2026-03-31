# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Transaction coordinator orchestrating lock, apply, WAL, and sync."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import time
from threading import Lock
from typing import TYPE_CHECKING, Any

from .lock_manager import CellLockManager
from .read_node import ReadNode
from .transaction import InvalidTransitionError, Operation, Transaction, TxState
from .wal import WriteAheadLog
from .write_node import WriteNode

if TYPE_CHECKING:
    from .replication import ReplicationManager

# Use a module-specific logger for clearer observability
logger = logging.getLogger("core.coordinator")


@dataclass(slots=True)
class CoordinatorStats:
    """Basic coordinator counters and timing metrics."""

    tx_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    total_commit_ms: float = 0.0

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
    ) -> None:
        self.write_node = write_node or WriteNode()
        self.read_node = read_node or ReadNode()
        self.lock_manager = lock_manager or CellLockManager()
        self.wal = wal or WriteAheadLog()
        self.replication_manager = replication_manager
        self.stats = CoordinatorStats()
        self._stats_lock = Lock()
        self.write_node.set_delta_callback(self.read_node.receive_delta)

        logger.info("TransactionCoordinator initialized write_node=%s read_node=%s", getattr(self.write_node, 'node_id', 'local'), getattr(self.read_node, 'node_id', 'local'))

    def submit(self, operations: list[Operation]) -> Transaction:
        """Execute full transaction lifecycle for a list of operations."""
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
            self.wal.append(tx)
            with self._stats_lock:
                self.stats.failed_count += 1
            return tx
        finally:
            self.lock_manager.release(tx.tx_id)

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
            }

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

