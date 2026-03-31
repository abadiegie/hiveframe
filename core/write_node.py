# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Write node backed by mutable pandas DataFrame."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock, Thread
from typing import Any
import logging

import pandas as pd

from .transaction import Transaction

DeltaCallback = Callable[[str, list[dict[str, Any]], int], None]


logger = logging.getLogger("core.write_node")


class WriteNode:
    """Thread-safe mutable store for transactional writes.

    If bulk_mode is True, apply() skips defensive copy and rollback for high-throughput batch ingest.
    This is now the default for all WriteNode instances.
    """

    def __init__(self, initial_data: dict[str, list[Any]] | None = None, bulk_mode: bool = True) -> None:
        self._lock = Lock()
        self._df = pd.DataFrame(initial_data or {})
        self._version = 0
        self._delta_callback: DeltaCallback | None = None
        self._bulk_mode = bulk_mode
        logger.info("WriteNode initialized columns=%s rows=%d bulk_mode=%s", list(self._df.columns), len(self._df.index), bulk_mode)

    def set_delta_callback(self, callback: DeltaCallback) -> None:
        """Register async/non-blocking delta callback for read node sync."""
        self._delta_callback = callback

    def apply(self, transaction: Transaction) -> bool:
        """Apply transaction operations atomically with rollback snapshot, unless bulk_mode is enabled.

        In bulk_mode, skips defensive copy and rollback for high-throughput batch ingest.
        """
        logger.debug("WriteNode.apply start tx_id=%s ops=%d bulk_mode=%s", transaction.tx_id, len(transaction.operations), self._bulk_mode)
        with self._lock:
            if self._bulk_mode or getattr(transaction, 'bulk_mode', False):
                # Bulk mode: skip defensive copy, apply all ops directly
                try:
                    delta: list[dict[str, Any]] = []
                    for op in transaction.operations:
                        col, row_idx = self._parse_cell_id(op.cell_id)
                        self._ensure_row(row_idx)
                        if col not in self._df.columns:
                            self._df[col] = [None] * len(self._df.index)
                        self._df.at[row_idx, col] = op.new_value
                        delta.append(op.to_dict())
                    self._version += 1
                except Exception as e:
                    logger.exception("WriteNode.apply (bulk) failed tx_id=%s error=%s", transaction.tx_id, e)
                    return False
            else:
                # Strict mode: defensive copy for rollback
                snapshot = self._df.copy(deep=True)
                try:
                    delta: list[dict[str, Any]] = []
                    for op in transaction.operations:
                        col, row_idx = self._parse_cell_id(op.cell_id)
                        self._ensure_row(row_idx)
                        if col not in self._df.columns:
                            self._df[col] = [None] * len(self._df.index)
                        self._df.at[row_idx, col] = op.new_value
                        delta.append(op.to_dict())
                    self._version += 1
                except Exception as e:
                    self._df = snapshot
                    logger.exception("WriteNode.apply failed tx_id=%s error=%s", transaction.tx_id, e)
                    return False

        if self._delta_callback is not None:
            logger.debug("WriteNode.apply dispatching delta callback tx_id=%s version=%d ops=%d", transaction.tx_id, self._version, len(delta))
            Thread(
                target=self._delta_callback,
                args=(transaction.tx_id, delta, self._version),
                daemon=True,
            ).start()
        logger.info("WriteNode.apply success tx_id=%s version=%d", transaction.tx_id, self._version)
        return True

    def get(self, cell_id: str) -> Any:
        """Read latest cell value directly from write node."""
        with self._lock:
            col, row_idx = self._parse_cell_id(cell_id)
            if col not in self._df.columns or row_idx >= len(self._df.index):
                logger.debug("WriteNode.get missing cell=%s", cell_id)
                return None
            val = self._df.at[row_idx, col]
            logger.debug("WriteNode.get cell=%s value=%s", cell_id, val)
            return val

    def snapshot(self) -> pd.DataFrame:
        """Return deep copy of current write dataframe."""
        with self._lock:
            logger.debug("WriteNode.snapshot called returning rows=%d cols=%d", len(self._df.index), len(self._df.columns))
            return self._df.copy(deep=True)

    def read_fresh_lazy(
        self, columns: list[str] | None = None, chunk_size: int = 1000
    ):
        """Yield DataFrame chunks lazily from the current write dataframe.

        Args:
            columns: Optional list of columns to include. If None, all columns are included.
            chunk_size: Number of rows per chunk (default: 1000).
        Yields:
            pandas.DataFrame: chunk of up to chunk_size rows.
        """
        with self._lock:
            df = self._df if columns is None else self._df[columns]
            total_rows = len(df)
            for start in range(0, total_rows, chunk_size):
                chunk = df.iloc[start : start + chunk_size].copy()
                logger.debug("WriteNode.read_fresh_lazy chunk start=%d size=%d", start, len(chunk))
                yield chunk

    def _ensure_row(self, row_idx: int) -> None:
        target_len = row_idx + 1
        if len(self._df.index) < target_len:
            self._df = self._df.reindex(range(target_len))

    @staticmethod
    def _parse_cell_id(cell_id: str) -> tuple[str, int]:
        if "_" not in cell_id:
            raise ValueError(f"Invalid cell_id format: {cell_id}")
        col, row = cell_id.rsplit("_", 1)
        if not col or not row.isdigit():
            raise ValueError(f"Invalid cell_id format: {cell_id}")
        return col, int(row)
