# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Write node backed by mutable pandas DataFrame."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any
import logging

import pandas as pd

from .transaction import Transaction

DeltaCallback = Callable[[str, list[dict[str, Any]], int], None]


logger = logging.getLogger("core.write_node")


class WriteNode:
    """Thread-safe mutable store for transactional writes.

    ``bulk_mode`` remains a throughput hint, but apply() still preserves atomic rollback
    semantics on conflicts or exceptions.
    """

    # Shared pool: one thread is plenty for sequential delta callbacks; max_workers=2
    # gives headroom for bursts without spawning unbounded threads.
    _callback_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="wn-delta")

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
        """Apply transaction operations atomically with rollback on failure."""
        logger.debug("WriteNode.apply start tx_id=%s ops=%d bulk_mode=%s", transaction.tx_id, len(transaction.operations), self._bulk_mode)
        with self._lock:
            # Snapshot only the columns touched by this transaction — not the entire shared DataFrame.
            # Rolling back means restoring just those series, which is orders of magnitude cheaper
            # when many DFrame columns coexist in _df.
            touched_cols: set[str] = set()
            for op in transaction.operations:
                col, row_idx = self._parse_cell_id(op.cell_id)
                if row_idx >= 0:
                    touched_cols.add(col)
            col_snapshots: dict[str, Any] = {
                c: self._df[c].copy() for c in touched_cols if c in self._df.columns
            }
            orig_len = len(self._df.index)
            try:
                delta: list[dict[str, Any]] = []
                for op in transaction.operations:
                    col, row_idx = self._parse_cell_id(op.cell_id)
                    # Skip bulk operation markers — not real cell writes
                    if row_idx < 0:
                        continue
                    self._ensure_row(row_idx)
                    if col not in self._df.columns:
                        self._df[col] = [None] * len(self._df.index)
                    current_value = self._df.at[row_idx, col]
                    if not self._values_match(current_value, op.old_value):
                        logger.warning(
                            "WriteNode.apply optimistic conflict tx_id=%s cell=%s current=%r expected=%r",
                            transaction.tx_id,
                            op.cell_id,
                            current_value,
                            op.old_value,
                        )
                        self._rollback(col_snapshots, orig_len)
                        return False
                    try:
                        self._df.at[row_idx, col] = op.new_value
                    except (TypeError, ValueError) as _dtype_err:
                        # Mixed type: coerce column to object and retry
                        import pandas.errors as _pd_errs
                        if not isinstance(_dtype_err, _pd_errs.LossySetitemError) and str(self._df[col].dtype) == "object":
                            raise
                        self._df[col] = self._df[col].astype(object)
                        self._df.at[row_idx, col] = op.new_value
                    except Exception:
                        self._df[col] = self._df[col].astype(object)
                        self._df.at[row_idx, col] = op.new_value
                        # Mixed type: coerce column to object and retry
                        if str(self._df[col].dtype) != "object":
                            self._df[col] = self._df[col].astype(object)
                            self._df.at[row_idx, col] = op.new_value
                        else:
                            raise
                    delta.append(op.to_dict())
                self._version += 1
            except Exception as e:
                self._rollback(col_snapshots, orig_len)
                logger.exception("WriteNode.apply failed tx_id=%s error=%s", transaction.tx_id, e)
                return False

        if self._delta_callback is not None:
            logger.debug("WriteNode.apply dispatching delta callback tx_id=%s version=%d ops=%d", transaction.tx_id, self._version, len(delta))
            cb, tx_id, ver = self._delta_callback, transaction.tx_id, self._version
            WriteNode._callback_pool.submit(cb, tx_id, delta, ver)
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

    def to_parquet(self, path: str | Path) -> Path:
        """Persist current write-node state to a parquet file for MinIO snapshot support.

        Returns the path written to.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._df.to_parquet(target)
        logger.info("WriteNode.to_parquet path=%s rows=%d cols=%d", target, len(self._df.index), len(self._df.columns))
        return target

    def from_parquet(self, path: str | Path) -> None:
        """Restore write-node state from a parquet snapshot.

        Replaces the in-memory DataFrame entirely.
        """
        target = Path(path)
        if not target.exists():
            logger.info("WriteNode.from_parquet: snapshot not found at %s, skipping", target)
            return
        frame = pd.read_parquet(target)
        with self._lock:
            self._df = frame
            self._version += 1
        logger.info("WriteNode.from_parquet path=%s rows=%d cols=%d version=%d", target, len(frame.index), len(frame.columns), self._version)

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
        # Take a shallow reference copy inside the lock, then release immediately.
        # Holding _lock across yield would block all writes for the entire iteration.
        with self._lock:
            df = self._df if columns is None else self._df[columns]
            df = df.copy()  # snapshot: shallow copy is enough; columns are immutable strings
        total_rows = len(df)
        for start in range(0, total_rows, chunk_size):
            chunk = df.iloc[start : start + chunk_size].copy()
            logger.debug("WriteNode.read_fresh_lazy chunk start=%d size=%d", start, len(chunk))
            yield chunk

    def _rollback(self, col_snapshots: dict[str, Any], orig_len: int) -> None:
        # Trim any rows added during the failed transaction.
        if len(self._df.index) > orig_len:
            self._df = self._df.iloc[:orig_len]
        for col, series in col_snapshots.items():
            self._df[col] = series

    def _ensure_row(self, row_idx: int) -> None:
        target_len = row_idx + 1
        if len(self._df.index) < target_len:
            self._df = self._df.reindex(range(target_len))

    @staticmethod
    def _values_match(current_value: Any, expected_value: Any) -> bool:
        if pd.isna(current_value) and expected_value is None:
            return True
        if pd.isna(expected_value) and current_value is None:
            return True
        if pd.isna(current_value) and pd.isna(expected_value):
            return True
        return current_value == expected_value

    @staticmethod
    def _parse_cell_id(cell_id: str) -> tuple[str, int]:
        if cell_id.endswith("::__bulk_init__") or cell_id.endswith("::__bulk_chunk__"):
            return ("__bulk__", -1)
        if "_" not in cell_id:
            raise ValueError(f"Invalid cell_id format: {cell_id}")
        col, row = cell_id.rsplit("_", 1)
        if not col or not row.isdigit():
            raise ValueError(f"Invalid cell_id format: {cell_id}")
        return col, int(row)
