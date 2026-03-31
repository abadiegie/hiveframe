# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Read node backed by pandas cache and parquet persistence.

This implementation avoids the polars <-> pandas conversion churn by keeping
an internal pandas.DataFrame as the analytical cache. It implements a simple
SQL path via sqlite for compatibility when SQL queries are requested.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
import threading
from typing import Any
import time
import sqlite3

import pandas as pd


class ReadNode:
    """Thread-safe analytical read store using pandas and parquet persistence."""

    def __init__(self, sync_delay_ms: int = 50, storage_dir: str = ".dframe_store") -> None:
        self._lock = Lock()
        self.sync_delay_ms = sync_delay_ms
        self._version = 0
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache = pd.DataFrame()
        # Buffering for incoming deltas to apply in batches
        self._delta_buffer: list[dict[str, Any]] = []
        self._buffer_lock = threading.Lock()
        self._buffer_timer: threading.Timer | None = None
        # Tuning parameters: flush when buffer reaches batch_size or after flush_interval_ms
        self._buffer_batch_size = 500  # default flush threshold
        self._buffer_flush_interval_ms = 50  # flush interval in milliseconds

    @staticmethod
    def _to_pandas_safe(frame: pd.DataFrame) -> pd.DataFrame:
        """Identity conversion for pandas frames kept for API compatibility."""
        return frame

    def receive_delta(self, tx_id: str, delta: list[dict[str, Any]], version: int) -> None:
        """Apply write-node delta into pandas cache with configured delay.

        This applies per-operation updates directly into the pandas DataFrame and
        avoids full-frame conversions.
        """
        # Simulate configured sync lag
        _ = tx_id
        time.sleep(self.sync_delay_ms / 1000.0)

        # Buffer incoming ops (flatten delta list) and schedule flush
        with self._buffer_lock:
            for op in delta:
                # keep op dict together and annotate with version
                copy_op = dict(op)
                copy_op["_version"] = version
                self._delta_buffer.append(copy_op)
            # If buffer reached threshold, flush synchronously
            if len(self._delta_buffer) >= self._buffer_batch_size:
                # flush under buffer_lock to avoid race with timer
                self._flush_buffer_locked()
                return
            # otherwise ensure a timer is scheduled to flush shortly
            if self._buffer_timer is None:
                def _timer_cb():
                    try:
                        self._flush_buffer()
                    except Exception:
                        pass

                self._buffer_timer = threading.Timer(self._buffer_flush_interval_ms / 1000.0, _timer_cb)
                self._buffer_timer.daemon = True
                self._buffer_timer.start()

    def query(self, cell_ids: list[str]) -> dict[str, Any]:
        """Read selected cells from pandas cache."""
        with self._lock:
            result: dict[str, Any] = {}
            pdf = self._cache
            for cell_id in cell_ids:
                col, row_idx = self._parse_cell_id(cell_id)
                if col in pdf.columns and row_idx < len(pdf.index):
                    result[cell_id] = pdf.at[row_idx, col]
                else:
                    result[cell_id] = None
            return result

    def query_sql(self, sql_string: str) -> pd.DataFrame:
        """Execute SQL against current cache using an in-memory sqlite table.

        This avoids adding a new dependency while providing full SQL expressiveness.
        """
        with self._lock:
            # create in-memory sqlite DB and load DataFrame as a table
            conn = sqlite3.connect(":memory:")
            try:
                self._cache.to_sql("dframe", conn, index=False, if_exists="replace")
                df = pd.read_sql_query(sql_string, conn)
                return df
            finally:
                conn.close()

    def get_dataframe(self) -> pd.DataFrame:
        """Return current cache for higher-level wrappers."""
        with self._lock:
            return self._cache.copy(deep=True)

    def persist(self, name: str) -> Path:
        """Persist cache to parquet and return path."""
        path = self._storage_dir / f"{name}.parquet"
        with self._lock:
            # pandas will choose an available engine (pyarrow/fastparquet) if installed
            self._cache.to_parquet(path)
        return path

    def load(self, name: str) -> pd.DataFrame:
        """Load parquet dataset into cache and return it."""
        path = self._storage_dir / f"{name}.parquet"
        frame = pd.read_parquet(path)
        with self._lock:
            self._cache = frame
        return frame

    @property
    def version(self) -> int:
        """Current sync version."""
        with self._lock:
            return self._version

    @staticmethod
    def _parse_cell_id(cell_id: str) -> tuple[str, int]:
        if "_" not in cell_id:
            raise ValueError(f"Invalid cell_id format: {cell_id}")
        col, row = cell_id.rsplit("_", 1)
        if not col or not row.isdigit():
            raise ValueError(f"Invalid cell_id format: {cell_id}")
        return col, int(row)

    def _flush_buffer(self) -> None:
        """Thread-safe flush callback wrapper: acquire buffer lock and flush.

        This function is safe to be called by the timer thread.
        """
        with self._buffer_lock:
            self._flush_buffer_locked()

    def _flush_buffer_locked(self) -> None:
        """Assumes _buffer_lock is held. Move buffered ops to local variable and apply in bulk."""
        if self._buffer_timer is not None:
            try:
                self._buffer_timer.cancel()
            except Exception:
                pass
            self._buffer_timer = None

        if not self._delta_buffer:
            return

        # Swap buffer
        ops = self._delta_buffer
        self._delta_buffer = []

        # Build aggregated col_updates and compute max version
        col_updates: dict[str, dict[int, Any]] = {}
        max_version = self._version
        max_row_idx = -1
        for op in ops:
            version = op.pop("_version", None)
            if version is not None and version > max_version:
                max_version = version
            col, row_idx = self._parse_cell_id(op["cell_id"])
            col_updates.setdefault(col, {})[row_idx] = op.get("new_value")
            if row_idx > max_row_idx:
                max_row_idx = row_idx

        # Apply updates in the main cache under main lock
        with self._lock:
            pdf = self._cache
            target_len = max_row_idx + 1
            if target_len > len(pdf.index):
                pdf = pdf.reindex(range(target_len))
            for col in col_updates.keys():
                if col not in pdf.columns:
                    pdf[col] = pd.Series([None] * len(pdf.index), dtype="object")
            updates_df = pd.DataFrame({col: pd.Series(mapping) for col, mapping in col_updates.items()})
            updates_df = updates_df.reindex(range(len(pdf.index)))
            pdf.update(updates_df)
            self._cache = pdf
            self._version = max(self._version, max_version)
