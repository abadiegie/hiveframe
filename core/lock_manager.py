# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Cell-level lock manager with all-or-nothing acquisition semantics."""

from __future__ import annotations

from threading import Lock
from typing import Any


class CellLockManager:
    """Thread-safe in-memory lock manager for cell IDs."""

    def __init__(self, lock_timeout_ms: int = 5000) -> None:
        self._guard = Lock()
        self._cell_to_tx: dict[str, str] = {}
        self._tx_to_cells: dict[str, set[str]] = {}
        self.lock_timeout_ms = lock_timeout_ms

    def acquire(self, tx_id: str, cell_ids: list[str]) -> bool:
        """Acquire all requested cells atomically, or fail without partial locks."""
        unique_cells = sorted(set(cell_ids))
        with self._guard:
            if any(cell in self._cell_to_tx and self._cell_to_tx[cell] != tx_id for cell in unique_cells):
                return False
            for cell in unique_cells:
                self._cell_to_tx[cell] = tx_id
            self._tx_to_cells.setdefault(tx_id, set()).update(unique_cells)
            return True

    def release(self, tx_id: str) -> None:
        """Release all cells held by a transaction."""
        with self._guard:
            cells = self._tx_to_cells.pop(tx_id, set())
            for cell in cells:
                owner = self._cell_to_tx.get(cell)
                if owner == tx_id:
                    del self._cell_to_tx[cell]

    def get_locked_cells(self) -> dict[str, Any]:
        """Return a copy of current lock ownership map."""
        with self._guard:
            return dict(self._cell_to_tx)
