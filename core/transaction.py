# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Transaction models and state machine for write operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import time
import uuid


class InvalidTransitionError(RuntimeError):
    """Raised when a transaction state transition is invalid."""


class TxState(str, Enum):
    """Lifecycle states for a transaction."""

    PENDING = "PENDING"
    VALIDATING = "VALIDATING"
    LOCKED = "LOCKED"
    APPLYING = "APPLYING"
    COMMITTED = "COMMITTED"
    SYNCING = "SYNCING"
    SYNCED = "SYNCED"
    FAILED = "FAILED"
    ROLLED_BACK = "ROLLED_BACK"


_ALLOWED_TRANSITIONS: dict[TxState, set[TxState]] = {
    TxState.PENDING: {TxState.VALIDATING, TxState.FAILED},
    TxState.VALIDATING: {TxState.LOCKED, TxState.FAILED},
    TxState.LOCKED: {TxState.APPLYING, TxState.FAILED},
    TxState.APPLYING: {TxState.COMMITTED, TxState.FAILED},
    TxState.COMMITTED: {TxState.SYNCING, TxState.FAILED},
    TxState.SYNCING: {TxState.SYNCED, TxState.FAILED},
    TxState.SYNCED: {TxState.FAILED},
    TxState.FAILED: {TxState.ROLLED_BACK},
    TxState.ROLLED_BACK: set(),
}


@dataclass(slots=True)
class Operation:
    """Single-cell mutation payload."""

    cell_id: str
    old_value: Any
    new_value: Any
    author_type: str
    author_id: str | None = None
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize operation for WAL/JSON output."""
        return {
            "cell_id": self.cell_id,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "author_type": self.author_type,
            "author_id": self.author_id,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class TransitionRecord:
    """Transition audit record with timing metadata."""

    from_state: TxState
    to_state: TxState
    timestamp: str
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize transition record."""
        return {
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "timestamp": self.timestamp,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass(slots=True)
class Transaction:
    """Atomic collection of operations with validated state machine."""

    operations: list[Operation]
    tx_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: TxState = TxState.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    transitions: list[TransitionRecord] = field(default_factory=list)
    error: str | None = None
    _last_transition_perf: float = field(default_factory=time.perf_counter)

    def transition(self, next_state: TxState) -> None:
        """Move to next state if transition is valid."""
        allowed = _ALLOWED_TRANSITIONS[self.state]
        if next_state not in allowed:
            raise InvalidTransitionError(
                f"Invalid transition from {self.state.value} to {next_state.value}"
            )

        now_perf = time.perf_counter()
        elapsed_ms = (now_perf - self._last_transition_perf) * 1000.0
        record = TransitionRecord(
            from_state=self.state,
            to_state=next_state,
            timestamp=datetime.now(timezone.utc).isoformat(),
            elapsed_ms=elapsed_ms,
        )
        self.transitions.append(record)
        self.state = next_state
        self._last_transition_perf = now_perf

    def fail(self) -> None:
        """Best-effort failure path: FAILED, then ROLLED_BACK when valid."""
        if self.state != TxState.FAILED:
            try:
                self.transition(TxState.FAILED)
            except InvalidTransitionError:
                self.state = TxState.FAILED
                return
        try:
            self.transition(TxState.ROLLED_BACK)
        except InvalidTransitionError:
            pass

    def to_dict(self) -> dict[str, Any]:
        """Serialize transaction for WAL/JSON storage."""
        return {
            "tx_id": self.tx_id,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "operations": [op.to_dict() for op in self.operations],
            "transitions": [item.to_dict() for item in self.transitions],
            "error": self.error,
        }
