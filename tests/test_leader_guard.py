# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from core.coordinator import NotLeaderError, TransactionCoordinator
from core.transaction import Operation


def _op() -> Operation:
    return Operation(cell_id="x_0", old_value=None, new_value=1, author_type="human", author_id="u")


def test_follower_write_is_rejected_by_default() -> None:
    coordinator = TransactionCoordinator(node_id="worker", leader_node_id="leader")
    with pytest.raises(NotLeaderError):
        coordinator.submit([_op()])


def test_follower_write_is_allowed_with_explicit_worker_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HIVEFRAME_ALLOW_FOLLOWER_WRITES", "1")
    coordinator = TransactionCoordinator(node_id="worker", leader_node_id="leader")
    tx = coordinator.submit([_op()])
    assert tx.state.value in {"COMMITTED", "SYNCING", "SYNCED"}


def test_set_local_write_enabled_allows_follower_write_without_env() -> None:
    coordinator = TransactionCoordinator(node_id="worker", leader_node_id="leader")
    with pytest.raises(NotLeaderError):
        coordinator.submit([_op()])
    coordinator.set_local_write_enabled(True)
    tx = coordinator.submit([_op()])
    assert tx.state.value in {"COMMITTED", "SYNCING", "SYNCED"}


def test_set_local_write_enabled_can_be_reverted() -> None:
    coordinator = TransactionCoordinator(node_id="worker", leader_node_id="leader")
    coordinator.set_local_write_enabled(True)
    coordinator.set_local_write_enabled(False)
    with pytest.raises(NotLeaderError):
        coordinator.submit([_op()])
