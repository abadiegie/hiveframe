# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

from core.transaction import InvalidTransitionError, Operation, Transaction, TxState


def _sample_op() -> Operation:
    return Operation(
        cell_id="city_0",
        old_value="jakarta",
        new_value="DKI Jakarta",
        author_type="human",
        author_id="user",
    )


def test_valid_transitions() -> None:
    tx = Transaction(operations=[_sample_op()])
    tx.transition(TxState.VALIDATING)
    tx.transition(TxState.LOCKED)
    tx.transition(TxState.APPLYING)
    tx.transition(TxState.COMMITTED)
    tx.transition(TxState.SYNCING)
    tx.transition(TxState.SYNCED)
    assert tx.state == TxState.SYNCED
    assert len(tx.transitions) == 6


def test_invalid_transition_raises() -> None:
    tx = Transaction(operations=[_sample_op()])
    try:
        tx.transition(TxState.COMMITTED)
        assert False, "Expected InvalidTransitionError"
    except InvalidTransitionError:
        assert True


def test_transaction_to_dict() -> None:
    tx = Transaction(operations=[_sample_op()])
    tx.transition(TxState.VALIDATING)
    payload = tx.to_dict()
    assert payload["tx_id"]
    assert payload["state"] == TxState.VALIDATING.value
    assert payload["operations"][0]["cell_id"] == "city_0"
    assert "transitions" in payload
    assert "error" in payload
    assert payload["error"] is None
