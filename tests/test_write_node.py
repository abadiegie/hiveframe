# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

from core.transaction import Operation, Transaction
from core.write_node import WriteNode


def _op(cell_id: str, old_value, new_value) -> Operation:
    return Operation(
        cell_id=cell_id,
        old_value=old_value,
        new_value=new_value,
        author_type="human",
        author_id="tester",
    )


def test_bulk_mode_conflict_rolls_back_entire_transaction() -> None:
    node = WriteNode(initial_data={"city": ["jakarta", "bandung"]}, bulk_mode=True)
    before = node.snapshot()
    version_before = node._version

    tx = Transaction(
        operations=[
            _op("city_0", "jakarta", "DKI Jakarta"),
            _op("city_1", "surabaya", "Jawa Barat"),
        ]
    )

    ok = node.apply(tx)

    assert ok is False
    assert node._version == version_before
    assert node.snapshot().equals(before)


def test_bulk_mode_exception_rolls_back_row_and_column_growth() -> None:
    node = WriteNode(initial_data={"city": ["jakarta"]}, bulk_mode=True)
    before = node.snapshot()
    version_before = node._version

    tx = Transaction(
        operations=[
            _op("city_0", "jakarta", "DKI Jakarta"),
            _op("badcell", None, "boom"),
        ]
    )

    ok = node.apply(tx)

    assert ok is False
    assert node._version == version_before
    assert node.snapshot().equals(before)

