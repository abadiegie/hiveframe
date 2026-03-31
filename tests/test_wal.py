# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor

from core.transaction import Operation, Transaction, TxState
from core.wal import WriteAheadLog


def _tx(i: int, state: TxState) -> Transaction:
    tx = Transaction(
        operations=[
            Operation(
                cell_id=f"c_{i}",
                old_value=None,
                new_value=i,
                author_type="human",
                author_id="user",
            )
        ]
    )
    tx.state = state
    return tx


def test_append_and_get_since() -> None:
    wal = WriteAheadLog()
    l1 = wal.append(_tx(1, TxState.COMMITTED))
    l2 = wal.append(_tx(2, TxState.SYNCED))
    assert l1 == 1
    assert l2 == 2
    entries = wal.get_since(1)
    assert len(entries) == 1
    assert entries[0]["lsn"] == 2


def test_thread_safety_concurrent_append() -> None:
    wal = WriteAheadLog()

    def append_one(i: int) -> int:
        return wal.append(_tx(i, TxState.COMMITTED))

    with ThreadPoolExecutor(max_workers=8) as ex:
        lsns = list(ex.map(append_one, range(50)))

    assert len(set(lsns)) == 50
    assert max(lsns) == 50


def test_get_committed_filters() -> None:
    wal = WriteAheadLog()
    wal.append(_tx(1, TxState.COMMITTED))
    wal.append(_tx(2, TxState.SYNCED))
    wal.append(_tx(3, TxState.FAILED))
    committed = wal.get_committed()
    assert len(committed) == 2
    assert all(item["state"] in {"COMMITTED", "SYNCED"} for item in committed)
