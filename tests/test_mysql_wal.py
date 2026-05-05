# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

import json

from core.transaction import Operation, Transaction, TxState
from core.wal import MySQLWriteAheadLog, create_default_wal


class _FakeMySQLCursor:
    def __init__(self, conn: "_FakeMySQLConn") -> None:
        self._conn = conn
        self.lastrowid = 0
        self.rowcount = 0
        self._result: list[tuple] = []

    def __enter__(self) -> "_FakeMySQLCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _ = (exc_type, exc, tb)
        return False

    def execute(self, sql: str, params=None) -> int:
        self._conn._executed_sql.append(sql)
        normalized = " ".join(sql.strip().split()).upper()
        self._result = []
        self.rowcount = 0

        if normalized.startswith("CREATE DATABASE IF NOT EXISTS"):
            return 0

        if normalized.startswith("CREATE TABLE"):
            return 0

        if normalized.startswith("INSERT INTO"):
            tx_id, state, ts, operations_json = params
            self._conn._next_lsn += 1
            lsn = self._conn._next_lsn
            self._conn._rows.append(
                {
                    "lsn": lsn,
                    "tx_id": tx_id,
                    "state": state,
                    "ts": ts,
                    "operations_json": operations_json,
                }
            )
            self.lastrowid = lsn
            self.rowcount = 1
            return 1

        if "WHERE LSN > %S" in normalized:
            threshold = int(params[0])
            rows = [r for r in self._conn._rows if int(r["lsn"]) > threshold]
            self._result = [
                (r["lsn"], r["tx_id"], r["state"], r["ts"], r["operations_json"])
                for r in sorted(rows, key=lambda item: int(item["lsn"]))
            ]
            return len(self._result)

        if "WHERE STATE IN (%S, %S)" in normalized:
            allowed = {str(params[0]), str(params[1])}
            rows = [r for r in self._conn._rows if str(r["state"]) in allowed]
            self._result = [
                (r["lsn"], r["tx_id"], r["state"], r["ts"], r["operations_json"])
                for r in sorted(rows, key=lambda item: int(item["lsn"]))
            ]
            return len(self._result)

        if normalized.startswith("SELECT LSN FROM"):
            self._result = [(r["lsn"],) for r in sorted(self._conn._rows, key=lambda item: int(item["lsn"]))]
            return len(self._result)

        if normalized.startswith("SELECT LSN, TX_ID, STATE, TS, OPERATIONS_JSON FROM"):
            self._result = [
                (r["lsn"], r["tx_id"], r["state"], r["ts"], r["operations_json"])
                for r in sorted(self._conn._rows, key=lambda item: int(item["lsn"]))
            ]
            return len(self._result)

        if normalized.startswith("DELETE FROM") and "WHERE LSN < %S" in normalized:
            threshold = int(params[0])
            before = len(self._conn._rows)
            self._conn._rows = [r for r in self._conn._rows if int(r["lsn"]) >= threshold]
            removed = before - len(self._conn._rows)
            self.rowcount = removed
            return removed

        raise AssertionError(f"Unsupported SQL in fake mysql cursor: {sql}")

    def fetchall(self) -> list[tuple]:
        return list(self._result)


class _FakeMySQLConn:
    def __init__(self) -> None:
        self._rows: list[dict] = []
        self._next_lsn = 0
        self.closed = False
        self._executed_sql: list[str] = []

    def cursor(self) -> _FakeMySQLCursor:
        return _FakeMySQLCursor(self)

    def close(self) -> None:
        self.closed = True


def _tx(i: int, state: TxState = TxState.COMMITTED) -> Transaction:
    tx = Transaction(
        operations=[
            Operation(
                cell_id=f"c_{i}",
                old_value=None,
                new_value=i,
                author_type="human",
                author_id="u",
            )
        ]
    )
    tx.state = state
    return tx


def test_mysql_wal_append_and_get_since() -> None:
    wal = MySQLWriteAheadLog(mysql_dsn="mysql://u:p@localhost:3306/db", mysql_conn=_FakeMySQLConn())
    l1 = wal.append(_tx(1, TxState.COMMITTED))
    l2 = wal.append(_tx(2, TxState.SYNCED))

    assert l1 == 1
    assert l2 == 2

    entries = wal.get_since(1)
    assert len(entries) == 1
    assert entries[0]["lsn"] == 2


def test_mysql_wal_get_committed_filters() -> None:
    wal = MySQLWriteAheadLog(mysql_dsn="mysql://u:p@localhost:3306/db", mysql_conn=_FakeMySQLConn())
    wal.append(_tx(1, TxState.COMMITTED))
    wal.append(_tx(2, TxState.SYNCED))
    wal.append(_tx(3, TxState.FAILED))

    committed = wal.get_committed()
    assert len(committed) == 2
    assert all(entry["state"] in {"COMMITTED", "SYNCED"} for entry in committed)


def test_mysql_wal_compaction_and_history() -> None:
    wal = MySQLWriteAheadLog(mysql_dsn="mysql://u:p@localhost:3306/db", mysql_conn=_FakeMySQLConn())
    for i in range(5):
        wal.append(_tx(i + 1))

    removed = wal.compact(keep_last_n=2)
    assert removed == 3

    remaining = wal.get_since(0)
    assert [entry["lsn"] for entry in remaining] == [4, 5]

    history = wal.get_cell_history("c_5")
    assert len(history) == 1
    assert history[0]["new_value"] == 5


def test_create_default_wal_from_env_mysql(monkeypatch) -> None:
    class _FakePyMySQLModule:
        calls: list[dict] = []

        @staticmethod
        def connect(**kwargs):
            _FakePyMySQLModule.calls.append(dict(kwargs))
            return _FakeMySQLConn()

    import sys

    monkeypatch.setenv("HIVEFRAME_WAL_BACKEND", "mysql")
    monkeypatch.setenv("HIVEFRAME_MYSQL_DSN", "mysql://u:p@localhost:3306/db")
    monkeypatch.setitem(sys.modules, "pymysql", _FakePyMySQLModule)

    wal = create_default_wal()
    assert isinstance(wal, MySQLWriteAheadLog)
    assert len(_FakePyMySQLModule.calls) == 1
    assert _FakePyMySQLModule.calls[0]["database"] == "db"

    lsn = wal.append(_tx(1))
    assert lsn == 1


def test_create_default_wal_from_env_mysql_auto_create_db(monkeypatch) -> None:
    class _FakePyMySQLModule:
        calls: list[dict] = []

        @staticmethod
        def connect(**kwargs):
            _FakePyMySQLModule.calls.append(dict(kwargs))
            return _FakeMySQLConn()

    import sys

    monkeypatch.setenv("HIVEFRAME_WAL_BACKEND", "mysql")
    monkeypatch.setenv("HIVEFRAME_MYSQL_DSN", "mysql://u:p@localhost:3306/db")
    monkeypatch.setenv("HIVEFRAME_WAL_MYSQL_AUTO_CREATE_DB", "1")
    monkeypatch.setitem(sys.modules, "pymysql", _FakePyMySQLModule)

    wal = create_default_wal()
    assert isinstance(wal, MySQLWriteAheadLog)

    assert len(_FakePyMySQLModule.calls) == 2
    assert "database" not in _FakePyMySQLModule.calls[0]
    assert _FakePyMySQLModule.calls[1]["database"] == "db"

    lsn = wal.append(_tx(1))
    assert lsn == 1


def test_create_default_wal_from_env_mysql_database_override(monkeypatch) -> None:
    class _FakePyMySQLModule:
        calls: list[dict] = []

        @staticmethod
        def connect(**kwargs):
            _FakePyMySQLModule.calls.append(dict(kwargs))
            return _FakeMySQLConn()

    import sys

    monkeypatch.setenv("HIVEFRAME_WAL_BACKEND", "mysql")
    monkeypatch.setenv("HIVEFRAME_MYSQL_DSN", "mysql://u:p@localhost:3306/default_db")
    monkeypatch.setenv("HIVEFRAME_WAL_MYSQL_DATABASE", "override_db")
    monkeypatch.setitem(sys.modules, "pymysql", _FakePyMySQLModule)

    wal = create_default_wal()
    assert isinstance(wal, MySQLWriteAheadLog)
    assert len(_FakePyMySQLModule.calls) == 1
    assert _FakePyMySQLModule.calls[0]["database"] == "override_db"


def test_create_default_wal_from_env_mysql_table_override(monkeypatch) -> None:
    class _FakePyMySQLModule:
        connections: list[_FakeMySQLConn] = []

        @staticmethod
        def connect(**kwargs):
            _ = kwargs
            conn = _FakeMySQLConn()
            _FakePyMySQLModule.connections.append(conn)
            return conn

    import sys

    monkeypatch.setenv("HIVEFRAME_WAL_BACKEND", "mysql")
    monkeypatch.setenv("HIVEFRAME_MYSQL_DSN", "mysql://u:p@localhost:3306/db")
    monkeypatch.setenv("HIVEFRAME_MYSQL_TABLE", "custom_wal_table")
    monkeypatch.setitem(sys.modules, "pymysql", _FakePyMySQLModule)

    wal = create_default_wal()
    assert isinstance(wal, MySQLWriteAheadLog)
    create_table_sql = _FakePyMySQLModule.connections[0]._executed_sql[0]
    assert "CREATE TABLE IF NOT EXISTS custom_wal_table" in create_table_sql


def test_mysql_wal_payload_is_json_roundtrip() -> None:
    wal = MySQLWriteAheadLog(mysql_dsn="mysql://u:p@localhost:3306/db", mysql_conn=_FakeMySQLConn())
    wal.append(_tx(1))

    entries = wal.get_since(0)
    assert entries
    json.dumps(entries[0])

