# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

import json

from core.transaction import Operation, Transaction, TxState
from core.wal import RedisWriteAheadLog, create_default_wal


class _FakeRedis:
    def __init__(self) -> None:
        self.kv: dict[str, int] = {}
        self.zsets: dict[str, list[tuple[float, str]]] = {}

    def incr(self, key: str) -> int:
        current = int(self.kv.get(key, 0)) + 1
        self.kv[key] = current
        return current

    def zadd(self, key: str, mapping: dict[str, float]) -> int:
        bucket = self.zsets.setdefault(key, [])
        for member, score in mapping.items():
            bucket[:] = [(s, m) for s, m in bucket if m != member]
            bucket.append((float(score), member))
        bucket.sort(key=lambda item: (item[0], item[1]))
        return len(mapping)

    def zrangebyscore(self, key: str, min_score, max_score) -> list[str]:
        _ = max_score
        bucket = self.zsets.get(key, [])
        if isinstance(min_score, str) and min_score.startswith("("):
            lower = float(min_score[1:])
            return [m for s, m in bucket if s > lower]
        lower = float(min_score)
        return [m for s, m in bucket if s >= lower]

    def zrange(self, key: str, start: int, end: int) -> list[str]:
        bucket = self.zsets.get(key, [])
        members = [m for _s, m in bucket]
        if end == -1:
            return members[start:]
        return members[start : end + 1]

    def zcard(self, key: str) -> int:
        return len(self.zsets.get(key, []))

    def zremrangebyrank(self, key: str, start: int, end: int) -> int:
        bucket = self.zsets.get(key, [])
        if not bucket:
            return 0
        end_idx = end + 1
        removed = bucket[start:end_idx]
        remain = bucket[:start] + bucket[end_idx:]
        self.zsets[key] = remain
        return len(removed)

    def zremrangebyscore(self, key: str, min_score, max_score) -> int:
        bucket = self.zsets.get(key, [])
        low = float("-inf") if min_score == "-inf" else float(min_score)
        high = float(max_score)
        kept = [(s, m) for s, m in bucket if not (low <= s <= high)]
        removed = len(bucket) - len(kept)
        self.zsets[key] = kept
        return removed


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


def test_redis_wal_append_and_get_since() -> None:
    wal = RedisWriteAheadLog(redis_url="redis://unused", prefix="test:wal", redis_client=_FakeRedis())
    l1 = wal.append(_tx(1, TxState.COMMITTED))
    l2 = wal.append(_tx(2, TxState.SYNCED))

    assert l1 == 1
    assert l2 == 2

    entries = wal.get_since(1)
    assert len(entries) == 1
    assert entries[0]["lsn"] == 2
    assert wal.get_metrics() == {"total_entries": 2, "last_lsn": 2}


def test_redis_wal_get_committed_filters() -> None:
    wal = RedisWriteAheadLog(redis_url="redis://unused", prefix="test:wal", redis_client=_FakeRedis())
    wal.append(_tx(1, TxState.COMMITTED))
    wal.append(_tx(2, TxState.SYNCED))
    wal.append(_tx(3, TxState.FAILED))

    committed = wal.get_committed()
    assert len(committed) == 2
    assert all(entry["state"] in {"COMMITTED", "SYNCED"} for entry in committed)


def test_redis_wal_compaction() -> None:
    wal = RedisWriteAheadLog(redis_url="redis://unused", prefix="test:wal", redis_client=_FakeRedis())
    for i in range(5):
        wal.append(_tx(i + 1))

    removed = wal.compact(keep_last_n=2)
    assert removed == 3

    remaining = wal.get_since(0)
    lsns = [entry["lsn"] for entry in remaining]
    assert lsns == [4, 5]


def test_create_default_wal_from_env_redis(monkeypatch) -> None:
    class _FakeRedisModule:
        class Redis:
            @staticmethod
            def from_url(_url: str, decode_responses: bool = True):
                _ = decode_responses
                return _FakeRedis()


    monkeypatch.setenv("HIVEFRAME_WAL_BACKEND", "redis")
    monkeypatch.setenv("HIVEFRAME_REDIS_URL", "redis://unused")
    monkeypatch.setitem(__import__("sys").modules, "redis", _FakeRedisModule)

    wal = create_default_wal()
    assert isinstance(wal, RedisWriteAheadLog)

    lsn = wal.append(_tx(1))
    assert lsn == 1


def test_redis_wal_payload_is_json_roundtrip() -> None:
    wal = RedisWriteAheadLog(redis_url="redis://unused", prefix="test:wal", redis_client=_FakeRedis())
    wal.append(_tx(1))

    entries = wal.get_since(0)
    assert entries
    json.dumps(entries[0])

