# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

from core.cluster_runtime import ClusterRuntime, RuntimeConfig
from core.coordinator import TransactionCoordinator
from core.transaction import Operation
from core.wal import RedisWriteAheadLog


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


def test_replay_applies_remote_commits(monkeypatch) -> None:
    async def run() -> None:
        monkeypatch.setenv("HIVEFRAME_WAL_REPLAY_ENABLED", "1")
        monkeypatch.setenv("HIVEFRAME_WAL_REPLAY_INTERVAL_S", "0.01")

        shared = RedisWriteAheadLog(
            redis_url="redis://unused",
            prefix="test:wal:replay",
            redis_client=_FakeRedis(),
        )
        writer = TransactionCoordinator(wal=shared)
        follower = TransactionCoordinator(wal=shared)

        await follower.start_wal_replay()
        try:
            tx = writer.submit(
                [
                    Operation(
                        cell_id="city_0",
                        old_value=None,
                        new_value="DKI Jakarta",
                        author_type="human",
                        author_id="user",
                    )
                ]
            )
            assert tx.error is None

            for _ in range(30):
                value = follower.read_fresh(["city_0"]).get("city_0")
                if value == "DKI Jakarta":
                    break
                await asyncio.sleep(0.01)

            assert follower.read_fresh(["city_0"]).get("city_0") == "DKI Jakarta"
        finally:
            await follower.stop_wal_replay()

    asyncio.run(run())


def test_replay_disabled_by_default_for_memory_wal(monkeypatch) -> None:
    async def run() -> None:
        monkeypatch.delenv("HIVEFRAME_WAL_REPLAY_ENABLED", raising=False)
        coordinator = TransactionCoordinator()
        await coordinator.start_wal_replay()
        # Memory WAL keeps replay disabled by default; no task should be started.
        assert coordinator._wal_replay_task is None

    asyncio.run(run())


def test_replay_cursor_checkpoint_persists_and_restores(monkeypatch, tmp_path) -> None:
    async def run() -> None:
        monkeypatch.setenv("HIVEFRAME_WAL_REPLAY_ENABLED", "1")
        monkeypatch.setenv("HIVEFRAME_WAL_REPLAY_INTERVAL_S", "0.01")

        cursor_path = tmp_path / "cursor.json"
        shared = RedisWriteAheadLog(
            redis_url="redis://unused",
            prefix="test:wal:cursor",
            redis_client=_FakeRedis(),
        )
        writer = TransactionCoordinator(wal=shared)
        follower = TransactionCoordinator(wal=shared)
        follower.set_wal_replay_cursor_path(str(cursor_path))

        await follower.start_wal_replay()
        try:
            writer.submit(
                [
                    Operation(
                        cell_id="city_0",
                        old_value=None,
                        new_value="DKI Jakarta",
                        author_type="human",
                        author_id="user",
                    )
                ]
            )
            for _ in range(30):
                if follower.read_fresh(["city_0"]).get("city_0") == "DKI Jakarta":
                    break
                await asyncio.sleep(0.01)
        finally:
            await follower.stop_wal_replay()

        assert cursor_path.exists()
        saved_lsn = int(json.loads(cursor_path.read_text())["last_lsn"])
        assert saved_lsn >= 1

        restored = TransactionCoordinator(wal=shared)
        restored.set_wal_replay_cursor_path(str(cursor_path))
        await restored.start_wal_replay()
        try:
            # No new writes; restored coordinator should keep cursor and not regress.
            await asyncio.sleep(0.03)
        finally:
            await restored.stop_wal_replay()

        assert restored._wal_replay_last_lsn >= saved_lsn

    asyncio.run(run())


def test_replay_cursor_save_every(monkeypatch, tmp_path) -> None:
    async def run() -> None:
        monkeypatch.setenv("HIVEFRAME_WAL_REPLAY_ENABLED", "1")
        monkeypatch.setenv("HIVEFRAME_WAL_REPLAY_INTERVAL_S", "0.01")
        monkeypatch.setenv("HIVEFRAME_WAL_REPLAY_SAVE_EVERY", "3")

        cursor_path = tmp_path / "cursor_every.json"
        shared = RedisWriteAheadLog(
            redis_url="redis://unused",
            prefix="test:wal:cursor_every",
            redis_client=_FakeRedis(),
        )
        follower = TransactionCoordinator(wal=shared)
        follower.set_wal_replay_cursor_path(str(cursor_path))

        follower._advance_wal_replay_cursor(1, persist=False)
        follower._advance_wal_replay_cursor(2, persist=False)
        assert not cursor_path.exists()

        follower._advance_wal_replay_cursor(3, persist=False)
        assert cursor_path.exists()
        assert int(json.loads(cursor_path.read_text())["last_lsn"]) == 3

    asyncio.run(run())


def test_runtime_sets_node_scoped_default_replay_cursor_path(monkeypatch) -> None:
    monkeypatch.setenv("HIVEFRAME_WAL_REPLAY_ENABLED", "1")
    monkeypatch.setenv("HIVEFRAME_WAL_BACKEND", "memory")

    runtime = ClusterRuntime(RuntimeConfig(node_id="node-a", role="write"))
    assert runtime.coordinator._wal_replay_cursor_path is not None
    assert str(runtime.coordinator._wal_replay_cursor_path).endswith("wal_replay_cursor_node-a.json")


