# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time

import pytest

from core.dataframe import DFrame


def test_setitem_getitem_groupby() -> None:
    df = DFrame({"city": ["jakarta", "bandung"], "score": [10, 20]})
    df["city"] = ["DKI Jakarta", "Jawa Barat"]
    time.sleep(0.1)
    city = df["city"]
    assert city.iloc[0] == "DKI Jakarta"
    grouped = df.groupby("city")["score"].sum()
    assert grouped.loc["DKI Jakarta"] == 10


def test_write_goes_through_transaction() -> None:
    df = DFrame({"a": [1]})
    before = df._coordinator.get_stats()["tx_count"]
    df["a"] = [2]
    after = df._coordinator.get_stats()["tx_count"]
    assert after == before + 1


def test_read_node_lag_and_read_fresh() -> None:
    df = DFrame({"name": ["Alice"]})
    df["name"] = ["Alicia"]
    fresh = df.read_fresh()
    assert fresh.at[0, "name"] == "Alicia"


def test_to_persistent_and_reload() -> None:
    df = DFrame({"x": [1, 2]})
    time.sleep(0.1)
    path = df.to_persistent("test_frame")
    assert path.exists()


def test_getitem_falls_back_to_fresh_when_read_cache_stale() -> None:
    df = DFrame({"city": ["jakarta", "bandung"]})
    # Simulate a stale read cache by increasing sync delay after seed.
    df._coordinator.read_node.sync_delay_ms = 300
    df["city"] = ["DKI Jakarta", "Jawa Barat"]

    city = df["city"]
    assert city.iloc[0] == "DKI Jakarta"


def test_groupby_falls_back_to_fresh_when_read_cache_stale() -> None:
    df = DFrame({"city": ["jakarta", "bandung"], "score": [10, 20]})
    # Simulate a stale read cache by increasing sync delay after seed.
    df._coordinator.read_node.sync_delay_ms = 300
    df["city"] = ["DKI Jakarta", "Jawa Barat"]

    grouped = df.groupby("city")["score"].sum()
    assert grouped.loc["DKI Jakarta"] == 10


def test_read_global_sync_returns_dataframe() -> None:
    df = DFrame({"city": ["jakarta", "bandung"], "score": [85, 90]})

    merged = df.read_global()

    assert list(merged.columns) == ["city", "score"]
    assert merged.shape == (2, 2)
    assert merged.iloc[0]["city"] == "jakarta"


def test_read_global_raises_when_event_loop_running() -> None:
    df = DFrame({"city": ["jakarta"]})

    async def _call_inside_loop() -> None:
        with pytest.raises(RuntimeError, match="read_global\\(\\) cannot be called while an event loop is running"):
            _ = df.read_global()

    asyncio.run(_call_inside_loop())


def test_read_fresh_lazy_chunking() -> None:
    df = DFrame({"city": [f"city_{i}" for i in range(105)], "score": list(range(105))})
    chunks = list(df.read_fresh_lazy(chunk_size=20))
    assert len(chunks) == 6  # 105/20 = 5.25 -> 6 chunks
    assert chunks[0].shape == (20, 2)
    assert chunks[-1].shape == (5, 2)
    # Check content
    assert chunks[0].iloc[0]["city"] == "city_0"
    assert chunks[-1].iloc[-1]["score"] == 104


def test_read_global_lazy_chunking() -> None:
    df = DFrame({"city": [f"city_{i}" for i in range(105)], "score": list(range(105))})
    chunks = list(df.read_global_lazy(chunk_size=20))
    assert len(chunks) == 6  # 105/20 = 5.25 -> 6 chunks
    assert chunks[0].shape == (20, 2)
    assert chunks[-1].shape == (5, 2)
    # Check content
    assert chunks[0].iloc[0]["city"] == "city_0"
    assert chunks[-1].iloc[-1]["score"] == 104
