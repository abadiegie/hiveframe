# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from datetime import date

import pandas as pd
import pytest

from core.dataframe import DFrame, read
from core.schema import ColumnSchema


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


def test_write_non_transactional_skips_tx_and_wal() -> None:
    df = DFrame({"a": [1]}, transactional=False)
    before = df._coordinator.get_stats()["tx_count"]
    before_wal = len(df._coordinator.wal._entries)

    df["a"] = [2]

    after = df._coordinator.get_stats()["tx_count"]
    after_wal = len(df._coordinator.wal._entries)
    assert after == before
    assert after_wal == before_wal
    assert df.read_fresh().at[0, "a"] == 2


def test_non_transactional_disables_audit_and_rollback_features() -> None:
    df = DFrame({"city": ["jakarta"]}, transactional=False)

    assert df.cell_history("city", 0) == []
    with pytest.raises(RuntimeError, match="checkpoint\(\) is unavailable"):
        df.checkpoint("cp1")
    with pytest.raises(RuntimeError, match="rollback\(\) is unavailable"):
        df.rollback("cp1")


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
    reloaded = read(str(path))
    assert reloaded.read_fresh().to_dict(orient="list") == {"x": [1, 2]}


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


def test_schema_validation_runs_during_seed() -> None:
    with pytest.raises(TypeError, match="Column 'score' expects int"):
        DFrame(
            {"score": ["bad"]},
            schema={"score": ColumnSchema(dtype="int", nullable=False)},
        )


def test_schema_validation_rejects_invalid_assignment() -> None:
    df = DFrame(
        {"score": [85]},
        schema={
            "score": ColumnSchema(
                dtype="int",
                nullable=False,
                validator=lambda value: 0 <= value <= 100,
            )
        },
    )

    with pytest.raises(ValueError, match="failed custom validation"):
        df["score"] = [120]

    assert df.read_fresh().at[0, "score"] == 85


def test_schema_date_validation_accepts_python_date() -> None:
    df = DFrame(
        {"created_at": [date(2026, 4, 4)]},
        schema={"created_at": ColumnSchema(dtype="date", nullable=False)},
    )

    assert df.read_fresh().at[0, "created_at"] == date(2026, 4, 4)


def test_schema_date_validation_rejects_non_date_assignment() -> None:
    df = DFrame(
        {"created_at": [date(2026, 4, 4)]},
        schema={"created_at": ColumnSchema(dtype="date", nullable=False)},
    )

    with pytest.raises(TypeError, match="Column 'created_at' expects date"):
        df["created_at"] = ["2026-04-04"]


def test_schema_coercion_is_opt_in() -> None:
    with pytest.raises(TypeError, match="Column 'score' expects int"):
        DFrame(
            {"score": ["85"]},
            schema={"score": ColumnSchema(dtype="int", nullable=False)},
        )


def test_schema_coercion_normalizes_seed_and_assignment_values() -> None:
    df = DFrame(
        {"score": ["85"]},
        schema={"score": ColumnSchema(dtype="int", nullable=False, coerce=True)},
    )

    assert df.read_fresh().at[0, "score"] == 85
    assert isinstance(df.read_fresh().at[0, "score"], int)

    df["score"] = ["90"]
    assert df.read_fresh().at[0, "score"] == 90
    assert isinstance(df.read_fresh().at[0, "score"], int)


def test_read_uses_schema_sidecar_for_reload(tmp_path) -> None:
    df = DFrame(
        {"score": ["85"]},
        schema={"score": ColumnSchema(dtype="int", nullable=False, coerce=True, description="0-100")},
    )
    df._coordinator.read_node._storage_dir = tmp_path

    path = df.to_persistent("schema_reload")
    reloaded = read(str(path))

    reloaded["score"] = ["91"]
    assert reloaded.read_fresh().at[0, "score"] == 91
    assert reloaded._schema["score"].coerce is True
    assert reloaded._schema["score"].description == "0-100"


def test_read_accepts_explicit_schema_for_parquet_path(tmp_path) -> None:
    path = tmp_path / "scores.parquet"
    pd.DataFrame({"score": ["85", "90"]}).to_parquet(path)

    reloaded = read(
        str(path),
        schema={"score": ColumnSchema(dtype="int", nullable=False, coerce=True)},
    )

    assert reloaded.read_fresh().to_dict(orient="list") == {"score": [85, 90]}


