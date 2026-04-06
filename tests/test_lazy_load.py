# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

import asyncio
import math

import pandas as pd
import pytest

from core.dataframe import DFrame
from core.schema import ColumnSchema


def test_from_csv_lazy_loads_all_rows(tmp_path) -> None:
    csv_path = tmp_path / "lazy.csv"
    source = pd.DataFrame({"city": [f"city_{i}" for i in range(1000)], "score": list(range(1000))})
    source.to_csv(csv_path, index=False)

    loaded = asyncio.run(DFrame.from_csv_lazy(str(csv_path), chunk_size=128))

    fresh = loaded.read_fresh()
    assert fresh.shape == (1000, 2)
    assert fresh.iloc[0]["city"] == "city_0"
    assert int(fresh.iloc[-1]["score"]) == 999


def test_from_csv_lazy_memory_is_chunked(tmp_path) -> None:
    csv_path = tmp_path / "chunked.csv"
    rows = 501
    source = pd.DataFrame({"x": list(range(rows))})
    source.to_csv(csv_path, index=False)

    loaded = asyncio.run(DFrame.from_csv_lazy(str(csv_path), chunk_size=100))

    chunk_count = math.ceil(rows / 100)
    wal_entries = [
        entry
        for entry in loaded._coordinator.wal._entries
        if entry.operations and entry.operations[0]["cell_id"].endswith("::__bulk_chunk__")
    ]
    assert len(wal_entries) == chunk_count


def test_from_csv_lazy_with_schema_validation(tmp_path) -> None:
    csv_path = tmp_path / "schema.csv"
    pd.DataFrame({"score": [1, "bad", 3]}).to_csv(csv_path, index=False)

    with pytest.raises(TypeError, match="Column 'score' expects int"):
        asyncio.run(
            DFrame.from_csv_lazy(
                str(csv_path),
                chunk_size=2,
                schema={"score": ColumnSchema(dtype="int", nullable=False)},
            )
        )


def test_from_csv_lazy_progress_callback(tmp_path) -> None:
    csv_path = tmp_path / "progress.csv"
    pd.DataFrame({"v": list(range(250))}).to_csv(csv_path, index=False)

    progress: list[int] = []
    _ = asyncio.run(
        DFrame.from_csv_lazy(
            str(csv_path),
            chunk_size=100,
            on_progress=lambda n: progress.append(n),
        )
    )

    assert progress == [100, 200, 250]


def test_from_excel_lazy_loads_all_rows(tmp_path) -> None:
    pytest.importorskip("openpyxl")

    path = tmp_path / "lazy.xlsx"
    source = pd.DataFrame({"name": [f"n{i}" for i in range(150)], "age": list(range(150))})
    source.to_excel(path, index=False)

    loaded = asyncio.run(DFrame.from_excel_lazy(str(path), chunk_size=40))
    fresh = loaded.read_fresh()

    assert fresh.shape == (150, 2)
    assert fresh.iloc[0]["name"] == "n0"
    assert int(fresh.iloc[-1]["age"]) == 149


def test_from_excel_lazy_progress_callback(tmp_path) -> None:
    pytest.importorskip("openpyxl")

    path = tmp_path / "progress.xlsx"
    pd.DataFrame({"v": list(range(205))}).to_excel(path, index=False)

    progress: list[int] = []
    _ = asyncio.run(
        DFrame.from_excel_lazy(
            str(path),
            chunk_size=80,
            on_progress=lambda n: progress.append(n),
        )
    )

    assert progress == [80, 160, 205]


def test_lazy_same_result_as_eager(tmp_path) -> None:
    path = tmp_path / "same.csv"
    source = pd.DataFrame({"city": ["a", "b", "c"], "score": [10, 20, 30]})
    source.to_csv(path, index=False)

    eager = DFrame.from_csv(str(path))
    lazy = asyncio.run(DFrame.from_csv_lazy(str(path), chunk_size=2))

    assert eager.read_fresh().to_dict(orient="list") == lazy.read_fresh().to_dict(orient="list")


def test_large_chunked_seed_wal_entries(tmp_path) -> None:
    path = tmp_path / "wal.csv"
    rows = 1005
    chunk_size = 200
    pd.DataFrame({"v": list(range(rows))}).to_csv(path, index=False)

    loaded = asyncio.run(DFrame.from_csv_lazy(str(path), chunk_size=chunk_size))

    expected_chunks = math.ceil(rows / chunk_size)
    wal_entries = [
        entry
        for entry in loaded._coordinator.wal._entries
        if entry.operations and entry.operations[0]["cell_id"].endswith("::__bulk_chunk__")
    ]
    assert len(wal_entries) == expected_chunks
    assert all(entry.operations[0]["new_value"]["rows"] <= chunk_size for entry in wal_entries)


def test_from_csv_lazy_non_transactional_skips_wal(tmp_path) -> None:
    path = tmp_path / "no_tx.csv"
    pd.DataFrame({"v": list(range(25))}).to_csv(path, index=False)

    loaded = asyncio.run(
        DFrame.from_csv_lazy(str(path), chunk_size=10, transactional=False)
    )

    assert loaded.read_fresh().shape == (25, 1)
    assert len(loaded._coordinator.wal._entries) == 0


