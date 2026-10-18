# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Partisi baris antar node dan penggabungan agregat lintas potongan.

Dua hal yang belum pernah diuji: batas kepemilikan baris pada rentang partisi,
dan kebenaran gabungan saat agregat dihitung dari lebih dari satu potongan.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from core.dataframe import DFrame
from core.sqlite_registry import SQLiteRegistry


def _sisipkan(reg: SQLiteRegistry, node_id: str, awal: int, akhir: int, status: str = "healthy") -> None:
    con = reg._require_connection()
    con.execute(
        "insert into nodes (node_id, host, port, role, region, partition_start, partition_end,"
        " last_seen, lsn, status) values (?,?,?,?,?,?,?,?,?,?)",
        (node_id, "127.0.0.1", 0, "write", "r", awal, akhir, time.time(), 0, status),
    )
    con.commit()


class TestKepemilikanBaris:
    def test_batas_rentang(self, tmp_path) -> None:
        reg = SQLiteRegistry(db_path=str(tmp_path / "registry.db"))
        asyncio.run(reg.connect())
        _sisipkan(reg, "A", 0, 500)
        _sisipkan(reg, "B", 500, 1000)

        assert reg.get_owner_for_row(0).node_id == "A"
        assert reg.get_owner_for_row(499).node_id == "A"
        assert reg.get_owner_for_row(500).node_id == "B"
        assert reg.get_owner_for_row(999).node_id == "B"

    def test_hanya_node_sehat_yang_dianggap_pemilik(self, tmp_path) -> None:
        reg = SQLiteRegistry(db_path=str(tmp_path / "registry.db"))
        asyncio.run(reg.connect())
        _sisipkan(reg, "A", 0, 1000, status="suspect")

        assert reg.get_owner_for_row(10) is None

    def test_tanpa_writer_mengembalikan_none(self, tmp_path) -> None:
        reg = SQLiteRegistry(db_path=str(tmp_path / "registry.db"))
        asyncio.run(reg.connect())

        assert reg.get_owner_for_row(0) is None


def _frame(baris: list[tuple[str, int]]) -> DFrame:
    df = DFrame(
        data={"platform": [b[0] for b in baris], "impresi": [b[1] for b in baris]},
        frame_id="partisi-uji",
    )
    # Paksa jalur cluster dan sediakan dua potongan global yang berbeda.
    df._runtime = object()
    potongan = [
        df.read_fresh().iloc[: len(baris) // 2].copy(),
        df.read_fresh().iloc[len(baris) // 2 :].copy(),
    ]
    df.read_fresh_global_lazy = lambda *a, **k: iter(potongan)  # type: ignore[method-assign]
    return df


class TestAgregasiLintasPotongan:
    def test_sum_digabung_dari_dua_potongan(self) -> None:
        df = _frame([("ig", 100), ("tt", 50), ("ig", 200), ("tt", 70)])

        hasil = df.groupby_aggregate(["platform"], ["impresi"], "sum")

        assert {r["platform"]: r["impresi"] for r in hasil} == {"ig": 300.0, "tt": 120.0}

    def test_count_digabung_dari_dua_potongan(self) -> None:
        df = _frame([("ig", 100), ("tt", 50), ("ig", 200), ("tt", 70)])

        hasil = df.groupby_aggregate(["platform"], [], "count")

        assert {r["platform"]: r["jumlah"] for r in hasil} == {"ig": 2, "tt": 2}

    def test_mean_memakai_total_dan_jumlah_baris(self) -> None:
        df = _frame([("ig", 100), ("tt", 50), ("ig", 200), ("tt", 70)])

        hasil = df.groupby_aggregate(["platform"], ["impresi"], "mean")

        assert {r["platform"]: r["impresi"] for r in hasil} == {"ig": 150.0, "tt": 60.0}

    def test_filter_diterapkan_per_potongan(self) -> None:
        df = _frame([("ig", 100), ("tt", 50), ("ig", 200), ("tt", 70)])

        hasil = df.groupby_aggregate(
            ["platform"], ["impresi"], "sum", filter_col="platform", filter_value="tt"
        )

        assert hasil == [{"platform": "tt", "impresi": 120.0}]
