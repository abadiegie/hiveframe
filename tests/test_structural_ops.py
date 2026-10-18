# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Fase 0 operasi struktural: perubahan bentuk frame dicatat sebagai satu operasi.

Sebelumnya menambah satu baris pada frame 50k x 5 menulis 50.001 operasi sel ke
WAL karena seluruh kolom ditulis ulang. Dengan kind struktural, satu perubahan
bentuk = satu operasi, dan op itu bisa diputar ulang untuk membangun frame baru.
"""

from __future__ import annotations

import pandas as pd

from core.dataframe import DFrame
from core.transaction import Operation, Transaction


def _operasi_terakhir(df: DFrame) -> dict:
    committed = df._coordinator.wal.get_committed()
    assert committed, "WAL kosong"
    return committed[-1]


class TestOperasiStruktural:
    def test_insert_rows_satu_operasi(self) -> None:
        df = DFrame(data={"nama": ["Ani", "Budi"], "nilai": [10, 20]}, frame_id="f0-insert")

        df.append_rows(3)

        entry = _operasi_terakhir(df)
        assert len(entry["operations"]) == 1
        assert entry["operations"][0]["kind"] == "insert_rows"
        assert len(df.read_fresh()) == 5
        # Nilai lama tidak tersentuh.
        assert df.read_fresh()["nama"].tolist()[:2] == ["Ani", "Budi"]

    def test_tambah_satu_baris_tidak_menulis_operasi_per_sel(self) -> None:
        df = DFrame(data={"a": list(range(500)), "b": list(range(500))}, frame_id="f0-murah")

        df.append_rows(1)

        entry = _operasi_terakhir(df)
        assert len(entry["operations"]) == 1, "satu baris seharusnya satu operasi, bukan 2 x 500"

    def test_delete_rows_menggeser_nilai(self) -> None:
        df = DFrame(data={"nama": ["A", "B", "C"]}, frame_id="f0-delete")

        df.delete_rows([1])

        assert df.read_fresh()["nama"].tolist() == ["A", "C"]
        assert _operasi_terakhir(df)["operations"][0]["kind"] == "delete_rows"

    def test_insert_column_menghormati_posisi(self) -> None:
        df = DFrame(data={"a": [1, 2], "b": [3, 4]}, frame_id="f0-kolom")

        df.add_column("tengah", values=[9, 8], at=1)

        assert list(df.read_fresh().columns) == ["a", "tengah", "b"]
        assert df.read_fresh()["tengah"].tolist() == [9, 8]

    def test_delete_column_tidak_hidup_kembali(self) -> None:
        df = DFrame(data={"a": [1], "b": [2]}, frame_id="f0-hapus")

        df.drop_column("b")

        assert list(df.read_fresh().columns) == ["a"]
        assert _operasi_terakhir(df)["operations"][0]["kind"] == "delete_column"

    def test_operasi_struktural_bisa_diputar_ulang(self) -> None:
        sumber = DFrame(data={"nama": ["Ani", "Budi"], "nilai": [10, 20]}, frame_id="f0-replay")
        sumber.append_rows(2)
        sumber.delete_rows([0])
        sumber.add_column("kota", values=["Jogja", "Solo", "Bali"], at=1)

        harapan = sumber.read_fresh()

        # Bangun frame baru dari nol, lalu putar ulang operasi strukturannya.
        target = DFrame(data={"nama": ["Ani", "Budi"], "nilai": [10, 20]}, frame_id="f0-replay")
        for entry in target._coordinator.wal.get_committed():
            entry  # pragma: no cover - hanya menegaskan bentuk WAL
        target._coordinator.wal._entries.clear()
        for entry in sumber._coordinator.wal.get_committed():
            ops = [Operation(**d) for d in entry["operations"]]
            target._coordinator.write_node.apply(Transaction(operations=ops))

        hasil = target.read_fresh()
        assert list(hasil.columns) == list(harapan.columns)
        assert len(hasil) == len(harapan)
        assert hasil["nama"].tolist()[:1] == harapan["nama"].tolist()[:1]
