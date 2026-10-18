# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Regresi: jumlah baris sebuah frame tidak boleh ditentukan frame lain.

WriteNode menyimpan satu DataFrame global berisi kolom SEMUA frame (namespaced
``{frame_id}::{col}``), sehingga kolom frame pendek ikut memanjang mengikuti
frame terpanjang di write node yang sama.

Gejala nyata (hivesheet, cluster runtime aktif): unggah xlsx 7 baris tersimpan
sebagai parquet 544 baris, hanya 7 baris pertama terisi dan sisanya null, dan
API melaporkan ``rows: 544``.
"""

from __future__ import annotations

from core.coordinator import TransactionCoordinator
from core.dataframe import DFrame


def test_frame_pendek_tidak_mewarisi_panjang_frame_lain() -> None:
    coord = TransactionCoordinator()

    panjang = DFrame(data={"x": list(range(544))}, coordinator=coord)
    pendek = DFrame(data={"y": list(range(7))}, coordinator=coord)

    assert len(panjang.read_fresh()) == 544
    assert len(pendek.read_fresh()) == 7
    assert list(pendek.read_fresh()["y"]) == list(range(7))
    # Frame panjang tidak boleh ikut terpotong.
    assert len(panjang.read_fresh()) == 544


def test_chunk_tambahan_dicatat_sebagai_baris_frame() -> None:
    """Jalur loader malas menambah data per chunk dengan row_offset.

    Yang diuji di sini hanya pencatatan jumlah barisnya (perubahan pada
    _append_chunk_to_write_node), bukan semantik baca jalur chunk.
    """
    coord = TransactionCoordinator()

    panjang = DFrame(data={"x": list(range(100))}, coordinator=coord)
    pendek = DFrame(coordinator=coord)

    pendek._append_chunk_to_write_node({"y": [1, 2, 3]}, row_offset=0)
    pendek._append_chunk_to_write_node({"y": [4, 5]}, row_offset=3)

    assert pendek._coordinator.write_node._frame_rows[pendek._frame_id] == 5
    assert len(panjang.read_fresh()) == 100
