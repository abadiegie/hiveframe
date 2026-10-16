# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Regresi: frame tanpa baris tetap punya kolom.

Sebelumnya _seed_initial_data kembali lebih awal saat panjang data 0, sehingga
kolom tidak pernah dibuat. Akibatnya frame kosong tidak punya skema: penulisan
sel ditolak dan urutan kolom tidak bisa diterapkan (ditemukan lewat hivesheet:
POST /sheets dengan daftar kolom lalu menulis sel menghasilkan 400).
"""

from __future__ import annotations

from core.coordinator import TransactionCoordinator
from core.dataframe import DFrame


def test_frame_tanpa_baris_tetap_punya_kolom() -> None:
    df = DFrame(data={"a": [], "b": []})

    assert list(df.columns) == ["a", "b"]
    assert len(df.read_fresh()) == 0


def test_kolom_kosong_bisa_diisi_kemudian() -> None:
    df = DFrame(data={"a": [], "b": []})

    df["a"] = [1, 2]

    fresh = df.read_fresh()
    assert fresh["a"].tolist() == [1, 2]
    assert "b" in fresh.columns


def test_dict_kosong_tidak_membuat_kolom() -> None:
    coord = TransactionCoordinator()
    df = DFrame(data={}, coordinator=coord)

    assert list(df.columns) == []
    assert coord.write_node._df.empty
