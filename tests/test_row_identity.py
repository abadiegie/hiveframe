# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Fase 1: identitas baris stabil.

Posisi baris tetap dipakai sebagai indeks penyimpanan, tetapi setiap baris punya
id yang tidak berubah saat baris lain dihapus atau disisipkan. Identitas yang
terhapus ikut tercatat di payload operasi struktural supaya pemulihan tidak
bergantung pada posisi yang bergeser.
"""

from __future__ import annotations

from core.dataframe import DFrame


def _frame(data: dict, frame_id: str) -> DFrame:
    return DFrame(data=data, frame_id=frame_id)


class TestIdentitasBaris:
    def test_id_dibangun_deterministik(self) -> None:
        df = _frame({"a": [1, 2, 3]}, "id-stabil")

        pertama = df.row_ids()
        kedua = df.row_ids()

        assert pertama == kedua == ["r0", "r1", "r2"]

    def test_id_tidak_berubah_saat_baris_lain_dihapus(self) -> None:
        df = _frame({"nama": ["A", "B", "C"]}, "id-hapus")
        awal = df.row_ids()

        df.delete_rows(rows=[1])

        assert df.row_ids() == [awal[0], awal[2]]
        assert df.read_fresh()["nama"].tolist() == ["A", "C"]

    def test_hapus_berdasarkan_id(self) -> None:
        df = _frame({"nama": ["A", "B", "C"]}, "id-hapus-id")
        awal = df.row_ids()

        # C sekarang di posisi 1, tapi identitasnya tetap r2.
        df.delete_rows(row_ids=[awal[2]])

        assert df.read_fresh()["nama"].tolist() == ["A", "B"]
        assert df.row_ids() == [awal[0], awal[1]]

    def test_sisip_di_tengah_menggeser_nilai(self) -> None:
        df = _frame({"nama": ["A", "B", "C"]}, "id-sisip")
        awal = df.row_ids()

        df.insert_rows(1, at=1, default="X")

        assert df.read_fresh()["nama"].tolist() == ["A", "X", "B", "C"]
        ids = df.row_ids()
        assert ids[0] == awal[0]
        assert ids[2:] == [awal[1], awal[2]], "id baris lama tetap, hanya bergeser"
        assert len(ids) == 4

    def test_payload_hapus_mencatat_identitas(self) -> None:
        df = _frame({"nama": ["A", "B", "C"]}, "id-payload")
        awal = df.row_ids()

        df.delete_rows(rows=[2])

        entry = df._coordinator.wal.get_committed()[-1]
        payload = entry["operations"][0]["payload"]
        assert payload["deleted_row_ids"] == [awal[2]]

    def test_id_baru_tidak_memakai_ulang_id_terhapus(self) -> None:
        df = _frame({"nama": ["A", "B"]}, "id-unik")
        awal = df.row_ids()

        df.delete_rows(rows=[1])
        df.append_rows(1, default="C")

        ids = df.row_ids()
        assert ids[0] == awal[0]
        assert ids[1] not in {awal[1]}, "id yang pernah dihapus tidak dipakai ulang"
        assert len(set(ids)) == len(ids)
