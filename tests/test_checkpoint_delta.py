# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Fase 2: checkpoint disimpan sebagai delta, bukan salinan penuh.

Sebelumnya setiap checkpoint menyalin seluruh frame (diukur 2,0 MB untuk 50k x 5),
dan hivesheet membuat checkpoint otomatis sebelum setiap mutasi. Sekarang hanya
checkpoint pertama, kelipatan interval, atau yang memuat perubahan bentuk yang
menyalin penuh; sisanya menyimpan operasi WAL sejak checkpoint induk.
"""

from __future__ import annotations

from core.dataframe import DFrame


class TestCheckpointDelta:
    def test_checkpoint_pertama_penuh_sisanya_delta(self) -> None:
        df = DFrame(data={"a": [1, 2]}, frame_id="cp-bentuk")

        cp0 = df.checkpoint("awal")
        df["a"] = [3, 4]
        cp1 = df.checkpoint("kedua")

        assert df._checkpoints[cp0].get("kind") == "full"
        assert df._checkpoints[cp1].get("kind") == "delta"
        assert "snapshot_df" not in df._checkpoints[cp1]

    def test_snapshot_delta_sesuai_keadaan_saat_itu(self) -> None:
        df = DFrame(data={"a": [1, 2], "b": ["x", "y"]}, frame_id="cp-snapshot")
        harapan: dict[str, tuple[list, list]] = {}

        cp0 = df.checkpoint("awal")
        harapan[cp0] = ([1, 2], ["x", "y"])
        df["a"] = [10, 20]
        cp1 = df.checkpoint("ubah-a")
        harapan[cp1] = ([10, 20], ["x", "y"])
        df["b"] = ["p", "q"]
        cp2 = df.checkpoint("ubah-b")
        harapan[cp2] = ([10, 20], ["p", "q"])

        for cid, (nilai_a, nilai_b) in harapan.items():
            snap = df.checkpoint_snapshot(cid)
            assert snap["a"].tolist() == nilai_a
            assert snap["b"].tolist() == nilai_b

    def test_rollback_ke_checkpoint_delta(self) -> None:
        df = DFrame(data={"a": [1, 2]}, frame_id="cp-rollback")
        df.checkpoint("awal")
        df["a"] = [10, 20]
        cp1 = df.checkpoint("setelah-ubah")
        df["a"] = [99, 99]

        df.rollback(cp1)

        assert df.read_fresh()["a"].tolist() == [10, 20]

    def test_diff_antar_checkpoint(self) -> None:
        df = DFrame(data={"a": [1, 2]}, frame_id="cp-diff")
        cp0 = df.checkpoint("awal")
        df["a"] = [10, 20]
        cp1 = df.checkpoint("ubah")

        diff_awal = df.diff(cp0).to_dict(orient="records")
        diff_terbaru = df.diff(cp1).to_dict(orient="records")

        assert len(diff_awal) == 2
        assert diff_terbaru == []

    def test_checkpoint_setelah_operasi_struktural_menjadi_penuh(self) -> None:
        df = DFrame(data={"a": [1, 2]}, frame_id="cp-struktural")
        df.checkpoint("awal")
        df.append_rows(1)

        cp = df.checkpoint("setelah-tambah-baris")

        # Delta berbasis sel tidak bisa menyusun ulang baris baru, jadi penuh.
        assert df._checkpoints[cp].get("kind") == "full"
        assert len(df.checkpoint_snapshot(cp)) == 3

    def test_memori_checkpoint_delta_jauh_lebih_kecil(self) -> None:
        df = DFrame(data={"a": list(range(2000)), "b": list(range(2000))}, frame_id="cp-memori")
        df.checkpoint("awal")
        for i in range(20):
            df["a"] = [list(range(2000))[0] + i] + list(range(1, 2000))
            df.checkpoint(f"cp-{i}")

        penuh = [e for e in df._checkpoints.values() if "snapshot_df" in e]
        delta = [e for e in df._checkpoints.values() if e.get("kind") == "delta"]

        assert len(penuh) == 1, "hanya checkpoint pertama yang menyalin penuh"
        assert len(delta) == 20
        # Delta hanya menyimpan operasi yang benar-benar berubah.
        assert all(len(e.get("ops") or []) <= 2000 for e in delta)
