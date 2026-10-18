# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Agregasi kelompok di DFrame: dipakai agent untuk menjawab pertanyaan tanpa
memindahkan seluruh baris."""

from __future__ import annotations

import pytest

from core.dataframe import DFrame


def _frame() -> DFrame:
    return DFrame(
        data={
            "platform": ["ig", "ig", "tt", "tt", "x"],
            "impresi": [100, 200, 50, 70, 30],
            "klik": [10, 20, 5, 7, 3],
        },
        frame_id="agregasi-uji",
    )


class TestGroupbyAggregate:
    def test_sum_per_kelompok(self) -> None:
        hasil = _frame().groupby_aggregate(["platform"], ["impresi"], "sum")

        assert hasil[0] == {"platform": "ig", "impresi": 300}
        assert len(hasil) == 3

    def test_count_tanpa_kolom_nilai(self) -> None:
        hasil = _frame().groupby_aggregate(["platform"], [], "count")

        assert hasil[0] == {"platform": "ig", "jumlah": 2}

    def test_mean_memakai_pembagi_jumlah_baris(self) -> None:
        hasil = _frame().groupby_aggregate(["platform"], ["impresi"], "mean")

        assert hasil[0]["impresi"] == pytest.approx(150.0)

    def test_min_dan_max(self) -> None:
        df = _frame()
        assert df.groupby_aggregate(["platform"], ["impresi"], "min")[-1]["impresi"] == 30
        assert df.groupby_aggregate(["platform"], ["klik"], "max")[0]["klik"] == 20

    def test_filter_sebelum_agregasi(self) -> None:
        hasil = _frame().groupby_aggregate(
            ["platform"], ["impresi"], "sum", filter_col="platform", filter_value="ig"
        )

        assert hasil == [{"platform": "ig", "impresi": 300}]

    def test_limit_membatasi_hasil(self) -> None:
        hasil = _frame().groupby_aggregate(["platform"], ["impresi"], "sum", limit=1)

        assert len(hasil) == 1
        assert hasil[0]["platform"] == "ig"

    def test_fungsi_tidak_dikenal(self) -> None:
        with pytest.raises(ValueError):
            _frame().groupby_aggregate(["platform"], ["impresi"], "median")

    def test_kolom_tidak_ada(self) -> None:
        with pytest.raises(KeyError):
            _frame().groupby_aggregate(["tidak-ada"], ["impresi"])
