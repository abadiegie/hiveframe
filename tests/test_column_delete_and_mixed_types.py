# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Regresi: hapus kolom dan pembacaan frame dengan kolom bercampur tipe.

Dua gejala nyata dari hivesheet:
- `del df[kolom]` selalu AttributeError karena DFrame tidak punya __delitem__,
  sehingga fitur hapus kolom gagal 400.
- Kolom numerik yang berisi string kosong (mis. baris baru ditulis "") tidak bisa
  dikonversi ke Arrow, sehingga pencarian dan pembacaan gagal total.
"""

from __future__ import annotations

import pandas as pd

from core.dataframe import DFrame


def test_hapus_kolom_membuang_nilai_dan_skema() -> None:
    df = DFrame(data={"a": [1, 2], "b": ["x", "y"]}, frame_id="uji-hapus")

    del df["b"]

    sisa = df.read_fresh()
    assert list(sisa.columns) == ["a"]
    assert sisa["a"].tolist() == [1, 2]


def test_hapus_kolom_tidak_ada_menghasilkan_keyerror() -> None:
    df = DFrame(data={"a": [1]}, frame_id="uji-hapus2")

    try:
        del df["tidak-ada"]
    except KeyError as exc:
        assert "tidak-ada" in str(exc)
    else:  # pragma: no cover - seharusnya tidak terjadi
        raise AssertionError("KeyError tidak dilempar")


def test_frame_tanpa_kolom_object_gagal_dibaca() -> None:
    """Kolom object bercampur (angka + teks kosong) tetap bisa dibaca sebagai batch."""
    df = DFrame(data={"nilai": [10, 20]}, frame_id="uji-campur")

    # Tulis nilai kosong seperti perilaku lama yang membuat dtype jadi object.
    df["nilai"] = [10, 20, ""]

    baris: list = []
    for batch in df.iter_record_batches(columns=["nilai"], batch_size=2):
        baris.extend(batch.to_pandas()["nilai"].tolist())

    assert len(baris) == 3
    # Nilai angka boleh kembali sebagai teks ("10" atau "10.0") karena kolom
    # bercampur dipaksa ke teks; yang penting baris kosong tetap kosong.
    assert float(baris[0]) == 10.0
    assert pd.isna(baris[2]) or str(baris[2]).strip() in {"", "nan", "<NA>"}
