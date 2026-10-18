# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Regresi baca malas: streaming per batch, lompatan start_row, dan batas memori.

Sebelumnya read_fresh_lazy membaca seluruh parquet ke memori
(pq.read_table(...).to_pandas()) lalu memotongnya per chunk, sehingga permintaan
jendela 1.000 baris pada frame besar tetap memuat seluruh frame.
"""

from __future__ import annotations

import gc

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from core.coordinator import TransactionCoordinator
from core.dataframe import DFrame
from core.read_node import ReadNode
from core.write_node import WriteNode

BARIS = 200_000


def _buat_parquet(tmp_path, frame_id: str, baris: int, grup: int = 10_000) -> None:
    tabel = pa.table({
        "nomor": pa.array(range(baris), type=pa.int64()),
        "teks": pa.array([f"b{i}" for i in range(baris)]),
    })
    pq.write_table(tabel, tmp_path / f"{frame_id}.parquet", row_group_size=grup)


def _df_dari_parquet(tmp_path, baris: int, frame_id: str = "frame-uji") -> tuple[DFrame, TransactionCoordinator]:
    _buat_parquet(tmp_path, frame_id, baris)
    coord = TransactionCoordinator(
        write_node=WriteNode(),
        read_node=ReadNode(storage_dir=str(tmp_path)),
    )
    return DFrame(coordinator=coord, frame_id=frame_id), coord


def _rss_mb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 1048576


def test_row_count_dari_metadata_tanpa_materialisasi(tmp_path) -> None:
    df, coord = _df_dari_parquet(tmp_path, BARIS)

    assert df.row_count() == BARIS
    # Write node tetap kosong: tidak ada data yang dimuat ke memori.
    assert coord.write_node._df.empty


def test_iter_record_batches_ukuran_dan_urutan(tmp_path) -> None:
    df, _ = _df_dari_parquet(tmp_path, BARIS)

    potongan = list(df.iter_record_batches(batch_size=10_000))
    jumlah = [p.num_rows for p in potongan]

    assert sum(jumlah) == BARIS
    assert set(jumlah[:-1]) == {10_000}
    assert potongan[0].column("nomor")[0].as_py() == 0
    assert potongan[-1].column("nomor")[-1].as_py() == BARIS - 1


def test_iter_record_batches_melewati_start_row(tmp_path) -> None:
    df, _ = _df_dari_parquet(tmp_path, BARIS)
    mulai = 123_456

    potongan = list(df.iter_record_batches(batch_size=10_000, start_row=mulai))
    nomor = [n for p in potongan for n in p.column("nomor").to_pylist()]

    assert len(nomor) == BARIS - mulai
    assert nomor[0] == mulai
    assert nomor[-1] == BARIS - 1


def test_read_fresh_lazy_mempertahankan_indeks_global(tmp_path) -> None:
    df, _ = _df_dari_parquet(tmp_path, BARIS)
    mulai = 50_000

    chunk = next(iter(df.read_fresh_lazy(chunk_size=1_000, start_row=mulai)))

    assert list(chunk.index) == list(range(mulai, mulai + 1_000))
    assert chunk["nomor"].tolist() == list(range(mulai, mulai + 1_000))


def test_memori_streaming_jauh_di_bawah_baca_penuh(tmp_path) -> None:
    baris = 1_000_000
    df, _ = _df_dari_parquet(tmp_path, baris)

    gc.collect()
    dasar = _rss_mb()
    puncak = dasar
    jumlah = 0
    for potongan in df.iter_record_batches(batch_size=1_000):
        jumlah += potongan.num_rows
        puncak = max(puncak, _rss_mb())
    delta_streaming = puncak - dasar

    gc.collect()
    dasar_penuh = _rss_mb()
    # Perbandingan yang adil: cara baca lama, yaitu seluruh parquet sekaligus.
    penuh = pq.read_table(str(tmp_path / "frame-uji.parquet")).to_pandas()
    delta_penuh = _rss_mb() - dasar_penuh

    assert jumlah == baris
    assert len(penuh) == baris
    # Streaming memakai memori jauh lebih sedikit daripada membaca seluruh frame.
    assert delta_streaming < 32, f"streaming {delta_streaming:.1f} MB terlalu besar"
    assert delta_penuh > delta_streaming, (
        f"baca penuh {delta_penuh:.1f} MB tidak lebih besar dari streaming {delta_streaming:.1f} MB"
    )
