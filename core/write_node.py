# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Write node backed by mutable pandas DataFrame."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any
import logging

import pandas as pd

from .transaction import Transaction

DeltaCallback = Callable[[str, list[dict[str, Any]], int], None]


logger = logging.getLogger("core.write_node")


class WriteNode:
    """Thread-safe mutable store for transactional writes.

    ``bulk_mode`` remains a throughput hint, but apply() still preserves atomic rollback
    semantics on conflicts or exceptions.
    """

    # Shared pool: one thread is plenty for sequential delta callbacks; max_workers=2
    # gives headroom for bursts without spawning unbounded threads.
    _callback_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="wn-delta")

    def __init__(self, initial_data: dict[str, list[Any]] | None = None, bulk_mode: bool = True) -> None:
        self._lock = Lock()
        self._df = pd.DataFrame(initial_data or {})
        self._version = 0
        # Jumlah baris MILIK tiap frame. WriteNode menyimpan satu DataFrame global
        # berisi kolom semua frame, sehingga tanpa catatan ini jumlah baris sebuah
        # frame ikut jumlah baris frame terpanjang (frame baru jadi penuh baris
        # kosong, dan parquet-nya ikut membengkak).
        self._frame_rows: dict[str, int] = {}
        # Fase 1: identitas baris stabil per frame. Posisi baris tetap dipakai
        # sebagai indeks penyimpanan, tetapi setiap baris punya id yang tidak
        # berubah saat baris lain dihapus atau disisipkan. Catatan ini in-memory
        # (dibangun ulang secara deterministik untuk frame lama); yang durabel
        # adalah id yang ikut tercatat di payload operasi struktural.
        self._frame_row_ids: dict[str, list[str]] = {}
        self._frame_row_seq: dict[str, int] = {}
        self._delta_callback: DeltaCallback | None = None
        self._bulk_mode = bulk_mode
        logger.info("WriteNode initialized columns=%s rows=%d bulk_mode=%s", list(self._df.columns), len(self._df.index), bulk_mode)

    def set_delta_callback(self, callback: DeltaCallback) -> None:
        """Register async/non-blocking delta callback for read node sync."""
        self._delta_callback = callback

    def apply(self, transaction: Transaction) -> bool:
        """Apply transaction operations atomically with rollback on failure."""
        logger.debug("WriteNode.apply start tx_id=%s ops=%d bulk_mode=%s", transaction.tx_id, len(transaction.operations), self._bulk_mode)
        with self._lock:
            # Snapshot only the columns touched by this transaction — not the entire shared DataFrame.
            # Rolling back means restoring just those series, which is orders of magnitude cheaper
            # when many DFrame columns coexist in _df.
            touched_cols: set[str] = set()
            for op in transaction.operations:
                if getattr(op, "kind", "set_cell") != "set_cell":
                    # Operasi struktural tidak punya cell_id sel; kolomnya
                    # disimpan terpisah di frame_snapshots di bawah.
                    continue
                col, row_idx = self._parse_cell_id(op.cell_id)
                if row_idx >= 0:
                    touched_cols.add(col)
            col_snapshots: dict[str, Any] = {
                c: self._df[c].copy() for c in touched_cols if c in self._df.columns
            }
            # Snapshot per frame untuk operasi struktural: cukup kolom milik frame
            # itu, bukan seluruh _df yang dipakai bersama banyak frame.
            frame_snapshots: dict[str, tuple[dict[str, Any], int | None]] = {}
            for op in transaction.operations:
                if getattr(op, "kind", "set_cell") == "set_cell":
                    continue
                fid = str((op.payload or {}).get("frame_id") or "")
                if not fid or fid in frame_snapshots:
                    continue
                prefix = f"{fid}::"
                frame_snapshots[fid] = (
                    {c: self._df[c].copy() for c in self._df.columns if c.startswith(prefix)},
                    self._frame_rows.get(fid),
                )
            orig_len = len(self._df.index)
            try:
                delta: list[dict[str, Any]] = []
                for op in transaction.operations:
                    if getattr(op, "kind", "set_cell") != "set_cell":
                        # Perubahan bentuk frame: satu operasi, bukan ribuan sel.
                        self._apply_structural(op)
                        delta.append(op.to_dict())
                        continue
                    col, row_idx = self._parse_cell_id(op.cell_id)
                    # Skip bulk operation markers — not real cell writes
                    if row_idx < 0:
                        continue
                    self._ensure_row(row_idx)
                    if col not in self._df.columns:
                        self._df[col] = [None] * len(self._df.index)
                    current_value = self._df.at[row_idx, col]
                    if not self._values_match(current_value, op.old_value):
                        logger.warning(
                            "WriteNode.apply optimistic conflict tx_id=%s cell=%s current=%r expected=%r",
                            transaction.tx_id,
                            op.cell_id,
                            current_value,
                            op.old_value,
                        )
                        self._rollback(col_snapshots, orig_len, frame_snapshots)
                        return False
                    try:
                        self._df.at[row_idx, col] = op.new_value
                    except (TypeError, ValueError) as _dtype_err:
                        # Mixed type: coerce column to object and retry
                        import pandas.errors as _pd_errs
                        if not isinstance(_dtype_err, _pd_errs.LossySetitemError) and str(self._df[col].dtype) == "object":
                            raise
                        self._df[col] = self._df[col].astype(object)
                        self._df.at[row_idx, col] = op.new_value
                    except Exception:
                        self._df[col] = self._df[col].astype(object)
                        self._df.at[row_idx, col] = op.new_value
                        # Mixed type: coerce column to object and retry
                        if str(self._df[col].dtype) != "object":
                            self._df[col] = self._df[col].astype(object)
                            self._df.at[row_idx, col] = op.new_value
                        else:
                            raise
                    delta.append(op.to_dict())
                # Catat jumlah baris milik tiap frame yang tersentuh transaksi ini.
                # Beberapa frame berbagi satu _df, jadi tanpa catatan ini frame
                # pendek terbaca sepanjang frame terpanjang (berisi null).
                for op in transaction.operations:
                    if getattr(op, "kind", "set_cell") != "set_cell":
                        continue  # pencatatan barisnya ditangani _apply_structural
                    col, row_idx = self._parse_cell_id(op.cell_id)
                    if row_idx < 0 or "::" not in col:
                        continue
                    self._catat_baris(col.split("::", 1)[0], row_idx + 1)
                self._version += 1
            except Exception as e:
                self._rollback(col_snapshots, orig_len, frame_snapshots)
                logger.exception("WriteNode.apply failed tx_id=%s error=%s", transaction.tx_id, e)
                return False

        if self._delta_callback is not None:
            logger.debug("WriteNode.apply dispatching delta callback tx_id=%s version=%d ops=%d", transaction.tx_id, self._version, len(delta))
            cb, tx_id, ver = self._delta_callback, transaction.tx_id, self._version
            WriteNode._callback_pool.submit(cb, tx_id, delta, ver)
        logger.info("WriteNode.apply success tx_id=%s version=%d", transaction.tx_id, self._version)
        return True

    def get(self, cell_id: str) -> Any:
        """Read latest cell value directly from write node."""
        with self._lock:
            col, row_idx = self._parse_cell_id(cell_id)
            if col not in self._df.columns or row_idx >= len(self._df.index):
                logger.debug("WriteNode.get missing cell=%s", cell_id)
                return None
            val = self._df.at[row_idx, col]
            logger.debug("WriteNode.get cell=%s value=%s", cell_id, val)
            return val

    def snapshot(self) -> pd.DataFrame:
        """Return deep copy of current write dataframe."""
        with self._lock:
            logger.debug("WriteNode.snapshot called returning rows=%d cols=%d", len(self._df.index), len(self._df.columns))
            return self._df.copy(deep=True)

    def to_parquet(self, path: str | Path) -> Path:
        """Persist current write-node state to a parquet file for MinIO snapshot support.

        Returns the path written to.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._df.to_parquet(target)
        logger.info("WriteNode.to_parquet path=%s rows=%d cols=%d", target, len(self._df.index), len(self._df.columns))
        return target

    def from_parquet(self, path: str | Path) -> None:
        """Restore write-node state from a parquet snapshot.

        Replaces the in-memory DataFrame entirely.
        """
        target = Path(path)
        if not target.exists():
            logger.info("WriteNode.from_parquet: snapshot not found at %s, skipping", target)
            return
        frame = pd.read_parquet(target)
        with self._lock:
            self._df = frame
            self._version += 1
        logger.info("WriteNode.from_parquet path=%s rows=%d cols=%d version=%d", target, len(frame.index), len(frame.columns), self._version)

    def read_fresh_lazy(
        self, columns: list[str] | None = None, chunk_size: int = 1000
    ):
        """Yield DataFrame chunks lazily from the current write dataframe.

        Args:
            columns: Optional list of columns to include. If None, all columns are included.
            chunk_size: Number of rows per chunk (default: 1000).
        Yields:
            pandas.DataFrame: chunk of up to chunk_size rows.
        """
        # Take a shallow reference copy inside the lock, then release immediately.
        # Holding _lock across yield would block all writes for the entire iteration.
        with self._lock:
            df = self._df if columns is None else self._df[columns]
            df = df.copy()  # snapshot: shallow copy is enough; columns are immutable strings
        total_rows = len(df)
        for start in range(0, total_rows, chunk_size):
            chunk = df.iloc[start : start + chunk_size].copy()
            logger.debug("WriteNode.read_fresh_lazy chunk start=%d size=%d", start, len(chunk))
            yield chunk

    def _rollback(
        self,
        col_snapshots: dict[str, Any],
        orig_len: int,
        frame_snapshots: dict[str, tuple[dict[str, Any], int | None]] | None = None,
    ) -> None:
        # Trim any rows added during the failed transaction.
        if len(self._df.index) > orig_len:
            self._df = self._df.iloc[:orig_len]
        for col, series in col_snapshots.items():
            self._df[col] = series
        # Pulihkan kolom per frame untuk operasi struktural: kolom baru dibuang,
        # kolom yang berubah dikembalikan, dan catatan barisnya direset.
        for fid, (snapshot, baris_lama) in (frame_snapshots or {}).items():
            prefix = f"{fid}::"
            for col in [c for c in self._df.columns if c.startswith(prefix)]:
                if col not in snapshot:
                    self._df = self._df.drop(columns=[col])
            for col, series in snapshot.items():
                self._df[col] = series.reindex(self._df.index)
            if baris_lama is None:
                self._frame_rows.pop(fid, None)
            else:
                self._frame_rows[fid] = baris_lama

    def _apply_structural(self, op: Any) -> None:
        """Terapkan satu operasi struktural pada frame pemiliknya.

        Fase 0 mendukung:
        - insert_rows: menambah baris DI AKHIR frame (payload: count, default)
        - delete_rows: menghapus baris berdasarkan indeks (payload: rows)
        - insert_column: menambah kolom pada posisi tertentu (payload: name, at, values)
        - delete_column: menghapus kolom (payload: name)
        """
        payload = dict(op.payload or {})
        frame_id = str(payload.get("frame_id") or "")
        if not frame_id:
            raise ValueError("operasi struktural butuh payload.frame_id")
        prefix = f"{frame_id}::"
        kolom_frame = [c for c in self._df.columns if c.startswith(prefix)]
        baris_kini = self._frame_rows.get(frame_id)
        if baris_kini is None:
            baris_kini = self._extent_frame(frame_id)

        if op.kind == "insert_rows":
            jumlah = int(payload.get("count", 0))
            if jumlah <= 0:
                return
            posisi = max(0, min(int(payload.get("at", baris_kini)), baris_kini))
            target = baris_kini + jumlah
            self._ensure_row(target - 1)
            ids = self._pastikan_row_ids(frame_id, baris_kini)
            if posisi < baris_kini:
                # Sisip di tengah: geser baris [posisi, baris_kini) turun sebanyak
                # jumlah baris baru. Nilainya tetap, posisinya yang berubah.
                for col in kolom_frame:
                    seri = self._df[col].iloc[:baris_kini]
                    gabung = pd.concat([
                        seri.iloc[:posisi],
                        pd.Series([None] * jumlah, index=range(posisi, posisi + jumlah)),
                        seri.iloc[posisi:],
                    ])
                    gabung.index = range(len(gabung))
                    self._df[col] = gabung.reindex(self._df.index)
            nilai = payload.get("default")
            if nilai is not None:
                for col in kolom_frame:
                    self._df.loc[posisi : posisi + jumlah - 1, col] = nilai
            # Identitas baris baru disisipkan di posisi yang sama.
            ids[posisi:posisi] = [self._next_row_id(frame_id) for _ in range(jumlah)]
            self._frame_rows[frame_id] = target
            return

        if op.kind == "delete_rows":
            ids = self._pastikan_row_ids(frame_id, baris_kini)
            # Baris boleh ditunjuk lewat posisi (rows) atau identitas (row_ids).
            dari_id = {ids.index(rid) for rid in (payload.get("row_ids") or []) if rid in ids}
            baris = sorted(dari_id | {
                int(i) for i in (payload.get("rows") or []) if 0 <= int(i) < baris_kini
            })
            if not baris or not kolom_frame:
                return
            for col in kolom_frame:
                seri = self._df[col].iloc[:baris_kini].drop(index=baris)
                seri.index = range(len(seri))
                # Panjang _df tetap; sisa di ekor menjadi kosong dan tidak
                # terlihat karena catatan baris frame ikut menyusut.
                self._df[col] = seri.reindex(self._df.index)
            # Catat identitas yang hilang agar pemulihan tidak bergantung posisi.
            payload["deleted_row_ids"] = [ids[i] for i in baris]
            op.payload = payload
            for i in sorted(baris, reverse=True):
                ids.pop(i)
            # Catatan baris frame harus bisa menyusut; _catat_baris hanya menumbuhkan.
            self._frame_rows[frame_id] = max(0, baris_kini - len(baris))
            return

        if op.kind == "insert_column":
            nama = str(payload.get("name") or "")
            if not nama:
                raise ValueError("insert_column butuh payload.name")
            namespaced = f"{prefix}{nama}"
            if namespaced in self._df.columns:
                raise ValueError(f"kolom {nama!r} sudah ada di frame {frame_id[:12]}")
            nilai = payload.get("values")
            if nilai is None:
                seri = pd.Series([None] * baris_kini, index=range(baris_kini), dtype=object)
            else:
                daftar = list(nilai)[:baris_kini]
                seri = pd.Series(daftar, index=range(len(daftar)))
            seri = seri.reindex(self._df.index)
            self._df[namespaced] = seri
            # Susun ulang urutan kolom agar posisi `at` dihormati.
            at = payload.get("at")
            if at is not None:
                posisi = max(0, min(int(at), len(kolom_frame)))
                urutan = list(self._df.columns)
                urutan.remove(namespaced)
                if posisi < len(kolom_frame):
                    urutan.insert(urutan.index(kolom_frame[posisi]), namespaced)
                else:
                    indeks_akhir = urutan.index(kolom_frame[-1]) + 1 if kolom_frame else len(urutan)
                    urutan.insert(indeks_akhir, namespaced)
                self._df = self._df[urutan]
            return

        if op.kind == "delete_column":
            nama = str(payload.get("name") or "")
            namespaced = f"{prefix}{nama}"
            if namespaced in self._df.columns:
                self._df = self._df.drop(columns=[namespaced])
            return

        raise ValueError(f"jenis operasi struktural tidak dikenal: {op.kind}")

    def _ensure_row(self, row_idx: int) -> None:
        target_len = row_idx + 1
        if len(self._df.index) < target_len:
            self._df = self._df.reindex(range(target_len))

    def _extent_frame(self, frame_id: str) -> int:
        """Perkiraan jumlah baris frame dari kolom namespaced yang sudah ada.

        Dipakai saat catatan baris frame belum ada (mis. frame hasil restore),
        supaya pencatatan tidak justru memotong baris yang sudah ada.
        """
        prefix = f"{frame_id}::"
        kolom = [c for c in self._df.columns if c.startswith(prefix)]
        if not kolom:
            return 0
        terisi = self._df[kolom].notna().any(axis=1)
        if not terisi.any():
            return 0
        return int(terisi[terisi].index[-1]) + 1

    def _catat_baris(self, frame_id: str, baris: int) -> None:
        """Catat jumlah baris MILIK satu frame. Lock harus sudah dipegang pemanggil."""
        if not frame_id or baris <= 0:
            return
        lama = self._frame_rows.get(frame_id)
        if lama is None:
            # Belum ada catatan (mis. frame hasil restore): ambil luas baris yang
            # sudah ada supaya pencatatan tidak memotong baris lama.
            self._frame_rows[frame_id] = max(self._extent_frame(frame_id), baris)
            return
        if baris > lama:
            self._frame_rows[frame_id] = baris

    def _pastikan_row_ids(self, frame_id: str, jumlah: int) -> list[str]:
        """Daftar id baris frame, dibangun deterministik bila belum ada.

        Lock harus sudah dipegang pemanggil.
        """
        ids = self._frame_row_ids.setdefault(frame_id, [])
        seq = self._frame_row_seq.get(frame_id, 0)
        while len(ids) < jumlah:
            ids.append(f"r{seq}")
            seq += 1
        self._frame_row_seq[frame_id] = max(seq, len(ids))
        return ids

    def _next_row_id(self, frame_id: str) -> str:
        seq = self._frame_row_seq.get(frame_id, 0)
        self._frame_row_seq[frame_id] = seq + 1
        return f"r{seq}"

    def row_ids(self, frame_id: str) -> list[str]:
        """Salinan daftar id baris frame (posisi ke-n = id baris ke-n)."""
        with self._lock:
            baris = self._frame_rows.get(frame_id)
            if baris is None:
                baris = self._extent_frame(frame_id)
            return list(self._pastikan_row_ids(frame_id, baris))

    @staticmethod
    def _values_match(current_value: Any, expected_value: Any) -> bool:
        if pd.isna(current_value) and expected_value is None:
            return True
        if pd.isna(expected_value) and current_value is None:
            return True
        if pd.isna(current_value) and pd.isna(expected_value):
            return True
        return current_value == expected_value

    @staticmethod
    def _parse_cell_id(cell_id: str) -> tuple[str, int]:
        if cell_id.endswith("::__bulk_init__") or cell_id.endswith("::__bulk_chunk__"):
            return ("__bulk__", -1)
        if "_" not in cell_id:
            raise ValueError(f"Invalid cell_id format: {cell_id}")
        col, row = cell_id.rsplit("_", 1)
        if not col or not row.isdigit():
            raise ValueError(f"Invalid cell_id format: {cell_id}")
        return col, int(row)
