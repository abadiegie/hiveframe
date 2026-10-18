# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Fase 3: commit berparent, ref, dan gc (pruning) checkpoint.

Checkpoint kini dipandang sebagai commit pada sebuah rantai: punya commit_id,
parent_commit, penanda auto, dan bisa dipangkas tanpa memutus rantai delta.
"""

from __future__ import annotations

from core.dataframe import DFrame


def _frame(frame_id: str) -> DFrame:
    return DFrame(data={"a": [1, 2]}, frame_id=frame_id)


class TestRantaiCommit:
    def test_commit_berparent_dan_head(self) -> None:
        df = _frame("f3-rantai")
        df.checkpoint("cp-0")
        df["a"] = [2, 3]
        df.checkpoint("cp-1")

        komit = df.list_commits()
        assert [k["commit_id"] for k in komit] == ["cp-0", "cp-1"]
        assert komit[0]["parent_commit"] is None
        assert komit[1]["parent_commit"] == "cp-0"
        assert df.head_commit() == "cp-1"

    def test_resolve_ref(self) -> None:
        df = _frame("f3-ref")
        df.checkpoint("cp-0")

        assert df.resolve_ref("HEAD") == "cp-0"
        assert df.resolve_ref("cp-0") == "cp-0"
        assert df.resolve_ref("tidak-ada") is None

    def test_auto_ditandai(self) -> None:
        df = _frame("f3-auto")
        df.checkpoint("manual")
        df["a"] = [2, 3]
        df.checkpoint("otomatis", auto=True)

        auto = [k["auto"] for k in df.list_commits()]
        assert auto == [False, True]


class TestPrune:
    def test_prune_menyisakan_n_terakhir(self) -> None:
        df = _frame("f3-prune")
        for i in range(5):
            df["a"] = [i, i + 1]
            df.checkpoint(f"cp-{i}", auto=True)

        dibuang = df.prune_checkpoints(keep_last=2, keep_named=False)

        assert dibuang == 3
        assert [k["commit_id"] for k in df.list_commits()] == ["cp-3", "cp-4"]

    def test_rantai_tetap_utuh_setelah_prune(self) -> None:
        df = _frame("f3-utuh")
        for i in range(5):
            df["a"] = [i, i + 1]
            df.checkpoint(f"cp-{i}", auto=True)

        df.prune_checkpoints(keep_last=2, keep_named=False)

        # Elemen tertua dimekarkan jadi penuh sehingga delta sesudahnya tetap bisa
        # disusun tanpa induk yang sudah dibuang.
        assert df.checkpoint_snapshot(df.head_commit())["a"].tolist() == [4, 5]

    def test_checkpoint_manual_lama_dipertahankan(self) -> None:
        df = _frame("f3-manual")
        df.checkpoint("titik-penting")
        for i in range(5):
            df["a"] = [i, i + 1]
            df.checkpoint(f"auto-{i}", auto=True)

        df.prune_checkpoints(keep_last=2, keep_named=True)

        ids = [k["commit_id"] for k in df.list_commits()]
        assert "titik-penting" in ids
        # Checkpoint manual itu sudah dimekarkan sehingga mandiri.
        entri = df._checkpoints["titik-penting"]
        assert entri.get("kind") == "full"
        assert df.checkpoint_snapshot("titik-penting")["a"].tolist() == [1, 2]

    def test_auto_prune_saat_checkpoint_otomatis_menumpuk(self) -> None:
        df = _frame("f3-auto-prune")
        batas = DFrame._CHECKPOINT_AUTO_MAX
        for i in range(batas + 5):
            df["a"] = [i, i + 1]
            df.checkpoint(f"auto-{i}", auto=True)

        # Pruning otomatis menjaga jumlah checkpoint otomatis tetap di batas.
        jumlah_auto = sum(1 for k in df.list_commits() if k["auto"])
        assert jumlah_auto <= batas
        assert df.checkpoint_snapshot(df.head_commit())["a"].tolist() == [batas + 4, batas + 5]

    def test_prune_tidak_membuang_bila_di_bawah_batas(self) -> None:
        df = _frame("f3-tidak-prune")
        df.checkpoint("cp-0")
        df["a"] = [2, 3]
        df.checkpoint("cp-1")

        assert df.prune_checkpoints(keep_last=10, keep_named=True) == 0
        assert len(df.list_commits()) == 2
