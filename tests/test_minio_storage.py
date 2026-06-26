# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Tests for MinIO data directory backend."""

from __future__ import annotations

import json
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

from core.minio_storage import MinioDataDir, create_data_dir


# ---------------------------------------------------------------------------
# Fake MinIO client for unit tests (no real MinIO required)
# ---------------------------------------------------------------------------


class _FakeMinioClient:
    """In-memory fake MinIO client that behaves like the real minio.Minio API."""

    def __init__(self) -> None:
        self._buckets: dict[str, dict[str, bytes]] = {}
        self._bucket_regions: dict[str, str] = {}

    def bucket_exists(self, bucket_name: str) -> bool:
        return bucket_name in self._buckets

    def make_bucket(self, bucket_name: str, location: str = "us-east-1") -> None:
        self._buckets.setdefault(bucket_name, {})
        self._bucket_regions[bucket_name] = location

    def fput_object(self, bucket_name: str, object_name: str, file_path: str) -> None:
        data = Path(file_path).read_bytes()
        self._buckets.setdefault(bucket_name, {})[object_name] = data

    def fget_object(self, bucket_name: str, object_name: str, file_path: str) -> None:
        data = self._buckets.get(bucket_name, {}).get(object_name, b"")
        Path(file_path).write_bytes(data)

    def put_object(
        self,
        bucket_name: str,
        object_name: str,
        data: BytesIO,
        length: int,
    ) -> None:
        _ = length
        self._buckets.setdefault(bucket_name, {})[object_name] = data.read()

    def get_object(self, bucket_name: str, object_name: str):
        data = self._buckets.get(bucket_name, {}).get(object_name, b"")

        class _FakeResponse:
            def read(self) -> bytes:
                return data

            def close(self) -> None:
                pass

            def release_conn(self) -> None:
                pass

        return _FakeResponse()

    def list_objects(
        self,
        bucket_name: str,
        prefix: str = "",
        recursive: bool = False,
    ) -> list[Any]:
        _ = recursive

        class _FakeObject:
            def __init__(self, name: str, size: int) -> None:
                self.object_name = name
                self.size = size

        bucket = self._buckets.get(bucket_name, {})
        return [
            _FakeObject(name, len(data))
            for name, data in bucket.items()
            if name.startswith(prefix)
        ]

    def remove_object(self, bucket_name: str, object_name: str) -> None:
        self._buckets.get(bucket_name, {}).pop(object_name, None)

    def stat_object(self, bucket_name: str, object_name: str) -> Any:
        """Return a fake stat result or raise if object not found."""
        bucket = self._buckets.get(bucket_name, {})
        if object_name not in bucket:
            raise Exception("Object does not exist")
        data = bucket[object_name]

        class _FakeStat:
            pass

        stat = _FakeStat()
        stat.object_name = object_name  # type: ignore[attr-defined]
        stat.size = len(data)  # type: ignore[attr-defined]
        return stat


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_minio_upload_download_bytes() -> None:
    client = _FakeMinioClient()
    dd = MinioDataDir(
        endpoint="localhost:9000",
        access_key="ak",
        secret_key="sk",
        bucket="test-bucket",
        minio_client=client,
    )

    obj = dd.upload_bytes(b"hello world", "data/hello.txt")
    assert "test-bucket" in client._buckets
    assert client._buckets["test-bucket"].get(obj) == b"hello world"

    result = dd.download_bytes("data/hello.txt")
    assert result == b"hello world"


def test_minio_upload_download_file(tmp_path: Path) -> None:
    client = _FakeMinioClient()
    dd = MinioDataDir(
        endpoint="localhost:9000",
        access_key="ak",
        secret_key="sk",
        bucket="test-bucket",
        minio_client=client,
    )

    local_file = tmp_path / "sample.parquet"
    local_file.write_bytes(b"parquet-data")

    dd.upload_file(local_file, "frames/sample.parquet")
    assert len(client._buckets["test-bucket"]) == 1

    restored = tmp_path / "restored.parquet"
    dd.download_file("frames/sample.parquet", restored)
    assert restored.read_bytes() == b"parquet-data"


def test_minio_list_objects() -> None:
    client = _FakeMinioClient()
    dd = MinioDataDir(
        endpoint="localhost:9000",
        access_key="ak",
        secret_key="sk",
        bucket="test-bucket",
        minio_client=client,
    )

    dd.upload_bytes(b"a", "data/a.txt")
    dd.upload_bytes(b"b", "data/b.txt")
    dd.upload_bytes(b"c", "other/c.txt")

    all_objects = dd.list_objects()
    assert len(all_objects) == 3

    data_objects = dd.list_objects("data")
    assert len(data_objects) == 2


def test_minio_delete_object() -> None:
    client = _FakeMinioClient()
    dd = MinioDataDir(
        endpoint="localhost:9000",
        access_key="ak",
        secret_key="sk",
        bucket="test-bucket",
        minio_client=client,
    )

    dd.upload_bytes(b"data", "temp/file.txt")
    assert len(dd.list_objects()) == 1

    dd.delete_object("temp/file.txt")
    assert len(dd.list_objects()) == 0


def test_minio_sync_restore_db(tmp_path: Path) -> None:
    client = _FakeMinioClient()
    dd = MinioDataDir(
        endpoint="localhost:9000",
        access_key="ak",
        secret_key="sk",
        bucket="test-bucket",
        minio_client=client,
    )

    db_path = tmp_path / "registry.db"
    db_path.write_bytes(b"sqlite-data-here")

    dd.sync_db(db_path, "registry/node1.db")
    objects = dd.list_objects("registry")
    assert len(objects) == 1

    # Simulate loss of local DB
    db_path.unlink()
    assert not db_path.exists()

    dd.restore_db(db_path, "registry/node1.db")
    assert db_path.exists()
    assert db_path.read_bytes() == b"sqlite-data-here"


def test_minio_get_metrics() -> None:
    client = _FakeMinioClient()
    dd = MinioDataDir(
        endpoint="localhost:9000",
        access_key="ak",
        secret_key="sk",
        bucket="test-bucket",
        minio_client=client,
    )

    dd.upload_bytes(b"x" * 100, "a.txt")
    dd.upload_bytes(b"y" * 50, "b.txt")

    metrics = dd.get_metrics()
    assert metrics["bucket"] == "test-bucket"
    assert metrics["endpoint"] == "localhost:9000"
    assert metrics["total_objects"] == 2
    assert metrics["total_size_bytes"] == 150


def test_minio_prefix_isolation() -> None:
    """Verify that objects from different prefixes don't collide."""
    client = _FakeMinioClient()

    dd_a = MinioDataDir(
        endpoint="localhost:9000",
        access_key="ak",
        secret_key="sk",
        bucket="shared-bucket",
        prefix="tenant-a/",
        minio_client=client,
    )
    dd_b = MinioDataDir(
        endpoint="localhost:9000",
        access_key="ak",
        secret_key="sk",
        bucket="shared-bucket",
        prefix="tenant-b/",
        minio_client=client,
    )

    dd_a.upload_bytes(b"data-a", "file.txt")
    dd_b.upload_bytes(b"data-b", "file.txt")

    assert dd_a.download_bytes("file.txt") == b"data-a"
    assert dd_b.download_bytes("file.txt") == b"data-b"

    a_objects = dd_a.list_objects()
    b_objects = dd_b.list_objects()
    assert len(a_objects) == 1
    assert len(b_objects) == 1


def test_create_data_dir_from_env_minio(monkeypatch) -> None:
    monkeypatch.setenv("HIVEFRAME_DATA_DIR_BACKEND", "minio")
    monkeypatch.setenv("HIVEFRAME_MINIO_ENDPOINT", "minio.example.com:9000")
    monkeypatch.setenv("HIVEFRAME_MINIO_ACCESS_KEY", "mykey")
    monkeypatch.setenv("HIVEFRAME_MINIO_SECRET_KEY", "mysecret")
    monkeypatch.setenv("HIVEFRAME_MINIO_BUCKET", "mybucket")
    monkeypatch.setenv("HIVEFRAME_MINIO_SECURE", "true")
    monkeypatch.setenv("HIVEFRAME_MINIO_REGION", "ap-southeast-1")
    monkeypatch.setenv("HIVEFRAME_MINIO_PREFIX", "prod/")

    import sys
    from unittest.mock import MagicMock

    # Inject a fake minio module so we don't need the real package installed.
    fake_minio = MagicMock()
    fake_minio.Minio = MagicMock(return_value=_FakeMinioClient())
    monkeypatch.setitem(sys.modules, "minio", fake_minio)

    dd = create_data_dir()
    assert dd is not None
    assert isinstance(dd, MinioDataDir)
    assert dd._endpoint == "minio.example.com:9000"
    assert dd._bucket == "mybucket"
    assert dd._secure is True
    assert dd._region == "ap-southeast-1"
    assert dd._prefix == "prod/"


def test_create_data_dir_local_returns_none(monkeypatch) -> None:
    monkeypatch.setenv("HIVEFRAME_DATA_DIR_BACKEND", "local")
    dd = create_data_dir()
    assert dd is None


def test_create_data_dir_unset_returns_none(monkeypatch) -> None:
    monkeypatch.delenv("HIVEFRAME_DATA_DIR_BACKEND", raising=False)
    dd = create_data_dir()
    assert dd is None


def test_create_data_dir_unknown_raises(monkeypatch) -> None:
    monkeypatch.setenv("HIVEFRAME_DATA_DIR_BACKEND", "gcs")
    try:
        create_data_dir()
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Unknown HIVEFRAME_DATA_DIR_BACKEND" in str(exc)


# ---------------------------------------------------------------------------
# JSON helper tests
# ---------------------------------------------------------------------------


def test_upload_download_json() -> None:
    client = _FakeMinioClient()
    dd = MinioDataDir(
        endpoint="localhost:9000",
        access_key="ak",
        secret_key="sk",
        bucket="test-bucket",
        minio_client=client,
    )

    payload = {"hello": "world", "count": 42}
    obj = dd.upload_json(payload, "data/config.json")
    assert obj in client._buckets["test-bucket"]

    result = dd.download_json("data/config.json")
    assert result == payload


def test_download_json_missing() -> None:
    client = _FakeMinioClient()
    dd = MinioDataDir(
        endpoint="localhost:9000",
        access_key="ak",
        secret_key="sk",
        bucket="test-bucket",
        minio_client=client,
    )

    result = dd.download_json("nonexistent/key.json")
    assert result is None


def test_object_exists() -> None:
    client = _FakeMinioClient()
    dd = MinioDataDir(
        endpoint="localhost:9000",
        access_key="ak",
        secret_key="sk",
        bucket="test-bucket",
        minio_client=client,
    )

    dd.upload_bytes(b"data", "present.txt")
    assert dd.object_exists("present.txt") is True
    assert dd.object_exists("missing.txt") is False


# ---------------------------------------------------------------------------
# MinioWriteAheadLog tests
# ---------------------------------------------------------------------------


def _make_fake_tx(tx_id: str = "tx-1", cell_id: str = "col_0",
                  old_value: Any = None, new_value: Any = "new-val",
                  state: str = "COMMITTED") -> Any:
    """Build a minimal Transaction-like object for WAL testing."""
    from core.transaction import Transaction, Operation, TxState

    op = Operation(
        cell_id=cell_id,
        old_value=old_value,
        new_value=new_value,
        author_type="human",
        author_id="test",
    )
    tx = Transaction(operations=[op], tx_id=tx_id)
    # Walk the state machine to the desired state
    tx.transition(TxState.VALIDATING)
    tx.transition(TxState.LOCKED)
    tx.transition(TxState.APPLYING)
    tx.transition(TxState.COMMITTED)
    if state == "SYNCED":
        tx.transition(TxState.SYNCING)
        tx.transition(TxState.SYNCED)
    return tx


def test_minio_wal_append_and_get_since() -> None:
    from core.wal import MinioWriteAheadLog

    client = _FakeMinioClient()
    dd = MinioDataDir(
        endpoint="localhost:9000",
        access_key="ak",
        secret_key="sk",
        bucket="test-bucket",
        minio_client=client,
    )

    wal = MinioWriteAheadLog(dd, node_id="node-1")
    assert wal.get_metrics()["total_entries"] == 0

    tx1 = _make_fake_tx("tx-1", new_value="a")
    tx2 = _make_fake_tx("tx-2", new_value="b")
    tx3 = _make_fake_tx("tx-3", new_value="c")

    lsn1 = wal.append(tx1)
    lsn2 = wal.append(tx2)
    lsn3 = wal.append(tx3)
    assert lsn1 == 1
    assert lsn2 == 2
    assert lsn3 == 3

    # get_since from 0 returns all
    since_0 = wal.get_since(0)
    assert len(since_0) == 3

    # get_since from 1 returns last two
    since_1 = wal.get_since(1)
    assert len(since_1) == 2
    assert since_1[0]["lsn"] == 2
    assert since_1[1]["lsn"] == 3


def test_minio_wal_recovery() -> None:
    from core.wal import MinioWriteAheadLog

    client = _FakeMinioClient()
    dd = MinioDataDir(
        endpoint="localhost:9000",
        access_key="ak",
        secret_key="sk",
        bucket="test-bucket",
        minio_client=client,
    )

    # Create WAL, append entries, then simulate restart
    wal1 = MinioWriteAheadLog(dd, node_id="node-2")
    wal1.append(_make_fake_tx("tx-a", new_value="x"))
    wal1.append(_make_fake_tx("tx-b", new_value="y"))

    # Second instance should recover from MinIO
    wal2 = MinioWriteAheadLog(dd, node_id="node-2")
    metrics = wal2.get_metrics()
    assert metrics["total_entries"] == 2
    assert metrics["last_lsn"] == 2

    since = wal2.get_since(0)
    assert len(since) == 2
    assert since[0]["tx_id"] == "tx-a"
    assert since[1]["tx_id"] == "tx-b"


def test_minio_wal_compact() -> None:
    from core.wal import MinioWriteAheadLog

    client = _FakeMinioClient()
    dd = MinioDataDir(
        endpoint="localhost:9000",
        access_key="ak",
        secret_key="sk",
        bucket="test-bucket",
        minio_client=client,
    )

    wal = MinioWriteAheadLog(dd, node_id="node-3")
    for i in range(10):
        wal.append(_make_fake_tx(f"tx-{i}", new_value=str(i)))

    assert wal.get_metrics()["total_entries"] == 10

    removed = wal.compact(keep_last_n=3)
    assert removed == 7
    assert wal.get_metrics()["total_entries"] == 3

    # Remaining entries have the highest LSNs
    since = wal.get_since(0)
    assert len(since) == 3
    lsns = [e["lsn"] for e in since]
    assert lsns == [8, 9, 10]


def test_minio_wal_compact_before_lsn() -> None:
    from core.wal import MinioWriteAheadLog

    client = _FakeMinioClient()
    dd = MinioDataDir(
        endpoint="localhost:9000",
        access_key="ak",
        secret_key="sk",
        bucket="test-bucket",
        minio_client=client,
    )

    wal = MinioWriteAheadLog(dd, node_id="node-4")
    for i in range(5):
        wal.append(_make_fake_tx(f"tx-{i}", new_value=str(i)))

    removed = wal.compact_before_lsn(3)
    assert removed == 2  # lsns 1 and 2

    since = wal.get_since(0)
    assert len(since) == 3
    assert since[0]["lsn"] == 3


def test_minio_wal_get_committed() -> None:
    from core.wal import MinioWriteAheadLog

    client = _FakeMinioClient()
    dd = MinioDataDir(
        endpoint="localhost:9000",
        access_key="ak",
        secret_key="sk",
        bucket="test-bucket",
        minio_client=client,
    )

    wal = MinioWriteAheadLog(dd, node_id="node-5")
    wal.append(_make_fake_tx("tx-c1", new_value="c1", state="COMMITTED"))
    wal.append(_make_fake_tx("tx-c2", new_value="c2", state="SYNCED"))

    committed = wal.get_committed()
    assert len(committed) == 2
    assert committed[0]["tx_id"] == "tx-c1"
    assert committed[1]["tx_id"] == "tx-c2"


def test_minio_wal_cell_history() -> None:
    from core.wal import MinioWriteAheadLog

    client = _FakeMinioClient()
    dd = MinioDataDir(
        endpoint="localhost:9000",
        access_key="ak",
        secret_key="sk",
        bucket="test-bucket",
        minio_client=client,
    )

    wal = MinioWriteAheadLog(dd, node_id="node-6")
    wal.append(_make_fake_tx("tx-h1", cell_id="price_0", old_value=None, new_value=100))
    wal.append(_make_fake_tx("tx-h2", cell_id="price_0", old_value=100, new_value=110))
    wal.append(_make_fake_tx("tx-h3", cell_id="qty_0", old_value=None, new_value=5))

    history = wal.get_cell_history("price_0")
    assert len(history) == 2
    assert history[0]["old_value"] is None
    assert history[0]["new_value"] == 100
    assert history[1]["old_value"] == 100
    assert history[1]["new_value"] == 110


def test_minio_wal_get_metrics() -> None:
    from core.wal import MinioWriteAheadLog

    client = _FakeMinioClient()
    dd = MinioDataDir(
        endpoint="localhost:9000",
        access_key="ak",
        secret_key="sk",
        bucket="test-bucket",
        minio_client=client,
    )

    wal = MinioWriteAheadLog(dd, node_id="node-7")
    assert wal.get_metrics() == {"total_entries": 0, "last_lsn": 0}

    wal.append(_make_fake_tx("tx-a", new_value="a"))
    wal.append(_make_fake_tx("tx-b", new_value="b"))

    metrics = wal.get_metrics()
    assert metrics["total_entries"] == 2
    assert metrics["last_lsn"] == 2


# ---------------------------------------------------------------------------
# ReadNode MinIO integration tests
# ---------------------------------------------------------------------------


def test_read_node_persist_to_minio(tmp_path: Path) -> None:
    import pandas as pd
    from core.read_node import ReadNode

    client = _FakeMinioClient()
    dd = MinioDataDir(
        endpoint="localhost:9000",
        access_key="ak",
        secret_key="sk",
        bucket="test-bucket",
        minio_client=client,
    )

    storage_dir = tmp_path / "dframe_store"
    rn = ReadNode(storage_dir=str(storage_dir), minio_data_dir=dd, node_id="test-node")

    # Put data into cache and persist
    rn._cache = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})
    rn.persist("mydataset")

    # Verify local parquet exists
    local_path = storage_dir / "mydataset.parquet"
    assert local_path.exists()

    # Verify MinIO upload
    expected_obj = "parquet/test-node/mydataset.parquet"
    all_objects = dd.list_objects()
    full_key = dd._object_name(expected_obj)
    assert any(full_key in o for o in all_objects) or any(
        o.endswith("mydataset.parquet") for o in all_objects
    ), f"Expected parquet object in MinIO, got: {all_objects}"


def test_read_node_load_from_minio(tmp_path: Path) -> None:
    import pandas as pd
    from core.read_node import ReadNode

    client = _FakeMinioClient()
    dd = MinioDataDir(
        endpoint="localhost:9000",
        access_key="ak",
        secret_key="sk",
        bucket="test-bucket",
        minio_client=client,
    )

    storage_dir = tmp_path / "dframe_store"

    # Pre-populate: persist via one ReadNode instance
    rn1 = ReadNode(storage_dir=str(storage_dir), minio_data_dir=dd, node_id="node-a")
    rn1._cache = pd.DataFrame({"val": [10, 20]})
    rn1.persist("dataset_x")

    # Delete local file to simulate cold start / remote node
    local_path = storage_dir / "dataset_x.parquet"
    local_path.unlink()
    assert not local_path.exists()

    # Second ReadNode loads — should pull from MinIO
    rn2 = ReadNode(storage_dir=str(storage_dir), minio_data_dir=dd, node_id="node-a")
    frame = rn2.load("dataset_x")
    assert frame is not None
    assert list(frame["val"]) == [10, 20]
    assert local_path.exists()  # downloaded locally


def test_read_node_restore_all_from_minio(tmp_path: Path) -> None:
    import pandas as pd
    from core.read_node import ReadNode

    client = _FakeMinioClient()
    dd = MinioDataDir(
        endpoint="localhost:9000",
        access_key="ak",
        secret_key="sk",
        bucket="test-bucket",
        minio_client=client,
    )

    storage_dir = tmp_path / "dframe_store"
    rn1 = ReadNode(storage_dir=str(storage_dir), minio_data_dir=dd, node_id="node-b")

    # Persist multiple datasets
    rn1._cache = pd.DataFrame({"a": [1]})
    rn1.persist("ds1")

    rn1._cache = pd.DataFrame({"b": [2]})
    rn1.persist("ds2")

    # Wipe local storage
    import shutil
    shutil.rmtree(storage_dir)
    storage_dir.mkdir()

    # Restore all from MinIO
    rn2 = ReadNode(storage_dir=str(storage_dir), minio_data_dir=dd, node_id="node-b")
    count = rn2.restore_all_from_minio()
    assert count >= 2

    # Verify local files restored
    assert (storage_dir / "ds1.parquet").exists()
    assert (storage_dir / "ds2.parquet").exists()
