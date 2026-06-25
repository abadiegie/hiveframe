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
