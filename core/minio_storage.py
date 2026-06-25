# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""MinIO (S3-compatible) data directory backend for shared durable storage."""

from __future__ import annotations

import logging
import os
import tempfile
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger("core.minio_storage")


class MinioDataDir:
    """MinIO-backed data directory for parquet, SQLite DBs, and WAL persistence.

    Uses the ``minio`` Python package for S3-compatible operations. Intended as a
    shared durable store across cluster nodes, with local filesystem as hot cache.
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str = "hiveframe",
        *,
        secure: bool = False,
        region: str = "us-east-1",
        prefix: str = "hiveframe/",
        minio_client: Any | None = None,
    ) -> None:
        self._lock = Lock()
        self._endpoint = endpoint
        self._bucket = bucket
        self._secure = secure
        self._region = region
        self._prefix = prefix.rstrip("/") + "/" if prefix else ""

        if minio_client is not None:
            self._client = minio_client
        else:
            try:
                from minio import Minio
            except ImportError as exc:
                raise ImportError(
                    "minio package required for MinioDataDir: pip install hiveframe[minio]"
                ) from exc
            self._client = Minio(
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure,
                region=region,
            )

        self._ensure_bucket()
        logger.info(
            "MinioDataDir initialized endpoint=%s bucket=%s prefix=%s secure=%s",
            endpoint,
            bucket,
            self._prefix,
            secure,
        )

    # -------------------------------------------------------------------------
    # Bucket management
    # -------------------------------------------------------------------------

    def _ensure_bucket(self) -> None:
        """Create bucket if it does not already exist."""
        try:
            found = self._client.bucket_exists(self._bucket)
            if not found:
                self._client.make_bucket(self._bucket, location=self._region)
                logger.info("MinioDataDir created bucket=%s region=%s", self._bucket, self._region)
            else:
                logger.debug("MinioDataDir bucket=%s already exists", self._bucket)
        except Exception as exc:
            logger.warning("MinioDataDir bucket check/creation failed: %s", exc)
            raise

    # -------------------------------------------------------------------------
    # Object name helpers
    # -------------------------------------------------------------------------

    def _object_name(self, key: str) -> str:
        """Prefix a key with the configured path prefix."""
        return f"{self._prefix}{key.lstrip('/')}"

    # -------------------------------------------------------------------------
    # Core upload / download
    # -------------------------------------------------------------------------

    def upload_file(self, local_path: str | Path, object_name: str) -> str:
        """Upload a local file to MinIO and return the object name used."""
        local = Path(local_path)
        obj = self._object_name(object_name)
        with self._lock:
            self._client.fput_object(
                bucket_name=self._bucket,
                object_name=obj,
                file_path=str(local),
            )
        logger.debug("MinioDataDir uploaded %s -> %s/%s", local, self._bucket, obj)
        return obj

    def download_file(self, object_name: str, local_path: str | Path) -> Path:
        """Download an object from MinIO to a local file path."""
        obj = self._object_name(object_name)
        local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._client.fget_object(
                bucket_name=self._bucket,
                object_name=obj,
                file_path=str(local),
            )
        logger.debug("MinioDataDir downloaded %s/%s -> %s", self._bucket, obj, local)
        return local

    def upload_bytes(self, data: bytes, object_name: str) -> str:
        """Upload raw bytes as an object."""
        obj = self._object_name(object_name)
        bio = BytesIO(data)
        with self._lock:
            self._client.put_object(
                bucket_name=self._bucket,
                object_name=obj,
                data=bio,
                length=len(data),
            )
        logger.debug("MinioDataDir uploaded %d bytes -> %s/%s", len(data), self._bucket, obj)
        return obj

    def download_bytes(self, object_name: str) -> bytes:
        """Download an object's contents as raw bytes."""
        obj = self._object_name(object_name)
        with self._lock:
            response = self._client.get_object(
                bucket_name=self._bucket,
                object_name=obj,
            )
            try:
                data = response.read()
            finally:
                response.close()
                response.release_conn()
        logger.debug("MinioDataDir downloaded %d bytes from %s/%s", len(data), self._bucket, obj)
        return data

    # -------------------------------------------------------------------------
    # Object listing and deletion
    # -------------------------------------------------------------------------

    def list_objects(self, prefix: str = "") -> list[str]:
        """List object names under the given prefix (relative to the data dir prefix)."""
        full_prefix = self._object_name(prefix)
        with self._lock:
            objects = self._client.list_objects(
                bucket_name=self._bucket,
                prefix=full_prefix,
                recursive=True,
            )
            names: list[str] = []
            for obj in objects:
                if obj.object_name:
                    names.append(obj.object_name)
            return names

    def delete_object(self, object_name: str) -> None:
        """Delete a single object."""
        obj = self._object_name(object_name)
        with self._lock:
            self._client.remove_object(
                bucket_name=self._bucket,
                object_name=obj,
            )
        logger.debug("MinioDataDir deleted %s/%s", self._bucket, obj)

    # -------------------------------------------------------------------------
    # SQLite DB sync helpers
    # -------------------------------------------------------------------------

    def sync_db(self, db_path: str | Path, object_name: str) -> str:
        """Upload a local SQLite database file to MinIO.

        Reads the file as bytes and uploads, so it works even if the DB is
        being actively written (the caller should hold appropriate locks).
        """
        local = Path(db_path)
        data = local.read_bytes()
        return self.upload_bytes(data, object_name)

    def restore_db(self, db_path: str | Path, object_name: str) -> Path:
        """Download a SQLite database from MinIO and write to local path.

        Returns the local path of the restored file.
        """
        data = self.download_bytes(object_name)
        local = Path(db_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        logger.info(
            "MinioDataDir restored DB %s <- %s/%s (%d bytes)",
            local,
            self._bucket,
            self._object_name(object_name),
            len(data),
        )
        return local

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    def get_metrics(self) -> dict[str, Any]:
        """Return operational metrics for observability."""
        with self._lock:
            try:
                objects = list(
                    self._client.list_objects(
                        bucket_name=self._bucket,
                        prefix=self._prefix,
                        recursive=True,
                    )
                )
                total_size = sum(getattr(o, "size", 0) or 0 for o in objects)
                total_objects = len(objects)
            except Exception:
                total_size = 0
                total_objects = 0
        return {
            "endpoint": self._endpoint,
            "bucket": self._bucket,
            "prefix": self._prefix,
            "total_objects": total_objects,
            "total_size_bytes": total_size,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_data_dir() -> MinioDataDir | None:
    """Create a MinIO data directory backend based on environment configuration.

    - ``HIVEFRAME_DATA_DIR_BACKEND`` unset or ``local`` → returns None (use local FS)
    - ``HIVEFRAME_DATA_DIR_BACKEND=minio`` → returns ``MinioDataDir``

    MinIO connection parameters are read from environment variables:

    - ``HIVEFRAME_MINIO_ENDPOINT`` (default: ``localhost:9000``)
    - ``HIVEFRAME_MINIO_ACCESS_KEY`` (default: ``minioadmin``)
    - ``HIVEFRAME_MINIO_SECRET_KEY`` (default: ``minioadmin``)
    - ``HIVEFRAME_MINIO_BUCKET`` (default: ``hiveframe``)
    - ``HIVEFRAME_MINIO_SECURE`` (default: ``false``)
    - ``HIVEFRAME_MINIO_REGION`` (default: ``us-east-1``)
    - ``HIVEFRAME_MINIO_PREFIX`` (default: ``hiveframe/``)
    """

    backend = os.getenv("HIVEFRAME_DATA_DIR_BACKEND", "local").strip().lower()
    if backend == "local":
        return None

    if backend == "minio":
        endpoint = os.getenv("HIVEFRAME_MINIO_ENDPOINT", "localhost:9000")
        access_key = os.getenv("HIVEFRAME_MINIO_ACCESS_KEY", "minioadmin")
        secret_key = os.getenv("HIVEFRAME_MINIO_SECRET_KEY", "minioadmin")
        bucket = os.getenv("HIVEFRAME_MINIO_BUCKET", "hiveframe")
        secure_raw = os.getenv("HIVEFRAME_MINIO_SECURE", "false").strip().lower()
        secure = secure_raw in {"1", "true", "yes", "on"}
        region = os.getenv("HIVEFRAME_MINIO_REGION", "us-east-1")
        prefix = os.getenv("HIVEFRAME_MINIO_PREFIX", "hiveframe/")

        return MinioDataDir(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            bucket=bucket,
            secure=secure,
            region=region,
            prefix=prefix,
        )

    raise ValueError(
        f"Unknown HIVEFRAME_DATA_DIR_BACKEND='{backend}'. Use local|minio."
    )
