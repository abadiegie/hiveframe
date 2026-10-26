# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Message protocol and MessagePack serialization for cluster communication."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Any
import uuid

import hashlib
import hmac
import os
import msgpack


class MessageType(Enum):
    """Cluster message categories for discovery, health, and replication."""

    HELLO = "hello"
    HELLO_ACK = "hello_ack"
    JOIN = "join"
    MEMBERSHIP_UPDATE = "membership_update"
    ROUTE_UPDATE = "route_update"
    HEARTBEAT = "heartbeat"
    HEARTBEAT_ACK = "heartbeat_ack"
    BYE = "bye"

    DELTA = "delta"
    SEED_CHUNK = "seed_chunk"
    DELTA_ACK = "delta_ack"
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"
    READ_SNAPSHOT_REQUEST = "read_snapshot_request"
    READ_SNAPSHOT_RESPONSE = "read_snapshot_response"

    OPLOG_PUSH = "oplog_push"
    OPLOG_PUSH_RESPONSE = "oplog_push_response"
    OPLOG_PULL = "oplog_pull"
    OPLOG_PULL_RESPONSE = "oplog_pull_response"
    OPLOG_FULL_RESYNC = "oplog_full_resync"
    OPLOG_FULL_RESYNC_RESPONSE = "oplog_full_resync_response"

    PARTITION_MAP = "partition_map"
    ELECT_LEADER = "elect_leader"
    REBALANCE = "rebalance"


@dataclass(slots=True)
class Message:
    """Serializable cross-node message payload."""

    type: MessageType
    sender_id: str
    sender_region: str
    timestamp: float
    payload: dict[str, Any]

    def serialize(self, auth_secret: str | bytes | None = None) -> bytes:
        """Serialize message to MessagePack bytes, optionally HMAC-signed."""
        raw = {
            "type": self.type.value,
            "sender_id": self.sender_id,
            "sender_region": self.sender_region,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }
        body = msgpack.packb(raw, use_bin_type=True)
        secret = auth_secret or os.environ.get("HIVEFRAME_CLUSTER_SECRET", "")
        if not secret:
            return body
        signature = hmac.new(
            secret.encode("utf-8") if isinstance(secret, str) else secret,
            body,
            hashlib.sha256,
        ).hexdigest()
        return body + signature.encode("ascii")

    @classmethod
    def deserialize(cls, data: bytes, auth_secret: str | bytes | None = None) -> "Message":
        """Deserialize and verify an optionally HMAC-signed message."""
        secret = auth_secret or os.environ.get("HIVEFRAME_CLUSTER_SECRET", "")
        if secret:
            if len(data) < 64:
                raise ValueError("Signed cluster message is truncated")
            body, signature = data[:-64], data[-64:]
            expected = hmac.new(
                secret.encode("utf-8") if isinstance(secret, str) else secret,
                body,
                hashlib.sha256,
            ).hexdigest().encode("ascii")
            if not hmac.compare_digest(signature, expected):
                raise ValueError("Cluster message authentication failed")
            data = body
        raw = msgpack.unpackb(data, raw=False)
        return cls(
            type=MessageType(raw["type"]),
            sender_id=raw["sender_id"],
            sender_region=raw["sender_region"],
            timestamp=float(raw["timestamp"]),
            payload=dict(raw.get("payload", {})),
        )

    @classmethod
    def build(
        cls,
        *,
        message_type: MessageType,
        sender_id: str,
        sender_region: str,
        payload: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> "Message":
        """Convenience constructor that assigns current wall time and injects a trace_id.

        The trace_id is used to correlate messages across components. If the caller
        provides a trace_id in the payload or via the `trace_id` argument, that value
        is preserved. Otherwise a new UUID4 string is generated.
        """
        payload = payload or {}
        # Prefer explicit trace_id argument, then payload['trace_id'], else generate.
        if trace_id is not None:
            payload.setdefault("trace_id", trace_id)
        else:
            payload.setdefault("trace_id", str(uuid.uuid4()))
        return cls(
            type=message_type,
            sender_id=sender_id,
            sender_region=sender_region,
            timestamp=time.time(),
            payload=payload or {},
        )
