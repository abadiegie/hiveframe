# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

from core.message import Message, MessageType


def test_message_roundtrip() -> None:
    msg = Message.build(
        message_type=MessageType.HEARTBEAT,
        sender_id="node-a",
        sender_region="ap-southeast-1",
        payload={"x": 1, "ok": True},
    )
    blob = msg.serialize()
    decoded = Message.deserialize(blob)

    assert decoded.type == MessageType.HEARTBEAT
    assert decoded.sender_id == "node-a"
    assert decoded.sender_region == "ap-southeast-1"
    assert decoded.payload["x"] == 1
    assert decoded.payload["ok"] is True


def test_seed_chunk_message_roundtrip() -> None:
    msg = Message.build(
        message_type=MessageType.SEED_CHUNK,
        sender_id="writer-a",
        sender_region="ap-southeast-1",
        payload={
            "frame_id": "frame-1",
            "row_offset": 100,
            "data": {"x": [1, 2, 3]},
        },
    )

    decoded = Message.deserialize(msg.serialize())

    assert decoded.type == MessageType.SEED_CHUNK
    assert decoded.payload["frame_id"] == "frame-1"
    assert int(decoded.payload["row_offset"]) == 100
    assert decoded.payload["data"]["x"] == [1, 2, 3]

