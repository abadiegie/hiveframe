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
