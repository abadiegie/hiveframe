# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

import asyncio

from core.message import Message, MessageType
from core.tcp_transport import InMemoryTCPTransport, TCPTransport


def test_tcp_transport_request_response_and_broadcast() -> None:
    async def run() -> None:
        node_a = TCPTransport(host="127.0.0.1", port=19200, node_id="node-a")
        node_b = TCPTransport(host="127.0.0.1", port=19201, node_id="node-b")
        node_c = TCPTransport(host="127.0.0.1", port=19202, node_id="node-c")

        seen: list[tuple[str, str]] = []

        async def handler_a(message: Message) -> None:
            seen.append(("a", message.type.value))

        async def handler_b(message: Message) -> None:
            seen.append(("b", message.type.value))
            if message.type == MessageType.READ_SNAPSHOT_REQUEST:
                response = Message.build(
                    message_type=MessageType.READ_SNAPSHOT_RESPONSE,
                    sender_id="node-b",
                    sender_region="local",
                    payload={
                        "request_id": message.payload["request_id"],
                        "snapshot": {"value": [42]},
                    },
                    trace_id=message.payload.get("trace_id"),
                )
                await node_b.send(message.sender_id, response)

        async def handler_c(message: Message) -> None:
            seen.append(("c", message.type.value))

        node_a.on_message(handler_a)
        node_b.on_message(handler_b)
        node_c.on_message(handler_c)

        await node_a.listen("127.0.0.1", 19200)
        await node_b.listen("127.0.0.1", 19201)
        await node_c.listen("127.0.0.1", 19202)
        try:
            await node_a.connect("127.0.0.1", 19201)
            await asyncio.sleep(0)
            assert "node-b" in node_a._connected_nodes
            assert ("b", MessageType.JOIN.value) in seen

            request = Message.build(
                message_type=MessageType.READ_SNAPSHOT_REQUEST,
                sender_id="node-a",
                sender_region="local",
                payload={"request_id": "req-1"},
            )
            response = await node_a.request("node-b", request, timeout=0.2)
            assert response is not None
            assert response.type == MessageType.READ_SNAPSHOT_RESPONSE
            assert response.payload["snapshot"] == {"value": [42]}

            heartbeat = Message.build(
                message_type=MessageType.HEARTBEAT,
                sender_id="node-a",
                sender_region="local",
                payload={"ts": 1.0},
            )
            await node_a.broadcast(heartbeat, exclude=["node-c"])
            await asyncio.sleep(0)

            assert ("b", "heartbeat") in seen
            assert ("c", "heartbeat") not in seen
        finally:
            await node_a.close()
            await node_b.close()
            await node_c.close()

    asyncio.run(run())


def test_tcp_transport_legacy_compatibility_shim() -> None:
    server = InMemoryTCPTransport(host="127.0.0.1", port=19210, node_id="legacy-server")
    client = InMemoryTCPTransport(host="127.0.0.1", port=19211, node_id="legacy-client")

    server.register_handler(
        "ping",
        lambda message: {
            "ok": True,
            "echo": message.get("payload", {}),
        },
    )
    server.start_server()

    response = client.send("127.0.0.1", 19210, {"type": "ping", "payload": {"n": 7}})

    assert response == {"ok": True, "echo": {"n": 7}}


def test_tcp_transport_legacy_shim_inside_running_event_loop() -> None:
    async def run() -> None:
        server = InMemoryTCPTransport(host="127.0.0.1", port=19212, node_id="legacy-server-async")
        client = InMemoryTCPTransport(host="127.0.0.1", port=19213, node_id="legacy-client-async")

        server.register_handler(
            "ping",
            lambda message: {
                "ok": True,
                "echo": message.get("payload", {}),
            },
        )
        server.start_server()

        response = await asyncio.to_thread(
            client.send,
            "127.0.0.1",
            19212,
            {"type": "ping", "payload": {"n": 9}},
        )

        assert response == {"ok": True, "echo": {"n": 9}}

    asyncio.run(run())
