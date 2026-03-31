# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0
"""Raw TCP transport with MessagePack framing for local cluster deployments."""

import socket
import threading
import msgpack
import logging
from typing import Callable

logger = logging.getLogger("core.tcp_transport")

class TCPTransport:
    def __init__(self, host: str = "127.0.0.1", port: int = 19100):
        self.host = host
        self.port = port
        self._server = None
        self._handlers: dict[str, Callable[[dict], dict]] = {}
        self._running = False

    def start_server(self):
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self.host, self.port))
        self._server.listen()
        self._running = True
        logger.info(f"TCPTransport server listening on {self.host}:{self.port}")
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self):
        while self._running:
            try:
                conn, addr = self._server.accept()
                threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True).start()
            except Exception as e:
                logger.error(f"Accept failed: {e}")

    def _handle_client(self, conn, addr):
        try:
            unpacker = msgpack.Unpacker(conn, raw=False)
            for msg in unpacker:
                logger.debug(f"Received from {addr}: {msg}")
                if "type" in msg and msg["type"] in self._handlers:
                    response = self._handlers[msg["type"]](msg)
                    conn.sendall(msgpack.packb(response, use_bin_type=True))
        except Exception as e:
            logger.error(f"Client handler error: {e}")
        finally:
            conn.close()

    def register_handler(self, msg_type: str, handler: Callable[[dict], dict]):
        self._handlers[msg_type] = handler

    def send(self, host: str, port: int, message: dict) -> dict:
        with socket.create_connection((host, port)) as sock:
            sock.sendall(msgpack.packb(message, use_bin_type=True))
            unpacker = msgpack.Unpacker(sock, raw=False)
            for response in unpacker:
                logger.debug(f"Response from {host}:{port}: {response}")
                return response
        return {}
