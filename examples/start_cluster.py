# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""
start_cluster.py — Contoh menjalankan satu atau lebih node dalam cluster mode.

Jalankan writer node:
    python examples/start_cluster.py --role write --node-id writer-1 --port 19000

Jalankan read node (di terminal berbeda):
    python examples/start_cluster.py --role read --node-id reader-1 --port 19001

Opsi lengkap:
    python examples/start_cluster.py --help
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("start_cluster")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Start a hiveframe cluster node")
    p.add_argument("--node-id", required=True, help="Unique node identifier")
    p.add_argument("--role", choices=["write", "read"], required=True, help="Node role")
    p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=19000, help="Bind port (default: 19000)")
    p.add_argument("--region", default="ap-southeast-1", help="Region label")
    p.add_argument("--nats-url", default="nats://127.0.0.1:4222", help="NATS URL for registry")
    p.add_argument(
        "--registry-backend",
        choices=["memory", "nats"],
        default="memory",
        help="Registry backend (default: memory)",
    )
    p.add_argument(
        "--transport-backend",
        choices=["memory", "quic"],
        default="memory",
        help="Transport backend (default: memory)",
    )
    p.add_argument(
        "--partition-start", type=int, default=0, help="Partition range start (default: 0)"
    )
    p.add_argument(
        "--partition-end", type=int, default=1000, help="Partition range end (default: 1000)"
    )
    return p.parse_args()


async def run(args: argparse.Namespace) -> None:
    # Import here to avoid circular-import at module level.
    from hiveframe.core.cluster_runtime import ClusterRuntime, RuntimeConfig

    config = RuntimeConfig(
        node_id=args.node_id,
        role=args.role,
        host=args.host,
        port=args.port,
        region=args.region,
        nats_url=args.nats_url,
        registry_backend=args.registry_backend,
        transport_backend=args.transport_backend,
        partition_start=args.partition_start,
        partition_end=args.partition_end,
        enable_cluster=True,
        required_cluster=False,
    )

    runtime = ClusterRuntime(config)
    await runtime.start()

    logger.info(
        "Node started  id=%s  role=%s  host=%s  port=%d  region=%s  registry=%s  transport=%s",
        config.node_id,
        config.role,
        config.host,
        config.port,
        config.region,
        config.registry_backend,
        config.transport_backend,
    )

    # Keep running until SIGINT / SIGTERM.
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    logger.info("Node is running. Press Ctrl+C to stop.")
    await stop_event.wait()

    logger.info("Shutting down node %s ...", config.node_id)
    await runtime.heartbeat.stop()
    logger.info("Node %s stopped.", config.node_id)


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
