# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""
Example: connect to a running hiveframe cluster node started via `examples/start_cluster.py`.

Workflow:
1) Start a writer node (in another terminal):
   python examples/start_cluster.py --role write --node-id writer-1 --port 19000 --registry-backend nats --transport-backend quic

   (This requires a running NATS server and aioquic installed for real QUIC transport.)

2) Start your application that creates/prints the `frame_id` on the writer (or note the frame_id from logs).
   The frame_id is available on any DFrame as `dframe._frame_id`.

3) Run this script to connect as a read node and fetch the global merged snapshot for that `frame_id`:
   python examples/connect_to_cluster.py --frame-id <FRAME_ID> --node-id reader-1 --port 19001

Notes:
- Uses the same `ClusterRuntime` wiring as `start_cluster.py` but runs in a separate process.
- For local in-memory registry/transport, separate processes cannot see each other. Use `--registry-backend nats` and a real transport (QUIC) for multi-process cluster.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from core.cluster_runtime import ClusterRuntime, RuntimeConfig


async def main_async(frame_id: str, node_id: str, port: int, nats_url: str, host: str) -> None:
    config = RuntimeConfig(node_id=node_id, role="read", host=host, port=port, nats_url=nats_url, enable_cluster=True)
    runtime = ClusterRuntime(config)
    await runtime.start()
    print(f"Connected to cluster as node {node_id}. Attempting to read frame: {frame_id}")

    # Retry fetching a global snapshot for a short period in case writer hasn't registered yet
    deadline = time.time() + 10.0
    snapshot = None
    while time.time() < deadline:
        snapshot = await runtime.read_global_snapshot_for(frame_id)
        if snapshot is not None and not snapshot.empty:
            break
        await asyncio.sleep(0.5)

    if snapshot is None or snapshot.empty:
        print("No snapshot available for given frame_id (timeout).")
    else:
        print("Merged snapshot:")
        print(snapshot)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--frame-id", required=True, help="Frame ID to fetch from the cluster")
    p.add_argument("--node-id", default="reader-1", help="This node's id")
    p.add_argument("--host", default="127.0.0.1", help="Host to bind for this node")
    p.add_argument("--port", type=int, default=19001, help="Port to bind for this node")
    p.add_argument("--nats-url", default="nats://127.0.0.1:4222", help="NATS URL for registry")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args.frame_id, args.node_id, args.port, args.nats_url, args.host))


if __name__ == "__main__":
    main()
