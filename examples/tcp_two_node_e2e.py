# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Minimal 2-node TCP end-to-end demo.

Run:
    python examples/tcp_two_node_e2e.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd

from core.cluster_runtime import ClusterRuntime, RuntimeConfig


async def main() -> None:
    db_path = Path(".hiveframe/tcp-two-node-demo.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    runtime_a = ClusterRuntime(
        RuntimeConfig(
            node_id="tcp-w1",
            role="write",
            host="127.0.0.1",
            port=19400,
            enable_cluster=True,
            registry_backend="sqlite",
            transport_backend="tcp",
            db_path=str(db_path),
        )
    )
    runtime_b = ClusterRuntime(
        RuntimeConfig(
            node_id="tcp-w2",
            role="write",
            host="127.0.0.1",
            port=19401,
            enable_cluster=True,
            registry_backend="sqlite",
            transport_backend="tcp",
            db_path=str(db_path),
        )
    )

    await runtime_a.start()
    await runtime_b.start()

    runtime_a.coordinator.write_node._df = pd.DataFrame({"left": [1]})
    runtime_b.coordinator.write_node._df = pd.DataFrame({"right": [2]})

    merged = await runtime_a.read_global_snapshot()
    print("Merged snapshot:")
    print(merged)

    await runtime_a.heartbeat.stop()
    await runtime_b.heartbeat.stop()
    await runtime_a.transport.close()
    await runtime_b.transport.close()
    runtime_a.registry.close()
    runtime_b.registry.close()


if __name__ == "__main__":
    asyncio.run(main())

