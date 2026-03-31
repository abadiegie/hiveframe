# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""
Example: load an Excel file into a hiveframe DFrame while connected to an in-process cluster runtime.

This script demonstrates:
- starting a `ClusterRuntime` with enable_cluster=True (in-process)
- creating or reading an Excel file (.xlsx)
- creating a `DFrame` backed by the runtime via `DFrame.from_runtime`
- performing a simple write
- doing a global read (fan-out merge) via runtime.read_global_snapshot_for

Usage:
    python examples/load_excel_cluster.py [--path PATH]

Note: requires pandas and openpyxl installed.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import pandas as pd

from core.cluster_runtime import ClusterRuntime, RuntimeConfig
from core.dataframe import DFrame


async def main_async(path: Path) -> None:
    # Start a single-node cluster runtime in-process. In real deployments you would
    # start multiple processes/nodes. This example keeps everything local but
    # exercises the cluster RPC/fan-out codepaths.
    runtime = ClusterRuntime(RuntimeConfig(node_id="writer-1", role="write", port=19201, enable_cluster=True))
    await runtime.start()

    try:
        # Ensure an Excel file exists
        if not path.exists():
            sample = pd.DataFrame(
                {
                    "name": ["Alice", "Bob", "Charlie"],
                    "city": ["jakarta", "bandung", "surabaya"],
                    "score": [85, 92, 78],
                }
            )
            sample.to_excel(path, index=False)
            print(f"Wrote sample Excel to {path}")

        # Read Excel with pandas
        pd_df = pd.read_excel(path)
        print("Pandas read (head):")
        print(pd_df.head())

        # Create a DFrame backed by the running ClusterRuntime
        dframe = DFrame.from_runtime(runtime, pd_df.to_dict(orient="list"))
        print("DFrame head:")
        print(dframe.head())

        # Perform a local write (transactional) — normalize names
        dframe["name"] = [n.title() for n in pd_df["name"].tolist()]
        print("After write (local view):")
        print(dframe.read_fresh())

        # Demonstrate global read/fan-out: ask the runtime to merge snapshots for this frame
        merged = await runtime.read_global_snapshot_for(dframe._frame_id)
        print("Global merged snapshot (from runtime.read_global_snapshot_for):")
        print(merged)

        # Persist via DFrame API
        p = dframe.to_persistent("employee_data_cluster_from_excel")
        print(f"Persisted DFrame to: {p}")

    finally:
        # Best-effort cleanup: stop network/heartbeat/replication services.
        # ClusterRuntime does not expose a public stop() method; the components
        # (transport/replication/heartbeat) might have stop methods in real use.
        # For this example we simply allow the process to exit after awaiting
        # background coroutines to register our node; a fuller implementation
        # may add a graceful shutdown API.
        await asyncio.sleep(0.1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="/tmp/employee_data_cluster.xlsx", help="Excel file path to load")
    args = parser.parse_args()
    path = Path(args.path)

    asyncio.run(main_async(path))


if __name__ == "__main__":
    main()
