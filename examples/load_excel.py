# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""
Example: load an Excel file (.xlsx) into a hiveframe DFrame.

Usage:
    python examples/load_excel.py [--path PATH] [--use-cluster --node-id ID --port PORT --nats-url URL]

If the Excel file does not exist, this script writes a small sample file and
then demonstrates loading it into a DFrame.

Note: Excel support requires `openpyxl` to be installed to write/read .xlsx.
For multi-process cluster mode you should run a NATS server and (optionally) have
aioquic installed for QUIC transport.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import asyncio
import pandas as pd
import hiveframe as hf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="/tmp/employee_data.xlsx", help="Excel file path to load")
    p.add_argument("--use-cluster", action="store_true", help="Connect to a running cluster via NATS/QUIC")
    p.add_argument("--node-id", default="writer-1", help="Node id to register for this process when using cluster")
    p.add_argument("--host", default="127.0.0.1", help="Host to bind for this node when using cluster")
    p.add_argument("--port", type=int, default=19000, help="Port to bind for this node when using cluster")
    p.add_argument("--nats-url", default="nats://127.0.0.1:4222", help="NATS URL for registry when using cluster")
    p.add_argument(
        "--registry-backend",
        choices=["memory", "nats"],
        default="nats",
        help="Registry backend to use when connecting to cluster (default: nats)",
    )
    p.add_argument(
        "--transport-backend",
        choices=["memory", "quic"],
        default="quic",
        help="Transport backend when using cluster (default: quic)",
    )
    return p.parse_args()


async def main_async(path: Path, use_cluster: bool, node_id: str, host: str, port: int, nats_url: str, registry_backend: str, transport_backend: str) -> None:
    runtime = None
    try:
        # Create sample Excel if missing
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

        # Read Excel with pandas and convert to hiveframe DFrame
        pd_df = pd.read_excel(path)
        print("Pandas read (head):")
        print(pd_df.head())

        data = pd_df.to_dict(orient="list")

        if use_cluster:
            # Import here to avoid module-level circular import and to keep
            # cluster-specific dependencies local to this branch.
            from core.cluster_runtime import ClusterRuntime, RuntimeConfig

            config = RuntimeConfig(
                node_id=node_id,
                role="write",
                host=host,
                port=port,
                nats_url=nats_url,
                registry_backend=registry_backend,
                transport_backend=transport_backend,
                enable_cluster=True,
            )

            runtime = ClusterRuntime(config)
            await runtime.start()
            print(f"Connected to cluster as node {node_id} (nats={nats_url})")

            # Create a DFrame that is backed by the runtime/coordinator
            dframe = hf.DFrame.from_runtime(runtime, data)
        else:
            dframe = hf.DFrame(data)

        print("DFrame head:")
        print(dframe.head())

        # Choose a text column to normalize (avoid KeyError if Excel has different column names)
        candidates = [c for c in pd_df.columns if c.lower() in ("name", "title")]
        if candidates:
            text_col = candidates[0]
        else:
            # fallback: first object (string) dtype column
            obj_cols = [c for c, dt in pd_df.dtypes.items() if dt == object]
            text_col = obj_cols[0] if obj_cols else pd_df.columns[0]

        print(f"Normalizing column: {text_col}")

        # Simple transform and persist as parquet via DFrame API
        dframe[text_col] = [str(n).upper() if pd.notna(n) else n for n in pd_df[text_col].tolist()]
        p = dframe.to_persistent("employee_data_from_excel")
        print(f"Persisted DFrame to: {p}")

        # In async contexts use the async global read helper
        merged = await dframe.read_fresh_global_async()
        print("Merged global snapshot:")
        print(merged)

    finally:
        # Best-effort cleanup when using cluster: stop heartbeat manager so the
        # node deregisters cleanly. A full stop() API would be nicer.
        if runtime is not None:
            try:
                await runtime.heartbeat.stop()
            except Exception:
                pass


def main() -> None:
    args = parse_args()
    asyncio.run(
        main_async(
            Path(args.path),
            args.use_cluster,
            args.node_id,
            args.host,
            args.port,
            args.nats_url,
            args.registry_backend,
            args.transport_backend,
        )
    )


if __name__ == "__main__":
    main()
