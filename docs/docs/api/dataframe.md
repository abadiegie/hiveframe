# DFrame API

## Overview

`DFrame` is the main DataFrame abstraction in hiveframe, providing transactional, distributed, and AI-augmented operations on top of pandas.

By default, `DFrame` runs in transactional mode. You can set `transactional=False` for single-writer, throughput-oriented pipelines.

## Key Methods

- `__init__(data, schema=None, transactional=True, ...)` — Create a new DFrame
- `from_csv(path)` — Load from CSV file
- `from_excel(path)` — Load from Excel file
- `await from_csv_lazy(path, chunk_size=500, transactional=True, ...)` — Chunked lazy CSV load (memory O(chunk_size))
- `await from_excel_lazy(path, chunk_size=500, transactional=True, ...)` — Chunked lazy Excel load (openpyxl read-only)
- `from_runtime(runtime, data, frame_id=None, transactional=True)` — Attach to a cluster runtime
- `read_fresh()` — Get the latest local snapshot
- `read_fresh_global()` — Get merged snapshot from all cluster nodes
- `await read_fresh_global_async()` — Get merged snapshot from all cluster nodes (async)
- `read_fresh_lazy(chunk_size=1000)` — Iterate over local data in chunks
- `read_fresh_global_lazy(chunk_size=1000)` — Iterate over global data in chunks
- `read_fresh_global_lazy_async(chunk_size=1000)` — Iterate over global data in chunks (async)
- `checkpoint(label=None)` — Save a checkpoint
- `rollback(checkpoint_id)` — Restore to a checkpoint
- `cell_history(col, row_idx)` — Get audit trail for a cell
- `describe_for_agent()` — Build LLM context string

For runtime observability (logs, agent attempt telemetry, and audit trail patterns), see
[Guides -> Telemetry](../guides/telemetry.md).

When `transactional=False`:
- writes bypass lock-manager + WAL lifecycle,
- distributed routing and global reads still work,
- reads come from writer snapshots directly,
- read-replica sync is not provided,
- `cell_history()` returns `[]`,
- `checkpoint()` and `rollback()` raise `RuntimeError`.

Async caveats:
- `read_fresh_global()` is sync-only. Inside an active event loop, call `await read_fresh_global_async()`.
- `read_fresh_global_lazy()` is sync-only. Inside an active event loop, use `async for ... in read_fresh_global_lazy_async(...)`.
- In cluster mode with `transactional=False`, global reads come from writer snapshots directly; WAL-backed replica convergence is not available.

## Example

```python
import hiveframe as hf

df = hf.DFrame({"city": ["jakarta"]})
df["city"] = ["DKI Jakarta"]
print(df.read_fresh())
```

## Lazy Load Example

```python
import asyncio
import hiveframe as hf

async def main() -> None:
    df = await hf.DFrame.from_csv_lazy(
        "large.csv",
        chunk_size=1000,
        on_progress=lambda n: print(f"loaded {n} rows"),
    )
    print(df.shape)

asyncio.run(main())
```

## Distributed Lazy Seed (Cluster)

```python
import asyncio
import hiveframe as hf
from hiveframe.core.cluster_runtime import ClusterRuntime, RuntimeConfig

async def main() -> None:
    runtime = ClusterRuntime(
        RuntimeConfig(node_id="w1", role="write", enable_cluster=True)
    )
    await runtime.start()

    df = await hf.DFrame.from_csv_lazy(
        "large.csv",
        chunk_size=1000,
        runtime=runtime,
        distribute=True,
    )
    print(await df.read_fresh_async())

asyncio.run(main())
```

Notes:
- `distribute=False` is default (backward compatible).
- `distribute=True` requires `runtime`.
- If cluster has only one healthy write node, ingestion falls back to local chunked seed.
- `from_excel_lazy()` requires `openpyxl` (`pip install hiveframe[excel]`).

