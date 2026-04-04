# hiveframe

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-20%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
Transactional, distributed-ready pandas-compatible DataFrame engine.

**hiveframe** scales DataFrame workloads across many small machines with transactions, persistence, and AI agent support built in. No new paradigm to learn. Just `import hiveframe as hf`.

Supports single-node standalone mode and optional multi-node cluster mode with QUIC/TCP transport, NATS/SQLite registry, heartbeat, WAL-based delta replication, **global read fan-out**, **dynamic partition assignment**, and **per-DFrame namespace isolation** — multiple independent DataFrames can run on the same cluster node without overlap.

Note: `TCPTransport` now uses real asyncio TCP sockets for cluster runtime traffic. The previous process-local compatibility behavior is preserved in `InMemoryTCPTransport` for legacy sync callers/tests.

---

## Table of Contents

- [Why hiveframe?](#why-hiveframe)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Development Mode (from clone)](#development-mode-from-clone)
- [Quick Start (hiveframe import)](#quick-start-hiveframe-import)
- [Read vs Write Model](#read-vs-write-model)
- [Modes](#modes)
- [Namespace Isolation](#namespace-isolation)
- [Dynamic Partitioning](#dynamic-partitioning)
- [RuntimeConfig](#runtimeconfig)
- [Install extras](#install-extras)
- [Pandas API Coverage](#pandas-api-coverage)
- [Advanced Features](#advanced-features)
- [Testing](#testing)
- [Start Cluster](#start-cluster)
- [Usage](#usage)
- [LLM Agent Prompt](#llm-agent-prompt)
- [Contributing](#contributing)
- [License](#license)

---

## Why hiveframe?

Most distributed DataFrame libraries solve one problem:
**scale computation**. hiveframe solves a different problem:
**scale data correctness**.

| | Dask / Modin | Snowpark | hiveframe |
|---|---|---|---|
| Scale computation | ✓ | ✓ | ✓ |
| ACID transactions | ✗ | partial | ✓ |
| Write-Ahead Log | ✗ | ✗ | ✓ |
| Built-in AI agent | ✗ | ✗ | ✓ |
| Minimal hardware | ✗ | ✗ | ✓ |
| No vendor lock-in | ✓ | ✗ | ✓ |
| Persistent by default | ✗ | ✓ | ✓ |

**Use hiveframe when:**
- You need data corrections to be auditable and reversible
- You want AI agents to write to your DataFrame safely
- You have many small machines, not one big one
- You need human + AI to collaborate on the same dataset

**Use Dask/Modin when:**
- You need maximum raw computation speed
- You have an existing Spark/Ray infrastructure
- You don't need transactional guarantees

---

## Architecture

```
core/
├── coordinator.py      # Transaction lifecycle (lock → apply → WAL → replicate)
├── write_node.py       # Mutable pandas write path
├── read_node.py        # Pandas read replica with sync lag + parquet persistence
├── lock_manager.py     # Cell-level lock manager
├── wal.py              # In-memory append-only WAL with LSN
├── transaction.py      # State machine + Operation model
├── dataframe.py        # DFrame public API + namespace isolation + pandas proxy layer
├── message.py          # MessagePack protocol (MessageType, Message)
├── quic_transport.py   # QUIC transport + in-memory fallback + request/response
├── registry.py         # Cluster node registry + dynamic partition management
├── heartbeat.py        # Periodic heartbeat + failure detection
├── replication.py      # WAL delta replication + snapshot request/response handler
└── cluster_runtime.py  # Runtime wiring + global snapshot fan-out + merge + rebalance
agent/
├── writer.py           # Async LLM agent writer with retry/backoff
└── prompt.py           # Structured prompt builder + JSON plan parser
```

---

## Quick Start

Install from pip for normal usage:

```bash
python -m venv .venv
source .venv/bin/activate
pip install hiveframe
```

## Development Mode (from clone)

Use this if you want to modify source code locally:

```bash
git clone <your-fork-or-upstream-url>
cd hiveframe
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

## Quick Start (hiveframe import)

Use the module import facade directly:

```python
import hiveframe as hf

df = hf.read(".dframe_store/employee_data.parquet")
print(df.head())
```

`hf.read(...)` accepts either a full parquet path or the persisted dataset name used with `to_persistent(...)`.

Run the ready-to-use example:

```bash
.venv/bin/python examples/hiveframe_import_usage.py
```

---

## Read vs Write Model

Core principle of DFrame:

| Operation | Path | Scope |
|---|---|---|
| **Write** (`df["col"] = ...`, `append`, `drop`, `rename`) | Transactional coordinator (local node) | Node-based, routed by partition |
| **Read sync** (`read_fresh()`, `head`, `groupby`, `to_csv`, etc.) | Local write-node snapshot | Local node only |
| **Read sync (global)** (`read_fresh_global()`) | Global fan-out → merge all nodes | Entire cluster |
| **Read global** (`read_fresh_global_async()`) | Global fan-out → merge all nodes | Entire cluster |
| **Read lazy** (`read_fresh_lazy()`) | Local write-node, chunked | Local node only, yields DataFrame chunks |
| **Read lazy (global)** (`read_fresh_global_lazy()`) | Global merged, chunked | Entire cluster, yields DataFrame chunks |

- **Write** is always sent to the local coordinator, locked at cell level, then replicated via WAL delta to read replicas. In cluster mode, writes are routed to the node that owns the partition for that row index.
- **`read_fresh()`** returns the local node snapshot synchronously — safe to call from anywhere including sync contexts.
- **`read_fresh_global()`** is a sync helper for global cluster reads and internally runs `read_fresh_global_async()`.
- **`read_fresh_global_async()`** fans out to all connected writer nodes and merges into one complete DataFrame — must be called from an async context.
- **`read_fresh_lazy()`** yields DataFrame chunks (default 1000 rows per chunk) from the local node, with columns matching the public DFrame API. Useful for iterating over large datasets without loading all rows at once.
- **`read_fresh_global_lazy()`** yields DataFrame chunks (default 1000 rows per chunk) from the global merged snapshot (all cluster nodes). Only available in sync context; raises if called in an event loop.
- If an event loop is already running, `read_fresh_global()` raises and you should use `await read_fresh_global_async()` directly.
- In **standalone** mode (no cluster), both methods behave identically and read from the local node.

---

## Modes

### Standalone (default)

Node runs independently without requiring peers or external services.
All transactional features remain fully active.

```python
import hiveframe as hf

df = hf.DFrame({"city": ["jakarta", "bandung"]})
df["city"] = ["DKI Jakarta", "West Java"]
print(df.head())
print(df.describe())
```

### Cluster mode (optional)

Enable with `enable_cluster=True`. The node will:
- listen on transport (`memory`, `quic`, or `tcp`),
- register to the registry (`memory`, `nats`, or `sqlite`),
- start heartbeat,
- broadcast WAL delta to read replicas after each commit,
- auto-assign and rebalance partitions when nodes join or fail,
- respond to `READ_SNAPSHOT_REQUEST` from other nodes for global read fan-out.

```python
import asyncio
import hiveframe as hf
from hiveframe.core.cluster_runtime import ClusterRuntime, RuntimeConfig

async def main():
    runtime1 = ClusterRuntime(RuntimeConfig(
        node_id="writer-1", role="write", port=19000, enable_cluster=True
    ))
    runtime2 = ClusterRuntime(RuntimeConfig(
        node_id="writer-2", role="write", port=19001, enable_cluster=True
    ))
    await runtime1.start()
    await runtime2.start()

    df1 = hf.DFrame.from_runtime(runtime1, {"city": ["jakarta", "bandung"], "score": [85, 90]})
    df2 = hf.DFrame.from_runtime(runtime2, {"city": ["surabaya", "medan"],  "score": [78, 82]})

    # Local read — fast, sync
    print(df1.read_fresh())

    # Global read — fan-out across all writer nodes, returns merged DataFrame
    merged = await df1.read_fresh_async()
    print(merged)
    #        city  score
    # 0   jakarta     85
    # 1   bandung     90
    # 2  surabaya     78
    # 3     medan     82

asyncio.run(main())
```

---

## Namespace Isolation

Each `DFrame` is assigned a unique `frame_id` at creation. All cell writes are prefixed with `{frame_id}::` so multiple DFrames sharing the same cluster node or coordinator never overlap.

```python
import asyncio
import hiveframe as hf
from hiveframe.core.cluster_runtime import ClusterRuntime, RuntimeConfig

async def main():
    runtime = ClusterRuntime(RuntimeConfig(
        node_id="writer-1", role="write", port=19000, enable_cluster=True
    ))
    await runtime.start()

    # Two completely independent DataFrames on the same runtime
    df_a = hf.DFrame.from_runtime(runtime, {"name": ["alice", "bob"], "age": [30, 25]})
    df_b = hf.DFrame.from_runtime(runtime, {"product": ["apple", "banana"], "price": [1.5, 0.8]})

    df_a["age"] = [31, 26]  # does NOT affect df_b

    print(df_a.read_fresh())
    #     name  age
    # 0  alice   31
    # 1    bob   26

    print(df_b.read_fresh())
    #   product  price
    # 0   apple    1.5
    # 1  banana    0.8

asyncio.run(main())
```

To share the same logical dataset across nodes, pass an explicit `frame_id`:

```python
df_node2 = hf.DFrame.from_runtime(runtime2, {"name": ["charlie"], "age": [40]}, frame_id=df_a._frame_id)

# Now read_fresh_async() on df_a returns all rows from both nodes
merged = await df_a.read_fresh_async()  # 3 rows total
```

---

## Dynamic Partitioning

Partitions are assigned and rebalanced automatically when writer nodes join or fail.

```
2 nodes:   writer-1: [0, 500)   writer-2: [500, 1000)
3 nodes:   writer-1: [0, 333)   writer-2: [333, 666)   writer-3: [666, 1000)
```

- Write routing: each row is mapped to a slot via `row_index % 1000` and sent to the node that owns that slot.
- Rebalance: triggered automatically on `register()` (node join) and `mark_failed()` (node failure).
- A `REBALANCE` broadcast message is sent to all peers after every partition change.

---

## RuntimeConfig

| Field | Default | Description |
|---|---|---|
| `node_id` | *(required)* | Unique node identifier |
| `role` | *(required)* | `"write"` or `"read"` |
| `region` | `"ap-southeast-1"` | Region label for routing |
| `host` | `"127.0.0.1"` | Transport bind address |
| `port` | `19000` | Transport port |
| `nats_url` | `"nats://127.0.0.1:4222"` | NATS server URL (optional) |
| `db_path` | `".hiveframe/registry.db"` | SQLite registry file path when `registry_backend="sqlite"` |
| `enable_cluster` | `False` | Enable transport + registry + heartbeat + global read |
| `registry_backend` | `"memory"` | `"memory"`, `"nats"`, or `"sqlite"` |
| `transport_backend` | `"memory"` | `"memory"`, `"quic"`, or `"tcp"` |
| `required_cluster` | `False` | Fail on startup if cluster is unavailable |
| `partition_start` | `0` | Initial partition range start (auto-overridden on rebalance) |
| `partition_end` | `1000` | Initial partition range end (auto-overridden on rebalance) |

### Registry/Transport Backend Options

- `registry_backend="sqlite"` — zero dependency, persistent registry for dev/homelab/single-machine
- `registry_backend="nats"` — production-ready, distributed registry (requires NATS)
- `registry_backend="memory"` — in-memory, for testing/dev only
- `transport_backend="tcp"` — network-real asyncio TCP socket transport (zero extra dependency)
- `transport_backend="quic"` — production-ready, cross-region, low-latency
- `transport_backend="memory"` — in-memory QUIC fallback for testing/dev only

Migration note: legacy sync shim methods (`start_server`, `register_handler`, `send(host, port, dict)`) now belong to `InMemoryTCPTransport`, not `TCPTransport`.

#### Example: SQLite registry and TCP transport

```python
from hiveframe.core.cluster_runtime import ClusterRuntime, RuntimeConfig

runtime = ClusterRuntime(RuntimeConfig(
    node_id="writer-1",
    role="write",
    port=19000,
    enable_cluster=True,
    registry_backend="sqlite",
    transport_backend="tcp",
    db_path=".hiveframe/dev-registry.db",
))
```

Use this combination for single-machine or homelab deployments where you want a persistent registry without running NATS.

---

## Install extras

```bash
# Real NATS backend
pip install -e .[nats]

# Real QUIC transport
pip install -e .[quic]

# Both
pip install -e .[transport]

# Excel export support
pip install openpyxl
```

---

## Pandas API Coverage

`DFrame` implements a **pandas proxy layer** — all pandas read APIs are automatically available via delegation to `read_fresh()`.

### Write API (transactional, node-based)

| Method | Description |
|---|---|
| `df["col"] = values` | Set a column via transaction |
| `df.append(row)` | Add a row via transaction |
| `df.drop(columns)` | Drop columns via transaction |
| `df.rename(mapping)` | Rename columns via transaction |

### Read API

| Method | Scope | Notes |
|---|---|---|
| `df.read_fresh()` | Local node | Sync, safe anywhere |
| `df.read_fresh_global()` | All cluster nodes | Sync helper (raises if called inside active event loop) |
| `await df.read_fresh_global_async()` | All cluster nodes | Async, fan-out + merge |
| `df.read_fresh_lazy()` | Local node | Yields DataFrame chunks (default 1000 rows) |
| `df.read_fresh_global_lazy()` | All cluster nodes | Yields DataFrame chunks (default 1000 rows, sync only) |
| `df.head()`, `df.tail()`, `df.loc[...]`, `df.iloc[...]` | Local node | Via proxy |
| `df.groupby()`, `df.describe()` | Local node | Via proxy |
| `df.sort_values()`, `df.fillna()`, `df.apply()` | Local node | Via proxy |
| `df.to_csv()`, `df.to_json()`, `df.to_parquet()` | Local node | Via proxy |
| `df.to_excel()` | Local node | Requires `pip install openpyxl` |
| `df.shape`, `df.columns`, `df.dtypes` | Local node | Properties via proxy |

> All pandas methods not explicitly listed above are automatically proxied via `__getattr__` to the local snapshot.

---

## Advanced Features

### Schema Validation

DFrame supports per-column schema validation:

```python
from hiveframe.core.schema import ColumnSchema
import hiveframe as hf

df = hf.DFrame(
    {"city": ["jakarta"], "score": [85]},
    schema={
        "city": ColumnSchema(dtype="str", nullable=False),
        "score": ColumnSchema(dtype="int", nullable=False, validator=lambda v: 0 <= v <= 100, description="Score 0-100"),
    }
)
# Invalid type or value will raise error on write
```

If you want normalization/coercion, make it explicit:

```python
df = hf.DFrame(
    {"score": ["85"]},
    schema={
        "score": ColumnSchema(dtype="int", nullable=False, coerce=True),
    },
)
```

When you persist a `DFrame`, HiveFrame writes a small sidecar schema file next to the parquet file so `hf.read(...)` can restore `dtype`, `nullable`, `description`, and `coerce`. If you rely on custom validators, pass the schema again explicitly on `read(...)`.

### WAL Persistence & Compaction

- WAL (Write-Ahead Log) can be persisted to disk for full auditability and recovery:

```python
from hiveframe.core.wal import WriteAheadLog
wal = WriteAheadLog(wal_path=".dframe_store/wal.jsonl")
```
- WAL compaction is available to control memory usage:

```python
wal.compact(keep_last_n=1000)
wal.compact_before_lsn(lsn=5000)
```

### Checkpoint & Rollback

Save and restore DataFrame state at any time:

```python
cp = df.checkpoint("before_ai")
# ... modify data ...
df.rollback(cp)  # Undo to checkpoint
```

### Cell History (Audit Trail)

Query all changes for a single cell:

```python
history = df.cell_history("city", 0)
for h in history:
    print(h)
```

### Streaming Agent (LLM, chunked)

Efficiently normalize large columns with LLM agent in chunks:

```python
async def llm_call(messages):
    # Call your LLM here, return list of {cell_id, value, confidence}
    ...
result = await writer.stream_normalize("city", llm_call, chunk_size=50)
```

### Metrics Endpoint

Expose DataFrame and cluster metrics for observability:

```python
metrics = df.get_metrics()
print(metrics)
# {'coordinator': {...}, 'wal': {...}, 'write_node': {...}, 'read_node': {...}, 'frame_id': ...}
```

### API Improvements

- `DFrame.from_csv()` and `DFrame.from_excel()` for direct file loading
- `df.describe_for_agent()` for LLM context building

---

## Testing

```bash
# Unit tests (default, no external services required)
pytest

# Integration tests (requires NATS + aioquic)
RUN_PHASE2_INTEGRATION=1 NATS_URL=nats://127.0.0.1:4222 pytest tests/test_phase2_runtime.py -v
```

---

## Start Cluster

Use `examples/start_cluster.py` to run a node in cluster mode.

**Writer node** (terminal 1):
```bash
python examples/start_cluster.py \
  --node-id writer-1 \
  --role write \
  --host 127.0.0.1 \
  --port 19000 \
  --region ap-southeast-1
```

**Read replica** (terminal 2):
```bash
python examples/start_cluster.py \
  --node-id reader-1 \
  --role read \
  --host 127.0.0.1 \
  --port 19001 \
  --region ap-southeast-1
```

**With real NATS registry** (requires a running NATS server):
```bash
python examples/start_cluster.py \
  --node-id writer-1 \
  --role write \
  --port 19000 \
  --registry-backend nats \
  --nats-url nats://127.0.0.1:4222
```

**With SQLite registry + TCP transport** (single machine / homelab):
```bash
python examples/start_cluster.py \
  --node-id writer-1 \
  --role write \
  --port 19000 \
  --registry-backend sqlite \
  --transport-backend tcp \
  --db-path .hiveframe/cluster-a.db
```

Minimal end-to-end 2-node TCP demo:

```bash
python examples/tcp_two_node_e2e.py
```

Manual multi-process runbook (Terminal 1/2/3): see `docs/docs/guides/homelab-setup.md`.

All available options:
```
--node-id           Unique node ID (required)
--role              write | read (required)
--host              Bind host (default: 127.0.0.1)
--port              Bind port (default: 19000)
--region            Region label (default: ap-southeast-1)
--registry-backend  memory | nats | sqlite (default: memory)
--transport-backend memory | quic | tcp (default: memory)
--nats-url          NATS server URL (default: nats://127.0.0.1:4222)
--db-path           SQLite registry path (default: .hiveframe/registry.db)
--partition-start   Initial partition range start (default: 0, auto-rebalanced)
--partition-end     Initial partition range end (default: 1000, auto-rebalanced)
```

---

## Usage

Start here for the module import flow:

- `examples/hiveframe_import_usage.py` (recommended first run)
- Run with `.venv/bin/python examples/hiveframe_import_usage.py`

### Standalone — DFrame directly

```python
import hiveframe as hf

df = hf.DFrame({"city": ["jakarta", "bandung"], "score": [85, 90]})

# Write (transactional)
df["city"] = ["DKI Jakarta", "West Java"]

# Read — full pandas API available
print(df.head())
print(df.sort_values("score"))
print(df.describe())
df.to_csv("output.csv", index=False)
df.to_json("output.json", orient="records", indent=2)

# Lazy chunked read (for large data)
for chunk in df.read_fresh_lazy(chunk_size=50):
    print(chunk)

# Lazy chunked global read (all cluster nodes)
for chunk in df.read_fresh_global_lazy(chunk_size=50):
    print(chunk)
```

### Standalone — LLM Agent writer

```python
import asyncio
import hiveframe as hf
from agent.writer import AgentWriter

async def main():
    df = hf.DFrame({"city": ["jakarta"]})
    writer = AgentWriter(df._coordinator, agent_id="normalizer", author_type="llm_normalization")
    await writer.normalize(f"{df._frame_id}::city_0", "DKI Jakarta", confidence=0.97)
    print(df.read_fresh())

asyncio.run(main())
```

### Cluster mode — multiple independent DFrames on same node

```python
import asyncio
import hiveframe as hf
from hiveframe.core.cluster_runtime import ClusterRuntime, RuntimeConfig

async def main():
    runtime = ClusterRuntime(RuntimeConfig(
        node_id="writer-1", role="write", port=19000, enable_cluster=True
    ))
    await runtime.start()

    # Fully isolated — different frame_id, no overlap
    df_users = hf.DFrame.from_runtime(runtime, {"name": ["alice", "bob"], "age": [30, 25]})
    df_orders = hf.DFrame.from_runtime(runtime, {"product": ["apple"], "qty": [3]})

    df_users["age"] = [31, 26]  # does NOT affect df_orders

    print(df_users.read_fresh())
    print(df_orders.read_fresh())

asyncio.run(main())
```

### Cluster mode — multi-writer global read

```python
import asyncio
import hiveframe as hf
from hiveframe.core.cluster_runtime import ClusterRuntime, RuntimeConfig

async def main():
    runtime1 = ClusterRuntime(RuntimeConfig(
        node_id="writer-1", role="write", port=19000, enable_cluster=True
    ))
    runtime2 = ClusterRuntime(RuntimeConfig(
        node_id="writer-2", role="write", port=19001, enable_cluster=True
    ))
    await runtime1.start()
    await runtime2.start()

    df1 = hf.DFrame.from_runtime(runtime1, {"city": ["jakarta", "bandung"], "score": [85, 90]})
    # df2 shares same frame_id as df1 — same logical dataset, different partition
    df2 = hf.DFrame.from_runtime(runtime2, {"city": ["surabaya", "medan"], "score": [78, 82]},
                               frame_id=df1._frame_id)

    # Local read (this node only)
    print(df1.read_fresh())   # 2 rows

    # Global read (all nodes merged)
    merged = await df1.read_fresh_async()
    print(merged)             # 4 rows from both nodes
    print(merged.shape)       # (4, 2)

asyncio.run(main())
```

### Cluster mode — writer + read replica

```python
import asyncio
import hiveframe as hf
from hiveframe.core.cluster_runtime import ClusterRuntime, RuntimeConfig

async def main():
    writer = ClusterRuntime(
        RuntimeConfig(node_id="writer-1", role="write", port=19000, enable_cluster=True)
    )
    reader = ClusterRuntime(
        RuntimeConfig(node_id="reader-1", role="read", port=19001, enable_cluster=True)
    )
    await writer.start()
    await reader.start()

    df = hf.DFrame.from_runtime(writer, {"score": [85, 92]})
    df["score"] = [99, 100]   # write to local writer node

    print(df.read_fresh())            # local snapshot
    print(await df.read_fresh_async()) # global — includes replicas

asyncio.run(main())
```

### Cluster mode — submit transaction directly

```python
import asyncio
from hiveframe.core.cluster_runtime import ClusterRuntime, RuntimeConfig
from hiveframe.core.transaction import Operation

async def main():
    runtime = ClusterRuntime(
        RuntimeConfig(node_id="writer-1", role="write", port=19000, enable_cluster=True)
    )
    await runtime.start()

    ops = [
        Operation(
            cell_id="my_frame::city_0",
            old_value=None,
            new_value="DKI Jakarta",
            author_type="human",
            author_id="user-1",
        )
    ]
    tx = runtime.coordinator.submit(ops)
    print(tx.state)  # TxState.SYNCED

asyncio.run(main())
```

### Cluster mode — LLM Agent on top of ClusterRuntime

```python
import asyncio
import hiveframe as hf
from hiveframe.agent.writer import AgentWriter
from hiveframe.core.cluster_runtime import ClusterRuntime, RuntimeConfig

async def main():
    runtime = ClusterRuntime(
        RuntimeConfig(node_id="writer-1", role="write", port=19000, enable_cluster=True)
    )
    await runtime.start()

    df = hf.DFrame.from_runtime(runtime, {"city": ["jakarta"]})
    writer = AgentWriter(
        runtime.coordinator,
        agent_id="normalizer",
        author_type="llm_normalization",
    )
    await writer.normalize(f"{df._frame_id}::city_0", "DKI Jakarta", confidence=0.97)
    print(df.read_fresh())

asyncio.run(main())
```

---

## LLM Agent Prompt

`agent/prompt.py` provides a structured prompt builder so an LLM can run queries and write to the DataFrame using the available API.

### Build prompt + parse plan

```python
import hiveframe as hf
from hiveframe.agent.prompt import build_messages, parse_plan
from hiveframe.agent.writer import AgentWriter

df = hf.DFrame({"city": ["jakarta", "bandung"], "score": [85, 92]})
writer = AgentWriter(df._coordinator, agent_id="normalizer", author_type="llm_normalization")

# 1. Send a DataFrame snapshot as context to the LLM
snapshot = df.read_fresh().to_string()
messages = build_messages(
    user_instruction="Normalize all city values to official Indonesian province names.",
    dataframe_snapshot=snapshot,
)

# 2. Send to LLM (example using OpenAI)
# response = openai_client.chat.completions.create(model="gpt-4o", messages=messages)
# llm_text = response.choices[0].message.content

# 3. Parse JSON plan from LLM response
plan = parse_plan(llm_text)

# 4. Execute via writer
if plan.get("action") == "batch_enrich":
    await writer.batch_enrich(plan["operations"])
elif plan.get("action") == "normalize":
    op = plan["operations"][0]
    await writer.normalize(op["cell_id"], op["value"], confidence=op["confidence"])
```

### LLM JSON plan format

```json
{
  "action": "batch_enrich",
  "reasoning": "Normalizing city values to official Indonesian province names.",
  "operations": [
    {"cell_id": "{frame_id}::city_0", "value": "DKI Jakarta", "confidence": 0.97},
    {"cell_id": "{frame_id}::city_1", "value": "West Java",   "confidence": 0.95}
  ]
}
```

Available actions: `normalize` | `batch_enrich` | `read` | `describe`

### Cell ID convention

```
{frame_id}::{column_name}_{row_index}   ← namespaced, zero-based row index

Examples:
  abc123::city_0   → DFrame "abc123", column "city", row 0
  abc123::score_3  → DFrame "abc123", column "score", row 3
```

> `frame_id` is available via `df._frame_id`.

### Confidence scoring

| Range | Meaning |
|---|---|
| 0.95 – 1.00 | Very confident (exact match, authoritative source) |
| 0.80 – 0.94 | Confident (strong inference) |
| 0.60 – 0.79 | Moderate (plausible but uncertain) |
| < 0.60 | Do not write — return clarification request to user |

See also:
- `examples/hiveframe_import_usage.py`
- `examples/basic_usage.py`
- `examples/llm_agent_usage.py`
- `examples/start_cluster.py`

---

## Contributing

Contributions are welcome. See `CONTRIBUTING.md` for dev setup, coding guidelines, and pull request checklist.

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.
