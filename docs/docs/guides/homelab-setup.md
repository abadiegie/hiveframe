# Homelab Setup Guide

## Overview

hiveframe can run on minimal hardware and is ideal for homelab or single-board cluster setups.

## Recommended Backends

- Use `registry_backend="sqlite"` for zero-dependency persistent registry
- Use `transport_backend="tcp"` for local network communication
- Point every node at the same `db_path` when they should share one registry state

## SQLite Concurrency Note

SQLite is great for single-machine and homelab clusters, but it is not a high-concurrency distributed registry.

- Use `registry_backend="sqlite"` for development, demos, and low-write-control-plane workloads.
- For heavier concurrent node join/fail traffic across multiple machines, prefer `registry_backend="nats"`.
- If you keep SQLite, use one shared stable path and avoid aggressive simultaneous process churn.

## Example

```python
from hiveframe.core.cluster_runtime import ClusterRuntime, RuntimeConfig

runtime = ClusterRuntime(RuntimeConfig(
    node_id="writer-1",
    role="write",
    port=19000,
    enable_cluster=True,
    registry_backend="sqlite",
    transport_backend="tcp",
    db_path=".hiveframe/homelab-registry.db",
))
```

This setup requires no external services and is perfect for home clusters. Use a stable shared path such as `.hiveframe/homelab-registry.db` for every local node process that should participate in the same registry.

## Minimal 2-Node TCP End-to-End

Run the dedicated demo script:

```bash
python examples/tcp_two_node_e2e.py
```

Expected output (merged snapshot from two writer nodes):

```text
Merged snapshot:
   left  right
0   1.0    NaN
1   NaN    2.0
```

This demo starts two writer runtimes on localhost with `registry_backend="sqlite"` + `transport_backend="tcp"`, writes one row on each node, and reads a global merged snapshot.

## Manual Multi-Process Runbook (Terminal 1/2/3)

Use this when you want to run each node manually in separate terminals.

### Terminal 1 — writer node

```bash
python examples/start_cluster.py \
  --node-id tcp-w1 \
  --role write \
  --host 127.0.0.1 \
  --port 19400 \
  --registry-backend sqlite \
  --transport-backend tcp \
  --db-path .hiveframe/tcp-manual.db
```

### Terminal 2 — second writer node

```bash
python examples/start_cluster.py \
  --node-id tcp-w2 \
  --role write \
  --host 127.0.0.1 \
  --port 19401 \
  --registry-backend sqlite \
  --transport-backend tcp \
  --db-path .hiveframe/tcp-manual.db
```

### Terminal 3 — verify both nodes are registered

```bash
python - <<'PY'
import sqlite3

conn = sqlite3.connect('.hiveframe/tcp-manual.db')
rows = conn.execute(
    "SELECT node_id, role, host, port, status FROM nodes ORDER BY node_id"
).fetchall()
conn.close()

for row in rows:
    print(row)
PY
```

Expected: both `tcp-w1` and `tcp-w2` appear with `status='healthy'`.

