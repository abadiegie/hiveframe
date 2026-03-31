# ClusterRuntime API

## Overview

`ClusterRuntime` manages the lifecycle of a cluster node, including registry, transport, replication, and heartbeat.

## Key Methods

- `__init__(config)` — Initialize with a `RuntimeConfig`
- `start()` — Start the node and join the cluster
- `route_write(row_index)` — Get the node responsible for a row
- `read_global_snapshot()` — Get merged snapshot from all writers

## Example

```python
from hiveframe.core.cluster_runtime import ClusterRuntime, RuntimeConfig

runtime = ClusterRuntime(RuntimeConfig(node_id="writer-1", role="write", enable_cluster=True))
await runtime.start()
```
