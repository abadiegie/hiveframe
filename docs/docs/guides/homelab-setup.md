# Homelab Setup Guide

## Overview

hiveframe can run on minimal hardware and is ideal for homelab or single-board cluster setups.

## Recommended Backends

- Use `registry_backend="sqlite"` for zero-dependency persistent registry
- Use `transport_backend="tcp"` for local network communication

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
))
```

This setup requires no external services and is perfect for home clusters.
