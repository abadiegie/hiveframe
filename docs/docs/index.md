# hiveframe

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-20%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
Transactional, distributed-ready pandas-compatible DataFrame engine.

**hiveframe** scales DataFrame workloads across many small machines with transactions, persistence, and AI agent support built in. No new paradigm to learn. Just `import hiveframe as hf`.

Supports single-node standalone mode and optional multi-node cluster mode with QUIC transport, NATS registry, heartbeat, WAL-based delta replication, **global read fan-out**, **dynamic partition assignment**, and **per-DFrame namespace isolation** — multiple independent DataFrames can run on the same cluster node without overlap.

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
├── read_node.py        # Polars read replica with sync lag
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

...existing code from README continues here...
