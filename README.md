# hiveframe

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-133%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Docs](https://img.shields.io/badge/docs-abadiegie.github.io%2Fhiveframe-orange)](https://abadiegie.github.io/hiveframe/)

Transactional, distributed-ready pandas-compatible DataFrame engine.

**hiveframe** scales DataFrame workloads across many small machines with transactions, persistence, and AI agent support built in. No new paradigm to learn — just `import hiveframe as hf`.

> hiveframe outputs data, not visualizations. What you do with that data is entirely up to you.

📖 **Full documentation: [abadiegie.github.io/hiveframe](https://abadiegie.github.io/hiveframe/)**

---

## Why hiveframe?

Most distributed DataFrame libraries solve one problem: **scale computation**.
hiveframe solves a different problem: **scale data correctness**.

| | Dask / Modin | Snowpark | hiveframe |
|---|---|---|---|
| Scale computation | ✓ | ✓ | ✓ |
| ACID transactions | ✗ | partial | ✓ |
| Write-Ahead Log | ✗ | ✗ | ✓ |
| Built-in AI agent | ✗ | ✗ | ✓ |
| Minimal hardware | ✗ | ✗ | ✓ |
| No vendor lock-in | ✓ | ✗ | ✓ |
| Persistent by default | ✗ | ✓ | ✓ |

**Use hiveframe when** you need data corrections to be auditable, want AI agents to write to your DataFrame safely, or have many small machines instead of one big one.

---

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install hiveframe
```

```python
import hiveframe as hf

df = hf.DFrame({"city": ["jakarta", "bandung"], "score": [85, 90]})
df["city"] = ["DKI Jakarta", "West Java"]   # transactional write
print(df.head())
print(df.describe())
```

### Development (from clone)

```bash
git clone https://github.com/abadiegie/hiveframe
cd hiveframe
pip install -e .[dev]
pytest
```

---

## Key Features

| Feature | Docs |
|---|---|
| DFrame API & pandas proxy | [API Reference → DFrame](https://abadiegie.github.io/hiveframe/api/dataframe/) |
| Cluster mode, RuntimeConfig, transport backends | [API Reference → Cluster](https://abadiegie.github.io/hiveframe/api/cluster/) |
| LLM AgentWriter & MultiFrameAgent | [API Reference → AgentWriter](https://abadiegie.github.io/hiveframe/api/agent/) |
| Checkpoint & Rollback | [Guides → Checkpoint](https://abadiegie.github.io/hiveframe/guides/checkpoint/) |
| Cell History (audit trail) | [Guides → Cell History](https://abadiegie.github.io/hiveframe/guides/cell-history/) |
| Telemetry & observability | [Guides → Telemetry](https://abadiegie.github.io/hiveframe/guides/telemetry/) |
| Homelab / multi-node setup | [Guides → Homelab Setup](https://abadiegie.github.io/hiveframe/guides/homelab-setup/) |
| Getting started walkthrough | [Getting Started](https://abadiegie.github.io/hiveframe/getting-started/) |

---

## Minimal Examples

### Standalone with LLM Agent

```python
import asyncio, hiveframe as hf
from hiveframe.agent.writer import AgentWriter

async def main():
    df = hf.DFrame({"city": ["jakarta"]})
    writer = AgentWriter(df._coordinator, agent_id="normalizer", author_type="llm_normalization")
    await writer.normalize(f"{df._frame_id}::city_0", "DKI Jakarta", confidence=0.97)
    print(df.read_fresh())

asyncio.run(main())
```

### MultiFrameAgent (analysis)

```python
import asyncio, hiveframe as hf
from hiveframe.agent import MultiFrameAgent

async def main():
    sales = hf.DFrame({"city": ["jakarta", "bandung"], "score": [90, 80]})
    inventory = hf.DFrame({"city": ["jakarta", "bandung"], "stock": [12, 4]})

    agent = MultiFrameAgent(
        frames={"sales": sales, "inventory": inventory},
        provider="anthropic",
    )
    result = await agent.analyze("City mana score tinggi tapi stock rendah?", mode="query")
    print(result.final_verdict, result.total_llm_calls)
    print(result.attempt_summaries)
    print(result.to_markdown())

asyncio.run(main())
```

`result.series` is structured data only. hiveframe outputs data, not visualizations.
What you do with that data is entirely up to you.

### Cluster mode (multi-writer global read)

```python
import asyncio, hiveframe as hf
from hiveframe.core.cluster_runtime import ClusterRuntime, RuntimeConfig

async def main():
    r1 = ClusterRuntime(RuntimeConfig(node_id="w1", role="write", port=19000, enable_cluster=True))
    r2 = ClusterRuntime(RuntimeConfig(node_id="w2", role="write", port=19001, enable_cluster=True))
    await r1.start()
    await r2.start()

    df1 = hf.DFrame.from_runtime(r1, {"city": ["jakarta", "bandung"], "score": [85, 90]})
    df2 = hf.DFrame.from_runtime(r2, {"city": ["surabaya", "medan"], "score": [78, 82]},
                                  frame_id=df1._frame_id)

    merged = await df1.read_fresh_async()   # fan-out across all nodes
    print(merged)   # 4 rows

asyncio.run(main())
```

For full cluster setup including TCP/QUIC transport and SQLite/NATS registry, see the [Homelab Setup guide](https://abadiegie.github.io/hiveframe/guides/homelab-setup/).

---

## Install extras

```bash
pip install hiveframe[excel]      # Excel import/export (openpyxl)
pip install hiveframe[nats]       # NATS registry backend
pip install hiveframe[quic]       # QUIC transport
pip install hiveframe[redis]      # Redis-backed WAL backend (optional, multi-user)
pip install hiveframe[mysql]      # MySQL-backed WAL backend (optional, multi-user)
pip install hiveframe[transport]  # NATS + QUIC
pip install hiveframe[all]        # All optional dependencies
```

---

## Testing

```bash
pytest                            # unit tests, no external services required
RUN_PHASE2_INTEGRATION=1 NATS_URL=nats://127.0.0.1:4222 pytest tests/test_phase2_runtime.py -v
```

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, coding guidelines, and the pull request checklist.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
