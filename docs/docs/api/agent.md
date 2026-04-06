# Agent API

## Overview

Hiveframe agent layer has two main interfaces:

- `AgentWriter` for transactional writes into a single `DFrame`
- `MultiFrameAgent` for analysis across one or many `DFrame` objects (sample/query mode)

## AgentWriter Key Methods

- `normalize(cell_id, value, confidence)` — Write a single cell with confidence
- `batch_enrich(operations)` — Write a batch of cell updates
- `stream_normalize(column, llm_call, chunk_size=50)` — Normalize a column in streaming chunks

## MultiFrameAgent Key Methods

- `analyze(instruction, mode="sample", output_frame=None, max_retries=0, ...)` — Analyze one or many frames
- `_safe_eval(query_str, df)` — Guarded pandas expression executor for query mode
- `_build_schema_context()` — Build schema + numeric stats context (without sample rows)

### Analysis Modes

- `sample` — one LLM call using `describe_for_agent()` context per frame
- `query` — iterative loop (`plan/act -> review -> optional retry -> final analysis`)

In `query` mode:
- each generated query must start with `df`
- forbidden patterns (`import`, `exec`, `eval`, `open`, `os`, `sys`, etc.) are rejected
- if no queries are generated, flow falls back to sample mode
- set `max_retries > 0` to enable iterative review/retry; keep `max_retries=0` for legacy simple flow

### Review Verdicts

- `accepted`
- `partial`
- `error`
- `plan`
- `rejected`
- `merge`

`MultiFrameResult` also includes iterative metadata:
- `review_history`
- `total_llm_calls`
- `converged`
- `final_verdict`

### Result Types

- `MultiFrameResult`
- `FrameInsight`

Both are exported from `hiveframe.agent`.

## Example

```python
from hiveframe.agent.writer import AgentWriter

writer = AgentWriter(df._coordinator, agent_id="normalizer", author_type="llm_normalization")
await writer.normalize(f"{df._frame_id}::city_0", "DKI Jakarta", confidence=0.97)
```

## MultiFrameAgent Example

```python
import asyncio
import hiveframe as hf
from hiveframe.agent import MultiFrameAgent

async def main() -> None:
	sales = hf.DFrame({"city": ["jakarta", "bandung", "jakarta"], "score": [90, 80, 70]})
	inventory = hf.DFrame({"city": ["jakarta", "bandung"], "stock": [12, 4]})

	agent = MultiFrameAgent(frames={"sales": sales, "inventory": inventory})
	result = await agent.analyze("City mana score tinggi tapi stock rendah?", mode="query")
	print(result.analysis)
	print(result.to_markdown())

asyncio.run(main())
```

