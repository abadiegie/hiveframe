# AgentWriter API

## Overview

`AgentWriter` enables LLM or programmatic agents to write to a DFrame with confidence scoring, batch enrichment, and normalization.

## Key Methods

- `normalize(cell_id, value, confidence)` — Write a single cell with confidence
- `batch_enrich(operations)` — Write a batch of cell updates
- `stream_normalize(column, llm_call, chunk_size=50)` — Normalize a column in streaming chunks

## Example

```python
from hiveframe.agent.writer import AgentWriter

writer = AgentWriter(df._coordinator, agent_id="normalizer", author_type="llm_normalization")
await writer.normalize(f"{df._frame_id}::city_0", "DKI Jakarta", confidence=0.97)
```
