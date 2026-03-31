# DFrame API

## Overview

`DFrame` is the main DataFrame abstraction in hiveframe, providing transactional, distributed, and AI-augmented operations on top of pandas.

## Key Methods

- `__init__(data, schema=None, ...)` — Create a new DFrame
- `from_csv(path)` — Load from CSV file
- `from_excel(path)` — Load from Excel file
- `from_runtime(runtime, data, frame_id=None)` — Attach to a cluster runtime
- `read_fresh()` — Get the latest local snapshot
- `read_fresh_global()` — Get merged snapshot from all cluster nodes
- `read_fresh_lazy(chunk_size=1000)` — Iterate over local data in chunks
- `read_fresh_global_lazy(chunk_size=1000)` — Iterate over global data in chunks
- `checkpoint(label=None)` — Save a checkpoint
- `rollback(checkpoint_id)` — Restore to a checkpoint
- `cell_history(col, row_idx)` — Get audit trail for a cell
- `get_metrics()` — Get DataFrame and cluster metrics
- `describe_for_agent()` — Build LLM context string

## Example

```python
import hiveframe as hf

df = hf.DFrame({"city": ["jakarta"]})
df["city"] = ["DKI Jakarta"]
print(df.read_fresh())
```
