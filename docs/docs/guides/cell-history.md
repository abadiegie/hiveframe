# Cell History Guide

## Overview

Cell history provides a full audit trail for every cell in a DFrame.

This feature is available only when `transactional=True` (default).
If you run a frame with `transactional=False`, `cell_history(...)` returns an empty list by design.

## Usage

```python
history = df.cell_history("city", 0)
for h in history:
    print(h)
```

This is useful for compliance, debugging, and understanding data provenance.

## Non-transactional mode note

```python
df = hf.DFrame({"city": ["jakarta"]}, transactional=False)
assert df.cell_history("city", 0) == []
```

Use transactional mode if you need audit trails, checkpoints, or rollback semantics.

