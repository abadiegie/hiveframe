# Cell History Guide

## Overview

Cell history provides a full audit trail for every cell in a DFrame.

## Usage

```python
history = df.cell_history("city", 0)
for h in history:
    print(h)
```

This is useful for compliance, debugging, and understanding data provenance.
