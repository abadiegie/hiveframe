# Checkpoint & Rollback Guide

## Overview

Checkpoints allow you to save and restore the state of a DFrame at any time.

## Usage

```python
cp = df.checkpoint("before_ai")
# ... modify data ...
df.rollback(cp)  # Undo to checkpoint
```

You can list and diff checkpoints as well.
