# TransactionCoordinator API

## Overview

The TransactionCoordinator manages transactional operations, lock management, and WAL for a DFrame or cluster node.

## Key Methods

- `submit(operations)` — Submit a list of operations as a transaction
- `get_stats()` — Get coordinator metrics

## Example

```python
from hiveframe.core.coordinator import TransactionCoordinator
from hiveframe.core.transaction import Operation

coordinator = TransactionCoordinator()
ops = [Operation(cell_id="frame::col_0", old_value=None, new_value="foo", author_type="human", author_id="user")]
tx = coordinator.submit(ops)
print(tx.state)
```
