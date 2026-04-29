# Telemetry

## Overview

Hiveframe exposes operational telemetry through:

- structured logs from core and agent modules,
- `MultiFrameResult` attempt metadata for query loops,
- write/audit APIs such as `cell_history(...)`.

Use these surfaces to debug failed LLM runs, monitor query quality, and track write outcomes.

---

## 1) Enable structured logs

Hiveframe uses Python `logging` namespaces. Set level to `DEBUG` when troubleshooting.

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
```

High-signal loggers:

- `hiveframe.agent.multi` - query generation/execution/review loop, fallback reasons, LLM request/response previews
- `hiveframe.agent.writer` - normalize/batch write lifecycle, confidence skips, retry outcomes
- `core.write_node` - transaction apply path, optimistic conflicts, delta callback dispatch
- `hiveframe.dataframe` - DataFrame load/read lifecycle and runtime integration

---

## 2) MultiFrameAgent telemetry

`MultiFrameAgent.analyze(..., mode="query")` returns `MultiFrameResult` with per-attempt metadata:

- `total_llm_calls`: total LLM calls used by the run
- `converged`: `True` when final review verdict is sufficient
- `final_verdict`: last review status (`accepted`, `partial`, `error`, etc.)
- `fallback_reason`: machine-readable fallback trigger when sample mode is used
- `review_history`: structured verdict history (`ReviewVerdict` list)
- `attempt_summaries`: attempt-level telemetry (`generated_labels`, `failed_labels`, rewrites, source)

```python
result = await agent.analyze("Analyze top trends", mode="query", max_retries=1)

print(result.total_llm_calls)
print(result.final_verdict, result.converged)
print(result.fallback_reason)
for attempt in result.attempt_summaries:
    print(attempt)
```

Tip: persist `result.to_dict()` to your own observability pipeline if you need long-term tracking.

---

## 3) AgentWriter telemetry

`AgentWriter` emits action-level debug logs for:

- operation building and confidence filtering,
- transaction submission attempts,
- lock-conflict retry/backoff,
- final write/skip counts.

```python
write_result = await writer.batch_enrich(operations)
print(write_result)  # {'written': ..., 'skipped': ..., 'tx_id': ..., 'success': ...}
```

When a write is skipped, inspect logs and confidence values first.

---

## 4) Core write-path telemetry

`core.write_node` logs transaction details:

- apply start/success with `tx_id` and version,
- optimistic conflicts (`current` vs `expected` values),
- callback dispatch for delta propagation.

These logs are useful for diagnosing transactional conflicts and replication lag symptoms.

---

## 5) Audit trail telemetry

Use `cell_history(column, row_idx)` for per-cell write history:

```python
history = df.cell_history("city", 0)
for event in history:
    print(event)
```

This gives you immutable, cell-level change provenance for compliance and debugging.

