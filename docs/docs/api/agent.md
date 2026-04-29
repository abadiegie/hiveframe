# Agent API

## Overview

Hiveframe agent layer has four main interfaces:

- `AgentWriter` for transactional writes into a single `DFrame`
- `RelationalAgentWriter` for LLM normalization using cross-frame relational context
- `MultiFrameAgent` for analysis across one or many `DFrame` objects (sample/query mode)
- `SeriesSpec` for structured chart-ready data output from LLM analysis

## AgentWriter Key Methods

- `normalize(cell_id, value, confidence)` — Write a single cell with confidence
- `batch_enrich(operations)` — Write a batch of cell updates
- `stream_normalize(column, llm_call, chunk_size=50, progress_callback=None, custom_instruction=None)` — Normalize a column in streaming chunks with context-aware prompts

### Stream Normalize

`stream_normalize()` uses context-aware prompts that include actual data + all columns to help LLM make better normalization decisions. Features:

- **Context-aware prompts**: Actual DataFrame snapshot with all columns for context
- **Confidence scoring**: Only writes values with confidence ≥ 0.60
- **Debug logging**: Full audit trail at every step for troubleshooting
- **Custom instructions**: Flexible normalization rules per use case
- **Progress tracking**: Optional callback for monitoring

Example with custom instruction:

```python
result = await writer.stream_normalize(
    column="city",
    llm_call=llm_call,
    chunk_size=50,
    custom_instruction=(
        "Normalize city values to official Indonesian province names. "
        "Format: 'Province: City'. Examples: 'jkt' → 'DKI Jakarta: Jakarta Pusat'"
    ),
)
```

For complete API reference, logging format, best practices, and debugging tips:
**[Agent → Stream Normalize](../agent/stream-normalize.md)**

## RelationalAgentWriter Key Methods

- `stream_normalize_relational(target_column, instruction, chunk_size=10, provider, ...)` — Normalize a column using cross-frame relations

For full API reference, parameter details, and examples see:
[Agent → Relational Writer](../agent/relational-writer.md)

Use `RelationalAgentWriter` when your LLM annotation workflow requires related data from other frames
(e.g., comment stance depends on parent post context).

## MultiFrameAgent Key Methods

- `analyze(instruction, mode="sample", output_frame=None, max_retries=0, ...)` — Analyze one or many frames
- `_safe_eval(query_str, df)` — Guarded pandas expression executor for query mode
- `_build_schema_context()` — Build schema + numeric stats context (without sample rows)

### Analysis Modes

- `sample` — one LLM call using `describe_for_agent()` context per frame
- `query` — iterative loop (`plan/act -> review -> optional retry -> final analysis`)

Token-efficient mode is available via `columns_hint` in `analyze(...)`.
See [Agent -> Columns Hint](../agent/columns-hint.md) and runnable example
at `examples/multiframe_columns_hint.py`.

In `query` mode:
- each generated query must start with `df`
- forbidden patterns (`import`, `exec`, `eval`, `open`, `os`, `sys`, etc.) are rejected
- if no queries are generated, flow falls back to sample mode
- query mode always performs a reviewed iterative pass (`query -> execute -> review -> final analysis`)
- `max_retries=0` means no additional retries after the first reviewed attempt
- increase `max_retries` to allow extra correction rounds when review verdicts are not yet sufficient

### Review Verdicts

| Verdict | Meaning |
|---|---|
| `accepted` | Results fully answer the instruction |
| `merge` | Sufficient partial results across queries |
| `partial` | Some results useful, specific parts missing |
| `error` | One or more queries failed technically |
| `plan` | Needs columns not yet visible in schema |
| `rejected` | Results not relevant — wrong approach |

`MultiFrameResult` also includes iterative metadata:
- `review_history` — `list[ReviewVerdict]`, one per reviewed attempt
- `total_llm_calls` — total LLM calls (query generation + review + final analysis)
- `converged` — `True` if the last verdict was `accepted` or `merge`
- `final_verdict` — last verdict status string
- `fallback_reason` — short machine-readable fallback reason when the flow drops to sample mode
- `attempt_summaries` — per-attempt telemetry (`source`, executed labels, failed labels, verdict, optional rewrites)

For logging setup and telemetry interpretation, see
[Guides -> Telemetry](../guides/telemetry.md).

For full loop behavior, verdict handling, and examples see
[Guides → Iterative Agent](../guides/iterative-agent.md).

### Result Types

- `MultiFrameResult` — top-level result with `analysis`, `insights`, `series`, metadata
- `FrameInsight` — single insight with `finding`, `frames`, `confidence`
- `SeriesSpec` — chart-ready data series (see below)

All three are exported from `hiveframe.agent`.

---

## SeriesSpec

`SeriesSpec` is a pure-data chart series format produced by the LLM analysis layer.
Hiveframe stores structured x/y arrays and labels, but it does not render charts for you.

> hiveframe outputs data, not visualizations. What you do with that data is entirely up to you.

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `label` | `str` | Series identifier |
| `x` | `list[Any]` | X-axis values |
| `y` | `list[Any]` | Y-axis values |
| `x_label` | `str` | Optional x-axis label |
| `y_label` | `str` | Optional y-axis label |
| `series_type` | `str` | Chart hint such as `bar`, `line`, `scatter`, or `pie` |

### Methods

| Method | Returns | Description |
|---|---|---|
| `to_dict()` | `dict` | Serialize to a JSON-friendly dictionary |
| `from_dict(payload)` | `SeriesSpec` | Parse current schema or a best-effort legacy payload |
| `to_dataframe()` | `pd.DataFrame` | Convert the x/y arrays into a pandas DataFrame |

### MultiFrameResult helpers

`MultiFrameResult.series` is a `list[SeriesSpec]` populated automatically when the LLM
returns structured series data.

| Method | Description |
|---|---|
| `get_series(name)` | Get `SeriesSpec` by label/name, `None` if not found |
| `to_dataframe(name)` | Convert one series to `pd.DataFrame`, empty if not found |

### Example

```python
import asyncio
import hiveframe as hf
from hiveframe.agent import MultiFrameAgent

async def main() -> None:
    sales = hf.DFrame({"month": ["Jan", "Feb", "Mar"], "revenue": [100, 150, 200]})
    agent = MultiFrameAgent(frames={"sales": sales}, provider="anthropic")

    result = await agent.analyze("Trend revenue per bulan", mode="query")

    # Check what series the LLM produced
    print(result.to_markdown())
    # ## Available Charts
    # - `revenue_trend` — Monthly revenue trend
    #   x: `month` | y: `revenue` | 3 rows

    # Access raw data
    df = result.to_dataframe("revenue_trend")
    print(df)

    # Use pandas, matplotlib, seaborn, Plotly, Vega-Lite, or anything else you prefer.
    # hiveframe only guarantees the structured data payload.

asyncio.run(main())
```

---

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
