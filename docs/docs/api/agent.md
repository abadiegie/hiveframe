# Agent API

## Overview

Hiveframe agent layer has three main interfaces:

- `AgentWriter` for transactional writes into a single `DFrame`
- `MultiFrameAgent` for analysis across one or many `DFrame` objects (sample/query mode)
- `SeriesSpec` for structured chart-ready data output from LLM analysis

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

| Verdict | Meaning |
|---|---|
| `accepted` | Results fully answer the instruction |
| `merge` | Sufficient partial results across queries |
| `partial` | Some results useful, specific parts missing |
| `error` | One or more queries failed technically |
| `plan` | Needs columns not yet visible in schema |
| `rejected` | Results not relevant — wrong approach |

`MultiFrameResult` also includes iterative metadata:
- `review_history` — `list[ReviewVerdict]`, one per attempt
- `total_llm_calls` — total LLM calls (query + review + final analysis)
- `converged` — `True` if last verdict was `accepted` or `merge`
- `final_verdict` — last verdict status string

For full loop behavior, verdict handling, and examples see
[Guides → Iterative Agent](../guides/iterative-agent.md).

### Result Types

- `MultiFrameResult` — top-level result with `analysis`, `insights`, `series`, metadata
- `FrameInsight` — single insight with `finding`, `frames`, `confidence`
- `SeriesSpec` — chart-ready data series (see below)

All three are exported from `hiveframe.agent`.

---

## SeriesSpec

`SeriesSpec` is a structured data output produced by the LLM analysis. It holds aggregated
data rows ready to be rendered as a chart. The user decides chart type and styling.

```
LLM query results → SeriesSpec.data → pd.DataFrame → plotly Figure → PNG
```

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `name` | `str` | Unique snake_case identifier |
| `description` | `str` | What the series shows — use as chart title |
| `data` | `list[dict]` | Actual aggregated data rows from query results |
| `suggested_x` | `str` | Column suggested for x-axis |
| `suggested_y` | `str \| list[str]` | Column(s) suggested for y-axis |
| `suggested_group_by` | `str \| None` | Column for color grouping |
| `unit` | `str` | Unit label, e.g. `"IDR"`, `"%"` |
| `source_frames` | `list[str]` | Frame labels that produced this data |

### Methods

| Method | Returns | Description |
|---|---|---|
| `to_dataframe()` | `pd.DataFrame` | Convert `data` to a pandas DataFrame |
| `to_plotly_figure(chart_type, **kwargs)` | `go.Figure` | Render with Plotly. `chart_type`: `line\|bar\|scatter\|area\|pie\|histogram` |
| `save_chart(path, chart_type, width, height, scale)` | `str` | Save as PNG, returns absolute path |

Requires `pip install hiveframe[charts]` (`plotly>=5.0`, `kaleido>=0.2`).

### MultiFrameResult chart helpers

`MultiFrameResult.series` is a `list[SeriesSpec]` populated automatically when the LLM
returns series data.

| Method | Description |
|---|---|
| `get_series(name)` | Get `SeriesSpec` by name, `None` if not found |
| `to_dataframe(name)` | Get data as `pd.DataFrame`, empty if not found |
| `to_plotly_figure(name, chart_type, **kwargs)` | Get Plotly figure, raises `KeyError` if not found |
| `save_chart(name, path, chart_type, **kwargs)` | Save one series as PNG |
| `save_all_charts(output_dir, chart_type)` | Save all series as PNGs, skips failures silently |

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

    # User decides chart type
    fig = result.to_plotly_figure("revenue_trend", chart_type="line", title="Revenue Q1")
    fig.show()

    # Save as PNG (requires kaleido)
    path = result.save_chart("revenue_trend", "output/revenue.png", chart_type="bar")
    print(f"Saved: {path}")

    # Save all series at once
    paths = result.save_all_charts("output/charts/")
    for name, p in paths.items():
        print(f"{name}: {p}")

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
