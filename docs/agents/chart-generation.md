# Chart Generation Guide

## Overview

Hiveframe provides two mechanisms for generating chart-ready data from DataFrames:

1. **ChartGenerator** - Deterministic, no-LLM chart generation
2. **MultiFrameAgent** - LLM-powered intelligent chart generation with analysis

> hiveframe outputs data, not visualizations. What you do with that data is entirely up to you.

## ChartGenerator — Programmatic Chart Creation

Use `ChartGenerator` when you want full control over which columns become which chart axes.

### Basic Usage

```python
import hiveframe as hf
from hiveframe.agent import ChartGenerator

# Create a DataFrame
df = hf.DFrame({
    "Category": ["A", "B", "C", "A", "B"],
    "Revenue": [100, 200, 150, 120, 180],
    "Region": ["East", "West", "East", "West", "East"],
})

# Generate a bar chart
gen = ChartGenerator(df, frame_label="sales")

# Value counts (simple bar chart data)
series = gen.generate("bar", x="Category")
print(series.to_dict())

# Aggregated bar chart with grouping
series = gen.generate(
    "bar",
    x="Category",
    y="Revenue",
    group_by="Region",
    agg="sum",
)
print(series.to_dataframe())
```

### Supported Chart Types

- **bar** - Bar chart (default)
- **line** - Line chart
- **area** - Area chart
- **scatter** - Scatter plot
- **pie** - Pie chart
- **histogram** - Histogram (distribution)
- **heatmap** - Heatmap (requires group_by)

### Method Signature

```python
gen.generate(
    chart_type: str,           # "bar", "line", "area", "scatter", "pie", "histogram", "heatmap"
    x: str,                    # x-axis column
    y: str | None = None,      # y-axis column (optional for bar/pie — uses value_counts)
    group_by: str | None = None,  # grouping column (required for heatmap)
    agg: str = "count",        # aggregation: "count", "sum", "mean", "median", "min", "max"
    top_n: int | None = 20,    # limit result rows
    title: str = "",           # custom chart title
    sort_by: str | None = None,  # column to sort by
    ascending: bool = False,   # sort direction
) -> SeriesSpec
```

### Configuration Suggestions

Get recommended column assignments for a chart type:

```python
config = gen.suggest_config("scatter")
print(config)
# {
#   "chart_type": "scatter",
#   "numeric_columns": ["Revenue", "Count"],
#   "category_columns": ["Category", "Region"],
#   "suggested_x": "Revenue",
#   "suggested_y": "Count",
#   "suggested_group_by": "Category",
# }
```

## SeriesSpec — Chart Data Structure

Both `ChartGenerator` and `MultiFrameAgent` return `SeriesSpec` objects that store:

```python
@dataclass
class SeriesSpec:
    label: str            # series identifier
    x: list               # x-axis values
    y: list               # y-axis values
    x_label: str = ""     # x-axis label
    y_label: str = ""     # y-axis label
    series_type: str = "bar"  # chart hint only
```

### Using Series Data

```python
# Convert to dict for API responses or custom front-end renderers
payload = series.to_dict()

# Convert to DataFrame for pandas/matplotlib/seaborn/etc.
df_plot = series.to_dataframe()
```

## MultiFrameAgent — Intelligent Chart Generation

Use `MultiFrameAgent` when you want the LLM to decide which chart type is most appropriate.

### Basic Usage

```python
import asyncio
from hiveframe.agent import MultiFrameAgent

async def main():
    df = hf.DFrame({
        "Date": pd.date_range("2026-01-01", periods=10),
        "Sales": [100, 120, 110, 150, 140, 160, 170, 180, 190, 200],
        "Region": ["East", "West"] * 5,
    })

    agent = MultiFrameAgent(
        frames={"sales": df},
        provider="anthropic",
    )

    # Request analysis + chart
    result = await agent.analyze(
        instruction="Show sales trends over time by region. Use a line chart.",
    )

    print(result.analysis)

    # Use generated chart-ready data
    if result.series:
        print(result.series[0].to_dict())

asyncio.run(main())
```

### How LLM Chooses Chart Type

The LLM responds with JSON including a `series_type` field:

```json
{
  "action": "analyze",
  "reasoning": "...",
  "analysis": "...",
  "series": [
    {
      "label": "sales_by_region_over_time",
      "x": ["2026-01-01", "2026-01-02"],
      "y": [100, 120],
      "x_label": "Date",
      "y_label": "Sales",
      "series_type": "line"
    }
  ]
}
```

The `SeriesSpec` stores this metadata as pure data so you can render it with any visualization stack.

## Chart Type Selection Guidelines

### When to Use Each Type

| Chart Type | Best For | Example |
|---|---|---|
| **bar** | Category comparisons, counts | Product sales by category |
| **line** | Time series, trends | Stock price over time |
| **area** | Cumulative trends, composition | Revenue stacked by product |
| **scatter** | Correlation, outliers | Age vs income |
| **pie** | Part-to-whole, percentages | Market share by vendor |
| **histogram** | Distributions | Income distribution |
| **heatmap** | Two-dimensional patterns | Temperature by hour/day |

### LLM Instructions

When requesting a chart from an LLM agent, be specific:

```python
# ✗ Vague — LLM may choose wrong type
await agent.analyze("Show data")

# ✓ Clear — LLM knows what you want
await agent.analyze("Show sales trends over time using a line chart")
await agent.analyze("Compare product categories by total revenue using a bar chart")
await agent.analyze("Show the correlation between price and quantity using a scatter plot")
```

## Combining Charts and Analysis

`MultiFrameResult` holds both analysis text AND chart data:

```python
result = await agent.analyze("Analyze sales by region...")

print(result.analysis)        # Text insights
print(result.insights)         # Structured findings
print(len(result.series))      # Number of charts generated

for spec in result.series:
    payload = spec.to_dict()
    print(payload["label"], len(payload["x"]))
```

## Performance Tips

### ChartGenerator

1. **Use top_n to limit output**: `top_n=20` returns only top 20 rows
2. **Specify agg early**: `agg="sum"` aggregates before export
3. **No LLM overhead**: Zero latency compared to agent mode

### MultiFrameAgent

1. **Use columns_hint** to reduce context size:
   ```python
   result = await agent.analyze(
       "Show revenue trends",
       columns_hint={"sales": ["Date", "Revenue", "Region"]},
   )
   ```

2. **Request specific chart type** in instruction to guide LLM

3. **Reuse agent instance** for multiple analyses

## API Reference

### ChartGenerator

- `available_columns() -> list[str]` — List all columns
- `suggest_config(chart_type: str) -> dict` — Get column suggestions
- `generate(...) -> SeriesSpec` — Generate chart

### SeriesSpec

- `to_dataframe() -> pd.DataFrame` — Get aggregated data
- `to_dict() -> dict` — Serialize chart-ready data
- `from_dict(payload) -> SeriesSpec` — Deserialize series payload

### MultiFrameResult

- `get_series(name: str) -> SeriesSpec | None` — Find series by label
- `to_dataframe(name: str) -> pd.DataFrame` — Convert one series to DataFrame
- `to_dict() -> dict` — Serialize complete analysis output

---

**Learn more**: [Agent Documentation](../agents.md) | [API Reference](../api/index.md)

