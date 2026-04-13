# Chart Generation Guide

## Overview

Hiveframe provides two mechanisms for generating charts from DataFrames:

1. **ChartGenerator** - Deterministic, no-LLM chart generation
2. **MultiFrameAgent** - LLM-powered intelligent chart generation with analysis

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

# Value counts (simple bar chart)
series = gen.generate("bar", x="Category")
fig = series.to_plotly_figure()
fig.show()

# Aggregated bar chart with grouping
series = gen.generate(
    "bar",
    x="Category",
    y="Revenue",
    group_by="Region",
    agg="sum",
)
fig = series.to_plotly_figure()
fig.show()
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
    name: str                            # snake_case identifier
    description: str                     # chart title
    data: list[dict[str, Any]]          # aggregated rows
    suggested_x: str = ""               # x-axis column
    suggested_y: str | list[str] = ""   # y-axis column(s)
    suggested_group_by: str | None = None  # grouping column
    chart_type: str = "bar"             # the chart type to use
    unit: str = ""                      # y-axis unit label
    source_frames: list[str] = []       # which frames contributed
```

### Rendering Charts

```python
# Render with auto-detected chart type (from series.chart_type)
fig = series.to_plotly_figure()
fig.show()

# Override chart type for this render
fig = series.to_plotly_figure(chart_type="line")
fig.show()

# Save as PNG
path = series.save_chart("output/my_chart.png")
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

    # Render the generated chart
    if result.series:
        fig = result.series[0].to_plotly_figure()
        fig.show()

asyncio.run(main())
```

### How LLM Chooses Chart Type

The LLM responds with JSON including a `chart_type` field:

```json
{
  "action": "analyze",
  "reasoning": "...",
  "analysis": "...",
  "series": [
    {
      "name": "sales_by_region_over_time",
      "description": "Sales trends by region",
      "chart_type": "line",
      "suggested_x": "Date",
      "suggested_y": "Sales",
      "suggested_group_by": "Region",
      "data": [...]
    }
  ]
}
```

The `SeriesSpec` stores this `chart_type` and uses it as the default when rendering.

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
    fig = spec.to_plotly_figure()
    fig.show()

# Save all charts to directory
paths = result.save_all_charts("output/")
```

## Performance Tips

### ChartGenerator

1. **Use top_n to limit output**: `top_n=20` returns only top 20 rows
2. **Specify agg early**: `agg="sum"` aggregates before plotting
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
- `to_plotly_figure(chart_type=None) -> go.Figure` — Render chart
- `save_chart(path, chart_type=None) -> str` — Save as PNG

### MultiFrameResult

- `get_series(name: str) -> SeriesSpec | None` — Find series by name
- `to_plotly_figure(name, chart_type=None) -> go.Figure` — Render specific chart
- `save_all_charts(output_dir, chart_type=None) -> dict` — Save all charts

---

**Learn more**: [Agent Documentation](../agents.md) | [API Reference](../api/index.md)

