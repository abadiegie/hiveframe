# Chart Generator Fix Summary

## Problem
Users consistently received bar charts regardless of what chart type they requested from `MultiFrameAgent` or `ChartGenerator`. The issue was that `SeriesSpec` didn't have a `chart_type` field to preserve the intended chart type.

## Root Cause Analysis
1. **No chart_type field**: `SeriesSpec` dataclass lacked a field to store which chart type should be used
2. **Default fallbacks**: `to_plotly_figure()` had hardcoded `chart_type="line"` default
3. **LLM couldn't specify**: MultiFrameAgent prompt didn't include `chart_type` in series response format
4. **No pass-through**: LLM response parsing ignored any chart type information

## Solution Implemented

### 1. Added `chart_type` Field to SeriesSpec
```python
@dataclass
class SeriesSpec:
    # ... existing fields ...
    chart_type: str = "bar"  # Default chart type for this series
```

### 2. Updated SeriesSpec.to_plotly_figure()
**Before**: Always defaulted to `"line"`
```python
def to_plotly_figure(self, chart_type: str = "line", **kwargs):
```

**After**: Uses self.chart_type as fallback
```python
def to_plotly_figure(self, chart_type: str | None = None, **kwargs):
    effective_chart_type = chart_type or self.chart_type or "bar"
```

### 3. ChartGenerator Now Sets chart_type
```python
return SeriesSpec(
    # ...
    chart_type=chart_type,  # Preserves the requested chart type
    # ...
)
```

### 4. MultiFrameAgent Parses chart_type from LLM Response
```python
def _plan_to_result(plan: dict[str, Any]) -> MultiFrameResult:
    for raw_series in plan.get("series", []):
        chart_type = str(raw_series.get("chart_type", "bar")).lower()
        # ... validation ...
        series.append(SeriesSpec(
            # ...
            chart_type=chart_type,
            # ...
        ))
```

### 5. Updated LLM Prompt Format
```json
{
  "series": [
    {
      "name": "chart_name",
      "description": "...",
      "chart_type": "bar|line|area|scatter|pie|histogram|heatmap",
      "suggested_x": "column",
      "suggested_y": "column",
      "data": [...]
    }
  ]
}
```

### 6. MultiFrameResult Respects Series chart_type
```python
def to_plotly_figure(self, name: str, chart_type: str | None = None, **kwargs):
    spec = self.get_series(name)
    return spec.to_plotly_figure(chart_type=chart_type)  # Uses series default
```

## Changes Summary

### Files Modified
1. **agent/result.py**
   - Added `chart_type: str = "bar"` field to SeriesSpec
   - Updated `to_plotly_figure()` to use self.chart_type as fallback
   - Updated `save_chart()` to use self.chart_type as fallback
   - Updated MultiFrameResult methods to respect series chart_type

2. **agent/chart_generator.py**
   - Updated return statement to include `chart_type=chart_type`

3. **agent/multi_agent.py**
   - Updated `_plan_to_result()` to parse and validate `chart_type`
   - Includes fallback and warning for invalid chart types

4. **agent/prompt.py**
   - Added `chart_type` field to LLM response format documentation
   - Added rule: "Specify the appropriate chart_type: bar, line, area, scatter, pie, histogram, heatmap"

5. **tests/test_chart_generator.py**
   - Updated 10+ test cases to verify `chart_type` is set correctly
   - All 25 tests passing

### Documentation Added
- **docs/agents/chart-generation.md** - Comprehensive guide covering:
  - ChartGenerator usage
  - SeriesSpec structure
  - Supported chart types
  - Chart type selection guidelines
  - Performance tips
  - API reference

## Behavior Changes

### Before Fix
```python
# ChartGenerator
series = gen.generate("line", x="Date", y="Sales")
series.chart_type  # ❌ Raises AttributeError

# When rendering
fig = series.to_plotly_figure()
# Always creates a line chart (default) regardless of what type was generated
```

### After Fix
```python
# ChartGenerator
series = gen.generate("line", x="Date", y="Sales")
series.chart_type  # ✅ Returns "line"

# When rendering
fig = series.to_plotly_figure()
# Creates a line chart (uses self.chart_type)

fig = series.to_plotly_figure(chart_type="bar")
# Override: Creates a bar chart
```

## Testing
✅ All 217 tests passing (2 skipped)
- ChartGenerator tests: 25/25 passed
- Series output tests: 34/34 passed
- Full test suite: No regressions

## Backward Compatibility
✅ Fully backward compatible:
- Default `chart_type="bar"` matches intuitive default
- `to_plotly_figure(chart_type=None)` uses series default
- Existing code continues to work
- No breaking API changes

## User Impact
Users can now:
1. **Programmatic**: `gen.generate("scatter", ...)` creates scatter plot
2. **LLM-powered**: LLM can select appropriate chart type
3. **Override**: Still manually override: `.to_plotly_figure(chart_type="pie")`
4. **Automatic**: No need to manually specify chart type in most cases

## Commits
1. **f43c8d9** - feat: add chart_type field to SeriesSpec for proper chart rendering
2. **15a6ff2** - docs: add chart generation guide

