# MultiFrameAgent `columns_hint`

Use `columns_hint` to send only relevant columns to the LLM.
This significantly reduces token usage for wide frames.

## When to use

- The frame has many columns (tens to hundreds)
- The model has a small context window
- The instruction only needs a subset of columns

## Signature

```python
result = await agent.analyze(
    instruction,
    mode="sample",            # or "query"
    max_sample_rows=5,         # default is now 5
    columns_hint={
        "news": ["city", "sentiment", "date"],
        "social": ["platform", "sentiment"],
    },
)
```

## Behavior

- `columns_hint=None` -> existing behavior (backward compatible)
- `columns_hint` provided -> context prioritizes selected columns
- Missing hint columns -> skipped + warning log (no exception)
- Empty hint (`[]`) or all-invalid hint -> fallback to all columns
- Hint state resets on every `analyze()` call (no cross-call leakage)

## Quick example

```python
import asyncio
from agent.multi_agent import MultiFrameAgent
from core.dataframe import DFrame


async def main() -> None:
    news = DFrame(
        {
            "title": ["A", "B", "C"],
            "city": ["jakarta", "bandung", "jakarta"],
            "sentiment": ["pos", "neg", "pos"],
            "source": ["media1", "media2", "media1"],
            "unused_big_col": ["...", "...", "..."],
        }
    )

    agent = MultiFrameAgent({"news": news})

    # Without hint: all columns are included in context
    # result = await agent.analyze("Analyze sentiment by city")

    # With hint: only relevant columns are included
    result = await agent.analyze(
        "Analyze sentiment by city",
        mode="sample",
        columns_hint={"news": ["city", "sentiment"]},
    )

    print(result.analysis)


asyncio.run(main())
```

## Runnable sample

See the runnable sample (no API key required):

- `examples/multiframe_columns_hint.py`

