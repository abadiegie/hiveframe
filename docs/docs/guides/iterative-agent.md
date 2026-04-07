# Iterative MultiFrameAgent

## Overview

By default (`max_retries=0`), `MultiFrameAgent` runs a single **query → execute → analyze**
pass. Activating `max_retries > 0` enables the **iterative review/retry loop** — the agent
reviews its own query results, decides if they are sufficient, and retries with corrections
until it converges or reaches the retry limit.

```
max_retries=0  →  query → execute → analyze
max_retries=N  →  (query → execute → review → [retry if needed]) × N+1 → final analyze
```

---

## Activation

```python
result = await agent.analyze(
    "City mana score tinggi tapi stock rendah?",
    mode="query",
    max_retries=2,      # allow up to 2 retries after first attempt
)
```

---

## The Loop

Each iteration:

1. **Query generation** — LLM generates pandas queries per frame schema
2. **Execute** — queries run through `_safe_eval()` sandbox
3. **Review** — LLM evaluates accumulated results vs. the original instruction
4. **Verdict** — one of `accepted | partial | error | plan | rejected | merge`
5. **Continue or stop** based on verdict

After the loop ends (convergence or max retries), a final **analysis LLM call** produces
`MultiFrameResult` from all accumulated results.

```
Attempt 1:  generate → execute → review → partial
Attempt 2:  generate (missing parts only) → execute → review → accepted
Final:      analysis → MultiFrameResult
```

---

## Review Verdicts

| Verdict | Meaning | What happens next |
|---|---|---|
| `accepted` | Results fully answer the instruction | Loop stops, proceed to final analysis |
| `merge` | Sufficient partial results across multiple queries | Loop stops, proceed to final analysis |
| `partial` | Some results useful, specific parts missing | Retry — only missing labels re-queried; accepted labels skipped |
| `error` | One or more queries failed technically | Retry — only failed queries re-run with corrected approach |
| `plan` | Needs columns not yet visible in schema | Inject extra schema info, retry with additional context |
| `rejected` | Results not relevant at all — wrong approach | Clear accumulated results, full retry from scratch |

---

## ReviewVerdict Structure

```python
@dataclass
class ReviewVerdict:
    status: str                         # verdict string above
    reason: str                         # brief explanation
    reflection: str                     # what to fix on next attempt
    missing_parts: list[str]            # description of missing data
    suggested_queries: dict[str, str]   # frame_label → corrected query
    accepted_labels: list[str]          # labels already OK (partial mode)
    needs_columns: list[str]            # columns to inject (plan mode)
    merge_ready: bool                   # True when merge verdict
```

---

## MultiFrameResult Iterative Metadata

```python
result.review_history    # list[ReviewVerdict] — one per attempt
result.total_llm_calls   # int — total LLM calls (query + review + final)
result.converged         # bool — True if last verdict was accepted or merge
result.final_verdict     # str — last verdict status
```

### LLM call count formula

```
total_llm_calls = (attempts × 2) + 1
                   └── query+review per attempt    └── final analysis
```

Example: `max_retries=2`, converged at attempt 2 → `(2 × 2) + 1 = 5` calls.

---

## Full Example

```python
import asyncio
import hiveframe as hf
from hiveframe.agent import MultiFrameAgent

async def main() -> None:
    sales = hf.DFrame({
        "product_id": ["A", "B", "C", "D"],
        "qty_sold": [500, 120, 340, 80],
    })
    inventory = hf.DFrame({
        "product_id": ["A", "B", "C", "D"],
        "stock_remaining": [5, 200, 12, 300],
    })

    agent = MultiFrameAgent(
        frames={"sales": sales, "inventory": inventory},
        provider="anthropic",
    )

    result = await agent.analyze(
        "Produk mana yang penjualannya tinggi tapi stok menipis?",
        mode="query",
        max_retries=2,
    )

    # Narrative analysis
    print(result.analysis)

    # Iterative metadata
    print(f"LLM calls: {result.total_llm_calls}")
    print(f"Converged: {result.converged}")
    print(f"Final verdict: {result.final_verdict}")

    # Per-attempt review history
    for i, verdict in enumerate(result.review_history, 1):
        print(f"Attempt {i}: {verdict.status} — {verdict.reason}")

    # Full markdown report (includes iteration history + chart series)
    print(result.to_markdown())

asyncio.run(main())
```

### Example output

```
LLM calls: 5
Converged: True
Final verdict: accepted

Attempt 1: partial — Sales data OK, inventory query used wrong column name
Attempt 2: accepted — Both datasets available and sufficient

## Analysis
Produk A memiliki penjualan tertinggi (500 unit) dengan stok tersisa hanya 5 unit...

## Key Insights
1. **Produk A: critical stock risk**
   Sources: `sales`, `inventory`
   Confidence: 95%

## Iteration History
1. ~ **partial** - Sales data OK, inventory query used wrong column name
2. ✓ **accepted** - Both datasets available and sufficient
```

---

## With Chart Series Output

The iterative agent also populates `result.series` from the final analysis:

```python
result = await agent.analyze(
    "Bandingkan qty_sold vs stock_remaining per produk",
    mode="query",
    max_retries=2,
)

# Access chart data
df = result.to_dataframe("sales_vs_stock")
fig = result.to_plotly_figure("sales_vs_stock", chart_type="bar")
fig.show()
```

See [API Reference → AgentWriter](../api/agent.md#seriesspec) for full SeriesSpec docs.

---

## Tips

- Start with `max_retries=1` — covers most real-world cases with one retry
- Use `max_retries=0` (default) for simple single-frame queries where one pass is enough
- `result.total_llm_calls` is your cost indicator — monitor it per query type
- `result.converged=False` with `final_verdict="unknown"` means max retries hit without convergence — consider increasing `max_retries` or simplifying the instruction
- The agent accumulates results across attempts (`partial` mode) — earlier successful results are not re-queried

