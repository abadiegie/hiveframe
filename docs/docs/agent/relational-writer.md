# RelationalAgentWriter

Use `RelationalAgentWriter` to normalize a single target frame using cross-frame relational context.
This enables LLM annotation workflows where each row's value depends on related data from other frames.

## When to use

- Annotating child rows that need parent context (e.g., comment stance depends on parent post)
- Enriching data that requires cross-frame lookup (e.g., comment → post → category)
- Multi-frame normalization where single-frame context is insufficient
- Self-referencing relations (e.g., nested comments: child → parent comment)

## Key Concepts

### FrameRelation

Defines a single cross-frame lookup rule.

```python
@dataclass
class FrameRelation:
    from_column: str           # Foreign key column in target frame
    to_column: str             # Primary key column in context frame
    context_frame: str         # Frame label to lookup from (or "self")
    include_columns: list[str] # Columns to include in context (empty = all)
    relation_type: str         # "many_to_one" or "one_to_many"
    context_label: str         # Display name in context (auto: context_frame)
    max_related: int           # For one_to_many, max rows to include (default 10)
```

### Relation Types

| Type | Lookup | Result | Use Case |
|------|--------|--------|----------|
| `many_to_one` | One FK value | At most 1 row | Comment → Parent Post |
| `one_to_many` | One FK value | Multiple rows (limited by `max_related`) | Post → All Comments |
| self-referencing | `context_frame="self"` | Lookup within same frame | Nested Comments (child → parent) |

## API Reference

### Constructor

```python
RelationalAgentWriter(
    target_frame: DFrame,
    relations: list[FrameRelation],
    context_frames: dict[str, DFrame] | None = None,
    agent_id: str = "relational_agent",
    confidence_threshold: float = 0.6,
    author_type: str = "llm_normalization",
)
```

**Parameters:**

- `target_frame` — Frame to write results to
- `relations` — List of `FrameRelation` definitions
- `context_frames` — Dictionary of context frames by label (required if not all are "self")
- `agent_id` — Agent ID for write operations
- `confidence_threshold` — Minimum LLM confidence (0-1) to write a cell; default 0.6
- `author_type` — Author type for write log; default "llm_normalization"

**Raises:**

- `ValueError` — If a relation references a `context_frame` not in `context_frames`

### Main Method

```python
result = await writer.stream_normalize_relational(
    target_column: str,
    instruction: str,
    chunk_size: int = 10,
    provider: str = "anthropic",
    model: str | None = None,
    anthropic_api_key: str | None = None,
    openai_api_key: str | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, int]
```

**Parameters:**

- `target_column` — Column to populate (created if missing)
- `instruction` — Annotation guideline for the LLM (e.g., "Classify as positive, negative, or neutral")
- `chunk_size` — Rows per LLM call; default 10 (balance between token cost and latency)
- `provider` — LLM provider: `"anthropic"` or `"openai"`
- `model` — Model name (optional; uses provider default if not specified)
- `anthropic_api_key` — For Anthropic provider; uses `ANTHROPIC_API_KEY` env var if None
- `openai_api_key` — For OpenAI provider; uses `OPENAI_API_KEY` env var if None
- `on_progress` — Callback: `on_progress(processed, total)` for progress tracking

**Returns:**

Dictionary with:

- `"written"` — Number of cells successfully written
- `"skipped"` — Number of cells skipped (low confidence or LLM error)
- `"total"` — Total rows processed

## Usage Examples

### Example 1: Comment Stance (Many-to-One)

Classify comment stance using parent post context.

```python
import asyncio
import os
import hiveframe as hf
from hiveframe.agent import RelationalAgentWriter, FrameRelation

async def main() -> None:
    # Posts frame
    posts = hf.DFrame({
        "post_id": [1, 2, 3],
        "topic": ["tech", "politics", "sports"],
        "title": ["New Python feature", "Election 2026", "World Cup"],
    })

    # Comments frame (many comments per post)
    comments = hf.DFrame({
        "comment_id": [101, 102, 103, 104],
        "post_id": [1, 1, 2, 3],
        "text": ["Great!", "Not useful", "Vote now", "Amazing game"],
        "stance": [None, None, None, None],  # To be filled
    })

    # Define relation: each comment points to one post
    relation = FrameRelation(
        from_column="post_id",
        to_column="post_id",
        context_frame="posts",
        include_columns=["topic", "title"],  # Only relevant columns
        relation_type="many_to_one",
        context_label="Parent Post",
    )

    # Create writer
    writer = RelationalAgentWriter(
        target_frame=comments,
        relations=[relation],
        context_frames={"posts": posts},
        confidence_threshold=0.7,
    )

    # Annotate stance column
    result = await writer.stream_normalize_relational(
        target_column="stance",
        instruction="Classify comment stance toward the post topic as: positive, negative, or neutral",
        chunk_size=5,
        provider="anthropic",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        on_progress=lambda p, t: print(f"Progress: {p}/{t}"),
    )

    print(f"Result: {result}")
    # Expected: {"written": 4, "skipped": 0, "total": 4}

    # Read results
    updated = comments.read_fresh()
    print(updated[["comment_id", "text", "stance"]])

asyncio.run(main())
```

### Example 2: Post Review Category (One-to-Many)

Classify post category using aggregate info from related comments.

```python
import asyncio
import os
import hiveframe as hf
from hiveframe.agent import RelationalAgentWriter, FrameRelation

async def main() -> None:
    # Posts (target)
    posts = hf.DFrame({
        "post_id": [1, 2, 3],
        "title": ["Python Tips", "Election News", "Sports"],
        "category": [None, None, None],  # To be filled
    })

    # Comments (context)
    comments = hf.DFrame({
        "comment_id": [101, 102, 103, 104, 105],
        "post_id": [1, 1, 2, 2, 3],
        "text": [
            "Good guide",
            "Very helpful",
            "Breaking news",
            "Important",
            "Exciting match",
        ],
    })

    # Define relation: each post has many comments
    relation = FrameRelation(
        from_column="post_id",
        to_column="post_id",
        context_frame="comments",
        include_columns=["text"],  # Include comment texts
        relation_type="one_to_many",
        context_label="User Comments",
        max_related=5,  # Show up to 5 comments per post
    )

    writer = RelationalAgentWriter(
        target_frame=posts,
        relations=[relation],
        context_frames={"comments": comments},
    )

    result = await writer.stream_normalize_relational(
        target_column="category",
        instruction=(
            "Based on post title and comment texts, categorize the post as: "
            "technology, news, or entertainment"
        ),
        chunk_size=10,
        provider="anthropic",
    )

    print(f"Result: {result}")

    # Read results
    updated = posts.read_fresh()
    print(updated[["post_id", "title", "category"]])

asyncio.run(main())
```

### Example 3: Self-Referencing (Nested Comments)

Classify comment type based on parent comment context (comments may reply to other comments).

```python
import asyncio
import os
import hiveframe as hf
from hiveframe.agent import RelationalAgentWriter, FrameRelation

async def main() -> None:
    # Nested comments (some are replies to other comments)
    comments = hf.DFrame({
        "comment_id": [1, 2, 3, 4, 5],
        "parent_id": [None, 1, 1, 2, 3],  # None = top-level, else parent comment_id
        "text": [
            "Main observation",
            "I agree!",
            "I disagree",
            "Good point",
            "Not relevant",
        ],
        "type": [None, None, None, None, None],  # To be filled: "original", "agreement", "rebuttal"
    })

    # Define self-referencing relation: reply → parent comment
    relation = FrameRelation(
        from_column="parent_id",
        to_column="comment_id",
        context_frame="self",  # Lookup within same frame
        include_columns=["text"],
        relation_type="many_to_one",
        context_label="Parent Comment",
    )

    writer = RelationalAgentWriter(
        target_frame=comments,
        relations=[relation],
        confidence_threshold=0.65,
    )

    result = await writer.stream_normalize_relational(
        target_column="type",
        instruction=(
            "Classify this comment as: 'original' (top-level), "
            "'agreement' (supports parent), or 'rebuttal' (opposes parent)"
        ),
        chunk_size=5,
        provider="anthropic",
    )

    print(f"Result: {result}")

    updated = comments.read_fresh()
    print(updated[["comment_id", "parent_id", "text", "type"]])

asyncio.run(main())
```

## Context Format

The LLM receives enriched context like this:

```
Row 0 (comment_id=101, post_id=1, text=Great!):
  Data:
    comment_id: 101
    post_id: 1
    text: Great!
    stance: None

  [Parent Post] (lookup post_id=1):
    post_id: 1
    topic: tech
    title: New Python feature

Row 1 (comment_id=102, post_id=1, text=Not useful):
  Data:
    comment_id: 102
    post_id: 1
    text: Not useful
    stance: None

  [Parent Post] (lookup post_id=1):
    post_id: 1
    topic: tech
    title: New Python feature
```

For one-to-many relations:

```
Row 0 (post_id=1, title=Python Tips):
  Data:
    post_id: 1
    title: Python Tips
    category: None

  [User Comments] (2 found, max 5 shown):
    1. comment_id: 101, text: Good guide
    2. comment_id: 102, text: Very helpful
```

## Response Format

The LLM must return raw JSON:

```json
{
  "action": "batch_enrich",
  "reasoning": "Based on parent post topic and comment sentiment...",
  "operations": [
    {
      "cell_id": "comments::stance_0",
      "value": "positive",
      "confidence": 0.95
    },
    {
      "cell_id": "comments::stance_1",
      "value": "negative",
      "confidence": 0.87
    }
  ]
}
```

Only operations with `confidence >= confidence_threshold` are written.

## Confidence Threshold

By default, only cells with confidence ≥ 0.6 are written.

| Range | Interpretation |
|-------|----------------|
| 0.95-1.00 | Very certain |
| 0.80-0.94 | Confident |
| 0.60-0.79 | Moderate |
| < 0.60 | Skipped (below threshold) |

Adjust via constructor:

```python
writer = RelationalAgentWriter(
    target_frame=comments,
    relations=[relation],
    context_frames={"posts": posts},
    confidence_threshold=0.8,  # Stricter: require 80%+ confidence
)
```

## Progress Tracking

Use `on_progress` callback to monitor processing:

```python
def on_progress(processed: int, total: int) -> None:
    pct = 100 * processed / total if total > 0 else 0
    print(f"Progress: {processed}/{total} ({pct:.0f}%)")

result = await writer.stream_normalize_relational(
    target_column="stance",
    instruction="...",
    on_progress=on_progress,
)
```

## Performance Tips

1. **Chunk size** — Larger chunks (e.g., 20-50) reduce API calls but increase token cost
2. **Include columns** — Limit context frame columns to only what's needed (token efficiency)
3. **Confidence threshold** — Stricter threshold (e.g., 0.8) means fewer writes but higher confidence
4. **Cache building** — One-time cache build at start; subsequent lookups are O(1)

## Error Handling

- Per-chunk LLM errors are logged as warnings; processing continues with remaining chunks
- Missing `to_column` in context frame is logged; cache entry remains empty
- Missing `target_column` is created automatically
- Missing `include_columns` in context frame are silently skipped

Example error log:

```
LLM call failed for chunk 0-9: 429 rate limit exceeded
chunk 10-19: written=8 skipped=2
chunk 0-9: skipped=10 (error from previous attempt)
```

## Runnable Example

See `examples/relational_writer_usage.py` for complete working examples (no API key required).
Demonstrates:
- Many-to-one relation (comment → parent post)
- One-to-many relation (post → comments)
- Self-referencing (nested comments)

## See Also

- [MultiFrameAgent](../agent/columns-hint.md) — Cross-frame analysis without writes
- [AgentWriter API](../api/agent.md) — Single-frame transactional writes
- [Guides → Iterative Agent](../guides/iterative-agent.md) — Iterative LLM workflows

