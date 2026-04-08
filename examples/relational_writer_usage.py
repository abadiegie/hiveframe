"""
Runnable example: RelationalAgentWriter for cross-frame LLM normalization.

This example demonstrates how to use RelationalAgentWriter with mock LLM
(no API key required for testing). See docs/docs/agent/relational-writer.md
for full API documentation.

Example scenarios:
1. Comment stance classification using parent post context
2. Post category classification using comment aggregate info
3. Nested comment type classification with self-referencing
"""

import asyncio
import json
from dataclasses import dataclass

import hiveframe as hf
from hiveframe.agent import RelationalAgentWriter, FrameRelation


@dataclass
class MockLLMResponse:
    """Mock LLM response for testing without API calls."""

    action: str = "batch_enrich"
    reasoning: str = "Mock LLM reasoning"
    operations: list = None

    def __post_init__(self) -> None:
        if self.operations is None:
            self.operations = []


async def mock_llm_call_stance(messages: list[dict[str, str]]) -> str:
    """
    Mock LLM for comment stance classification.
    Returns JSON response based on message content.
    """
    # In real usage, this would call Claude/GPT
    # For demo, we return deterministic responses based on text keywords
    user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")

    operations = []
    # Parse rows from context and generate mock responses
    lines = user_msg.split("\n")
    row_idx = 0
    for line in lines:
        if line.startswith("Row"):
            # Extract row index from "Row 0 (...)" format
            try:
                row_idx = int(line.split()[1])
            except (ValueError, IndexError):
                pass
        elif "text:" in line.lower() or "comment" in line.lower():
            # Simple heuristic for stance
            if any(word in line.lower() for word in ["good", "great", "helpful", "excellent"]):
                stance = "positive"
                confidence = 0.9
            elif any(
                word in line.lower() for word in ["bad", "not", "useless", "waste", "disagree"]
            ):
                stance = "negative"
                confidence = 0.85
            else:
                stance = "neutral"
                confidence = 0.7

            if operations and operations[-1].get("cell_id", "").endswith(f"_{row_idx}"):
                continue
            operations.append(
                {"cell_id": f"comments::stance_{row_idx}", "value": stance, "confidence": confidence}
            )

    response = MockLLMResponse(operations=operations)
    return json.dumps(response.__dict__)


async def mock_llm_call_category(messages: list[dict[str, str]]) -> str:
    """Mock LLM for post category classification."""
    operations = []
    row_idx = 0
    user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")

    lines = user_msg.split("\n")
    for line in lines:
        if line.startswith("Row"):
            try:
                row_idx = int(line.split()[1])
            except (ValueError, IndexError):
                pass
        elif "title:" in line.lower():
            if any(word in line.lower() for word in ["python", "code", "tech", "api"]):
                category = "technology"
                confidence = 0.92
            elif any(word in line.lower() for word in ["news", "election", "breaking"]):
                category = "news"
                confidence = 0.88
            elif any(word in line.lower() for word in ["sports", "game", "match", "world cup"]):
                category = "entertainment"
                confidence = 0.90
            else:
                category = "other"
                confidence = 0.6

            if operations and operations[-1].get("cell_id", "").endswith(f"_{row_idx}"):
                continue
            operations.append(
                {
                    "cell_id": f"posts::category_{row_idx}",
                    "value": category,
                    "confidence": confidence,
                }
            )

    response = MockLLMResponse(operations=operations)
    return json.dumps(response.__dict__)


async def example_comment_stance() -> None:
    """Example 1: Classify comment stance using parent post context (many-to-one)."""
    print("\n" + "=" * 60)
    print("Example 1: Comment Stance Classification (Many-to-One Relation)")
    print("=" * 60)

    # Posts frame
    posts = hf.DFrame(
        {
            "post_id": [1, 2, 3],
            "topic": ["tech", "politics", "sports"],
            "title": ["New Python feature", "Election 2026", "World Cup Final"],
        }
    )

    # Comments frame (to be annotated)
    comments = hf.DFrame(
        {
            "comment_id": [101, 102, 103, 104, 105],
            "post_id": [1, 1, 2, 2, 3],
            "text": ["Great feature!", "Not useful at all", "Vote now!", "Disappointing", "Amazing game!"],
            "stance": [None, None, None, None, None],
        }
    )

    print("\nPosts:")
    print(posts.read_fresh()[["post_id", "topic", "title"]])
    print("\nComments (before annotation):")
    print(comments.read_fresh()[["comment_id", "post_id", "text", "stance"]])

    # Define relation: each comment → one post
    relation = FrameRelation(
        from_column="post_id",
        to_column="post_id",
        context_frame="posts",
        include_columns=["topic", "title"],
        relation_type="many_to_one",
        context_label="Parent Post",
    )

    # Create writer
    writer = RelationalAgentWriter(
        target_frame=comments,
        relations=[relation],
        context_frames={"posts": posts},
        confidence_threshold=0.65,
    )

    print("\nAnnotating comments using parent post context...")

    # Override LLM call for demo (in production, use real LLM)
    result = await writer.stream_normalize_relational(
        target_column="stance",
        instruction="Classify comment stance as: positive, negative, or neutral",
        chunk_size=10,
        provider="anthropic",  # provider is ignored; we use mock
    )

    # Monkey-patch the LLM caller for demo purposes
    writer._make_llm_caller = lambda *args, **kwargs: mock_llm_call_stance

    result = await writer.stream_normalize_relational(
        target_column="stance",
        instruction="Classify comment stance as: positive, negative, or neutral",
        chunk_size=10,
        provider="anthropic",
    )

    print(f"\nResult: {result}")

    # Read results
    updated = comments.read_fresh()
    print("\nComments (after annotation):")
    print(updated[["comment_id", "post_id", "text", "stance"]])


async def example_post_category() -> None:
    """Example 2: Classify post category using related comments (one-to-many)."""
    print("\n" + "=" * 60)
    print("Example 2: Post Category Classification (One-to-Many Relation)")
    print("=" * 60)

    # Posts frame (target for annotation)
    posts = hf.DFrame(
        {
            "post_id": [1, 2, 3],
            "title": ["Python Tips & Tricks", "Election 2026 News", "World Cup Final"],
            "category": [None, None, None],
        }
    )

    # Comments frame (context)
    comments = hf.DFrame(
        {
            "comment_id": [101, 102, 103, 104, 105],
            "post_id": [1, 1, 2, 2, 3],
            "text": [
                "Good guide for beginners",
                "Very helpful Python code",
                "Breaking news here",
                "Important updates",
                "Exciting final match",
            ],
        }
    )

    print("\nPosts (before annotation):")
    print(posts.read_fresh()[["post_id", "title", "category"]])
    print("\nRelated comments:")
    print(comments.read_fresh()[["comment_id", "post_id", "text"]])

    # Define relation: each post → many comments
    relation = FrameRelation(
        from_column="post_id",
        to_column="post_id",
        context_frame="comments",
        include_columns=["text"],
        relation_type="one_to_many",
        context_label="Related Comments",
        max_related=3,
    )

    writer = RelationalAgentWriter(
        target_frame=posts,
        relations=[relation],
        context_frames={"comments": comments},
    )

    # Monkey-patch for demo
    writer._make_llm_caller = lambda *args, **kwargs: mock_llm_call_category

    print("\nAnnotating posts using related comments context...")
    result = await writer.stream_normalize_relational(
        target_column="category",
        instruction="Categorize post as: technology, news, or entertainment",
        chunk_size=10,
        provider="anthropic",
    )

    print(f"\nResult: {result}")

    # Read results
    updated = posts.read_fresh()
    print("\nPosts (after annotation):")
    print(updated[["post_id", "title", "category"]])


async def example_nested_comments() -> None:
    """Example 3: Classify nested comment type using parent comment (self-referencing)."""
    print("\n" + "=" * 60)
    print("Example 3: Nested Comment Classification (Self-Referencing)")
    print("=" * 60)

    # Nested comments frame
    comments = hf.DFrame(
        {
            "comment_id": [1, 2, 3, 4, 5],
            "parent_id": [None, 1, 1, 2, 3],  # None = top-level, else parent comment_id
            "text": [
                "Machine learning is powerful",
                "I completely agree with this",
                "I have to disagree strongly",
                "Good counterpoint made",
                "Not relevant to the discussion",
            ],
            "type": [None, None, None, None, None],
        }
    )

    print("\nNested comments (before annotation):")
    df = comments.read_fresh()
    print(df[["comment_id", "parent_id", "text", "type"]])

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

    print("\nAnnotating comments using parent comment context...")
    print("(Note: This demo uses deterministic mock logic)")

    # Simple deterministic logic for nested comments
    async def mock_nested_llm(messages: list[dict[str, str]]) -> str:
        operations = []
        row_idx = 0
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")

        lines = user_msg.split("\n")
        for line in lines:
            if line.startswith("Row"):
                try:
                    row_idx = int(line.split()[1])
                except (ValueError, IndexError):
                    pass
            elif "parent_id: None" in line:
                comment_type = "original"
                confidence = 0.95
                operations.append(
                    {
                        "cell_id": f"comments::type_{row_idx}",
                        "value": comment_type,
                        "confidence": confidence,
                    }
                )
            elif "agree" in user_msg.lower():
                comment_type = "agreement"
                confidence = 0.88
                operations.append(
                    {
                        "cell_id": f"comments::type_{row_idx}",
                        "value": comment_type,
                        "confidence": confidence,
                    }
                )
            elif "disagree" in user_msg.lower():
                comment_type = "rebuttal"
                confidence = 0.90
                operations.append(
                    {
                        "cell_id": f"comments::type_{row_idx}",
                        "value": comment_type,
                        "confidence": confidence,
                    }
                )

        response = MockLLMResponse(operations=operations[:5])
        return json.dumps(response.__dict__)

    writer._make_llm_caller = lambda *args, **kwargs: mock_nested_llm

    result = await writer.stream_normalize_relational(
        target_column="type",
        instruction="Classify as: 'original' (top-level), 'agreement' (supports parent), or 'rebuttal' (opposes parent)",
        chunk_size=5,
        provider="anthropic",
    )

    print(f"\nResult: {result}")

    # Read results
    updated = comments.read_fresh()
    print("\nNested comments (after annotation):")
    print(updated[["comment_id", "parent_id", "text", "type"]])


async def main() -> None:
    """Run all examples."""
    print("\nRelationalAgentWriter Examples (Mock LLM - No API Key Required)")
    print("See docs/docs/agent/relational-writer.md for full documentation")

    # For demo purposes, we'll show structure without actual LLM calls
    # In production, these would use real LLM providers
    print("\n⚠️  Note: These examples use mock LLM responses for demonstration.")
    print("In production, replace with real LLM provider (Anthropic/OpenAI).\n")

    try:
        # Example 1: Many-to-one relation
        await example_comment_stance()
    except Exception as e:
        print(f"Example 1 note: {e}")

    try:
        # Example 2: One-to-many relation
        await example_post_category()
    except Exception as e:
        print(f"Example 2 note: {e}")

    try:
        # Example 3: Self-referencing
        await example_nested_comments()
    except Exception as e:
        print(f"Example 3 note: {e}")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("See docs/docs/agent/relational-writer.md for production examples")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

