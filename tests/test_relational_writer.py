# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import re

import pytest

from agent.relational_writer import FrameRelation, RelationalAgentWriter
from core.dataframe import DFrame


def _posts_frame() -> DFrame:
    return DFrame(
        {
            "post_id": ["P001", "P002", "P003"],
            "text": ["Inflation up 5%", "Rupiah strengthened", "Fuel subsidy updated"],
            "author": ["a1", "a2", "a3"],
        }
    )


def _comments_frame() -> DFrame:
    return DFrame(
        {
            "comment_id": ["C001", "C002", "C003", "C004", "C005"],
            "post_id": ["P001", "P001", "P002", "P001", "P003"],
            "text": ["Agree", "Not true", "Neutral", "Makes sense", "Unsure"],
            "stance": [None, None, None, None, None],
            "author": ["u1", "u2", "u3", "u4", "u5"],
        }
    )


def _writer_many_to_one() -> RelationalAgentWriter:
    posts = _posts_frame()
    comments = _comments_frame()
    return RelationalAgentWriter(
        target_frame=comments,
        context_frames={"posts": posts},
        relations=[
            FrameRelation(
                from_column="post_id",
                to_column="post_id",
                context_frame="posts",
                include_columns=["text", "author"],
                relation_type="many_to_one",
                context_label="parent_post",
            )
        ],
    )


def _build_mock_llm(target_column: str, value_prefix: str = "val", confidence: float = 0.9):
    async def _call(messages):
        system = next(msg["content"] for msg in messages if msg["role"] == "system")
        user = next(msg["content"] for msg in messages if msg["role"] == "user")
        frame_id = re.search(r"frame_id = ([^\n]+)", system).group(1)
        indices = [int(idx) for idx in re.findall(r"Row (\d+) \(", user)]
        operations = [
            {
                "cell_id": f"{frame_id}::{target_column}_{idx}",
                "value": f"{value_prefix}_{idx}",
                "confidence": confidence,
            }
            for idx in indices
        ]
        return json.dumps({"action": "batch_enrich", "reasoning": "mock", "operations": operations})

    return _call


# FrameRelation

def test_frame_relation_defaults() -> None:
    rel = FrameRelation(from_column="a", to_column="b", context_frame="posts")
    assert rel.relation_type == "many_to_one"
    assert rel.max_related == 10
    assert rel.context_label == "posts"


def test_frame_relation_context_label_fallback() -> None:
    rel = FrameRelation(from_column="a", to_column="b", context_frame="ctx", context_label="")
    assert rel.context_label == "ctx"


def test_frame_relation_invalid_type() -> None:
    with pytest.raises(ValueError, match="relation_type"):
        FrameRelation(from_column="a", to_column="b", context_frame="ctx", relation_type="invalid")


# _validate_relations

def test_validate_relations_missing_context_frame() -> None:
    with pytest.raises(ValueError, match="not found in context_frames"):
        RelationalAgentWriter(
            target_frame=_comments_frame(),
            context_frames={},
            relations=[FrameRelation(from_column="post_id", to_column="post_id", context_frame="posts")],
        )


def test_validate_relations_self_ref_always_valid() -> None:
    _ = RelationalAgentWriter(
        target_frame=_comments_frame(),
        context_frames={},
        relations=[FrameRelation(from_column="comment_id", to_column="comment_id", context_frame="self")],
    )


def test_validate_relations_valid() -> None:
    _ = _writer_many_to_one()


# _build_cache

def test_build_cache_many_to_one() -> None:
    writer = _writer_many_to_one()
    writer._build_cache()
    assert "posts:post_id" in writer._cache
    assert writer._cache["posts:post_id"]["P001"][0]["text"] == "Inflation up 5%"


def test_build_cache_one_to_many() -> None:
    posts = _posts_frame()
    comments = _comments_frame()
    writer = RelationalAgentWriter(
        target_frame=posts,
        context_frames={"comments": comments},
        relations=[
            FrameRelation(
                from_column="post_id",
                to_column="post_id",
                context_frame="comments",
                relation_type="one_to_many",
            )
        ],
    )
    writer._build_cache()
    assert len(writer._cache["comments:post_id"]["P001"]) == 3


def test_build_cache_self_ref() -> None:
    target = DFrame({"comment_id": ["C1", "C2"], "parent_id": [None, "C1"], "text": ["a", "b"]})
    writer = RelationalAgentWriter(
        target_frame=target,
        context_frames={},
        relations=[FrameRelation(from_column="parent_id", to_column="comment_id", context_frame="self")],
    )
    writer._build_cache()
    assert "self:comment_id" in writer._cache
    assert writer._cache["self:comment_id"]["C1"][0]["text"] == "a"


def test_build_cache_missing_to_column(caplog: pytest.LogCaptureFixture) -> None:
    writer = RelationalAgentWriter(
        target_frame=_comments_frame(),
        context_frames={"posts": _posts_frame()},
        relations=[FrameRelation(from_column="post_id", to_column="missing_col", context_frame="posts")],
    )
    with caplog.at_level("WARNING"):
        writer._build_cache()
    assert writer._cache["posts:missing_col"] == {}
    assert "to_column 'missing_col' not found" in caplog.text


def test_build_cache_called_once() -> None:
    posts = _posts_frame()
    comments = _comments_frame()
    writer = RelationalAgentWriter(
        target_frame=comments,
        context_frames={"posts": posts},
        relations=[FrameRelation(from_column="post_id", to_column="post_id", context_frame="posts")],
    )
    called = {"count": 0}

    original = posts.read_fresh

    def counted_read_fresh():
        called["count"] += 1
        return original()

    posts.read_fresh = counted_read_fresh  # type: ignore[assignment]

    writer._build_cache()
    writer._build_cache()
    assert called["count"] == 1


# _lookup

def test_lookup_many_to_one_found() -> None:
    writer = _writer_many_to_one()
    writer._build_cache()
    rel = writer._relations[0]
    found = writer._lookup(rel, "P001")
    assert len(found) == 1


def test_lookup_many_to_one_not_found() -> None:
    writer = _writer_many_to_one()
    writer._build_cache()
    rel = writer._relations[0]
    assert writer._lookup(rel, "P999") == []


def test_lookup_one_to_many_found() -> None:
    posts = _posts_frame()
    comments = _comments_frame()
    rel = FrameRelation(
        from_column="post_id",
        to_column="post_id",
        context_frame="comments",
        relation_type="one_to_many",
        max_related=3,
    )
    writer = RelationalAgentWriter(target_frame=posts, context_frames={"comments": comments}, relations=[rel])
    writer._build_cache()
    found = writer._lookup(rel, "P001")
    assert len(found) == 3


def test_lookup_one_to_many_max_related() -> None:
    posts = DFrame({"post_id": ["P1"]})
    comments = DFrame({"post_id": ["P1"] * 100, "text": [f"t{i}" for i in range(100)]})
    rel = FrameRelation(
        from_column="post_id",
        to_column="post_id",
        context_frame="comments",
        relation_type="one_to_many",
        max_related=10,
    )
    writer = RelationalAgentWriter(target_frame=posts, context_frames={"comments": comments}, relations=[rel])
    writer._build_cache()
    assert len(writer._lookup(rel, "P1")) == 10


def test_lookup_include_columns_filter() -> None:
    writer = _writer_many_to_one()
    writer._build_cache()
    rel = writer._relations[0]
    rel.include_columns = ["text"]
    found = writer._lookup(rel, "P001")
    assert set(found[0].keys()) == {"text"}


def test_lookup_empty_include_columns() -> None:
    writer = _writer_many_to_one()
    writer._build_cache()
    rel = writer._relations[0]
    rel.include_columns = []
    found = writer._lookup(rel, "P001")
    assert "text" in found[0] and "author" in found[0]


# _build_enriched_context

def test_enriched_context_many_to_one_found() -> None:
    writer = _writer_many_to_one()
    writer._build_cache()
    row = writer._target.read_fresh().iloc[0].to_dict()
    ctx = writer._build_enriched_context([(0, row)])
    assert "[parent_post]" in ctx
    assert "Inflation up 5%" in ctx


def test_enriched_context_many_to_one_not_found() -> None:
    writer = _writer_many_to_one()
    writer._build_cache()
    row = {"post_id": "P999", "text": "x"}
    ctx = writer._build_enriched_context([(0, row)])
    assert "(not found)" in ctx


def test_enriched_context_one_to_many() -> None:
    posts = _posts_frame()
    comments = _comments_frame()
    writer = RelationalAgentWriter(
        target_frame=posts,
        context_frames={"comments": comments},
        relations=[
            FrameRelation(
                from_column="post_id",
                to_column="post_id",
                context_frame="comments",
                include_columns=["text"],
                relation_type="one_to_many",
                context_label="related_comments",
                max_related=5,
            )
        ],
    )
    writer._build_cache()
    row = writer._target.read_fresh().iloc[0].to_dict()
    ctx = writer._build_enriched_context([(0, row)])
    assert "[related_comments]" in ctx
    assert "3 found" in ctx


def test_enriched_context_multiple_relations() -> None:
    posts = _posts_frame()
    comments = _comments_frame()
    writer = RelationalAgentWriter(
        target_frame=comments,
        context_frames={"posts": posts, "siblings": comments},
        relations=[
            FrameRelation("post_id", "post_id", "posts", include_columns=["text"], context_label="parent_post"),
            FrameRelation(
                "post_id",
                "post_id",
                "siblings",
                include_columns=["text"],
                relation_type="one_to_many",
                context_label="related_comments",
            ),
        ],
    )
    writer._build_cache()
    row = writer._target.read_fresh().iloc[0].to_dict()
    ctx = writer._build_enriched_context([(0, row)])
    assert "[parent_post]" in ctx
    assert "[related_comments]" in ctx


def test_enriched_context_self_ref() -> None:
    thread = DFrame(
        {
            "comment_id": ["C1", "C2"],
            "parent_id": [None, "C1"],
            "text": ["Root", "Reply"],
            "stance": [None, None],
        }
    )
    writer = RelationalAgentWriter(
        target_frame=thread,
        relations=[
            FrameRelation(
                from_column="parent_id",
                to_column="comment_id",
                context_frame="self",
                include_columns=["text"],
                relation_type="many_to_one",
                context_label="parent_comment",
            )
        ],
    )
    writer._build_cache()
    row = writer._target.read_fresh().iloc[1].to_dict()
    ctx = writer._build_enriched_context([(1, row)])
    assert "[parent_comment]" in ctx
    assert "Root" in ctx


# stream_normalize_relational

def test_stream_normalize_writes_to_target(monkeypatch: pytest.MonkeyPatch) -> None:
    writer = _writer_many_to_one()
    monkeypatch.setattr(
        writer,
        "_make_llm_caller",
        lambda *args, **kwargs: _build_mock_llm(target_column="stance", value_prefix="stance", confidence=0.9),
    )

    result = asyncio.run(
        writer.stream_normalize_relational(
            target_column="stance",
            instruction="classify stance",
            chunk_size=2,
            provider="anthropic",
        )
    )

    writer._target._invalidate_snapshot_cache()
    fresh = writer._target.read_fresh()
    assert result["written"] > 0
    assert fresh["stance"].notna().all()


def test_stream_normalize_does_not_write_to_context(monkeypatch: pytest.MonkeyPatch) -> None:
    writer = _writer_many_to_one()
    before = writer._context_frames["posts"].read_fresh().copy()
    monkeypatch.setattr(
        writer,
        "_make_llm_caller",
        lambda *args, **kwargs: _build_mock_llm(target_column="stance", value_prefix="stance", confidence=0.9),
    )

    _ = asyncio.run(
        writer.stream_normalize_relational(
            target_column="stance",
            instruction="classify",
            chunk_size=2,
            provider="anthropic",
        )
    )

    after = writer._context_frames["posts"].read_fresh()
    assert before.equals(after)


def test_stream_normalize_chunk_size_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    writer = _writer_many_to_one()
    calls = {"count": 0}

    async def fake_llm(messages):
        calls["count"] += 1
        return await _build_mock_llm("stance", "s", 0.9)(messages)

    monkeypatch.setattr(writer, "_make_llm_caller", lambda *args, **kwargs: fake_llm)

    _ = asyncio.run(
        writer.stream_normalize_relational(
            target_column="stance",
            instruction="classify",
            chunk_size=2,
            provider="anthropic",
        )
    )

    assert calls["count"] == 3


def test_stream_normalize_low_confidence_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    writer = _writer_many_to_one()
    monkeypatch.setattr(
        writer,
        "_make_llm_caller",
        lambda *args, **kwargs: _build_mock_llm(target_column="stance", value_prefix="s", confidence=0.3),
    )

    result = asyncio.run(
        writer.stream_normalize_relational(
            target_column="stance",
            instruction="classify",
            chunk_size=5,
            provider="anthropic",
        )
    )

    assert result["written"] == 0
    assert result["skipped"] >= result["total"]


def test_stream_normalize_llm_error_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    writer = _writer_many_to_one()
    calls = {"n": 0}

    async def fake_llm(messages):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return await _build_mock_llm("stance", "s", 0.9)(messages)

    monkeypatch.setattr(writer, "_make_llm_caller", lambda *args, **kwargs: fake_llm)

    result = asyncio.run(
        writer.stream_normalize_relational(
            target_column="stance",
            instruction="classify",
            chunk_size=2,
            provider="anthropic",
        )
    )

    assert result["written"] > 0
    assert result["skipped"] > 0


def test_stream_normalize_missing_target_column_created(monkeypatch: pytest.MonkeyPatch) -> None:
    comments = DFrame(
        {
            "comment_id": ["C1", "C2"],
            "post_id": ["P001", "P002"],
            "text": ["a", "b"],
        }
    )
    writer = RelationalAgentWriter(
        target_frame=comments,
        context_frames={"posts": _posts_frame()},
        relations=[FrameRelation(from_column="post_id", to_column="post_id", context_frame="posts")],
    )
    monkeypatch.setattr(
        writer,
        "_make_llm_caller",
        lambda *args, **kwargs: _build_mock_llm(target_column="stance", value_prefix="s", confidence=0.9),
    )

    _ = asyncio.run(
        writer.stream_normalize_relational(
            target_column="stance",
            instruction="classify",
            chunk_size=2,
            provider="anthropic",
        )
    )

    writer._target._invalidate_snapshot_cache()
    assert "stance" in writer._target.read_fresh().columns


def test_stream_normalize_returns_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    writer = _writer_many_to_one()
    monkeypatch.setattr(
        writer,
        "_make_llm_caller",
        lambda *args, **kwargs: _build_mock_llm(target_column="stance", value_prefix="s", confidence=0.9),
    )

    result = asyncio.run(
        writer.stream_normalize_relational(
            target_column="stance",
            instruction="classify",
            chunk_size=2,
            provider="anthropic",
        )
    )

    assert set(result.keys()) == {"written", "skipped", "total"}
    assert result["written"] + result["skipped"] >= result["total"]


def test_stream_normalize_on_progress_called(monkeypatch: pytest.MonkeyPatch) -> None:
    writer = _writer_many_to_one()
    progress: list[tuple[int, int]] = []

    def on_progress(done: int, total: int) -> None:
        progress.append((done, total))

    monkeypatch.setattr(
        writer,
        "_make_llm_caller",
        lambda *args, **kwargs: _build_mock_llm(target_column="stance", value_prefix="s", confidence=0.9),
    )

    _ = asyncio.run(
        writer.stream_normalize_relational(
            target_column="stance",
            instruction="classify",
            chunk_size=2,
            provider="anthropic",
            on_progress=on_progress,
        )
    )

    assert len(progress) == 3
    assert progress[-1][0] == progress[-1][1]


def test_stream_normalize_self_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    thread = DFrame(
        {
            "comment_id": ["C1", "C2", "C3"],
            "parent_id": [None, "C1", "C1"],
            "text": ["Root", "Agree", "Disagree"],
            "stance": [None, None, None],
        }
    )
    writer = RelationalAgentWriter(
        target_frame=thread,
        context_frames={},
        relations=[
            FrameRelation(
                from_column="parent_id",
                to_column="comment_id",
                context_frame="self",
                include_columns=["text"],
                context_label="parent_comment",
            )
        ],
    )
    monkeypatch.setattr(
        writer,
        "_make_llm_caller",
        lambda *args, **kwargs: _build_mock_llm(target_column="stance", value_prefix="s", confidence=0.9),
    )

    result = asyncio.run(
        writer.stream_normalize_relational(
            target_column="stance",
            instruction="classify replies",
            chunk_size=2,
            provider="anthropic",
        )
    )

    assert result["written"] > 0


# invalid provider

def test_unknown_provider_raises() -> None:
    writer = _writer_many_to_one()
    with pytest.raises(ValueError, match="Unknown provider"):
        _ = writer._make_llm_caller("unknown", None, None, None)


# integration

def test_full_flow_comment_stance(monkeypatch: pytest.MonkeyPatch) -> None:
    posts = DFrame(
        {
            "post_id": ["P001", "P002", "P003"],
            "text": ["Inflation up 5%", "Rupiah strengthened", "Policy unchanged"],
        }
    )
    comments = DFrame(
        {
            "comment_id": [f"C{i:03d}" for i in range(1, 10)],
            "post_id": ["P001", "P001", "P001", "P002", "P002", "P002", "P003", "P003", "P003"],
            "text": [
                "Agree", "Not true", "Neutral", "Agree", "Not true", "Neutral", "Agree", "Not true", "Neutral"
            ],
            "stance": [None] * 9,
        }
    )

    writer = RelationalAgentWriter(
        target_frame=comments,
        context_frames={"posts": posts},
        relations=[
            FrameRelation(
                from_column="post_id",
                to_column="post_id",
                context_frame="posts",
                include_columns=["text"],
                relation_type="many_to_one",
                context_label="parent_post",
            )
        ],
    )

    before_posts = posts.read_fresh().copy()

    async def fake_llm(messages):
        system = next(msg["content"] for msg in messages if msg["role"] == "system")
        user = next(msg["content"] for msg in messages if msg["role"] == "user")
        frame_id = re.search(r"frame_id = ([^\n]+)", system).group(1)
        rows = re.findall(r"Row (\d+) \([\s\S]*?\n\s+text: ([^\n]+)", user)
        operations = []
        for idx, text in rows:
            idx_int = int(idx)
            text_lower = text.lower()
            if "not true" in text_lower or "disagree" in text_lower:
                value = "against"
            elif "neutral" in text_lower:
                value = "neutral"
            else:
                value = "pro"
            operations.append(
                {
                    "cell_id": f"{frame_id}::stance_{idx_int}",
                    "value": value,
                    "confidence": 0.92,
                }
            )
        return json.dumps({"action": "batch_enrich", "operations": operations})

    monkeypatch.setattr(writer, "_make_llm_caller", lambda *args, **kwargs: fake_llm)

    result = asyncio.run(
        writer.stream_normalize_relational(
            target_column="stance",
            instruction="Evaluate stance toward parent post: pro/against/neutral",
            chunk_size=3,
            provider="anthropic",
        )
    )

    comments._invalidate_snapshot_cache()
    updated = comments.read_fresh()
    assert result["written"] == 9
    assert updated["stance"].notna().all()
    assert posts.read_fresh().equals(before_posts)


