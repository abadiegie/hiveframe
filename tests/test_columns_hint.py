# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from agent.multi_agent import MultiFrameAgent
from core.dataframe import DFrame


@pytest.fixture
def frame_data() -> DFrame:
    return DFrame(
        {
            "city": ["jakarta", "bandung", "surabaya", "medan", "denpasar", "makassar"],
            "score": [90, 80, 70, 88, 75, 65],
            "sentiment": ["pos", "neg", "pos", "neu", "pos", "neg"],
            "platform": ["x", "ig", "x", "tiktok", "ig", "x"],
            "title": ["t0", "t1", "t2", "t3", "t4", "t5"],
            "extra_col": ["e0", "e1", "e2", "e3", "e4", "e5"],
        }
    )


def test_columns_hint_default_state_none(frame_data: DFrame) -> None:
    agent = MultiFrameAgent({"data": frame_data})
    assert agent._columns_hint is None


def test_build_context_with_hint_basic(frame_data: DFrame) -> None:
    agent = MultiFrameAgent({"data": frame_data})
    ctx = agent._build_context_with_hint("data", frame_data, ["city", "score"], max_rows=5)
    assert "showing_columns: ['city', 'score']" in ctx
    assert "extra_col" not in ctx


def test_build_context_with_hint_includes_sample_rows(frame_data: DFrame) -> None:
    agent = MultiFrameAgent({"data": frame_data})
    ctx = agent._build_context_with_hint("data", frame_data, ["city", "score"], max_rows=5)
    assert "Sample (5 of 6 rows):" in ctx
    assert "jakarta" in ctx


def test_build_context_with_hint_missing_column_skipped(frame_data: DFrame, caplog: pytest.LogCaptureFixture) -> None:
    agent = MultiFrameAgent({"data": frame_data})
    with caplog.at_level("WARNING"):
        ctx = agent._build_context_with_hint("data", frame_data, ["city", "nonexistent"], max_rows=5)
    assert "showing_columns: ['city']" in ctx
    assert "columns not found" in caplog.text


def test_build_context_with_hint_all_missing_fallback(frame_data: DFrame, caplog: pytest.LogCaptureFixture) -> None:
    agent = MultiFrameAgent({"data": frame_data})
    with caplog.at_level("WARNING"):
        ctx = agent._build_context_with_hint("data", frame_data, ["nonexistent1", "nonexistent2"], max_rows=5)
    assert "falling back to all columns" in caplog.text
    assert "extra_col" in ctx


def test_build_context_with_hint_includes_dtype(frame_data: DFrame) -> None:
    agent = MultiFrameAgent({"data": frame_data})
    ctx = agent._build_context_with_hint("data", frame_data, ["city", "score"], max_rows=5)
    assert "Column types:" in ctx
    assert "city:" in ctx
    assert "score:" in ctx


def test_build_context_with_hint_includes_value_counts(frame_data: DFrame) -> None:
    agent = MultiFrameAgent({"data": frame_data})
    ctx = agent._build_context_with_hint("data", frame_data, ["city", "sentiment"], max_rows=5)
    assert "Top values per categorical column:" in ctx
    assert "sentiment" in ctx


def test_build_context_with_hint_numeric_stats(frame_data: DFrame) -> None:
    agent = MultiFrameAgent({"data": frame_data})
    ctx = agent._build_context_with_hint("data", frame_data, ["score"], max_rows=5)
    assert "Numeric statistics:" in ctx
    assert "mean" in ctx


def test_analyze_sample_with_hint_uses_hint_context(frame_data: DFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = MultiFrameAgent({"data": frame_data})
    calls: list[list[dict[str, str]]] = []

    async def fake_call(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return '{"action":"analyze","analysis":"ok","insights":[],"operations":[]}'

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    _ = asyncio.run(
        agent.analyze(
            "analyze",
            mode="sample",
            columns_hint={"data": ["city", "score"]},
        )
    )

    context_blob = "\n".join(msg["content"] for msg in calls[0] if msg["role"] == "system")
    assert "showing_columns: ['city', 'score']" in context_blob
    assert "extra_col" not in context_blob


def test_analyze_sample_without_hint_uses_describe_for_agent(frame_data: DFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = MultiFrameAgent({"data": frame_data})
    calls: list[list[dict[str, str]]] = []

    async def fake_call(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return '{"action":"analyze","analysis":"ok","insights":[],"operations":[]}'

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    _ = asyncio.run(agent.analyze("analyze", mode="sample"))

    context_blob = "\n".join(msg["content"] for msg in calls[0] if msg["role"] == "system")
    assert "## Sample (first" in context_blob
    assert "extra_col" in context_blob


def test_analyze_query_with_hint_schema_context(frame_data: DFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = MultiFrameAgent({"data": frame_data})
    calls: list[list[dict[str, str]]] = []
    responses = [
        '{"queries":{"data":"df.head(2)"},"reasoning":"q"}',
        '{"action":"analyze","analysis":"ok","insights":[],"operations":[]}',
    ]

    async def fake_call(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return responses.pop(0)

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    _ = asyncio.run(
        agent.analyze(
            "query",
            mode="query",
            columns_hint={"data": ["city", "score"]},
        )
    )

    first_blob = "\n".join(msg["content"] for msg in calls[0] if msg["role"] == "system")
    assert "Relevant columns (use these for queries):" in first_blob
    assert "city" in first_blob


def test_analyze_query_without_hint_schema_context(frame_data: DFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = MultiFrameAgent({"data": frame_data})
    calls: list[list[dict[str, str]]] = []
    responses = [
        '{"queries":{"data":"df.head(2)"},"reasoning":"q"}',
        '{"action":"analyze","analysis":"ok","insights":[],"operations":[]}',
    ]

    async def fake_call(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return responses.pop(0)

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    _ = asyncio.run(agent.analyze("query", mode="query"))

    first_blob = "\n".join(msg["content"] for msg in calls[0] if msg["role"] == "system")
    assert "Column dtypes:" in first_blob
    assert "Relevant columns (use these for queries):" not in first_blob


def test_columns_hint_partial_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    news = DFrame({"title": ["a", "b"], "city": ["jkt", "bdg"], "source": ["n1", "n2"]})
    social = DFrame({"content": ["x", "y"], "platform": ["ig", "x"], "region": ["jkt", "bdg"]})
    agent = MultiFrameAgent({"news": news, "social": social})
    calls: list[list[dict[str, str]]] = []

    async def fake_call(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return '{"action":"analyze","analysis":"ok","insights":[],"operations":[]}'

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    _ = asyncio.run(agent.analyze("x", mode="sample", columns_hint={"news": ["title", "city"]}))

    context_blob = "\n".join(msg["content"] for msg in calls[0] if msg["role"] == "system")
    assert "showing_columns: ['title', 'city']" in context_blob
    assert "## DataFrame: `social`" in context_blob
    assert "## Sample (first" in context_blob


def test_columns_hint_empty_list_fallback(frame_data: DFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = MultiFrameAgent({"data": frame_data})
    calls: list[list[dict[str, str]]] = []

    async def fake_call(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return '{"action":"analyze","analysis":"ok","insights":[],"operations":[]}'

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    _ = asyncio.run(agent.analyze("x", mode="sample", columns_hint={"data": []}))

    context_blob = "\n".join(msg["content"] for msg in calls[0] if msg["role"] == "system")
    assert "showing_columns:" in context_blob
    assert "extra_col" in context_blob


def test_hint_context_smaller_than_full_context(frame_data: DFrame) -> None:
    wide = DFrame(
        {
            **{f"num_{idx}": list(range(1000)) for idx in range(15)},
            **{f"cat_{idx}": [f"v{row % 5}" for row in range(1000)] for idx in range(5)},
        }
    )
    agent = MultiFrameAgent({"data": wide})
    hint_ctx = agent._build_context_with_hint("data", wide, ["num_0", "num_1", "cat_0"], max_rows=5)
    full_ctx = wide.describe_for_agent(max_rows=50, include_schema=True, include_stats=True)
    assert len(hint_ctx) < len(full_ctx)


def test_hint_max_rows_respected(frame_data: DFrame) -> None:
    agent = MultiFrameAgent({"data": frame_data})
    ctx = agent._build_context_with_hint("data", frame_data, ["title", "city"], max_rows=3)
    sample_block = ctx.split("\nColumn types:", 1)[0]
    assert "Sample (3 of 6 rows):" in ctx
    assert "t2" in sample_block
    assert "t3" not in sample_block


def test_columns_hint_resets_between_calls(frame_data: DFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = MultiFrameAgent({"data": frame_data})
    calls: list[list[dict[str, str]]] = []

    async def fake_call(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return '{"action":"analyze","analysis":"ok","insights":[],"operations":[]}'

    monkeypatch.setattr(agent, "_call_llm", fake_call)

    _ = asyncio.run(
        agent.analyze(
            "x",
            mode="sample",
            columns_hint={"data": ["city"]},
        )
    )
    _ = asyncio.run(agent.analyze("x", mode="sample"))

    first_blob = "\n".join(msg["content"] for msg in calls[0] if msg["role"] == "system")
    second_blob = "\n".join(msg["content"] for msg in calls[1] if msg["role"] == "system")

    assert "showing_columns: ['city']" in first_blob
    assert "## Sample (first" in second_blob
    assert agent._columns_hint is None


