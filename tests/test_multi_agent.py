# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging

import pandas as pd
import pytest

from agent.multi_agent import MultiFrameAgent, _rewrite_generated_code
from agent.result import FrameInsight, MultiFrameResult
from core.dataframe import DFrame


@pytest.fixture
def frame_sales() -> DFrame:
    return DFrame({"city": ["jakarta", "bandung", "jakarta"], "score": [90, 80, 70]})


@pytest.fixture
def frame_inventory() -> DFrame:
    return DFrame({"product_id": ["a", "b", "c"], "stock_qty": [5, 20, 2], "reorder_point": [10, 10, 5]})


def test_result_to_markdown_with_insights() -> None:
    result = MultiFrameResult(
        analysis="Top finding",
        insights=[
            FrameInsight(finding="A", frames=["sales"], confidence=0.9),
            FrameInsight(finding="B", frames=["inventory"], confidence=0.8),
        ],
    )
    md = result.to_markdown()
    assert "Top finding" in md
    assert "`sales`" in md
    assert "`inventory`" in md


def test_result_to_markdown_empty() -> None:
    result = MultiFrameResult()
    assert isinstance(result.to_markdown(), str)


def test_result_to_dict() -> None:
    result = MultiFrameResult(
        action="analyze",
        insights=[FrameInsight(finding="A", frames=["sales"], confidence=0.7)],
    )
    payload = result.to_dict()
    assert payload["action"] == "analyze"
    json.dumps(payload)


def test_safe_eval_valid_query(frame_sales: DFrame) -> None:
    agent = MultiFrameAgent(frames={"sales": frame_sales})
    df = frame_sales.read_fresh()
    out = agent._safe_eval("df.nlargest(2, 'score')", df)
    assert len(out.index) == 2


def test_safe_eval_forbidden_import(frame_sales: DFrame) -> None:
    agent = MultiFrameAgent(frames={"sales": frame_sales})
    with pytest.raises(ValueError, match="Forbidden pattern"):
        agent._safe_eval("import os; df.head()", frame_sales.read_fresh())


def test_safe_eval_forbidden_exec(frame_sales: DFrame) -> None:
    agent = MultiFrameAgent(frames={"sales": frame_sales})
    with pytest.raises(ValueError, match="Forbidden pattern"):
        agent._safe_eval("exec('x=1'); df", frame_sales.read_fresh())


def test_safe_eval_must_start_with_df(frame_sales: DFrame) -> None:
    agent = MultiFrameAgent(frames={"sales": frame_sales})
    with pytest.raises(ValueError, match="must start with 'df'"):
        agent._safe_eval("pd.DataFrame()", frame_sales.read_fresh())


def test_safe_eval_series_converted_to_dataframe(frame_sales: DFrame) -> None:
    agent = MultiFrameAgent(frames={"sales": frame_sales})
    out = agent._safe_eval("df['city'].value_counts()", frame_sales.read_fresh())
    assert isinstance(out, pd.DataFrame)
    assert len(out.columns) == 2


def test_safe_eval_rejects_frame_label_alias_without_rewrite(frame_sales: DFrame) -> None:
    agent = MultiFrameAgent(frames={"sales": frame_sales})
    with pytest.raises(NameError):
        agent._safe_eval(
            "result = sales['city'].value_counts()",
            frame_sales.read_fresh(),
        )


def test_rewrite_generated_code_rewrites_frame_variable_and_column_case(frame_sales: DFrame) -> None:
    rewritten, applied = _rewrite_generated_code(
        code="result = sales['CITY'].value_counts()",
        frame_label="sales",
        columns=list(frame_sales.read_fresh().columns),
        known_labels={"sales"},
    )

    assert "result = df['city'].value_counts()" == rewritten
    assert "frame_variable_to_df" in applied
    assert "column_case_match" in applied


def test_rewrite_generated_code_preserves_non_selector_string_literals(frame_sales: DFrame) -> None:
    rewritten, applied = _rewrite_generated_code(
        code="result = df[df['city'] == 'CITY']",
        frame_label="sales",
        columns=list(frame_sales.read_fresh().columns),
        known_labels={"sales"},
    )

    assert rewritten == "result = df[df['city'] == 'CITY']"
    assert "column_case_match" not in applied


def test_rewrite_generated_code_rewrites_column_lists(frame_sales: DFrame) -> None:
    rewritten, applied = _rewrite_generated_code(
        code="result = df[['CITY', 'score']]",
        frame_label="sales",
        columns=list(frame_sales.read_fresh().columns),
        known_labels={"sales"},
    )

    assert rewritten == "result = df[['city', 'score']]"
    assert "column_case_match" in applied


def test_schema_context_contains_shape(frame_sales: DFrame) -> None:
    agent = MultiFrameAgent(frames={"sales": frame_sales})
    ctx = agent._build_schema_context()["sales"]
    assert "rows" in ctx and "columns" in ctx


def test_schema_context_contains_dtypes(frame_sales: DFrame) -> None:
    agent = MultiFrameAgent(frames={"sales": frame_sales})
    ctx = agent._build_schema_context()["sales"]
    assert "city" in ctx
    assert "score" in ctx


def test_schema_context_no_sample_rows(frame_sales: DFrame) -> None:
    agent = MultiFrameAgent(frames={"sales": frame_sales})
    ctx = agent._build_schema_context()["sales"]
    assert "Sample values" not in ctx
    assert "jakarta" not in ctx.lower()


def test_analyze_sample_single_frame(frame_sales: DFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = MultiFrameAgent(frames={"sales": frame_sales})
    calls: list[list[dict[str, str]]] = []

    async def fake_call(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return '{"action":"analyze","analysis":"ok","insights":[],"operations":[]}'

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    result = asyncio.run(agent.analyze("analyze this", mode="sample"))

    assert isinstance(result, MultiFrameResult)
    assert result.analysis == "ok"
    assert len(calls) == 1


def test_analyze_sample_multi_frame(frame_sales: DFrame, frame_inventory: DFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = MultiFrameAgent(frames={"sales": frame_sales, "inventory": frame_inventory})
    calls: list[list[dict[str, str]]] = []

    async def fake_call(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return '{"action":"analyze","analysis":"ok","insights":[],"operations":[]}'

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    _ = asyncio.run(agent.analyze("cross frame", mode="sample"))

    assert len(calls) == 1
    context_blob = "\n".join(msg["content"] for msg in calls[0] if msg["role"] == "system")
    assert "DataFrame: `sales`" in context_blob
    assert "DataFrame: `inventory`" in context_blob


def test_analyze_sample_returns_multi_frame_result(frame_sales: DFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = MultiFrameAgent(frames={"sales": frame_sales})

    async def fake_call(messages: list[dict[str, str]]) -> str:
        return '{"action":"analyze","analysis":"done","insights":[],"operations":[]}'

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    result = asyncio.run(agent.analyze("x", mode="sample"))
    assert isinstance(result, MultiFrameResult)


def test_analyze_sample_reuses_single_snapshot_per_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    sales = DFrame({"product_id": ["a", "b"], "qty": [1, 2]})
    agent = MultiFrameAgent(frames={"sales": sales})
    responses = ['{"action":"analyze","analysis":"ok","insights":[],"operations":[]}']

    reads = {"count": 0}
    original_read_fresh = sales.read_fresh

    def counted_read_fresh():
        reads["count"] += 1
        return original_read_fresh()

    async def fake_call(_messages: list[dict[str, str]]) -> str:
        return responses.pop(0)

    sales.read_fresh = counted_read_fresh  # type: ignore[assignment]
    monkeypatch.setattr(agent, "_call_llm", fake_call)

    result = asyncio.run(agent.analyze("x", mode="sample"))

    assert result.analysis == "ok"
    assert reads["count"] == 1


def test_analyze_query_two_llm_calls(frame_sales: DFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = MultiFrameAgent(frames={"sales": frame_sales})
    calls: list[list[dict[str, str]]] = []
    responses = [
        '{"queries":{"sales":"df.head(2)"},"reasoning":"q"}',
        '{"action":"analyze","analysis":"ok","insights":[],"operations":[]}',
    ]

    async def fake_call(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return responses.pop(0)

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    _ = asyncio.run(agent.analyze("x", mode="query"))
    assert len(calls) == 2


def test_analyze_query_defaults_to_iterative_review_loop(frame_sales: DFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = MultiFrameAgent(frames={"sales": frame_sales})
    responses = [
        '{"queries":{"sales":"df.head(1)"}}',
        '{"status":"accepted","reason":"ok"}',
        '{"action":"analyze","analysis":"done","insights":[],"operations":[]}',
    ]

    async def fake_call(_messages: list[dict[str, str]]) -> str:
        return responses.pop(0)

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    result = asyncio.run(agent.analyze("x", mode="query"))

    assert result.analysis == "done"
    assert result.total_llm_calls == 3
    assert len(result.review_history) == 1
    assert result.final_verdict == "accepted"


def test_analyze_query_reuses_single_snapshot_per_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    sales = DFrame({"product_id": ["a", "b"], "qty": [1, 2]})
    inventory = DFrame({"product_id": ["a", "b"], "stock": [3, 4]})
    agent = MultiFrameAgent(frames={"sales": sales, "inventory": inventory})
    responses = [
        '{"queries":{"sales":"df.head(1)","inventory":"df.head(1)"}}',
        '{"status":"accepted","reason":"ok"}',
        '{"action":"analyze","analysis":"ok","insights":[],"operations":[]}',
    ]

    sales_reads = {"count": 0}
    inventory_reads = {"count": 0}
    sales_original = sales.read_fresh
    inventory_original = inventory.read_fresh

    def counted_sales_read_fresh():
        sales_reads["count"] += 1
        return sales_original()

    def counted_inventory_read_fresh():
        inventory_reads["count"] += 1
        return inventory_original()

    sales.read_fresh = counted_sales_read_fresh  # type: ignore[assignment]
    inventory.read_fresh = counted_inventory_read_fresh  # type: ignore[assignment]

    async def fake_call(_messages: list[dict[str, str]]) -> str:
        return responses.pop(0)

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    result = asyncio.run(agent.analyze("x", mode="query", max_retries=1))

    assert result.analysis == "ok"
    assert sales_reads["count"] == 1
    assert inventory_reads["count"] == 1


def test_snapshot_fresh_frames_is_lazy() -> None:
    sales = DFrame({"product_id": ["a"], "qty": [1]})
    inventory = DFrame({"product_id": ["a"], "stock": [3]})
    agent = MultiFrameAgent(frames={"sales": sales, "inventory": inventory})

    sales_reads = {"count": 0}
    inventory_reads = {"count": 0}
    sales_original = sales.read_fresh
    inventory_original = inventory.read_fresh

    def counted_sales_read_fresh():
        sales_reads["count"] += 1
        return sales_original()

    def counted_inventory_read_fresh():
        inventory_reads["count"] += 1
        return inventory_original()

    sales.read_fresh = counted_sales_read_fresh  # type: ignore[assignment]
    inventory.read_fresh = counted_inventory_read_fresh  # type: ignore[assignment]

    snapshots = agent._snapshot_fresh_frames()

    assert sales_reads["count"] == 0
    assert inventory_reads["count"] == 0

    sales_snapshot = snapshots.get("sales")

    assert sales_snapshot is not None
    assert sales_reads["count"] == 1
    assert inventory_reads["count"] == 0


def test_analyze_query_executes_generated_queries(frame_sales: DFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = MultiFrameAgent(frames={"sales": frame_sales})
    calls: list[list[dict[str, str]]] = []
    responses = [
        '{"queries":{"sales":"df.nlargest(1, \'score\')"},"reasoning":"q"}',
        '{"action":"analyze","analysis":"ok","insights":[],"operations":[]}',
    ]

    async def fake_call(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return responses.pop(0)

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    _ = asyncio.run(agent.analyze("top", mode="query"))

    second_call_blob = "\n".join(msg["content"] for msg in calls[1])
    assert "Query result: `sales`" in second_call_blob
    assert "Query: `df.nlargest(1, 'score')`" in second_call_blob


def test_analyze_query_simple_preserves_executed_query_text(frame_sales: DFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = MultiFrameAgent(frames={"sales": frame_sales})
    responses = [
        "```python\n# frame: sales\nresult = sales['CITY'].value_counts()\n```",
        '{"action":"analyze","analysis":"ok","insights":[],"operations":[]}',
    ]

    async def fake_call(_messages: list[dict[str, str]]) -> str:
        return responses.pop(0)

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    result = asyncio.run(agent._analyze_query_mode_simple("count cities", max_result_rows=200))

    assert result.queries_executed["sales"] == "result = df['city'].value_counts()"


def test_analyze_query_simple_fallback_sets_reason(frame_sales: DFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = MultiFrameAgent(frames={"sales": frame_sales})
    responses = [
        '{"queries":{"sales":"df.not_existing_method()"},"reasoning":"q"}',
        '{"action":"analyze","analysis":"fallback-sample","insights":[],"operations":[]}',
    ]

    async def fake_call(_messages: list[dict[str, str]]) -> str:
        return responses.pop(0)

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    result = asyncio.run(agent._analyze_query_mode_simple("x", max_result_rows=200))

    assert result.analysis == "fallback-sample"
    assert result.fallback_reason == "query_executor_no_results"
    assert result.attempt_summaries


def test_analyze_query_self_heals_frame_label_variable(
    frame_sales: DFrame,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    agent = MultiFrameAgent(frames={"sales": frame_sales})
    responses = [
        "```python\n# frame: sales\nresult = sales['city'].value_counts()\n```",
        '{"action":"analyze","analysis":"ok","insights":[],"operations":[]}',
    ]

    async def fake_call(messages: list[dict[str, str]]) -> str:
        return responses.pop(0)

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    caplog.set_level(logging.DEBUG, logger="hiveframe.agent.multi")
    result = asyncio.run(agent._analyze_query_mode_simple("count cities", max_result_rows=200))

    assert result.analysis == "ok"
    assert result.query_errors == {}
    assert "QueryExecutor REWRITE_PREVIEW" in caplog.text


def test_analyze_query_self_heals_column_case(frame_sales: DFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = MultiFrameAgent(frames={"sales": frame_sales})
    responses = [
        "```python\n# frame: sales\nresult = df['CITY'].value_counts()\n```",
        '{"action":"analyze","analysis":"ok","insights":[],"operations":[]}',
    ]

    async def fake_call(messages: list[dict[str, str]]) -> str:
        return responses.pop(0)

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    result = asyncio.run(agent._analyze_query_mode_simple("count cities", max_result_rows=200))

    assert result.analysis == "ok"
    assert result.query_errors == {}


def test_analyze_query_unknown_frame_logged_as_error(frame_sales: DFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = MultiFrameAgent(frames={"sales": frame_sales})
    responses = [
        '{"queries":{"missing":"df.head(2)"},"reasoning":"q"}',
        '{"action":"analyze","analysis":"fallback-sample","insights":[],"operations":[]}',
    ]

    async def fake_call(messages: list[dict[str, str]]) -> str:
        return responses.pop(0)

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    result = asyncio.run(agent.analyze("x", mode="query"))
    assert "missing" in result.query_errors
    # When frame is unknown, falls back to sample mode
    assert result.analysis == "fallback-sample"


def test_analyze_query_bad_pandas_query(frame_sales: DFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = MultiFrameAgent(frames={"sales": frame_sales})
    responses = [
        '{"queries":{"sales":"df.not_existing_method()"},"reasoning":"q"}',
        '{"action":"analyze","analysis":"fallback-sample","insights":[],"operations":[]}',
    ]

    async def fake_call(messages: list[dict[str, str]]) -> str:
        return responses.pop(0)

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    result = asyncio.run(agent.analyze("x", mode="query"))
    assert "sales" in result.query_errors
    # When query fails, falls back to sample mode
    assert result.analysis == "fallback-sample"


def test_analyze_query_fallback_to_sample_if_no_queries(frame_sales: DFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = MultiFrameAgent(frames={"sales": frame_sales})
    calls: list[list[dict[str, str]]] = []
    responses = [
        '{"queries":{},"reasoning":"empty"}',
        '{"action":"analyze","analysis":"sample-fallback","insights":[],"operations":[]}',
    ]

    async def fake_call(messages: list[dict[str, str]]) -> str:
        calls.append(messages)
        return responses.pop(0)

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    result = asyncio.run(agent.analyze("x", mode="query"))
    assert result.analysis == "sample-fallback"
    assert len(calls) == 2


def test_analyze_with_output_frame(frame_sales: DFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    output = DFrame()
    agent = MultiFrameAgent(frames={"sales": frame_sales})

    async def fake_call(messages: list[dict[str, str]]) -> str:
        return (
            '{"action":"batch_enrich","analysis":"ok","insights":[],'
            '"operations":[{"cell_id":"%s::result_0","value":"A","confidence":0.9}]}'
            % output._frame_id
        )

    async def fake_write(operations, output_frame):
        assert output_frame is output
        return {"written": len(operations), "skipped": 0}

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    monkeypatch.setattr(agent, "_write_to_frame", fake_write)

    result = asyncio.run(agent.analyze("x", mode="sample", output_frame=output))
    assert result.write_result is not None
    assert result.write_result["written"] == 1


def test_analyze_no_output_frame(frame_sales: DFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = MultiFrameAgent(frames={"sales": frame_sales})

    async def fake_call(messages: list[dict[str, str]]) -> str:
        return '{"action":"analyze","analysis":"ok","insights":[],"operations":[]}'

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    result = asyncio.run(agent.analyze("x", mode="sample"))
    assert result.write_result is None


def test_unknown_provider_raises(frame_sales: DFrame) -> None:
    agent = MultiFrameAgent(frames={"sales": frame_sales}, provider="unknown")
    with pytest.raises(ValueError, match="Unknown provider"):
        asyncio.run(agent._call_llm([]))


def test_call_llm_logs_debug_request_and_response(
    frame_sales: DFrame,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    agent = MultiFrameAgent(frames={"sales": frame_sales}, provider="openai", model="gpt-4o")

    async def fake_openai(messages: list[dict[str, str]]) -> str:
        return '{"action":"analyze","analysis":"ok"}'

    monkeypatch.setattr(agent, "_call_openai", fake_openai)
    caplog.set_level(logging.DEBUG, logger="hiveframe.agent.multi")

    raw = asyncio.run(agent._call_llm([
        {"role": "system", "content": "You are a tester."},
        {"role": "user", "content": "Explain sales."},
    ]))

    assert raw == '{"action":"analyze","analysis":"ok"}'
    assert "multi_agent LLM_REQUEST" in caplog.text
    assert "Explain sales." in caplog.text
    assert "multi_agent LLM_RESPONSE" in caplog.text
    assert '"analysis":"ok"' in caplog.text


def test_call_llm_timeout_raises(frame_sales: DFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = MultiFrameAgent(
        frames={"sales": frame_sales},
        provider="openai",
        model="gpt-4o",
        llm_timeout_seconds=0.01,
    )

    async def fake_openai(_messages: list[dict[str, str]]) -> str:
        await asyncio.sleep(0.05)
        return '{"action":"analyze","analysis":"ok"}'

    monkeypatch.setattr(agent, "_call_openai", fake_openai)
    with pytest.raises(TimeoutError, match="timed out"):
        asyncio.run(agent._call_llm([{"role": "user", "content": "x"}]))


def test_empty_frames_raises() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        MultiFrameAgent(frames={})

