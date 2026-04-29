# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from agent.multi_agent import MultiFrameAgent
from agent.prompt import build_query_generation_messages, build_review_messages
from agent.result import MultiFrameResult, ReviewVerdict
from core.dataframe import DFrame


def _agent() -> MultiFrameAgent:
    sales = DFrame({"product_id": ["a", "b", "c"], "qty_sold": [30, 20, 10], "price": [100, 150, 200]})
    inventory = DFrame({"product_id": ["a", "b", "c"], "stock_remaining": [3, 40, 2], "margin": [20, 10, 8]})
    return MultiFrameAgent(frames={"sales": sales, "inventory": inventory})


def test_review_verdict_defaults() -> None:
    verdict = ReviewVerdict(status="accepted")
    assert verdict.status == "accepted"
    assert verdict.merge_ready is False
    assert verdict.suggested_queries == {}


def test_review_verdict_merge_ready() -> None:
    verdict = ReviewVerdict(status="merge", merge_ready=True)
    assert verdict.merge_ready is True


def test_build_review_messages_basic() -> None:
    messages = build_review_messages("test", {}, {}, {})
    assert any(item["role"] == "user" for item in messages)
    assert any("test" in item["content"] for item in messages)


def test_build_review_messages_with_errors() -> None:
    messages = build_review_messages("inst", {}, {}, {"sales": "boom"})
    blob = "\n".join(item["content"] for item in messages)
    assert "Query errors" in blob
    assert "sales" in blob


def test_build_review_messages_with_previous_verdicts() -> None:
    messages = build_review_messages(
        "inst",
        {},
        {},
        {},
        previous_verdicts=[{"status": "partial", "reason": "need more"}],
    )
    blob = "\n".join(item["content"] for item in messages)
    assert "Previous attempts" in blob
    assert "partial" in blob


def test_build_review_messages_includes_schema_context() -> None:
    messages = build_review_messages(
        "inst",
        {"data": "df.head(2)"},
        {},
        {},
        frame_schemas={"data": "label: data\n  Category: object"},
    )
    blob = "\n".join(item["content"] for item in messages)
    assert "Available schemas" in blob
    assert "Frame schema: `data`" in blob


def test_build_review_messages_filters_mismatched_few_shot_examples() -> None:
    messages = build_review_messages(
        "inst",
        {"data": "df.head(2)"},
        {},
        {},
        frame_schemas={"data": "label: data\n  Category: object"},
    )
    blob = "\n".join(item["content"] for item in messages)
    assert "Produk mana yang penjualannya tinggi tapi stok menipis?" not in blob


def test_build_query_generation_messages_includes_matching_few_shot() -> None:
    messages = build_query_generation_messages(
        "cek produk",
        {
            "sales": "label: sales",
            "inventory": "label: inventory",
        },
    )
    blob = "\n".join(item["content"] for item in messages)
    assert "Produk mana yang penjualannya tinggi tapi stok menipis?" in blob


def test_build_query_generation_messages_skips_mismatched_few_shot() -> None:
    messages = build_query_generation_messages(
        "cek data",
        {"data": "label: data"},
    )
    blob = "\n".join(item["content"] for item in messages)
    assert "Produk mana yang penjualannya tinggi tapi stok menipis?" not in blob


def test_iterative_accepted_first_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent()
    responses = [
        '{"queries":{"sales":"df.head(2)"}}',
        '{"status":"accepted","reason":"enough"}',
        '{"action":"analyze","analysis":"done","insights":[],"operations":[]}',
    ]

    async def fake_call(_messages):
        return responses.pop(0)

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    result = asyncio.run(agent.analyze("x", mode="query", max_retries=1))

    assert result.converged is True
    assert result.total_llm_calls == 3
    assert len(result.review_history) == 1


def test_iterative_partial_additive(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent()
    seen_query_prompts: list[list[dict[str, str]]] = []
    responses = [
        '{"queries":{"sales":"df.head(2)"}}',
        '{"status":"partial","reason":"need inventory","accepted_labels":["sales"],'
        '"suggested_queries":{"inventory":"df.head(2)"},"reflection":"query inventory"}',
        '{"status":"accepted","reason":"ok"}',
        '{"action":"analyze","analysis":"joined","insights":[],"operations":[]}',
    ]

    async def fake_call(messages):
        text = responses.pop(0)
        if messages and "generate pandas queries" in messages[0]["content"]:
            seen_query_prompts.append(messages)
        return text

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    result = asyncio.run(agent.analyze("x", mode="query", max_retries=1))

    assert set(result.queries_executed.keys()) == {"sales", "inventory"}
    assert len(seen_query_prompts) == 1
    assert "sales" not in result.query_errors


def test_iterative_error_selective_replace(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent()
    responses = [
        '{"queries":{"sales":"df.head(2)","inventory":"df.not_existing_method()"}}',
        '{"status":"error","reason":"inventory failed","suggested_queries":{"inventory":"df.head(1)"}}',
        '{"status":"accepted","reason":"fixed"}',
        '{"action":"analyze","analysis":"ok","insights":[],"operations":[]}',
    ]

    async def fake_call(_messages):
        return responses.pop(0)

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    result = asyncio.run(agent.analyze("x", mode="query", max_retries=1))

    assert "sales" in result.queries_executed
    assert "inventory" in result.queries_executed


def test_iterative_uses_reviewer_suggested_queries_directly(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent()
    captured_messages: list[list[dict[str, str]]] = []
    responses = [
        '{"queries":{"sales":"df.head(2)"}}',
        '{"status":"partial","reason":"need inventory","accepted_labels":["sales"],'
        '"suggested_queries":{"inventory":"df.head(1)"},"reflection":"use inventory"}',
        '{"status":"accepted","reason":"ok"}',
        '{"action":"analyze","analysis":"ok","insights":[],"operations":[]}',
    ]

    async def fake_call(messages):
        captured_messages.append(messages)
        return responses.pop(0)

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    result = asyncio.run(agent.analyze("x", mode="query", max_retries=1))

    query_generation_calls = [
        messages for messages in captured_messages
        if messages and "generate pandas queries" in messages[0]["content"]
    ]
    assert len(query_generation_calls) == 1
    assert result.total_llm_calls == 4


def test_iterative_query_generation_uses_sample_rich_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent()
    captured_messages: list[list[dict[str, str]]] = []
    responses = [
        '{"queries":{"sales":"df.head(1)"}}',
        '{"status":"accepted","reason":"ok"}',
        '{"action":"analyze","analysis":"ok","insights":[],"operations":[]}',
    ]

    async def fake_call(messages):
        captured_messages.append(messages)
        return responses.pop(0)

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    _ = asyncio.run(agent.analyze("x", mode="query", max_retries=1))

    first_prompt_blob = "\n".join(msg["content"] for msg in captured_messages[0])
    assert "Column details (name | dtype | sample values):" in first_prompt_blob
    assert "Sample rows (first" in first_prompt_blob


def test_iterative_rewrites_frame_variable_before_strict_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent()
    responses = [
        '{"queries":{"sales":"sales[\'qty_sold\'].head(1)"}}',
        '{"status":"accepted","reason":"ok"}',
        '{"action":"analyze","analysis":"ok","insights":[],"operations":[]}',
    ]

    async def fake_call(_messages):
        return responses.pop(0)

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    result = asyncio.run(agent.analyze("x", mode="query", max_retries=1))

    assert result.analysis == "ok"
    assert result.queries_executed["sales"].startswith("df[")


def test_iterative_plan_injects_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent()
    captured_messages = []
    responses = [
        '{"queries":{"sales":"df.head(1)"}}',
        '{"status":"plan","reason":"need price+margin","needs_columns":["price","margin"],"reflection":"include extra cols"}',
        '{"queries":{"sales":"df[[\"price\"]].head(1)","inventory":"df[[\"margin\"]].head(1)"}}',
        '{"status":"accepted","reason":"ok"}',
        '{"action":"analyze","analysis":"ok","insights":[],"operations":[]}',
    ]

    async def fake_call(messages):
        captured_messages.append(messages)
        return responses.pop(0)

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    _ = asyncio.run(agent.analyze("x", mode="query", max_retries=1))

    second_plan_blob = "\n".join(msg["content"] for msg in captured_messages[2] if msg["role"] == "system")
    assert "Additional columns available" in second_plan_blob


def test_iterative_rejected_clears_results(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent()
    responses = [
        '{"queries":{"sales":"df.head(2)"}}',
        '{"status":"rejected","reason":"wrong approach","reflection":"use inventory"}',
        '{"queries":{"inventory":"df.head(2)"}}',
        '{"status":"accepted","reason":"ok"}',
        '{"action":"analyze","analysis":"ok","insights":[],"operations":[]}',
    ]

    async def fake_call(_messages):
        return responses.pop(0)

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    result = asyncio.run(agent.analyze("x", mode="query", max_retries=1))

    assert "inventory" in result.queries_executed


def test_iterative_merge_proceeds_to_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent()
    responses = [
        '{"queries":{"sales":"df.head(1)","inventory":"df.head(1)"}}',
        '{"status":"merge","reason":"sufficient","merge_ready":true}',
        '{"action":"analyze","analysis":"ok","insights":[],"operations":[]}',
    ]

    async def fake_call(_messages):
        return responses.pop(0)

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    result = asyncio.run(agent.analyze("x", mode="query", max_retries=1))
    assert result.converged is True
    assert result.final_verdict == "merge"


def test_iterative_max_retries_proceeds_anyway(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent()
    responses = [
        '{"queries":{"sales":"df.head(1)"}}',
        '{"status":"partial","reason":"need more","suggested_queries":{"inventory":"df.head(1)"}}',
        '{"status":"partial","reason":"still partial"}',
        '{"action":"analyze","analysis":"best effort","insights":[],"operations":[]}',
    ]

    async def fake_call(_messages):
        return responses.pop(0)

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    result = asyncio.run(agent.analyze("x", mode="query", max_retries=1))

    assert result.analysis
    assert result.converged is False
    assert result.total_llm_calls <= 5


def test_iterative_fallback_to_sample_if_no_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent()

    async def fake_sample(_instruction, _max_rows):
        return MultiFrameResult(action="analyze", analysis="sample fallback")

    responses = ['{"queries":{}}']

    async def fake_call(_messages):
        return responses.pop(0)

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    monkeypatch.setattr(agent, "_analyze_sample_mode", fake_sample)
    result = asyncio.run(agent.analyze("x", mode="query", max_retries=1))
    assert result.analysis == "sample fallback"
    assert result.fallback_reason == "iterative_no_queries"
    assert result.attempt_summaries[-1]["reason"] == "no_queries_generated"


def test_result_to_markdown_with_review_history() -> None:
    result = MultiFrameResult(
        analysis="ok",
        review_history=[ReviewVerdict(status="accepted", reason="done")],
        total_llm_calls=3,
        converged=True,
        final_verdict="accepted",
    )
    md = result.to_markdown()
    assert "Iteration History" in md
    assert "accepted" in md


def test_result_to_markdown_with_attempt_summaries() -> None:
    result = MultiFrameResult(
        analysis="ok",
        attempt_summaries=[
            {
                "attempt": 1,
                "source": "generated",
                "succeeded_labels": ["sales"],
                "failed_labels": ["inventory"],
                "verdict": "partial",
                "reason": "inventory missing",
                "suggested_query_labels": ["inventory"],
            }
        ],
        fallback_reason="iterative_no_queries",
    )
    md = result.to_markdown()
    assert "Attempt Summaries" in md
    assert "Fallback reason" in md
    assert "reason: inventory missing" in md
    assert "suggested: inventory" in md


def test_result_to_dict_includes_attempt_telemetry() -> None:
    result = MultiFrameResult(
        attempt_summaries=[{"attempt": 1, "source": "generated"}],
        fallback_reason="query_executor_no_results",
    )
    payload = result.to_dict()
    assert payload["attempt_summaries"] == [{"attempt": 1, "source": "generated"}]
    assert payload["fallback_reason"] == "query_executor_no_results"


def test_total_llm_calls_counted_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent()
    responses = [
        '{"queries":{"sales":"df.head(1)"}}',
        '{"status":"accepted","reason":"ok"}',
        '{"action":"analyze","analysis":"ok","insights":[],"operations":[]}',
    ]

    async def fake_call(_messages):
        return responses.pop(0)

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    result = asyncio.run(agent.analyze("x", mode="query", max_retries=1))
    assert result.total_llm_calls == 3


def test_total_llm_calls_with_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent()
    responses = [
        '{"queries":{"sales":"df.head(1)"}}',
        '{"status":"partial","reason":"need inventory","suggested_queries":{"inventory":"df.head(1)"}}',
        '{"status":"accepted","reason":"ok"}',
        '{"action":"analyze","analysis":"ok","insights":[],"operations":[]}',
    ]

    async def fake_call(_messages):
        return responses.pop(0)

    monkeypatch.setattr(agent, "_call_llm", fake_call)
    result = asyncio.run(agent.analyze("x", mode="query", max_retries=1))
    assert result.total_llm_calls <= 5

