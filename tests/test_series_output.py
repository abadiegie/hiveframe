# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Tests for pure-data SeriesSpec and MultiFrameResult behavior."""

from __future__ import annotations

import pytest

from agent.multi_agent import MultiFrameAgent
from agent.result import MultiFrameResult, SeriesSpec


def _make_spec(label: str = "revenue") -> SeriesSpec:
    return SeriesSpec(
        label=label,
        x=["Jan", "Feb"],
        y=[100, 150],
        x_label="month",
        y_label="value",
        series_type="bar",
    )


def _make_result(*specs: SeriesSpec, analysis: str = "Test analysis") -> MultiFrameResult:
    return MultiFrameResult(analysis=analysis, series=list(specs))


def test_series_spec_to_dict_roundtrip() -> None:
    spec = _make_spec("sales")
    parsed = SeriesSpec.from_dict(spec.to_dict())
    assert parsed.label == "sales"
    assert parsed.x == ["Jan", "Feb"]
    assert parsed.y == [100, 150]
    assert parsed.series_type == "bar"


def test_series_spec_from_dict_legacy_payload() -> None:
    payload = {
        "name": "legacy_series",
        "suggested_x": "month",
        "suggested_y": "value",
        "chart_type": "line",
        "data": [
            {"month": "Jan", "value": 10},
            {"month": "Feb", "value": 20},
        ],
    }
    spec = SeriesSpec.from_dict(payload)
    assert spec.label == "legacy_series"
    assert spec.x == ["Jan", "Feb"]
    assert spec.y == [10, 20]
    assert spec.series_type == "line"


def test_series_spec_to_dataframe() -> None:
    df = _make_spec().to_dataframe()
    assert list(df.columns) == ["month", "value"]
    assert len(df) == 2


def test_series_spec_to_dataframe_empty() -> None:
    spec = SeriesSpec(label="empty", x=[], y=[])
    assert spec.to_dataframe().empty


def test_result_get_series_found() -> None:
    spec = _make_spec("sales")
    result = _make_result(spec)
    assert result.get_series("sales") is spec


def test_result_get_series_not_found() -> None:
    result = _make_result(_make_spec("sales"))
    assert result.get_series("nonexistent") is None


def test_result_to_dataframe_found() -> None:
    result = _make_result(_make_spec("sales"))
    assert len(result.to_dataframe("sales")) == 2


def test_result_to_dataframe_not_found() -> None:
    result = _make_result(_make_spec("sales"))
    assert result.to_dataframe("nonexistent").empty


def test_result_to_markdown_with_series() -> None:
    result = _make_result(_make_spec("revenue_by_region"))
    md = result.to_markdown()
    assert "revenue_by_region" in md
    assert "points" in md


def test_result_to_dict_includes_series() -> None:
    result = _make_result(_make_spec("sales"))
    payload = result.to_dict()
    assert "series" in payload
    assert payload["series"][0]["label"] == "sales"
    assert payload["series"][0]["x_label"] == "month"


def test_plan_to_result_parses_new_series_schema() -> None:
    plan = {
        "analysis": "ok",
        "series": [
            {
                "label": "rev",
                "x": ["Jan", "Feb"],
                "y": [1, 2],
                "x_label": "month",
                "y_label": "value",
                "series_type": "line",
            }
        ],
    }
    result = MultiFrameAgent._plan_to_result(plan)
    assert len(result.series) == 1
    assert result.series[0].label == "rev"
    assert result.series[0].series_type == "line"


def test_plan_to_result_parses_legacy_series_schema() -> None:
    plan = {
        "series": [
            {
                "name": "legacy",
                "suggested_x": "month",
                "suggested_y": "value",
                "chart_type": "bar",
                "data": [{"month": "Jan", "value": 99}],
            }
        ]
    }
    result = MultiFrameAgent._plan_to_result(plan)
    assert len(result.series) == 1
    assert result.series[0].label == "legacy"
    assert result.series[0].x == ["Jan"]
    assert result.series[0].y == [99]


@pytest.mark.asyncio
async def test_analyze_query_mode_populates_series(monkeypatch):
    from core.dataframe import DFrame

    df = DFrame.from_dict({"month": ["Jan", "Feb"], "revenue": [100, 200]})

    query_response = '{"queries": {"sales": "df.head(10)"}, "reasoning": "get data"}'
    analysis_response = """{
        "action": "analyze",
        "reasoning": "test",
        "analysis": "Revenue is growing",
        "insights": [{"finding": "Revenue up", "frames": ["sales"], "confidence": 0.9}],
        "series": [
            {
                "label": "revenue_trend",
                "x": ["Jan", "Feb"],
                "y": [100, 200],
                "x_label": "month",
                "y_label": "revenue",
                "series_type": "line"
            }
        ],
        "operations": []
    }"""

    call_counter = {"n": 0}

    async def fake_call_llm(messages):
        call_counter["n"] += 1
        return query_response if call_counter["n"] == 1 else analysis_response

    agent = MultiFrameAgent(frames={"sales": df}, provider="anthropic")
    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    result = await agent.analyze("Trend revenue", mode="query")
    assert len(result.series) == 1
    assert result.series[0].label == "revenue_trend"
    assert result.series[0].x == ["Jan", "Feb"]
    assert result.series[0].y == [100, 200]


@pytest.mark.asyncio
async def test_analyze_sample_mode_populates_series(monkeypatch):
    from core.dataframe import DFrame

    df = DFrame.from_dict({"x": [1, 2], "y": [3, 4]})

    sample_response = """{
        "action": "analyze",
        "reasoning": "test",
        "analysis": "Sample analysis",
        "insights": [],
        "series": [
            {
                "label": "xy_data",
                "x": [1, 2],
                "y": [3, 4],
                "x_label": "x",
                "y_label": "y",
                "series_type": "scatter"
            }
        ],
        "operations": []
    }"""

    async def fake_call_llm(messages):
        return sample_response

    agent = MultiFrameAgent(frames={"df": df}, provider="anthropic")
    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    result = await agent.analyze("analyze xy", mode="sample")
    assert len(result.series) == 1
    assert result.series[0].label == "xy_data"
