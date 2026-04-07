# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Tests untuk SeriesSpec dan MultiFrameResult chart series output."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.result import MultiFrameResult, SeriesSpec
from agent.multi_agent import MultiFrameAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_spec(name: str = "revenue", data: list[dict] | None = None) -> SeriesSpec:
    return SeriesSpec(
        name=name,
        description=f"{name} trend",
        data=data if data is not None else [
            {"month": "Jan", "value": 100},
            {"month": "Feb", "value": 150},
        ],
        suggested_x="month",
        suggested_y="value",
        suggested_group_by=None,
        unit="IDR",
        source_frames=["sales"],
    )


def _make_result(*specs: SeriesSpec, analysis: str = "Test analysis") -> MultiFrameResult:
    return MultiFrameResult(analysis=analysis, series=list(specs))


@pytest.fixture
def mock_plotly(monkeypatch):
    """Mock plotly.express so tests don't need plotly installed."""
    fake_fig = MagicMock()
    fake_fig.write_image = MagicMock()

    px_mock = MagicMock()
    px_mock.line.return_value = fake_fig
    px_mock.bar.return_value = fake_fig
    px_mock.scatter.return_value = fake_fig
    px_mock.area.return_value = fake_fig
    px_mock.pie.return_value = fake_fig
    px_mock.histogram.return_value = fake_fig

    monkeypatch.setitem(sys.modules, "plotly", MagicMock())
    monkeypatch.setitem(sys.modules, "plotly.express", px_mock)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", MagicMock())
    return px_mock, fake_fig


# ---------------------------------------------------------------------------
# SeriesSpec.to_dataframe
# ---------------------------------------------------------------------------

def test_series_spec_to_dataframe():
    spec = _make_spec()
    df = spec.to_dataframe()
    assert len(df) == 2
    assert list(df.columns) == ["month", "value"]
    assert df["value"].tolist() == [100, 150]


def test_series_spec_to_dataframe_empty():
    spec = _make_spec(data=[])
    df = spec.to_dataframe()
    assert df.empty


# ---------------------------------------------------------------------------
# SeriesSpec.to_plotly_figure
# ---------------------------------------------------------------------------

def test_series_spec_to_plotly_figure_line(mock_plotly):
    px_mock, fake_fig = mock_plotly
    spec = _make_spec()
    fig = spec.to_plotly_figure("line")
    px_mock.line.assert_called_once()
    call_kwargs = px_mock.line.call_args.kwargs
    assert call_kwargs["x"] == "month"
    assert call_kwargs["y"] == "value"
    assert fig is fake_fig


def test_series_spec_to_plotly_figure_bar(mock_plotly):
    px_mock, fake_fig = mock_plotly
    spec = _make_spec()
    fig = spec.to_plotly_figure("bar")
    px_mock.bar.assert_called_once()
    assert fig is fake_fig


def test_series_spec_to_plotly_figure_unknown_type(mock_plotly):
    spec = _make_spec()
    with pytest.raises(ValueError, match="Unknown chart_type"):
        spec.to_plotly_figure("unknown")


def test_series_spec_to_plotly_figure_user_override(mock_plotly):
    px_mock, _ = mock_plotly
    spec = _make_spec()
    spec.to_plotly_figure("line", title="Custom Title")
    call_kwargs = px_mock.line.call_args.kwargs
    assert call_kwargs["title"] == "Custom Title"
    # suggested_x still used since not overridden
    assert call_kwargs["x"] == "month"


def test_series_spec_to_plotly_figure_no_plotly(monkeypatch):
    monkeypatch.setitem(sys.modules, "plotly.express", None)
    # Remove cached module so import fails
    with patch("builtins.__import__", side_effect=ImportError("No module named 'plotly'")):
        spec = _make_spec()
        with pytest.raises(ImportError, match="plotly"):
            spec.to_plotly_figure("line")


def test_series_spec_to_plotly_figure_empty_data(mock_plotly):
    spec = _make_spec(data=[])
    with pytest.raises(ValueError, match="has no data"):
        spec.to_plotly_figure("line")


# ---------------------------------------------------------------------------
# SeriesSpec.save_chart
# ---------------------------------------------------------------------------

def test_series_spec_save_chart(tmp_path, mock_plotly):
    _, fake_fig = mock_plotly
    spec = _make_spec()
    path = str(tmp_path / "chart.png")
    result = spec.save_chart(path, chart_type="bar")
    fake_fig.write_image.assert_called_once()
    assert result.endswith("chart.png")


def test_series_spec_save_chart_creates_dir(tmp_path, mock_plotly):
    _, fake_fig = mock_plotly
    spec = _make_spec()
    nested = str(tmp_path / "sub" / "dir" / "chart.png")
    spec.save_chart(nested)
    fake_fig.write_image.assert_called_once()


def test_series_spec_save_chart_no_kaleido(mock_plotly):
    _, fake_fig = mock_plotly
    fake_fig.write_image.side_effect = Exception("kaleido is required")
    spec = _make_spec()
    with pytest.raises(ImportError, match="kaleido"):
        spec.save_chart("/tmp/chart.png")


# ---------------------------------------------------------------------------
# MultiFrameResult.get_series
# ---------------------------------------------------------------------------

def test_result_get_series_found():
    spec = _make_spec("sales")
    result = _make_result(spec)
    found = result.get_series("sales")
    assert found is spec


def test_result_get_series_not_found():
    result = _make_result(_make_spec("sales"))
    assert result.get_series("nonexistent") is None


# ---------------------------------------------------------------------------
# MultiFrameResult.to_dataframe
# ---------------------------------------------------------------------------

def test_result_to_dataframe_found():
    spec = _make_spec("sales")
    result = _make_result(spec)
    df = result.to_dataframe("sales")
    assert len(df) == 2


def test_result_to_dataframe_not_found():
    result = _make_result(_make_spec("sales"))
    df = result.to_dataframe("nonexistent")
    assert df.empty


# ---------------------------------------------------------------------------
# MultiFrameResult.to_plotly_figure
# ---------------------------------------------------------------------------

def test_result_to_plotly_figure_found(mock_plotly):
    px_mock, fake_fig = mock_plotly
    spec = _make_spec("sales")
    result = _make_result(spec)
    fig = result.to_plotly_figure("sales", "bar")
    px_mock.bar.assert_called_once()
    assert fig is fake_fig


def test_result_to_plotly_figure_not_found():
    result = _make_result(_make_spec("sales"))
    with pytest.raises(KeyError, match="nonexistent"):
        result.to_plotly_figure("nonexistent")


def test_result_to_plotly_figure_not_found_shows_available():
    result = _make_result(_make_spec("sales"), _make_spec("inventory"))
    with pytest.raises(KeyError) as exc_info:
        result.to_plotly_figure("missing")
    assert "sales" in str(exc_info.value)


# ---------------------------------------------------------------------------
# MultiFrameResult.save_chart
# ---------------------------------------------------------------------------

def test_result_save_chart_found(tmp_path, mock_plotly):
    _, fake_fig = mock_plotly
    spec = _make_spec("sales")
    result = _make_result(spec)
    path = result.save_chart("sales", str(tmp_path / "sales.png"))
    fake_fig.write_image.assert_called_once()
    assert path.endswith("sales.png")


def test_result_save_chart_not_found():
    result = _make_result(_make_spec("sales"))
    with pytest.raises(KeyError, match="nonexistent"):
        result.save_chart("nonexistent", "chart.png")


# ---------------------------------------------------------------------------
# MultiFrameResult.save_all_charts
# ---------------------------------------------------------------------------

def test_result_save_all_charts(tmp_path, mock_plotly):
    _, fake_fig = mock_plotly
    result = _make_result(_make_spec("a"), _make_spec("b"))
    paths = result.save_all_charts(str(tmp_path))
    assert set(paths.keys()) == {"a", "b"}
    assert fake_fig.write_image.call_count == 2


def test_result_save_all_charts_skips_failed(tmp_path, mock_plotly):
    _, fake_fig = mock_plotly
    call_count = 0

    def write_image_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("simulated failure")

    fake_fig.write_image.side_effect = write_image_side_effect

    result = _make_result(_make_spec("a"), _make_spec("b"))
    paths = result.save_all_charts(str(tmp_path))
    # First series failed, second succeeded
    assert len(paths) == 1
    assert "b" in paths


# ---------------------------------------------------------------------------
# to_markdown with series
# ---------------------------------------------------------------------------

def test_result_to_markdown_with_series():
    spec = _make_spec("revenue_by_region")
    result = _make_result(spec)
    md = result.to_markdown()
    assert "Available Charts" in md
    assert "revenue_by_region" in md


def test_result_to_markdown_no_series():
    result = MultiFrameResult(analysis="some analysis", series=[])
    md = result.to_markdown()
    assert "Available Charts" not in md


def test_result_to_markdown_series_with_group():
    spec = SeriesSpec(
        name="grouped",
        description="grouped data",
        data=[{"x": 1, "y": 2, "g": "A"}],
        suggested_x="x",
        suggested_y="y",
        suggested_group_by="g",
    )
    result = _make_result(spec)
    md = result.to_markdown()
    assert "group" in md
    assert "`g`" in md


def test_result_to_markdown_series_multi_y():
    spec = SeriesSpec(
        name="multi_y",
        description="multi y",
        data=[{"x": 1, "a": 2, "b": 3}],
        suggested_x="x",
        suggested_y=["a", "b"],
    )
    result = _make_result(spec)
    md = result.to_markdown()
    assert "multi_y" in md


# ---------------------------------------------------------------------------
# to_dict with series
# ---------------------------------------------------------------------------

def test_result_to_dict_includes_series():
    spec = _make_spec("sales")
    result = _make_result(spec)
    d = result.to_dict()
    assert "series" in d
    assert len(d["series"]) == 1
    s = d["series"][0]
    assert s["name"] == "sales"
    assert s["row_count"] == 2
    assert "data" not in s  # data not included in to_dict


def test_result_to_dict_no_series():
    result = MultiFrameResult(analysis="test", series=[])
    d = result.to_dict()
    # series key absent or empty list when no series
    assert d.get("series", []) == []


# ---------------------------------------------------------------------------
# _plan_to_result parsing
# ---------------------------------------------------------------------------

def test_plan_to_result_parses_series():
    plan = {
        "analysis": "test analysis",
        "series": [
            {
                "name": "revenue",
                "description": "revenue trend",
                "data": [{"month": "Jan", "val": 100}, {"month": "Feb", "val": 200}],
                "suggested_x": "month",
                "suggested_y": "val",
                "suggested_group_by": None,
                "unit": "IDR",
                "source_frames": ["sales"],
            }
        ],
    }
    result = MultiFrameAgent._plan_to_result(plan)
    assert len(result.series) == 1
    assert result.series[0].name == "revenue"
    assert len(result.series[0].data) == 2
    assert result.series[0].suggested_x == "month"


def test_plan_to_result_skips_invalid_series():
    plan = {
        "series": [
            {"name": "bad_data", "data": "not_a_list"},          # data not list
            {"name": "empty_dicts", "data": ["str", 123]},        # items not dict
            {"name": "valid", "data": [{"x": 1, "y": 2}]},       # valid
        ]
    }
    result = MultiFrameAgent._plan_to_result(plan)
    assert len(result.series) == 1
    assert result.series[0].name == "valid"


def test_plan_to_result_empty_series():
    plan = {"analysis": "no charts needed"}
    result = MultiFrameAgent._plan_to_result(plan)
    assert result.series == []


def test_plan_to_result_series_list_y():
    plan = {
        "series": [{
            "name": "comparison",
            "description": "compare",
            "data": [{"month": "Jan", "a": 1, "b": 2}],
            "suggested_x": "month",
            "suggested_y": ["a", "b"],
        }]
    }
    result = MultiFrameAgent._plan_to_result(plan)
    assert isinstance(result.series[0].suggested_y, list)
    assert result.series[0].suggested_y == ["a", "b"]


# ---------------------------------------------------------------------------
# Integration: analyze flow populates series
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_analyze_query_mode_populates_series(monkeypatch):
    """Mock LLM to return series in final analysis JSON."""
    import pandas as pd
    from core.dataframe import DFrame

    df = DFrame.from_dict({"month": ["Jan", "Feb"], "revenue": [100, 200]})

    analysis_response = """{
        "action": "analyze",
        "reasoning": "test",
        "analysis": "Revenue is growing",
        "insights": [{"finding": "Revenue up", "frames": ["sales"], "confidence": 0.9}],
        "series": [
            {
                "name": "revenue_trend",
                "description": "Monthly revenue",
                "data": [{"month": "Jan", "revenue": 100}, {"month": "Feb", "revenue": 200}],
                "suggested_x": "month",
                "suggested_y": "revenue",
                "suggested_group_by": null,
                "unit": "IDR",
                "source_frames": ["sales"]
            }
        ],
        "operations": []
    }"""

    query_response = '{"queries": {"sales": "df.head(10)"}, "reasoning": "get data"}'

    call_counter = {"n": 0}

    async def fake_call_llm(messages):
        call_counter["n"] += 1
        if call_counter["n"] == 1:
            return query_response
        return analysis_response

    agent = MultiFrameAgent(frames={"sales": df}, provider="anthropic")
    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    result = await agent.analyze("Trend revenue", mode="query")
    assert len(result.series) == 1
    assert result.series[0].name == "revenue_trend"
    assert len(result.series[0].data) == 2


@pytest.mark.asyncio
async def test_analyze_sample_mode_populates_series(monkeypatch):
    """sample mode juga bisa populate series via _plan_to_result."""
    from core.dataframe import DFrame

    df = DFrame.from_dict({"x": [1, 2], "y": [3, 4]})

    sample_response = """{
        "action": "analyze",
        "reasoning": "test",
        "analysis": "Sample analysis",
        "insights": [],
        "series": [
            {
                "name": "xy_data",
                "description": "xy relationship",
                "data": [{"x": 1, "y": 3}],
                "suggested_x": "x",
                "suggested_y": "y"
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
    assert result.series[0].name == "xy_data"

