# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Tests for ChartGenerator."""

import pytest
import pandas as pd

from agent.chart_generator import ChartGenerator
from agent.result import SeriesSpec


@pytest.fixture
def df():
    return pd.DataFrame({
        "Category": ["A", "B", "A", "C", "B", "A"],
        "Region": ["X", "X", "Y", "Y", "X", "Y"],
        "Revenue": [100, 200, 150, 300, 250, 120],
        "Count": [1, 2, 1, 3, 2, 1],
    })


@pytest.fixture
def gen(df):
    return ChartGenerator(df, frame_label="test_frame")


# -- available_columns ------------------------------------------------

def test_available_columns(gen, df):
    assert gen.available_columns() == list(df.columns)


# -- suggest_config ---------------------------------------------------

def test_suggest_config_bar(gen):
    cfg = gen.suggest_config("bar")
    assert cfg["chart_type"] == "bar"
    assert cfg["suggested_x"] in cfg["category_columns"]
    assert cfg["suggested_y"] in cfg["numeric_columns"]


def test_suggest_config_scatter(gen):
    cfg = gen.suggest_config("scatter")
    assert cfg["suggested_x"] in cfg["numeric_columns"]
    assert cfg["suggested_y"] in cfg["numeric_columns"]


def test_suggest_config_unknown_raises(gen):
    with pytest.raises(ValueError, match="Unknown chart_type"):
        gen.suggest_config("unknown")


# -- generate: bar ----------------------------------------------------

def test_bar_value_counts(gen):
    series = gen.generate("bar", x="Category")
    assert isinstance(series, SeriesSpec)
    assert series.x_label == "Category"
    assert series.y_label == "count"
    assert series.series_type == "bar"
    assert len(series.x) <= 20
    assert len(series.x) == len(series.y)


def test_bar_with_y(gen):
    series = gen.generate("bar", x="Category", y="Revenue", agg="sum")
    assert series.y_label == "Revenue"
    cats = set(series.x)
    assert cats == {"A", "B", "C"}


def test_bar_with_group_by(gen):
    series = gen.generate("bar", x="Category", y="Revenue", group_by="Region", agg="sum")
    assert series.y_label == "Revenue"
    assert len(series.x) == len(series.y)


def test_bar_top_n(gen):
    series = gen.generate("bar", x="Category", top_n=2)
    assert len(series.x) == 2


def test_bar_top_n_zero_returns_empty(gen):
    series = gen.generate("bar", x="Category", top_n=0)
    assert len(series.x) == 0
    assert len(series.y) == 0


# -- generate: line / area --------------------------------------------

def test_line(gen):
    series = gen.generate("line", x="Category", y="Revenue", agg="mean")
    assert series.x_label == "Category"
    assert series.y_label == "Revenue"
    assert series.series_type == "line"


def test_line_respects_ascending(gen):
    series = gen.generate("line", x="Category", y="Revenue", agg="sum", ascending=False)
    assert series.x == ["C", "B", "A"]


def test_area(gen):
    series = gen.generate("area", x="Category", y="Revenue")
    assert series.x_label == "Category"
    assert series.series_type == "area"


def test_line_missing_y_raises(gen):
    with pytest.raises(ValueError):
        gen.generate("line", x="Category")


# -- generate: scatter ------------------------------------------------

def test_scatter(gen):
    series = gen.generate("scatter", x="Revenue", y="Count")
    assert series.x_label == "Revenue"
    assert series.y_label == "Count"
    assert series.series_type == "scatter"
    assert len(series.x) == 6


def test_scatter_missing_raises(gen):
    with pytest.raises(ValueError):
        gen.generate("scatter", x="Revenue")


# -- generate: pie ----------------------------------------------------

def test_pie_count(gen):
    series = gen.generate("pie", x="Category")
    assert series.x_label == "Category"
    assert series.y_label == "count"
    assert series.series_type == "pie"


def test_pie_with_y(gen):
    series = gen.generate("pie", x="Category", y="Revenue", agg="sum")
    assert series.y_label == "Revenue"
    assert series.series_type == "pie"


# -- generate: histogram ----------------------------------------------

def test_histogram(gen):
    series = gen.generate("histogram", x="Revenue")
    assert series.x_label == "Revenue"
    assert series.series_type == "histogram"
    assert len(series.x) == 6


# -- generate: heatmap ------------------------------------------------

def test_heatmap_count(gen):
    series = gen.generate("heatmap", x="Category", group_by="Region")
    assert series.x_label == "Category"
    assert series.y_label == "count"
    assert series.series_type == "heatmap"


def test_heatmap_with_y(gen):
    series = gen.generate("heatmap", x="Category", y="Revenue", group_by="Region", agg="sum")
    assert series.y_label == "Revenue"
    assert series.series_type == "heatmap"


def test_heatmap_missing_group_by_raises(gen):
    with pytest.raises(ValueError):
        gen.generate("heatmap", x="Category")


# -- error cases ------------------------------------------------------

def test_unknown_chart_type_raises(gen):
    with pytest.raises(ValueError, match="Unsupported chart_type"):
        gen.generate("funnel", x="Category")


def test_unknown_agg_raises(gen):
    with pytest.raises(ValueError, match="Unsupported agg"):
        gen.generate("bar", x="Category", y="Revenue", agg="variance")


def test_multi_y_raises(gen):
    with pytest.raises(ValueError, match="Multiple y columns"):
        gen.generate("bar", x="Category", y=["Revenue", "Count"], agg="sum")


def test_missing_column_raises(gen):
    with pytest.raises(ValueError, match="not found"):
        gen.generate("bar", x="NonExistent")


# -- name / description -----------------------------------------------

def test_auto_title(gen):
    series = gen.generate("bar", x="Category")
    assert "bar" in series.name.lower() or "category" in series.name.lower()


def test_custom_title(gen):
    series = gen.generate("bar", x="Category", title="My Chart")
    assert "my_chart" in series.label


# -- source_frames ----------------------------------------------------

def test_source_frames(gen):
    series = gen.generate("bar", x="Category")
    assert series.label.endswith("_test_frame")

