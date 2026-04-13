# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Simple deterministic chart generator — no LLM required.

User picks columns and a chart type; pandas does the aggregation and the
result is returned as a pure-data :class:`~agent.result.SeriesSpec`.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from .result import SeriesSpec

if TYPE_CHECKING:
    import pandas as pd

    from core.dataframe import DFrame

logger = logging.getLogger("hiveframe.agent.chart")

CHART_TYPES = frozenset({"bar", "line", "area", "scatter", "pie", "histogram", "heatmap"})
AGG_FUNCS = frozenset({"count", "sum", "mean", "median", "min", "max"})


class ChartGenerator:
    """Generate a :class:`~agent.result.SeriesSpec` from a frame without LLM.

    User selects columns and chart type; aggregation is performed with
    pandas and the result is returned as a structured :class:`SeriesSpec`.

    Args:
        source: A :class:`~core.dataframe.DFrame` or a plain
            ``pandas.DataFrame``.
        frame_label: Label stored in the generated series label suffix.

    Example::

        gen = ChartGenerator(my_dframe, frame_label="sales")
        series = gen.generate(
            chart_type="bar",
            x="Category",
            agg="count",
            top_n=20,
        )
        payload = series.to_dict()
    """

    def __init__(
        self,
        source: "DFrame | pd.DataFrame",
        frame_label: str = "frame",
    ) -> None:
        self._source = source
        self._frame_label = frame_label

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def available_columns(self) -> list[str]:
        """Return all column names available in the source frame."""
        return list(self._read().columns)

    def suggest_config(self, chart_type: str) -> dict[str, Any]:
        """Suggest sensible column assignments for *chart_type* based on dtypes.

        Returns a dict with keys ``chart_type``, ``available_columns``,
        ``numeric_columns``, ``category_columns``, ``suggested_x``,
        ``suggested_y``, and ``suggested_group_by``.
        """
        df = self._read()
        numeric_cols = list(df.select_dtypes(include="number").columns)
        category_cols = list(
            df.select_dtypes(include=["object", "string", "category"]).columns
        )
        date_cols = list(
            df.select_dtypes(include=["datetime64", "datetimetz"]).columns
        )

        ct = chart_type.lower()
        if ct not in CHART_TYPES:
            raise ValueError(
                f"Unknown chart_type '{chart_type}'. "
                f"Supported: {', '.join(sorted(CHART_TYPES))}"
            )

        suggestions: dict[str, Any] = {
            "chart_type": ct,
            "available_columns": list(df.columns),
            "numeric_columns": numeric_cols,
            "category_columns": category_cols,
        }

        if ct in ("bar", "pie"):
            suggestions["suggested_x"] = category_cols[0] if category_cols else None
            suggestions["suggested_y"] = numeric_cols[0] if numeric_cols else None
            suggestions["suggested_group_by"] = (
                category_cols[1] if len(category_cols) > 1 else None
            )
        elif ct in ("line", "area"):
            suggestions["suggested_x"] = (
                date_cols[0]
                if date_cols
                else (category_cols[0] if category_cols else None)
            )
            suggestions["suggested_y"] = numeric_cols[0] if numeric_cols else None
            suggestions["suggested_group_by"] = (
                category_cols[0] if category_cols else None
            )
        elif ct == "scatter":
            suggestions["suggested_x"] = numeric_cols[0] if numeric_cols else None
            suggestions["suggested_y"] = (
                numeric_cols[1] if len(numeric_cols) > 1 else None
            )
            suggestions["suggested_group_by"] = (
                category_cols[0] if category_cols else None
            )
        elif ct == "histogram":
            suggestions["suggested_x"] = numeric_cols[0] if numeric_cols else None
            suggestions["suggested_y"] = None
            suggestions["suggested_group_by"] = (
                category_cols[0] if category_cols else None
            )
        elif ct == "heatmap":
            suggestions["suggested_x"] = category_cols[0] if category_cols else None
            suggestions["suggested_y"] = numeric_cols[0] if numeric_cols else None
            suggestions["suggested_group_by"] = (
                category_cols[1] if len(category_cols) > 1 else None
            )

        return suggestions

    def generate(
        self,
        chart_type: str,
        x: str | None = None,
        y: str | list[str] | None = None,
        group_by: str | None = None,
        agg: str = "count",
        top_n: int | None = 20,
        title: str = "",
        sort_by: str | None = None,
        ascending: bool = False,
    ) -> SeriesSpec:
        """Aggregate the data and return a :class:`SeriesSpec`.

        Args:
            chart_type: ``"bar"`` | ``"line"`` | ``"area"`` | ``"scatter"``
                | ``"pie"`` | ``"histogram"`` | ``"heatmap"``
            x: Column for the x-axis (or category labels for pie/heatmap).
                Required for all chart types except histogram (where it is
                the numeric distribution column).
            y: Column(s) for the y-axis.  When *None* for bar/pie, the
                generator uses ``value_counts()`` (count aggregation).
            group_by: Column used for color grouping / heatmap y-axis.
            agg: Aggregation applied to *y*: ``"count"`` | ``"sum"``
                | ``"mean"`` | ``"median"`` | ``"min"`` | ``"max"``.
            top_n: Limit the number of result rows.  ``None`` returns all.
            title: Chart title. Auto-generated when empty.
            sort_by: Column to sort results by (defaults to *y*).
            ascending: Sort direction (default descending).

        Returns:
            A :class:`SeriesSpec` containing aggregated data and rendering
            hints.

        Raises:
            ValueError: On unknown chart_type, missing required columns,
                or unsupported agg function.
        """
        chart_type = chart_type.lower().strip()
        if chart_type not in CHART_TYPES:
            raise ValueError(
                f"Unsupported chart_type '{chart_type}'. "
                f"Supported: {', '.join(sorted(CHART_TYPES))}"
            )

        agg = agg.lower().strip()
        if agg not in AGG_FUNCS:
            raise ValueError(
                f"Unsupported agg '{agg}'. "
                f"Supported: {', '.join(sorted(AGG_FUNCS))}"
            )

        df = self._read()
        self._validate_columns(df, x=x, y=y, group_by=group_by)

        y_cols: list[str] = (
            [y] if isinstance(y, str) else list(y) if y else []
        )

        _builders = {
            "bar": self._bar,
            "line": self._line_area,
            "area": self._line_area,
            "scatter": self._scatter,
            "pie": self._pie,
            "histogram": self._histogram,
            "heatmap": self._heatmap,
        }

        result_df, sx, sy, sgb = _builders[chart_type](
            df=df,
            x=x,
            y=y_cols,
            group_by=group_by,
            agg=agg,
            top_n=top_n,
            sort_by=sort_by,
            ascending=ascending,
        )

        if not title:
            y_part = ", ".join(y_cols) if y_cols else "count"
            title = f"{chart_type.title()}: {x or y_part}"
            if group_by:
                title += f" by {group_by}"

        name = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")

        logger.debug(
            "ChartGenerator.generate: chart_type=%s x=%s y=%s group_by=%s rows=%d",
            chart_type,
            sx,
            sy,
            sgb,
            len(result_df),
        )

        y_column = sy[0] if isinstance(sy, list) and sy else (sy or "")
        if isinstance(sy, list) and len(sy) > 1:
            logger.debug(
                "ChartGenerator.generate: multi-y result detected, using first y column '%s'",
                y_column,
            )

        if sx and sx in result_df.columns:
            x_values = result_df[sx].tolist()
        else:
            x_values = result_df.index.tolist()

        if y_column and y_column in result_df.columns:
            y_values = result_df[y_column].tolist()
        else:
            y_values = [None] * len(result_df)

        series_label = f"{name}_{self._frame_label}" if self._frame_label else name
        return SeriesSpec(
            label=series_label,
            x=x_values,
            y=y_values,
            x_label=sx or "",
            y_label=str(y_column),
            series_type=chart_type,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read(self) -> "pd.DataFrame":
        import pandas as pd

        if isinstance(self._source, pd.DataFrame):
            return self._source.copy()
        return self._source.read_fresh()

    def _validate_columns(
        self,
        df: "pd.DataFrame",
        x: str | None,
        y: "str | list[str] | None",
        group_by: str | None,
    ) -> None:
        available = list(df.columns)
        if x is not None and x not in df.columns:
            raise ValueError(f"Column '{x}' not found. Available: {available}")
        y_cols = [y] if isinstance(y, str) else list(y) if y else []
        for col in y_cols:
            if col not in df.columns:
                raise ValueError(
                    f"Column '{col}' not found. Available: {available}"
                )
        if group_by is not None and group_by not in df.columns:
            raise ValueError(
                f"Column '{group_by}' not found. Available: {available}"
            )

    # -- chart-type builders ------------------------------------------

    def _bar(
        self, df, x, y, group_by, agg, top_n, sort_by, ascending
    ):
        if x is None:
            raise ValueError("chart_type='bar' requires x column")

        if not y:
            result = df[x].value_counts().reset_index()
            result.columns = [x, "count"]
            if top_n:
                result = result.head(top_n)
            return result, x, "count", None

        y_col: str | list[str] = y[0] if len(y) == 1 else y
        groupby_cols = [x] if not group_by else [x, group_by]

        if isinstance(y_col, list):
            result = df.groupby(groupby_cols)[y_col].agg(agg).reset_index()
        else:
            result = df.groupby(groupby_cols)[y_col].agg(agg).reset_index()

        _sort = sort_by or (y_col if isinstance(y_col, str) else y_col[0])
        if _sort in result.columns:
            result = result.sort_values(_sort, ascending=ascending)
        if top_n:
            result = result.head(top_n)
        return result, x, y_col, group_by

    def _line_area(
        self, df, x, y, group_by, agg, top_n, sort_by, ascending
    ):
        if x is None:
            raise ValueError("chart_type='line'/'area' requires x column")
        if not y:
            raise ValueError("chart_type='line'/'area' requires at least one y column")

        y_col: str | list[str] = y[0] if len(y) == 1 else y
        groupby_cols = [x] if not group_by else [x, group_by]

        if isinstance(y_col, list):
            result = df.groupby(groupby_cols)[y_col].agg(agg).reset_index()
        else:
            result = df.groupby(groupby_cols)[y_col].agg(agg).reset_index()

        _sort = sort_by or x
        if _sort in result.columns:
            result = result.sort_values(_sort, ascending=True)
        if top_n:
            result = result.head(top_n)
        return result, x, y_col, group_by

    def _scatter(
        self, df, x, y, group_by, agg, top_n, sort_by, ascending
    ):
        if x is None or not y:
            raise ValueError("chart_type='scatter' requires both x and y columns")

        y_col = y[0]
        select_cols = [x, y_col]
        if group_by:
            select_cols.append(group_by)

        result = df[select_cols].dropna()
        if top_n:
            result = result.head(top_n)
        return result, x, y_col, group_by

    def _pie(
        self, df, x, y, group_by, agg, top_n, sort_by, ascending
    ):
        if x is None:
            raise ValueError("chart_type='pie' requires x column (category labels)")

        if not y:
            result = df[x].value_counts().reset_index()
            result.columns = [x, "count"]
            if top_n:
                result = result.head(top_n)
            return result, x, "count", None

        y_col = y[0]
        result = df.groupby(x)[y_col].agg(agg).reset_index()
        result = result.sort_values(y_col, ascending=False)
        if top_n:
            result = result.head(top_n)
        return result, x, y_col, None

    def _histogram(
        self, df, x, y, group_by, agg, top_n, sort_by, ascending
    ):
        if x is None:
            raise ValueError("chart_type='histogram' requires x column")

        select_cols = [x]
        if group_by:
            select_cols.append(group_by)

        result = df[select_cols].dropna()
        if top_n:
            result = result.head(top_n)
        return result, x, "", group_by

    def _heatmap(
        self, df, x, y, group_by, agg, top_n, sort_by, ascending
    ):
        if x is None or group_by is None:
            raise ValueError(
                "chart_type='heatmap' requires x and group_by columns"
            )

        if not y:
            result = (
                df.groupby([x, group_by])
                .size()
                .reset_index(name="count")
            )
            return result, x, "count", group_by

        y_col = y[0]
        result = df.groupby([x, group_by])[y_col].agg(agg).reset_index()
        return result, x, y_col, group_by



