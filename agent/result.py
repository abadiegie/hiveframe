# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Result dataclasses untuk MultiFrameAgent."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    import plotly.graph_objects as go

_logger = logging.getLogger("hiveframe.result")


@dataclass
class SeriesSpec:
    """Structured data output dari LLM analysis, siap di-render sebagai chart."""

    name: str
    description: str
    data: list[dict[str, Any]]
    suggested_x: str = ""
    suggested_y: str | list[str] = ""
    suggested_group_by: str | None = None
    chart_type: str = "bar"  # Default chart type: bar, line, area, scatter, pie, histogram, heatmap
    unit: str = ""
    source_frames: list[str] = field(default_factory=list)

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert data ke pandas DataFrame. Return empty DataFrame kalau data kosong."""
        import pandas as pd

        if not self.data:
            return pd.DataFrame()
        return pd.DataFrame(self.data)

    def to_plotly_figure(
        self,
        chart_type: str | None = None,
        **kwargs: Any,
    ) -> "go.Figure":
        """Convert ke Plotly figure dengan chart_type yang user pilih.

        Args:
            chart_type: "line"|"bar"|"scatter"|"area"|"pie"|"histogram"|"heatmap"
                       Jika None, gunakan self.chart_type (dari LLM response).
            **kwargs: Diteruskan ke plotly.express function.

        Raises:
            ImportError: kalau plotly tidak terinstall.
            ValueError: kalau chart_type tidak dikenali atau data kosong.
        """
        # Use self.chart_type as fallback if chart_type not provided
        effective_chart_type = chart_type or self.chart_type or "bar"
        try:
            import importlib as _importlib
            px = _importlib.import_module("plotly.express")
        except ImportError:
            raise ImportError("plotly required for chart rendering: pip install plotly")

        df = self.to_dataframe()
        if df.empty:
            raise ValueError(f"SeriesSpec '{self.name}' has no data to plot")

        plot_kwargs: dict[str, Any] = {"title": self.description}
        if self.suggested_x:
            plot_kwargs["x"] = self.suggested_x
        if self.suggested_y:
            plot_kwargs["y"] = self.suggested_y
        if self.suggested_group_by:
            plot_kwargs["color"] = self.suggested_group_by

        plot_kwargs.update(kwargs)
        plot_kwargs["data_frame"] = df

        _CHART_BUILDERS = {
            "line": px.line,
            "bar": px.bar,
            "scatter": px.scatter,
            "area": px.area,
            "pie": px.pie,
            "histogram": px.histogram,
        }

        chart_lower = effective_chart_type.lower()

        if chart_lower == "heatmap":
            import plotly.graph_objects as _go

            if not self.suggested_x or not self.suggested_y or not self.suggested_group_by:
                raise ValueError(
                    "heatmap requires suggested_x, suggested_y, and suggested_group_by"
                )
            pivot = df.pivot(
                index=self.suggested_group_by,
                columns=self.suggested_x,
                values=self.suggested_y,
            )
            fig = _go.Figure(
                data=_go.Heatmap(
                    z=pivot.values,
                    x=list(pivot.columns),
                    y=list(pivot.index),
                    colorscale="Blues",
                )
            )
            fig.update_layout(title=self.description, **{k: v for k, v in kwargs.items() if k not in ("data_frame",)})
            return fig

        builder = _CHART_BUILDERS.get(chart_lower)
        if builder is None:
            supported = ", ".join(list(_CHART_BUILDERS.keys()) + ["heatmap"])
            raise ValueError(f"Unknown chart_type '{effective_chart_type}'. Supported: {supported}")

        return builder(**plot_kwargs)

    def save_chart(
        self,
        path: str,
        chart_type: str | None = None,
        width: int = 900,
        height: int = 500,
        scale: float = 2.0,
        **kwargs: Any,
    ) -> str:
        """Render chart dan save sebagai PNG.

        Args:
            chart_type: Chart type ke gunakan. Jika None, gunakan self.chart_type.

        Returns:
            Absolute path dari file yang disimpan.

        Raises:
            ImportError: kalau plotly atau kaleido tidak terinstall.
        """
        from pathlib import Path

        fig = self.to_plotly_figure(chart_type=chart_type, **kwargs)
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            fig.write_image(str(output_path), width=width, height=height, scale=scale)
        except Exception as exc:
            if "kaleido" in str(exc).lower():
                raise ImportError(
                    "kaleido required for PNG export: pip install kaleido"
                ) from exc
            raise

        return str(output_path.resolve())


@dataclass
class FrameInsight:
    """Satu insight dari analisis cross-frame."""

    finding: str
    frames: list[str]
    confidence: float
    row_references: list[str] = field(default_factory=list)


@dataclass
class ReviewVerdict:
    """Verdict dari LLM setelah review query results."""

    status: str
    reason: str = ""
    reflection: str = ""
    missing_parts: list[str] = field(default_factory=list)
    suggested_queries: dict[str, str] = field(default_factory=dict)
    accepted_labels: list[str] = field(default_factory=list)
    needs_columns: list[str] = field(default_factory=list)
    merge_ready: bool = False


@dataclass
class MultiFrameResult:
    """Result dari MultiFrameAgent.analyze()."""

    action: str = ""
    reasoning: str = ""
    analysis: str = ""
    insights: list[FrameInsight] = field(default_factory=list)
    operations: list[dict[str, Any]] = field(default_factory=list)
    queries_executed: dict[str, str] = field(default_factory=dict)
    query_errors: dict[str, str] = field(default_factory=dict)
    write_result: dict[str, Any] | None = None
    mode: str = "sample"
    review_history: list[ReviewVerdict] = field(default_factory=list)
    total_llm_calls: int = 0
    converged: bool = False
    final_verdict: str = ""
    series: list[SeriesSpec] = field(default_factory=list)

    def get_series(self, name: str) -> SeriesSpec | None:
        """Get SeriesSpec by name. Return None kalau tidak ditemukan."""
        for s in self.series:
            if s.name == name:
                return s
        return None

    def to_dataframe(self, name: str) -> "pd.DataFrame":
        """Get data dari series tertentu sebagai pandas DataFrame.

        Returns empty DataFrame kalau name tidak ditemukan.
        """
        spec = self.get_series(name)
        if spec is None:
            import pandas as pd
            return pd.DataFrame()
        return spec.to_dataframe()

    def to_plotly_figure(
        self,
        name: str,
        chart_type: str | None = None,
        **kwargs: Any,
    ) -> "go.Figure":
        """Get Plotly figure dari series tertentu.

        Args:
            name: Series name.
            chart_type: Chart type ke gunakan. Jika None, gunakan default dari series.

        Raises:
            KeyError: kalau name tidak ditemukan.
        """
        spec = self.get_series(name)
        if spec is None:
            raise KeyError(
                f"Series '{name}' not found. "
                f"Available: {[s.name for s in self.series]}"
            )
        return spec.to_plotly_figure(chart_type=chart_type, **kwargs)

    def save_chart(
        self,
        name: str,
        path: str,
        chart_type: str = "line",
        **kwargs: Any,
    ) -> str:
        """Save chart dari series tertentu sebagai PNG.

        Returns:
            Absolute path dari file yang disimpan.

        Raises:
            KeyError: kalau name tidak ditemukan.
        """
        spec = self.get_series(name)
        if spec is None:
            raise KeyError(
                f"Series '{name}' not found. "
                f"Available: {[s.name for s in self.series]}"
            )
        return spec.save_chart(path, chart_type=chart_type, **kwargs)

    def save_all_charts(
        self,
        output_dir: str = ".",
        chart_type: str | None = None,
        **kwargs: Any,
    ) -> dict[str, str]:
        """Save semua series sebagai PNG files.

        Args:
            chart_type: Chart type untuk semua series. Jika None, gunakan default dari masing-masing series.

        Returns:
            Dict name → absolute path. Series yang gagal di-skip.
        """
        import os

        paths: dict[str, str] = {}
        for spec in self.series:
            try:
                file_path = os.path.join(output_dir, f"{spec.name}.png")
                saved = spec.save_chart(file_path, chart_type=chart_type, **kwargs)
                paths[spec.name] = saved
            except Exception as exc:
                _logger.warning("Failed to save chart '%s': %s", spec.name, exc)
        return paths

    def to_markdown(self) -> str:
        """Format hasil sebagai markdown report."""

        parts: list[str] = []

        if self.analysis:
            parts.append(f"## Analysis\n\n{self.analysis}")

        if self.insights:
            parts.append("\n## Key Insights\n")
            for idx, insight in enumerate(self.insights, 1):
                frames_str = ", ".join(f"`{frame}`" for frame in insight.frames)
                parts.append(
                    f"{idx}. **{insight.finding}**\n"
                    f"   Sources: {frames_str}\n"
                    f"   Confidence: {insight.confidence:.0%}"
                )

        if self.series:
            parts.append("\n## Available Charts\n")
            for spec in self.series:
                y_str = (
                    ", ".join(spec.suggested_y)
                    if isinstance(spec.suggested_y, list)
                    else spec.suggested_y
                )
                group_part = (
                    f" | group: `{spec.suggested_group_by}`"
                    if spec.suggested_group_by
                    else ""
                )
                parts.append(
                    f"- **`{spec.name}`** — {spec.description}\n"
                    f"  x: `{spec.suggested_x}` | y: `{y_str}`"
                    f"{group_part} | {len(spec.data)} rows"
                )
            parts.append(
                "\n_Use `result.to_plotly_figure(name, chart_type)` to render._"
            )

        if self.review_history:
            parts.append("\n## Iteration History\n")
            icons = {
                "accepted": "v",
                "partial": "~",
                "error": "x",
                "plan": "->",
                "rejected": "x",
                "merge": "+",
            }
            for idx, verdict in enumerate(self.review_history, 1):
                icon = icons.get(verdict.status, "?")
                parts.append(f"{idx}. {icon} **{verdict.status}** - {verdict.reason}")

        if self.total_llm_calls:
            converged_str = "yes" if self.converged else "no"
            parts.append(
                f"\n_LLM calls: {self.total_llm_calls} | "
                f"Converged: {converged_str} | "
                f"Final verdict: {self.final_verdict or '?'}_"
            )

        if self.queries_executed:
            parts.append("\n## Queries Executed\n")
            for label, query in self.queries_executed.items():
                parts.append(f"**{label}:**\n```python\n{query}\n```")

        if self.query_errors:
            parts.append("\n## Query Errors\n")
            for label, error in self.query_errors.items():
                parts.append(f"- `{label}`: {error}")

        if self.write_result:
            written = self.write_result.get("written", 0)
            skipped = self.write_result.get("skipped", 0)
            parts.append(
                "\n## Write Result\n\n"
                f"Written: {written} cells | Skipped: {skipped} cells"
            )

        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize ke dict untuk JSON response."""

        base: dict[str, Any] = {
            "action": self.action,
            "reasoning": self.reasoning,
            "analysis": self.analysis,
            "insights": [
                {
                    "finding": insight.finding,
                    "frames": insight.frames,
                    "confidence": insight.confidence,
                    "row_references": insight.row_references,
                }
                for insight in self.insights
            ],
            "operations": self.operations,
            "queries_executed": self.queries_executed,
            "query_errors": self.query_errors,
            "write_result": self.write_result,
            "mode": self.mode,
            "review_history": [
                {
                    "status": verdict.status,
                    "reason": verdict.reason,
                    "reflection": verdict.reflection,
                    "missing_parts": verdict.missing_parts,
                    "suggested_queries": verdict.suggested_queries,
                    "accepted_labels": verdict.accepted_labels,
                    "needs_columns": verdict.needs_columns,
                    "merge_ready": verdict.merge_ready,
                }
                for verdict in self.review_history
            ],
            "total_llm_calls": self.total_llm_calls,
            "converged": self.converged,
            "final_verdict": self.final_verdict,
        }

        if self.series:
            base["series"] = [
                {
                    "name": s.name,
                    "description": s.description,
                    "suggested_x": s.suggested_x,
                    "suggested_y": s.suggested_y,
                    "suggested_group_by": s.suggested_group_by,
                    "unit": s.unit,
                    "source_frames": s.source_frames,
                    "row_count": len(s.data),
                    # data tidak di-include — gunakan to_dataframe() untuk akses data
                }
                for s in self.series
            ]

        return base

