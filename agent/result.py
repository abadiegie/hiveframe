# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Result dataclasses for MultiFrameAgent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

@dataclass
class SeriesSpec:
    """Pure data chart series spec (no plotting/rendering dependencies)."""

    label: str
    x: list[Any]
    y: list[Any]
    x_label: str = ""
    y_label: str = ""
    series_type: str = "bar"

    @property
    def name(self) -> str:
        """Compatibility alias for older callers that read `name`."""
        return self.label

    def to_dict(self) -> dict[str, Any]:
        """Serialize into a JSON-friendly dictionary."""
        return {
            "label": self.label,
            "x": self.x,
            "y": self.y,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "series_type": self.series_type,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SeriesSpec":
        """Deserialize from new schema and best-effort legacy schema."""
        if "x" in payload and "y" in payload:
            return cls(
                label=str(payload.get("label", "series")),
                x=list(payload.get("x", [])),
                y=list(payload.get("y", [])),
                x_label=str(payload.get("x_label", "")),
                y_label=str(payload.get("y_label", "")),
                series_type=str(payload.get("series_type", "bar") or "bar"),
            )

        data = payload.get("data")
        if not isinstance(data, list):
            data = []
        rows = [item for item in data if isinstance(item, dict)]

        x_key = str(payload.get("suggested_x", ""))
        raw_y = payload.get("suggested_y", "")
        if isinstance(raw_y, list):
            y_key = str(raw_y[0]) if raw_y else ""
        else:
            y_key = str(raw_y)

        x_values = [row.get(x_key) for row in rows] if x_key else list(range(len(rows)))
        y_values = [row.get(y_key) for row in rows] if y_key else [None for _ in rows]
        return cls(
            label=str(payload.get("label") or payload.get("name") or "series"),
            x=x_values,
            y=y_values,
            x_label=x_key,
            y_label=y_key,
            series_type=str(payload.get("series_type") or payload.get("chart_type") or "bar"),
        )

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert x/y arrays to pandas DataFrame."""
        import pandas as pd

        if not self.x and not self.y:
            return pd.DataFrame()
        max_len = max(len(self.x), len(self.y))
        x_values = list(self.x) + [None] * max(0, max_len - len(self.x))
        y_values = list(self.y) + [None] * max(0, max_len - len(self.y))
        return pd.DataFrame({self.x_label or "x": x_values, self.y_label or "y": y_values})


@dataclass
class FrameInsight:
    """One insight produced by cross-frame analysis."""

    finding: str
    frames: list[str]
    confidence: float
    row_references: list[str] = field(default_factory=list)


@dataclass
class ReviewVerdict:
    """Verdict returned by the LLM after reviewing query results."""

    status: str
    reason: str = ""
    reflection: str = ""
    missing_parts: list[str] = field(default_factory=list)
    suggested_queries: dict[str, str] = field(default_factory=dict)
    accepted_labels: list[str] = field(default_factory=list)
    needs_columns: list[str] = field(default_factory=list)
    merge_ready: bool = False


@dataclass
class ColumnProfile:
    """Statistical profile untuk satu column."""

    column_name: str
    dtype: str
    null_count: int
    null_pct: float
    unique_count: int
    is_numeric: bool
    is_categorical: bool
    is_temporal: bool

    # Numeric statistics
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None

    # Categorical top values: [(value, count), ...]
    top_values: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-friendly dict."""
        return {
            "column_name": self.column_name,
            "dtype": self.dtype,
            "null_count": self.null_count,
            "null_pct": round(self.null_pct, 4),
            "unique_count": self.unique_count,
            "is_numeric": self.is_numeric,
            "is_categorical": self.is_categorical,
            "is_temporal": self.is_temporal,
            "min": self.min,
            "max": self.max,
            "mean": round(self.mean, 4) if self.mean is not None else None,
            "median": round(self.median, 4) if self.median is not None else None,
            "std": round(self.std, 4) if self.std is not None else None,
            "top_values": [{"value": str(v), "count": int(c)} for v, c in self.top_values],
        }


@dataclass
class FrameProfile:
    """Complete statistical profile untuk satu frame."""

    frame_label: str
    row_count: int
    col_count: int
    columns: dict[str, ColumnProfile]

    # Top aggregations auto-detected
    # Format: {"group_column": [{"value": "A", "count": 10, "pct": 0.15}, ...]}
    top_groupby_results: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-friendly dict."""
        return {
            "frame_label": self.frame_label,
            "row_count": self.row_count,
            "col_count": self.col_count,
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
            "top_groupby_results": self.top_groupby_results,
        }


@dataclass
class AggregationSnapshot:
    """Auto-generated aggregation snapshot dari frame."""

    frame_label: str
    aggregation_column: str
    aggregation_type: str  # "value_counts", "groupby", "describe"
    data: list[dict[str, Any]]  # [{"value": "A", "count": 10}, ...]
    title: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_label": self.frame_label,
            "aggregation_column": self.aggregation_column,
            "aggregation_type": self.aggregation_type,
            "data": self.data,
            "title": self.title,
        }


@dataclass
class MultiFrameResult:
    """Result returned by ``MultiFrameAgent.analyze()``."""

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
    fallback_reason: str = ""
    attempt_summaries: list[dict[str, Any]] = field(default_factory=list)

    # NEW: Comprehensive profiling data
    frame_profiles: dict[str, FrameProfile] = field(default_factory=dict)
    aggregation_snapshots: list[AggregationSnapshot] = field(default_factory=list)

    def get_series(self, name: str) -> SeriesSpec | None:
        """Get a SeriesSpec by label/name."""
        for s in self.series:
            if s.label == name or s.name == name:
                return s
        return None

    def to_dataframe(self, name: str) -> "pd.DataFrame":
        """Return one series as a pandas DataFrame.

        Returns an empty DataFrame when the requested series is not found.
        """
        spec = self.get_series(name)
        if spec is None:
            import pandas as pd
            return pd.DataFrame()
        return spec.to_dataframe()

    def to_markdown(self) -> str:
        """Render the result as a markdown report."""

        parts: list[str] = []

        # Frame profiles section
        if self.frame_profiles:
            parts.append("## 📊 Data Overview\n")
            for label, profile in self.frame_profiles.items():
                parts.append(f"### {label}\n")
                parts.append(f"- **Shape:** {profile.row_count:,} rows × {profile.col_count} columns")

                # Data quality indicators
                null_cols = [
                    (col_name, col_prof.null_pct)
                    for col_name, col_prof in profile.columns.items()
                    if col_prof.null_pct > 0.1
                ]
                if null_cols:
                    parts.append("- **Data Quality Issues:**")
                    for col_name, null_pct in sorted(null_cols, key=lambda x: -x[1])[:5]:
                        parts.append(f"  - `{col_name}`: {null_pct:.1%} null")

                # Key statistics
                numeric_cols = [
                    (col_name, col_prof)
                    for col_name, col_prof in profile.columns.items()
                    if col_prof.is_numeric
                ]
                if numeric_cols:
                    parts.append("- **Numeric Columns:**")
                    for col_name, col_prof in numeric_cols[:5]:
                        parts.append(f"  - `{col_name}`: μ={col_prof.mean:.2f}, σ={col_prof.std:.2f}, range=[{col_prof.min}, {col_prof.max}]")

                parts.append("")

        # Aggregation snapshots
        if self.aggregation_snapshots:
            parts.append("## 📈 Aggregation Snapshots\n")
            for snap in self.aggregation_snapshots:
                parts.append(f"### {snap.frame_label} - {snap.aggregation_column}\n")
                if isinstance(snap.data, list) and snap.data:
                    for item in snap.data[:15]:
                        value_str = str(item.get("value", "?"))
                        count = item.get("count", 0)
                        pct = item.get("pct", 0)
                        parts.append(f"- {value_str}: {count} ({pct:.1%})")
                parts.append("")

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
                parts.append(
                    f"- **`{spec.label}`** ({spec.series_type})\n"
                    f"  x: `{spec.x_label or 'x'}` | y: `{spec.y_label or 'y'}` | "
                    f"{len(spec.x)} points"
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

        if self.attempt_summaries:
            parts.append("\n## Attempt Summaries\n")
            for summary in self.attempt_summaries:
                attempt = summary.get("attempt", "?")
                source = summary.get("source", "generated")
                succeeded = ", ".join(summary.get("succeeded_labels", [])) or "-"
                failed = ", ".join(summary.get("failed_labels", [])) or "-"
                verdict = summary.get("verdict") or "-"
                line = (
                    f"- Attempt {attempt} [{source}] | succeeded: {succeeded} | "
                    f"failed: {failed} | verdict: {verdict}"
                )
                reason = summary.get("reason")
                if reason:
                    line += f" | reason: {reason}"
                rewrites = summary.get("rewrites")
                if rewrites:
                    line += f" | rewrites: {rewrites}"
                suggested = ", ".join(summary.get("suggested_query_labels", []))
                if suggested:
                    line += f" | suggested: {suggested}"
                parts.append(line)

        if self.total_llm_calls:
            converged_str = "yes" if self.converged else "no"
            parts.append(
                f"\n_LLM calls: {self.total_llm_calls} | "
                f"Converged: {converged_str} | "
                f"Final verdict: {self.final_verdict or '?'}_"
            )

        if self.fallback_reason:
            parts.append(f"\n_Fallback reason: {self.fallback_reason}_")

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
        """Serialize the result into a JSON-friendly dictionary."""

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
            "fallback_reason": self.fallback_reason,
            "attempt_summaries": self.attempt_summaries,
        }

        if self.series:
            base["series"] = [s.to_dict() for s in self.series]

        return base

