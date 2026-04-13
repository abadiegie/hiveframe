# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Result dataclasses untuk MultiFrameAgent."""

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
        """Get a SeriesSpec by label/name."""
        for s in self.series:
            if s.label == name or s.name == name:
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
            base["series"] = [s.to_dict() for s in self.series]

        return base

