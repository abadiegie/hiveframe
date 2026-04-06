# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Result dataclasses untuk MultiFrameAgent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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

        return {
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

