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
        }

