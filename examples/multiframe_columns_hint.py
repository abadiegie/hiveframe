# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Runnable MultiFrameAgent columns_hint example without an API key.

Run:
    python examples/multiframe_columns_hint.py
"""

from __future__ import annotations

import asyncio
import json

from agent.multi_agent import MultiFrameAgent
from core.dataframe import DFrame


class DemoMultiFrameAgent(MultiFrameAgent):
    """Override LLM calls so this sample runs locally without a provider."""

    def __init__(self, frames: dict[str, DFrame]) -> None:
        super().__init__(frames=frames, provider="openai")
        self._step = 0

    async def _call_llm(self, messages: list[dict[str, str]]) -> str:
        self._step += 1
        if self._step == 1:
            # sample mode analysis response
            if any("DataFrame:" in msg.get("content", "") for msg in messages):
                return json.dumps(
                    {
                        "action": "analyze",
                        "reasoning": "Use visible columns only",
                        "analysis": "Positive sentiment is dominant in Jakarta.",
                        "insights": [
                            {
                                "finding": "Jakarta trends positive",
                                "frames": ["news"],
                                "confidence": 0.9,
                            }
                        ],
                        "operations": [],
                    }
                )

        # query mode simple flow: first call returns queries, second returns analysis
        if self._step == 2:
            return json.dumps({"queries": {"news": "df.groupby('city')['sentiment'].value_counts().reset_index()"}})
        return json.dumps(
            {
                "action": "analyze",
                "reasoning": "Aggregate sentiment per city",
                "analysis": "Sentiment distribution by city has been summarized.",
                "insights": [
                    {
                        "finding": "Cities with negative sentiment still exist but are smaller",
                        "frames": ["news"],
                        "confidence": 0.86,
                    }
                ],
                "operations": [],
            }
        )


async def run_sample_mode() -> None:
    news = DFrame(
        {
            "title": ["A", "B", "C", "D"],
            "city": ["jakarta", "bandung", "jakarta", "surabaya"],
            "sentiment": ["pos", "neg", "pos", "neu"],
            "source": ["media1", "media2", "media1", "media3"],
            "author": ["u1", "u2", "u3", "u4"],
        }
    )

    agent = DemoMultiFrameAgent({"news": news})
    result = await agent.analyze(
        "Analyze sentiment by city",
        mode="sample",
        columns_hint={"news": ["city", "sentiment"]},
    )

    print("=== SAMPLE MODE (with columns_hint) ===")
    print(result.analysis)


async def run_query_mode() -> None:
    news = DFrame(
        {
            "title": ["A", "B", "C", "D"],
            "city": ["jakarta", "bandung", "jakarta", "surabaya"],
            "sentiment": ["pos", "neg", "pos", "neu"],
            "source": ["media1", "media2", "media1", "media3"],
            "author": ["u1", "u2", "u3", "u4"],
        }
    )

    agent = DemoMultiFrameAgent({"news": news})
    # Force the simple query path (2 calls), matching backward-compatible defaults.
    agent._step = 1
    result = await agent.analyze(
        "Summarize sentiment distribution by city",
        mode="query",
        columns_hint={"news": ["city", "sentiment"]},
        max_retries=0,
    )

    print("=== QUERY MODE (with columns_hint) ===")
    print(result.analysis)
    print("queries_executed:", result.queries_executed)


async def main() -> None:
    await run_sample_mode()
    await run_query_mode()


if __name__ == "__main__":
    asyncio.run(main())

