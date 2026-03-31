# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""
LLM Agent usage example — structured prompt + AgentWriter.

Run:
    python examples/llm_agent_usage.py

For real LLM integration (OpenAI, Anthropic, etc.), see with_real_llm_example() below.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from agent.prompt import build_messages, parse_plan
from agent.writer import AgentWriter
from core.cluster_runtime import ClusterRuntime, RuntimeConfig
from core.dataframe import DFrame


# ---------------------------------------------------------------------------
# Mock LLM (no API key needed)
# ---------------------------------------------------------------------------

def mock_llm(messages: list[dict[str, str]], frame_id: str) -> str:
    """Simulate LLM response — replace with a real client for production."""
    last_user = next(m["content"] for m in reversed(messages) if m["role"] == "user")

    if "normalize" in last_user.lower() or "city" in last_user.lower():
        return json.dumps({
            "action": "batch_enrich",
            "reasoning": "Normalizing city values to official Indonesian province names.",
            "operations": [
                {"cell_id": f"{frame_id}::city_0", "value": "DKI Jakarta", "confidence": 0.97},
                {"cell_id": f"{frame_id}::city_1", "value": "Jawa Barat",  "confidence": 0.95},
            ],
        })

    if "score" in last_user.lower() and "set" in last_user.lower():
        return json.dumps({
            "action": "normalize",
            "reasoning": "User explicitly set score for row 0.",
            "operations": [
                {"cell_id": f"{frame_id}::score_0", "value": 99, "confidence": 1.0},
            ],
        })

    return json.dumps({
        "action": "read",
        "reasoning": "User wants to read data, no write needed.",
        "operations": [],
    })


# ---------------------------------------------------------------------------
# Agent executor
# ---------------------------------------------------------------------------

async def execute_plan(plan: dict[str, Any], writer: AgentWriter, df: DFrame) -> None:
    """Translate LLM plan into DFrame API calls."""
    action = plan.get("action")
    ops = plan.get("operations", [])
    reasoning = plan.get("reasoning", "-")

    print(f"\n[Agent] action={action}")
    print(f"[Agent] reasoning: {reasoning}")

    if action == "normalize" and ops:
        item = ops[0]
        await writer.normalize(item["cell_id"], item["value"], confidence=item["confidence"])
        print(f"[Agent] normalized {item['cell_id']} → {item['value']} (conf={item['confidence']})")

    elif action == "batch_enrich" and ops:
        await writer.batch_enrich(ops)
        for item in ops:
            print(f"[Agent] enriched {item['cell_id']} → {item['value']} (conf={item['confidence']})")

    elif action == "read":
        print("[Agent] read result:")
        print(df.read_fresh())

    elif action == "describe":
        print("[Agent] describe result:")
        print(df.describe())

    else:
        print("[Agent] no write action performed.")


# ---------------------------------------------------------------------------
# Demo — standalone mode
# ---------------------------------------------------------------------------

async def standalone_demo() -> None:
    print("=" * 50)
    print("DEMO: Standalone mode")
    print("=" * 50)

    df = DFrame({"city": ["jakarta", "bandung"], "score": [85, 92]})
    writer = AgentWriter(df._coordinator, agent_id="normalizer", author_type="llm_normalization")

    snapshot = df.read_fresh().to_string()
    messages = build_messages(
        user_instruction="Normalize all city names to official Indonesian province names.",
        dataframe_snapshot=snapshot,
    )

    llm_response = mock_llm(messages, df._frame_id)
    plan = parse_plan(llm_response)
    await execute_plan(plan, writer, df)

    print("\n[Result] after normalization:")
    print(df.read_fresh())


# ---------------------------------------------------------------------------
# Demo — cluster mode
# ---------------------------------------------------------------------------

async def cluster_demo() -> None:
    print("\n" + "=" * 50)
    print("DEMO: Cluster mode")
    print("=" * 50)

    runtime = ClusterRuntime(
        RuntimeConfig(node_id="writer-1", role="write", port=19200, enable_cluster=True)
    )
    await runtime.start()

    df = DFrame.from_runtime(runtime, {"city": ["jakarta", "bandung"], "score": [85, 92]})
    writer = AgentWriter(
        runtime.coordinator,
        agent_id="normalizer",
        author_type="llm_normalization",
    )

    snapshot = df.read_fresh().to_string()
    messages = build_messages(
        user_instruction="Set score row 0 to 99.",
        dataframe_snapshot=snapshot,
    )

    llm_response = mock_llm(messages, df._frame_id)
    plan = parse_plan(llm_response)
    await execute_plan(plan, writer, df)

    print("\n[Result] after cluster write:")
    print(df.read_fresh())

    # Show global read from all nodes
    merged = await df.read_fresh_async()
    print("\n[Result] global read (all nodes merged):")
    print(merged)


# ---------------------------------------------------------------------------
# Real LLM integration example (not executed directly)
# ---------------------------------------------------------------------------

def with_real_llm_example() -> None:
    """
    Example using OpenAI client (requires OPENAI_API_KEY):

        from openai import OpenAI
        from agent.prompt import build_messages, parse_plan

        client = OpenAI()
        df = DFrame({"city": ["jakarta"]})
        snapshot = df.read_fresh().to_string()

        messages = build_messages(
            user_instruction="Normalize city names to proper Indonesian province names.",
            dataframe_snapshot=snapshot,
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )

        plan = parse_plan(response.choices[0].message.content)
        # frame_id is available via df._frame_id
        # The LLM will use it in cell_id: "{frame_id}::city_0"
        await execute_plan(plan, writer, df)
    """


if __name__ == "__main__":
    asyncio.run(standalone_demo())
    asyncio.run(cluster_demo())
