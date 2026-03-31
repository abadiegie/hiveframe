# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0
"""Structured prompt builder for LLM agents interacting with the DataFrame API."""
from __future__ import annotations
from typing import Any
SYSTEM_PROMPT = (
    "You are a data transformation agent with access to a transactional distributed DataFrame engine.\n"
    "Your job is to process user instructions and translate them into precise API calls.\n\n"
    "## Available API\n\n"
    "### 1. Read data\n\n"
    "df.read_fresh() -> pd.DataFrame  # local snapshot, sync\n"
    "await df.read_fresh_async() -> pd.DataFrame  # global fan-out, async\n"
    "df.head(), df.groupby(), df.describe(), df.to_csv(), ...  # full pandas proxy\n\n"
    "### 2. Write data (transactional)\n\n"
    "await writer.normalize('{frame_id}::col_row', value, confidence=0.0..1.0)\n"
    "await writer.batch_enrich([{'cell_id': '{frame_id}::col_row', 'value': v, 'confidence': c}, ...])\n\n"
    "### 3. Cell ID convention\n\n"
    "Format: {frame_id}::{{column_name}}_{{row_index}}  (row_index is zero-based)\n"
    "frame_id is available via df._frame_id\n"
    "Example: '{frame_id}::city_0'  -> DFrame {frame_id}, column city, row 0\n\n"
    "### 4. Confidence scoring\n\n"
    "0.95-1.00: very high  |  0.80-0.94: high  |  0.60-0.79: medium  |  <0.60: do not write\n\n"
    "### 5. Rules\n\n"
    "- Never invent data.\n"
    "- Always include confidence for every write.\n"
    "- Always use the full namespaced cell_id with frame_id prefix.\n"
    "- Prefer batch_enrich() for bulk column updates.\n\n"
    "## Response format\n\n"
    "Respond with a raw JSON object (no markdown fences):\n"
    '{"action":"normalize|batch_enrich|read|describe","reasoning":"...","operations":[...]}'
)
FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {
        "role": "user",
        "content": "Normalize all city values to proper Indonesian province names. frame_id=abc123",
    },
    {
        "role": "assistant",
        "content": '{"action":"batch_enrich","reasoning":"Normalizing city cells to official province names.","operations":[{"cell_id":"abc123::city_0","value":"DKI Jakarta","confidence":0.97},{"cell_id":"abc123::city_1","value":"Jawa Barat","confidence":0.95}]}',
    },
    {
        "role": "user",
        "content": "What is the average score per city?",
    },
    {
        "role": "assistant",
        "content": '{"action":"read","reasoning":"User wants aggregated statistics via groupby.","operations":[]}',
    },
    {
        "role": "user",
        "content": "Set score for row 2 to 88. frame_id=abc123",
    },
    {
        "role": "assistant",
        "content": '{"action":"normalize","reasoning":"User explicitly set score for row 2.","operations":[{"cell_id":"abc123::score_2","value":88,"confidence":1.0}]}',
    },
]
def build_messages(
    user_instruction: str,
    dataframe_snapshot: str | None = None,
    frame_id: str | None = None,
    include_few_shot: bool = True,
) -> list[dict[str, str]]:
    """Build a full message list ready to send to an LLM chat API.
    Args:
        user_instruction: Natural language task from user or orchestrator.
        dataframe_snapshot: Optional string representation of current df state.
        frame_id: The DFrame frame_id so the LLM can build correct namespaced cell_ids.
        include_few_shot: Whether to include few-shot examples (default True).
    Returns:
        List of dicts with 'role' and 'content' keys.
    Example::
        messages = build_messages(
            user_instruction="Normalize city names",
            dataframe_snapshot=df.read_fresh().to_string(),
            frame_id=df._frame_id,
        )
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    context_parts: list[str] = []
    if frame_id:
        context_parts.append(
            f"## DFrame ID\n\n`{frame_id}`\n\n"
            f"Use this as the prefix for all cell_ids.\n"
            f"Format: `{frame_id}::{{column_name}}_{{row_index}}`\n"
            f"Example: `{frame_id}::city_0` means column 'city', row 0"
        )
    if dataframe_snapshot:
        context_parts.append(f"## Current DataFrame State\n\n```\n{dataframe_snapshot}\n```")
    if context_parts:
        messages.append({"role": "system", "content": "\n\n".join(context_parts)})
    if include_few_shot:
        messages.extend(FEW_SHOT_EXAMPLES)
    messages.append({"role": "user", "content": user_instruction})
    return messages
def build_system_only() -> str:
    """Return just the system prompt string."""
    return SYSTEM_PROMPT
def parse_plan(response_text: str) -> dict[str, Any]:
    """Parse JSON plan from LLM response text.
    Handles raw JSON, ```json fenced blocks, and embedded JSON.
    Returns parsed dict or empty dict if no valid JSON found.
    """
    import json
    import re
    # 1. Try direct parse (raw JSON — most mock and well-behaved LLMs)
    try:
        return json.loads(response_text.strip())
    except (json.JSONDecodeError, ValueError):
        pass
    # 2. Try ```json ... ``` code fence
    match = re.search(r"```json\s*([\s\S]*?)\s*```", response_text)
    if match:
        try:
            return json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            pass
    # 3. Try first { ... } block
    match = re.search(r"{[\s\S]*}", response_text)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            pass
    return {}
