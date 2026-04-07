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


MULTI_FRAME_FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {
        "role": "user",
        "content": (
            "Bandingkan total penjualan per region antara "
            "frame sales_q1 dan sales_q2"
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"action":"cross_reference",'
            '"reasoning":"Compare regional sales between Q1 and Q2.",'
            '"analysis":"Region Jawa Barat menunjukkan pertumbuhan '
            '23% dari Q1 ke Q2, sementara DKI Jakarta stagnan di '
            '+2%. Region Sulawesi mengalami penurunan 8%.",'
            '"insights":[{'
            '"finding":"Jawa Barat growth 23% Q1->Q2",'
            '"frames":["sales_q1","sales_q2"],'
            '"confidence":0.95}],'
            '"operations":[]}'
        ),
    },
]


QUERY_GEN_FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {
        "role": "user",
        "content": (
            "Produk mana yang penjualannya tinggi tapi stok menipis? "
            "frames: sales, inventory"
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"queries":{'
            '"sales":"df.groupby(\'product_id\')[\'qty_sold\']'
            '.sum().nlargest(50).reset_index()",'
            '"inventory":"df[df[\'stock_qty\'] < '
            'df[\'reorder_point\']][['
            '\'product_id\',\'stock_qty\',\'reorder_point\']]"'
            '},'
            '"reasoning":"Get top 50 products by sales, '
            'cross with items below reorder point."}'
        ),
    },
]


REVIEW_FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {
        "role": "user",
        "content": (
            "Instruction: Produk mana yang sales tinggi tapi stok menipis?\n"
            "Results:\n"
            "sales: top 20 products by qty_sold OK\n"
            "inventory: ERROR - column 'stock_qty' not found"
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"status":"partial",'
            '"reason":"Sales data is complete but inventory query failed",'
            '"reflection":"inventory query used wrong column name. '
            'Use stock_remaining instead of stock_qty",'
            '"missing_parts":["inventory stock data"],'
            '"suggested_queries":{"inventory":"df.nsmallest(50, '
            '\'stock_remaining\')[[\'product_id\',\'stock_remaining\']]"},'
            '"accepted_labels":["sales"],'
            '"needs_columns":[],'
            '"merge_ready":false}'
        ),
    },
    {
        "role": "user",
        "content": (
            "Instruction: Bandingkan revenue Q1 vs Q2 per region\n"
            "Results:\n"
            "sales_q1: revenue per region OK\n"
            "sales_q2: revenue per region OK"
        ),
    },
    {
        "role": "assistant",
        "content": (
            '{"status":"merge",'
            '"reason":"Both quarters data is available and sufficient",'
            '"reflection":"",'
            '"missing_parts":[],'
            '"suggested_queries":{},'
            '"accepted_labels":["sales_q1","sales_q2"],'
            '"needs_columns":[],'
            '"merge_ready":true}'
        ),
    },
]


_REVIEW_SYSTEM_PROMPT = (
    "You are evaluating query results to determine if they are "
    "sufficient to answer the original instruction.\n\n"
    "## Verdict options\n\n"
    "accepted  -> Results fully answer the instruction. Proceed to analysis.\n"
    "partial   -> Some results are useful, but specific parts are missing. "
    "Declare which frame labels are already accepted and what additional queries are needed.\n"
    "error     -> One or more queries failed technically. Suggest corrected queries.\n"
    "plan      -> Results are insufficient because you need columns not yet visible in the schema. "
    "List the column names you need.\n"
    "rejected  -> Results are not relevant to the instruction. The query approach was wrong. "
    "Suggest a completely different approach.\n"
    "merge     -> You have sufficient partial results across multiple queries "
    "that can be combined to answer the instruction.\n\n"
    "## Rules\n\n"
    "- Be specific about what is missing or wrong\n"
    "- For 'partial': always list accepted_labels so accepted parts are not re-queried\n"
    "- For 'plan': only request columns from frames that already exist in the frames dict\n"
    "- For 'suggested_queries': start each query with 'df'\n"
    "- Prefer 'partial' or 'merge' over 'rejected' when some data is useful\n\n"
    "## Response format\n\n"
    "Raw JSON only:\n"
    '{"status":"accepted|partial|error|plan|rejected|merge",'
    '"reason":"brief explanation",'
    '"reflection":"what to fix (empty if accepted)",'
    '"missing_parts":["description of what is missing"],'
    '"suggested_queries":{"frame_label":"df_expression"},'
    '"accepted_labels":["frame_labels_already_ok"],'
    '"needs_columns":["col_name_1","col_name_2"],'
    '"merge_ready":false}'
)


def build_multi_frame_messages(
    instruction: str,
    frame_contexts: dict[str, str],
    mode: str = "sample",
    include_few_shot: bool = True,
) -> list[dict[str, str]]:
    """Build messages untuk sample mode analysis."""
    _ = mode  # reserved for future prompt variants
    system = (
        "You are a data analysis agent with access to one or more "
        "DataFrames.\n"
        "Each DataFrame is identified by a label.\n\n"
        "## Your job\n\n"
        "Analyze the data across all provided DataFrames and "
        "generate insights based on the user instruction.\n\n"
        "## Rules\n\n"
        "- Reference frames by their label\n"
        "- Never invent data - only analyze what is shown\n"
        "- Cite which frames support each finding\n"
        "- Include confidence for each insight (0.0-1.0)\n"
        "- If asked to write results, include operations array\n"
        "  with cell_id, value, confidence per item\n\n"
        "## Response format\n\n"
        "Respond with raw JSON (no markdown fences):\n"
        '{"action":"analyze|cross_reference|batch_enrich",'
        '"reasoning":"...","analysis":"narrative text",'
        '"insights":[{"finding":"...","frames":[...],'
        '"confidence":0.0}],'
        '"operations":[]}'
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]

    context_parts: list[str] = []
    for label, snapshot in frame_contexts.items():
        context_parts.append(f"## DataFrame: `{label}`\n\n{snapshot}")

    if context_parts:
        messages.append({"role": "system", "content": "\n\n---\n\n".join(context_parts)})

    if include_few_shot:
        messages.extend(MULTI_FRAME_FEW_SHOT_EXAMPLES)

    messages.append({"role": "user", "content": instruction})
    return messages


def build_query_generation_messages(
    instruction: str,
    frame_schemas: dict[str, str],
    reflection: str = "",
    iteration: int = 0,
) -> list[dict[str, str]]:
    """Build messages untuk fase 1 query mode."""
    system = (
        "You are a data analyst. Given DataFrame schemas and "
        "an instruction, generate pandas queries to extract "
        "the most relevant data for analysis.\n\n"
        "## Rules\n\n"
        "- Each query MUST start with 'df'\n"
        "- Only use pandas methods - no imports, no file I/O\n"
        "- Return aggregated/filtered data, not raw full frames\n"
        "- Keep results focused - prefer top-N over all rows\n"
        "- If instruction only needs one frame, query one frame\n"
        "- Forbidden: import, exec, eval, open, os, sys\n\n"
        "## Allowed pandas methods\n\n"
        "groupby, filter, query, nlargest, nsmallest, "
        "sort_values, head, tail, describe, value_counts, "
        "merge, pivot_table, agg, apply, loc, iloc\n\n"
        "## Response format\n\n"
        "Raw JSON only:\n"
        '{"queries":{"frame_label":"df_expression",...},'
        '"reasoning":"why these queries"}'
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]

    schema_parts: list[str] = []
    for label, schema_str in frame_schemas.items():
        schema_parts.append(f"## DataFrame schema: `{label}`\n\n{schema_str}")

    if schema_parts:
        messages.append({"role": "system", "content": "\n\n---\n\n".join(schema_parts)})

    if reflection and iteration > 0:
        messages.append({
            "role": "system",
            "content": (
                f"## Reflection from previous attempt (attempt {iteration})\n\n"
                f"{reflection}\n\n"
                "Generate different queries based on this reflection. "
                "Do not repeat the same approach that failed."
            ),
        })

    messages.extend(QUERY_GEN_FEW_SHOT_EXAMPLES)
    messages.append({"role": "user", "content": instruction})
    return messages


def build_review_messages(
    instruction: str,
    queries_executed: dict[str, str],
    query_results: dict[str, str],
    query_errors: dict[str, str],
    iteration: int = 0,
    previous_verdicts: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build messages untuk Review phase."""
    messages: list[dict[str, str]] = [{"role": "system", "content": _REVIEW_SYSTEM_PROMPT}]

    parts = [
        f"## Original instruction\n\n{instruction}",
        f"\n## Iteration {iteration + 1}",
    ]

    if previous_verdicts:
        history_lines: list[str] = []
        for idx, verdict in enumerate(previous_verdicts, 1):
            history_lines.append(
                f"  Attempt {idx}: {verdict.get('status', '?')} - {verdict.get('reason', '')}"
            )
        parts.append("\n## Previous attempts\n" + "\n".join(history_lines))

    if queries_executed:
        parts.append("\n## Queries executed")
        for label, query in queries_executed.items():
            parts.append(f"**{label}:** `{query}`")

    if query_results:
        parts.append("\n## Query results")
        for label, result_str in query_results.items():
            parts.append(f"**{label}:**\n```\n{result_str}\n```")

    if query_errors:
        parts.append("\n## Query errors")
        for label, error in query_errors.items():
            parts.append(f"- `{label}`: {error}")

    messages.append({"role": "system", "content": "\n\n".join(parts)})
    messages.extend(REVIEW_FEW_SHOT_EXAMPLES)
    messages.append({
        "role": "user",
        "content": (
            "Evaluate the results above. Are they sufficient to answer "
            "the instruction? Provide your verdict."
        ),
    })
    return messages


_ANALYSIS_WITH_SERIES_PROMPT = (
    "You are a data analyst generating insights from query results.\n\n"
    "## Output format\n\n"
    "Respond with raw JSON:\n"
    "{\n"
    '  "action": "analyze",\n'
    '  "reasoning": "brief explanation of approach",\n'
    '  "analysis": "narrative analysis text",\n'
    '  "insights": [\n'
    '    {"finding": "specific finding", "frames": ["frame_label"], "confidence": 0.0}\n'
    "  ],\n"
    '  "series": [\n'
    "    {\n"
    '      "name": "snake_case_identifier",\n'
    '      "description": "what this data shows",\n'
    '      "suggested_x": "column_name",\n'
    '      "suggested_y": "column_name or [list]",\n'
    '      "suggested_group_by": "column_name or null",\n'
    '      "unit": "IDR|units|%|etc or empty string",\n'
    '      "source_frames": ["frame_label"],\n'
    '      "data": [{"col": "val"}, ...]\n'
    "    }\n"
    "  ],\n"
    '  "operations": []\n'
    "}\n\n"
    "## Series rules\n\n"
    "- Include series ONLY when data is suitable for visualization\n"
    "- name must be snake_case, unique, descriptive\n"
    "- data must be the ACTUAL aggregated data from query results\n"
    "- Keep data rows focused: max 200 rows per series\n"
    '- suggested_y can be a list for multi-line charts e.g. ["revenue", "target"]\n'
    "- If no visualization makes sense, series = []\n"
    "- Do NOT invent data — only use data from query results\n"
)


def build_analysis_messages(
    instruction: str,
    query_results: dict[str, str],
    query_errors: dict[str, str],
    original_queries: dict[str, str],
) -> list[dict[str, str]]:
    """Build messages untuk fase 2 query mode."""
    messages: list[dict[str, str]] = [{"role": "system", "content": _ANALYSIS_WITH_SERIES_PROMPT}]

    result_parts = [f"## Original instruction\n\n{instruction}"]

    for label, result_str in query_results.items():
        query_used = original_queries.get(label, "")
        result_parts.append(
            f"## Query result: `{label}`\n\n"
            f"Query: `{query_used}`\n\n"
            f"Result:\n```\n{result_str}\n```"
        )

    if query_errors:
        error_parts = [f"- `{label}`: {err}" for label, err in query_errors.items()]
        result_parts.append(
            "## Query errors (frames with errors could not be analyzed)\n\n"
            + "\n".join(error_parts)
        )

    messages.append({"role": "system", "content": "\n\n---\n\n".join(result_parts)})
    messages.append({"role": "user", "content": "Generate analysis based on the query results above."})
    return messages



