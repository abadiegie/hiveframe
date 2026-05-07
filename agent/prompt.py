# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0
"""Structured prompt builder for LLM agents interacting with the DataFrame API."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("hiveframe.agent.prompt")

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


_KNOWN_EXAMPLE_FRAMES = {"sales", "inventory", "sales_q1", "sales_q2"}


def _filter_example_dialogues(
    examples: list[dict[str, str]],
    available_frames: list[str],
) -> list[dict[str, str]]:
    """Return only example dialogues whose frame references fit the current session."""
    available = set(available_frames)
    filtered: list[dict[str, str]] = []
    for idx in range(0, len(examples), 2):
        dialogue = examples[idx:idx + 2]
        if not dialogue:
            continue
        text = "\n".join(message.get("content", "") for message in dialogue)
        mentioned = {frame for frame in _KNOWN_EXAMPLE_FRAMES if frame in text}
        if not mentioned or mentioned.issubset(available):
            filtered.extend(dialogue)
    return filtered


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
    """Build messages for sample mode analysis.

    Args:
        instruction: Natural language instruction from the user.
        frame_contexts: Dict label -> context string per frame.
        mode: Reserved for future prompt variants.
        include_few_shot: Whether to include few-shot examples.

    Returns:
        List of message dicts ready to send to a chat LLM.
    """
    _ = mode  # reserved for future prompt variants

    available_frames = list(frame_contexts.keys())
    frames_list = ", ".join(f"`{f}`" for f in available_frames)

    system = (
        "You are a data analysis agent with access to one or more "
        "DataFrames.\n"
        "Each DataFrame is identified by a label.\n\n"
        "## CRITICAL: Available DataFrames (ONLY these)\n\n"
        f"EXACTLY these frames are available: {frames_list}\n"
        "Do NOT reference frames that are not listed above.\n"
        "Words in the instruction that are not in the list above "
        "are COLUMN NAMES, not frame labels.\n\n"
        "## Your job\n\n"
        "Analyze the data across all provided DataFrames and "
        "generate insights based on the user instruction.\n\n"
        "## Rules\n\n"
        "- Reference frames by their label (only from the list above)\n"
        "- Never invent data - only analyze what is shown\n"
        "- Cite which frames support each finding\n"
        "- Include confidence for each insight (0.0-1.0)\n"
        "- If asked to write results, include operations array\n"
        "  with cell_id, value, confidence per item\n"
        "- If the instruction asks for a chart or visualization, include a `series` array\n"
        "  with the aggregated data ready for plotting. Use value_counts or groupby from\n"
        "  the sample data shown. If sample size is too small, note it in analysis.\n\n"
        "## Response format\n\n"
        "Respond with raw JSON (no markdown fences):\n"
        "{\n"
        '  "action": "analyze|cross_reference|batch_enrich",\n'
        '  "reasoning": "...",\n'
        '  "analysis": "narrative text",\n'
        '  "insights": [{"finding": "...", "frames": [...], "confidence": 0.0}],\n'
        '  "series": [\n'
        "    {\n"
        '      "label": "series_name",\n'
        '      "x": ["A", "B", "C"],\n'
        '      "y": [10, 7, 4],\n'
        '      "x_label": "column_name",\n'
        '      "y_label": "metric_name",\n'
        '      "series_type": "bar|line|area|scatter|pie|histogram|heatmap"\n'
        "    }\n"
        "  ],\n"
        '  "operations": []\n'
        "}\n\n"
        "series rules:\n"
        "- Include series ONLY when the instruction asks for a chart/visualization\n"
        "- x and y must contain ACTUAL aggregated values from the sample shown\n"
        "- x and y lengths must match\n"
        "- max 50 points per series\n"
        "- If no visualization makes sense, series = []"
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]

    context_parts: list[str] = []
    for label, snapshot in frame_contexts.items():
        context_parts.append(f"## DataFrame: `{label}`\n\n{snapshot}")

    if context_parts:
        messages.append({"role": "system", "content": "\n\n---\n\n".join(context_parts)})

    if include_few_shot:
        relevant_examples = _filter_example_dialogues(
            MULTI_FRAME_FEW_SHOT_EXAMPLES,
            available_frames,
        )
        if relevant_examples:
            messages.extend(relevant_examples)
        else:
            logger.debug(
                "build_multi_frame_messages: skipping few-shot examples "
                "(frame mismatch). Available: %s",
                available_frames,
            )

    messages.append({"role": "user", "content": instruction})
    return messages


def build_query_generation_messages(
    instruction: str,
    frame_schemas: dict[str, str],
    reflection: str = "",
    iteration: int = 0,
) -> list[dict[str, str]]:
    """Build messages untuk fase 1 query mode."""
    available_frames = list(frame_schemas.keys())
    frames_list = ", ".join(f"`{f}`" for f in available_frames)
    first_frame = available_frames[0] if available_frames else "data"
    
    system = (
        "You are a data analyst. Given DataFrame schemas and "
        "an instruction, generate pandas queries to extract "
        "the most relevant data for analysis.\n\n"
        "## CRITICAL: Available DataFrames (ONLY these)\n\n"
        f"EXACTLY these frames exist and can be queried: {frames_list}\n"
        f"Frame labels in your response MUST match one of: {frames_list}\n"
        "Do NOT invent frame names — if the instruction mentions a word, "
        "it is most likely a COLUMN NAME inside an existing frame, not a new frame label.\n\n"
        "## Rules\n\n"
        "- Each query MUST start with 'df' — write actual executable pandas code\n"
        "- Only use pandas methods - no imports, no file I/O, no text descriptions\n"
        "- Return aggregated/filtered data, not raw full frames\n"
        "- Keep results focused - prefer top-N over all rows\n"
        "- CRITICAL: frame_label MUST be exactly one from available list above\n"
        "- CRITICAL: column names are CASE-SENSITIVE — use the EXACT column name from the schema\n"
        "- Forbidden: import, exec, eval, open, os, sys, and any non-code text\n"
        "- For visualization requests (bar chart, pie chart, histogram, line chart, scatter, etc.):\n"
        "  translate directly to an aggregation query — value_counts, groupby, describe, etc.\n"
        "  Do NOT ask for clarification. Assume a sensible default (e.g. top-20 by count).\n"
        "  Example: 'bar chart for Category' → df['Category'].value_counts().head(20)\n\n"
        "## Allowed pandas methods\n\n"
        "groupby, filter, query, nlargest, nsmallest, "
        "sort_values, head, tail, describe, value_counts, "
        "merge, pivot_table, agg, apply, loc, iloc\n\n"
        "## Response format\n\n"
        "Raw JSON only. frame_label MUST be from available list. queries values must be EXECUTABLE pandas code:\n\n"
        '{"queries":{"frame_label":"df.groupby(...).size().nlargest(10)"},'
        '"reasoning":"why this query"}\n\n'
        "## Concrete Examples\n\n"
        f"If available frame is `{first_frame}` and instruction mentions a column (e.g. 'Category'):\n\n"
        f"✓ CORRECT (use exact case from schema):\n"
        f'{{"queries": {{"{first_frame}": "df[\'Category\'].value_counts().head(10)"}},\n'
        f'"reasoning": "Count top 10 Category values"}}\n\n'
        f"❌ WRONG — made-up frame name:\n"
        f'{{"queries": {{"Category": "df[\'Category\'].value_counts()"}},\n'
        f'"reasoning": "..."}}\n\n'
        f"❌ WRONG — wrong column case (column is 'Category' not 'category'):\n"
        f'{{"queries": {{"{first_frame}": "df[\'category\'].value_counts()"}},\n'
        f'"reasoning": "..."}}\n\n'
        "All query frame_labels MUST exactly match available frames above. "
        "All column names MUST match the exact case shown in the schema."
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
                f"Generate different queries based on this reflection. "
                f"CRITICAL: Use frame labels EXACTLY as listed above: {frames_list}\n"
                f"Do not repeat the same approach that failed. "
                f"Ensure all query values are executable pandas code starting with 'df'."
            ),
        })

    if available_frames:
        relevant_examples = _filter_example_dialogues(
            QUERY_GEN_FEW_SHOT_EXAMPLES,
            available_frames,
        )
        if relevant_examples:
            messages.extend(relevant_examples)
        else:
            logger.debug(
                "build_query_generation_messages: skipping few-shot examples (frame mismatch). "
                "Available: %s", available_frames
            )

    # Strip output-mode wrapper if present (e.g. "Output mode: chart.\n...\nUser request:\n<bare>")
    # Query generation only needs the bare user request; the wrapper is for analysis phase.
    import re as _re
    _match = _re.search(r"User request:\s*\n(.+)", instruction, _re.DOTALL)
    query_instruction = _match.group(1).strip() if _match else instruction.strip()

    messages.append({"role": "user", "content": query_instruction})
    return messages


def build_code_gen_messages(
    instruction: str,
    frame_schemas: dict[str, str],
    attempt_history: list[dict] | None = None,
) -> list[dict[str, str]]:
    """Build messages for QueryExecutor code generation.

    Args:
        instruction: Original user instruction.
        frame_schemas: Dict label -> schema string with sample values.
        attempt_history: Previous failed attempts with label, code, and error.

    Returns:
        Message list ready to send to the LLM.
    """
    import re as _re

    available_frames = list(frame_schemas.keys())
    frames_list = ", ".join(f"`{f}`" for f in available_frames)

    system = (
        "You are a pandas query code generator for Hiveframe. Given DataFrame "
        "schemas and a user instruction, generate executable Python pandas code "
        "that assigns the final output to a variable named `result`.\n\n"
        "## CRITICAL: Available DataFrames (ONLY these)\n\n"
        f"EXACTLY these frames exist and can be queried: {frames_list}\n"
        "Frame labels in your response MUST match one of the labels above.\n"
        "Words in the instruction that are not frame labels are COLUMN NAMES, not new frames.\n\n"
        "## Response format\n\n"
        "Return one fenced Python block per frame, and the FIRST line inside each block "
        "must be `# frame: <label>`. Example:\n\n"
        "```python\n"
        "# frame: data\n"
        "result = df['Category'].value_counts().head(20).reset_index()\n"
        "result.columns = ['category', 'count']\n"
        "```\n\n"
        "## Rules\n\n"
        "- Assign the final output to a variable named `result`\n"
        "- CRITICAL: the ONLY DataFrame variable available in code is `df`\n"
        "- Do NOT write `data[...]`, `sales[...]`, or any frame label as a Python variable\n"
        "- Use column names EXACTLY as shown in the schema - they are CASE-SENSITIVE\n"
        "- Use only pandas operations on `df`\n"
        "- Do not use imports, file I/O, exec, eval, open, os, sys, or subprocess\n"
        "- Keep results aggregated and focused - prefer top-N over full frames\n"
        "- For chart requests, do NOT ask clarifying questions. Assume a sensible aggregation "
        "such as value_counts(), groupby(), or top-N counts\n"
        "- Return code blocks only, with no prose before or after them\n\n"
        "## Wrong vs correct\n\n"
        "❌ WRONG:\n"
        "```python\n"
        "# frame: data\n"
        "result = data['Category'].value_counts()\n"
        "```\n\n"
        "✅ CORRECT:\n"
        "```python\n"
        "# frame: data\n"
        "result = df['Category'].value_counts()\n"
        "```\n\n"
        "## Allowed pandas methods\n\n"
        "groupby, filter, query, nlargest, nsmallest, sort_values, head, tail, "
        "describe, value_counts, merge, pivot_table, agg, apply, loc, iloc, "
        "reset_index, rename, drop, dropna, fillna, astype"
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]

    schema_parts: list[str] = []
    for label, schema_str in frame_schemas.items():
        schema_parts.append(f"## DataFrame schema: `{label}`\n\n{schema_str}")
    if schema_parts:
        messages.append({"role": "system", "content": "\n\n---\n\n".join(schema_parts)})

    if attempt_history:
        history_parts = [
            "## Previous failed attempts",
            "Fix the specific errors shown below. Use the exact frame labels and exact column names from the schema.",
            "Remember: only `df` exists as a Python variable. Frame labels are metadata, not variable names.",
        ]
        for idx, attempt in enumerate(attempt_history, 1):
            history_parts.append(
                f"\n### Failed attempt {idx} — frame `{attempt.get('label', '?')}`\n"
                f"Code:\n```python\n{attempt.get('code', '').strip()}\n```\n\n"
                f"Traceback:\n```text\n{attempt.get('error', '').strip()}\n```"
            )
        messages.append({"role": "system", "content": "\n".join(history_parts)})

    match = _re.search(r"User request:\s*\n(.+)", instruction, _re.DOTALL)
    query_instruction = match.group(1).strip() if match else instruction.strip()

    messages.append({"role": "user", "content": query_instruction})
    return messages


def build_query_correction_messages(
    instruction: str,
    failed_queries: dict[str, str],
    query_errors: dict[str, str],
    frame_schemas: dict[str, str],
) -> list[dict[str, str]]:
    """Build correction prompt for failed pandas queries.

    Gives the LLM the exact error messages and correct schema so it can
    fix column name casing or other simple execution errors.

    Args:
        instruction: Original analysis instruction.
        failed_queries: Dict label -> original query string that failed.
        query_errors: Dict label -> error message from execution.
        frame_schemas: Dict label -> schema context string (from _build_schema_context).

    Returns:
        List of message dicts ready to send to a chat LLM.
    """
    available_frames = list(frame_schemas.keys())
    frames_list = ", ".join(f"`{f}`" for f in available_frames)

    system = (
        "You are a data analyst fixing failed pandas queries.\n\n"
        "## CRITICAL: Available DataFrames (ONLY these)\n\n"
        f"EXACTLY these frames exist: {frames_list}\n\n"
        "## Rules\n\n"
        "- Column names are CASE-SENSITIVE — use the EXACT name from the schema\n"
        "- Each query MUST start with 'df'\n"
        "- Only use pandas methods — no imports, no file I/O\n"
        "- CRITICAL: frame_label in response MUST be exactly one of: "
        f"{frames_list}\n\n"
        "## Response format\n\n"
        "Raw JSON only, same format as before:\n"
        '{"queries":{"frame_label":"df.corrected_expression()"},'
        '"reasoning":"what was wrong and how it was fixed"}'
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]

    # Schema context so LLM can see correct column names
    schema_parts: list[str] = []
    for label, schema_str in frame_schemas.items():
        schema_parts.append(f"## DataFrame schema: `{label}`\n\n{schema_str}")
    if schema_parts:
        messages.append({
            "role": "system",
            "content": "\n\n---\n\n".join(schema_parts),
        })

    # Failed queries + errors as the user turn
    error_lines = [f"## Original instruction\n\n{instruction}\n"]
    error_lines.append("## Failed queries and errors\n")
    for label, query_str in failed_queries.items():
        error = query_errors.get(label, "unknown error")
        error_lines.append(
            f"**{label}:**\n"
            f"  Query:  `{query_str}`\n"
            f"  Error:  `{error}`"
        )
    error_lines.append(
        "\nFix the queries above using the EXACT column names from the schema. "
        "Column names are case-sensitive. "
        "Return corrected queries in the same JSON format."
    )

    messages.append({"role": "user", "content": "\n".join(error_lines)})
    return messages


def build_review_messages(
    instruction: str,
    queries_executed: dict[str, str],
    query_results: dict[str, str],
    query_errors: dict[str, str],
    frame_schemas: dict[str, str] | None = None,
    iteration: int = 0,
    previous_verdicts: list[dict[str, str]] | None = None,
    include_few_shot: bool = True,
) -> list[dict[str, str]]:
    """Build messages for the review phase."""
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

    if frame_schemas:
        schema_parts = []
        for label, schema in frame_schemas.items():
            schema_parts.append(f"## Frame schema: `{label}`\n\n{schema}")
        parts.append("\n## Available schemas\n\n" + "\n\n---\n\n".join(schema_parts))

    if queries_executed:
        parts.append("\n## Queries executed")
        for label, query in queries_executed.items():
            parts.append(f"**{label}:** `{query}`")

    if query_results:
        for label, result_str in query_results.items():
            query_used = queries_executed.get(label, "")
            parts.append(
                f"\n## Query result: `{label}`\n\n"
                f"Query: `{query_used}`\n\n"
                f"Result:\n```\n{result_str}\n```"
            )

    if query_errors:
        parts.append("\n## Query errors")
        for label, error in query_errors.items():
            parts.append(f"- `{label}`: {error}")

    messages.append({"role": "system", "content": "\n\n".join(parts)})

    if include_few_shot:
        available_frames = sorted(
            set(frame_schemas or {})
            | set(queries_executed)
            | set(query_results)
            | set(query_errors)
        )
        relevant_examples = _filter_example_dialogues(
            REVIEW_FEW_SHOT_EXAMPLES,
            available_frames,
        )
        if relevant_examples:
            messages.extend(relevant_examples)
        else:
            logger.debug(
                "build_review_messages: skipping few-shot examples (frame mismatch). Available: %s",
                available_frames,
            )

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
    '      "label": "series_name",\n'
    '      "x": ["A", "B", "C"],\n'
    '      "y": [10, 7, 4],\n'
    '      "x_label": "column_name",\n'
    '      "y_label": "metric_name",\n'
    '      "series_type": "bar|line|area|scatter|pie|histogram|heatmap"\n'
    "    }\n"
    "  ],\n"
    '  "operations": []\n'
    "}\n\n"
    "## Series rules\n\n"
    "- Include series ONLY when data is suitable for visualization\n"
    "- label must be concise and descriptive\n"
    "- x and y must be the ACTUAL aggregated arrays from query results\n"
    "- x and y lengths must match\n"
    "- Keep data focused: max 200 points per series\n"
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

    if query_results:
        available = ", ".join(
            f"`{label}` ({result_str.count(chr(10))} rows)"
            for label, result_str in query_results.items()
        )
        user_content = (
            f"Query results available: {available}. "
            "Generate analysis based on the query results above."
        )
    else:
        user_content = "Generate analysis based on the query results above."

    messages.append({"role": "user", "content": user_content})
    return messages


_NORMALIZE_SYSTEM_PROMPT = (
    "You are a data normalization agent. Your task is to standardize and enrich values in a DataFrame.\n\n"
    "## Context\n\n"
    "You will receive:\n"
    "1. A normalization instruction (e.g., 'standardize province names')\n"
    "2. Chunk of rows from target column (with context columns)\n"
    "3. Data patterns to help you standardize consistently\n\n"
    "## Your job\n\n"
    "For each row in the chunk:\n"
    "1. Analyze the current value in context of other columns\n"
    "2. Apply the normalization rule consistently\n"
    "3. Return transformed value with confidence score\n\n"
    "## Cell ID convention\n\n"
    "Format: {frame_id}::{column_name}_{row_index}\n"
    "- frame_id: unique identifier for the DataFrame\n"
    "- column_name: target column being normalized\n"
    "- row_index: zero-based row number\n"
    "Example: 'abc123::city_5' means frame 'abc123', column 'city', row 5\n\n"
    "## Confidence scoring\n\n"
    "1.00 (certain):  Exact match or very clear transformation\n"
    "0.80-0.94:       High confidence, minor ambiguity\n"
    "0.60-0.79:       Medium confidence, some inference needed\n"
    "<0.60:           Do NOT write — return empty operations\n\n"
    "## Rules\n\n"
    "- Always analyze all available context (other columns in same row)\n"
    "- Never invent values — only transform what exists\n"
    "- If transformation is unclear, set confidence <0.60 instead of guessing\n"
    "- Be consistent: same input → same output across chunk\n"
    "- Include reasoning for each value in the cell\n\n"
    "## Response format\n\n"
    "Respond with raw JSON (no markdown fences):\n"
    "{\n"
    '  "action": "batch_enrich",\n'
    '  "reasoning": "narrative explanation of normalization logic",\n'
    '  "operations": [\n'
    "    {\n"
    '      "cell_id": "{frame_id}::{column_name}_{row_index}",\n'
    '      "value": "normalized_value",\n'
    '      "confidence": 0.85,\n'
    '      "cell_reasoning": "why this value at this row"\n'
    "    }\n"
    "  ]\n"
    "}"
)


def build_normalize_messages(
    instruction: str,
    chunk_snapshot: str,
    frame_id: str,
    column_name: str,
    chunk_start: int,
    context_columns: list[str] | None = None,
    chunk_row_count: int | None = None,
) -> list[dict[str, str]]:
    """
    Build LLM messages for stream_normalize with context awareness.

    Args:
        instruction: Normalization rule/instruction (e.g., "standardize to proper province names").
        chunk_snapshot: String representation of the chunk (with target column + context).
        frame_id: DFrame identifier for cell_id prefixing.
        column_name: Name of the column being normalized.
        chunk_start: Starting row index of this chunk.
        context_columns: Optional list of context columns included (for display in prompt).

    Returns:
        List of messages ready to send to LLM.

    Example::

        chunk_str = df.iloc[0:10][['city', 'region', 'population']].to_string()
        messages = build_normalize_messages(
            instruction="Standardize city names to proper Indonesian city format",
            chunk_snapshot=chunk_str,
            frame_id="abc123",
            column_name="city",
            chunk_start=0,
            context_columns=["city", "region", "population"],
        )
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": _NORMALIZE_SYSTEM_PROMPT}]

    context_parts: list[str] = [
        f"## Normalization Task\n\n{instruction}",
        f"\n## Target Information\n\n"
        f"Frame ID: `{frame_id}`\n"
        f"Column: `{column_name}`\n"
        f"Row index range in this chunk: {chunk_start} to "
        f"{chunk_start + (chunk_row_count - 1 if chunk_row_count else max(0, chunk_snapshot.count(chr(10)) - 1))} "
        f"(0-based, these are the exact row numbers to use in cell_ids)",
    ]

    if context_columns:
        context_parts.append(
            f"\n## Context Columns\n\n"
            f"Use these columns to inform your normalization decision:\n"
            f"{', '.join(f'`{c}`' for c in context_columns)}"
        )

    context_parts.append(
        f"\n## Data to Normalize\n\n"
        f"```\n{chunk_snapshot}\n```\n"
        f"*Rows above are indexed starting from {chunk_start}. "
        f"Use these row numbers in your cell_ids.*"
    )

    messages.append({"role": "system", "content": "\n".join(context_parts)})

    messages.append({
        "role": "user",
        "content": (
            f"Normalize the '{column_name}' column according to the instruction above. "
            f"Generate cell_ids using the row numbers shown (starting from {chunk_start}). "
            f"Include reasoning for each normalized value."
        ),
    })

    return messages