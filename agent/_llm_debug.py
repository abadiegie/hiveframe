"""Helpers for compact LLM debug logging."""

from __future__ import annotations

import json
from typing import Any


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def truncate_text(value: Any, limit: int = 500) -> str:
    text = _stringify(value).replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…"


def summarize_messages(messages: list[dict[str, Any]], preview_limit: int = 700) -> str:
    parts: list[str] = []
    for index, message in enumerate(messages[:3], start=1):
        role = message.get("role", "?")
        content = truncate_text(message.get("content", ""), limit=220)
        parts.append(f"{index}:{role}={content}")
    if len(messages) > 3:
        parts.append(f"...(+{len(messages) - 3} more)")
    return truncate_text(" | ".join(parts), limit=preview_limit)


def summarize_operations(operations: list[dict[str, Any]], preview_limit: int = 700) -> str:
    sample: list[dict[str, Any]] = []
    for item in operations[:3]:
        sample.append({
            "cell_id": item.get("cell_id"),
            "confidence": item.get("confidence"),
            "value": truncate_text(item.get("value", item.get("new_value")), limit=120),
        })
    suffix = ""
    if len(operations) > 3:
        suffix = f" ...(+{len(operations) - 3} more)"
    return truncate_text(f"{_stringify(sample)}{suffix}", limit=preview_limit)