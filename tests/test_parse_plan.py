# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for agent.prompt.parse_plan()."""

import pytest

from agent.prompt import parse_plan


# --- Happy path ---

def test_raw_json_object() -> None:
    raw = '{"action":"batch_enrich","operations":[]}'
    result = parse_plan(raw)
    assert result["action"] == "batch_enrich"
    assert result["operations"] == []


def test_raw_json_with_whitespace() -> None:
    raw = '  \n  {"action":"read","operations":[]}\n  '
    result = parse_plan(raw)
    assert result["action"] == "read"


def test_json_fenced_block() -> None:
    raw = '```json\n{"action":"normalize","operations":[]}\n```'
    result = parse_plan(raw)
    assert result["action"] == "normalize"


def test_json_fenced_block_with_prose() -> None:
    raw = (
        "Here is the result:\n"
        "```json\n"
        '{"action":"batch_enrich","reasoning":"ok","operations":[{"cell_id":"abc::col_0","value":"x","confidence":0.9}]}\n'
        "```\n"
        "That's it."
    )
    result = parse_plan(raw)
    assert result["action"] == "batch_enrich"
    assert len(result["operations"]) == 1


def test_embedded_json_no_fence() -> None:
    raw = 'The agent returned: {"action":"describe","operations":[]}'
    result = parse_plan(raw)
    assert result["action"] == "describe"


def test_full_batch_enrich_payload() -> None:
    raw = (
        '{"action":"batch_enrich","reasoning":"classified stance",'
        '"operations":[{"cell_id":"abc123::stance_0","value":"pro","confidence":0.92},'
        '{"cell_id":"abc123::stance_1","value":"against","confidence":0.87}]}'
    )
    result = parse_plan(raw)
    assert len(result["operations"]) == 2
    assert result["operations"][0]["value"] == "pro"
    assert result["operations"][1]["confidence"] == 0.87


# --- Fallback / error cases ---

def test_empty_string_returns_empty_dict() -> None:
    assert parse_plan("") == {}


def test_plain_text_returns_empty_dict() -> None:
    assert parse_plan("Sorry, I cannot process that.") == {}


def test_malformed_json_returns_empty_dict() -> None:
    assert parse_plan('{"action": "batch_enrich", "operations": [}') == {}


def test_fenced_block_malformed_json_falls_through_to_empty() -> None:
    raw = "```json\n{broken json\n```"
    result = parse_plan(raw)
    assert result == {}


def test_partial_json_in_prose_extracted() -> None:
    raw = 'Prefix text {"action":"read"} suffix text'
    result = parse_plan(raw)
    assert result.get("action") == "read"


def test_nested_operations_preserved() -> None:
    raw = (
        '{"action":"batch_enrich","operations":['
        '{"cell_id":"f1::col_3","value":42,"confidence":1.0,"cell_reasoning":"exact match"}]}'
    )
    result = parse_plan(raw)
    op = result["operations"][0]
    assert op["value"] == 42
    assert op["cell_reasoning"] == "exact match"


def test_unicode_content_handled() -> None:
    raw = '{"action":"batch_enrich","reasoning":"Provinsi Jawa Barat","operations":[]}'
    result = parse_plan(raw)
    assert "Jawa Barat" in result["reasoning"]


def test_none_string_returns_falsy() -> None:
    # json.loads("null") == None, which is falsy — callers should treat it as empty
    result = parse_plan("null")
    assert not result


def test_list_json_returns_non_empty_list() -> None:
    # parse_plan returns the list as-is when JSON root is an array
    result = parse_plan('[{"action":"batch_enrich"}]')
    assert isinstance(result, list)



