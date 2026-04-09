# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

import asyncio

from agent.writer import AgentWriter
from core.dataframe import DFrame


def test_stream_normalize_uses_default_instruction(monkeypatch):
    df = DFrame({"city": ["jakarta", "bandung"]})
    writer = AgentWriter(
        coordinator=df._coordinator,
        agent_id="normalizer",
        author_type="llm_normalization",
        frame_id=df._frame_id,
    )

    captured_user_instructions: list[str] = []

    async def fake_llm_call(messages):
        captured_user_instructions.append(messages[-1]["content"])
        return [{"cell_id": "ignored", "value": "x", "confidence": 0.9}]

    async def fake_batch_enrich(items):
        return {"written": len(items), "skipped": 0}

    monkeypatch.setattr(writer, "batch_enrich", fake_batch_enrich)

    result = asyncio.run(writer.stream_normalize("city", fake_llm_call, chunk_size=1))

    assert result["total"] == 2
    assert captured_user_instructions == ["Normalize column 'city'", "Normalize column 'city'"]


def test_stream_normalize_uses_custom_instruction(monkeypatch):
    df = DFrame({"city": ["jakarta", "bandung"]})
    writer = AgentWriter(
        coordinator=df._coordinator,
        agent_id="normalizer",
        author_type="llm_normalization",
        frame_id=df._frame_id,
    )

    custom_instruction = "Standardize city names to official province names in English."
    captured_user_instructions: list[str] = []

    async def fake_llm_call(messages):
        captured_user_instructions.append(messages[-1]["content"])
        return [{"cell_id": "ignored", "value": "x", "confidence": 0.9}]

    async def fake_batch_enrich(items):
        return {"written": len(items), "skipped": 0}

    monkeypatch.setattr(writer, "batch_enrich", fake_batch_enrich)

    _ = asyncio.run(
        writer.stream_normalize(
            "city",
            fake_llm_call,
            chunk_size=1,
            custom_instruction=custom_instruction,
        )
    )

    assert captured_user_instructions == [custom_instruction, custom_instruction]


def test_stream_normalize_backwards_compatible_positional_progress(monkeypatch):
    df = DFrame({"city": ["jakarta", "bandung", "medan"]})
    writer = AgentWriter(
        coordinator=df._coordinator,
        agent_id="normalizer",
        author_type="llm_normalization",
        frame_id=df._frame_id,
    )

    progress_events: list[tuple[int, int]] = []

    async def fake_llm_call(messages):
        return [{"cell_id": "ignored", "value": "x", "confidence": 0.9}]

    async def fake_batch_enrich(items):
        return {"written": len(items), "skipped": 0}

    def progress_callback(done, total):
        progress_events.append((done, total))

    monkeypatch.setattr(writer, "batch_enrich", fake_batch_enrich)

    result = asyncio.run(
        writer.stream_normalize(
            "city",
            fake_llm_call,
            1,
            progress_callback,
        )
    )

    assert result["total"] == 3
    assert progress_events == [(1, 3), (2, 3), (3, 3)]

