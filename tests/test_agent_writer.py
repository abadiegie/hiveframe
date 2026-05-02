# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import pytest

from agent.writer import AgentWriter
from core.coordinator import TransactionCoordinator
from core.dataframe import DFrame


def test_stream_normalize_uses_default_instruction(monkeypatch):
    df = DFrame({"city": ["jakarta", "bandung"]})
    writer = AgentWriter(
        coordinator=df._coordinator,
        agent_id="normalizer",
        author_type="llm_normalization",
        frame_id=df._frame_id,
    )

    captured_messages: list[list[dict]] = []

    async def fake_llm_call(messages):
        captured_messages.append(messages)
        return [{"cell_id": "ignored", "value": "x", "confidence": 0.9}]

    async def fake_batch_enrich(items):
        return {"written": len(items), "skipped": 0}

    monkeypatch.setattr(writer, "batch_enrich", fake_batch_enrich)

    result = asyncio.run(writer.stream_normalize("city", fake_llm_call, chunk_size=1))

    assert result["total"] == 2
    # Check that default instruction is in the context message
    assert len(captured_messages) == 2
    for messages in captured_messages:
        # System message should contain "Normalize column 'city'"
        context_msg = [m for m in messages if m["role"] == "system" and "Normalize column" in m["content"]]
        assert len(context_msg) > 0, f"Expected 'Normalize column 'city'' in context, got {messages}"


def test_stream_normalize_uses_custom_instruction(monkeypatch):
    df = DFrame({"city": ["jakarta", "bandung"]})
    writer = AgentWriter(
        coordinator=df._coordinator,
        agent_id="normalizer",
        author_type="llm_normalization",
        frame_id=df._frame_id,
    )

    custom_instruction = "Standardize city names to official province names in English."
    captured_messages: list[list[dict]] = []

    async def fake_llm_call(messages):
        captured_messages.append(messages)
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

    # Check that custom instruction is included in context message
    assert len(captured_messages) == 2
    for messages in captured_messages:
        context_msg = [m for m in messages if m["role"] == "system" and custom_instruction in m["content"]]
        assert len(context_msg) > 0, f"Expected custom instruction in context, got {messages}"


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


def test_stream_normalize_logs_llm_debug(monkeypatch, caplog):
    df = DFrame({"city": ["jakarta"]})
    writer = AgentWriter(
        coordinator=df._coordinator,
        agent_id="normalizer",
        author_type="llm_normalization",
        frame_id=df._frame_id,
    )

    async def fake_llm_call(messages):
        return [{
            "cell_id": f"{df._frame_id}::0::city",
            "value": "Jakarta",
            "confidence": 0.95,
        }]

    async def fake_batch_enrich(items):
        return {"written": len(items), "skipped": 0}

    monkeypatch.setattr(writer, "batch_enrich", fake_batch_enrich)
    caplog.set_level(logging.DEBUG, logger="hiveframe.agent.writer")

    asyncio.run(writer.stream_normalize("city", fake_llm_call, chunk_size=1))

    assert "stream_normalize LLM_CALL" in caplog.text
    assert "Normalize column 'city'" in caplog.text
    assert "stream_normalize LLM_RESPONSE" in caplog.text
    assert "Jakarta" in caplog.text


def test_normalize_low_confidence_is_skipped() -> None:
    df = DFrame({"city": ["jakarta"]})
    writer = AgentWriter(
        coordinator=df._coordinator,
        agent_id="normalizer",
        author_type="llm_normalization",
        frame_id=df._frame_id,
    )

    result = asyncio.run(
        writer.normalize(
            f"{df._frame_id}::city_0",
            "DKI Jakarta",
            confidence=0.30,
        )
    )

    assert result["written"] == 0
    assert result["skipped"] == 1
    assert df.read_fresh().at[0, "city"] == "jakarta"


def test_batch_enrich_mixed_confidence_writes_only_above_threshold() -> None:
    df = DFrame({"city": ["jakarta", "bandung"]})
    writer = AgentWriter(
        coordinator=df._coordinator,
        agent_id="normalizer",
        author_type="llm_normalization",
        frame_id=df._frame_id,
    )

    result = asyncio.run(
        writer.batch_enrich([
            {
                "cell_id": f"{df._frame_id}::city_0",
                "value": "DKI Jakarta",
                "confidence": 0.90,
            },
            {
                "cell_id": f"{df._frame_id}::city_1",
                "value": "Jawa Barat",
                "confidence": 0.10,
            },
        ])
    )

    fresh = df.read_fresh()
    assert result["written"] == 1
    assert result["skipped"] == 1
    assert fresh.at[0, "city"] == "DKI Jakarta"
    assert fresh.at[1, "city"] == "bandung"


def test_stream_normalize_invalid_chunk_size_raises() -> None:
    df = DFrame({"city": ["jakarta"]})
    writer = AgentWriter(
        coordinator=df._coordinator,
        agent_id="normalizer",
        author_type="llm_normalization",
        frame_id=df._frame_id,
    )

    async def fake_llm_call(_messages):
        return []

    with pytest.raises(ValueError, match="chunk_size must be > 0"):
        asyncio.run(writer.stream_normalize("city", fake_llm_call, chunk_size=0))


def test_stream_normalize_uses_only_target_frame_columns(monkeypatch) -> None:
    coordinator = TransactionCoordinator()
    target = DFrame({"city": ["jakarta", "bandung"]}, coordinator=coordinator)
    _other = DFrame({"country": ["ID", "ID"]}, coordinator=coordinator)

    writer = AgentWriter(
        coordinator=coordinator,
        agent_id="normalizer",
        author_type="llm_normalization",
        frame_id=target._frame_id,
    )

    captured_messages: list[list[dict]] = []

    async def fake_llm_call(messages):
        captured_messages.append(messages)
        return []

    async def fake_batch_enrich(items):
        return {"written": 0, "skipped": len(items)}

    monkeypatch.setattr(writer, "batch_enrich", fake_batch_enrich)

    result = asyncio.run(writer.stream_normalize("city", fake_llm_call, chunk_size=2))

    assert result["total"] == 2
    assert captured_messages
    blob = "\n".join(msg["content"] for msg in captured_messages[0] if isinstance(msg, dict))
    assert "country" not in blob


