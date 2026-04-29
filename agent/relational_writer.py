# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Relational writer for cross-frame LLM normalization."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from ._llm_debug import summarize_messages, summarize_operations, truncate_text

if TYPE_CHECKING:
    from core.dataframe import DFrame

logger = logging.getLogger("hiveframe.agent.relational")


@dataclass
class FrameRelation:
    """Define a relation used for cross-frame lookup."""

    from_column: str
    to_column: str
    context_frame: str
    include_columns: list[str] = field(default_factory=list)
    relation_type: str = "many_to_one"
    context_label: str = ""
    max_related: int = 10

    def __post_init__(self) -> None:
        if self.relation_type not in ("many_to_one", "one_to_many"):
            raise ValueError(
                "relation_type must be 'many_to_one' or 'one_to_many', "
                f"got '{self.relation_type}'"
            )
        if not self.context_label:
            self.context_label = self.context_frame


class RelationalAgentWriter:
    """Agent writer that enriches each target row with related frame context."""

    def __init__(
        self,
        target_frame: "DFrame",
        relations: list[FrameRelation],
        context_frames: dict[str, "DFrame"] | None = None,
        agent_id: str = "relational_agent",
        confidence_threshold: float = 0.6,
        author_type: str = "llm_normalization",
        llm_timeout_seconds: float | None = 45.0,
    ) -> None:
        self._target = target_frame
        self._context_frames = context_frames or {}
        self._relations = relations
        self._agent_id = agent_id
        self._confidence_threshold = confidence_threshold
        self._author_type = author_type
        self._llm_timeout_seconds = llm_timeout_seconds

        self._cache: dict[str, dict[Any, list[dict[str, Any]]]] = {}
        self._cache_built = False

        self._validate_relations()

    def _validate_relations(self) -> None:
        """Validate relation definitions before processing starts."""
        for rel in self._relations:
            if rel.context_frame == "self":
                continue
            if rel.context_frame not in self._context_frames:
                raise ValueError(
                    f"FrameRelation context_frame='{rel.context_frame}' not found in context_frames. "
                    f"Available: {list(self._context_frames.keys())}"
                )

    def _build_cache(self) -> None:
        """Build in-memory lookup indexes once for all relations."""
        if self._cache_built:
            return

        for rel in self._relations:
            if rel.context_frame == "self":
                df = self._target.read_fresh()
                cache_key = f"self:{rel.to_column}"
            else:
                df = self._context_frames[rel.context_frame].read_fresh()
                cache_key = f"{rel.context_frame}:{rel.to_column}"

            if cache_key in self._cache:
                continue

            if rel.to_column not in df.columns:
                logger.warning(
                    "_build_cache: to_column '%s' not found in frame '%s', skipping",
                    rel.to_column,
                    rel.context_frame,
                )
                self._cache[cache_key] = {}
                continue

            index: dict[Any, list[dict[str, Any]]] = {}
            grouped = df.groupby(rel.to_column, dropna=False, sort=False)
            for key, group in grouped:
                index[key] = group.to_dict(orient="records")

            self._cache[cache_key] = index
            logger.info(
                "_build_cache: indexed %d unique values for '%s' in frame '%s'",
                len(index),
                rel.to_column,
                rel.context_frame,
            )

        self._cache_built = True

    def _lookup(self, rel: FrameRelation, fk_value: Any) -> list[dict[str, Any]]:
        """Lookup related rows from cache for a single FK value."""
        if rel.context_frame == "self":
            cache_key = f"self:{rel.to_column}"
        else:
            cache_key = f"{rel.context_frame}:{rel.to_column}"

        rows = self._cache.get(cache_key, {}).get(fk_value, [])

        if rel.include_columns and rows:
            rows = [{k: v for k, v in row.items() if k in rel.include_columns} for row in rows]

        if rel.relation_type == "one_to_many":
            return rows[: rel.max_related]
        return rows[:1]

    def _build_enriched_context(self, chunk_rows: list[tuple[int, dict[str, Any]]]) -> str:
        """Build a text block that includes target row data and related context."""
        parts: list[str] = []

        for row_index, row in chunk_rows:
            id_cols = [
                f"{k}={v}" for k, v in list(row.items())[:3] if not str(k).startswith("_")
            ]
            parts.append(f"\nRow {row_index} ({', '.join(id_cols)}):")
            parts.append("  Data:")
            for key, value in row.items():
                if not str(key).startswith("_context_"):
                    parts.append(f"    {key}: {value}")

            for rel in self._relations:
                fk_value = row.get(rel.from_column)
                if fk_value is None:
                    continue

                related = self._lookup(rel, fk_value)
                if not related:
                    parts.append(f"  [{rel.context_label}]: (not found)")
                    continue

                if rel.relation_type == "many_to_one":
                    parts.append(f"  [{rel.context_label}] (lookup {rel.from_column}={fk_value}):")
                    for key, value in related[0].items():
                        parts.append(f"    {key}: {value}")
                else:
                    parts.append(
                        f"  [{rel.context_label}] ({len(related)} found, max {rel.max_related} shown):"
                    )
                    for idx, related_row in enumerate(related, start=1):
                        row_str = ", ".join(f"{k}: {v}" for k, v in related_row.items())
                        parts.append(f"    {idx}. {row_str}")

        return "\n".join(parts)

    def _build_messages(
        self,
        enriched_context: str,
        target_column: str,
        instruction: str,
        frame_id: str,
        start_idx: int,
        chunk_size: int,
    ) -> list[dict[str, str]]:
        """Build LLM messages for relational normalization."""
        system = (
            "You are a data annotation agent. "
            "For each row, analyze the data AND its related context, "
            "then determine the appropriate value for the target column.\n\n"
            f"## Target column\n{target_column}\n\n"
            f"## Instruction\n{instruction}\n\n"
            "## Cell ID convention\n"
            "Format: {frame_id}::{column}_{row_index}\n"
            f"frame_id = {frame_id}\n"
            f"Chunk rows start at absolute index {start_idx} with {chunk_size} rows\n"
            f"Example: {frame_id}::{target_column}_{start_idx}\n\n"
            "## Confidence scoring\n"
            "0.95-1.00: very certain\n"
            "0.80-0.94: confident\n"
            "0.60-0.79: moderate\n"
            f"< {self._confidence_threshold}: skip - do not write\n\n"
            "## Rules\n"
            "- Base your decision on BOTH the row data AND its related context\n"
            "- Never invent information not present in the data\n"
            "- If context is missing (not found), use available data and lower confidence\n\n"
            "## Response format\n"
            "Raw JSON only, no markdown:\n"
            '{"action":"batch_enrich","reasoning":"...",'
            '"operations":[{'
            f'"cell_id":"{frame_id}::{target_column}_N",'
            '"value":"...","confidence":0.0}]}'
        )

        return [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"Annotate {target_column} for each row based on data and context below:\n\n"
                    f"{enriched_context}"
                ),
            },
        ]

    def _make_llm_caller(
        self,
        provider: str,
        model: str | None,
        anthropic_api_key: str | None,
        openai_api_key: str | None,
    ) -> Callable[[list[dict[str, str]]], Awaitable[str]]:
        """Return async callable: messages -> raw LLM response string."""
        if provider == "anthropic":

            async def call_anthropic(messages: list[dict[str, str]]) -> str:
                try:
                    import anthropic
                except ImportError as exc:
                    raise ImportError("anthropic required: pip install anthropic") from exc

                client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
                chosen_model = model or "claude-sonnet-4-20250514"
                system = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
                user_messages = [msg for msg in messages if msg["role"] != "system"]
                logger.debug(
                    "relational LLM_REQUEST: provider=%s model=%s messages=%d preview=%s",
                    provider,
                    chosen_model,
                    len(messages),
                    summarize_messages(messages),
                )
                response = await client.messages.create(
                    model=chosen_model,
                    max_tokens=1000,
                    system=system,
                    messages=user_messages,
                )
                raw = response.content[0].text
                logger.debug(
                    "relational LLM_RESPONSE: provider=%s model=%s chars=%d preview=%s",
                    provider,
                    chosen_model,
                    len(raw),
                    truncate_text(raw),
                )
                return raw

            return call_anthropic

        if provider == "openai":

            async def call_openai(messages: list[dict[str, str]]) -> str:
                try:
                    from openai import AsyncOpenAI
                except ImportError as exc:
                    raise ImportError("openai required: pip install openai") from exc

                client = AsyncOpenAI(api_key=openai_api_key)
                chosen_model = model or "gpt-4o"
                new_param = ("o1", "o3", "o4", "gpt-4.1")
                kwargs: dict[str, Any] = {"model": chosen_model, "messages": messages}
                if any(chosen_model.startswith(prefix) for prefix in new_param):
                    kwargs["max_completion_tokens"] = 1000
                else:
                    kwargs["max_tokens"] = 1000
                logger.debug(
                    "relational LLM_REQUEST: provider=%s model=%s messages=%d preview=%s",
                    provider,
                    chosen_model,
                    len(messages),
                    summarize_messages(messages),
                )
                response = await client.chat.completions.create(**kwargs)
                raw = response.choices[0].message.content or ""
                logger.debug(
                    "relational LLM_RESPONSE: provider=%s model=%s chars=%d preview=%s",
                    provider,
                    chosen_model,
                    len(raw),
                    truncate_text(raw),
                )
                return raw

            return call_openai

        raise ValueError(f"Unknown provider: {provider}. Use 'anthropic' or 'openai'.")

    async def stream_normalize_relational(
        self,
        target_column: str,
        instruction: str,
        chunk_size: int = 10,
        provider: str = "anthropic",
        model: str | None = None,
        anthropic_api_key: str | None = None,
        openai_api_key: str | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        llm_timeout_seconds: float | None = None,
    ) -> dict[str, int]:
        """Normalize target_column using cross-frame context per chunk."""
        from .prompt import parse_plan
        from .writer import AgentWriter

        self._build_cache()

        writer = AgentWriter(
            coordinator=self._target._coordinator,
            agent_id=self._agent_id,
            author_type=self._author_type,
            frame_id=self._target._frame_id,
        )
        writer.CONFIDENCE_THRESHOLD = self._confidence_threshold

        llm_call = self._make_llm_caller(
            provider=provider,
            model=model,
            anthropic_api_key=anthropic_api_key,
            openai_api_key=openai_api_key,
        )

        fresh = self._target.read_fresh()
        total = len(fresh)
        total_written = 0
        total_skipped = 0

        if target_column not in fresh.columns:
            logger.warning("target_column '%s' not found in target frame, will be created", target_column)

        for i in range(0, total, chunk_size):
            chunk = fresh.iloc[i : i + chunk_size]
            chunk_rows = [(i + j, row.to_dict()) for j, (_, row) in enumerate(chunk.iterrows())]

            enriched_context = self._build_enriched_context(chunk_rows)
            messages = self._build_messages(
                enriched_context=enriched_context,
                target_column=target_column,
                instruction=instruction,
                frame_id=self._target._frame_id,
                start_idx=i,
                chunk_size=len(chunk),
            )

            try:
                timeout = self._llm_timeout_seconds if llm_timeout_seconds is None else llm_timeout_seconds
                if timeout is None:
                    raw = await llm_call(messages)
                else:
                    raw = await asyncio.wait_for(llm_call(messages), timeout=timeout)
                plan = parse_plan(raw)
                operations = list(plan.get("operations", []))
                logger.debug(
                    "relational LLM_PARSED: chunk=%d-%d operations=%d preview=%s",
                    i,
                    i + len(chunk) - 1,
                    len(operations),
                    summarize_operations(operations),
                )
            except Exception as exc:
                logger.warning("LLM call failed for chunk %d-%d: %s", i, i + len(chunk) - 1, exc)
                total_skipped += len(chunk)
                if on_progress:
                    on_progress(min(i + len(chunk), total), total)
                continue

            if operations:
                result = await writer.batch_enrich(operations)
                written = int(result.get("written", 0))
                skipped = int(result.get("skipped", 0))
            else:
                written = 0
                skipped = len(chunk)
                logger.warning("No operations from LLM for chunk %d-%d", i, i + len(chunk) - 1)

            total_written += written
            total_skipped += skipped

            logger.info("chunk %d-%d: written=%d skipped=%d", i, i + len(chunk) - 1, written, skipped)

            if on_progress:
                on_progress(min(i + len(chunk), total), total)

        return {"written": total_written, "skipped": total_skipped, "total": total}

