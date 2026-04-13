# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Async LLM agent writer interface with retry/backoff."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Awaitable, Optional

from core.coordinator import TransactionCoordinator
from core.transaction import Operation, TxState
from ._llm_debug import summarize_messages, summarize_operations


logger = logging.getLogger("hiveframe.agent.writer")


class AgentWriter:
    """Async helper for LLM-originated writes through coordinator."""

    CONFIDENCE_THRESHOLD = 0.60

    def __init__(
        self,
        coordinator: TransactionCoordinator,
        agent_id: str,
        author_type: str = "llm_agent",
        max_retries: int = 3,
        frame_id: str | None = None,
    ) -> None:
        if author_type not in {"llm_agent", "llm_normalization"}:
            raise ValueError("author_type must be 'llm_agent' or 'llm_normalization'")
        self._coordinator = coordinator
        self._agent_id = agent_id
        self._author_type = author_type
        self._max_retries = max_retries
        self._frame_id = frame_id
        logger.debug(
            "AgentWriter initialized agent_id=%s author_type=%s frame_id=%s max_retries=%d",
            agent_id,
            author_type,
            frame_id,
            max_retries,
        )

    def _validate_cell_id(self, cell_id: str) -> bool:
        if self._frame_id is None:
            return True
        return cell_id.startswith(f"{self._frame_id}::")

    def _build_operations(self, items: list[dict[str, Any]]) -> tuple[list[Operation], list[str]]:
        logger.debug("_build_operations STARTED: items=%d", len(items))
        cell_ids = [item["cell_id"] for item in items]

        logger.debug(
            "_build_operations READING: cells=%d cell_ids=%s",
            len(cell_ids),
            cell_ids[:3] + (["..."] if len(cell_ids) > 3 else []),
        )

        old_values = self._coordinator.read_fresh(cell_ids)

        logger.debug(
            "_build_operations READ_COMPLETE: values_read=%d",
            len(old_values),
        )

        ops = []
        skipped = []
        for idx, item in enumerate(items):
            confidence: float = item.get("confidence")  # type: ignore
            if confidence is None:
                raise ValueError("confidence must be provided for LLM writes")
            cell_id = item["cell_id"]

            if confidence < self.CONFIDENCE_THRESHOLD:
                skipped.append(cell_id)
                logger.debug(
                    "_build_operations SKIP: idx=%d cell_id=%s confidence=%.4f (below threshold %.2f)",
                    idx,
                    cell_id,
                    confidence,
                    self.CONFIDENCE_THRESHOLD,
                )
                continue

            if not self._validate_cell_id(cell_id):
                skipped.append(cell_id)
                logger.debug(
                    "_build_operations SKIP: idx=%d cell_id=%s (frame mismatch, expected prefix=%s::)",
                    idx,
                    cell_id,
                    self._frame_id,
                )
                continue

            value = item["value"] if "value" in item else item.get("new_value")
            old_value = old_values.get(cell_id)

            logger.debug(
                "_build_operations ACCEPT: idx=%d cell_id=%s old=%r new=%r confidence=%.4f",
                idx,
                cell_id,
                old_value,
                value,
                confidence,
            )

            ops.append(Operation(
                cell_id=cell_id,
                old_value=old_value,
                new_value=value,
                author_type=self._author_type,
                author_id=self._agent_id,
                confidence=confidence,
            ))

        if skipped:
            logger.warning(
                "_build_operations SUMMARY: skipped %d cells: %s",
                len(skipped),
                skipped[:5] + (["..."] if len(skipped) > 5 else []),
            )

        logger.debug(
            "_build_operations SUMMARY: total=%d accepted=%d skipped=%d",
            len(items),
            len(ops),
            len(skipped),
        )

        return ops, skipped

    async def normalize(self, cell_id: str, new_value: Any, confidence: float) -> dict:
        """Write one normalization result. Returns result summary."""
        logger.debug(
            "normalize SINGLE: cell_id=%s confidence=%.4f value=%r",
            cell_id,
            confidence,
            new_value,
        )
        result = await self._submit_with_retry([
            {
                "cell_id": cell_id,
                "value": new_value,
                "confidence": confidence,
            }
        ])
        return result or {}

    async def batch_enrich(self, items: list[dict[str, Any]]):
        """Write multiple enrichment values as one transaction. Returns result summary."""
        logger.debug("batch_enrich STARTED: items=%d", len(items))
        logger.debug(
            "batch_enrich DETAILS: items=%s",
            [f"{item.get('cell_id')}={item.get('value')}" for item in items[:3]]
            + (["..."] if len(items) > 3 else []),
        )
        return await self._submit_with_retry(items)

    async def _submit_with_retry(self, items: list[dict[str, Any]]):
        # Build ops and skipped once outside retry loop
        logger.debug("_submit_with_retry START: items=%d agent_id=%s author_type=%s", len(items), self._agent_id, self._author_type)
        ops, skipped = self._build_operations(items)
        if not ops:
            logger.debug(
                "_submit_with_retry NO_OPS: no operations to submit, skipped=%d all_below_threshold",
                len(skipped),
            )
            return {
                "written": 0,
                "skipped": len(skipped),
                "skipped_cells": skipped,
                "tx_id": None,
                "success": False,
            }
        delay = 0.05
        for attempt in range(self._max_retries):
            logger.debug(
                "_submit_with_retry SUBMIT: attempt=%d/%d ops=%d",
                attempt + 1,
                self._max_retries,
                len(ops),
            )
            tx = await asyncio.to_thread(self._coordinator.submit, ops)
            if tx.state == TxState.SYNCED:
                logger.info(
                    "_submit_with_retry SUCCESS: attempt=%d tx_id=%s written=%d skipped=%d state=%s",
                    attempt + 1,
                    getattr(tx, "tx_id", None),
                    len(ops),
                    len(skipped),
                    tx.state,
                )
                return {
                    "written": len(ops),
                    "skipped": len(skipped),
                    "skipped_cells": skipped,
                    "tx_id": getattr(tx, "tx_id", None),
                    "success": True,
                }
            error = tx.error or "transaction failed"
            logger.debug(
                "_submit_with_retry FAILED: attempt=%d/%d state=%s error=%s",
                attempt + 1,
                self._max_retries,
                tx.state,
                error,
            )
            # Retry only lock conflicts; fail fast for other write errors.
            if "Lock conflict" not in error:
                logger.error(
                    "_submit_with_retry NON_RETRYABLE: attempt=%d state=%s error=%s",
                    attempt + 1,
                    tx.state,
                    error,
                )
                raise RuntimeError(f"Agent write failed: {error}")

            if attempt == self._max_retries - 1:
                logger.error("_submit_with_retry RETRIES_EXHAUSTED: lock conflict persisted after %d attempts", self._max_retries)
                raise RuntimeError("Agent write failed after retries: lock conflict")
            logger.debug(
                "_submit_with_retry RETRY: attempt=%d lock conflict, backoff=%.2fs",
                attempt + 1,
                delay,
            )
            await asyncio.sleep(delay)
            delay *= 2

    def _build_operation(self, item: dict[str, Any]) -> Operation | None:
        import warnings
        warnings.warn(
            "_build_operation is deprecated, use _build_operations",
            DeprecationWarning,
            stacklevel=2,
        )
        ops, _ = self._build_operations([item])
        return ops[0] if ops else None

    async def stream_normalize(
        self,
        column: str,
        llm_call: Callable[[list[dict]], Awaitable[list[dict]]],
        chunk_size: int = 50,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        custom_instruction: str | None = None,
    ) -> dict:
        """
        Normalize one column in streaming mode using chunked LLM calls.

        Uses context-aware prompts that include surrounding columns to inform normalization.

        Args:
            column: Public column name to normalize.
            llm_call: Async function(messages) -> list of write operations.
            chunk_size: Number of rows sent per LLM call.
            progress_callback: Optional callback(processed, total).
            custom_instruction: Optional custom instruction for normalization rule.
                                If None, uses generic "Normalize column '{column}'".
                                Example: "Standardize city names to proper Indonesian format"

        Returns:
            {"written": int, "total": int}

        Example::

            async def my_llm_call(messages):
                # Your LLM call here
                response = await openai.ChatCompletion.acreate(
                    model="gpt-4",
                    messages=messages,
                )
                return parse_plan(response)["operations"]

            result = await writer.stream_normalize(
                column="city",
                llm_call=my_llm_call,
                chunk_size=50,
                custom_instruction="Standardize to proper Indonesian city names",
            )
        """
        df = self._coordinator.write_node._df
        # Find the internal namespaced column from the public column name.
        col_candidates = [c for c in df.columns if c.endswith(f"::{column}") or c == column]
        if not col_candidates:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        col_name = col_candidates[0]
        total = len(df)
        written = 0
        skipped = 0
        instruction = custom_instruction or f"Normalize column '{column}'"

        logger.info(
            "stream_normalize STARTED: column=%s total_rows=%d chunk_size=%d frame_id=%s instruction=%r",
            column,
            total,
            chunk_size,
            self._frame_id,
            instruction,
        )

        for chunk_idx, start in enumerate(range(0, total, chunk_size)):
            end = min(start + chunk_size, total)
            chunk = df.iloc[start:end]

            logger.debug(
                "stream_normalize CHUNK %d: rows %d-%d (size=%d)",
                chunk_idx + 1,
                start,
                end - 1,
                len(chunk),
            )

            # Build snapshot with target column + context columns
            # Get all columns for context
            all_cols = list(chunk.columns)
            snapshot = chunk[all_cols].to_string()

            logger.debug(
                "stream_normalize CONTEXT: chunk %d snapshot_lines=%d columns=%s",
                chunk_idx + 1,
                snapshot.count("\n"),
                ", ".join(f"'{c}'" for c in all_cols[:3]) + ("..." if len(all_cols) > 3 else ""),
            )

            # Use new context-aware prompt builder
            from .prompt import build_normalize_messages
            messages = build_normalize_messages(
                instruction=instruction,
                chunk_snapshot=snapshot,
                frame_id=self._frame_id or "unknown",
                column_name=column,
                chunk_start=start,
                context_columns=all_cols,
            )

            logger.debug(
                "stream_normalize LLM_CALL: chunk %d messages=%d preview=%s",
                chunk_idx + 1,
                len(messages),
                summarize_messages(messages),
            )

            # Call LLM for this chunk
            try:
                ops = await llm_call(messages)
                logger.debug(
                    "stream_normalize LLM_RESPONSE: chunk %d operations=%d preview=%s",
                    chunk_idx + 1,
                    len(ops) if ops else 0,
                    summarize_operations(ops or []),
                )
            except Exception as exc:
                logger.error(
                    "stream_normalize LLM_ERROR: chunk %d error=%s",
                    chunk_idx + 1,
                    str(exc),
                    exc_info=True,
                )
                skipped += len(chunk)
                continue

            # Write batch to frame
            if ops:
                logger.debug(
                    "stream_normalize BATCH_ENRICH: chunk %d submitting operations=%d",
                    chunk_idx + 1,
                    len(ops),
                )
                result = await self.batch_enrich(ops)
                chunk_written = result.get("written", 0)
                chunk_skipped = result.get("skipped", 0)

                logger.info(
                    "stream_normalize BATCH_RESULT: chunk %d written=%d skipped=%d",
                    chunk_idx + 1,
                    chunk_written,
                    chunk_skipped,
                )

                written += chunk_written
                skipped += chunk_skipped
            else:
                logger.warning(
                    "stream_normalize NO_OPS: chunk %d (no operations from LLM)",
                    chunk_idx + 1,
                )
                skipped += len(chunk)

            if progress_callback:
                progress_callback(end, total)

        logger.info(
            "stream_normalize FINISHED: total=%d written=%d skipped=%d success_rate=%.1f%%",
            total,
            written,
            skipped,
            (written / total * 100) if total > 0 else 0,
        )

        return {"written": written, "skipped": skipped, "total": total}
