# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Async LLM agent writer interface with retry/backoff."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Awaitable, Optional

from core.coordinator import TransactionCoordinator
from core.transaction import Operation, TxState


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

    def _validate_cell_id(self, cell_id: str) -> bool:
        if self._frame_id is None:
            return True
        return cell_id.startswith(f"{self._frame_id}::")

    def _build_operations(self, items: list[dict[str, Any]]) -> tuple[list[Operation], list[str]]:
        cell_ids = [item["cell_id"] for item in items]
        old_values = self._coordinator.read_fresh(cell_ids)
        ops = []
        skipped = []
        for item in items:
            confidence = item.get("confidence")
            if confidence is None:
                raise ValueError("confidence must be provided for LLM writes")
            cell_id = item["cell_id"]
            if confidence < self.CONFIDENCE_THRESHOLD:
                skipped.append(cell_id)
                continue
            if not self._validate_cell_id(cell_id):
                skipped.append(cell_id)
                continue
            value = item["value"] if "value" in item else item.get("new_value")
            ops.append(Operation(
                cell_id=cell_id,
                old_value=old_values.get(cell_id),
                new_value=value,
                author_type=self._author_type,
                author_id=self._agent_id,
                confidence=confidence,
            ))
        if skipped:
            import logging
            logger = logging.getLogger("hiveframe.agent.writer")
            logger.warning(
                "AgentWriter skipped %d cells: %s", len(skipped), skipped
            )
        return ops, skipped

    async def normalize(self, cell_id: str, new_value: Any, confidence: float) -> dict:
        """Write one normalization result. Returns result summary."""
        return await self._submit_with_retry([
            {
                "cell_id": cell_id,
                "value": new_value,
                "confidence": confidence,
            }
        ])

    async def batch_enrich(self, items: list[dict[str, Any]]):
        """Write multiple enrichment values as one transaction. Returns result summary."""
        return await self._submit_with_retry(items)

    async def _submit_with_retry(self, items: list[dict[str, Any]]):
        # Build ops and skipped once outside retry loop
        ops, skipped = self._build_operations(items)
        if not ops:
            return {
                "written": 0,
                "skipped": len(skipped),
                "skipped_cells": skipped,
                "tx_id": None,
                "success": False,
            }
        delay = 0.05
        for attempt in range(self._max_retries):
            tx = await asyncio.to_thread(self._coordinator.submit, ops)
            if tx.state == TxState.SYNCED:
                return {
                    "written": len(ops),
                    "skipped": len(skipped),
                    "skipped_cells": skipped,
                    "tx_id": getattr(tx, "tx_id", None),
                    "success": True,
                }
            error = tx.error or "transaction failed"
            # Retry only lock conflicts; fail fast for other write errors.
            if "Lock conflict" not in error:
                raise RuntimeError(f"Agent write failed: {error}")

            if attempt == self._max_retries - 1:
                raise RuntimeError("Agent write failed after retries: lock conflict")
            await asyncio.sleep(delay)
            delay *= 2

    def _build_operation(self, item: dict[str, Any]) -> Operation:
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

        Args:
            column: Public column name to normalize.
            llm_call: Async function(messages) -> list of write operations.
            chunk_size: Number of rows sent per LLM call.
            progress_callback: Optional callback(processed, total).
            custom_instruction: Optional custom instruction passed to
                `build_messages`. If None, uses
                ``Normalize column '{column}'``.
        """
        df = self._coordinator.write_node._df
        # Find the internal namespaced column from the public column name.
        col_candidates = [c for c in df.columns if c.endswith(f"::{column}") or c == column]
        if not col_candidates:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        col_name = col_candidates[0]
        total = len(df)
        written = 0
        instruction = custom_instruction or f"Normalize column '{column}'"
        for start in range(0, total, chunk_size):
            chunk = df.iloc[start:start + chunk_size]
            # Build a snapshot for this chunk only.
            snapshot = chunk[[col_name]].to_string()
            from .prompt import build_messages
            messages = build_messages(
                user_instruction=instruction,
                dataframe_snapshot=snapshot,
                frame_id=self._frame_id,
            )
            # Call LLM for this chunk.
            ops = await llm_call(messages)
            # ops: list[dict] with cell_id, value, confidence
            result = await self.batch_enrich(ops)
            written += result.get("written", 0)
            if progress_callback:
                progress_callback(min(start + len(chunk), total), total)
        return {"written": written, "total": total}
