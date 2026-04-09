# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""MultiFrameAgent untuk analisis satu atau banyak DFrame."""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Any

from ._llm_debug import summarize_messages, truncate_text
from .prompt import parse_plan
from .result import FrameInsight, MultiFrameResult, ReviewVerdict, SeriesSpec

if TYPE_CHECKING:
    import pandas as pd

    from core.dataframe import DFrame


logger = logging.getLogger("hiveframe.agent.multi")

_ALLOWED_PANDAS_METHODS = frozenset(
    {
        "groupby",
        "filter",
        "query",
        "nlargest",
        "nsmallest",
        "sort_values",
        "sort_index",
        "head",
        "tail",
        "sample",
        "describe",
        "value_counts",
        "nunique",
        "count",
        "sum",
        "mean",
        "median",
        "std",
        "min",
        "max",
        "merge",
        "join",
        "concat",
        "pivot",
        "pivot_table",
        "melt",
        "stack",
        "unstack",
        "reset_index",
        "set_index",
        "rename",
        "drop",
        "dropna",
        "fillna",
        "astype",
        "apply",
        "map",
        "agg",
        "transform",
        "loc",
        "iloc",
        "at",
        "iat",
    }
)

_FORBIDDEN_PATTERNS = (
    "import ",
    "exec(",
    "eval(",
    "open(",
    "os.",
    "sys.",
    "subprocess",
    "__import__",
    "globals(",
    "locals(",
    "vars(",
    "getattr(",
    "setattr(",
    "delattr(",
)


class MultiFrameAgent:
    """LLM agent for analyzing one or multiple DFrame objects."""

    def __init__(
        self,
        frames: dict[str, "DFrame"],
        agent_id: str = "multi_frame_agent",
        provider: str = "anthropic",
        model: str | None = None,
        anthropic_api_key: str | None = None,
        openai_api_key: str | None = None,
    ) -> None:
        if not frames:
            raise ValueError("frames dict cannot be empty")
        self._frames = frames
        self._agent_id = agent_id
        self._provider = provider
        self._model = model
        self._anthropic_api_key = anthropic_api_key
        self._openai_api_key = openai_api_key
        self._columns_hint: dict[str, list[str]] | None = None

        logger.info(
            "MultiFrameAgent created frames=%s provider=%s",
            list(frames.keys()),
            provider,
        )

    async def analyze(
        self,
        instruction: str,
        mode: str = "sample",
        max_sample_rows: int = 5,
        max_result_rows: int = 200,
        output_frame: "DFrame | None" = None,
        max_retries: int = 0,
        columns_hint: dict[str, list[str]] | None = None,
    ) -> MultiFrameResult:
        """Run LLM analysis over one or multiple frames.

        Args:
            instruction: Natural language instruction from the user.
            mode: "sample" or "query".
            max_sample_rows: Number of sample rows per frame in sample mode.
            max_result_rows: Maximum query-result rows sent to the LLM.
            output_frame: Optional destination frame for write operations.
            max_retries: Number of retries for iterative query mode.
            columns_hint: Dict label -> list of relevant columns. When set,
                sample/query context prioritizes these columns to reduce
                payload size. When None, behavior remains unchanged.
        """
        if mode not in ("sample", "query"):
            raise ValueError("mode must be 'sample' or 'query'")

        # Reset per call to avoid state leaks across analyze() invocations.
        self._columns_hint = columns_hint

        if mode == "query":
            # Always use iterative path; guarantee at least one correction pass.
            effective_retries = max(1, max_retries)
            result = await self._analyze_query_mode_iterative(
                instruction,
                max_result_rows,
                effective_retries,
            )
        else:
            result = await self._analyze_sample_mode(instruction, max_sample_rows)

        result.mode = mode

        if output_frame is not None and result.operations:
            result.write_result = await self._write_to_frame(result.operations, output_frame)

        return result

    async def _analyze_sample_mode(self, instruction: str, max_rows: int) -> MultiFrameResult:
        from .prompt import build_multi_frame_messages

        contexts: dict[str, str] = {}
        for label, frame in self._frames.items():
            hint_cols = self._columns_hint.get(label) if self._columns_hint is not None else None

            if hint_cols is not None:
                contexts[label] = self._build_context_with_hint(
                    label=label,
                    frame=frame,
                    columns=hint_cols,
                    max_rows=max_rows,
                )
                logger.info("Using columns_hint for frame '%s': %s", label, hint_cols)
            else:
                contexts[label] = frame.describe_for_agent(
                    max_rows=max_rows,
                    include_schema=True,
                    include_stats=True,
                )

        messages = build_multi_frame_messages(
            instruction=instruction,
            frame_contexts=contexts,
            mode="sample",
        )
        raw = await self._call_llm(messages)
        plan = parse_plan(raw)
        return self._plan_to_result(plan)

    async def _analyze_query_mode(self, instruction: str, max_result_rows: int) -> MultiFrameResult:
        """Backward-compatible alias for direct callers."""
        return await self._analyze_query_mode_simple(instruction, max_result_rows)

    async def _analyze_query_mode_simple(self, instruction: str, max_result_rows: int) -> MultiFrameResult:
        from .prompt import build_analysis_messages, build_query_correction_messages, build_query_generation_messages

        schema_contexts = self._build_schema_context()
        query_gen_messages = build_query_generation_messages(
            instruction=instruction,
            frame_schemas=schema_contexts,
        )

        raw_queries = await self._call_llm(query_gen_messages)
        query_plan = parse_plan(raw_queries)
        pandas_queries = query_plan.get("queries", {})
        logger.debug(
            "_analyze_query_mode_simple PARSED: queries=%d plan_keys=%s",
            len(pandas_queries) if isinstance(pandas_queries, dict) else 0,
            list(query_plan.keys()) if query_plan else "empty",
        )
        if not isinstance(pandas_queries, dict) or not pandas_queries:
            logger.warning("LLM did not generate any queries, falling back to sample mode")
            return await self._analyze_sample_mode(instruction, 50)

        query_results: dict[str, "pd.DataFrame"] = {}
        query_errors: dict[str, str] = {}

        for label, query_str in pandas_queries.items():
            query_str = str(query_str).strip()
            label = str(label).strip()

            # CRITICAL: Validate frame label matches exactly
            frame = self._frames.get(label)
            if frame is None:
                available = list(self._frames.keys())
                query_errors[label] = (
                    f"Frame label '{label}' not found. Available frames: {available}. "
                    f"Frame label MUST match exactly."
                )
                logger.warning(
                    "_analyze_query_mode_simple FRAME_MISMATCH: requested_label=%s available_frames=%s",
                    label,
                    available,
                )
                continue

            # Validate query format early
            if not query_str.startswith("df"):
                query_errors[label] = f"Invalid query format: must start with 'df', got '{query_str[:50]}'"
                logger.debug(
                    "_analyze_query_mode_simple INVALID_QUERY: label=%s query=%s",
                    label,
                    query_str[:80],
                )
                continue
            try:
                fresh = frame.read_fresh()
                result_df = self._safe_eval(str(query_str), fresh)
                query_results[label] = result_df.head(max_result_rows)
                logger.debug(
                    "_analyze_query_mode_simple EXEC_OK: label=%s query=%s rows=%d",
                    label,
                    query_str[:80],
                    len(result_df),
                )
            except Exception as exc:
                query_errors[label] = str(exc)
                logger.debug(
                    "_analyze_query_mode_simple EXEC_ERROR: label=%s query=%s error=%s",
                    label,
                    query_str[:80],
                    str(exc),
                )

        result_contexts = {
            label: result_df.to_string(max_rows=max_result_rows)
            for label, result_df in query_results.items()
        }
        logger.debug(
            "_analyze_query_mode_simple BUILD_ANALYSIS: result_contexts=%d total_context_chars=%d query_errors=%s",
            len(result_contexts),
            sum(len(v) for v in result_contexts.values()),
            query_errors if query_errors else "none",
        )

        if not result_contexts and query_errors:
            error_details = "; ".join(f"{label}: {err}" for label, err in query_errors.items())

            # Detect frame mismatch errors specifically
            frame_mismatch_errors = [e for e in query_errors.values() if "not found" in e]
            if frame_mismatch_errors:
                logger.error(
                    "_analyze_query_mode_simple FRAME_MISMATCH_DETECTED: "
                    "LLM generated query with wrong frame labels. "
                    "Available: %s, Requested: %s. "
                    "See prompt and few-shot examples.",
                    list(self._frames.keys()),
                    list(pandas_queries.keys()),
                )

            logger.warning(
                "_analyze_query_mode_simple NO_RESULTS: all queries failed. "
                "Available frames: %s, Query labels: %s. Errors: %s. "
                "Ensure LLM generates valid pandas expressions starting with 'df'.",
                list(self._frames.keys()),
                list(pandas_queries.keys()),
                error_details,
            )

            # Attempt one self-correction pass before falling back to sample mode
            logger.info(
                "_analyze_query_mode_simple CORRECTION_ATTEMPT: errors=%s",
                query_errors,
            )
            correction_messages = build_query_correction_messages(
                instruction=instruction,
                failed_queries={k: str(v) for k, v in pandas_queries.items()},
                query_errors=query_errors,
                frame_schemas=schema_contexts,
            )
            raw_corrected = await self._call_llm(correction_messages)
            corrected_plan = parse_plan(raw_corrected)
            corrected_queries = corrected_plan.get("queries", {})

            if isinstance(corrected_queries, dict) and corrected_queries:
                for label, query_str in corrected_queries.items():
                    query_str = str(query_str).strip()
                    frame = self._frames.get(label)
                    if frame is None:
                        continue
                    try:
                        fresh = frame.read_fresh()
                        result_df = self._safe_eval(query_str, fresh)
                        query_results[label] = result_df.head(max_result_rows)
                        query_errors.pop(label, None)
                        logger.info(
                            "_analyze_query_mode_simple CORRECTION_OK: label=%s", label
                        )
                    except Exception as exc:
                        query_errors[label] = str(exc)
                        logger.debug(
                            "_analyze_query_mode_simple CORRECTION_FAILED: label=%s error=%s",
                            label, exc,
                        )

                result_contexts = {
                    label: df.to_string(max_rows=max_result_rows)
                    for label, df in query_results.items()
                }

            if not result_contexts:
                logger.info("Falling back to sample mode after correction attempt failed")
                fallback_result = await self._analyze_sample_mode(instruction, 50)
                fallback_result.query_errors = query_errors
                return fallback_result

        analysis_messages = build_analysis_messages(
            instruction=instruction,
            query_results=result_contexts,
            query_errors=query_errors,
            original_queries={k: str(v) for k, v in pandas_queries.items()},
        )

        raw_analysis = await self._call_llm(analysis_messages)
        analysis_plan = parse_plan(raw_analysis)

        result = self._plan_to_result(analysis_plan)
        result.queries_executed = {k: str(v) for k, v in pandas_queries.items()}
        result.query_errors = query_errors
        return result

    async def _analyze_query_mode_iterative(
        self,
        instruction: str,
        max_result_rows: int,
        max_retries: int,
    ) -> MultiFrameResult:
        """Iterative query mode with review+retry loop."""
        from .prompt import (
            build_analysis_messages,
            build_query_generation_messages,
            build_review_messages,
        )

        schema_contexts = self._build_schema_context()

        accumulated_results: dict[str, str] = {}
        accumulated_queries: dict[str, str] = {}
        accumulated_errors: dict[str, str] = {}
        review_history: list[ReviewVerdict] = []
        total_llm_calls = 0
        reflection = ""
        extra_schema: dict[str, str] = {}
        started_at = time.perf_counter()
        skip_labels: set[str] = set()
        only_labels: set[str] | None = None

        for attempt in range(max_retries + 1):
            logger.info(
                "MultiFrameAgent attempt %d/%d instruction=%r",
                attempt + 1,
                max_retries + 1,
                instruction[:60],
            )

            current_schema = {**schema_contexts, **extra_schema}
            query_messages = build_query_generation_messages(
                instruction=instruction,
                frame_schemas=current_schema,
                reflection=reflection,
                iteration=attempt,
            )
            raw_queries = await self._call_llm(query_messages)
            total_llm_calls += 1

            query_plan = parse_plan(raw_queries)
            new_queries = query_plan.get("queries", {})

            if isinstance(new_queries, dict):
                new_queries = {k: str(v) for k, v in new_queries.items()}
                if only_labels is not None:
                    new_queries = {k: v for k, v in new_queries.items() if k in only_labels}
                if skip_labels:
                    new_queries = {k: v for k, v in new_queries.items() if k not in skip_labels}

            if not isinstance(new_queries, dict) or not new_queries:
                logger.warning(
                    "No queries generated at attempt %d, falling back to sample mode",
                    attempt + 1,
                )
                result = await self._analyze_sample_mode(instruction, 50)
                result.total_llm_calls = total_llm_calls + 1
                result.converged = False
                result.final_verdict = "fallback"
                return result

            new_results: dict[str, str] = {}
            new_errors: dict[str, str] = {}
            for label, query_str in new_queries.items():
                frame = self._frames.get(label)
                if frame is None:
                    new_errors[label] = f"Frame '{label}' not found"
                    continue
                try:
                    fresh = frame.read_fresh()
                    result_df = self._safe_eval(str(query_str), fresh)
                    new_results[label] = result_df.head(max_result_rows).to_string()
                    logger.info("Query OK: frame=%s rows=%d", label, len(result_df.head(max_result_rows)))
                except Exception as exc:
                    new_errors[label] = str(exc)
                    logger.warning("Query failed: frame=%s error=%s", label, exc)

            accumulated_queries.update({k: str(v) for k, v in new_queries.items()})
            accumulated_results.update(new_results)
            accumulated_errors.update(new_errors)

            # Pre-review: detect column KeyError and inject schema hint into
            # reflection so next query generation uses correct column casing.
            column_key_errors = {
                label: err for label, err in new_errors.items()
                if "KeyError" in err or (err.startswith("'") and err.endswith("'"))
            }
            if column_key_errors and not reflection:
                schema_ctx = self._build_schema_context()
                col_hints: list[str] = []
                for lbl, schema_str in schema_ctx.items():
                    for line in schema_str.splitlines():
                        stripped = line.strip()
                        if stripped.startswith("  ") or (stripped and ":" in stripped and not stripped.startswith("label") and not stripped.startswith("frame_id") and not stripped.startswith("shape")):
                            col_hints.append(f"  frame `{lbl}`: {stripped}")
                if col_hints:
                    reflection = (
                        "Column name KeyError detected. "
                        "Use EXACT column names as shown in schema (case-sensitive):\n"
                        + "\n".join(col_hints[:30])  # cap to avoid token overflow
                    )
                    logger.info(
                        "_analyze_query_mode_iterative KEYERROR_HINT injected for labels: %s",
                        list(column_key_errors.keys()),
                    )

            review_messages = build_review_messages(
                instruction=instruction,
                queries_executed=accumulated_queries,
                query_results=accumulated_results,
                query_errors=accumulated_errors,
                iteration=attempt,
                previous_verdicts=[{"status": v.status, "reason": v.reason} for v in review_history],
            )
            raw_review = await self._call_llm(review_messages)
            total_llm_calls += 1
            verdict_dict = parse_plan(raw_review)

            # Legacy compatibility: older callers/tests may return final analysis JSON
            # directly in the second call without a review verdict payload.
            if "status" not in verdict_dict and (
                "analysis" in verdict_dict or "action" in verdict_dict
            ):
                legacy_result = self._plan_to_result(verdict_dict)
                legacy_result.queries_executed = dict(accumulated_queries)
                legacy_result.query_errors = dict(accumulated_errors)
                legacy_result.total_llm_calls = total_llm_calls
                legacy_result.converged = True
                legacy_result.final_verdict = "accepted"
                return legacy_result

            verdict = ReviewVerdict(
                status=str(verdict_dict.get("status", "rejected")),
                reason=str(verdict_dict.get("reason", "")),
                reflection=str(verdict_dict.get("reflection", "")),
                missing_parts=list(verdict_dict.get("missing_parts", [])),
                suggested_queries={k: str(v) for k, v in dict(verdict_dict.get("suggested_queries", {})).items()},
                accepted_labels=list(verdict_dict.get("accepted_labels", [])),
                needs_columns=list(verdict_dict.get("needs_columns", [])),
                merge_ready=bool(verdict_dict.get("merge_ready", False)),
            )
            review_history.append(verdict)

            if verdict.status == "accepted":
                break
            if verdict.status == "merge" or verdict.merge_ready:
                break
            if attempt >= max_retries:
                logger.warning("Max retries (%d) reached, proceeding with accumulated results", max_retries)
                break

            if verdict.status == "partial":
                reflection = verdict.reflection
                skip_labels = set(verdict.accepted_labels)
                only_labels = set(verdict.suggested_queries.keys()) if verdict.suggested_queries else None
            elif verdict.status == "error":
                failed_labels = list(accumulated_errors.keys())
                retry_labels = [label for label in verdict.suggested_queries if label in failed_labels]
                for label in retry_labels:
                    accumulated_errors.pop(label, None)
                    accumulated_results.pop(label, None)
                reflection = verdict.reflection
                skip_labels = set()
                if retry_labels:
                    only_labels = set(retry_labels)
                else:
                    only_labels = set(failed_labels) if failed_labels else None
            elif verdict.status == "plan":
                for label, frame in self._frames.items():
                    try:
                        fresh = frame.read_fresh()
                    except Exception:
                        continue
                    needed = [column for column in verdict.needs_columns if column in fresh.columns]
                    if not needed:
                        continue
                    shape_line = f"shape: {len(fresh):,} rows x {len(fresh.columns)} columns"
                    dtypes_lines = [
                        f"  {column}: {fresh.dtypes[column]}"
                        for column in needed
                    ]
                    extra_schema[label] = (
                        "Additional columns available:\n"
                        f"{shape_line}\n"
                        "Columns:\n"
                        + "\n".join(dtypes_lines)
                    )
                reflection = verdict.reflection
                skip_labels = set()
                only_labels = None
            elif verdict.status == "rejected":
                # Only clear if not the last iteration — preserve results for final analysis
                if attempt < max_retries:
                    accumulated_results.clear()
                    accumulated_queries.clear()
                    accumulated_errors.clear()
                    reflection = verdict.reflection
                    skip_labels = set()
                    only_labels = None
                else:
                    logger.warning(
                        "Last attempt rejected, but preserving accumulated results for final analysis"
                    )

        final_messages = build_analysis_messages(
            instruction=instruction,
            query_results=accumulated_results,
            query_errors=accumulated_errors,
            original_queries=accumulated_queries,
        )
        raw_analysis = await self._call_llm(final_messages)
        total_llm_calls += 1
        analysis_plan = parse_plan(raw_analysis)
        result = self._plan_to_result(analysis_plan)
        result.queries_executed = dict(accumulated_queries)
        result.query_errors = dict(accumulated_errors)
        result.review_history = review_history
        result.total_llm_calls = total_llm_calls
        result.converged = bool(review_history) and review_history[-1].status in ("accepted", "merge")
        result.final_verdict = review_history[-1].status if review_history else "unknown"

        logger.info(
            "MultiFrameAgent done: verdict=%s converged=%s llm_calls=%d attempts=%d elapsed_ms=%.2f",
            result.final_verdict,
            result.converged,
            total_llm_calls,
            len(review_history),
            (time.perf_counter() - started_at) * 1000.0,
        )
        return result

    def _safe_eval(self, query_str: str, df: "pd.DataFrame") -> "pd.DataFrame":
        """Execute pandas expression dengan sandbox minimal."""
        import pandas as pd

        query_stripped = query_str.strip()

        for pattern in _FORBIDDEN_PATTERNS:
            if pattern in query_stripped:
                raise ValueError(f"Forbidden pattern '{pattern}' in query")

        if not query_stripped.startswith("df"):
            raise ValueError(f"Query must start with 'df'. Got: {query_stripped[:50]}")

        for method in re.findall(r"\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", query_stripped):
            if method not in _ALLOWED_PANDAS_METHODS:
                raise ValueError(f"Method '{method}' is not allowed")

        result = eval(  # noqa: S307
            query_stripped,
            {"__builtins__": {}},
            {"df": df, "pd": pd},
        )

        if not isinstance(result, (pd.DataFrame, pd.Series)):
            raise ValueError(f"Query must return DataFrame or Series, got {type(result).__name__}")

        if isinstance(result, pd.Series):
            series_result: pd.Series = result
            frame_result = series_result.reset_index()
            frame_result.columns = [str(col) for col in frame_result.columns]
            return frame_result

        frame_result: pd.DataFrame = result
        return frame_result

    def _build_schema_context(self) -> dict[str, str]:
        """Build schema + statistics context without sample rows."""
        contexts: dict[str, str] = {}
        for label, frame in self._frames.items():
            fresh = frame.read_fresh()
            hint_cols = self._columns_hint.get(label) if self._columns_hint is not None else None

            parts = [
                f"label: {label}",
                f"frame_id: {frame._frame_id}",
                f"shape: {len(fresh):,} rows x {len(fresh.columns)} columns",
            ]

            if hint_cols is not None:
                available_hints = [c for c in hint_cols if c in fresh.columns]
                missing_hints = [c for c in hint_cols if c not in fresh.columns]
                if missing_hints:
                    logger.warning(
                        "_build_schema_context: columns not found in frame '%s': %s",
                        label,
                        missing_hints,
                    )

                if not available_hints:
                    logger.warning(
                        "_build_schema_context: no valid hint columns for frame '%s', "
                        "falling back to full schema",
                        label,
                    )
                    parts.append(f"\nColumn dtypes:\n{fresh.dtypes.to_string()}")
                else:
                    parts.append("\nRelevant columns (use these for queries):")
                    for col in available_hints:
                        dtype = fresh[col].dtype
                        n_unique = fresh[col].nunique(dropna=True)
                        parts.append(f"  {col}: {dtype} ({n_unique:,} unique values)")

                    other_cols = [c for c in fresh.columns if c not in available_hints]
                    if other_cols:
                        parts.append("\nOther available columns (not selected for this analysis):")
                        parts.append(f"  {other_cols}")
            else:
                parts.append(f"\nColumn dtypes:\n{fresh.dtypes.to_string()}")

                numeric = fresh.select_dtypes(include="number")
                if not numeric.empty:
                    try:
                        stats = numeric.describe()
                        parts.append(f"\nStatistics:\n{stats.to_string()}")
                    except Exception:
                        pass

                if frame._schema:
                    parts.append("\nColumn descriptions:")
                    for col, schema in frame._schema.items():
                        desc = schema.description or schema.dtype
                        parts.append(f"  {col}: {desc}")

            contexts[label] = "\n".join(parts)

        return contexts

    def _build_context_with_hint(
        self,
        label: str,
        frame: "DFrame",
        columns: list[str],
        max_rows: int = 5,
    ) -> str:
        """Build token-efficient context using selected columns and sample rows."""
        fresh = frame.read_fresh()

        available = [c for c in columns if c in fresh.columns]
        missing = [c for c in columns if c not in fresh.columns]

        if missing:
            logger.warning(
                "_build_context_with_hint: columns not found in frame '%s': %s",
                label,
                missing,
            )

        if not available:
            logger.warning(
                "_build_context_with_hint: no valid columns for frame '%s', "
                "falling back to all columns",
                label,
            )
            available = list(fresh.columns)

        subset = fresh[available].head(max_rows)

        parts = [
            f"label: {label}",
            f"frame_id: {frame._frame_id}",
            f"total_rows: {len(fresh):,}",
            f"showing_columns: {available}",
            "(other columns not shown - not relevant to instruction)",
            f"\nSample ({min(max_rows, len(fresh))} of {len(fresh):,} rows):",
            subset.to_string(index=True),
            "\nColumn types:",
        ]

        for col in available:
            parts.append(f"  {col}: {fresh[col].dtype}")

        numeric = subset.select_dtypes(include="number")
        if not numeric.empty:
            try:
                parts.append(f"\nNumeric statistics:\n{numeric.describe().to_string()}")
            except Exception:
                pass

        obj_cols = subset.select_dtypes(include=["object", "string"]).columns
        if len(obj_cols) > 0:
            parts.append("\nTop values per categorical column:")
            for col in obj_cols[:3]:
                top = fresh[col].value_counts().head(5)
                parts.append(f"  {col}: {top.to_dict()}")

        return "\n".join(parts)

    async def _write_to_frame(self, operations: list[dict[str, Any]], output_frame: "DFrame") -> dict[str, Any]:
        from .writer import AgentWriter

        writer = AgentWriter(
            coordinator=output_frame._coordinator,
            agent_id=self._agent_id,
            author_type="llm_agent",
            frame_id=output_frame._frame_id,
        )
        return await writer.batch_enrich(operations)

    async def _call_llm(self, messages: list[dict[str, str]]) -> str:
        model_name = self._model or ("claude-sonnet-4-20250514" if self._provider == "anthropic" else "gpt-4o")
        logger.debug(
            "multi_agent LLM_REQUEST: provider=%s model=%s messages=%d preview=%s",
            self._provider,
            model_name,
            len(messages),
            summarize_messages(messages),
        )
        if self._provider == "anthropic":
            raw = await self._call_anthropic(messages)
        elif self._provider == "openai":
            raw = await self._call_openai(messages)
        else:
            raise ValueError(f"Unknown provider: {self._provider}. Use 'anthropic' or 'openai'.")
        logger.debug(
            "multi_agent LLM_RESPONSE: provider=%s model=%s chars=%d preview=%s",
            self._provider,
            model_name,
            len(raw),
            truncate_text(raw),
        )
        return raw

    async def _call_anthropic(self, messages: list[dict[str, str]]) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("anthropic package required: pip install anthropic") from exc

        client = anthropic.AsyncAnthropic(api_key=self._anthropic_api_key)
        model = self._model or "claude-sonnet-4-20250514"

        system = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
        user_messages = [msg for msg in messages if msg["role"] != "system"]

        response = await client.messages.create(
            model=model,
            max_tokens=2000,
            system=system,
            messages=user_messages,
        )
        return response.content[0].text

    async def _call_openai(self, messages: list[dict[str, str]]) -> str:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("openai package required: pip install openai") from exc

        client = AsyncOpenAI(api_key=self._openai_api_key)
        model = self._model or "gpt-4o"

        new_param_models = ("o1", "o3", "o4", "gpt-4.1")
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if any(model.startswith(prefix) for prefix in new_param_models):
            kwargs["max_completion_tokens"] = 2000
        else:
            kwargs["max_tokens"] = 2000

        response = await client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    @staticmethod
    def _plan_to_result(plan: dict[str, Any]) -> MultiFrameResult:
        insights: list[FrameInsight] = []
        for raw in plan.get("insights", []):
            insights.append(
                FrameInsight(
                    finding=str(raw.get("finding", "")),
                    frames=list(raw.get("frames", [])),
                    confidence=float(raw.get("confidence", 0.8)),
                    row_references=list(raw.get("row_references", [])),
                )
            )

        series: list[SeriesSpec] = []
        for raw_series in plan.get("series", []):
            raw_data = raw_series.get("data", [])
            if not isinstance(raw_data, list):
                logger.warning("Skipping series '%s': data is not a list", raw_series.get("name", "?"))
                continue
            clean_data = [item for item in raw_data if isinstance(item, dict)]
            if not clean_data:
                logger.warning("Skipping series '%s': no valid data rows", raw_series.get("name", "?"))
                continue
            series.append(SeriesSpec(
                name=str(raw_series.get("name", f"series_{len(series)}")),
                description=str(raw_series.get("description", "")),
                data=clean_data,
                suggested_x=str(raw_series.get("suggested_x", "")),
                suggested_y=raw_series.get("suggested_y", ""),
                suggested_group_by=raw_series.get("suggested_group_by"),
                unit=str(raw_series.get("unit", "")),
                source_frames=list(raw_series.get("source_frames", [])),
            ))

        return MultiFrameResult(
            action=str(plan.get("action", "analyze")),
            reasoning=str(plan.get("reasoning", "")),
            analysis=str(plan.get("analysis", "")),
            insights=insights,
            operations=list(plan.get("operations", [])),
            series=series,
        )



