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


def _parse_code_blocks(raw: str) -> dict[str, str]:
    """Parse an LLM response into per-frame Python code blocks."""
    import json

    blocks: dict[str, str] = {}

    for match in re.finditer(r"```python\s*\n(.*?)```", raw, re.DOTALL | re.IGNORECASE):
        code = match.group(1).strip()
        label_match = re.search(r"#\s*frame:\s*(\S+)", code)
        if label_match:
            label = label_match.group(1)
            code = re.sub(r"#\s*frame:\s*\S+\s*\n?", "", code, count=1).strip()
            blocks[label] = code

    if blocks:
        return blocks

    try:
        parsed = json.loads(raw.strip())
        queries = parsed.get("queries", {})
        if isinstance(queries, dict):
            return {str(k): str(v) for k, v in queries.items()}
    except (json.JSONDecodeError, ValueError):
        pass

    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            parsed = json.loads(match.group(0))
            queries = parsed.get("queries", {})
            if isinstance(queries, dict):
                return {str(k): str(v) for k, v in queries.items()}
        except (json.JSONDecodeError, ValueError):
            pass

    return blocks


def _replace_frame_label_variables(code: str, known_labels: set[str]) -> tuple[str, bool]:
    """Rewrite obvious frame-label variables to the canonical ``df`` name."""
    rewritten = code
    changed = False

    for label in sorted(known_labels, key=len, reverse=True):
        if label == "df" or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", label):
            continue
        updated = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(label)}(?=\s*[\[\.])", "df", rewritten)
        if updated != rewritten:
            rewritten = updated
            changed = True

    return rewritten, changed


def _replace_column_case_literals(code: str, columns: list[str]) -> tuple[str, bool]:
    """Rewrite quoted column-name literals that are a case-only mismatch with real columns.

    Works line-by-line so newlines in the original code are always preserved.
    """
    if not columns:
        return code, False

    canonical_by_lower: dict[str, str] = {}
    ambiguous: set[str] = set()
    for column in columns:
        lower = column.lower()
        if lower in canonical_by_lower and canonical_by_lower[lower] != column:
            ambiguous.add(lower)
            canonical_by_lower.pop(lower, None)
            continue
        if lower not in ambiguous:
            canonical_by_lower[lower] = column

    lines = code.split("\n")
    changed = False
    result_lines: list[str] = []
    for line in lines:
        new_line = line
        for wrong_lower, correct in canonical_by_lower.items():
            new_line = re.sub(
                rf"(['\"]){re.escape(wrong_lower)}\1",
                lambda m, c=correct: f"{m.group(1)}{c}{m.group(1)}",
                new_line,
                flags=re.IGNORECASE,
            )
        if new_line != line:
            changed = True
        result_lines.append(new_line)

    return "\n".join(result_lines), changed


def _rewrite_generated_code(
    code: str,
    frame_label: str,
    columns: list[str],
    known_labels: set[str],
) -> tuple[str, list[str]]:
    """Apply conservative local rewrites to generated code before execution."""
    rewritten = code
    applied: list[str] = []

    updated, changed = _replace_frame_label_variables(rewritten, set(known_labels) | {frame_label})
    if changed:
        rewritten = updated
        applied.append("frame_variable_to_df")

    updated, changed = _replace_column_case_literals(rewritten, columns)
    if changed:
        rewritten = updated
        applied.append("column_case_match")

    return rewritten, applied


class QueryExecutor:
    """PandasAI-style self-healing query executor."""

    MAX_RETRIES: int = 3

    def __init__(
        self,
        frames: dict[str, "DFrame"],
        call_llm,
        build_schema,
        safe_eval,
        max_retries: int = 3,
        max_result_rows: int = 200,
    ) -> None:
        self.frames = frames
        self.call_llm = call_llm
        self.build_schema = build_schema
        self.safe_eval = safe_eval
        self.max_retries = max_retries
        self.max_result_rows = max_result_rows

    async def run(
        self,
        instruction: str,
    ) -> tuple[dict[str, "pd.DataFrame"], dict[str, str]]:
        """Run code generation and self-healing execution."""
        import traceback

        from .prompt import build_code_gen_messages

        schema_contexts = self.build_schema()
        attempt_history: list[dict[str, str]] = []
        results: dict[str, "pd.DataFrame"] = {}
        errors: dict[str, str] = {}
        pending: set[str] = set(self.frames.keys())

        for attempt in range(self.max_retries):
            if not pending:
                break

            messages = build_code_gen_messages(
                instruction=instruction,
                frame_schemas={k: v for k, v in schema_contexts.items() if k in pending},
                attempt_history=[h for h in attempt_history if h.get("label") in pending],
            )
            raw = await self.call_llm(messages)
            code_blocks = _parse_code_blocks(raw)

            newly_failed: list[dict[str, str]] = []
            seen_known_labels: set[str] = set()

            for label, code in code_blocks.items():
                label = str(label).strip()
                code = str(code).strip()

                frame = self.frames.get(label)
                if frame is None:
                    error = (
                        f"Frame label '{label}' not found. Available frames: {list(self.frames.keys())}. "
                        "Frame label MUST match exactly."
                    )
                    newly_failed.append({"label": label, "code": code, "error": error})
                    errors[label] = error
                    logger.debug(
                        "QueryExecutor FAIL: attempt=%d label=%s\n%s",
                        attempt + 1,
                        label,
                        error,
                    )
                    continue

                if label not in pending:
                    continue

                seen_known_labels.add(label)
                try:
                    fresh = frame.read_fresh()
                    rewritten_code, applied_rewrites = _rewrite_generated_code(
                        code=code,
                        frame_label=label,
                        columns=[str(column) for column in fresh.columns],
                        known_labels=set(self.frames.keys()),
                    )
                    if applied_rewrites:
                        before_preview = truncate_text(" ".join(code.split()), 180)
                        after_preview = truncate_text(" ".join(rewritten_code.split()), 180)
                        logger.info(
                            "QueryExecutor REWRITE: attempt=%d label=%s rewrites=%s",
                            attempt + 1,
                            label,
                            applied_rewrites,
                        )
                        logger.debug(
                            "QueryExecutor REWRITE_PREVIEW: attempt=%d label=%s before=%s after=%s delta=%+d",
                            attempt + 1,
                            label,
                            before_preview,
                            after_preview,
                            len(rewritten_code) - len(code),
                        )
                    result_df = self.safe_eval(rewritten_code, fresh)
                    results[label] = result_df.head(self.max_result_rows)
                    errors.pop(label, None)
                    pending.discard(label)
                    logger.info(
                        "QueryExecutor OK: attempt=%d label=%s rows=%d",
                        attempt + 1,
                        label,
                        len(result_df),
                    )
                except Exception:
                    full_tb = traceback.format_exc()
                    if "NameError: name '" in full_tb and "' is not defined" in full_tb:
                        full_tb += (
                            "\nHint: only `df` is available as the DataFrame variable in generated code. "
                            "Do not use frame labels like `data` or `sales` as Python variables."
                        )
                    newly_failed.append({"label": label, "code": code, "error": full_tb})
                    errors[label] = full_tb
                    if any(
                        marker in full_tb
                        for marker in (
                            "Forbidden pattern",
                            "Method '",
                            "Query must start with 'df'",
                            "Code block must assign final output to 'result'",
                        )
                    ):
                        pending.discard(label)
                    logger.debug(
                        "QueryExecutor FAIL: attempt=%d label=%s\n%s",
                        attempt + 1,
                        label,
                        full_tb,
                    )

            for label in sorted(pending - seen_known_labels):
                error = (
                    "No code block returned for this frame. Return one fenced Python block per frame "
                    "with `# frame: <label>` as the first line."
                )
                newly_failed.append({"label": label, "code": "", "error": error})
                errors[label] = error
                logger.debug(
                    "QueryExecutor FAIL: attempt=%d label=%s\n%s",
                    attempt + 1,
                    label,
                    error,
                )

            attempt_history.extend(newly_failed)

            if not seen_known_labels:
                break

            if newly_failed and pending:
                logger.info(
                    "QueryExecutor retry: attempt=%d/%d pending=%s",
                    attempt + 1,
                    self.max_retries,
                    list(pending),
                )

        return results, errors


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
            if max_retries > 0:
                result = await self._analyze_query_mode_iterative(
                    instruction,
                    max_result_rows,
                    max_retries,
                )
            else:
                result = await self._analyze_query_mode_simple(instruction, max_result_rows)
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
        from .prompt import build_analysis_messages

        executor = QueryExecutor(
            frames=self._frames,
            call_llm=self._call_llm,
            build_schema=lambda: self._build_schema_context_with_samples(use_hint=False),
            safe_eval=self._safe_eval,
            max_retries=QueryExecutor.MAX_RETRIES,
            max_result_rows=max_result_rows,
        )
        query_results, query_errors = await executor.run(instruction)

        if not query_results:
            logger.info(
                "_analyze_query_mode_simple NO_RESULTS after %d attempts, falling back to sample mode",
                QueryExecutor.MAX_RETRIES,
            )
            fallback_result = await self._analyze_sample_mode(instruction, 50)
            fallback_result.query_errors = query_errors
            return fallback_result

        result_contexts = {
            label: result_df.to_string(max_rows=max_result_rows)
            for label, result_df in query_results.items()
        }

        analysis_messages = build_analysis_messages(
            instruction=instruction,
            query_results=result_contexts,
            query_errors=query_errors,
            original_queries={},
        )

        raw_analysis = await self._call_llm(analysis_messages)
        analysis_plan = parse_plan(raw_analysis)

        result = self._plan_to_result(analysis_plan)
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

        schema_contexts = self._build_schema_context(use_hint=False)

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
                schema_ctx = self._build_schema_context(use_hint=False)
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
        """Execute pandas code in a minimal sandbox."""
        import pandas as pd

        query_stripped = query_str.strip()
        has_result_assignment = bool(re.search(r"(^|[\n;])\s*result\s*=", query_stripped))

        for pattern in _FORBIDDEN_PATTERNS:
            if pattern in query_stripped:
                raise ValueError(f"Forbidden pattern '{pattern}' in query")

        if not query_stripped.startswith("df") and not has_result_assignment:
            raise ValueError(
                f"Query must start with 'df' or assign to 'result'. Got: {query_stripped[:50]}"
            )

        for method in re.findall(r"\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", query_stripped):
            if method not in _ALLOWED_PANDAS_METHODS:
                raise ValueError(f"Method '{method}' is not allowed")

        local_ns: dict[str, Any] = {"df": df, "pd": pd}
        for frame_label in self._frames:
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", frame_label):
                local_ns.setdefault(frame_label, df)

        if "\n" in query_stripped or has_result_assignment:
            exec(query_stripped, {"__builtins__": {}}, local_ns)  # noqa: S102
            result = local_ns.get("result")
            if result is None:
                raise ValueError("Code block must assign final output to 'result'")
        else:
            result = eval(  # noqa: S307
                query_stripped,
                {"__builtins__": {}},
                local_ns,
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

    def _build_schema_context(self, use_hint: bool = True) -> dict[str, str]:
        """Build schema + statistics context without sample rows."""
        contexts: dict[str, str] = {}
        for label, frame in self._frames.items():
            fresh = frame.read_fresh()
            hint_cols = (
                self._columns_hint.get(label)
                if use_hint and self._columns_hint is not None
                else None
            )

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

    def _build_schema_context_with_samples(
        self,
        n_sample: int = 3,
        use_hint: bool = True,
    ) -> dict[str, str]:
        """Build schema context with sample values for self-healing query generation."""
        contexts: dict[str, str] = {}
        for label, frame in self._frames.items():
            fresh = frame.read_fresh()
            hint_cols = (
                self._columns_hint.get(label)
                if use_hint and self._columns_hint
                else None
            )

            parts = [
                f"label: {label}",
                f"frame_id: {frame._frame_id}",
                f"shape: {len(fresh):,} rows x {len(fresh.columns)} columns",
            ]

            if hint_cols is not None:
                available = [col for col in hint_cols if col in fresh.columns]
                missing = [col for col in hint_cols if col not in fresh.columns]
                if missing:
                    logger.warning(
                        "_build_schema_context_with_samples: columns not found in frame '%s': %s",
                        label,
                        missing,
                    )
                if not available:
                    logger.warning(
                        "_build_schema_context_with_samples: no valid hint columns for frame '%s', "
                        "falling back to all columns",
                        label,
                    )
                    available = list(fresh.columns)
                    parts.append(f"\nColumn dtypes:\n{fresh.dtypes.to_string()}")
                else:
                    parts.append("\nRelevant columns (use these for queries):")
                    for col in available:
                        dtype = fresh[col].dtype
                        n_unique = fresh[col].nunique(dropna=True)
                        parts.append(f"  {col}: {dtype} ({n_unique:,} unique values)")

                    other_cols = [col for col in fresh.columns if col not in available]
                    if other_cols:
                        parts.append("\nOther available columns (not selected for this analysis):")
                        parts.append(f"  {other_cols}")
            else:
                available = list(fresh.columns)
                parts.append(f"\nColumn dtypes:\n{fresh.dtypes.to_string()}")

            sample_rows = fresh[available].head(n_sample)
            parts.extend(
                [
                    f"\ncolumns used for this query: {available}",
                    "",
                    "Column details (name | dtype | sample values):",
                ]
            )

            for col in available:
                try:
                    samples = fresh[col].dropna().head(n_sample).tolist()
                    samples_str = ", ".join(repr(sample) for sample in samples) if samples else "—"
                except Exception:
                    samples_str = "—"
                parts.append(f"  {col} | {fresh[col].dtype} | {samples_str}")

            parts.extend(
                [
                    "",
                    f"Sample rows (first {min(n_sample, len(sample_rows))}):",
                    sample_rows.to_string(index=True),
                ]
            )
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

        # Anthropic accepts only one `system` string — join all system messages.
        system_parts = [msg["content"] for msg in messages if msg["role"] == "system"]
        system = "\n\n".join(system_parts) if system_parts else None
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
            
            # Extract chart_type, default to "bar" if not specified by LLM
            chart_type = str(raw_series.get("chart_type", "bar")).lower()
            if chart_type not in ("bar", "line", "area", "scatter", "pie", "histogram", "heatmap"):
                logger.warning("Invalid chart_type '%s' for series '%s', using 'bar'", chart_type, raw_series.get("name", "?"))
                chart_type = "bar"
            
            series.append(SeriesSpec(
                name=str(raw_series.get("name", f"series_{len(series)}")),
                description=str(raw_series.get("description", "")),
                data=clean_data,
                suggested_x=str(raw_series.get("suggested_x", "")),
                suggested_y=raw_series.get("suggested_y", ""),
                suggested_group_by=raw_series.get("suggested_group_by"),
                chart_type=chart_type,
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



