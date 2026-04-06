# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""MultiFrameAgent untuk analisis satu atau banyak DFrame."""

from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING, Any

from .prompt import parse_plan
from .result import FrameInsight, MultiFrameResult, ReviewVerdict

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
    """LLM agent untuk analisis satu atau banyak DFrame."""

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

        logger.info(
            "MultiFrameAgent created frames=%s provider=%s",
            list(frames.keys()),
            provider,
        )

    async def analyze(
        self,
        instruction: str,
        mode: str = "sample",
        max_sample_rows: int = 50,
        max_result_rows: int = 200,
        output_frame: "DFrame | None" = None,
        max_retries: int = 0,
    ) -> MultiFrameResult:
        """Jalankan analisis LLM terhadap satu atau banyak frame."""
        if mode not in ("sample", "query"):
            raise ValueError("mode must be 'sample' or 'query'")

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
        from .prompt import build_analysis_messages, build_query_generation_messages

        schema_contexts = self._build_schema_context()
        query_gen_messages = build_query_generation_messages(
            instruction=instruction,
            frame_schemas=schema_contexts,
        )

        raw_queries = await self._call_llm(query_gen_messages)
        query_plan = parse_plan(raw_queries)
        pandas_queries = query_plan.get("queries", {})
        if not isinstance(pandas_queries, dict) or not pandas_queries:
            logger.warning("LLM did not generate any queries, falling back to sample mode")
            return await self._analyze_sample_mode(instruction, 50)

        query_results: dict[str, pd.DataFrame] = {}
        query_errors: dict[str, str] = {}

        for label, query_str in pandas_queries.items():
            frame = self._frames.get(label)
            if frame is None:
                query_errors[label] = f"Frame '{label}' not found"
                continue
            try:
                fresh = frame.read_fresh()
                result_df = self._safe_eval(str(query_str), fresh)
                query_results[label] = result_df.head(max_result_rows)
            except Exception as exc:
                query_errors[label] = str(exc)

        result_contexts = {
            label: result_df.to_string(max_rows=max_result_rows)
            for label, result_df in query_results.items()
        }
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
        """Iterative query mode dengan review+retry loop."""
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
                accumulated_results.clear()
                accumulated_queries.clear()
                accumulated_errors.clear()
                reflection = verdict.reflection
                skip_labels = set()
                only_labels = None

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
        """Build schema + statistics tanpa sample rows."""
        contexts: dict[str, str] = {}
        for label, frame in self._frames.items():
            fresh = frame.read_fresh()
            parts = [
                f"label: {label}",
                f"frame_id: {frame._frame_id}",
                f"shape: {len(fresh):,} rows x {len(fresh.columns)} columns",
                f"\nColumn dtypes:\n{fresh.dtypes.to_string()}",
            ]

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
        if self._provider == "anthropic":
            return await self._call_anthropic(messages)
        if self._provider == "openai":
            return await self._call_openai(messages)
        raise ValueError(f"Unknown provider: {self._provider}. Use 'anthropic' or 'openai'.")

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

        return MultiFrameResult(
            action=str(plan.get("action", "analyze")),
            reasoning=str(plan.get("reasoning", "")),
            analysis=str(plan.get("analysis", "")),
            insights=insights,
            operations=list(plan.get("operations", [])),
        )



