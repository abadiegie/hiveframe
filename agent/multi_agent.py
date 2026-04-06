# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""MultiFrameAgent untuk analisis satu atau banyak DFrame."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from .prompt import parse_plan
from .result import FrameInsight, MultiFrameResult

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
    ) -> MultiFrameResult:
        """Jalankan analisis LLM terhadap satu atau banyak frame."""
        if mode not in ("sample", "query"):
            raise ValueError("mode must be 'sample' or 'query'")

        if mode == "query":
            result = await self._analyze_query_mode(instruction, max_result_rows)
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



