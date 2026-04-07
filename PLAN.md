# hiveframe — Chart Series Output dari MultiFrameAgent

## Context

Codebase hiveframe v0.1.1. Baca file-file ini sebelum mulai:
- agent/result.py      → MultiFrameResult, FrameInsight, ReviewVerdict
- agent/multi_agent.py → MultiFrameAgent, _analyze_query_mode_iterative
- agent/prompt.py      → build_analysis_messages, semua prompts

Fitur ini adalah EXTENSION dari MultiFrameResult.
Tidak ada breaking change ke API yang sudah ada.

---

## FILOSOFI

LLM hanya provide DATA yang sudah di-aggregate.
User yang decide chart type dan rendering.

```
hiveframe core:
  ✓ SeriesSpec dataclass — structured data output
  ✓ MultiFrameResult.series — list of SeriesSpec
  ✓ result.to_dataframe(name) → pd.DataFrame
  ✓ result.to_plotly_figure(name) → go.Figure
  ✓ result.save_chart(name, path) → PNG file

User responsibility:
  → Pilih chart type (line/bar/scatter/dll)
  → Customize styling
  → Embed ke PDF, PowerPoint, email, dll
```

**Zero PDF dependency di core.**
Kalau user mau PDF: `fig.write_image("chart.png")` lalu embed manual.

---

## DATACLASS: SeriesSpec

Tambahkan ke `agent/result.py` SEBELUM MultiFrameResult.

```python
@dataclass
class SeriesSpec:
    """
    Structured data output dari LLM analysis.
    Siap di-render sebagai chart dengan Plotly atau library lain.

    LLM populate ini berdasarkan query results.
    User yang decide bagaimana cara render-nya.

    Attributes:
        name: Identifier unik untuk series ini.
              Dipakai sebagai key di result.to_dataframe(name).
        description: Apa yang series ini tunjukkan.
                     Bisa dipakai sebagai chart title atau caption.
        data: List of dicts — actual data rows.
              Setiap dict adalah satu row.
        suggested_x: Kolom yang LLM suggest sebagai x-axis.
        suggested_y: Kolom yang LLM suggest sebagai y-axis.
                     Bisa single column atau list untuk multi-line.
        suggested_group_by: Kolom untuk grouping/color.
                            None kalau tidak perlu grouping.
        unit: Unit dari y-axis kalau relevan.
              Contoh: "IDR", "units", "percentage", "%"
        source_frames: Frame labels yang jadi sumber data ini.

    Example::

        # LLM generate SeriesSpec ini dari query results
        spec = SeriesSpec(
            name="revenue_by_region",
            description="Monthly revenue per region Q1 2026",
            data=[
                {"month": "Jan", "revenue": 1200, "region": "Jakarta"},
                {"month": "Jan", "revenue": 890, "region": "Surabaya"},
                {"month": "Feb", "revenue": 1450, "region": "Jakarta"},
            ],
            suggested_x="month",
            suggested_y="revenue",
            suggested_group_by="region",
            unit="IDR (juta)",
            source_frames=["sales"],
        )

        # User render sesuka hati
        import plotly.express as px
        df = spec.to_dataframe()
        fig = px.line(df, x="month", y="revenue", color="region",
                      title=spec.description)
        fig.write_image("revenue_chart.png")
    """

    name: str
    description: str
    data: list[dict[str, Any]]
    suggested_x: str = ""
    suggested_y: str | list[str] = ""
    suggested_group_by: str | None = None
    unit: str = ""
    source_frames: list[str] = field(default_factory=list)

    def to_dataframe(self) -> "pd.DataFrame":
        """
        Convert data ke pandas DataFrame.
        Return empty DataFrame kalau data kosong.
        """
        import pandas as pd
        if not self.data:
            return pd.DataFrame()
        return pd.DataFrame(self.data)

    def to_plotly_figure(
        self,
        chart_type: str = "line",
        **kwargs: Any,
    ) -> "go.Figure":
        """
        Convert ke Plotly figure dengan chart_type yang user pilih.

        Args:
            chart_type: "line"|"bar"|"scatter"|"area"|"pie"|"histogram"
            **kwargs: Diteruskan ke plotly.express function.
                      Bisa override x, y, color, title, dll.

        Returns:
            go.Figure siap di-render atau di-save.

        Raises:
            ImportError: kalau plotly tidak terinstall.
            ValueError: kalau chart_type tidak dikenali.

        Example::

            fig = spec.to_plotly_figure("bar", title="Custom Title")
            fig.show()
            fig.write_image("chart.png")
        """
        try:
            import plotly.express as px
        except ImportError:
            raise ImportError(
                "plotly required for chart rendering: "
                "pip install plotly"
            )

        df = self.to_dataframe()
        if df.empty:
            raise ValueError(
                f"SeriesSpec '{self.name}' has no data to plot"
            )

        # Build kwargs dengan suggested values sebagai default
        # User-provided kwargs override suggested values
        plot_kwargs: dict[str, Any] = {
            "title": self.description,
        }

        if self.suggested_x:
            plot_kwargs["x"] = self.suggested_x
        if self.suggested_y:
            plot_kwargs["y"] = self.suggested_y
        if self.suggested_group_by:
            plot_kwargs["color"] = self.suggested_group_by

        # User kwargs override defaults
        plot_kwargs.update(kwargs)
        plot_kwargs["data_frame"] = df

        _CHART_BUILDERS = {
            "line": px.line,
            "bar": px.bar,
            "scatter": px.scatter,
            "area": px.area,
            "pie": px.pie,
            "histogram": px.histogram,
        }

        builder = _CHART_BUILDERS.get(chart_type.lower())
        if builder is None:
            supported = ", ".join(_CHART_BUILDERS.keys())
            raise ValueError(
                f"Unknown chart_type '{chart_type}'. "
                f"Supported: {supported}"
            )

        return builder(**plot_kwargs)

    def save_chart(
        self,
        path: str,
        chart_type: str = "line",
        width: int = 900,
        height: int = 500,
        scale: float = 2.0,
        **kwargs: Any,
    ) -> str:
        """
        Render chart dan save sebagai PNG.

        Args:
            path: Output file path. Extension .png disarankan.
            chart_type: Lihat to_plotly_figure().
            width: Image width dalam pixels.
            height: Image height dalam pixels.
            scale: Scale factor untuk resolution (default 2.0 = retina).
            **kwargs: Diteruskan ke to_plotly_figure().

        Returns:
            Absolute path dari file yang disimpan.

        Raises:
            ImportError: kalau plotly atau kaleido tidak terinstall.

        Example::

            path = spec.save_chart("revenue.png", chart_type="bar")
            print(f"Chart saved to {path}")
            # Embed ke PDF: reader.insertImage(page, rect, path)
        """
        from pathlib import Path

        fig = self.to_plotly_figure(chart_type=chart_type, **kwargs)

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            fig.write_image(
                str(output_path),
                width=width,
                height=height,
                scale=scale,
            )
        except Exception as exc:
            if "kaleido" in str(exc).lower():
                raise ImportError(
                    "kaleido required for PNG export: "
                    "pip install kaleido"
                ) from exc
            raise

        return str(output_path.resolve())
```

---

## UPDATE MultiFrameResult

Tambahkan field dan methods baru ke MultiFrameResult.
Tidak ada yang dihapus.

```python
@dataclass
class MultiFrameResult:
    # ... semua existing fields tetap ...

    # NEW field:
    series: list[SeriesSpec] = field(default_factory=list)

    # NEW methods:

    def get_series(self, name: str) -> SeriesSpec | None:
        """
        Get SeriesSpec by name.
        Return None kalau tidak ditemukan.

        Example::

            spec = result.get_series("revenue_by_region")
            if spec:
                df = spec.to_dataframe()
        """
        for s in self.series:
            if s.name == name:
                return s
        return None

    def to_dataframe(self, name: str) -> "pd.DataFrame":
        """
        Get data dari series tertentu sebagai pandas DataFrame.

        Args:
            name: SeriesSpec.name yang ingin diambil.

        Returns:
            pd.DataFrame. Empty DataFrame kalau name tidak ditemukan.

        Example::

            df = result.to_dataframe("revenue_by_region")
            fig = px.line(df, x="month", y="revenue")
        """
        spec = self.get_series(name)
        if spec is None:
            import pandas as pd
            return pd.DataFrame()
        return spec.to_dataframe()

    def to_plotly_figure(
        self,
        name: str,
        chart_type: str = "line",
        **kwargs: Any,
    ) -> "go.Figure":
        """
        Get Plotly figure dari series tertentu.

        Args:
            name: SeriesSpec.name.
            chart_type: "line"|"bar"|"scatter"|"area"|"pie"|"histogram"
            **kwargs: Diteruskan ke SeriesSpec.to_plotly_figure().

        Raises:
            KeyError: kalau name tidak ditemukan.

        Example::

            fig = result.to_plotly_figure(
                "revenue_by_region",
                chart_type="bar",
                title="Q1 Revenue",
            )
            fig.show()
        """
        spec = self.get_series(name)
        if spec is None:
            raise KeyError(
                f"Series '{name}' not found. "
                f"Available: {[s.name for s in self.series]}"
            )
        return spec.to_plotly_figure(chart_type=chart_type, **kwargs)

    def save_chart(
        self,
        name: str,
        path: str,
        chart_type: str = "line",
        **kwargs: Any,
    ) -> str:
        """
        Save chart dari series tertentu sebagai PNG.

        Args:
            name: SeriesSpec.name.
            path: Output file path.
            chart_type: Chart type untuk rendering.
            **kwargs: Diteruskan ke SeriesSpec.save_chart().

        Returns:
            Absolute path dari file yang disimpan.

        Raises:
            KeyError: kalau name tidak ditemukan.

        Example::

            path = result.save_chart(
                "revenue_by_region",
                "output/revenue.png",
                chart_type="line",
            )
            # path siap di-embed ke PDF
        """
        spec = self.get_series(name)
        if spec is None:
            raise KeyError(
                f"Series '{name}' not found. "
                f"Available: {[s.name for s in self.series]}"
            )
        return spec.save_chart(path, chart_type=chart_type, **kwargs)

    def save_all_charts(
        self,
        output_dir: str = ".",
        chart_type: str = "line",
        **kwargs: Any,
    ) -> dict[str, str]:
        """
        Save semua series sebagai PNG files.

        Args:
            output_dir: Directory untuk output files.
            chart_type: Default chart type untuk semua series.
            **kwargs: Diteruskan ke setiap SeriesSpec.save_chart().

        Returns:
            Dict name → absolute path untuk setiap chart yang disimpan.
            Series yang gagal di-skip (tidak raise exception).

        Example::

            paths = result.save_all_charts("output/charts/")
            for name, path in paths.items():
                print(f"{name}: {path}")
                # Embed setiap path ke PDF
        """
        import logging
        logger = logging.getLogger("hiveframe.result")

        paths: dict[str, str] = {}
        for spec in self.series:
            try:
                import os
                file_path = os.path.join(
                    output_dir,
                    f"{spec.name}.png",
                )
                saved = spec.save_chart(
                    file_path,
                    chart_type=chart_type,
                    **kwargs,
                )
                paths[spec.name] = saved
            except Exception as exc:
                logger.warning(
                    "Failed to save chart '%s': %s",
                    spec.name, exc,
                )
        return paths

    # UPDATE to_markdown() — tambahkan series summary
    def to_markdown(self) -> str:
        """Format hasil sebagai markdown report."""
        parts: list[str] = []

        if self.analysis:
            parts.append(f"## Analysis\n\n{self.analysis}")

        if self.insights:
            parts.append("\n## Key Insights\n")
            for idx, insight in enumerate(self.insights, 1):
                frames_str = ", ".join(
                    f"`{f}`" for f in insight.frames
                )
                parts.append(
                    f"{idx}. **{insight.finding}**\n"
                    f"   Sources: {frames_str}\n"
                    f"   Confidence: {insight.confidence:.0%}"
                )

        # NEW — series summary
        if self.series:
            parts.append("\n## Available Charts\n")
            for spec in self.series:
                y_str = (
                    ", ".join(spec.suggested_y)
                    if isinstance(spec.suggested_y, list)
                    else spec.suggested_y
                )
                parts.append(
                    f"- **`{spec.name}`** — {spec.description}\n"
                    f"  x: `{spec.suggested_x}` | "
                    f"y: `{y_str}`"
                    + (
                        f" | group: `{spec.suggested_group_by}`"
                        if spec.suggested_group_by
                        else ""
                    )
                    + f" | {len(spec.data)} rows"
                )
            parts.append(
                "\n_Use `result.to_plotly_figure(name, chart_type)` "
                "to render._"
            )

        # existing sections...
        if hasattr(self, "review_history") and self.review_history:
            parts.append("\n## Iteration History\n")
            for i, v in enumerate(self.review_history, 1):
                icon = {
                    "accepted": "✓", "partial": "◑",
                    "error": "✗", "plan": "→",
                    "rejected": "✗", "merge": "⊕",
                }.get(v.status, "?")
                parts.append(
                    f"{i}. {icon} **{v.status}** — {v.reason}"
                )

        if self.queries_executed:
            parts.append("\n## Queries Executed\n")
            for label, query in self.queries_executed.items():
                parts.append(
                    f"**{label}:**\n```python\n{query}\n```"
                )

        if self.query_errors:
            parts.append("\n## Query Errors\n")
            for label, error in self.query_errors.items():
                parts.append(f"- `{label}`: {error}")

        if self.write_result:
            written = self.write_result.get("written", 0)
            skipped = self.write_result.get("skipped", 0)
            parts.append(
                f"\n## Write Result\n\n"
                f"Written: {written} cells | "
                f"Skipped: {skipped} cells"
            )

        return "\n".join(parts)

    # UPDATE to_dict() — tambahkan series
    def to_dict(self) -> dict[str, Any]:
        """Serialize ke dict untuk JSON response."""
        base = {
            "action": self.action,
            "reasoning": self.reasoning,
            "analysis": self.analysis,
            "insights": [
                {
                    "finding": i.finding,
                    "frames": i.frames,
                    "confidence": i.confidence,
                    "row_references": i.row_references,
                }
                for i in self.insights
            ],
            "operations": self.operations,
            "queries_executed": self.queries_executed,
            "query_errors": self.query_errors,
            "write_result": self.write_result,
            "mode": self.mode,
        }

        # NEW — series
        if self.series:
            base["series"] = [
                {
                    "name": s.name,
                    "description": s.description,
                    "suggested_x": s.suggested_x,
                    "suggested_y": s.suggested_y,
                    "suggested_group_by": s.suggested_group_by,
                    "unit": s.unit,
                    "source_frames": s.source_frames,
                    "row_count": len(s.data),
                    # data tidak di-include di to_dict() default
                    # karena bisa sangat besar
                    # gunakan to_dataframe() untuk akses data
                }
                for s in self.series
            ]

        return base
```

---

## UPDATE agent/prompt.py — tambah series ke analysis prompt

Update `build_analysis_messages()` untuk instruksikan LLM
generate series specification.

Update bagian system prompt di `build_analysis_messages()`:

```python
# Tambahkan ke system prompt di build_analysis_messages():

_ANALYSIS_WITH_SERIES_PROMPT = (
    "You are a data analyst generating insights from query results.\n\n"
    "## Output format\n\n"
    "Respond with raw JSON:\n"
    "{\n"
    '  "action": "analyze",\n'
    '  "reasoning": "brief explanation of approach",\n'
    '  "analysis": "narrative analysis text",\n'
    '  "insights": [\n'
    '    {\n'
    '      "finding": "specific finding",\n'
    '      "frames": ["frame_label"],\n'
    '      "confidence": 0.0\n'
    "    }\n"
    "  ],\n"
    '  "series": [\n'
    "    {\n"
    '      "name": "snake_case_identifier",\n'
    '      "description": "what this data shows",\n'
    '      "suggested_x": "column_name",\n'
    '      "suggested_y": "column_name or [list]",\n'
    '      "suggested_group_by": "column_name or null",\n'
    '      "unit": "IDR|units|%|etc or empty string",\n'
    '      "source_frames": ["frame_label"],\n'
    '      "data": [{"col": "val"}, ...]\n'
    "    }\n"
    "  ],\n"
    '  "operations": []\n'
    "}\n\n"
    "## Series rules\n\n"
    "- Include series ONLY when data is suitable for visualization\n"
    "- name must be snake_case, unique, descriptive\n"
    "- data must be the ACTUAL aggregated data from query results\n"
    "- Keep data rows focused: max 200 rows per series\n"
    "- suggested_y can be a list for multi-line charts\n"
    "  e.g. [\"revenue\", \"target\"] for comparison\n"
    "- If no visualization makes sense, series = []\n"
    "- Do NOT invent data — only use data from query results\n"
)
```

Inject prompt ini ke `build_analysis_messages()`:

```python
def build_analysis_messages(
    instruction: str,
    query_results: dict[str, str],
    query_errors: dict[str, str],
    original_queries: dict[str, str],
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        # Ganti system prompt lama dengan yang include series
        {"role": "system", "content": _ANALYSIS_WITH_SERIES_PROMPT}
    ]
    # ... rest of function sama seperti sebelumnya ...
```

---

## UPDATE agent/multi_agent.py — parse series dari LLM response

Di `_plan_to_result()`, parse series dari plan:

```python
@staticmethod
def _plan_to_result(plan: dict[str, Any]) -> MultiFrameResult:
    from .result import FrameInsight, MultiFrameResult, SeriesSpec

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

    # NEW — parse series
    series: list[SeriesSpec] = []
    for raw_series in plan.get("series", []):
        # Validate data adalah list of dicts
        raw_data = raw_series.get("data", [])
        if not isinstance(raw_data, list):
            continue

        # Pastikan setiap item adalah dict
        clean_data = [
            item for item in raw_data
            if isinstance(item, dict)
        ]

        if not clean_data:
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
        series=series,    # NEW
    )
```

---

## pyproject.toml — tambah optional dependency

```toml
[project.optional-dependencies]
charts = [
    "plotly>=5.0",
    "kaleido>=0.2",
]
excel = [
    "openpyxl>=3.1",
]
```

Note: plotly dan kaleido adalah OPTIONAL.
Kalau tidak terinstall — ImportError dengan helpful message.
Jangan add ke main dependencies.

---

## TESTS: tests/test_series_output.py

```python
# File test baru

# ── SeriesSpec basic ──────────────────────────────────────────────

def test_series_spec_to_dataframe()
    # SeriesSpec dengan data list of dicts
    # to_dataframe() return DataFrame dengan kolom yang benar
    # Row count sesuai

def test_series_spec_to_dataframe_empty()
    # SeriesSpec dengan data=[]
    # to_dataframe() return empty DataFrame
    # Tidak raise exception

def test_series_spec_to_plotly_figure_line(mock_plotly)
    # chart_type="line" → px.line dipanggil
    # suggested_x dan suggested_y dipakai sebagai default

def test_series_spec_to_plotly_figure_bar(mock_plotly)
    # chart_type="bar" → px.bar dipanggil

def test_series_spec_to_plotly_figure_unknown_type()
    # chart_type="unknown" → ValueError dengan list supported types

def test_series_spec_to_plotly_figure_user_override(mock_plotly)
    # User pass title="Custom" → override suggested title
    # suggested_x tetap dipakai kalau tidak di-override

def test_series_spec_to_plotly_figure_no_plotly()
    # Mock: import plotly raises ImportError
    # → ImportError dengan helpful message

def test_series_spec_save_chart(tmp_path, mock_plotly)
    # save_chart("chart.png") → file exists di tmp_path
    # Return value adalah absolute path

def test_series_spec_save_chart_creates_dir(tmp_path, mock_plotly)
    # Path dengan subdirectory yang belum ada
    # → directory dibuat otomatis

def test_series_spec_save_chart_no_kaleido(mock_plotly_kaleido_missing)
    # write_image raise kaleido error
    # → ImportError dengan helpful message

# ── MultiFrameResult series methods ──────────────────────────────

def test_result_get_series_found()
    # result.get_series("sales") → SeriesSpec yang benar

def test_result_get_series_not_found()
    # result.get_series("nonexistent") → None (tidak raise)

def test_result_to_dataframe_found()
    # result.to_dataframe("sales") → DataFrame dengan data

def test_result_to_dataframe_not_found()
    # result.to_dataframe("nonexistent") → empty DataFrame

def test_result_to_plotly_figure_found(mock_plotly)
    # result.to_plotly_figure("sales", "bar") → Figure

def test_result_to_plotly_figure_not_found()
    # result.to_plotly_figure("nonexistent") → KeyError
    # Error message menyebut available series

def test_result_save_chart_found(tmp_path, mock_plotly)
    # result.save_chart("sales", str(tmp_path/"sales.png"))
    # → file exists, return absolute path

def test_result_save_chart_not_found()
    # result.save_chart("nonexistent", "chart.png") → KeyError

def test_result_save_all_charts(tmp_path, mock_plotly)
    # result dengan 2 series
    # save_all_charts(str(tmp_path)) → dict dengan 2 paths
    # Kedua files exist

def test_result_save_all_charts_skips_failed(tmp_path, mock_plotly)
    # Satu series gagal save (mock error)
    # save_all_charts() tidak raise
    # Return dict hanya berisi yang berhasil

# ── to_markdown dengan series ─────────────────────────────────────

def test_result_to_markdown_with_series()
    # to_markdown() mengandung "Available Charts"
    # Series name muncul di output

def test_result_to_markdown_no_series()
    # series=[] → tidak ada "Available Charts" section

# ── to_dict series ────────────────────────────────────────────────

def test_result_to_dict_includes_series()
    # to_dict() mengandung "series" key
    # Setiap series punya name, description, row_count
    # "data" TIDAK ada di to_dict() (terlalu besar)

def test_result_to_dict_no_series()
    # series=[] → "series" key tidak ada atau empty list

# ── _plan_to_result parsing ───────────────────────────────────────

def test_plan_to_result_parses_series()
    # plan dict dengan "series" array
    # → MultiFrameResult.series populated dengan benar

def test_plan_to_result_skips_invalid_series()
    # series item dengan data bukan list → di-skip
    # series item dengan data berisi non-dict → di-skip
    # Valid series tetap masuk

def test_plan_to_result_empty_series()
    # plan tanpa "series" key → result.series == []

# ── Integration: full analyze flow dengan series ──────────────────

def test_analyze_query_mode_populates_series(monkeypatch)
    # Mock LLM final analysis return series dalam JSON
    # result.series tidak kosong
    # result.series[0].data tidak kosong

def test_analyze_sample_mode_populates_series(monkeypatch)
    # sample mode juga bisa populate series
    # (karena _plan_to_result sama)
```

---

## BUILD ORDER

```
Step 1: agent/result.py
        → Tambah SeriesSpec dataclass dengan semua methods
        → Tambah field series ke MultiFrameResult
        → Tambah get_series(), to_dataframe(), to_plotly_figure(),
          save_chart(), save_all_charts()
        → Update to_markdown() dan to_dict()
        TEST: from agent.result import SeriesSpec
              spec = SeriesSpec(
                  name="test",
                  description="test series",
                  data=[{"x": 1, "y": 2}, {"x": 2, "y": 4}],
                  suggested_x="x",
                  suggested_y="y",
              )
              df = spec.to_dataframe()
              assert len(df) == 2

Step 2: agent/prompt.py
        → Tambah _ANALYSIS_WITH_SERIES_PROMPT
        → Update build_analysis_messages() pakai prompt baru
        TEST: messages = build_analysis_messages(
                  "test", {"df": "data"}, {}, {}
              )
              assert "series" in messages[0]["content"]

Step 3: agent/multi_agent.py
        → Update _plan_to_result() untuk parse series
        TEST: plan = {
                  "analysis": "test",
                  "series": [{
                      "name": "revenue",
                      "description": "revenue trend",
                      "data": [{"month": "Jan", "val": 100}],
                      "suggested_x": "month",
                      "suggested_y": "val",
                  }]
              }
              result = MultiFrameAgent._plan_to_result(plan)
              assert len(result.series) == 1
              assert result.series[0].name == "revenue"

Step 4: pyproject.toml
        → Tambah [charts] optional dependency

Step 5: tests/test_series_output.py
        RUN: pytest tests/test_series_output.py -v

Step 6: Full test suite
        RUN: pytest -q
        Semua existing tests harus pass
```

---

## USAGE EXAMPLE (untuk docstring atau README)

```python
import hiveframe as hf
from hiveframe.agent import MultiFrameAgent
import plotly.express as px

# Setup
df_sales = hf.DFrame(sales_data)
agent = MultiFrameAgent(
    frames={"sales": df_sales},
    provider="anthropic",
)

# Analyze
result = await agent.analyze(
    "Trend penjualan per region Q1 2026",
    mode="query",
)

# Lihat apa yang tersedia
print(result.to_markdown())
# → ## Analysis
# → Trend menunjukkan...
# → ## Available Charts
# → - `revenue_by_region` — Monthly revenue per region
# →   x: `month` | y: `revenue` | group: `region` | 12 rows

# Akses data mentah
df = result.to_dataframe("revenue_by_region")
print(df.head())

# User decide chart type
fig = result.to_plotly_figure(
    "revenue_by_region",
    chart_type="line",           # user pilih
    title="Q1 2026 Revenue",    # user customize
)
fig.show()   # preview di browser

# Save sebagai PNG untuk embed ke PDF
path = result.save_chart(
    "revenue_by_region",
    "output/revenue_chart.png",
    chart_type="bar",    # user bisa ganti type saat save
    width=1200,
    height=600,
)
print(f"Chart saved: {path}")
# User embed path ke PDF dengan library pilihan mereka

# Save semua charts sekaligus
paths = result.save_all_charts("output/charts/")
for name, path in paths.items():
    print(f"{name}: {path}")
```

---

## CRITICAL RULES

1. SeriesSpec.data berisi ACTUAL data dari query results
   — LLM tidak boleh invent data
2. to_dict() TIDAK include data array — terlalu besar untuk JSON
   User gunakan to_dataframe() untuk akses data
3. plotly dan kaleido adalah OPTIONAL dependencies
   → ImportError dengan helpful message kalau tidak ada
   → Jangan add ke main dependencies
4. save_all_charts() TIDAK raise exception kalau satu gagal
   → Skip dan continue, log warning
5. suggested_y bisa string ATAU list of strings
   → Handle keduanya di to_plotly_figure()
6. Semua existing tests di test_multi_agent.py harus pass
7. _plan_to_result() skip series item yang invalid
   → Jangan raise, log warning saja

---

## WHAT NOT TO BUILD

- PDF generation (urusan user)
- Chart template system
- Animation atau interactive features
- Server-side chart rendering
- Chart storage atau caching
- Automatic chart type selection (user yang decide)
- Cross-series comparison logic
- Dashboard layout