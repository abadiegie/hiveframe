# Stream Normalize

## Overview

`stream_normalize()` adalah method di `AgentWriter` untuk melakukan normalisasi kolom dengan LLM secara streaming. Digunakan untuk standardisasi, pembersihan, atau transformasi data pada skala besar tanpa membebani context window model.

## Konsep

Sebelumnya, prompt yang dikirim ke LLM adalah **generic/dummy** — tidak contextual terhadap data actual:

```python
# OLD: Generic prompt
"Normalize column 'city' to proper format"
# LLM tidak melihat: tipe data, kolom lain, pola data
```

Sekarang menggunakan **context-aware prompts**:

```python
# NEW: Smart prompt dengan context
- Instruction yang spesifik
- Actual data chunk dengan semua kolom (untuk konteks)
- Row numbers yang akurat
- Type information per kolom
- Sample patterns
```

Ini memungkinkan LLM membuat keputusan normalisasi yang lebih akurat dengan confidence score yang realitas.

## Fitur

- ✅ **Chunked processing**: Kirim data dalam batch kecil untuk efisiensi token
- ✅ **Context-aware prompts**: Data + konteks kolom lain untuk keputusan lebih baik
- ✅ **Confidence scoring**: LLM menentukan confidence, hanya values confident yang ditulis
- ✅ **Debug logging**: Full audit trail di logs untuk troubleshooting
- ✅ **Custom instructions**: Instruction rule yang spesifik per use case
- ✅ **Progress tracking**: Optional callback untuk monitor progress

## API

```python
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
        llm_call: Async function that takes LLM messages and returns
                  list of write operations. Example:
                  async def my_llm_call(messages):
                      response = await client.chat.completions.create(...)
                      return parse_plan(response)["operations"]
        chunk_size: Number of rows per LLM call. Default 50.
                    Reduce for smaller context windows (e.g., 10-20).
        progress_callback: Optional callback(processed, total) for progress tracking.
        custom_instruction: Natural language instruction for normalization.
                           If None, uses "Normalize column '{column}'".
                           Example: "Standardize province names to official format"

    Returns:
        {
            "written": int,     # rows successfully written
            "skipped": int,     # rows skipped (low confidence)
            "total": int        # total rows processed
        }
    """
```

## Prompt Structure

Prompt yang dikirim ke LLM sekarang context-aware:

### System Message
- Role dan tanggung jawab agent
- Cell ID convention yang jelas
- Confidence scoring guidelines
- Rules untuk avoid hallucination

### Context Message
- **Normalization Task**: Deskripsi instruction
- **Target Information**: Frame ID, kolom target, range rows
- **Context Columns**: Kolom lain yang bisa membantu keputusan
- **Data to Normalize**: Actual data chunk (all columns)

### User Message
- Instruksi untuk normalize sesuai rule dengan row numbers yang akurat

## Logging

Setiap step dicatat dengan level DEBUG/INFO untuk transparency:

```
2026-04-09 21:22:40,300 INFO stream_normalize STARTED: column=city total_rows=1000 chunk_size=50
2026-04-09 21:22:40,301 DEBUG stream_normalize CHUNK 1: rows 0-49 (size=50)
2026-04-09 21:22:40,302 DEBUG stream_normalize CONTEXT: chunk 1 snapshot_lines=51 columns='city', 'region', 'population'
2026-04-09 21:22:40,310 DEBUG stream_normalize LLM_CALL: chunk 1 messages=5
2026-04-09 21:22:40,350 DEBUG stream_normalize LLM_RESPONSE: chunk 1 operations=45
2026-04-09 21:22:40,351 DEBUG _build_operations STARTED: items=45
2026-04-09 21:22:40,352 DEBUG _build_operations READING: cells=45 cell_ids=['...', '...', '...']
2026-04-09 21:22:40,360 DEBUG _build_operations ACCEPT: idx=0 cell_id=abc123::city_0 old='jakarta' new='DKI Jakarta' confidence=0.9700
2026-04-09 21:22:40,390 DEBUG _submit_with_retry SUBMIT: attempt=1/3 ops=45
2026-04-09 21:22:40,400 INFO _submit_with_retry SUCCESS: attempt=1 tx_id=tx_xyz written=45 skipped=0 state=<TxState.SYNCED>
2026-04-09 21:22:40,400 INFO stream_normalize BATCH_RESULT: chunk 1 written=45 skipped=0
...
2026-04-09 21:22:45,500 INFO stream_normalize FINISHED: total=1000 written=980 skipped=20 success_rate=98.0%
```

## Contoh Penggunaan

### Basic: Standardize City Names

```python
from openai import AsyncOpenAI
from hiveframe.agent import AgentWriter
from hiveframe.agent.prompt import parse_plan

# Setup
client = AsyncOpenAI(api_key="...")
writer = AgentWriter(
    coordinator=df._coordinator,
    agent_id="city_standardizer",
    frame_id=df._frame_id,
)

# LLM caller
async def llm_call(messages):
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=1000,
        temperature=0.1,  # Low temp untuk konsistency
    )
    # Parse response
    return parse_plan(response.choices[0].message.content).get("operations", [])

# Normalize
result = await writer.stream_normalize(
    column="city",
    llm_call=llm_call,
    chunk_size=50,
    custom_instruction="Standardize city names to proper Indonesian city format with province prefix. "
                       "Examples: 'jakarta' → 'DKI Jakarta', 'bandung' → 'Jawa Barat: Bandung'"
)

print(f"Normalized {result['written']}/{result['total']} rows")
```

### With Progress Tracking

```python
# Progress callback
def on_progress(processed, total):
    pct = (processed / total * 100) if total > 0 else 0
    print(f"Progress: {processed}/{total} ({pct:.1f}%)")

result = await writer.stream_normalize(
    column="city",
    llm_call=llm_call,
    chunk_size=50,
    progress_callback=on_progress,
    custom_instruction="Standardize city names to proper Indonesian city format"
)
```

### Small Chunk for Small Context Models

```python
# Untuk model kecil (Qwen 1.5B, Llama 3.2B, dll)
# Gunakan chunk_size lebih kecil agar tetap dalam context window

result = await writer.stream_normalize(
    column="category",
    llm_call=llm_call_small_model,
    chunk_size=10,  # ← Lebih kecil dari default 50
    custom_instruction="Normalize product categories ke standard ecommerce taxonomy"
)
```

### Multiple Columns (Sequential)

```python
# Normalize beberapa kolom secara berurutan
columns_to_normalize = [
    ("city", "Standardize city names to proper Indonesian format"),
    ("category", "Normalize product categories to standard taxonomy"),
    ("status", "Standardize status values: active/inactive/pending"),
]

for col, instruction in columns_to_normalize:
    result = await writer.stream_normalize(
        column=col,
        llm_call=llm_call,
        chunk_size=50,
        custom_instruction=instruction,
    )
    print(f"{col}: {result['written']}/{result['total']} written")
```

## Confidence Threshold

Default confidence threshold: **0.60**

Hanya values dengan confidence ≥ 0.60 yang ditulis:

| Confidence | Status | Meaning |
|-----------|--------|---------|
| 0.95-1.00 | ✅ Write | Very certain |
| 0.80-0.94 | ✅ Write | High confidence |
| 0.60-0.79 | ✅ Write | Medium confidence |
| <0.60 | ❌ Skip | Too uncertain |

Jika LLM tidak confident, row di-skip (tidak diubah). Ini mencegah data corruption dari hallucination.

Dapat dikonfigurasi via `AgentWriter`:

```python
writer = AgentWriter(
    coordinator=df._coordinator,
    confidence_threshold=0.75,  # Raise threshold untuk strictness
)
```

## Best Practices

### 1. Custom Instructions Yang Jelas

❌ Buruk:
```python
custom_instruction="Normalize city"
```

✅ Baik:
```python
custom_instruction="Standardize city names to official Indonesian city format "
                   "with province prefix. Format: 'Province: City'. "
                   "Example: 'jkt' → 'DKI Jakarta: Jakarta Pusat'"
```

### 2. Chunk Size

- **Kecil (10-20)**: Untuk model dengan context window kecil (8K tokens)
- **Medium (50)**: Default, balanced untuk model standar
- **Besar (100+)**: Untuk model dengan large context (32K+ tokens)

### 3. Temperature

Gunakan low temperature (0.0-0.3) untuk normalisasi:

```python
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=0.1,  # ← Low temp untuk konsistency
    max_tokens=1000,
)
```

### 4. Monitor Logging

Enable DEBUG logging untuk troubleshooting:

```python
import logging
logging.getLogger("hiveframe.agent.writer").setLevel(logging.DEBUG)
```

## Debugging

### Semua rows di-skip

**Penyebab**: LLM tidak confident atau parsing error

**Debug**:
```python
# Set logging ke DEBUG
import logging
logging.basicConfig(level=logging.DEBUG)

# Lihat di log:
# _build_operations SKIP: cell_id=... confidence=0.45 (below threshold 0.60)
# → Confidence terlalu rendah, naikkan instruction clarity
# 
# stream_normalize LLM_ERROR: ...
# → LLM call error, check API key dan model availability
```

### Rows normalized tapi dengan salah value

**Penyebab**: LLM hallucinating atau instruction tidak jelas

**Fix**:
- Perbaiki instruction untuk lebih spesifik
- Tambah examples dalam instruction
- Turunkan chunk_size untuk lebih detail per batch
- Raise confidence_threshold untuk lebih strict

### Timeout atau slow processing

**Penyebab**: Chunk size terlalu besar atau LLM call slow

**Fix**:
```python
# Reduce chunk size
result = await writer.stream_normalize(
    column="city",
    llm_call=llm_call,
    chunk_size=20,  # ← Kurangi dari 50
)

# Atau increase LLM timeout
# (tergantung LLM provider client)
```

## Perbandingan dengan Alternatives

| Approach | Pro | Con |
|----------|-----|-----|
| **stream_normalize** | Context-aware, debug logging, low hallucination | Perlu async, API calls |
| `normalize()` single | Simple, synchronous | Per-row, tidak scalable |
| `batch_enrich()` manual | Flexible | Harus manage prompt sendiri |
| SQL/pandas | Fast, deterministic | Tidak flexible untuk complex rules |

## Limitations

1. **Iterative refinement tidak supported** — satu pass saja
   - Gunakan `MultiFrameAgent` mode="query" untuk iterative analysis

2. **Cross-frame context limited** — hanya satu frame
   - Gunakan `RelationalAgentWriter` untuk multi-frame normalization

3. **No automatic retry per row** — fail chunk menaffect semua rows dalam chunk
   - Gunakan chunk_size kecil untuk isolation

## Changelog

### v0.1.2 (April 2026)
- ✨ Context-aware prompts dengan actual data + all columns
- ✨ Enhanced debug logging di setiap step
- ✨ Support custom_instruction untuk flexible normalization rules
- ✨ Progress callback untuk monitoring

### v0.1.1 (sebelumnya)
- Generic prompt tanpa context
- Minimal logging

