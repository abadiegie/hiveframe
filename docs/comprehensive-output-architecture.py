"""
Visual Architecture: Comprehensive Analysis Output

Shows how everything fits together.
"""

# ============================================================================
# BEFORE vs AFTER
# ============================================================================

"""
BEFORE:
┌─────────────────────────┐
│   User Instruction      │
│  "Analyze sales trends" │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Analysis (LLM)         │
│  - Generate insights    │
│  - Execute queries      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Result                │
│  - analysis (text)      │
│  - insights (list)      │
│  - queries_executed     │
└─────────────────────────┘
             │
             ▼
         User sees:
      - 2-3 insights only
      - No data context
      - No quality metrics


AFTER:
┌──────────────────────────┐
│  User Instruction        │
│ "Analyze sales trends"   │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│ Step 1: Analysis (LLM)   │
│ - Generate insights      │
│ - Execute queries        │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│ Step 2: PROFILING (NEW)  │◄─────── NEW!
│ - Compute statistics     │
│ - Generate aggregations  │
│ - Create snapshots       │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│ Step 3: Attach Results   │
│ - Combine all data       │
│ - Format outputs         │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│   Result (ENRICHED)      │
│ - analysis (text)        │
│ - insights (list)        │
│ - queries_executed       │
│ ────────────────────     │
│ - frame_profiles (NEW)   │◄─ NOW!
│ - aggregation_snapshots  │◄─ NOW!
└──────────────────────────┘
            │
            ▼
         User sees:
      - Data overview (rows, cols, quality)
      - Key metrics (distributions)
      - Top values per category
      - LLM insights
      - Executed queries
      - ALL IN ONE PLACE!
"""

# ============================================================================
# DATA FLOW
# ============================================================================

"""
Input Data:
┌──────────────────────────────────────────┐
│ Frame: sales                             │
│ ├─ region: [Jakarta, Bandung, Surabaya]  │
│ ├─ amount: [10000, None, 8000]           │
│ ├─ quantity: [100, 150, 80]              │
│ └─ category: [Electronics, ...]          │
└──────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│ _profile_frame()                         │
│                                          │
│ For each column:                         │
│ ├─ Compute dtype, nulls, unique count    │
│ ├─ If numeric: min, max, mean, std       │
│ ├─ If categorical: top-10 values         │
│ └─ Identify good groupby columns         │
└──────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│ FrameProfile                             │
│                                          │
│ {                                        │
│   frame_label: "sales",                  │
│   row_count: 1000,                       │
│   col_count: 4,                          │
│   columns: {                             │
│     "region": ColumnProfile {            │
│       unique_count: 3,                   │
│       top_values: [                      │
│         ("Jakarta", 250),                │
│         ("Bandung", 180),                │
│         ("Surabaya", 120)                │
│       ],                                 │
│       ...                                │
│     },                                   │
│     "amount": ColumnProfile {            │
│       is_numeric: true,                  │
│       null_pct: 0.021,                   │
│       mean: 50234.50,                    │
│       std: 12345.67,                     │
│       ...                                │
│     }                                    │
│   },                                     │
│   top_groupby_results: {                 │
│     "region": [                          │
│       {"value": "Jakarta", "count": 250, │
│        "pct": 0.25},                     │
│       {"value": "Bandung", "count": 180, │
│        "pct": 0.18}                      │
│     ]                                    │
│   }                                      │
│ }                                        │
└──────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────┐
│ AggregationSnapshot                      │
│                                          │
│ [                                        │
│   {                                      │
│     frame_label: "sales",                │
│     aggregation_column: "region",        │
│     aggregation_type: "value_counts",    │
│     data: [                              │
│       {"value": "Jakarta", "count": 250, │
│        "pct": 0.25},                     │
│       {"value": "Bandung", "count": 180, │
│        "pct": 0.18},                     │
│       ...                                │
│     ]                                    │
│   },                                     │
│   ... (more snapshots for other frames)  │
│ ]                                        │
└──────────────────────────────────────────┘
"""

# ============================================================================
# OUTPUT GENERATION
# ============================================================================

"""
MultiFrameResult
├─ analysis (text from LLM)
├─ insights (list from LLM)
├─ queries_executed (dict)
├─ query_errors (dict)
├─ frame_profiles (dict) ◄─ NEW!
├─ aggregation_snapshots (list) ◄─ NEW!
└─ ...

        │
        ├─── to_markdown() ────────────────────────────┐
        │                                              │
        │ Output Format 1: RICH MARKDOWN              │
        │ ─────────────────────────────────────────── │
        │                                              │
        │ ## 📊 Data Overview                          │
        │ ### sales                                   │
        │ - Shape: 1,000 rows × 4 columns            │
        │ - Data Quality:                            │
        │   - amount: 2.1% null                       │
        │ - Numeric Stats:                           │
        │   - amount: μ=50234.50, σ=12345.67         │
        │                                              │
        │ ## 📈 Aggregation Snapshots                  │
        │ ### sales - region                          │
        │ - Jakarta: 250 (25.0%)                      │
        │ - Bandung: 180 (18.0%)                      │
        │                                              │
        │ ## Analysis                                  │
        │ [LLM analysis...]                           │
        │                                              │
        └──────────────────────────────────────────────┘
        │
        ├─── to_dict() ─────────────────────────────────┐
        │                                               │
        │ Output Format 2: JSON DICT                   │
        │ ──────────────────────────────────────────── │
        │                                               │
        │ {                                            │
        │   "action": "analyze",                       │
        │   "analysis": "...",                         │
        │   "insights": [...],                         │
        │   "queries_executed": {...},                 │
        │   "frame_profiles": {                        │
        │     "sales": {                               │
        │       "row_count": 1000,                      │
        │       "columns": {...},                      │
        │       "top_groupby_results": {...}           │
        │     }                                        │
        │   },                                         │
        │   "aggregation_snapshots": [...]             │
        │ }                                            │
        │                                               │
        └───────────────────────────────────────────────┘
        │
        └─── Programmatic Access ───────────────┐
                                               │
            Output Format 3: OBJECTS          │
            ─────────────────────────────    │
                                              │
            result.frame_profiles             │
            ├─ dict[label, FrameProfile]      │
            ├─ Each has .columns              │
            └─ Each has .top_groupby_results  │
                                              │
            result.aggregation_snapshots      │
            ├─ list[AggregationSnapshot]      │
            ├─ Each has .frame_label          │
            ├─ Each has .aggregation_column   │
            └─ Each has .data (list)          │
                                              │
                                              └──┘
"""

# ============================================================================
# SAMPLE OUTPUT
# ============================================================================

"""
User calls:
    result = await agent.analyze("Analyze sales trends")

Markdown Output (result.to_markdown()):
─────────────────────────────────────────────────────────

## 📊 Data Overview

### sales
- **Shape:** 1,000 rows × 4 columns
- **Data Quality Issues:**
  - amount: 2.1% null
- **Numeric Columns:**
  - amount: μ=50234.50, σ=12345.67, range=[1000, 99999]
  - quantity: μ=145.23, σ=87.45, range=[1, 500]

### inventory
- **Shape:** 500 rows × 3 columns
- **Data Quality Issues:**
  - stock: 5.0% null

## 📈 Aggregation Snapshots

### sales - region
- Jakarta: 250 (25.0%)
- Bandung: 180 (18.0%)
- Surabaya: 120 (12.0%)
- Others: 450 (45.0%)

### sales - category
- Electronics: 450 (45.0%)
- Clothing: 350 (35.0%)
- Furniture: 200 (20.0%)

### inventory - category
- Electronics: 200 (40.0%)
- Clothing: 150 (30.0%)
- Furniture: 150 (30.0%)

## Analysis

Based on the data analysis, Jakarta is the dominant market with 25% of total sales.
The Electronics category represents 45% of sales volume. Inventory levels show a
concerning 5% null rate in the stock column, indicating potential data quality issues.

## Key Insights

1. **Strong regional concentration in Jakarta**
   Sources: `sales`
   Confidence: 95%

2. **Electronics dominates product category**
   Sources: `sales`, `inventory`
   Confidence: 90%

3. **Data quality issues detected**
   Sources: `inventory`
   Confidence: 100%

## Queries Executed

**sales:** df.groupby('region')['quantity'].sum().nlargest(10).reset_index()
**inventory:** df[df['stock'].notna()][['category', 'stock']].groupby('category').mean()


───────────────────────────────────────────────────────
"""

# ============================================================================
# COMPARISON: DETAIL LEVELS
# ============================================================================

"""
User Question: "What are the key findings?"

WITHOUT Profiling:
┌─────────────────────────────────────────┐
│ "Jakarta shows strong sales performance │
│  in Electronics category. Further       │
│  investigation recommended."            │
└─────────────────────────────────────────┘
        Problem: What is "strong"?

WITH Profiling:
┌─────────────────────────────────────────┐
│ "Jakarta is the strongest region with   │
│  25% market share (250 out of 1,000).   │
│  Electronics dominates at 45% of sales. │
│  Note: Inventory data has 5% nulls in   │
│  stock column - may need cleaning."     │
└─────────────────────────────────────────┘
        Better: Specific, with evidence!
"""

# ============================================================================
# INTEGRATION WITH EXISTING FEATURES
# ============================================================================

"""
┌────────────────────────────────────────────────────────┐
│             MultiFrameAgent.analyze()                  │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Parameters:                                           │
│  ├─ instruction (string)                              │
│  ├─ mode ("sample" or "query")                        │
│  ├─ max_retries (int)                                 │
│  ├─ columns_hint (dict)                               │
│  ├─ include_profile (bool) ◄─ NEW!                    │
│  └─ ...others                                         │
│                                                        │
├────────────────────────────────────────────────────────┤
│ Step 1: Regular Analysis (existing)                   │
│ ├─ Build contexts/schemas                            │
│ ├─ Call LLM                                           │
│ ├─ Execute queries (if query mode)                    │
│ └─ Generate insights                                 │
│                                                        │
│ Step 2: Profile Generation (NEW)                     │
│ ├─ If include_profile=True:                           │
│ │  ├─ Read fresh frames                              │
│ │  ├─ Call _profile_frame() for each                │
│ │  ├─ Generate AggregationSnapshot                  │
│ │  └─ Attach to result                              │
│ └─ If include_profile=False:                         │
│    └─ Skip (result has empty profiles)               │
│                                                        │
├────────────────────────────────────────────────────────┤
│ Returns: MultiFrameResult (with profiles!)           │
│ ├─ analysis (text)                                    │
│ ├─ insights (list)                                    │
│ ├─ frame_profiles (dict) ◄─ NEW!                      │
│ ├─ aggregation_snapshots (list) ◄─ NEW!             │
│ └─ ...others                                         │
│                                                        │
└────────────────────────────────────────────────────────┘
"""

# ============================================================================
# EXTENSIBILITY
# ============================================================================

"""
Current (Level 1): Basic Profiling
├─ Column statistics (null %, unique count)
├─ Numeric stats (mean, std, min, max, median)
├─ Categorical distributions (top-10 values)
└─ Auto-aggregations (value_counts)

Future (Level 2): Enhanced Profiling
├─ Correlation matrix (numeric columns)
├─ Anomaly detection (outliers, unusual patterns)
├─ Data quality scoring (0-100)
└─ Conditional profiling (based on frame size)

Future (Level 3): Smart Profiling
├─ Column relationships (foreign keys)
├─ Trend analysis (time series)
├─ Cardinality analysis (identifier detection)
└─ Custom profilers (user-defined)

All built on top of current architecture!
"""

