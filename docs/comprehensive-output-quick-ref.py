"""
Quick Reference: Comprehensive Analysis Output

Use this as a quick guide untuk menggunakan comprehensive output feature.
"""

# ============================================================================
# BASIC USAGE
# ============================================================================

# 1. DEFAULT (WITH PROFILING)
result = await agent.analyze(
    instruction="Analyze sales trends",
    mode="sample",
    # include_profile=True  ← default, automatically enabled
)

print(result.to_markdown())  # Includes profiles + aggregation snapshots


# ============================================================================
# ACCESSING PROFILES
# ============================================================================

# Get frame profiles
for label, profile in result.frame_profiles.items():
    print(f"Frame: {label}")
    print(f"  Rows: {profile.row_count:,}")
    print(f"  Columns: {profile.col_count}")

    # Check data quality
    for col_name, col_prof in profile.columns.items():
        if col_prof.null_pct > 0.2:  # More than 20% nulls
            print(f"  ⚠️  {col_name}: {col_prof.null_pct:.1%} null")

        # Show numeric statistics
        if col_prof.is_numeric:
            print(f"  📊 {col_name}: μ={col_prof.mean:.2f}, σ={col_prof.std:.2f}")

        # Show categorical distribution
        if col_prof.top_values:
            top_str = ", ".join(f"{v[0]}({v[1]})" for v in col_prof.top_values[:3])
            print(f"  🏆 {col_name}: {top_str}")


# ============================================================================
# ACCESSING AGGREGATIONS
# ============================================================================

# Get aggregation snapshots
for snap in result.aggregation_snapshots:
    print(f"\n{snap.frame_label} - {snap.aggregation_column}:")
    for item in snap.data[:10]:  # Top 10 items
        value = item['value']
        count = item['count']
        pct = item.get('pct', 0)
        print(f"  {value:20s} {count:6d} ({pct:5.1%})")


# ============================================================================
# EXPORTING TO JSON
# ============================================================================

import json

result_dict = result.to_dict()

# Full export (includes profiles)
json_str = json.dumps(result_dict, indent=2)

# Just profiles
profiles_dict = {
    label: prof.to_dict()
    for label, prof in result.frame_profiles.items()
}
print(json.dumps(profiles_dict, indent=2))

# Just snapshots
snapshots_list = [snap.to_dict() for snap in result.aggregation_snapshots]
print(json.dumps(snapshots_list, indent=2))


# ============================================================================
# COMMON PATTERNS
# ============================================================================

# Pattern 1: Data Quality Check
def check_data_quality(result):
    """Check all frames for data quality issues."""
    issues = []
    for label, profile in result.frame_profiles.items():
        for col_name, col_prof in profile.columns.items():
            if col_prof.null_pct > 0.5:
                issues.append(f"{label}.{col_name}: {col_prof.null_pct:.0%} null")
            if col_prof.is_numeric and col_prof.std is None:
                issues.append(f"{label}.{col_name}: constant value (std=0)")
    return issues


# Pattern 2: Extract Top Values
def get_top_values(result, frame_label, column_name, top_n=10):
    """Extract top N values dari aggregation snapshot."""
    for snap in result.aggregation_snapshots:
        if snap.frame_label == frame_label and snap.aggregation_column == column_name:
            return snap.data[:top_n]
    return []


# Pattern 3: Compare Distributions Across Frames
def compare_distributions(result, column_name):
    """Compare column distributions across frames."""
    distributions = {}
    for snap in result.aggregation_snapshots:
        if snap.aggregation_column == column_name:
            distributions[snap.frame_label] = {
                item['value']: item['count']
                for item in snap.data
            }
    return distributions


# ============================================================================
# DISABLE PROFILING (IF NEEDED)
# ============================================================================

# For large datasets or performance reasons
result = await agent.analyze(
    instruction="Quick analysis",
    mode="sample",
    include_profile=False  # Skip profiling
)
# result.frame_profiles will be empty
# result.aggregation_snapshots will be empty


# ============================================================================
# DATA STRUCTURES
# ============================================================================

"""
ColumnProfile:
  - column_name: str
  - dtype: str
  - null_count: int
  - null_pct: float (0.0-1.0)
  - unique_count: int
  - is_numeric: bool
  - is_categorical: bool
  - is_temporal: bool
  - min: float | None  (if numeric)
  - max: float | None  (if numeric)
  - mean: float | None  (if numeric)
  - median: float | None  (if numeric)
  - std: float | None  (if numeric)
  - top_values: list[tuple[str, int]]  (if categorical)

FrameProfile:
  - frame_label: str
  - row_count: int
  - col_count: int
  - columns: dict[str, ColumnProfile]
  - top_groupby_results: dict[str, list[dict]]
    Format: {"column": [{"value": "A", "count": 10, "pct": 0.1}, ...]}

AggregationSnapshot:
  - frame_label: str
  - aggregation_column: str
  - aggregation_type: str
  - data: list[dict[str, Any]]
    Format: [{"value": "A", "count": 10, "pct": 0.1}, ...]
  - title: str

MultiFrameResult:
  - ... (existing fields)
  - frame_profiles: dict[str, FrameProfile]
  - aggregation_snapshots: list[AggregationSnapshot]
"""


# ============================================================================
# MARKDOWN OUTPUT EXAMPLE
# ============================================================================

"""
Output dari result.to_markdown():

## 📊 Data Overview

### sales
- **Shape:** 1,000 rows × 5 columns
- **Data Quality Issues:**
  - amount: 2.1% null
- **Numeric Columns:**
  - amount: μ=50234.50, σ=12345.67, range=[1000, 99999]
  - quantity: μ=145.23, σ=87.45, range=[1, 500]

## 📈 Aggregation Snapshots

### sales - region
- Jakarta: 250 (25.0%)
- Bandung: 180 (18.0%)
- Surabaya: 120 (12.0%)

## Analysis
[LLM-generated analysis]

## Key Insights
[LLM-generated insights]
"""


# ============================================================================
# TIPS & TRICKS
# ============================================================================

# Tip 1: Filter problematic columns
def get_problematic_columns(result, threshold=0.2):
    """Get columns dengan nulls > threshold."""
    problems = {}
    for label, profile in result.frame_profiles.items():
        problems[label] = [
            col_name
            for col_name, col_prof in profile.columns.items()
            if col_prof.null_pct > threshold
        ]
    return {k: v for k, v in problems.items() if v}


# Tip 2: Get summary statistics
def get_numeric_summary(result, frame_label):
    """Get summary untuk semua numeric columns."""
    if frame_label not in result.frame_profiles:
        return {}

    profile = result.frame_profiles[frame_label]
    return {
        col_name: {
            'mean': col_prof.mean,
            'std': col_prof.std,
            'min': col_prof.min,
            'max': col_prof.max,
        }
        for col_name, col_prof in profile.columns.items()
        if col_prof.is_numeric
    }


# Tip 3: Check for imbalanced categories
def get_imbalanced_categories(result, threshold=0.01):
    """Get categories dengan very low percentage."""
    imbalanced = {}
    for snap in result.aggregation_snapshots:
        low_pct = [
            (item['value'], item['pct'])
            for item in snap.data
            if item.get('pct', 0) < threshold
        ]
        if low_pct:
            imbalanced[f"{snap.frame_label}.{snap.aggregation_column}"] = low_pct
    return imbalanced


# ============================================================================
# INTEGRATION DENGAN LLM INSIGHTS
# ============================================================================

"""
Profiles dan snapshots memberikan konteks untuk LLM insights:

Example workflow:
1. Analyze dengan profiling
2. LLM generate insights berdasarkan query results
3. Combine dengan frame profiles untuk mendeteksi patterns
4. User lihat: profiles (what) + insights (why) + snapshots (evidence)

Contoh:
- Profile show: Jakarta = 25% dari sales (WHAT)
- Snapshot show: Jakarta adalah top region (EVIDENCE)
- Insight say: "Jakarta adalah strongest market dengan 25% market share" (WHY)
"""

