"""
Example: Comprehensive Analysis with Profiles and Aggregation Snapshots

This demonstrates the new comprehensive output feature that includes:
- Frame profiles (statistics, data quality, distributions)
- Aggregation snapshots (auto-detected value_counts)
- Rich markdown output with data overview
"""

import asyncio
import pandas as pd
from core.dataframe import DFrame
from agent.multi_agent import MultiFrameAgent


async def main():
    """Run comprehensive analysis example."""

    # Create sample data
    sales_data = {
        "product_id": ["P001", "P002", "P003", "P001", "P002", "P003", "P001"],
        "region": ["Jakarta", "Bandung", "Surabaya", "Jakarta", "Surabaya", "Jakarta", "Bandung"],
        "sales_amount": [10000, 15000, 8000, 12000, 9000, 11000, None],  # Note: None for missing value
        "quantity": [100, 150, 80, 120, 90, 110, 0],
    }

    inventory_data = {
        "product_id": ["P001", "P002", "P003"],
        "stock": [500, 200, None],
        "reorder_point": [100, 150, 100],
        "category": ["Electronics", "Electronics", "Clothing"],
    }

    # Create DFrames
    sales_frame = DFrame(sales_data)
    inventory_frame = DFrame(inventory_data)

    # Create agent with multiple frames
    agent = MultiFrameAgent(
        frames={
            "sales": sales_frame,
            "inventory": inventory_frame,
        }
    )

    # Run comprehensive analysis
    # include_profile=True (default) will automatically generate profiles and snapshots
    result = await agent.analyze(
        instruction="Analyze sales trends and inventory levels across regions",
        mode="sample",
        include_profile=True,  # NEW: Enable comprehensive profiling
    )

    # Output 1: Rich Markdown Report (includes profiles + snapshots)
    print("=" * 80)
    print("COMPREHENSIVE ANALYSIS REPORT")
    print("=" * 80)
    print(result.to_markdown())
    print()

    # Output 2: Access profiles programmatically
    print("=" * 80)
    print("FRAME PROFILES (Programmatic Access)")
    print("=" * 80)
    for label, profile in result.frame_profiles.items():
        print(f"\n📊 Frame: {label}")
        print(f"   Rows: {profile.row_count}, Columns: {profile.col_count}")

        print(f"\n   Column Details:")
        for col_name, col_prof in profile.columns.items():
            print(f"   - {col_name}:")
            print(f"     dtype: {col_prof.dtype}")
            print(f"     unique: {col_prof.unique_count}")
            print(f"     nulls: {col_prof.null_count} ({col_prof.null_pct:.1%})")

            if col_prof.is_numeric:
                print(f"     stats: min={col_prof.min}, max={col_prof.max}, mean={col_prof.mean:.2f}, std={col_prof.std:.2f}")

            if col_prof.top_values:
                top_str = ", ".join(f"{v[0]}({v[1]})" for v in col_prof.top_values[:3])
                print(f"     top values: {top_str}")

    # Output 3: Access aggregation snapshots
    print("\n" + "=" * 80)
    print("AGGREGATION SNAPSHOTS")
    print("=" * 80)
    for snap in result.aggregation_snapshots:
        print(f"\n📈 {snap.frame_label} - {snap.aggregation_column}")
        print(f"   Type: {snap.aggregation_type}")
        for item in snap.data:
            print(f"   - {item['value']}: {item['count']} ({item.get('pct', 0):.1%})")

    # Output 4: Export to JSON (for APIs/frontend)
    print("\n" + "=" * 80)
    print("JSON EXPORT (for APIs/Frontend)")
    print("=" * 80)
    result_dict = result.to_dict()

    print(f"Result action: {result_dict['action']}")
    print(f"Analysis: {result_dict['analysis'][:200]}...")
    print(f"Insights: {len(result_dict['insights'])} found")

    # Check if profiles are included in JSON
    print(f"\nProfiles in JSON export: {len(result_dict.get('frame_profiles', {}))} frames")

    # Pretty print profile structure
    import json
    if result.frame_profiles:
        first_frame_label = list(result.frame_profiles.keys())[0]
        first_profile = result.frame_profiles[first_frame_label]
        print(f"\nFirst frame profile structure (sample):")
        profile_dict = first_profile.to_dict()
        print(json.dumps({
            "frame_label": profile_dict["frame_label"],
            "row_count": profile_dict["row_count"],
            "col_count": profile_dict["col_count"],
            "columns_sample": {
                k: v for k, v in list(profile_dict["columns"].items())[:2]
            },
            "top_groupby_results": profile_dict["top_groupby_results"],
        }, indent=2))

    print("\n" + "=" * 80)
    print("✅ Analysis complete with comprehensive profiling!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

