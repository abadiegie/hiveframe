# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""
Basic usage example — standalone mode.

Run:
    python examples/basic_usage.py
"""

from core.dataframe import DFrame

df = DFrame(
    {
        "name": ["Alice", "Bob"],
        "city": ["jakarta", "bandung"],
        "score": [85, 92],
    }
)

# Write (transactional)
df["city"] = ["DKI Jakarta", "Jawa Barat"]

# Read — full pandas API available via proxy layer
print(df.head())
print(df.groupby("city")["score"].mean())
print(df.describe())
df.to_csv("/tmp/employee_data.csv", index=False)
print("CSV written to /tmp/employee_data.csv")

path = df.to_persistent("employee_data")
print(f"Persisted to: {path}")
