# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""Minimal example that imports hiveframe as a module."""

from __future__ import annotations

import hiveframe as hf


def main() -> None:
    df = hf.read(".dframe_store/employee_data.parquet")

    print("Columns:", list(df.columns))
    print("Head:")
    print(df.head())

    if "salary" in df.columns:
        print("\nAverage salary:", float(df["salary"].mean()))


if __name__ == "__main__":
    main()
