# Copyright 2026 Abadi Gilang
# SPDX-License-Identifier: Apache-2.0

"""
Example: load a CSV file into a hiveframe DFrame.

Usage:
    python examples/load_csv.py [--path PATH]

If the CSV does not exist, the script writes a small sample to the path and
then demonstrates loading it into a DFrame.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import hiveframe as hf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="/tmp/employee_data.csv", help="CSV file path to load")
    args = parser.parse_args()

    path = Path(args.path)

    # Create a small sample CSV if it doesn't exist
    if not path.exists():
        sample = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "city": ["jakarta", "bandung", "surabaya"],
                "score": [85, 92, 78],
            }
        )
        sample.to_csv(path, index=False)
        print(f"Wrote sample CSV to {path}")

    # Read CSV with pandas and convert to hiveframe DFrame
    pd_df = pd.read_csv(path)
    print("Pandas read (head):")
    print(pd_df.head())

    # Construct DFrame from column-lists
    data = pd_df.to_dict(orient="list")
    dframe = hf.DFrame(data)

    print("DFrame head:")
    print(dframe.head())

    # Show a small operation and persist
    dframe["city"] = [c.upper() for c in pd_df["city"].tolist()]
    out_path = "/tmp/from_dframe.csv"
    dframe.to_csv(out_path, index=False)
    print(f"Saved DFrame-backed CSV to {out_path}")


if __name__ == "__main__":
    main()
