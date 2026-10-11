"""
Builds a labeled CSV from the FlakyCat dataset (test_files_v0 folder), with
columns our evaluate.py script expects: test_code + category.
"""

import argparse
import os
import pandas as pd

CATEGORY_MAP = {
    "async wait": "Timing",
    "time": "Timing",
    "test case timeout": "Timing",
    "concurrency": "Concurrency",
    "network": "Network/External I-O",
    "i_o": "Network/External I-O",
    "randomness": "Randomness",
    "test order dependency": "Order Dependency",
    "unordered collections": "Unordered Collection",
    "platform dependency": "Platform Dependency",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the test_files_v0 folder")
    parser.add_argument("--output", required=True, help="Where to save the cleaned CSV")
    args = parser.parse_args()

    rows = []
    skipped = 0

    for fname in os.listdir(args.input):
        if not fname.endswith(".txt") or "@" not in fname:
            skipped += 1
            continue

        raw_category = fname.rsplit("@", 1)[1][:-4].strip().lower()
        category = CATEGORY_MAP.get(raw_category)
        if category is None:
            skipped += 1
            continue

        path = os.path.join(args.input, fname)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()

        rows.append({"test_code": code, "category": category})

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)

    print(f"Wrote {len(out)} rows to {args.output}")
    print(f"Skipped {skipped} files (no usable category match)")
    print(out["category"].value_counts())


if __name__ == "__main__":
    main()
    