"""
Converts the raw Flakify CSV files (which use columns like 'full_code' and
'flaky') into the test_code/label format our pipeline expects.
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the raw Flakify CSV")
    parser.add_argument("--output", required=True, help="Where to save the cleaned CSV")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="If set, keep only this many rows (balanced flaky/not_flaky) for a faster first run",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    df = df[df["flaky"].isin([0, 1, 0.0, 1.0])]
    df["flaky"] = df["flaky"].astype(int)

    df = df.dropna(subset=["full_code"])

    out = pd.DataFrame(
        {
            "test_code": df["full_code"],
            "label": df["flaky"].map({1: "flaky", 0: "not_flaky"}),
        }
    )

    if args.sample:
        flaky = out[out["label"] == "flaky"]
        not_flaky = out[out["label"] == "not_flaky"]
        half = args.sample // 2
        out = pd.concat(
            [flaky.sample(min(half, len(flaky)), random_state=42),
             not_flaky.sample(min(half, len(not_flaky)), random_state=42)]
        ).sample(frac=1, random_state=42)

    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")
    print(out["label"].value_counts())


if __name__ == "__main__":
    main()